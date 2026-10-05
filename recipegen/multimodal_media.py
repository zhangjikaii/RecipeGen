"""Prepare real test media one recipe at a time, without running a model.

The verified member inventories replace repeated remote ZIP central-directory
reads. Payloads are read through the existing strict RangeFile, then streamed
through ZIP size/CRC and SHA checks. Video windows use observed PTS, never steps.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import http.client
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sqlite3
import ssl
import struct
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zlib

from .test_graph import ARCHIVES, REPO, uid


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_id(archive: str, member: str) -> str:
    # Match test_graph.build_graph.source_node exactly, including the dataset.
    return uid("source", f"{REPO}\n{archive}\n{member}")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_records(records_path: Path, limit: int = 0, offset: int = 0, recipe_id: str | None = None):
    """Sort IDs/byte offsets, keeping full recipe/media dictionaries off RAM."""
    if limit < 0 or offset < 0:
        raise ValueError("limit and offset must be non-negative; limit=0 means all")
    positions = []
    seen = set()
    with Path(records_path).open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError("Duplicate recipe stable ID")
            seen.add(row["id"])
            if recipe_id is None or row["id"] == recipe_id:
                positions.append((row["id"], position))
        if recipe_id is not None and not positions:
            raise ValueError(f"Unknown recipe ID: {recipe_id}")
        positions.sort()
        positions = positions[offset:] if limit == 0 else positions[offset:offset + limit]
        for _, position in positions:
            handle.seek(position)
            yield json.loads(handle.readline())


@dataclass(frozen=True)
class MediaLimits:
    max_file_bytes: int = 1024 * 1024 * 1024
    max_recipe_bytes: int = 2 * 1024 * 1024 * 1024
    max_download_bytes: int = 2 * 1024 * 1024 * 1024 + 1024 * 1024
    max_workspace_bytes: int = 4 * 1024 * 1024 * 1024
    chunk_bytes: int = 4 * 1024 * 1024
    http_attempts: int = 3
    clip_seconds: float = 5.0
    frames_per_clip: int = 3
    max_clips: int = 2000
    max_frames: int = 6000
    max_scanned_frames: int = 1000000
    frame_max_side: int = 448
    subprocess_timeout: float = 1800.0

    def __post_init__(self):
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("Every media resource limit must be positive")
        if self.http_attempts > 5 or self.chunk_bytes > 16 * 1024 * 1024:
            raise ValueError("HTTP retries/chunk size exceed the bounded reader limits")


def _range_base():
    script = Path(__file__).resolve().parents[1] / "scripts/import_recipegen.py"
    spec = importlib.util.spec_from_file_location("recipegen_media_range_base", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RangeFile, module.ssl_context


_RangeFile, _ssl_context = _range_base()


class MediaRangeFile(_RangeFile):
    """RangeFile with persistent HTTPS, limited retries and no payload cache."""

    def __init__(self, archive: str, revision: str, size: int, limits: MediaLimits):
        if archive not in ARCHIVES:
            raise ValueError("Archive is outside the test-only whitelist")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A full immutable dataset revision is required")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{archive}"
        super().__init__(url, size, limits.max_download_bytes, _ssl_context())
        self.limits = limits
        self.signed_url = None
        self.connection = None
        self.attempted_bytes = 0
        self.attempts = 0

    def reset_budget(self):
        self.downloaded = self.requests = self.attempted_bytes = self.attempts = 0

    @staticmethod
    def verify_response(status: int, headers, start: int, end: int, size: int):
        if status != 206 or headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise ValueError("Exact HTTP Range was not honored; full ZIP response refused")

    def _disconnect(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.limits.chunk_bytes:
            raise ValueError("Unbounded or oversized media range read refused")
        if not 0 <= self.position <= self.size:
            raise ValueError("ZIP byte position out of bounds")
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        start, end = self.position, self.position + size - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "RecipeGen-multimodal-media/1"}
        for attempt in range(self.limits.http_attempts):
            if self.attempted_bytes + size > self.budget:
                raise ValueError("Recipe HTTP byte budget exceeded")
            # Count even failed attempts conservatively against the budget.
            self.attempted_bytes += size
            self.attempts += 1
            try:
                if self.signed_url is None:
                    with urlopen(Request(self.url, headers=headers), context=self.context, timeout=60) as response:
                        self.verify_response(response.status, response.headers, start, end, self.size)
                        raw = response.read(size + 1)
                        self.signed_url = response.geturl()
                else:
                    parts = urlsplit(self.signed_url)
                    if self.connection is None:
                        self.connection = http.client.HTTPSConnection(parts.hostname, parts.port, context=self.context, timeout=60)
                    target = parts.path + ("?" + parts.query if parts.query else "")
                    self.connection.request("GET", target, headers=headers)
                    response = self.connection.getresponse()
                    try:
                        if response.status in {401, 403}:
                            self.signed_url = None
                            raise OSError("Temporary signed URL needs refreshing")
                        if response.status in {429, 500, 502, 503, 504}:
                            raise OSError(f"Transient media HTTP status {response.status}")
                        self.verify_response(response.status, response.headers, start, end, self.size)
                        raw = response.read(size + 1)
                    finally:
                        response.close()
                if len(raw) != size:
                    raise ValueError("Truncated or oversized media HTTP range")
                self.position += size
                self.requests += 1
                self.downloaded += size
                return raw
            except (HTTPError, URLError, OSError, http.client.HTTPException, socket.timeout):
                self._disconnect()
                if attempt + 1 == self.limits.http_attempts:
                    raise
                time.sleep(min(0.5 * 2 ** attempt, 2.0))
        raise OSError("Media range retries exhausted")

    def close(self):
        self._disconnect()
        super().close()


def read_zip_member(remote, entry: dict, target: Path, limits: MediaLimits) -> dict:
    """Use authoritative central metadata and verify the local header + CRC."""
    if entry.get("source", {}).get("archive") not in ARCHIVES or entry.get("source", {}).get("split") != "test":
        raise ValueError("Non-test media entry rejected")
    if entry["kind"] not in {"image", "video"}:
        raise ValueError("Only genuine inventory image/video entries are allowed")
    if entry["size"] > limits.max_file_bytes or entry["compressed_size"] > limits.max_file_bytes:
        raise ValueError("Media exceeds max_file_bytes; no payload downloaded")
    remote.seek(entry["header_offset"])
    header = remote.read(30)
    if len(header) != 30:
        raise ValueError("Truncated ZIP local header")
    signature, _, flags, compression, _, _, local_crc, local_compressed, local_size, name_length, extra_length = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or flags & 1 or flags != entry["flags"] or compression != entry["compression"]:
        raise ValueError("ZIP local header disagrees with the verified inventory")
    if compression not in {0, 8}:
        raise ValueError("Unsupported ZIP media compression")
    raw_name = remote.read(name_length)
    name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
    if name != entry["member"]:
        raise ValueError("ZIP local member name mismatch")
    if not flags & 8 and (local_crc != int(entry["crc"], 16) or local_compressed != entry["compressed_size"] or local_size != entry["size"]):
        raise ValueError("ZIP local size/CRC metadata mismatch")
    payload_start = entry["header_offset"] + 30 + name_length + extra_length
    if payload_start + entry["compressed_size"] - 1 > entry["entry_end"]:
        raise ValueError("ZIP compressed payload crosses its member boundary")
    remote.seek(payload_start)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    crc = total = 0
    digest = hashlib.sha256()
    decoder = zlib.decompressobj(-15) if compression == 8 else None
    try:
        with temporary.open("wb") as output:
            remaining = entry["compressed_size"]
            while remaining:
                amount = min(remaining, limits.chunk_bytes)
                payload = remote.read(amount)
                if len(payload) != amount:
                    raise ValueError("Truncated compressed ZIP media payload")
                remaining -= amount
                # Bound deflate expansion by declared size, even for corrupt ZIPs.
                plain = decoder.decompress(payload, entry["size"] - total + 1) if decoder else payload
                total += len(plain)
                if total > entry["size"] or (decoder and decoder.unconsumed_tail):
                    raise ValueError("ZIP decompressed payload exceeds the declared size")
                digest.update(plain)
                crc = zlib.crc32(plain, crc)
                output.write(plain)
            if decoder:
                if not decoder.eof or decoder.unused_data:
                    raise ValueError("Invalid deflated ZIP media stream")
            if total != entry["size"] or f"{crc & 0xffffffff:08x}" != entry["crc"]:
                raise ValueError("ZIP media size/CRC verification failed")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"sha256": digest.hexdigest(), "size": total, "crc32": entry["crc"], "zip_crc_verified": True}


def _run_json(command: list[str], timeout: float) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def probe_video(path: Path, limits: MediaLimits) -> tuple[dict, list[dict]]:
    """Read true stream metadata and decoded-frame presentation timestamps."""
    info = _run_json(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=index,width,height,time_base,start_time,duration,avg_frame_rate:format=duration,start_time", "-of", "json", str(path)], limits.subprocess_timeout)
    if not info.get("streams"):
        raise ValueError("Video has no decodable video stream")
    stream = info["streams"][0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe did not provide a finite positive video duration")
    planned_count = math.ceil(duration / limits.clip_seconds)
    if planned_count > limits.max_clips or planned_count * limits.frames_per_clip > limits.max_frames:
        raise ValueError("Video clip/frame count exceeds configured limits; not silently truncated")
    time_base = stream.get("time_base")
    if not time_base or Fraction(time_base) <= 0:
        raise ValueError("ffprobe did not provide a valid video time_base")
    # Stream compact lines rather than allocating ffprobe's full frame JSON.
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time", "-of", "compact=p=0:nk=0", str(path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    timer = threading.Timer(limits.subprocess_timeout, process.kill)
    timer.daemon = True
    timer.start()
    frames = []
    started = time.monotonic()
    try:
        for line in process.stdout:
            if time.monotonic() - started > limits.subprocess_timeout:
                raise TimeoutError("ffprobe frame scan exceeded subprocess_timeout")
            fields = dict(part.split("=", 1) for part in line.strip().split("|") if "=" in part)
            if "best_effort_timestamp" not in fields:
                continue
            pts = int(fields["best_effort_timestamp"])
            timestamp = float(Fraction(pts) * Fraction(time_base))
            if not math.isfinite(timestamp) or (frames and timestamp <= frames[-1]["timestamp_sec"]):
                raise ValueError("Video PTS is non-finite or not monotonically ordered")
            frames.append({"decode_index": len(frames), "pts": pts, "timestamp_sec": timestamp, "time_base": time_base})
            if len(frames) > limits.max_scanned_frames:
                raise ValueError("Video exceeds max_scanned_frames; not silently truncated")
        if process.wait(timeout=30) != 0 or not frames:
            raise ValueError("ffprobe frame scan failed or returned no frames")
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    return {"duration_sec": duration, "stream_start_time_sec": float(stream.get("start_time") or 0), "timeline_origin_sec": frames[0]["timestamp_sec"], "time_base": time_base, "width": int(stream["width"]), "height": int(stream["height"]), "avg_frame_rate": stream.get("avg_frame_rate"), "decoded_frame_count": len(frames), "probe_method": "ffprobe_decoded_best_effort_timestamp", "time_reference": "clip seconds relative to first decoded video frame; frame timestamp_sec is original PTS*time_base"}, frames


def plan_clips(media_id: str, metadata: dict, frames: list[dict], limits: MediaLimits) -> list[dict]:
    """Select existing frames near start/middle/end of fixed real-time windows."""
    duration = metadata["duration_sec"]
    count = math.ceil(duration / limits.clip_seconds)
    if count > limits.max_clips or count * limits.frames_per_clip > limits.max_frames:
        raise ValueError("Video clip/frame count exceeds configured limits; not silently truncated")
    origin = metadata["timeline_origin_sec"]
    offsets = [frame["timestamp_sec"] - origin for frame in frames]
    clips = []
    for number in range(count):
        start, end = number * limits.clip_seconds, min((number + 1) * limits.clip_seconds, duration)
        lo, hi = bisect_left(offsets, start), bisect_left(offsets, end)
        if hi == lo:
            raise ValueError(f"No observed video frames in time window {start}-{end}; window cannot be invented")
        indices = []
        targets = [start + (end - start) * j / max(limits.frames_per_clip - 1, 1) for j in range(limits.frames_per_clip)]
        for target in targets:
            at = bisect_left(offsets, target, lo, hi)
            options = [candidate for candidate in (at - 1, at) if lo <= candidate < hi]
            chosen = min(options, key=lambda candidate: (abs(offsets[candidate] - target), candidate))
            if chosen not in indices:
                indices.append(chosen)
        indices.sort()
        selected = [{**frames[index], "offset_sec": offsets[index], "frame_id": uid("frame", f"{media_id}\n{frames[index]['pts']}\n{metadata['time_base']}")} for index in indices]
        clips.append({"clip_id": uid("video_clip", f"{media_id}\n{start:.9f}\n{end:.9f}"), "order": number + 1, "start_sec": start, "end_sec": end, "duration_sec": end - start, "media_id": media_id, "method": "fixed_real_time_window", "clip_file_created": False, "frames": selected, "quality_flags": ["fewer_distinct_frames_than_requested"] if len(indices) < limits.frames_per_clip else []})
    return clips


def extract_video_frames(video: Path, media_id: str, output_dir: Path, project_root: Path, limits: MediaLimits, *, workspace_root: Path | None = None) -> dict:
    metadata, observed = probe_video(video, limits)
    clips = plan_clips(media_id, metadata, observed, limits)
    selected = sorted({frame["decode_index"]: frame for clip in clips for frame in clip["frames"]}.values(), key=lambda frame: frame["decode_index"])
    # Conservative JPEG bound includes raw RGB pixels and encoder overhead.
    projected_frame_bytes = len(selected) * (limits.frame_max_side ** 2 * 3 + 65536)
    bounded_root = workspace_root or output_dir.parent
    existing_bytes = sum(path.stat().st_size for path in bounded_root.rglob("*") if path.is_file()) if bounded_root.exists() else 0
    if existing_bytes + projected_frame_bytes > limits.max_workspace_bytes or shutil.disk_usage(video.parent).free < projected_frame_bytes + 64 * 1024 * 1024:
        raise ValueError("Frame output exceeds the available bounded temporary workspace")
    output_dir.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n,{frame['decode_index']})" for frame in selected)
    filter_path = output_dir / "select.filter"
    filter_path.write_text(f"select='{expression}',scale=w='min({limits.frame_max_side},iw)':h='min({limits.frame_max_side},ih)':force_original_aspect_ratio=decrease,showinfo", encoding="utf-8")
    output_template = output_dir / "sample-%08d.jpg"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-copyts", "-i", str(video), "-map", "0:v:0", "-filter_script:v", str(filter_path), "-fps_mode", "passthrough", "-frames:v", str(len(selected)), "-q:v", "3", str(output_template)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=limits.subprocess_timeout, check=True)
    emitted = [(int(pts), int(width), int(height)) for pts, width, height in re.findall(r"\bpts:\s*(-?\d+)\s+pts_time:[^\s]+.*?\bs:(\d+)x(\d+)", result.stderr)]
    emitted_pts = [value[0] for value in emitted]
    if emitted_pts != [frame["pts"] for frame in selected]:
        raise ValueError("ffmpeg emitted frame PTS differs from the observed ffprobe selection")
    for number, frame in enumerate(selected, 1):
        temporary = output_dir / f"sample-{number:08d}.jpg"
        if not temporary.is_file():
            raise ValueError("ffmpeg did not emit all requested frames")
        target = output_dir / (frame["frame_id"].split(":", 1)[1] + ".jpg")
        temporary.replace(target)
        _, width, height = emitted[number - 1]
        if max(width, height) > limits.frame_max_side:
            raise ValueError("Decoded frame exceeds frame_max_side")
        frame.update({"path": str(target.relative_to(project_root)), "sha256": file_sha256(target), "size": target.stat().st_size, "width": width, "height": height, "storage_state": "local", "pts_verified_by_ffmpeg": True})
    lookup = {frame["decode_index"]: frame for frame in selected}
    for clip in clips:
        clip["frames"] = [dict(lookup[frame["decode_index"]]) for frame in clip["frames"]]
    filter_path.unlink(missing_ok=True)
    return {**metadata, "clips": clips, "clip_count": len(clips), "sampled_frame_count": len(selected), "requested_frames_per_clip": limits.frames_per_clip}


class MediaPreparer:
    """Single-process preparation/cleanup API for the root's model pipeline.

    ``prepare_record`` appends evidence; ``cleanup_record`` removes only owned
    temporary media and appends the cleaned state. The caller must finish model
    outputs/checkpoints before cleanup. Raw source hashes survive cleanup.
    """

    def __init__(self, project_root: Path, corpus_dir: str | Path = "data/test_graph", workspace_dir: str | Path = "data/multimodal/work", limits: MediaLimits | None = None, *, reader_factory=None):
        self.root = Path(project_root).resolve()
        self.corpus = self._under_root(corpus_dir)
        self.workspace = self._under_root(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_path = self.workspace.parent / "media-workspace.jsonl"
        self.limits = limits or MediaLimits()
        self.reader_factory = reader_factory or MediaRangeFile
        self.readers = {}
        self.manifests = {}
        for archive in ARCHIVES:
            manifest = json.loads((self.corpus / "source" / f"{archive[:-4]}.manifest.json").read_text())
            if manifest.get("archive") != archive or manifest.get("split") != "test" or manifest.get("status") != "complete" or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
                raise ValueError("Complete immutable test manifests are required")
            self.manifests[archive] = manifest
        revisions = {manifest["revision"] for manifest in self.manifests.values()}
        if len(revisions) != 1:
            raise ValueError("Mixed dataset revisions are forbidden")
        self.revision = revisions.pop()
        self._build_inventory_cache()
        self.samples = {}
        sample_report = self.corpus / "media_samples/materialization-report.json"
        if sample_report.is_file():
            for row in json.loads(sample_report.read_text()).get("media", []):
                if row["archive"] in ARCHIVES and row["revision"] == self.revision and row["split"] == "test" and row.get("zip_crc_verified") is True:
                    self.samples[(row["archive"], row["member"])] = row

    def _under_root(self, value: str | Path) -> Path:
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Media paths must remain under project_root")
        return path

    def _build_inventory_cache(self):
        fingerprint = hashlib.sha256(json.dumps({archive: manifest["index_sha256"] for archive, manifest in self.manifests.items()}, sort_keys=True).encode()).hexdigest()
        path = self.workspace.parent / "member-inventory.sqlite3"
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS media (archive TEXT, member TEXT, row_json TEXT NOT NULL, PRIMARY KEY(archive,member))")
        prior = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        # Hash current files even when a previous cache is usable.
        for archive, manifest in self.manifests.items():
            if file_sha256(self.corpus / "source" / f"{archive[:-4]}.members.jsonl") != manifest["index_sha256"]:
                raise ValueError("Member index SHA does not agree with its manifest")
        if prior and prior[0] == fingerprint:
            return
        self.db.execute("DELETE FROM media")
        for archive, manifest in self.manifests.items():
            count = 0
            for entry in _read_jsonl(self.corpus / "source" / f"{archive[:-4]}.members.jsonl"):
                count += 1
                if entry["source"] != {"archive": archive, "revision": self.revision, "split": "test"}:
                    raise ValueError("ZIP index entry source does not match pinned test manifest")
                if entry["kind"] in {"image", "video"}:
                    self.db.execute("INSERT INTO media VALUES (?,?,?)", (archive, entry["member"], json.dumps(entry, ensure_ascii=False)))
            if count != manifest["member_count"]:
                raise ValueError("ZIP member count does not match manifest")
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint',?)", (fingerprint,))
        self.db.commit()

    def _entry(self, archive: str, member: str) -> dict:
        if archive not in ARCHIVES:
            raise ValueError("Archive is outside the test-only whitelist")
        row = self.db.execute("SELECT row_json FROM media WHERE archive=? AND member=?", (archive, member)).fetchone()
        if not row:
            raise ValueError("Media member is absent from the verified image/video inventory")
        return json.loads(row[0])

    def _recipe_dir(self, recipe_id: str) -> Path:
        return self.workspace / uid("workspace", recipe_id).split(":", 1)[1]

    def _append(self, row: dict):
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    def _reader(self, archive: str):
        if archive not in self.readers:
            manifest = self.manifests[archive]
            self.readers[archive] = self.reader_factory(archive, self.revision, manifest["archive_size"], self.limits)
        return self.readers[archive]

    def _materialize(self, entry: dict, directory: Path) -> dict:
        archive, member = entry["source"]["archive"], entry["member"]
        media_id = uid(entry["kind"], f"{archive}\n{member}")
        extension = PurePosixPath(entry["display_path"]).suffix.lower()
        target = directory / (media_id.split(":", 1)[1] + extension)
        sidecar = target.with_suffix(target.suffix + ".json")
        if target.exists():
            if not sidecar.is_file():
                raise ValueError("Existing media has no integrity checkpoint")
            checked = json.loads(sidecar.read_text())
            if checked.get("member") != member or checked.get("archive") != archive or checked.get("revision") != self.revision or checked.get("size") != entry["size"] or checked.get("crc32") != entry["crc"] or checked.get("zip_crc_verified") is not True or file_sha256(target) != checked["sha256"]:
                raise ValueError("Existing media failed source/SHA checkpoint verification")
            checked["reused"] = True
        else:
            sample = self.samples.get((archive, member))
            if sample:
                source = self.corpus / "media_samples" / Path(sample["path"]).name
                if source.stat().st_size != entry["size"] or sample["crc32"] != entry["crc"] or file_sha256(source) != sample["sha256"]:
                    raise ValueError("Existing original media sample failed SHA/CRC identity check")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                checked = {"sha256": sample["sha256"], "size": entry["size"], "crc32": entry["crc"], "zip_crc_verified": True, "reused_original_sample": True}
            else:
                checked = read_zip_member(self._reader(archive), entry, target, self.limits)
            checked.update({"archive": archive, "member": member, "revision": self.revision, "split": "test"})
            sidecar.write_text(json.dumps(checked, ensure_ascii=False) + "\n")
        return {**checked, "media_id": media_id, "source_id": source_id(archive, member), "kind": entry["kind"], "display_path": entry["display_path"], "compressed_size": entry["compressed_size"], "path": str(target.relative_to(self.root)), "storage_state": "local", "recognition_status": "not_run"}

    def prepare_record(self, record: dict) -> dict:
        src = record["source"]
        if record.get("split") != "test" or src.get("split") != "test" or src.get("archive") not in ARCHIVES or src.get("revision") != self.revision:
            raise ValueError("Non-test or mixed-revision recipe rejected")
        if any(step.get("order") != number for number, step in enumerate(record.get("steps", []), 1)):
            raise ValueError("Original Step order must remain contiguous and unchanged")
        directory = self._recipe_dir(record["id"])
        directory.mkdir(parents=True, exist_ok=True)
        row = {"schema_version": 1, "recipe_id": record["id"], "title": record["title"], "split": "test", "revision": self.revision, "source": src, "steps": [{"id": uid("step", f"{record['id']}\n{number}"), "order": number, "text": step["text"], "source_id": source_id(src["archive"], src["steps_member"])} for number, step in enumerate(record.get("steps", []), 1)], "images": [], "videos": [], "status": "preparing", "errors": [], "storage_state": "local", "limits": asdict(self.limits), "media_directory": str(directory.relative_to(self.root)), "expected_media_count": len(record.get("media", []))}
        for reader in self.readers.values():
            reader.reset_budget()
        try:
            entries = []
            for medium in record.get("media", []):
                archive = medium["source"]["archive"]
                if medium["source"] != {"archive": archive, "revision": self.revision, "split": "test"}:
                    raise ValueError("Recipe media provenance mismatch")
                entry = self._entry(archive, medium["member"])
                if any(medium[field] != entry[field] for field in ("kind", "size", "compressed_size", "crc", "header_offset")):
                    raise ValueError("Recipe media does not match verified inventory")
                if entry["size"] > self.limits.max_file_bytes or entry["compressed_size"] > self.limits.max_file_bytes:
                    raise ValueError("Media exceeds max_file_bytes; complete recipe cannot be prepared")
                entries.append(entry)
            if len({(entry["source"]["archive"], entry["member"]) for entry in entries}) != len(entries):
                raise ValueError("Duplicate media inventory members in recipe")
            total = sum(entry["size"] for entry in entries)
            if total > self.limits.max_recipe_bytes:
                raise ValueError("Recipe media exceeds max_recipe_bytes; not silently truncated")
            existing_bytes = sum(path.stat().st_size for path in self.workspace.rglob("*") if path.is_file())
            existing_recipe_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            remaining_estimate = max(total - existing_recipe_bytes, 0)
            if existing_bytes + remaining_estimate > self.limits.max_workspace_bytes or shutil.disk_usage(self.workspace).free < remaining_estimate + 256 * 1024 * 1024:
                raise ValueError("Insufficient bounded temporary workspace; cleanup previous recipes first")
            for entry in sorted(entries, key=lambda entry: (entry["source"]["archive"], entry["header_offset"])):
                medium = self._materialize(entry, directory)
                if entry["kind"] == "image":
                    row["images"].append(medium)
                else:
                    # Preserve the downloaded source SHA even if probing fails.
                    medium.update({"clips": [], "clip_count": 0, "sampled_frame_count": 0})
                    row["videos"].append(medium)
                    medium.update(extract_video_frames(self.root / medium["path"], medium["media_id"], directory / (medium["media_id"].split(":", 1)[1] + "-frames"), self.root, self.limits, workspace_root=self.workspace))
            row["status"] = "complete"
        except (ValueError, OSError, RuntimeError, zlib.error, subprocess.SubprocessError, TimeoutError) as error:
            row["status"] = "failed"
            row["errors"].append({"type": type(error).__name__, "message": str(error)})
        row["counts"] = {"images": len(row["images"]), "videos": len(row["videos"]), "clips": sum(video["clip_count"] for video in row["videos"]), "frames": sum(video["sampled_frame_count"] for video in row["videos"])}
        row["range"] = {archive: {"requests": reader.requests, "downloaded_bytes": reader.downloaded, "attempted_bytes": reader.attempted_bytes, "attempts": reader.attempts} for archive, reader in self.readers.items()}
        checkpoint = directory / "prepared.json"
        checkpoint.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self._append(row)
        return row

    def cleanup_record(self, recipe_id: str) -> dict:
        directory = self._recipe_dir(recipe_id)
        checkpoint = directory / "prepared.json"
        row = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {"schema_version": 1, "recipe_id": recipe_id, "split": "test", "revision": self.revision, "status": "cleanup_without_checkpoint", "images": [], "videos": []}
        if directory.exists():
            shutil.rmtree(directory)
        row["storage_state"] = "cleaned"
        for medium in row.get("images", []) + row.get("videos", []):
            medium["storage_state"] = "cleaned"
            for clip in medium.get("clips", []):
                for frame in clip["frames"]:
                    frame["storage_state"] = "cleaned"
        row["cleanup_method"] = "removed_only_recipe_owned_temporary_directory"
        self._append(row)
        return row

    def iter_orphan_media(self):
        """Inventory media without an original recipe owner; do not invent one."""
        owned = {(media["source"]["archive"], media["member"]) for record in _read_jsonl(self.corpus / "records.jsonl") for media in record.get("media", [])}
        for archive, member, encoded in self.db.execute("SELECT archive,member,row_json FROM media ORDER BY archive,member"):
            if (archive, member) not in owned:
                entry = json.loads(encoded)
                yield {"media_id": uid(entry["kind"], f"{archive}\n{member}"), "source_id": source_id(archive, member), "recipe_id": None, **entry}

    def prepare_orphan_media(self, entry: dict) -> dict:
        """Prepare an unowned image/video without creating a fictional Recipe.

        Clean this row with ``cleanup_record(row['work_id'])`` after inference.
        """
        archive, member = entry["source"]["archive"], entry["member"]
        canonical = self._entry(archive, member)
        work_id = "orphan:" + uid(canonical["kind"], f"{archive}\n{member}")
        directory = self._recipe_dir(work_id)
        directory.mkdir(parents=True, exist_ok=True)
        row = {"schema_version": 1, "recipe_id": None, "work_id": work_id, "title": None, "split": "test", "revision": self.revision, "source": canonical["source"], "steps": [], "images": [], "videos": [], "status": "preparing", "errors": [], "storage_state": "local", "limits": asdict(self.limits), "media_directory": str(directory.relative_to(self.root)), "expected_media_count": 1, "association_status": "no_original_recipe_owner"}
        for reader in self.readers.values():
            reader.reset_budget()
        try:
            if canonical["size"] > min(self.limits.max_file_bytes, self.limits.max_recipe_bytes) or canonical["compressed_size"] > self.limits.max_file_bytes:
                raise ValueError("Orphan media exceeds configured byte limits")
            current_bytes = sum(path.stat().st_size for path in self.workspace.rglob("*") if path.is_file())
            if current_bytes + canonical["size"] > self.limits.max_workspace_bytes or shutil.disk_usage(directory).free < canonical["size"] + 256 * 1024 * 1024:
                raise ValueError("Insufficient bounded orphan media workspace")
            medium = self._materialize(canonical, directory)
            if medium["kind"] == "image":
                row["images"].append(medium)
            else:
                medium.update({"clips": [], "clip_count": 0, "sampled_frame_count": 0})
                row["videos"].append(medium)
                medium.update(extract_video_frames(self.root / medium["path"], medium["media_id"], directory / "frames", self.root, self.limits, workspace_root=self.workspace))
            row["status"] = "complete"
        except (ValueError, OSError, RuntimeError, zlib.error, subprocess.SubprocessError, TimeoutError) as error:
            row["status"] = "failed"
            row["errors"].append({"type": type(error).__name__, "message": str(error)})
        row["counts"] = {"images": len(row["images"]), "videos": len(row["videos"]), "clips": sum(video["clip_count"] for video in row["videos"]), "frames": sum(video["sampled_frame_count"] for video in row["videos"])}
        (directory / "prepared.json").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self._append(row)
        return row

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
