#!/usr/bin/env python3
"""Read ALL official RecipeGen test text and media metadata, never train archives.

Strict byte-range fetching, exact-entry merging, bounded concurrency, SHA/CRC
validation and JSONL checkpoints keep this practical without full media ZIPs.
No ingredients, cooking time, labels or multimodal model outputs are invented.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import socket
import ssl
import struct
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import zipfile
import zlib


REPO = "RUOXUAN123/RecipeGen"
HUB = "https://huggingface.co"
ALLOWED_ARCHIVES = ("test.zip", "test-video.zip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_TEXT_BYTES = 2 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def tls_context() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE")
    if not ca and Path("/etc/ssl/cert.pem").is_file():
        ca = "/etc/ssl/cert.pem"
    return ssl.create_default_context(cafile=ca)


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def jsonl_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    repaired = False
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise ValueError(f"Corrupt non-final checkpoint line: {path}:{number + 1}")
            repaired = True
            break
        rows[row["member"]] = row
    if repaired:
        jsonl_write(path, list(rows.values()))
    return rows


def get_json(url: str, context: ssl.SSLContext) -> dict | list:
    with urlopen(Request(url, headers={"User-Agent": "RecipeGen-test-corpus/1"}), context=context, timeout=45) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Repository metadata exceeds 1 MiB")
    return json.loads(raw)


def display_path(entry: zipfile.ZipInfo) -> str:
    if not entry.flag_bits & 0x800:
        try:
            return entry.filename.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return entry.filename


def member_kind(name: str, is_dir: bool = False) -> str:
    path = PurePosixPath(name)
    if "__MACOSX" in path.parts or ".ipynb_checkpoints" in path.parts or path.name.startswith("._") or path.name == ".DS_Store":
        return "archive_metadata"
    if is_dir:
        return "directory"
    if path.name in {"goal.txt", "steps.txt"}:
        return "recipe_text"
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "other"


class RangeClient:
    """Persistent per-worker HTTPS connections; cache exact immutable ranges."""

    def __init__(self, archive: str, revision: str, size: int, cache: Path, budget: int, context: ssl.SSLContext):
        if archive not in ALLOWED_ARCHIVES:
            raise ValueError("Archive is outside the test-only whitelist")
        self.archive, self.revision, self.size = archive, revision, size
        self.url = f"{HUB}/datasets/{REPO}/resolve/{revision}/{quote(archive, safe='')}"
        self.cache = cache / archive.replace(".zip", "")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.context, self.budget = context, budget
        self.downloaded = self.requests = self.cache_hits = 0
        self.signed_url = None  # Ephemeral access URL is never saved or printed.
        self.lock = threading.Lock()
        self.local = threading.local()

    @staticmethod
    def _verify(status: int, headers, start: int, end: int, size: int) -> None:
        expected = f"bytes {start}-{end}/{size}"
        if status != 206 or headers.get("Content-Range") != expected:
            raise ValueError("Server rejected exact HTTP range; refusing a full ZIP response")

    def _network(self, start: int, end: int) -> bytes:
        amount = end - start + 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "RecipeGen-test-corpus/1"}
        for attempt in range(5):
            try:
                if not self.signed_url:
                    with self.lock:
                        if not self.signed_url:
                            with urlopen(Request(self.url, headers=headers), context=self.context, timeout=60) as response:
                                self._verify(response.status, response.headers, start, end, self.size)
                                raw = response.read(amount + 1)
                                self.signed_url = response.geturl()
                            if len(raw) != amount:
                                raise ValueError("Truncated or oversized byte range")
                            return raw
                parsed = urlsplit(self.signed_url)
                connection = getattr(self.local, "connection", None)
                if connection is None or getattr(self.local, "host", None) != parsed.netloc:
                    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port, context=self.context, timeout=60)
                    self.local.connection, self.local.host = connection, parsed.netloc
                path = parsed.path + ("?" + parsed.query if parsed.query else "")
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                if response.status in {401, 403}:
                    response.close()
                    connection.close()
                    self.local.connection = None
                    self.signed_url = None
                    continue
                if response.status in {429, 500, 502, 503, 504}:
                    response.close()
                    raise OSError(f"Transient HTTP status {response.status}")
                self._verify(response.status, response.headers, start, end, self.size)
                raw = response.read(amount + 1)
                response.close()
                if len(raw) != amount:
                    raise ValueError("Truncated or oversized byte range")
                return raw
            except (HTTPError, URLError, OSError, http.client.HTTPException, socket.timeout):
                connection = getattr(self.local, "connection", None)
                if connection:
                    connection.close()
                self.local.connection = None
                if attempt == 4:
                    raise
                time.sleep(min(0.5 * 2**attempt, 5))
        raise OSError("Could not refresh byte-range URL")

    def read(self, start: int, end: int) -> bytes:
        if not 0 <= start <= end < self.size:
            raise ValueError("Archive byte range out of bounds")
        cache_file = self.cache / f"{start}-{end}.bin"
        expected_size = end - start + 1
        if cache_file.exists():
            raw = cache_file.read_bytes()
            if len(raw) == expected_size:
                with self.lock:
                    self.cache_hits += 1
                return raw
            cache_file.unlink()
        with self.lock:
            if self.downloaded + expected_size > self.budget:
                raise ValueError("Source download budget exceeded; no complete archive attempted")
            self.downloaded += expected_size
            self.requests += 1
        raw = self._network(start, end)
        fd, temp_name = tempfile.mkstemp(prefix="range-", suffix=".part", dir=self.cache)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        Path(temp_name).replace(cache_file)
        return raw


class SeekableRange(io.RawIOBase):
    def __init__(self, client: RangeClient):
        self.client, self.position = client, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.client.size}[whence] + offset
        if self.position < 0:
            raise ValueError("Seek before archive")
        return self.position

    def read(self, amount=-1):
        remaining = self.client.size - self.position
        if amount < 0:
            if remaining > 65558:
                raise ValueError("Unbounded ZIP reads forbidden")
            amount = remaining
        amount = min(amount, remaining)
        if amount <= 0:
            return b""
        raw = self.client.read(self.position, self.position + amount - 1)
        self.position += amount
        return raw


def build_index(client: RangeClient) -> tuple[list[dict], int]:
    with zipfile.ZipFile(SeekableRange(client)) as archive:
        central_start = archive.start_dir
        entries = archive.infolist()
    if len({entry.filename for entry in entries}) != len(entries):
        raise ValueError("Duplicate exact ZIP member names must be resolved before import")
    ordered = sorted(entries, key=lambda entry: entry.header_offset)
    rows = []
    for number, entry in enumerate(ordered):
        following = ordered[number + 1].header_offset if number + 1 < len(ordered) else central_start
        if following <= entry.header_offset:
            raise ValueError("Overlapping ZIP entry offsets")
        rows.append({
            "member": entry.filename, "display_path": display_path(entry),
            "kind": member_kind(entry.filename, entry.is_dir()),
            "size": entry.file_size, "compressed_size": entry.compress_size,
            "crc": f"{entry.CRC:08x}", "compression": entry.compress_type, "flags": entry.flag_bits,
            "header_offset": entry.header_offset, "entry_end": following - 1,
            "source": {"archive": client.archive, "revision": client.revision, "split": "test"},
        })
    return rows, central_start


def decode(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("Source text encoding cannot be decoded without replacing bytes")


def parse_text_entry(entry: dict, chunk: bytes, chunk_start: int) -> dict:
    offset = entry["header_offset"] - chunk_start
    header = chunk[offset:offset + 30]
    if len(header) != 30:
        raise ValueError("Truncated local ZIP header")
    signature, version, flags, compression, mtime, mdate, crc, compressed, size, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or flags & 1:
        raise ValueError("Invalid or encrypted local ZIP text header")
    if compression != entry["compression"]:
        raise ValueError("Local and central compression metadata disagree")
    start = offset + 30 + name_len + extra_len
    payload = chunk[start:start + entry["compressed_size"]]
    if len(payload) != entry["compressed_size"]:
        raise ValueError("Incomplete compressed text member")
    if compression == zipfile.ZIP_STORED:
        raw = payload
    elif compression == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"Unsupported text compression method: {compression}")
    if len(raw) != entry["size"] or f"{zlib.crc32(raw) & 0xffffffff:08x}" != entry["crc"]:
        raise ValueError("Text size/CRC verification failed")
    text, encoding = decode(raw)
    return {"member": entry["member"], "display_path": entry["display_path"], "text": text,
            "sha256": sha(raw), "encoding": encoding, "size": len(raw), "crc": entry["crc"], "source": entry["source"]}


def chunks_for(entries: list[dict], existing: dict[str, dict], max_bytes: int) -> list[dict]:
    # Merge neighboring text, allowing small directory/AppleDouble metadata gaps
    # but never image/video/other file payloads.
    selected = [(position, entry) for position, entry in enumerate(entries) if entry["kind"] == "recipe_text" and entry["member"] not in existing]
    chunks = []
    previous_position = None
    for position, entry in selected:
        if entry["size"] > MAX_TEXT_BYTES or entry["entry_end"] - entry["header_offset"] + 1 > MAX_TEXT_BYTES:
            raise ValueError("Text member exceeds the bounded text limit")
        gap_is_metadata = previous_position is not None and all(item["kind"] in {"directory", "archive_metadata"} for item in entries[previous_position + 1:position])
        small_gap = bool(chunks) and entry["header_offset"] - chunks[-1]["end"] - 1 <= 16384
        if chunks and gap_is_metadata and small_gap and entry["entry_end"] - chunks[-1]["start"] + 1 <= max_bytes:
            chunks[-1]["end"] = entry["entry_end"]
            chunks[-1]["entries"].append(entry)
        else:
            chunks.append({"start": entry["header_offset"], "end": entry["entry_end"], "entries": [entry]})
        previous_position = position
    return chunks


def fetch_chunk(client: RangeClient, chunk: dict) -> list[dict]:
    raw = client.read(chunk["start"], chunk["end"])
    return [parse_text_entry(entry, raw, chunk["start"]) for entry in chunk["entries"]]


def records_for(archive_name: str, revision: str, entries: list[dict], texts: dict[str, dict]) -> list[dict]:
    by_folder = defaultdict(dict)
    media_by_folder = defaultdict(list)
    for entry in entries:
        folder = str(PurePosixPath(entry["member"]).parent)
        if entry["kind"] == "recipe_text":
            by_folder[folder][PurePosixPath(entry["member"]).name] = texts[entry["member"]]
        elif entry["kind"] in {"image", "video"}:
            media_by_folder[folder].append(entry)
    records = []
    for folder, fields in sorted(by_folder.items()):
        goal, steps = fields.get("goal.txt"), fields.get("steps.txt")
        representative = goal or steps
        display_dir = str(PurePosixPath(representative["display_path"]).parent)
        identity = f"test\n{archive_name}\n{folder}"
        recipe_id = f"recipegen:test:{archive_name[:-4]}:{sha(identity)[:20]}"
        media = [{
            "id": "media:test:" + sha(f"{archive_name}\n{entry['member']}")[:24],
            "kind": entry["kind"], "member": entry["member"], "display_path": entry["display_path"],
            "size": entry["size"], "compressed_size": entry["compressed_size"], "crc": entry["crc"],
            "header_offset": entry["header_offset"],
            "source": entry["source"],
        } for entry in media_by_folder[folder]]
        flags = ["ingredients_not_extracted", "cooking_time_not_provided", "tags_not_verified", "media_metadata_only"]
        if not goal:
            flags.append("missing_goal_text")
        if not steps:
            flags.append("missing_steps_text")
        step_lines = [line.strip() for line in steps["text"].splitlines() if line.strip()] if steps else []
        if goal and not goal["text"].strip():
            flags.append("empty_goal_text")
        if steps and not step_lines:
            flags.append("empty_steps_text")
        records.append({
            "id": recipe_id, "split": "test", "title": goal["text"].strip() if goal else None,
            "steps": [{"order": number, "text": line} for number, line in enumerate(step_lines, 1)],
            "source": {"archive": archive_name, "revision": revision, "split": "test", "recipe_dir": folder,
                       "display_dir": display_dir, "goal_member": goal["member"] if goal else None,
                       "steps_member": steps["member"] if steps else None,
                       "goal_sha256": goal["sha256"] if goal else None,
                       "steps_sha256": steps["sha256"] if steps else None,
                       "url": f"{HUB}/datasets/{REPO}/blob/{revision}/{archive_name}",
                       "step_split_method": "nonempty_original_lines"},
            "media": media, "quality_flags": flags,
            "provenance": {"text_extraction": "verified-http-range-zip-entry", "text_integrity": "size+crc32+sha256",
                           "goal_encoding": goal["encoding"] if goal else None,
                           "steps_encoding": steps["encoding"] if steps else None,
                           "source_texts_file": f"source/{archive_name[:-4]}.texts.jsonl", "media_state": "metadata_only"},
        })
    return records


def process_archive(archive_name: str, revision: str, size: int, source_dir: Path, workers: int, max_chunk: int, budget: int, context: ssl.SSLContext) -> tuple[list[dict], dict]:
    stem = archive_name[:-4]
    client = RangeClient(archive_name, revision, size, source_dir / "cache" / revision, budget, context)
    entries, central_start = build_index(client)
    kinds = Counter(entry["kind"] for entry in entries)
    index_path = source_dir / f"{stem}.members.jsonl"
    jsonl_write(index_path, entries)
    text_path = source_dir / f"{stem}.texts.jsonl"
    texts = load_checkpoint(text_path)
    selected = {entry["member"]: entry for entry in entries if entry["kind"] == "recipe_text"}
    for member, row in texts.items():
        if member not in selected or row["source"]["revision"] != revision or row["source"]["archive"] != archive_name:
            raise ValueError("Checkpoint contains another archive/revision or an unknown member")
        if row["crc"] != selected[member]["crc"] or row["size"] != selected[member]["size"]:
            raise ValueError("Checkpoint no longer agrees with immutable ZIP metadata")
    manifest_path = source_dir / f"{stem}.manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prior_runs = list(previous.get("prior_runs", []))
    if previous:
        prior_runs.append({key: previous.get(key) for key in ("started_at", "updated_at", "finished_at", "status", "range_requests", "downloaded_range_bytes")})
    manifest = {"dataset": REPO, "archive": archive_name, "revision": revision, "split": "test", "archive_size": size,
                "member_count": len(entries), "member_kind_counts": dict(kinds), "central_directory_start": central_start,
                "text_member_count": len(selected), "completed_text_members": len(texts), "status": "in_progress",
                "index_file": str(index_path), "texts_file": str(text_path), "started_at": now(),
                "prior_runs": prior_runs,
                "media_downloaded": False, "license": {"hub_metadata": "cc-by-4.0", "card_and_paper": "CC BY-NC 4.0", "conflict": True}}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"event": "index_ready", "archive": archive_name, **{key: manifest[key] for key in ("archive_size", "member_count", "member_kind_counts", "text_member_count", "completed_text_members")}}, ensure_ascii=False), flush=True)
    chunks = chunks_for(entries, texts, max_chunk)
    failures = []
    with text_path.open("a", encoding="utf-8") as checkpoint, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_chunk, client, chunk): chunk for chunk in chunks}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                rows = future.result()
            except Exception as error:
                failures.append({"start": futures[future]["start"], "end": futures[future]["end"], "error": str(error)})
                continue
            for row in rows:
                texts[row["member"]] = row
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            if number % 100 == 0 or number == len(chunks):
                manifest.update(completed_text_members=len(texts), range_requests=client.requests,
                                downloaded_range_bytes=client.downloaded, cache_hits=client.cache_hits, updated_at=now())
                atomic_json(manifest_path, manifest)
                print(json.dumps({"event": "progress", "archive": archive_name, "texts": len(texts), "total_texts": len(selected),
                                  "chunks": number, "total_chunks": len(chunks), "network_bytes": client.downloaded}, ensure_ascii=False), flush=True)
    manifest.update(completed_text_members=len(texts), range_requests=client.requests, downloaded_range_bytes=client.downloaded,
                    cache_hits=client.cache_hits, finished_at=now(), failed_chunks=failures)
    if failures or len(texts) != len(selected):
        manifest["status"] = "incomplete"
        atomic_json(manifest_path, manifest)
        raise ValueError(f"{archive_name}: incomplete text import; rerun to resume checkpoint ({len(failures)} failed chunks)")
    records = records_for(archive_name, revision, entries, texts)
    assigned = Counter(media["kind"] for recipe in records for media in recipe["media"])
    complete = sum(recipe["title"] is not None and "missing_steps_text" not in recipe["quality_flags"] for recipe in records)
    manifest.update(status="complete", recipe_records=len(records), complete_text_pairs=complete,
                    assigned_media_counts=dict(assigned), unassigned_media_counts={kind: kinds[kind] - assigned[kind] for kind in ("image", "video")},
                    texts_sha256=sha(text_path.read_bytes()), index_sha256=sha(index_path.read_bytes()),
                    cumulative_downloaded_range_bytes=previous.get("cumulative_downloaded_range_bytes", previous.get("downloaded_range_bytes", 0)) + client.downloaded,
                    cumulative_range_requests=previous.get("cumulative_range_requests", previous.get("range_requests", 0)) + client.requests,
                    cached_range_bytes=sum(path.stat().st_size for path in client.cache.glob("*.bin")),
                    uncompressed_text_bytes=sum(row["size"] for row in texts.values()),
                    archive_metadata_policy="AppleDouble/__MACOSX and notebook checkpoints are metadata, not media")
    atomic_json(manifest_path, manifest)
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", choices=ALLOWED_ARCHIVES, default=list(ALLOWED_ARCHIVES))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "test_graph")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-chunk-bytes", type=int, default=256 * 1024)
    parser.add_argument("--max-download-mb", type=int, default=128, help="Per archive range-byte budget, not full ZIPs")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.max_chunk_bytes < 1 or args.max_download_mb < 1:
        parser.error("workers must be 1..16; chunk size and byte budget must be positive")
    if len(set(args.archives)) != len(args.archives):
        parser.error("Duplicate archives are not allowed")
    context = tls_context()
    source_dir = args.output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    repository = get_json(f"{HUB}/api/datasets/{REPO}/revision/{quote(args.revision, safe='')}", context)
    revision = repository["sha"]
    tree = get_json(f"{HUB}/api/datasets/{REPO}/tree/{revision}", context)
    tree = [entry for entry in tree if entry["path"] in ALLOWED_ARCHIVES]
    atomic_json(source_dir / "repository.json", {"dataset": REPO, "revision": revision, "retrieved_at": now(), "tree": tree})
    archive_sizes = {entry["path"]: entry["size"] for entry in tree if entry["path"] in ALLOWED_ARCHIVES}
    if any(archive not in archive_sizes for archive in args.archives):
        raise ValueError("A required official test archive is absent")
    manifest = {"dataset": REPO, "split": "test", "revision": revision, "requested_archives": args.archives,
                "status": "in_progress", "started_at": now(), "archives": [], "media_downloaded": False,
                "processing_scope": "all goal.txt/steps.txt and all ZIP member metadata in the selected TEST archives only"}
    atomic_json(args.output_dir / "manifest.json", manifest)
    records = []
    for archive in args.archives:
        archive_records, archive_manifest = process_archive(archive, revision, archive_sizes[archive], source_dir, args.workers,
                                                           args.max_chunk_bytes, args.max_download_mb * 1024 * 1024, context)
        records.extend(archive_records)
        manifest["archives"].append(archive_manifest)
        atomic_json(args.output_dir / "manifest.json", manifest)
    jsonl_write(args.output_dir / "records.jsonl", records)
    totals = Counter()
    unassigned = Counter()
    for archive in manifest["archives"]:
        totals.update(archive["member_kind_counts"])
        unassigned.update(archive["unassigned_media_counts"])
    manifest.update(status="complete", record_count=len(records), finished_at=now(),
                    records_sha256=sha((args.output_dir / "records.jsonl").read_bytes()),
                    member_count_total=sum(totals.values()), member_kind_counts_total=dict(totals),
                    text_member_count_total=sum(item["text_member_count"] for item in manifest["archives"]),
                    complete_text_pairs_total=sum(item["complete_text_pairs"] for item in manifest["archives"]),
                    media_counts_total={kind: totals[kind] for kind in ("image", "video")},
                    unassigned_media_counts_total={kind: unassigned[kind] for kind in ("image", "video")},
                    deduplication="none; test.zip and test-video.zip source records remain distinct",
                    complete_test_scope=set(args.archives) == set(ALLOWED_ARCHIVES))
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "revision": revision, "records": len(records),
                      "output": str(args.output_dir.resolve()), "complete_test_scope": manifest["complete_test_scope"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Corpus build failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
