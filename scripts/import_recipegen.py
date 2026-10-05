#!/usr/bin/env python3
"""Import only RecipeGen goal.txt/steps.txt into an honest text staging format.

Remote mode uses verified HTTP byte ranges and refuses full-archive responses.
The output is NOT the application's recipe graph: absent ingredients, tags and
cooking time remain absent. Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import ssl
import sys
from typing import BinaryIO
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


REPO_ID = "RUOXUAN123/RecipeGen"
HUB = "https://huggingface.co"
MAX_TEXT_BYTES = 1024 * 1024


def ssl_context() -> ssl.SSLContext:
    """Use verified TLS, including the system CA bundle on macOS when present."""
    ca_file = os.environ.get("SSL_CERT_FILE")
    if not ca_file and Path("/etc/ssl/cert.pem").is_file():
        ca_file = "/etc/ssl/cert.pem"
    return ssl.create_default_context(cafile=ca_file)


def fetch_json(url: str, context: ssl.SSLContext) -> dict | list:
    request = Request(url, headers={"User-Agent": "RecipeGen-text-import/1"})
    with urlopen(request, context=context, timeout=30) as response:
        raw = response.read(MAX_TEXT_BYTES + 1)
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError("Repository metadata exceeds the 1 MiB safety limit")
    return json.loads(raw)


class RangeFile(io.RawIOBase):
    """Seekable ZIP input that fetches exactly requested ranges, within a budget."""

    def __init__(self, url: str, size: int, budget: int, context: ssl.SSLContext):
        self.url = url
        self.size = size
        self.budget = budget
        self.context = context
        self.position = 0
        self.downloaded = 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in bases:
            raise ValueError("Unsupported seek mode")
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek before the archive")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            # zipfile asks for an unsized read only when inspecting the EOCD
            # tail. Bound that operation to the maximum comment-bearing tail.
            size = self.size - self.position
            if size > 65558:
                raise ValueError("Unbounded archive reads are forbidden")
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b""
        if self.downloaded + size > self.budget:
            raise ValueError("Text import exceeded its byte budget; no full ZIP download attempted")
        start, end = self.position, self.position + size - 1
        request = Request(self.url, headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "RecipeGen-text-import/1",
        })
        with urlopen(request, context=self.context, timeout=45) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor the exact byte range; refusing archive download")
            raw = response.read(size + 1)
        if len(raw) != size:
            raise ValueError("Incomplete or oversized HTTP range response")
        self.position += size
        self.downloaded += size
        self.requests += 1
        return raw


def decode_text(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("Text is neither valid UTF-8 nor GB18030; do not replace source bytes silently")


def display_member_path(entry: zipfile.ZipInfo) -> str:
    """Repair a display-only UTF-8 path when a ZIP omitted its UTF-8 flag."""
    if not entry.flag_bits & 0x800:
        try:
            return entry.filename.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return entry.filename


def import_texts(archive: zipfile.ZipFile, archive_name: str, revision: str, limit: int) -> tuple[list[dict], int]:
    pairs: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for entry in archive.infolist():
        member = PurePosixPath(entry.filename)
        if entry.is_dir() or member.name not in {"goal.txt", "steps.txt"}:
            continue
        if entry.file_size > MAX_TEXT_BYTES:
            raise ValueError(f"Text member exceeds 1 MiB: {entry.filename}")
        bucket = pairs.setdefault(str(member.parent), {})
        if member.name in bucket:
            raise ValueError(f"Duplicate text member: {entry.filename}")
        bucket[member.name] = entry
    complete = sorted(folder for folder, members in pairs.items() if {"goal.txt", "steps.txt"} <= members.keys())
    records = []
    for folder in complete[:limit]:
        members = pairs[folder]
        raw = {name: archive.read(members[name]) for name in ("goal.txt", "steps.txt")}
        decoded = {name: decode_text(value) for name, value in raw.items()}
        goal = decoded["goal.txt"][0].strip()
        step_text = decoded["steps.txt"][0].strip()
        if not goal or not step_text:
            raise ValueError(f"Empty title or steps: {folder}")
        # Preserve original non-empty lines; do not infer missing steps or timing.
        steps = [line.strip() for line in step_text.splitlines() if line.strip()]
        identity = f"{REPO_ID}\n{archive_name}\n{folder}"
        stable_id = "recipegen:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        records.append({
            "id": stable_id,
            "name": goal,
            "steps": steps,
            "ingredients": None,
            "seasonings": None,
            "minutes": None,
            "tags": [],
            "quality_flags": ["ingredients_not_provided", "seasonings_not_classified", "cooking_time_not_provided", "tags_not_verified"],
            "source": {
                "id": stable_id, "title": goal,
                "url": f"{HUB}/datasets/{REPO_ID}/blob/{quote(revision, safe='')}/{quote(archive_name, safe='')}",
                "dataset": REPO_ID, "revision": revision, "archive": archive_name,
                "recipe_directory": folder,
                "recipe_directory_display": str(PurePosixPath(display_member_path(members["goal.txt"])).parent),
                "goal_member": members["goal.txt"].filename,
                "steps_member": members["steps.txt"].filename,
            },
            "raw_text": {"goal": decoded["goal.txt"][0], "steps": decoded["steps.txt"][0]},
            "provenance": {
                "goal_sha256": hashlib.sha256(raw["goal.txt"]).hexdigest(),
                "steps_sha256": hashlib.sha256(raw["steps.txt"]).hexdigest(),
                "encodings": {name: pair[1] for name, pair in decoded.items()},
                "step_split_method": "nonempty_original_lines",
            },
        })
    return records, len(complete)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="test.zip", help="Official archive name; defaults to test.zip")
    parser.add_argument("--revision", default="main", help="HF revision; remote mode resolves it to a commit")
    parser.add_argument("--limit", type=int, default=5, help="First N directories in lexical order, not a representative sample")
    parser.add_argument("--max-download-mb", type=int, default=16, help="Strict total ZIP range byte budget")
    parser.add_argument("--local-zip", type=Path, help="Read only text members of an existing local ZIP; never extract media")
    parser.add_argument("--output", type=Path, required=True, help="Text staging JSON; keep separate from example_graph.json")
    args = parser.parse_args()
    if args.limit < 1 or args.max_download_mb < 1:
        parser.error("--limit and --max-download-mb must be positive")
    if PurePosixPath(args.archive).name != args.archive or not args.archive.endswith(".zip"):
        parser.error("--archive must be a ZIP filename without path components")
    if args.output.name == "example_graph.json":
        parser.error("Do not overwrite the manually synthesized application demo graph")
    revision = args.revision
    source_file: BinaryIO
    if args.local_zip:
        source_file = args.local_zip.open("rb")
        args.archive = args.local_zip.name
        import_method = "local-zip-text-only"
    else:
        context = ssl_context()
        info = fetch_json(f"{HUB}/api/datasets/{REPO_ID}/revision/{quote(revision, safe='')}", context)
        if not isinstance(info, dict) or not isinstance(info.get("sha"), str):
            raise ValueError("Cannot pin the dataset revision")
        revision = info["sha"]
        tree = fetch_json(f"{HUB}/api/datasets/{REPO_ID}/tree/{revision}", context)
        entry = next((entry for entry in tree if entry.get("path") == args.archive), None)
        if entry is None or entry.get("type") != "file":
            raise ValueError(f"Official archive does not exist: {args.archive}")
        archive_url = f"{HUB}/datasets/{REPO_ID}/resolve/{revision}/{quote(args.archive, safe='')}"
        source_file = RangeFile(archive_url, entry["size"], args.max_download_mb * 1024 * 1024, context)
        import_method = "http-range-text-only"
    try:
        with zipfile.ZipFile(source_file) as archive:
            records, available = import_texts(archive, args.archive, revision, args.limit)
    finally:
        source_file.close()
    if not records:
        raise ValueError("No complete goal.txt/steps.txt pairs found; no output written")
    result = {
        "schema_version": 1,
        "format": "recipegen-text-staging-v1",
        "dataset": {
            "id": REPO_ID, "is_demo": False,
            "description": "Official RecipeGen text subset; not a complete ingredient graph or a verified cooking guide.",
            "source": f"{HUB}/datasets/{REPO_ID}",
            "license": {"hub_metadata": "cc-by-4.0", "dataset_card_text": "CC BY-NC 4.0", "conflict": True},
        },
        "import": {
            "method": import_method, "revision": revision, "archive": args.archive,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "sampling": "first N complete text pairs in lexical directory order",
            "available_text_pairs": available, "record_count": len(records),
            "zip_range_bytes": getattr(source_file, "downloaded", 0),
            "zip_range_requests": getattr(source_file, "requests", 0),
            "media_members_read": 0,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **result["import"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        raise SystemExit(1)
