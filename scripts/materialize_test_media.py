#!/usr/bin/env python3
"""Fetch selected test media with bounded HTTP Range reads and ZIP CRC checks.

Only explicitly selected members are read; no full archive download is allowed.
Use for viewing evidence or later visual extraction. This does not run a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from import_recipegen import RangeFile, ssl_context

ALLOWED = ("test.zip", "test-video.zip")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=ALLOWED, required=True)
    parser.add_argument("--member", action="append", required=True, help="Exact raw member name from the member inventory")
    parser.add_argument("--source-dir", type=Path, default=Path("data/test_graph/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_graph/media_samples"))
    parser.add_argument("--max-download-mb", type=int, default=96)
    parser.add_argument("--max-file-mb", type=int, default=64)
    args = parser.parse_args()
    if min(args.max_download_mb, args.max_file_mb) < 1:
        parser.error("Byte budgets must be positive")
    manifest = json.loads((args.source_dir / f"{args.archive[:-4]}.manifest.json").read_text())
    if manifest["split"] != "test" or manifest["archive"] != args.archive or manifest["status"] != "complete":
        raise ValueError("A complete immutable test archive inventory is required")
    index = {e["member"]: e for e in (json.loads(line) for line in (args.source_dir / f"{args.archive[:-4]}.members.jsonl").read_text().splitlines())}
    for member in args.member:
        if member not in index or index[member]["kind"] not in ("image", "video"):
            raise ValueError(f"Not a verified test media member: {member}")
        if index[member]["size"] > args.max_file_mb * 1024 * 1024:
            raise ValueError(f"Selected file exceeds per-file limit: {member}")
    url = f"https://huggingface.co/datasets/RUOXUAN123/RecipeGen/resolve/{manifest['revision']}/{args.archive}"
    remote = RangeFile(url, manifest["archive_size"], args.max_download_mb * 1024 * 1024, ssl_context())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with zipfile.ZipFile(remote) as archive:
        for member in args.member:
            info = archive.getinfo(member)
            expected = index[member]
            if info.file_size != expected["size"] or f"{info.CRC:08x}" != expected["crc"]:
                raise ValueError("Member inventory mismatch")
            suffix = Path(expected["display_path"]).suffix.lower()
            target = args.output_dir / (hashlib.sha256(f"{args.archive}\n{member}".encode()).hexdigest()[:24] + suffix)
            part = target.with_suffix(target.suffix + ".part")
            h = hashlib.sha256()
            with archive.open(info) as source, part.open("wb") as out:
                while chunk := source.read(1024 * 1024):
                    h.update(chunk)
                    out.write(chunk)
            # ZIP CRC is checked by zipfile on EOF before this rename.
            part.replace(target)
            results.append({"archive": args.archive, "member": member, "revision": manifest["revision"], "split": "test", "path": str(target.resolve()),
                            "size": target.stat().st_size, "crc32": expected["crc"], "sha256": h.hexdigest(), "zip_crc_verified": True})
    report = {"media": results, "range_requests": remote.requests, "downloaded_range_bytes": remote.downloaded, "visual_model_calls": 0}
    report_path = args.output_dir / "materialization-report.json"
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        all_rows = {(r["archive"], r["member"]): r for r in prior.get("media", []) + results}
        report["media"] = list(all_rows.values())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
