#!/usr/bin/env python3
"""Prepare verified test media + real PTS frames; no model/database calls.

Small standalone probe by default. The full model runner should use
MediaPreparer.prepare_record -> inference/checkpoint -> cleanup_record.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.multimodal_media import MediaLimits, MediaPreparer, select_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--corpus-dir", default="data/test_graph")
    parser.add_argument("--workspace-dir", default="data/multimodal/work")
    parser.add_argument("--limit", type=int, default=1, help="0 means all recipes; default 1 for preparation-only validation")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--recipe-id")
    parser.add_argument("--cleanup", action="store_true", help="Remove recipe temporary media after recording its evidence")
    parser.add_argument("--max-file-mb", type=int, default=1024)
    parser.add_argument("--max-recipe-mb", type=int, default=2048)
    parser.add_argument("--max-download-mb", type=int, default=2049)
    parser.add_argument("--max-clips", type=int, default=2000)
    parser.add_argument("--max-frames", type=int, default=6000)
    args = parser.parse_args()
    if args.limit == 0 and not args.cleanup and not args.recipe_id:
        parser.error("Full preparation-only runs require --cleanup; use the streaming model API to consume frames before cleanup")
    limits = MediaLimits(max_file_bytes=args.max_file_mb * 1024 * 1024, max_recipe_bytes=args.max_recipe_mb * 1024 * 1024, max_download_bytes=args.max_download_mb * 1024 * 1024, max_clips=args.max_clips, max_frames=args.max_frames)
    failed = prepared = 0
    with MediaPreparer(args.project_root, args.corpus_dir, args.workspace_dir, limits) as helper:
        records = select_records(helper.corpus / "records.jsonl", args.limit, args.offset, args.recipe_id)
        for record in records:
            row = helper.prepare_record(record)
            prepared += row["status"] == "complete"
            failed += row["status"] != "complete"
            if args.cleanup:
                helper.cleanup_record(record["id"])
            print(json.dumps({"recipe_id": row["recipe_id"], "status": row["status"], "counts": row["counts"], "errors": row["errors"]}, ensure_ascii=False), flush=True)
        print(json.dumps({"prepared": prepared, "failed": failed, "evidence": str(helper.output_path), "models_run": 0, "database_writes": 0}, ensure_ascii=False))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
