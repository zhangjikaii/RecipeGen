#!/usr/bin/env python3
"""Run the authorized full test batch once, then independently audit Neo4j."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-run", help="兼容旧批次的真实观测缓存")
    parser.add_argument("--modality", choices=("all", "images"), default="all")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    command = [sys.executable, str(ROOT / "scripts/run_multimodal_pipeline.py"),
               "--limit", "0", "--import-neo4j", "--modality", args.modality]
    if args.reuse_run:
        command.extend(["--reuse-run", args.reuse_run])
    worker = subprocess.run(command, cwd=ROOT)
    audit_path = ROOT / "reports/multimodal-full-audit.json"
    audit = subprocess.run([sys.executable, str(ROOT / "scripts/verify_multimodal_progress.py"),
                            "--neo4j", "--output", str(audit_path)], cwd=ROOT)
    audited = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    completion_key = "completed_image_scope" if args.modality == "images" else "completed_full_scope"
    complete = worker.returncode == 0 and audit.returncode == 0 and audited.get(completion_key) is True
    report = {"status": "completed" if complete else "incomplete", "completed_full_scope": complete and args.modality == "all",
              "completed_image_scope": complete and args.modality == "images", "processing_modalities": ["image"] if args.modality == "images" else ["image", "video"],
              "worker_exit_code": worker.returncode, "audit_exit_code": audit.returncode,
              "audit": str(audit_path.relative_to(ROOT)), "started_at": started,
              "finished_at": datetime.now(timezone.utc).isoformat(), "semantic_accuracy_verified": False}
    target = ROOT / "reports/multimodal-batch-result.json"
    part = target.with_suffix(".json.part")
    part.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    part.replace(target)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
