#!/usr/bin/env python3
"""Construct graph artifacts from the complete RecipeGen test corpus."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recipegen.test_graph import build_graph

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/test_graph/records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/test_graph"))
    args = parser.parse_args()
    report = build_graph(args.records, args.output)
    print(json.dumps({k: report[k] for k in ("build_id", "dataset_revision", "records_seen", "node_count", "relationship_count", "node_counts", "media_counts", "associated_media_counts")}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
