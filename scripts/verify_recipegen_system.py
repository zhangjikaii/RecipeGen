#!/usr/bin/env python3
"""Explicit real Neo4j/local-model smoke verification, never media processing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Read the active user Neo4j catalog")
    parser.add_argument("--local", action="store_true", help="Run one real, offline Qwen evidence selection")
    args = parser.parse_args()
    if not args.native and not args.local:
        parser.error("Choose --native or --local explicitly")
    from recipegen.catalog import RecipeCatalog
    from recipegen.config import Settings
    from recipegen.system_generation import RecipeGenerator

    generator = RecipeGenerator(project_root=ROOT)
    if args.native:
        catalog = RecipeCatalog(settings=Settings.from_env())
        try:
            status = catalog.status()
            search = catalog.search("番茄 鸡蛋", limit=3)
            assert search["recipes"], "Expected a real tomato/egg recipe in the fixed graph"
            recipe = catalog.recipe(search["recipes"][0]["id"])
            graph = catalog.graph(recipe["id"])
            generated = generator.generate("根据原图谱整理番茄和鸡蛋的食谱做法", search["recipes"], mode="grounded", limit=2)
            excluded = catalog.search("番茄，不要鸡蛋", limit=3)
            assert all("egg" not in {m["name"] for m in row["ingredient_mentions"]} for row in excluded["recipes"])
            assert status["graph"]["recipes"] == 5898 and status["dataset"]["is_demo"] is False
            assert generated["status"] == "ok" and generated["validation"]["passed"]
            assert generated["generation"]["llm_called"] is False
            report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                      "status": status, "search": search, "recipe": recipe, "graph": graph,
                      "generation": generated, "negated_ingredient_search": excluded,
                      "media_processing": json.loads((ROOT / "reports/multimodal-progress.json").read_text())["status"],
                      "read_only": True, "semantic_accuracy_verified": False}
            assert report["media_processing"] == "stopped_by_user"
            (ROOT / "reports/system-native-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"native_verified": True, "graph": status["graph"], "search_count": search["count"],
                              "first_recipe": recipe["title"], "steps": len(recipe["steps"]),
                              "visual_candidates": len(recipe["visual_evidence"]),
                              "graph_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]),
                              "grounded_generation_verified": True, "media_processing": report["media_processing"]},
                             ensure_ascii=False, indent=2))
        finally:
            catalog.close()
    if args.local:
        source = json.loads((ROOT / "reports/system-native-smoke.json").read_text())
        generated = generator.generate("请根据原步骤和来源组织这道食谱的做法", source["search"]["recipes"][:1], mode="local", limit=1)
        report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "actual_model_result": generated, "semantic_accuracy_verified": False,
                  "local_model_verified": generated["generation"]["llm_called"] and not generated["generation"]["fallback"]}
        (ROOT / "reports/system-local-model-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"local_model_verified": report["local_model_verified"], "generation": generated["generation"],
                          "validation": generated["validation"], "status": generated["status"]}, ensure_ascii=False, indent=2))
        return 0 if report["local_model_verified"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
