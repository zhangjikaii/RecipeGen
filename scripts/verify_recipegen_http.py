#!/usr/bin/env python3
"""在实际运行的网页 API 上验收检索、来源、生成与可选本地模型。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--local", action="store_true", help="额外执行一次已有本地 Qwen 模型推理")
    args = parser.parse_args()

    def request(path, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        req = Request(args.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=180) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    code, status = request("/api/system/status")
    assert code == 200 and status["graph"]["recipes"] == 5898 and not status["dataset"]["is_demo"]
    assert status["media_processing"]["status"] == "stopped_by_user"
    code, search = request("/api/search", {"query": "番茄 鸡蛋", "limit": 3})
    assert code == 200 and search["count"] > 0
    recipe_id = search["recipes"][0]["id"]
    code, recipe = request("/api/recipes/" + recipe_id)
    assert code == 200 and recipe["steps"] and recipe["sources"]
    code, graph = request("/api/system/graph?recipe_id=" + recipe_id)
    assert code == 200 and graph["nodes"] and graph["edges"]
    payload = {"question": "按顺序整理这道食谱的原始做法，并列出来源", "recipe_ids": [recipe_id], "limit": 1}
    code, generated = request("/api/generate", payload)
    assert code == 200 and generated["status"] == "ok" and generated["validation"]["passed"]
    assert generated["generation"]["llm_called"] is False
    assert all(step["text"] in generated["answer"] for step in recipe["steps"])
    assert generated["evidence"] and all(item.get("graph_id") for item in generated["evidence"])
    code, excluded = request("/api/search", {"query": "番茄，不要鸡蛋", "limit": 3})
    assert code == 200 and all("egg" not in {m["name"] for m in row["ingredient_mentions"]} for row in excluded["recipes"])
    assert request("/api/search", {"query": "番茄", "limit": 0})[0] == 422
    assert request("/api/recipes/missing-recipe")[0] == 404
    assert request("/api/generate", {**payload, "mode": "unsupported"})[0] == 422
    # 已停止批处理的成功检查点仅用于选择既有视觉记录，不重新识别媒体。
    progress = json.loads((ROOT / "reports/multimodal-progress.json").read_text())
    checkpoint = ROOT / "data/multimodal/runs" / progress["pipeline_run_id"] / "state.json"
    records = json.loads(checkpoint.read_text()).get("records", {})
    visual_recipe_id = next(key for key, value in records.items()
                            if value.get("status") == "verified" and value.get("counts", {}).get("images", 0) > 0)
    code, visual_recipe = request("/api/recipes/" + visual_recipe_id)
    assert code == 200 and visual_recipe["visual_evidence"]
    code, titled = request("/api/search", {"query": visual_recipe["title"], "limit": 6})
    assert code == 200 and titled["recipes"] and titled["recipes"][0]["id"] == visual_recipe_id
    assert all(item["candidate"] is True and item["caption"] and item["source_id"]
               for item in visual_recipe["visual_evidence"])
    code, visual_graph = request("/api/system/graph?recipe_id=" + visual_recipe_id)
    assert code == 200 and any(node["label"] == "VisualObservation" for node in visual_graph["nodes"])
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "url": args.url,
              "status": status, "search": search, "recipe": recipe, "graph": graph,
              "generation": generated, "negated_ingredient_search": excluded,
              "existing_visual_recipe": visual_recipe, "existing_visual_graph": visual_graph,
              "exact_title_search": titled,
              "checks": {"native_catalog": True, "source_traceability": True, "grounded_generation": True,
                         "original_steps_preserved": True, "existing_visual_candidates": True,
                         "exact_title_ranked_first": True,
                         "bad_input_422": True, "missing_recipe_404": True},
              "semantic_accuracy_verified": False}
    (ROOT / "reports/system-http-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"http_verified": True, "recipes": status["graph"]["recipes"], "first_recipe": recipe["title"],
                      "checks": report["checks"]}, ensure_ascii=False), flush=True)
    if args.local:
        print("正在进行一次真实本地模型推理…", flush=True)
        code, local = request("/api/generate", {**payload, "mode": "local"})
        verified = code == 200 and local["generation"]["llm_called"] and not local["generation"]["fallback"]
        local_report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "http_status": code,
                        "actual_model_result": local, "local_model_verified": bool(verified),
                        "semantic_accuracy_verified": False}
        (ROOT / "reports/system-local-model-smoke.json").write_text(json.dumps(local_report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"local_model_verified": verified, "generation": local.get("generation"),
                          "validation": local.get("validation")}, ensure_ascii=False, indent=2))
        return 0 if verified else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
