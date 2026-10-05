from __future__ import annotations

import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from .config import PROJECT_ROOT
from .models import RecommendRequest
from .pipeline import Recommender


def evaluate(service: Recommender, cases_path: Path | None = None, use_llm: bool = False) -> dict:
    cases_path = cases_path or PROJECT_ROOT / "data" / "evaluation_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    rows = []
    for case in cases:
        kind = case["category"].split("/", 1)[0]
        request = RecommendRequest(question=case["question"], constraints=case["constraints"] if kind == "structured" else None, use_llm=use_llm)
        result = service.recommend(request)
        from .models import Constraints
        actual = service.store.retrieve(Constraints.model_validate(result["constraints"])) if result["status"] != "needs_clarification" else []
        ids = sorted(recipe.id for recipe in actual)
        expected = sorted(case["expected_recipe_ids"])
        status_ok = result["status"] == case["expected_status"]
        candidates_ok = ids == expected
        recommendation_ids = [recipe["recipe_id"] for recipe in result["recommendations"]]
        recommendations_ok = set(recommendation_ids).issubset(expected) and len(recommendation_ids) <= result["constraints"]["top_k"]
        constraint_ok = result["validation"]["passed"]
        passed = status_ok and candidates_ok and recommendations_ok and constraint_ok
        row = {
            "id": case["id"], "category": case["category"], "question": case["question"], "passed": passed,
            "actual_status": result["status"], "expected_status": case["expected_status"],
            "candidate_set_match": candidates_ok, "actual_candidate_ids": ids, "expected_candidate_ids": expected,
            "recommendation_ids": recommendation_ids, "validation_passed": constraint_ok,
            "parsed_constraints": result["constraints"], "expected_constraints": case["constraints"],
            "generation_mode": result["generation"]["mode"], "elapsed_ms": result["elapsed_ms"],
        }
        rows.append(row)
        groups[kind].append(row)
    summary = {}
    for name, values in groups.items():
        summary[name] = {
            "total": len(values), "passed": sum(v["passed"] for v in values),
            "pass_rate": round(sum(v["passed"] for v in values) / len(values), 4),
            "candidate_set_accuracy": round(sum(v["candidate_set_match"] for v in values) / len(values), 4),
            "median_latency_ms": round(median(v["elapsed_ms"] for v in values), 2),
        }
    document = service.store.document()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "backend": service.settings.graph_backend, "dataset": document.dataset.model_dump(),
        "dataset_sha256": hashlib.sha256(document.model_dump_json().encode()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "llm_requested": use_llm, "llm_configured": service.settings.llm_configured,
        "successful_llm_calls": len(service.llm.calls), "generation_modes": dict((mode, sum(row["generation_mode"] == mode for row in rows)) for mode in {row["generation_mode"] for row in rows}),
        "scope": "手工示例图谱上的功能回归；不代表真实 RecipeGen 全量数据、真实 Neo4j 或 LLM 泛化能力" if document.dataset.is_demo else "当前导入图谱上的功能测试；评测集仍需检查是否与数据匹配",
        "summary": summary, "total": len(rows), "passed": sum(row["passed"] for row in rows), "cases": rows,
    }
