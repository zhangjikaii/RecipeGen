#!/usr/bin/env python3
"""Freeze one chat case, then opt in to the real hybrid-to-API serial smoke.

Preparation is local and never loads credentials or calls a model. Execution
uses the existing local settings, makes one FastAPI chat request, and records
only the public response plus allowlisted API usage. There are no retries.
This verifies one integration case, not semantic accuracy or retrieval P/R.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "recipegen-web-hybrid-serial-plan-v1"
QUESTION = "请从食谱图谱推荐一道同时使用西兰花和胡萝卜、不要牛奶的菜，依据原始步骤说明主要处理和烹调操作，并标明来源。"
CODE_FILES = ("recipegen/app.py", "recipegen/chef_chat.py", "recipegen/graphrag_retrieval.py",
              "recipegen/web_rag_answer.py", "recipegen/experimental_qa.py", "recipegen/llm.py",
              "recipegen/config.py", "recipegen/catalog.py", "recipegen/production_hybrid.py",
              "recipegen/text_embeddings.py",
              "scripts/verify_web_hybrid_serial.py")
DATA_FILES = ("data/graphrag/index-manifest.json", "data/graphrag/full/corpus.jsonl")
STAGES = ("llm_routing", "rrf_fusion", "graph_evidence_expansion", "graph_evidence",
          "bounded_api_context", "api_rerank", "api_qa_context", "api_question_answering",
          "api_answer_validation")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    return {name: digest(ROOT / name) for name in CODE_FILES}


def data_identity():
    return {name: digest(ROOT / name) for name in DATA_FILES}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_api_usage(app):
    """Keep successful call records even when an earlier result check fails."""
    calls = app.state.chat_service.llm_client.calls + app.state.api_answerer.qa.llm_client.calls
    return [{key: value for key, value in call.items()
             if key in {"stage", "configured", "provider", "model", "usage", "success"}}
            for call in calls]


def prepare(path, expected_method):
    if path.exists():
        raise ValueError("plan_already_exists")
    plan = {"schema": SCHEMA, "status": "prepared", "prepared_at": now(),
            "request": {"message": QUESTION, "history": []},
            "endpoint": "/api/chat", "expected_method": expected_method,
            "expected_required_ingredients": ["broccoli", "carrot"],
            "expected_excluded_ingredients": ["milk"], "ordered_stages": list(STAGES),
            "maximum_chat_requests": 1, "maximum_model_calls": 3, "retries": 0,
            "protocol": {"question_frozen_before_model_output": True,
                         "route_must_retrieve": True, "production_hybrid_required": True,
                         "semantic_accuracy_evaluated": False, "human_labeled": False,
                         "retrieval_precision_recall_evaluated": False,
                         "context_budget_unit": "characters_not_model_tokens",
                         "no_response_dependent_question_or_candidate_repair": True},
            "code_identity": code_identity(), "data_identity": data_identity()}
    write_json(path, plan)
    return {"status": "prepared", "plan": str(path.resolve()), "maximum_model_calls": 3}


def run(plan_path, output):
    if output.exists():
        raise ValueError("output_already_exists")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("schema") != SCHEMA or plan.get("status") != "prepared"
            or plan.get("maximum_model_calls") != 3 or plan.get("maximum_chat_requests") != 1
            or plan.get("retries") != 0 or plan["code_identity"] != code_identity()
            or plan["data_identity"] != data_identity()):
        raise ValueError("frozen_protocol_or_code_changed")
    from fastapi.testclient import TestClient
    from recipegen.app import create_app
    from recipegen.catalog import normalize_ingredient
    from recipegen.experimental_qa import load_experimental_settings, safe_configuration

    settings = load_experimental_settings()
    if not settings.llm_configured or settings.llm_provider != "chat_completions":
        raise ValueError("configured_API_required")
    report = {"schema": "recipegen-web-hybrid-serial-result-v1", "status": "running",
              "started_at": now(), "plan": str(plan_path.resolve()), "plan_sha256": digest(plan_path),
              "configuration": safe_configuration(settings), "protocol": plan["protocol"],
              "request": plan["request"], "code_identity": plan["code_identity"],
              "data_identity": plan["data_identity"],
              "checks": {}, "API_usage": [], "human_labeled": False,
              "semantic_accuracy_evaluated": False, "retrieval_precision_recall_evaluated": False}

    def check(name, condition):
        report["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    app = create_app(settings)
    try:
        with TestClient(app) as client:
            response = client.post(plan["endpoint"], json=plan["request"])
            report["http_status"] = response.status_code
            result = response.json()
            # Never persist credentials, including accidental echoes in error data.
            serialized = json.dumps(result, ensure_ascii=False)
            secrets = [settings.llm_api_key, settings.neo4j_password]
            check("public_response_has_no_configured_credential", not any(s and s in serialized for s in secrets))
            report["response"] = result
            check("chat_HTTP200", response.status_code == 200)
            check("route_is_LLM_retrieval", result["route"]["action"] == "retrieve" and result["route"]["llm_decided"])
            check("routing_required_ingredients_preserved", set(map(normalize_ingredient, result["route"]["ingredients"])) == set(plan["expected_required_ingredients"]))
            check("routing_excluded_ingredients_preserved", set(map(normalize_ingredient, result["route"]["excluded_ingredients"])) == set(plan["expected_excluded_ingredients"]))
            retrieval = result["retrieval"]
            check("production_hybrid_method", retrieval["method"] == plan["expected_method"])
            check("graph_candidates_scored_before_limits", retrieval["graph_candidate_count"] == retrieval["scored_recipe_count"] and retrieval["scored_recipe_count"] > 0)
            check("RRF_is_present", retrieval.get("rrf_k") == 60)
            actual_stages = [item["stage"] for item in result["trace"]]
            report["trace_stages"] = actual_stages
            check("all_serial_stages_recorded", all(stage in actual_stages for stage in plan["ordered_stages"]))
            positions = [actual_stages.index(stage) for stage in plan["ordered_stages"]]
            check("serial_stage_order", positions == sorted(positions))
            check("three_model_calls_no_fallback", result["generation"]["api_calls_attempted"] == 3 and result["generation"]["api_status"] == "ok" and result["generation"]["fallback"] is False)
            check("rerank_valid", result["rerank"]["status"] == "ok" and result["rerank"]["validation"]["candidate_permutation_passed"])
            check("QA_answered_and_structurally_valid", result["qa"]["status"] == "answered" and result["validation"]["passed"])
            check("rerank_context_bounded_complete", result["rerank_context"]["complete_original_steps"] and result["rerank_context"]["supplied_context_chars"] <= result["rerank_context"]["max_context_chars"])
            check("QA_context_bounded_complete", result["qa_context"]["complete_original_steps"] and result["qa_context"]["supplied_context_chars"] <= result["qa_context"]["max_context_chars"])
            check("rerank_context_hash_is_actual_input", result["rerank_context"]["context_sha256"] == result["rerank"]["context_sha256"])
            check("QA_context_hash_is_actual_input", result["qa_context"]["context_sha256"] == result["qa"]["context_sha256"])
            evidence = {item["id"]: item for item in result["evidence"]}
            citations = [eid for p in result["qa"]["paragraphs"] for eid in p["evidence_ids"]]
            check("answer_citations_resolve", bool(citations) and all(eid in evidence for eid in citations))
            originals = {}
            for eid in set(citations):
                item = evidence[eid]
                if item["recipe_id"] not in originals:
                    originals[item["recipe_id"]] = app.state.catalog.recipe(item["recipe_id"])
                check("citation_" + eid + "_matches_live_original_source", any(
                    step["id"] == item["graph_id"] and step["text"] == item["text"] and step["source_id"] == item["source_id"]
                    for step in originals[item["recipe_id"]]["steps"]))
            state = app.state.catalog.status()
            report["live_graph"] = {"backend": state["backend"], "build": state["build"], "counts": state["graph"]}
            check("live_Neo4j_graph", state["backend"] == "neo4j" and state["graph"]["recipes"] == 5898 and state["graph"]["steps"] == 44802)
            report["API_usage"] = safe_api_usage(app)
            check("three_successful_JSON_API_records", len(report["API_usage"]) == 3 and all(call.get("success") for call in report["API_usage"]))
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        if isinstance(error, AssertionError):
            report["failed_check"] = str(error)
    finally:
        report["API_usage"] = safe_api_usage(app)
        report["finished_at"] = now()
        report["code_identity_unchanged_after_run"] = code_identity() == plan["code_identity"]
        report["data_identity_unchanged_after_run"] = data_identity() == plan["data_identity"]
        report["plan_hash_unchanged_after_run"] = digest(plan_path) == report["plan_sha256"]
        if not all(report[name] for name in ("code_identity_unchanged_after_run",
                                             "data_identity_unchanged_after_run",
                                             "plan_hash_unchanged_after_run")):
            report["status"] = "failed"
            report["identity_failure"] = "frozen_protocol_or_code_changed_during_run"
        text = json.dumps(report, ensure_ascii=False)
        if any(secret and secret in text for secret in [settings.llm_api_key, settings.neo4j_password]):
            # Drop all response material instead of attempting fragile partial redaction.
            report = {"schema": report["schema"], "status": "failed", "error_type": "sensitive_report_blocked"}
        write_json(output, report)
    return {"status": report["status"], "report": str(output.resolve()),
            "checks": len(report.get("checks", {})), "failed_check": report.get("failed_check")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-method", default="graph_first_bm25_e5_rrf")
    args = parser.parse_args()
    try:
        if args.prepare:
            result = prepare(args.plan, args.expected_method)
        else:
            if not args.output:
                parser.error("--run requires --output")
            result = run(args.plan, args.output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] == "failed":
            raise SystemExit(1)
    except Exception:
        print(json.dumps({"status": "failed", "error": "inspect_nonsecret_protocol_or_configuration"}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
