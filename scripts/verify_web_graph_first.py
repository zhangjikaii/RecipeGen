#!/usr/bin/env python3
"""Read-only live Neo4j regression plus opt-in real web API QA validation.

No credentials are included in reports. Queries and labels reuse the frozen
condition benchmark; this is a regression, not a new semantic evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.catalog import RecipeCatalog
from recipegen.config import Settings
from recipegen.graphrag_retrieval import RecipeGraphRAGRetriever
from recipegen.text_embeddings import MODEL, REVISION
from scripts.evaluate_real_retrieval import lines, write_json, digest, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-api", action="store_true", help="Run authorized real API reranking and QA via the local website")
    parser.add_argument("--report-label", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    import httpx
    tick = time.monotonic()
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "live_Neo4j_graph_first_regression_and_web_DeepSeek_API",
              "human_annotated": False, "semantic_accuracy_verified": False,
              "queries_are_previously_viewed_frozen_regression": True,
              "api_requested": args.run_api, "checks": {}, "queries": []}
    target = ROOT / f"reports/web-graph-first-{args.report_label}.json"
    settings = Settings.from_env()
    catalog = RecipeCatalog(settings=settings)

    def check(name, condition):
        report["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    try:
        retriever = RecipeGraphRAGRetriever(catalog)
        state = catalog.status()
        report["graph"] = {"backend": state["backend"], "build": state["build"], "counts": state["graph"]}
        check("real_recipe_count_5898", state["graph"]["recipes"] == 5898)
        check("real_step_count_44802", state["graph"]["steps"] == 44802)
        data = ROOT / "data/evaluation_hybrid_v2"
        queries = [q for q in lines(data / "queries.jsonl") if q["category"] in {"multi_ingredient", "ingredient_exclusion"}]
        vectors = {q["query_id"]: q for q in lines(data / "query_embeddings.jsonl")}
        expected = {r["query_id"]: r for r in lines(ROOT / "data/evaluation_graph_first_v1/rankings.jsonl")
                    if r["method"] == "C_graph_then_e5"}
        gold = defaultdict(set)
        for qrel in lines(data / "qrels.jsonl"):
            gold[qrel["query_id"]].add(qrel["recipe_id"])
        for number, query in enumerate(queries, 1):
            qid = query["query_id"]
            found = retriever.search(query["query"], ingredients=query["ingredients"],
                                     excluded_ingredients=query["excluded_ingredients"], limit=10,
                                     mode="semantic", query_vector=vectors[qid]["embedding"],
                                     embedding_model=MODEL, embedding_revision=REVISION)
            ids = [r["id"] for r in found["recipes"]]
            expected_ids = [r["recipe_id"] for r in expected[qid]["ranking"][:10]]
            matches = len(set(ids) & gold[qid])
            metadata = found["retrieval"]
            check(qid + "_graph_first_method", metadata["method"] == "graph_first_e5_exact_cosine")
            check(qid + "_full_graph_candidate_scoring", metadata["graph_candidate_count"] == metadata["scored_recipe_count"] == expected[qid]["predicate_candidate_count"])
            check(qid + "_gold_hits_match_offline", matches == expected[qid]["gold_hits_at_10"])
            report["queries"].append({"query_id": qid, "split": query["split"], "category": query["category"],
                "returned_ids": ids, "exact_order_matches_offline": ids == expected_ids,
                "same_top10_set": set(ids) == set(expected_ids), "graph_candidates": metadata["graph_candidate_count"],
                "hits_at_10": matches, "precision_at_10": matches / 10,
                "recall_at_10": matches / len(gold[qid]), "empty_result": not ids})
            if number % 10 == 0:
                log("live_neo4j_regression", completed=number, total=len(queries))
        test = [q for q in report["queries"] if q["split"] == "test"]
        report["test_metrics"] = {"queries": len(test),
            "precision_at_10": sum(q["precision_at_10"] for q in test) / len(test),
            "recall_at_10": sum(q["recall_at_10"] for q in test) / len(test),
            "empty_results": sum(q["empty_result"] for q in test)}
        report["exact_order_matches_offline_queries"] = sum(q["exact_order_matches_offline"] for q in report["queries"])
        report["same_top10_set_queries"] = sum(q["same_top10_set"] for q in report["queries"])
        with httpx.Client(base_url="http://127.0.0.1:8765", timeout=150, trust_env=False) as client:
            status = client.get("/api/system/status")
            check("web_status_http200", status.status_code == 200)
            public_status = status.json()
            check("web_default_semantic", public_status["retrieval"]["default_mode"] == "semantic")
            check("web_api_rerank_and_question_answering_ready", public_status["generation"]["api_configured"]
                  and public_status["generation"]["api_rerank_supported"] and public_status["generation"]["api_question_answering_supported"])
            search = client.post("/api/search", json={"query": "同时包含西兰花和胡萝卜的食谱", "limit": 10})
            check("web_search_http200", search.status_code == 200)
            found = search.json()
            check("web_search_graph40_scored40", found["retrieval"]["graph_candidate_count"] == found["retrieval"]["scored_recipe_count"] == 40)
            check("web_search_returns10", found["count"] == 10)
            report["web_search"] = {"retrieval": found["retrieval"], "filters": found["applied_filters"],
                                    "recipe_ids": [r["id"] for r in found["recipes"]]}
            if args.run_api:
                payload = {"question": "请推荐一道同时包含西兰花和胡萝卜的食谱，根据原始步骤说明关键做法并标注来源。",
                           "ingredients": ["西兰花", "胡萝卜"], "retrieval_mode": "semantic", "mode": "api", "limit": 1}
                response = client.post("/api/generate", json=payload)
                check("full_flow_http200", response.status_code == 200)
                result = response.json()
                report["api_request"] = payload
                report["api_response"] = result
                check("full_flow_graph_first40", result["retrieval"]["method"] == "graph_first_e5_exact_cosine" and result["retrieval"]["graph_candidate_count"] == 40)
                check("api_rerank_called_and_valid", result["rerank"]["api_call_attempted"] and result["rerank"]["status"] == "ok" and result["rerank"]["validation"]["candidate_permutation_passed"])
                check("api_qa_called_and_answered", result["qa"]["generation"]["api_call_attempted"] and result["qa"]["status"] == "answered")
                check("api_two_calls_success_no_fallback", result["generation"]["llm_called"] and result["generation"]["api_calls_attempted"] == 2 and result["generation"]["fallback"] is False and result["generation"]["effective_mode"] == "graphrag_api_evidence_qa")
                evidence = {e["id"]: e for e in result["evidence"]}
                source_records = {r["id"]: r for r in result["recipes"]}
                citations = [eid for p in result["qa"]["paragraphs"] for eid in p["evidence_ids"]]
                check("qa_citations_resolve", bool(citations) and all(eid in evidence for eid in citations))
                for eid in citations:
                    item = evidence[eid]
                    original = catalog.recipe(item["recipe_id"])
                    check("citation_" + eid + "_exact_original_source", any(
                        step["id"] == item["graph_id"] and step["text"] == item["text"] and step["source_id"] == item["source_id"]
                        for step in original["steps"]))
                check("question_answer_cites_chosen_recipe", set(source_records) <= set(result["rerank"]["ordered_recipe_ids"][:1]))
        report["input_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in [data / "queries.jsonl", data / "qrels.jsonl", data / "query_embeddings.jsonl", ROOT / "data/evaluation_graph_first_v1/rankings.jsonl"]}
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        if isinstance(error, AssertionError):
            report["failed_check"] = str(error)
        raise
    finally:
        catalog.close()
        report["elapsed_seconds"] = round(time.monotonic() - tick, 3)
        write_json(target, report)
        log("web_graph_first_verification", status=report["status"], report=str(target),
            metrics=report.get("test_metrics"), checks=len(report["checks"]))


if __name__ == "__main__":
    main()
