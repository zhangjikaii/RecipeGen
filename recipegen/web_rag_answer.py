"""Web GraphRAG: API reranking followed by source-bound question answering.

Reuse the tested JSON, candidate-permutation, citation and number contracts.
The API never supplies new recipe IDs or cooking facts to the graph.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .experimental_qa import ExperimentalEvidenceQA, build_qa_context, ExperimentalQAError, json_hash
from .system_generation import RecipeGenerator, _candidate_records, _terms

MAX_RERANK_CANDIDATES = 20
LIMITATIONS = [
    "问答依据检索食谱的原始步骤，引用可展开核对；程序校验引用与数字，语义仍需核对原文。",
    "食材条件按图谱提及关系筛选，不能证明完整食材清单或饮食安全。",
]


class WebRAGAnswerer:
    def __init__(self, project_root: str | Path | None = None, *, settings: Settings | None = None,
                 qa: ExperimentalEvidenceQA | None = None):
        self.settings = settings or getattr(qa, "settings", None) or Settings()
        self.qa = qa or ExperimentalEvidenceQA(settings=self.settings, allow_api_calls=True)
        self.grounded = RecipeGenerator(project_root)

    def status(self) -> dict[str, Any]:
        state = self.qa.status()
        return {"api_configured": state["configured"],
                "api_status": "configured" if state["configured"] else "not_configured",
                "provider": state["provider"], "model": state["model"],
                "default_effective_mode": "graphrag_api_evidence_qa" if state["configured"] else "grounded",
                "structured_output_schema": state["schema"], "api_rerank_supported": True,
                "api_question_answering_supported": True, "local_model_used": False,
                "semantic_accuracy_verified": False, "limitations": list(LIMITATIONS)}

    @staticmethod
    def _public_api_result(result):
        public = copy.deepcopy(result)
        if public.get("status") not in {"ok", "answered", "insufficient_evidence"}:
            public["response_json"] = None
            if isinstance(public.get("generation"), dict):
                public["generation"]["response_json"] = None
        return public

    @staticmethod
    def _evidence(items, records):
        by_id = {r["id"]: r for r in records}
        public = []
        for original in items:
            item = copy.deepcopy(original)
            item.update(kind="step", graph_id=item["step_id"], text=item["original_text"])
            source = next((s for s in by_id[item["recipe_id"]]["sources"]
                           if s.get("source_id") == item["source_id"]), None)
            if source and isinstance(source.get("url"), str):
                item["source_url"] = source["url"]
            public.append(item)
        return public

    def answer(self, question: str, recipes: list[dict[str, Any]], *, retrieval_context=None,
               ingredients=None, excluded_ingredients=None, limit: int = 3) -> dict[str, Any]:
        # The grounded builder validates complete steps and actual Source ownership.
        result = self.grounded.generate(question, recipes, mode="grounded", ingredients=ingredients,
                                        excluded_ingredients=excluded_ingredients, limit=limit)
        state = self.status()
        generation = {"mode": "api", "effective_mode": "grounded", "provider": state["provider"],
                      "model": state["model"], "pending_api": not state["api_configured"],
                      "api_status": "not_called" if state["api_configured"] else "not_configured",
                      "llm_called": False, "api_call_attempted": False, "fallback": False,
                      "errors": [], "response_json": None, "api_calls_attempted": 0}
        result["generation"] = generation
        result["rerank"] = {"status": "not_called", "api_call_attempted": False, "ordered_recipe_ids": []}
        result["qa"] = {"status": "not_called", "generation": {"api_call_attempted": False}}
        result["graphrag"] = {"retrieval": copy.deepcopy(retrieval_context or {}),
                              "answer_schema": state["structured_output_schema"], "source_bound": True,
                              "new_recipe_creation_supported": False}
        result["recipe_ids"] = [r["id"] for r in result["recipes"]]
        result["generated_recipes"] = []
        result["limitations"] = list(dict.fromkeys([*result.get("limitations", []), *LIMITATIONS]))
        if result["status"] != "ok":
            result["trace"].append({"stage": "api_rerank", "status": "no_match", "elapsed_ms": 0})
            return result
        if not state["api_configured"]:
            result["trace"].append({"stage": "api_question_answering", "status": "not_configured", "elapsed_ms": 0})
            return result

        valid, unavailable = _candidate_records(recipes, _terms(excluded_ingredients, "excluded_ingredients"))
        # Keep complete documents. Shrink the candidate prefix to the context
        # budget rather than silently slicing any original cooking step.
        context_tick = time.perf_counter()
        bounded, rerank_payload = [], None
        for recipe in valid[:MAX_RERANK_CANDIDATES]:
            trial = bounded + [recipe]
            try:
                payload = build_qa_context(question, trial, [r["id"] for r in trial],
                                           max_context_chars=self.qa.max_context_chars)
            except ExperimentalQAError:
                break
            bounded = trial
            rerank_payload = payload
        result["rerank_context"] = {"input_candidates": len(valid), "supplied_candidates": len(bounded),
                                    "complete_original_steps": True, "budget_omitted_candidates": len(valid) - len(bounded),
                                    "budget_unit": "characters", "max_context_chars": self.qa.max_context_chars,
                                    "supplied_context_chars": len(json.dumps(rerank_payload, ensure_ascii=False, allow_nan=False)) if rerank_payload else 0,
                                    "context_sha256": json_hash(rerank_payload) if rerank_payload else None}
        result["trace"].append({"stage": "bounded_api_context", "status": "ok" if bounded else "context_error",
                                "elapsed_ms": round((time.perf_counter() - context_tick) * 1000, 3),
                                **result["rerank_context"]})
        if not bounded:
            generation.update(fallback=True, api_status="context_error")
            generation["errors"] = [{"code": "context_error", "message": "原始步骤超过上下文预算，已返回原始证据。"}]
            return result
        candidate_ids = [r["id"] for r in bounded]
        reranked = self.qa.rerank(question, bounded, candidate_ids)
        result["rerank"] = self._public_api_result(reranked)
        result["trace"].append({"stage": "api_rerank", "status": reranked["status"],
                                "elapsed_ms": reranked.get("elapsed_ms", 0)})
        ordered = reranked["ordered_recipe_ids"] if reranked["status"] == "ok" else candidate_ids
        chosen = ordered[:limit]
        context_tick = time.perf_counter()
        try:
            qa_payload = build_qa_context(question, bounded, chosen, max_context_chars=self.qa.max_context_chars)
            qa_context = {"status": "ok", "candidate_count": len(chosen), "complete_original_steps": True,
                          "budget_unit": "characters", "max_context_chars": self.qa.max_context_chars,
                          "supplied_context_chars": len(json.dumps(qa_payload, ensure_ascii=False, allow_nan=False)),
                          "context_sha256": json_hash(qa_payload)}
        except ExperimentalQAError:
            # The service below retains its existing explicit failure/fallback path.
            qa_context = {"status": "context_error", "candidate_count": len(chosen)}
        result["qa_context"] = copy.deepcopy(qa_context)
        result["trace"].append({"stage": "api_qa_context", **qa_context,
                                "elapsed_ms": round((time.perf_counter() - context_tick) * 1000, 3)})
        qa_result = self.qa.answer(question, bounded, chosen)
        result["qa"] = self._public_api_result(qa_result)
        result["trace"].append({"stage": "api_question_answering", "status": qa_result["status"],
                                "elapsed_ms": qa_result.get("elapsed_ms", 0)})
        result["trace"].append({"stage": "api_answer_validation",
                                "status": "ok" if qa_result["validation"]["schema_citation_number_checks_passed"] else "not_passed",
                                "checks": "schema_citation_ID_and_numeric_literals",
                                "semantic_accuracy_verified": False})
        calls = int(bool(reranked.get("api_call_attempted"))) + int(bool(qa_result["generation"].get("api_call_attempted")))
        generation.update(llm_called=bool(calls), api_call_attempted=bool(calls), api_calls_attempted=calls)
        if reranked["status"] != "ok":
            generation["errors"].append({"code": "rerank_fallback", "message": "API重排未通过校验，保留检索排序并继续证据问答。"})
        if qa_result["status"] in {"answered", "insufficient_evidence"}:
            result["answer"] = qa_result["answer"]
            result["evidence"] = self._evidence(qa_result["evidence"], bounded)
            result["paragraphs"] = copy.deepcopy(qa_result["paragraphs"])
            cited = {e["recipe_id"] for e in result["evidence"]
                     if any(e["id"] in p["evidence_ids"] for p in result["paragraphs"])}
            selected = cited or set(chosen)
            by_id = {r["id"]: r for r in bounded}
            result["recipes"] = [by_id[rid] for rid in chosen if rid in selected]
            result["recipe_ids"] = [r["id"] for r in result["recipes"]]
            generation.update(effective_mode="graphrag_api_evidence_qa", api_status="ok" if qa_result["status"] == "answered" else "insufficient_evidence")
            generation["response_json"] = copy.deepcopy(qa_result["generation"].get("response_json"))
            result["status"] = "ok" if qa_result["status"] == "answered" else "insufficient_evidence"
            result["limitations"] = list(LIMITATIONS)
            # Grounded assembly checks refer to a different answer and must
            # not be presented as validation of model-written paragraphs.
            result["validation"] = {"passed": True, "semantic_accuracy_verified": False, "checks": [
                {"name": "qa_schema_citation_number_checks_passed", "passed": True},
                {"name": "qa_evidence_retains_original_steps_and_source_ownership", "passed": True},
            ]}
        else:
            # The rejected answer is not returned; preserve cited original text.
            by_id = {r["id"]: r for r in bounded}
            fallback = self.grounded.generate(question, [by_id[rid] for rid in chosen], mode="grounded",
                                             ingredients=ingredients, excluded_ingredients=excluded_ingredients, limit=limit)
            for key in ["answer", "recipes", "evidence", "validation"]:
                result[key] = fallback[key]
            result["recipe_ids"] = [r["id"] for r in result["recipes"]]
            generation.update(fallback=True, api_status=qa_result["status"])
            generation["errors"].append({"code": qa_result["generation"].get("error_code") or qa_result["status"],
                                         "message": "API问答未完成或引用校验失败，已返回原始步骤证据。"})
        return result
