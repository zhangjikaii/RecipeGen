"""Web GraphRAG contracts use the real API adapter with isolated HTTP only."""
from __future__ import annotations

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from recipegen.app import create_app
from recipegen.config import Settings
from recipegen.experimental_qa import ExperimentalEvidenceQA, ExperimentalLLMClient, build_qa_context
from recipegen.web_rag_answer import WebRAGAnswerer


KEY = "PRIVATE-WEB-CONTRACT-KEY"
IDS = ["recipe:soup", "recipe:tea"]


@pytest.fixture
def records():
    return [
        {"id": IDS[0], "title": "Tomato Soup", "ingredient_mentions": [],
         "sources": [{"source_id": "source:soup", "role": "steps", "url": "https://example.test/soup"}],
         "steps": [
             {"id": "step:heat", "order": 1, "text": "Heat water for 5 minutes.", "source_id": "source:soup"},
             {"id": "step:tomato", "order": 2, "text": "Add tomatoes and stir.", "source_id": "source:soup"},
         ]},
        {"id": IDS[1], "title": "Tea", "ingredient_mentions": [],
         "sources": [{"source_id": "source:tea", "role": "steps", "url": "https://example.test/tea"}],
         "steps": [{"id": "step:tea", "order": 1, "text": "Steep tea for 10 minutes.", "source_id": "source:tea"}]},
    ]


def api_response(output):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 13, "total_tokens": 30},
    })


def request_context(request):
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer " + KEY
    assert KEY not in json.dumps(body)
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0 and body["stream"] is False
    return json.loads(body["messages"][1]["content"])


def answer(text="将水加热5分钟。", evidence_ids=None):
    return {"status": "answered", "paragraphs": [{"text": text, "evidence_ids": evidence_ids or ["E1"]}]}


def service(tmp_path, handler):
    settings = Settings(llm_provider="chat_completions", llm_base_url="https://api.deepseek.com",
                        llm_model="contract-only", llm_api_key=KEY)
    client = ExperimentalLLMClient(settings, transport=httpx.MockTransport(handler))
    qa = ExperimentalEvidenceQA(settings=settings, llm_client=client, allow_api_calls=True)
    return WebRAGAnswerer(project_root=tmp_path, settings=settings, qa=qa), client


def test_rerank_then_question_answer_uses_reordered_complete_sources(tmp_path, records):
    contexts = []

    def handler(request):
        context = request_context(request)
        contexts.append(context)
        if len(contexts) == 1:
            assert context["candidate_recipe_ids"] == IDS
            assert len(context["evidence"]) == 3
            return api_response({"ordered_recipe_ids": IDS[::-1]})
        assert context["candidate_recipe_ids"] == [IDS[1]]
        assert context["original_steps_complete"] is True
        assert context["evidence"][0]["original_text"] == records[1]["steps"][0]["text"]
        return api_response(answer("根据原步骤，将茶浸泡10分钟。"))

    original = copy.deepcopy(records)
    web, client = service(tmp_path, handler)
    result = web.answer("茶应该浸泡多久？", records, limit=1,
                        retrieval_context={"method": "graph_filter_then_e5_cosine", "candidate_count": 2})

    assert len(contexts) == 2
    assert [call["stage"] for call in client.calls] == ["experimental_rerank", "experimental_qa"]
    assert records == original
    assert result["rerank"]["status"] == "ok"
    assert result["rerank"]["ordered_recipe_ids"] == IDS[::-1]
    assert result["qa"]["status"] == "answered"
    assert result["answer"] == "根据原步骤，将茶浸泡10分钟。 [E1]"
    assert [recipe["id"] for recipe in result["recipes"]] == [IDS[1]]
    assert result["generation"]["mode"] == "api"
    assert result["generation"]["effective_mode"] == "graphrag_api_evidence_qa"
    assert result["generation"]["api_status"] == "ok"
    assert result["generation"]["llm_called"] is True
    assert result["generation"]["fallback"] is False
    assert result["validation"]["passed"] is True
    evidence = result["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["id"] == "E1" and evidence[0]["kind"] == "step"
    assert evidence[0]["graph_id"] == "step:tea"
    assert evidence[0]["text"] == "Steep tea for 10 minutes."
    assert evidence[0]["source_id"] == "source:tea"
    assert evidence[0]["source_url"] == "https://example.test/tea"
    assert result["qa"]["validation"]["semantic_evaluated"] is False
    stages = [item["stage"] for item in result["trace"]]
    serial = ["graph_evidence", "bounded_api_context", "api_rerank", "api_qa_context",
              "api_question_answering", "api_answer_validation"]
    assert [stages.index(stage) for stage in serial] == sorted(stages.index(stage) for stage in serial)
    assert result["rerank_context"]["context_sha256"] == result["rerank"]["context_sha256"]
    assert result["qa_context"]["context_sha256"] == result["qa"]["context_sha256"]
    assert result["rerank_context"]["budget_unit"] == result["qa_context"]["budget_unit"] == "characters"
    assert result["qa_context"]["supplied_context_chars"] <= result["qa_context"]["max_context_chars"]
    assert KEY not in json.dumps(result)


@pytest.mark.parametrize("bad_ids", [
    [IDS[1], "recipe:outside-graph"], [IDS[0], IDS[0]], [IDS[1]],
])
def test_invalid_rerank_permutation_retains_retrieval_order_for_qa(tmp_path, records, bad_ids):
    contexts = []

    def handler(request):
        context = request_context(request)
        contexts.append(context)
        if len(contexts) == 1:
            return api_response({"ordered_recipe_ids": bad_ids})
        assert context["candidate_recipe_ids"] == [IDS[0]]
        return api_response(answer())

    web, _ = service(tmp_path, handler)
    result = web.answer("水加热多久？", records, limit=1)
    assert len(contexts) == 2
    assert result["rerank"]["status"] == "fallback"
    assert result["rerank"]["ordered_recipe_ids"] == IDS
    assert result["rerank"]["error_code"] == "rerank_not_candidate_permutation"
    assert result["qa"]["status"] == "answered"
    assert [row["id"] for row in result["recipes"]] == [IDS[0]]
    assert "outside-graph" not in result["answer"]
    assert "保留检索排序" in result["generation"]["errors"][0]["message"]


@pytest.mark.parametrize("invalid,error", [
    (answer(evidence_ids=["E99"]), "unknown_or_duplicate_citation"),
    (answer("将水加热12分钟。"), "unsupported_numeric_literal"),
])
def test_qa_citation_or_number_rejection_returns_original_steps(tmp_path, records, invalid, error):
    calls = []

    def handler(request):
        calls.append(request_context(request))
        return api_response({"ordered_recipe_ids": IDS} if len(calls) == 1 else invalid)

    web, _ = service(tmp_path, handler)
    result = web.answer("水加热多久？", records, limit=1)
    assert len(calls) == 2
    assert result["qa"]["status"] == "validation_failed"
    assert result["qa"]["generation"]["error_code"] == error
    assert result["generation"]["fallback"] is True
    assert result["generation"]["effective_mode"] == "grounded"
    assert result["generation"]["api_status"] != "ok"
    assert all(step["text"] in result["answer"] for step in records[0]["steps"])
    assert "12分钟" not in result["answer"] and "E99" not in result["answer"]
    assert next(item for item in result["trace"] if item["stage"] == "api_answer_validation")["status"] == "not_passed"


def test_no_candidates_never_calls_api(tmp_path):
    web, client = service(tmp_path, lambda _: pytest.fail("No evidence must never trigger HTTP"))
    result = web.answer("番茄怎么做？", [], limit=1)
    assert result["status"] == "no_match"
    assert result["generation"]["llm_called"] is False
    assert result["generation"]["api_status"] != "ok"
    assert result["recipes"] == result["evidence"] == []
    assert client.calls == []


def test_insufficient_evidence_is_displayed_without_invented_answer(tmp_path, records):
    calls = []

    def handler(request):
        calls.append(request_context(request))
        return api_response({"ordered_recipe_ids": IDS} if len(calls) == 1 else
                            {"status": "insufficient_evidence", "paragraphs": []})

    web, _ = service(tmp_path, handler)
    result = web.answer("这道汤的总热量是多少？", records, limit=1)
    assert len(calls) == 2
    assert result["qa"]["status"] == "insufficient_evidence"
    assert result["qa"]["paragraphs"] == []
    assert "不足" in result["answer"]
    assert "Heat water" not in result["answer"]
    assert result["generation"]["api_status"] != "ok"


def test_rerank_http_failure_keeps_order_and_can_still_answer(tmp_path, records):
    contexts = []

    def handler(request):
        contexts.append(request_context(request))
        if len(contexts) == 1:
            return httpx.Response(503, text="private provider details")
        assert contexts[-1]["candidate_recipe_ids"] == [IDS[0]]
        return api_response(answer())

    web, _ = service(tmp_path, handler)
    result = web.answer("水加热多久？", records, limit=1)
    assert len(contexts) == 2
    assert result["rerank"]["status"] == "fallback"
    assert result["rerank"]["error_code"] == "upstream_error"
    assert result["qa"]["status"] == "answered"
    assert result["answer"] == "将水加热5分钟。 [E1]"
    assert "private provider details" not in json.dumps(result)


def test_qa_http_failure_returns_original_steps(tmp_path, records):
    calls = []

    def handler(request):
        calls.append(request_context(request))
        if len(calls) == 1:
            return api_response({"ordered_recipe_ids": IDS})
        raise httpx.ReadTimeout("private provider details", request=request)

    web, _ = service(tmp_path, handler)
    result = web.answer("水加热多久？", records, limit=1)
    assert len(calls) == 2
    assert result["qa"]["status"] == "upstream_error"
    assert result["generation"]["fallback"] is True
    assert result["generation"]["effective_mode"] == "grounded"
    assert result["generation"]["api_status"] != "ok"
    assert all(step["text"] in result["answer"] for step in records[0]["steps"])
    assert "private provider details" not in json.dumps(result)


def test_sensitive_model_output_is_not_returned_even_as_diagnostic(tmp_path, records):
    calls = []

    def handler(request):
        calls.append(request_context(request))
        return api_response({"ordered_recipe_ids": IDS} if len(calls) == 1 else
                            answer("模型输出的凭据为" + KEY))

    web, _ = service(tmp_path, handler)
    result = web.answer("水加热多久？", records, limit=1)
    assert len(calls) == 2
    assert result["generation"]["fallback"] is True
    assert result["qa"]["generation"]["error_code"] == "sensitive_output"
    assert result["qa"]["generation"]["response_json"] is None
    assert KEY not in json.dumps(result)
    assert all(step["text"] in result["answer"] for step in records[0]["steps"])


@pytest.mark.parametrize("problem", ["missing_steps", "foreign_step_source"])
def test_unusable_step_records_do_not_enter_any_api_context(tmp_path, records, problem):
    broken = copy.deepcopy(records[0])
    if problem == "missing_steps":
        broken["steps"] = []
    else:
        broken["steps"][0]["source_id"] = "source:outside-recipe"
    contexts = []

    def handler(request):
        context = request_context(request)
        contexts.append(context)
        assert context["candidate_recipe_ids"] == [IDS[1]]
        return api_response({"ordered_recipe_ids": [IDS[1]]} if len(contexts) == 1 else
                            answer("将茶浸泡10分钟。"))

    web, _ = service(tmp_path, handler)
    result = web.answer("茶浸泡多久？", [broken, records[1]], limit=1)
    assert len(contexts) == 2
    assert result["qa"]["status"] == "answered"
    assert [record["id"] for record in result["recipes"]] == [IDS[1]]
    assert all(e["source_id"] == "source:tea" for e in result["evidence"])


def test_generate_http_endpoint_runs_retrieval_rerank_and_qa(tmp_path, records):
    """Exercise the public endpoint, not just the answerer helper contract."""
    retrieval_calls, api_contexts = [], []

    class Catalog:
        def close(self):
            pass

    class Retriever:
        def search(self, question, **kwargs):
            retrieval_calls.append({"question": question, **kwargs})
            return {"recipes": copy.deepcopy(records),
                    "retrieval": {"mode": "semantic", "effective_mode": "semantic",
                                  "status": "ok", "method": "graph_filter_then_e5_cosine",
                                  "candidate_count": len(records)},
                    "applied_filters": {"excluded_ingredients": []}}

    def handler(request):
        context = request_context(request)
        api_contexts.append(context)
        if len(api_contexts) == 1:
            assert context["candidate_recipe_ids"] == IDS
            return api_response({"ordered_recipe_ids": IDS[::-1]})
        assert context["candidate_recipe_ids"] == [IDS[1]]
        return api_response(answer("将茶浸泡10分钟。"))

    web, _ = service(tmp_path, handler)
    with TestClient(create_app(web.settings, catalog=Catalog(), retriever=Retriever(),
                              api_answerer=web)) as client:
        response = client.post("/api/generate", json={"question": "茶浸泡多久？", "mode": "api",
                                                       "retrieval_mode": "semantic", "limit": 1})
    assert response.status_code == 200
    assert len(retrieval_calls) == 1 and retrieval_calls[0]["limit"] == 20
    assert retrieval_calls[0]["question"] == "茶浸泡多久？"
    assert len(api_contexts) == 2
    result = response.json()
    assert result["retrieval"]["method"] == "graph_filter_then_e5_cosine"
    assert result["rerank"]["status"] == "ok" and result["qa"]["status"] == "answered"
    assert result["answer"] == "将茶浸泡10分钟。 [E1]"
    assert result["evidence"][0]["text"] == records[1]["steps"][0]["text"]
    assert result["generation"]["api_status"] == "ok"


def test_explicit_selected_recipe_does_not_claim_new_vector_retrieval(tmp_path, records):
    api_contexts = []

    class Catalog:
        def recipe(self, identifier):
            return copy.deepcopy(next(row for row in records if row["id"] == identifier))

        def close(self):
            pass

    class Retriever:
        def search(self, *_args, **_kwargs):
            pytest.fail("Explicit selected evidence must not be replaced by unrelated search candidates")

    def handler(request):
        context = request_context(request)
        api_contexts.append(context)
        assert context["candidate_recipe_ids"] == [IDS[1]]
        return api_response({"ordered_recipe_ids": [IDS[1]]} if len(api_contexts) == 1 else
                            answer("将茶浸泡10分钟。"))

    web, _ = service(tmp_path, handler)
    with TestClient(create_app(web.settings, catalog=Catalog(), retriever=Retriever(),
                              api_answerer=web)) as client:
        response = client.post("/api/generate", json={"question": "茶浸泡多久？", "mode": "api",
                                                       "recipe_ids": [IDS[1]], "limit": 1})
    assert response.status_code == 200
    result = response.json()
    assert len(api_contexts) == 2
    assert result["retrieval"]["selection"] == "explicit_recipe_ids"
    assert result["retrieval"]["method"] == "verified_recipe_ids"
    assert result["generation"]["api_status"] == "ok"


def test_context_budget_removes_whole_candidates_and_preserves_complete_steps(tmp_path, records):
    question, contexts = "水加热多久？", []

    def handler(request):
        context = request_context(request)
        contexts.append(context)
        assert context["candidate_recipe_ids"] == IDS[:1]
        assert [e["original_text"] for e in context["evidence"]] == [s["text"] for s in records[0]["steps"]]
        return api_response({"ordered_recipe_ids": IDS[:1]} if len(contexts) == 1 else answer())

    web, _ = service(tmp_path, handler)
    first = build_qa_context(question, records[:1], IDS[:1])
    web.qa.max_context_chars = len(json.dumps(first, ensure_ascii=False, allow_nan=False)) + 1
    result = web.answer(question, records, limit=1,
                        retrieval_context={"method": "graph_first_bm25_e5_rrf"})

    assert len(contexts) == 2
    assert result["rerank_context"]["supplied_candidates"] == 1
    assert result["rerank_context"]["budget_omitted_candidates"] == 1
    assert result["rerank_context"]["complete_original_steps"] is True
    assert result["rerank_context"]["supplied_context_chars"] <= web.qa.max_context_chars
    assert result["qa"]["status"] == "answered" and result["generation"]["fallback"] is False


def test_no_complete_candidate_fits_context_budget_never_calls_api(tmp_path, records):
    web, client = service(tmp_path, lambda _: pytest.fail("An oversized source cannot enter an API context"))
    web.qa.max_context_chars = 1
    result = web.answer("水加热多久？", records, limit=1)

    assert client.calls == []
    assert result["generation"]["api_status"] == "context_error"
    assert result["generation"]["fallback"] is True
    assert result["rerank_context"]["supplied_candidates"] == 0
    assert next(item for item in result["trace"] if item["stage"] == "bounded_api_context")["status"] == "context_error"
    assert all(step["text"] in result["answer"] for step in records[0]["steps"])
