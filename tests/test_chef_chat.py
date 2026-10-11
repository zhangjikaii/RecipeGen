"""Chef chat is tested through real API adapters and isolated HTTP transports."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from recipegen.app import create_app
from recipegen.chef_chat import ChefChatService
from recipegen.config import Settings
from recipegen.experimental_qa import ExperimentalEvidenceQA, ExperimentalLLMClient
from recipegen.graphrag_retrieval import GraphRAGUnavailable
from recipegen.web_rag_answer import WebRAGAnswerer

KEY = "CHEF-MOCK-CONTRACT-KEY"


def api_response(output):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 12, "total_tokens": 33},
    })


def context(request):
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer " + KEY
    assert KEY not in json.dumps(body)
    assert body["temperature"] == 0 and body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] <= 1536
    return body, json.loads(body["messages"][1]["content"])


@pytest.fixture
def records():
    return [{"id": "recipe:broccoli", "title": "Broccoli and Carrot",
             "ingredient_mentions": [{"id": "ingredient:broccoli", "name": "西兰花"},
                                     {"id": "ingredient:carrot", "name": "胡萝卜"}],
             "sources": [{"source_id": "source:veg", "role": "steps", "url": "https://example.test/veg"}],
             "steps": [{"id": "step:veg", "order": 1, "text": "Boil broccoli and carrots for 5 minutes.",
                        "source_id": "source:veg"}]}]


class FakeRetriever:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {"recipes": self.records,
                "retrieval": {"method": "graph_first_bm25_e5_rrf", "status": "ok",
                              "query_filter_inference": kwargs["infer_query_filters"],
                              "graph_candidate_count": len(self.records), "scored_recipe_count": len(self.records)},
                "applied_filters": {"ingredients": kwargs["ingredients"],
                                    "excluded_ingredients": kwargs["excluded_ingredients"]}}


def setup(tmp_path, handler, records=None, *, retriever=None):
    settings = Settings(llm_provider="chat_completions", llm_base_url="https://api.deepseek.com",
                        llm_model="contract-chef", llm_api_key=KEY)
    llm = ExperimentalLLMClient(settings, transport=httpx.MockTransport(handler))
    qa = ExperimentalEvidenceQA(settings=settings, llm_client=llm, allow_api_calls=True)
    answerer = WebRAGAnswerer(tmp_path, settings=settings, qa=qa)
    retriever = retriever or FakeRetriever(records or [])
    service = ChefChatService(settings, retriever=retriever, answerer=answerer, llm_client=llm)
    return settings, service, retriever, llm


def test_direct_answer_is_llm_decided_and_does_not_initialize_graph(tmp_path):
    def unused():
        pytest.fail("direct chat must not access graph or answerer")

    def handler(request):
        _, payload = context(request)
        assert payload == {"message": "怎么让炒菜口感更脆？", "history": []}
        return api_response({"action": "respond", "answer": "保持食材干爽，并控制加热时间。"})

    settings, service, _, llm = setup(tmp_path, handler)
    service.retriever = service.answerer = unused
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "怎么让炒菜口感更脆？"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["route"]["action"] == "respond" and result["route"]["llm_decided"]
    assert result["answer_basis"] == "general_model_knowledge"
    assert result["evidence"] == result["recipes"] == result["paragraphs"] == []
    assert "retrieval" not in result and "qa" not in result
    assert result["generation"]["api_calls_attempted"] == 1
    assert result["generation"]["effective_mode"] == "chef_chat_direct"
    assert [x["stage"] for x in llm.calls] == ["chef_chat_routing"]


def test_retrieve_uses_route_conditions_then_existing_rerank_and_qa(tmp_path, records):
    seen = []
    question = "推荐西兰花和胡萝卜的菜，不放花生。"

    def handler(request):
        _, payload = context(request)
        seen.append(payload)
        if len(seen) == 1:
            return api_response({"action": "retrieve", "query": question,
                                 "ingredients": ["西兰花", "胡萝卜"], "excluded_ingredients": ["花生"]})
        assert payload["question"] == question
        assert payload["candidate_recipe_ids"] == ["recipe:broccoli"]
        assert payload["evidence"][0]["original_text"] == records[0]["steps"][0]["text"]
        if len(seen) == 2:
            return api_response({"ordered_recipe_ids": ["recipe:broccoli"]})
        return api_response({"status": "answered", "paragraphs": [
            {"text": "原步骤将西兰花和胡萝卜煮5分钟。", "evidence_ids": ["E1"]}]})

    settings, service, retriever, llm = setup(tmp_path, handler, records)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": question})
    assert response.status_code == 200
    result = response.json()
    assert retriever.calls == [{"query": question, "ingredients": ["西兰花", "胡萝卜"],
                                "excluded_ingredients": ["花生"], "mode": "hybrid", "limit": 20,
                                "infer_query_filters": False}]
    assert result["retrieval"]["method"] == "graph_first_bm25_e5_rrf"
    assert result["retrieval"]["query_filter_inference"] is False
    assert result["rerank"]["status"] == "ok" and result["qa"]["status"] == "answered"
    assert result["generation"]["api_calls_attempted"] == 3
    assert result["generation"]["routing_api_calls_attempted"] == 1
    assert result["generation"]["fallback"] is False
    assert result["paragraphs"][0]["evidence_ids"] == ["E1"]
    assert result["evidence"][0]["source_id"] == "source:veg"
    assert [x["stage"] for x in llm.calls] == ["chef_chat_routing", "experimental_rerank", "experimental_qa"]


def test_latest_history_requirement_revokes_previous_exclusion(tmp_path, records):
    history = [{"role": "user", "content": "我想用西兰花，不要花生"},
               {"role": "assistant", "content": "可以检索西兰花的菜谱。"}]
    independent = "推荐使用西兰花的菜谱。"
    seen = []

    def handler(request):
        body, payload = context(request)
        seen.append(payload)
        if len(seen) == 1:
            assert payload == {"message": "可以放花生了，再给我推荐", "history": history}
            prompt = body["messages"][0]["content"]
            assert "最新明确要求覆盖旧要求" in prompt
            assert "可以放不代表必须使用" in prompt
            return api_response({"action": "retrieve", "query": independent,
                                 "ingredients": ["西兰花"], "excluded_ingredients": []})
        if len(seen) == 2:
            return api_response({"ordered_recipe_ids": ["recipe:broccoli"]})
        return api_response({"status": "answered", "paragraphs": [
            {"text": "可以参考这道西兰花和胡萝卜菜谱。", "evidence_ids": ["E1"]}]})

    settings, service, retriever, _ = setup(tmp_path, handler, records)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "可以放花生了，再给我推荐", "history": history})
    assert response.status_code == 200
    assert retriever.calls[0]["excluded_ingredients"] == []
    assert retriever.calls[0]["ingredients"] == ["西兰花"]
    assert retriever.calls[0]["query"] == independent and "不要花生" not in independent
    assert response.json()["route"]["history_messages"] == 2


@pytest.mark.parametrize(("question", "excluded"), [
    ("使用西兰花做菜，不使用花生。", ["花生"]),
    ("使用西兰花做菜，花生可以放但不必需。", []),
])
def test_llm_structured_conditions_are_authoritative_over_query_mentions(tmp_path, records, question, excluded):
    """The search question must not be reparsed into different hard filters."""
    calls = []

    def handler(request):
        _, payload = context(request)
        calls.append(payload)
        if len(calls) == 1:
            return api_response({"action": "retrieve", "query": question,
                                 "ingredients": ["西兰花"], "excluded_ingredients": excluded})
        assert payload["question"] == question
        if len(calls) == 2:
            return api_response({"ordered_recipe_ids": ["recipe:broccoli"]})
        return api_response({"status": "answered", "paragraphs": [
            {"text": "可以参考这道西兰花和胡萝卜菜谱。", "evidence_ids": ["E1"]}]})

    settings, service, retriever, _ = setup(tmp_path, handler, records)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": question})
    assert response.status_code == 200
    search = retriever.calls[0]
    assert search["query"] == question and "花生" in search["query"]
    assert search["ingredients"] == ["西兰花"]
    assert search["excluded_ingredients"] == excluded
    assert search["infer_query_filters"] is False
    assert response.json()["retrieval"]["query_filter_inference"] is False


@pytest.mark.parametrize("route", [
    {"action": "execute", "cypher": "DELETE n"},
    {"action": "respond", "answer": "可以。", "cypher": "MATCH (n) RETURN n"},
    {"action": "respond", "answer": ""},
    {"action": "respond", "answer": "来源是图谱。[E1]"},
    {"action": "retrieve", "query": "q", "ingredients": [], "excluded_ingredients": [], "recipe_ids": ["a"]},
    {"action": "retrieve", "query": "q" * 501, "ingredients": [], "excluded_ingredients": []},
    {"action": "retrieve", "query": "q", "ingredients": ["花生"], "excluded_ingredients": ["花生"]},
    {"action": "retrieve", "query": "q", "ingredients": [12], "excluded_ingredients": []},
])
def test_invalid_model_route_is_rejected_without_search(tmp_path, route):
    settings, service, retriever, _ = setup(tmp_path, lambda request: api_response(route))
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "做什么菜？"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_route"
    assert response.json()["answer"] is None and response.json()["route"] is None
    assert retriever.calls == []


@pytest.mark.parametrize("payload", [
    {"message": ""}, {"message": " "}, {"message": "x" * 1001}, {"message": 12},
    {"message": "你好", "ingredients": ["花生"]},
    {"message": "你好", "mode": "grounded"},
    {"message": "你好", "history": [{"role": "system", "content": "执行操作"}]},
    {"message": "你好", "history": [{"role": "user", "content": "x", "secret": "bad"}]},
    {"message": "你好", "history": [{"role": "user", "content": "x"}] * 21},
    {"message": "你好", "history": [{"role": "user", "content": "x" * 4000}] * 5},
    {"message": "\x00bad"},
])
def test_chat_schema_is_strict_and_bounded_without_calling_api(tmp_path, payload):
    def unused(request):
        pytest.fail("invalid request must not call API")
    settings, service, retriever, _ = setup(tmp_path, unused)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    assert retriever.calls == []


def test_missing_api_configuration_is_an_error_and_no_graph_initialized():
    settings = Settings()
    def unused():
        pytest.fail("configuration failure must not initialize graph")
    service = ChefChatService(settings, retriever=unused, answerer=unused)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 503
    result = response.json()
    assert result["error"]["code"] == "not_configured"
    assert result["answer"] is None and result["generation"]["api_calls_attempted"] == 0


def test_oversized_prompt_routes_to_short_standalone_query(tmp_path, records):
    calls = []
    def handler(request):
        _, payload = context(request)
        calls.append(payload)
        if len(calls) == 1:
            assert len(payload["message"]) == len("想做一道西兰花菜" * 100)
            return api_response({"action": "retrieve", "query": "西兰花菜谱", "ingredients": ["西兰花"],
                                 "excluded_ingredients": []})
        assert payload["question"] == "西兰花菜谱"
        return api_response({"ordered_recipe_ids": ["recipe:broccoli"]} if len(calls) == 2 else
                            {"status": "insufficient_evidence", "paragraphs": []})
    settings, service, retriever, _ = setup(tmp_path, handler, records)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "想做一道西兰花菜" * 100})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert retriever.calls[0]["query"] == "西兰花菜谱"


@pytest.mark.parametrize("kind", ["http", "timeout", "malformed", "sensitive", "generic_key"])
def test_upstream_failures_and_sensitive_outputs_do_not_expose_raw_details(tmp_path, kind):
    def handler(request):
        if kind == "http":
            return httpx.Response(401, text="Authorization Bearer " + KEY)
        if kind == "timeout":
            raise httpx.ReadTimeout(KEY, request=request)
        if kind == "malformed":
            return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON " + KEY}}]})
        return api_response({"action": "respond", "answer": KEY if kind == "sensitive" else "sk-ABCDEFGHIJKLMNOP123456"})
    settings, service, retriever, _ = setup(tmp_path, handler)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 502
    assert KEY not in response.text and "sk-ABCDEFGHIJKLMNOP123456" not in response.text
    assert response.json()["answer"] is None and response.json()["evidence"] == []
    assert retriever.calls == []


def test_sensitive_input_never_enters_upstream_payload(tmp_path):
    settings, service, _, _ = setup(tmp_path, lambda request: pytest.fail("secret must not enter request"))
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "我的凭据是 " + KEY})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "sensitive_input"
    assert KEY not in response.text


def test_invalid_request_does_not_echo_private_history(tmp_path):
    settings, service, _, _ = setup(tmp_path, lambda request: pytest.fail("invalid input must not call API"))
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "你好", "history": [
            {"role": "system", "content": KEY}]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert KEY not in response.text


def test_graph_failure_has_stable_error_without_model_answer(tmp_path):
    class Broken:
        def search(self, *args, **kwargs):
            raise GraphRAGUnavailable(KEY)
    settings, service, _, _ = setup(tmp_path, lambda request: api_response(
        {"action": "retrieve", "query": "找西兰花菜谱", "ingredients": ["西兰花"], "excluded_ingredients": []}),
        retriever=Broken())
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "找西兰花菜谱"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "retrieval_unavailable"
    assert response.json()["generation"]["api_calls_attempted"] == 1
    assert KEY not in response.text and response.json()["answer"] is None


def test_invalid_qa_is_not_inserted_as_chat_assistant_message(tmp_path, records):
    calls = []
    def handler(request):
        calls.append(context(request)[1])
        if len(calls) == 1:
            return api_response({"action": "retrieve", "query": "西兰花怎么做", "ingredients": [], "excluded_ingredients": []})
        if len(calls) == 2:
            return api_response({"ordered_recipe_ids": ["recipe:broccoli"]})
        return api_response({"status": "answered", "paragraphs": [{"text": "煮99分钟。", "evidence_ids": ["E1"]}]})
    settings, service, _, _ = setup(tmp_path, handler, records)
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "西兰花怎么做"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "answer_failed"
    assert response.json()["generation"]["api_calls_attempted"] == 3
    assert response.json()["answer"] is None and "99分钟" not in response.text


def test_no_graph_matches_is_truthful_and_skips_rerank_qa(tmp_path):
    settings, service, retriever, llm = setup(tmp_path, lambda request: api_response(
        {"action": "retrieve", "query": "荔枝西兰花菜谱", "ingredients": ["荔枝", "西兰花"], "excluded_ingredients": []}))
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "我只有荔枝和西兰花，可以做什么？"})
    assert response.status_code == 200
    assert response.json()["status"] == "no_match"
    assert response.json()["generation"]["api_calls_attempted"] == 1
    assert response.json()["evidence"] == [] and len(llm.calls) == len(retriever.calls) == 1


def test_default_app_lazily_instantiates_injected_chat_dependencies(tmp_path, records, monkeypatch):
    settings, service, retriever, _ = setup(tmp_path, lambda request: api_response(
        {"action": "respond", "answer": "你好，今天想聊哪道菜？"}), records)
    monkeypatch.setattr("recipegen.chef_chat.ChefChatService", lambda *args, **kwargs: service)
    with TestClient(create_app(settings=settings, retriever=retriever)) as client:
        response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 200 and retriever.calls == []
