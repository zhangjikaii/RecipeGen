"""Public chat orchestration with real RRF/QA contracts and isolated I/O.

Scores and graph records are deliberate fixtures. This proves ordering and
evidence preservation, not live Neo4j retrieval quality or model accuracy.
"""
from __future__ import annotations

import copy
import json

import httpx
from fastapi.testclient import TestClient

from recipegen.app import create_app
from recipegen.chef_chat import ChefChatService
from recipegen.config import Settings
from recipegen.experimental_qa import ExperimentalEvidenceQA, ExperimentalLLMClient
from recipegen.experimental_retrieval import pool_rrf
from recipegen.web_rag_answer import WebRAGAnswerer


def test_chat_routes_hybrid_fuses_reads_sources_then_reranks_and_answers(tmp_path):
    key, events, contexts = "PRIVATE-SERIAL-CONTRACT-KEY", [], []
    first, second, excluded = "recipe:first", "recipe:second", "recipe:milk"

    def record(identifier, duration):
        source = "source:" + identifier
        return {"id": identifier, "title": "Broccoli and Carrot Fixture",
                "ingredient_mentions": [{"id": "ingredient:broccoli", "name": "broccoli"},
                                        {"id": "ingredient:carrot", "name": "carrot"}],
                "sources": [{"source_id": source, "role": "steps"}],
                "steps": [{"id": "step:" + identifier, "order": 1,
                           "text": f"Cook broccoli and carrots for {duration} minutes.", "source_id": source}]}

    records = {first: record(first, 5), second: record(second, 3)}

    class Retriever:
        def search(self, question, **kwargs):
            assert kwargs["mode"] == "hybrid" and kwargs["infer_query_filters"] is False
            assert kwargs["ingredients"] == ["西兰花", "胡萝卜"]
            assert kwargs["excluded_ingredients"] == ["牛奶"]
            assert events == ["LLM_route"]
            # A high lexical score from a condition-ineligible recipe must not
            # survive the common hard-condition intersection or enter the API.
            fused = pool_rrf({first: 10., second: 8., excluded: 100.},
                             {first: .7, second: .9}, graph_ids=records,
                             eligible_ids=records, rrf_k=60)
            ids = [row["recipe_id"] for row in fused["rankings"]]
            assert ids == [first, second] and excluded not in fused["candidate_pool"]
            events.extend(["RRF", "graph_readback"])
            return {"recipes": [copy.deepcopy(records[rid]) for rid in ids],
                    "applied_filters": {"excluded_ingredients": ["milk"]},
                    "retrieval": {"status": "ok", "mode": "hybrid", "effective_mode": "hybrid",
                                  "method": "graph_first_bm25_e5_rrf", "rrf_k": 60,
                                  "graph_has_rank": False, "rankings": fused["rankings"]},
                    "trace": [{"stage": "rrf_fusion", "status": "ok"},
                              {"stage": "graph_evidence_expansion", "status": "ok", "one_hop_only": True}]}

    def response(output):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(output, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 13, "total_tokens": 30}})

    def handler(request):
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer " + key
        assert key not in json.dumps(body)
        payload = json.loads(body["messages"][1]["content"])
        contexts.append(payload)
        if len(contexts) == 1:
            events.append("LLM_route")
            return response({"action": "retrieve", "query": "推荐一道西兰花和胡萝卜菜，不要牛奶。",
                             "ingredients": ["西兰花", "胡萝卜"], "excluded_ingredients": ["牛奶"]})
        if len(contexts) == 2:
            assert events[-1] == "graph_readback"
            assert payload["candidate_recipe_ids"] == [first, second]
            assert [e["original_text"] for e in payload["evidence"]] == [records[rid]["steps"][0]["text"] for rid in [first, second]]
            events.append("LLM_rerank")
            return response({"ordered_recipe_ids": [second, first]})
        assert len(contexts) == 3 and events[-1] == "LLM_rerank"
        assert payload["candidate_recipe_ids"] == [second, first]
        assert payload["evidence"][0]["original_text"] == records[second]["steps"][0]["text"]
        events.append("LLM_QA")
        return response({"status": "answered", "paragraphs": [
            {"text": "根据原步骤，将西兰花和胡萝卜煮3分钟。", "evidence_ids": ["E1"]}]})

    settings = Settings(llm_provider="chat_completions", llm_base_url="https://api.deepseek.com",
                        llm_model="contract-only", llm_api_key=key)
    llm = ExperimentalLLMClient(settings, transport=httpx.MockTransport(handler))
    qa = ExperimentalEvidenceQA(settings=settings, llm_client=llm, allow_api_calls=True)
    answerer = WebRAGAnswerer(tmp_path, settings=settings, qa=qa)
    chat = ChefChatService(settings, retriever=Retriever(), answerer=answerer, llm_client=llm)
    with TestClient(create_app(settings, chat_service=chat)) as client:
        actual = client.post("/api/chat", json={"message": "请从图谱推荐西兰花和胡萝卜菜，不要牛奶。", "history": []})
    assert actual.status_code == 200
    result = actual.json()
    assert events == ["LLM_route", "RRF", "graph_readback", "LLM_rerank", "LLM_QA"]
    expected = ["llm_routing", "rrf_fusion", "graph_evidence_expansion", "bounded_api_context",
                "api_rerank", "api_qa_context", "api_question_answering", "api_answer_validation"]
    stages = [s["stage"] for s in result["trace"]]
    assert [stages.index(s) for s in expected] == sorted(stages.index(s) for s in expected)
    assert result["generation"]["api_calls_attempted"] == 3 and result["generation"]["fallback"] is False
    assert [call["stage"] for call in llm.calls] == ["chef_chat_routing", "experimental_rerank", "experimental_qa"]
    assert result["answer"] == "根据原步骤，将西兰花和胡萝卜煮3分钟。 [E1]"
    assert result["evidence"][0]["recipe_id"] == second
    assert result["evidence"][0]["original_text"] == records[second]["steps"][0]["text"]
    assert result["qa_context"]["context_sha256"] == result["qa"]["context_sha256"]
    assert result["validation"]["semantic_accuracy_verified"] is False
    assert key not in json.dumps(result) and excluded not in json.dumps(result)
