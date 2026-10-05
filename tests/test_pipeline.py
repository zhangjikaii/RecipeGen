import json

import httpx
import pytest

from recipegen.config import PROJECT_ROOT, Settings
from recipegen.evaluate import evaluate
from recipegen.llm import LLMClient
from recipegen.models import RecommendRequest
from recipegen.pipeline import Recommender


def mock_llm(handler):
    settings = Settings(llm_provider="chat_completions", llm_base_url="https://example.invalid/v1", llm_model="test-only", llm_api_key="secret-that-must-not-leak")
    return settings, LLMClient(settings, transport=httpx.MockTransport(handler))


def response(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]})


def test_all_handwritten_cases(graph_store, settings):
    report = evaluate(Recommender(settings, graph_store))
    assert report["passed"] == report["total"] == 50
    assert report["successful_llm_calls"] == 0
    assert report["dataset"]["is_demo"] is True


def test_graph_provenance_and_native_edges(graph_store, settings):
    result = Recommender(settings, graph_store).recommend(RecommendRequest(question="我有番茄和鸡蛋，20分钟内，不吃辣"))
    assert result["status"] == "ok"
    assert result["generation"]["mode"] == "template"
    assert result["validation"]["passed"]
    for item in result["recommendations"]:
        evidence = {e["id"]: e for e in result["evidence"]}
        assert all(evidence[identifier]["recipe_id"] == item["recipe_id"] for identifier in item["evidence_ids"])
        graph = graph_store.graph(item["recipe_id"])
        assert any(edge["label"] == "USES_INGREDIENT" for edge in graph["edges"])


def test_llm_success_is_real_path_but_mocked_test(graph_store):
    def handler(request):
        payload = json.loads(request.content)
        question = json.loads(payload["messages"][1]["content"])
        assert "secret-that-must-not-leak" not in payload["messages"][0]["content"]
        if "candidates" not in question:
            return response({"available_ingredients": ["番茄", "鸡蛋"], "excluded_ingredients": [], "max_minutes": 20, "required_tags": ["不辣"], "allow_missing": False, "top_k": 3, "recipe_name": None})
        return response({"recommendations": [{"recipe_id": "r001", "reason_codes": ["ingredient_match", "within_time", "tag_match"]}]})
    config, client = mock_llm(handler)
    result = Recommender(config, graph_store, client).recommend(RecommendRequest(question="我有番茄和鸡蛋，20分钟内，不吃辣", use_llm=True))
    assert result["generation"]["mode"] == "llm"
    assert len(client.calls) == 2
    assert result["recommendations"][0]["recipe_id"] == "r001"
    assert result["validation"]["passed"]


@pytest.mark.parametrize("bad_choice", [
    {"recipe_id": "invented", "reason_codes": ["ingredient_match"]},
    {"recipe_id": "r001", "reason_codes": ["needs_shopping"]},
    {"recipe_id": "r001", "reason_codes": ["nutrition_is_safe"]},
])
def test_model_cannot_invent_recipe_or_reason(graph_store, bad_choice):
    def handler(request):
        data = json.loads(json.loads(request.content)["messages"][1]["content"])
        if "candidates" not in data:
            return response({"available_ingredients": ["番茄", "鸡蛋"], "max_minutes": 20})
        return response({"recommendations": [bad_choice]})
    config, client = mock_llm(handler)
    result = Recommender(config, graph_store, client).recommend(RecommendRequest(question="我有番茄和鸡蛋，20分钟内", use_llm=True))
    assert result["generation"]["mode"] == "fallback"
    assert all(r["recipe_id"] != "invented" for r in result["recommendations"])
    assert result["validation"]["passed"]


def test_timeout_returns_labeled_fallback_without_secrets(graph_store):
    def handler(request):
        raise httpx.ReadTimeout("secret-that-must-not-leak", request=request)
    config, client = mock_llm(handler)
    result = Recommender(config, graph_store, client).recommend(RecommendRequest(question="我有番茄和鸡蛋", use_llm=True))
    assert result["generation"]["mode"] == "fallback"
    assert "超时" in result["generation"]["error"]
    assert "secret-that-must-not-leak" not in json.dumps(result)


def test_llm_cannot_relax_exclusion_time_or_opt_into_shopping(graph_store):
    def handler(request):
        return response({"available_ingredients": ["番茄", "鸡蛋"], "excluded_ingredients": [], "max_minutes": 120, "required_tags": [], "allow_missing": True})
    config, client = mock_llm(handler)
    result = Recommender(config, graph_store, client).recommend(RecommendRequest(question="我有番茄和鸡蛋，20分钟内，不能吃鸡蛋", use_llm=True))
    assert result["status"] == "no_match"
    assert result["constraints"]["max_minutes"] == 20
    assert result["constraints"]["allow_missing"] is False
    assert "鸡蛋" in result["constraints"]["excluded_ingredients"]


def test_unknown_inventory_is_not_silently_deleted(graph_store, settings):
    result = Recommender(settings, graph_store).recommend(RecommendRequest(question="我只有龙虾和海参，20分钟内能做什么"))
    assert result["constraints"]["available_ingredients"] == ["龙虾", "海参"]
    assert result["status"] == "no_match"
