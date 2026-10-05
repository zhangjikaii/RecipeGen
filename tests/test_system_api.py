"""Real-system HTTP contracts with isolated graph and generation dependencies."""
from __future__ import annotations

import copy
from fastapi.testclient import TestClient
import pytest

from recipegen.app import create_app
from recipegen.config import Settings


RECIPE = {"id": "recipe:tomato", "title": "Tomato soup", "steps": [{"id": "step:1", "order": 1,
          "text": "Cook tomatoes.", "source_id": "source:steps"}],
          "ingredient_mentions": [{"id": "ingredient:tomato", "name": "tomato", "zh_name": "番茄"}],
          "sources": [{"source_id": "source:steps", "url": "https://example.test/source", "role": "steps"}],
          "media": {"images": 1, "videos": 0, "recognized_images": 0},
          "semantics": {"ingredients_complete": False, "duration_known": False}}


class Catalog:
    def __init__(self): self.closed = False; self.calls = []
    def status(self):
        return {"backend": "neo4j", "dataset": {"id": "contract", "name": "isolated contract data", "is_demo": False},
                "graph": {"recipes": 1}}
    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {"query": query, "count": 1, "recipes": [copy.deepcopy(RECIPE)], "applied_filters": kwargs}
    def recipe(self, identifier):
        from recipegen.catalog import RecipeNotFound
        if identifier != RECIPE["id"]: raise RecipeNotFound("missing")
        return copy.deepcopy(RECIPE)
    def graph(self, identifier):
        self.recipe(identifier)
        return {"nodes": [{"id": identifier, "label": "Recipe", "name": "Tomato soup"}], "edges": []}
    def close(self): self.closed = True


class Generator:
    def status(self): return {"default_mode": "grounded", "local_model_available": False}
    def generate(self, question, recipes, **kwargs):
        return {"status": "ok" if recipes else "no_match", "answer": "bounded contract response",
                "recipes": recipes, "generation": {"mode": kwargs["mode"], "llm_called": False},
                "trace": [], "validation": {"passed": True, "checks": []}, "evidence": []}


class Retriever:
    """仅隔离HTTP转发合同；真实GraphRAG查询另有独立/原生验收。"""
    def __init__(self, catalog): self.catalog = catalog; self.calls = []
    def status(self):
        return {"keyword": {"available": True}, "semantic": {"available": False}, "status": "unavailable"}
    def search(self, query, *, mode="hybrid", **kwargs):
        self.calls.append({"query": query, "mode": mode, **kwargs})
        result = self.catalog.search(query, **kwargs)
        result["retrieval"] = {"mode": mode, "effective_mode": "keyword", "status": "fallback"}
        return result


@pytest.fixture
def system_client(tmp_path):
    catalog = Catalog()
    with TestClient(create_app(Settings(db_path=tmp_path / "never-created.sqlite3"),
                              catalog=catalog, generator=Generator(), retriever=Retriever(catalog))) as client:
        yield client, catalog, tmp_path
    assert catalog.closed


def test_real_system_flow_uses_catalog_and_never_creates_demo_database(system_client):
    client, catalog, path = system_client
    status = client.get("/api/system/status")
    assert status.status_code == 200 and status.json()["backend"] == "neo4j"
    assert status.json()["dataset"]["is_demo"] is False
    assert client.post("/api/search", json={"query": "番茄", "ingredients": ["西红柿"]}).json()["count"] == 1
    assert client.get("/api/recipes/recipe:tomato").json()["steps"][0]["text"] == "Cook tomatoes."
    assert client.get("/api/system/graph", params={"recipe_id": "recipe:tomato"}).json()["nodes"]
    result = client.post("/api/generate", json={"question": "番茄怎么做", "recipe_ids": ["recipe:tomato"]})
    assert result.status_code == 200 and result.json()["status"] == "ok"
    assert result.json()["trace"][0]["stage"] == "graph_retrieval"
    assert not (path / "never-created.sqlite3").exists()


def test_generation_without_selection_retrieves_real_evidence(system_client):
    client, catalog, _ = system_client
    result = client.post("/api/generate", json={"question": "番茄怎么做", "ingredients": ["番茄"]})
    assert result.status_code == 200
    assert catalog.calls[-1][0] == "番茄怎么做"
    assert catalog.calls[-1][1]["ingredients"] == ["番茄"]


def test_explicit_selection_respects_known_ingredient_exclusion(system_client):
    client, _, _ = system_client
    result = client.post("/api/generate", json={"question": "生成", "recipe_ids": ["recipe:tomato"],
                                               "excluded_ingredients": ["西红柿"]})
    assert result.status_code == 200 and result.json()["status"] == "no_match"
    assert result.json()["retrieval"]["inventory_or_exclusion_verified"] is False


def test_explicit_selection_still_parses_exclusions_in_changed_question(system_client):
    client, _, _ = system_client
    result = client.post("/api/generate", json={"question": "生成做法，不要番茄", "recipe_ids": ["recipe:tomato"]})
    assert result.status_code == 200 and result.json()["status"] == "no_match"


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/search", {"limit": 200}), ("/api/search", {"ingredients": [""]}),
    ("/api/search", {"query": "a" * 501}), ("/api/search", {"admin": True}),
    ("/api/generate", {"question": ""}), ("/api/generate", {"question": "generate", "mode": "remote"}),
    ("/api/generate", {"question": "generate", "recipe_ids": ["recipe:tomato", "recipe:tomato"]}),
    ("/api/generate", {"question": "generate", "recipe_ids": ["MATCH (n) DELETE n"]}),
])
def test_invalid_system_requests_are_rejected(system_client, endpoint, payload):
    client, _, _ = system_client
    assert client.post(endpoint, json=payload).status_code == 422


def test_missing_real_recipe_is_404(system_client):
    client, _, _ = system_client
    assert client.get("/api/recipes/recipe:missing").status_code == 404
    assert client.post("/api/generate", json={"question": "generate", "recipe_ids": ["recipe:missing"]}).status_code == 404


def test_real_graph_unavailable_cannot_fall_back_to_demo(tmp_path):
    class Broken(Catalog):
        def status(self): raise ConnectionError("private connection details must not escape")
        def search(self, *_args, **_kwargs): raise ConnectionError("private secret")
    broken = Broken()
    with TestClient(create_app(Settings(db_path=tmp_path / "never.sqlite3"), catalog=broken,
                              generator=Generator(), retriever=Retriever(broken))) as client:
        result = client.get("/api/system/status")
        assert result.status_code == 503 and "人工示例" in result.json()["detail"]
        assert "private" not in result.text
        assert client.post("/api/search", json={"query": "tomato"}).status_code == 503
        assert not (tmp_path / "never.sqlite3").exists()


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/generate", {"question": "x" * 501, "recipe_ids": ["recipe:tomato"]}),
    ("/api/generate", {"question": "tea", "ingredients": ["x" * 81], "recipe_ids": ["recipe:tomato"]}),
    ("/api/search", {"ingredients": ["item" + str(i) for i in range(21)]}),
    ("/api/search", {"excluded_ingredients": ["item" + str(i) for i in range(21)]}),
    ("/api/search", {"ingredients": ["x" * 81]}),
])
def test_public_boundaries_reject_before_graph_calls(system_client, endpoint, payload):
    client, catalog, _ = system_client
    assert client.post(endpoint, json=payload).status_code == 422
    assert catalog.calls == []


def test_maximum_public_generation_boundaries_reach_real_generator(tmp_path):
    from recipegen.system_generation import RecipeGenerator
    with TestClient(create_app(Settings(db_path=tmp_path / "unused.sqlite3"),
                              catalog=Catalog(), generator=RecipeGenerator())) as client:
        response = client.post("/api/generate", json={"question": "x" * 500,
                               "recipe_ids": [RECIPE["id"]], "ingredients": ["x" * 80]})
    assert response.status_code == 200
    assert response.json()["validation"]["passed"]
    assert response.json()["generation"]["llm_called"] is False


def test_actual_catalog_query_condition_error_is_422_without_connecting(tmp_path):
    from recipegen.catalog import RecipeCatalog
    catalog = RecipeCatalog()
    with TestClient(create_app(Settings(db_path=tmp_path / "unused.sqlite3"), catalog=catalog,
                              generator=Generator())) as client:
        response = client.post("/api/search", json={"query": " ".join("unknown" + str(i) for i in range(13)),
                                                   "retrieval_mode": "keyword"})
        bad_id = client.get("/api/recipes/" + "x" * 201)
    assert response.status_code == 422
    assert "Neo4j" not in response.json()["detail"]
    assert bad_id.status_code == 422
    assert catalog._driver is None


def test_catalog_evidence_failures_remain_503(tmp_path):
    from recipegen.catalog import CatalogError
    class InvalidGraph(Catalog):
        def search(self, *_args, **_kwargs):
            raise CatalogError("private invalid fact source")
    invalid = InvalidGraph()
    with TestClient(create_app(Settings(db_path=tmp_path / "unused.sqlite3"), catalog=invalid,
                              generator=Generator(), retriever=Retriever(invalid))) as client:
        response = client.post("/api/search", json={"query": "tomato"})
    assert response.status_code == 503
    assert "private invalid fact" not in response.text


def test_frontend_generation_limit_matches_public_request_contract():
    from pathlib import Path
    import re
    from recipegen.system_api_models import GenerateRequest
    html = (Path(__file__).resolve().parents[1] / "static/index.html").read_text()
    field = re.search(r'<textarea\b[^>]*\bid="generation-question"[^>]*>', html).group()
    assert 'maxlength="500"' in field
    assert GenerateRequest(question="x" * 500).question == "x" * 500


def test_retrieval_mode_is_forwarded_to_graph_workflow(tmp_path):
    catalog = Catalog()
    retriever = Retriever(catalog)
    with TestClient(create_app(Settings(db_path=tmp_path / "unused.sqlite3"), catalog=catalog,
                              generator=Generator(), retriever=retriever)) as client:
        assert client.post("/api/search", json={"query": "清爽的凉菜", "retrieval_mode": "semantic"}).status_code == 200
        assert retriever.calls[-1]["mode"] == "semantic"
        assert client.post("/api/generate", json={"question": "清爽的凉菜", "retrieval_mode": "hybrid"}).status_code == 200
        assert retriever.calls[-1]["mode"] == "hybrid"
        assert client.post("/api/search", json={"query": "番茄", "retrieval_mode": "unknown"}).status_code == 422


def test_api_answer_can_be_previewed_while_key_is_pending(tmp_path):
    from recipegen.graphrag_answer import GraphRAGAnswerer
    settings = Settings(db_path=tmp_path / "unused.sqlite3")
    with TestClient(create_app(settings, catalog=Catalog(), generator=Generator(),
                              api_answerer=GraphRAGAnswerer(settings=settings))) as client:
        response = client.post("/api/generate", json={"question": "整理做法", "mode": "api",
                                                     "recipe_ids": [RECIPE["id"]]})
        assert response.status_code == 200
        result = response.json()
        assert result["generation"]["pending_api"] is True
        assert result["generation"]["llm_called"] is False
        assert result["generation"]["fallback"] is False
        assert result["generation"]["effective_mode"] == "grounded"
        assert "Cook tomatoes." in result["answer"]


def test_omitted_retrieval_mode_uses_server_configuration(tmp_path):
    catalog = Catalog()
    retriever = Retriever(catalog)
    settings = Settings(db_path=tmp_path / "unused.sqlite3", retrieval_mode="keyword")
    with TestClient(create_app(settings, catalog=catalog, generator=Generator(), retriever=retriever)) as client:
        assert client.post("/api/search", json={"query": "番茄"}).status_code == 200
        assert retriever.calls[-1]["mode"] == "keyword"
        assert client.post("/api/generate", json={"question": "番茄怎么做"}).status_code == 200
        assert retriever.calls[-1]["mode"] == "keyword"


def test_semantic_unavailable_has_same_safe_message_on_search_and_generate(tmp_path):
    from recipegen.graphrag_retrieval import GraphRAGUnavailable
    class Missing(Retriever):
        def search(self, *_args, **_kwargs):
            raise GraphRAGUnavailable("private_error")
    catalog = Catalog()
    with TestClient(create_app(Settings(db_path=tmp_path / "unused.sqlite3"), catalog=catalog,
                              generator=Generator(), retriever=Missing(catalog))) as client:
        search = client.post("/api/search", json={"query": "番茄", "retrieval_mode": "semantic"})
        generated = client.post("/api/generate", json={"question": "番茄", "retrieval_mode": "semantic"})
    assert search.status_code == generated.status_code == 503
    assert search.json()["detail"] == generated.json()["detail"]
    assert "private_error" not in search.text + generated.text
