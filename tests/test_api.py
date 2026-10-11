from fastapi.testclient import TestClient
from urllib.parse import quote
import re

from recipegen.app import create_app
from recipegen.pipeline import Recommender


def test_api_and_static_flow(graph_store, settings):
    with TestClient(create_app(settings, Recommender(settings, graph_store))) as client:
        assert client.get("/").status_code == 200
        status = client.get("/api/status").json()
        assert status["dataset"]["is_demo"]
        assert status["graph"]["recipes"] == 20
        assert not status["llm"]["configured"]
        result = client.post("/api/recommend", json={"question": "我有番茄和鸡蛋，20分钟以内"})
        assert result.status_code == 200
        assert result.json()["status"] == "ok"
        rid = result.json()["recommendations"][0]["recipe_id"]
        graph = client.get("/api/graph", params={"recipe_id": rid}).json()
        assert graph["nodes"] and graph["edges"]
        assert client.get("/static/app.js").status_code == 200


def test_schema_errors_and_explicit_overrides(graph_store, settings):
    with TestClient(create_app(settings, Recommender(settings, graph_store))) as client:
        assert client.post("/api/recommend", json={"question": ""}).status_code == 422
        assert client.post("/api/recommend", json={"question": "推荐", "constraints": {"max_minutes": -1}}).status_code == 422
        assert client.post("/api/recommend", json={"question": "推荐", "malicious": "ignored?"}).status_code == 422
        result = client.post("/api/recommend", json={"question": "我有鸡蛋", "constraints": {"available_ingredients": ["西红柿", "鸡蛋"], "max_minutes": 10}}).json()
        assert result["constraints"]["available_ingredients"] == ["番茄", "鸡蛋"]
        assert result["status"] == "ok"


def test_remote_graph_failure_does_not_switch_to_demo(settings):
    class BrokenStore:
        def document(self):
            raise ConnectionError("a remote database is unavailable")
    with TestClient(create_app(settings, Recommender(settings, BrokenStore()))) as client:
        assert client.get("/api/status").status_code == 503
        response = client.post("/api/recommend", json={"question": "我有鸡蛋"})
        assert response.status_code == 503
        assert "不会自动切换" in response.json()["detail"]


def test_main_workspace_navigation_and_legacy_routes_preserve_api_404(settings):
    """Main-entry modes survive refresh while existing links still work."""
    recipe_id = "recipegen:test:test-video:004de93faeb0fba12069"
    pages = {
        "/": "home",
        "/?view=explore": "explore",
        "/?view=chat": "chef-chat",
        "/?view=../../.env": "home",
        "/explore": "explore",
        "/recipes/" + quote(recipe_id, safe=""): "recipe",
        "/compose": "chef-chat",
        "/about": "about",
    }
    with TestClient(create_app(settings)) as client:
        for url, page in pages.items():
            response = client.get(url)
            assert response.status_code == 200, url
            assert response.headers["content-type"].startswith("text/html"), url
            body = re.search(r"<body\b[^>]*>", response.text).group()
            assert f'data-page="{page}"' in body, url
        home = client.get("/").text
        assert 'id="search-form"' not in home
        assert 'id="generate-form"' not in home
        assert 'id="stat-recipes"' not in home
        assert client.get("/api/unknown-route").status_code == 404
