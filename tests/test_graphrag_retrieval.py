"""手工 stub 仅验证只读合同，不代表真实 embedding 或语义效果。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from types import SimpleNamespace

import pytest

from recipegen import graphrag_retrieval as g
from recipegen.catalog import CatalogError, CatalogRequestError, RecipeCatalog

BUILD = {"build_id": "test-fixture", "dataset_revision": g.REVISION, "split": "test",
         "active": True, "status": "verified"}
IDS = ["recipegen:test:test:" + str(i) * 20 for i in range(1, 4)]
MANIFEST = {"schema_version": 1, "status": "verified", "namespace": g.TEXT_NAMESPACE,
    "signature": "e" * 64,
    "dataset": {"id": g.DATASET_ID, "revision": g.REVISION, "split": "test", "is_demo": False},
    "build_id": BUILD["build_id"], "index": {"name": "recipegen_text_fixture", "label": g.TEXT_LABEL,
        "property": "embedding", "dimensions": 384, "similarity_function": "cosine"},
    "embedding": {"model": g.EMBEDDING_MODEL, "revision": g.EMBEDDING_REVISION, "dimension": 384}}


def detail(rid, ingredient="pork"):
    return {"id": rid, "title": "Fixture original recipe", "build": copy.deepcopy(BUILD),
        "steps": [{"id": rid + ":step", "order": 1, "text": "Original step.", "source_id": rid + ":source"}],
        "sources": [{"source_id": rid + ":source", "url": "https://example.invalid/original"}],
        "ingredient_mentions": [{"id": rid + ":ingredient", "name": ingredient, "zh_name": "猪肉" if ingredient == "pork" else "鸡蛋"}],
        "media": {"images": 0, "videos": 0}, "semantics": {"ingredients_complete": False, "duration_known": False}}


def hit(rid, score=.8, **changes):
    original = detail(rid)
    text = original["title"] + "\n" + "\n".join(step["text"] for step in original["steps"])
    return {"recipe_id": rid, "source_recipe_id": rid, "score": score, "embedding_id": "text:" + rid,
        "namespace": g.TEXT_NAMESPACE, "build_id": BUILD["build_id"], "dataset_revision": g.REVISION,
        "split": "test", "embedding_model": g.EMBEDDING_MODEL, "embedding_revision": g.EMBEDDING_REVISION,
        "index_signature": MANIFEST["signature"],
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "step_ids": [rid + ":step"], "source_ids": [rid + ":source"],
        "ingredient_ids": [rid + ":ingredient"], **changes}


class Record(dict):
    def data(self):
        return copy.deepcopy(dict(self))


class Catalog:
    def __init__(self):
        self.details = {rid: detail(rid) for rid in IDS}
        self.keyword_ids = IDS[:2]
        self.calls = []
        self._connection = SimpleNamespace(database="actual-fixture-db")
        self._settings = None
        self._driver = None

    def status(self):
        return {"backend": "neo4j", "build": copy.deepcopy(BUILD)}

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"backend": "neo4j", "build": copy.deepcopy(BUILD),
                "recipes": [copy.deepcopy(self.details[rid]) for rid in self.keyword_ids]}

    def recipe(self, rid):
        self.calls.append(("recipe", rid))
        return copy.deepcopy(self.details[rid])

    def _connect(self):
        self.calls.append(("connect",))
        return self._driver


class Driver:
    def __init__(self):
        self.rows = [{"name": MANIFEST["index"]["name"], "state": "ONLINE", "labelsOrTypes": [g.TEXT_LABEL],
            "properties": ["embedding"], "options": {"indexConfig": {"vector.dimensions": 384,
            "vector.similarity_function": "cosine"}}}]
        self.calls = []
        self.failure = None

    def execute_query(self, query, parameters, **kwargs):
        assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b", query, re.I)
        assert kwargs["routing_"] == "r"
        self.calls.append((query, parameters, kwargs))
        if self.failure:
            raise self.failure
        return [Record(row) for row in self.rows], None, None


class Retriever:
    def __init__(self):
        self.rows = [hit(IDS[1], .9), hit(IDS[2], .8)]
        self.calls = []
        self.failure = None

    def get_search_results(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return SimpleNamespace(records=[Record(row) for row in self.rows])


@pytest.fixture
def backend(monkeypatch):
    catalog, driver, retriever = Catalog(), Driver(), Retriever()
    catalog._driver = driver
    factory_calls = []

    def factory(actual_driver, manifest, database):
        factory_calls.append((actual_driver, manifest, database))
        return retriever

    monkeypatch.setattr(g, "_make_vector_retriever", factory)
    service = g.RecipeGraphRAGRetriever(catalog, driver, copy.deepcopy(MANIFEST))
    return service, catalog, driver, retriever, factory_calls


def vector():
    return [1.] + [0.] * 383  # 手工合同输入，不是真实模型输出。


def vector_request(revision=g.EMBEDDING_REVISION):
    return {"query_vector": vector(), "embedding_model": g.EMBEDDING_MODEL, "embedding_revision": revision}


def test_semantic_uses_vectorcypher_contract_and_verified_original_details(backend):
    service, catalog, driver, retriever, factories = backend
    output = service.search("请找猪肉，不要鸡蛋", ingredients=["猪肉"], mode="semantic", **vector_request(), limit=2)
    call = retriever.calls[0]
    assert set(call) == {"query_vector", "top_k", "effective_search_ratio", "query_params"}
    assert call["query_vector"] == vector() and call["top_k"] == 50
    assert call["query_params"]["owner"] == g.NAMESPACE
    assert call["query_params"]["text_owner"] == g.TEXT_NAMESPACE
    assert call["query_params"]["embedding_revision"] == MANIFEST["embedding"]["revision"]
    assert call["query_params"]["index_signature"] == MANIFEST["signature"]
    assert call["query_params"]["ingredients"][0]["names"] == ["pork", "猪肉"]
    assert call["query_params"]["excluded"][0]["names"] == ["egg"]
    assert factories[0][0] is driver and factories[0][2] == "actual-fixture-db"
    assert output["recipes"] == [catalog.details[IDS[1]], catalog.details[IDS[2]]]
    assert output["graph_evidence"][0]["source_ids"] == [IDS[1] + ":source"]
    assert output["applied_filters"]["excluded_ingredients"] == ["egg"]
    assert output["retrieval"]["retriever"] == "neo4j_graphrag.VectorCypherRetriever"
    assert output["retrieval"]["llm_called"] is False
    assert all(call[0] != "search" for call in catalog.calls)


def test_expansion_is_bounded_fixed_scope_and_filters_nodes_and_edges():
    query = g.GRAPH_EXPANSION_QUERY
    assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b", query, re.I)
    assert "*" not in query and "node:RecipeGenText" in query
    assert "node.source_recipe_id = r.kg_csv_id" in query
    assert "node.index_signature = $index_signature" in query
    assert "link.embedding_model = $embedding_model" in query
    assert "link.embedding_revision = $embedding_revision" in query
    assert "all(term IN $ingredients" in query and "none(term IN $excluded" in query
    for name in ("node", "link", "r", "h", "s", "i"):
        assert f"{name}.kg_import_namespace" in query
        assert f"{name}.build_id = $build_id" in query
        assert f"{name}.dataset_revision = $dataset_revision" in query
        assert f"{name}.split = 'test'" in query
    assert "HAS_STEP" in query and "HAS_SOURCE" in query and "HAS_INGREDIENT" in query


def test_keyword_does_not_need_manifest_vector_library_or_model(backend):
    service, catalog, driver, retriever, factories = backend
    service.manifest_input = {"not": "an index"}
    service.embed_query = lambda _: pytest.fail("keyword cannot invoke embedding")
    output = service.search("pork", mode="keyword", limit=1)
    assert output["recipes"] == [catalog.details[IDS[0]]]
    assert output["retrieval"]["effective_mode"] == "keyword"
    assert not driver.calls and not retriever.calls and not factories
    assert output["retrieval"]["returned_candidates"] == 2
    assert output["retrieval"]["eligible_candidates"] == 2


def test_keyword_term_budget_rejects_before_catalog_status_or_connection(monkeypatch):
    catalog = RecipeCatalog()
    monkeypatch.setattr(catalog, "status", lambda: pytest.fail("invalid request cannot inspect database status"))
    monkeypatch.setattr(catalog, "_connect", lambda: pytest.fail("invalid request cannot connect"))
    service = g.RecipeGraphRAGRetriever(catalog)
    query = " ".join(f"unknown{i}" for i in range(13))
    with pytest.raises(CatalogRequestError, match="12 个关键词"):
        service.search(query, mode="keyword")


def test_hybrid_keyword_term_budget_can_continue_semantic(backend, monkeypatch):
    service, catalog, _, retriever, _ = backend
    # 使用实际目录的本地校验；失败发生在任何真实连接之前。
    actual_catalog = RecipeCatalog()
    monkeypatch.setattr(actual_catalog, "_connect", lambda: pytest.fail("keyword validation should reject locally"))
    monkeypatch.setattr(catalog, "search", actual_catalog.search)
    query = " ".join(f"unknown{i}" for i in range(13))
    output = service.search(query, **vector_request())
    assert retriever.calls and output["count"] == 2
    assert output["retrieval"]["effective_mode"] == "semantic"
    assert output["retrieval"]["fallback_reason"] == "keyword_query_not_supported"
    assert output["retrieval"]["keyword"] == {"available": False, "returned_candidates": 0}
    assert output["retrieval"]["returned_candidates"] == 2
    assert output["retrieval"]["eligible_candidates"] == 2


def test_hybrid_neither_keyword_nor_semantic_available_is_an_error(backend, monkeypatch, tmp_path):
    service, catalog, _, _, _ = backend
    monkeypatch.setattr(catalog, "search", RecipeCatalog().search)
    service.manifest_input = tmp_path / "not-created.json"
    query = " ".join(f"unknown{i}" for i in range(13))
    with pytest.raises(g.GraphRAGUnavailable) as captured:
        service.search(query, **vector_request())
    assert captured.value.reason == "keyword_and_semantic_unavailable"


def test_hybrid_rrf_deduplicates_each_list_and_preserves_original_facts(backend):
    service, catalog, _, retriever, _ = backend
    # B is keyword rank 2 + semantic rank 1; A keyword rank 1; C semantic rank 2.
    retriever.rows.append(hit(IDS[1], .7, embedding_id="text:duplicate"))
    keyword = catalog.search("pork")
    keyword["recipes"].append(copy.deepcopy(keyword["recipes"][0]))
    catalog.calls.clear()
    output = service.search("pork", **vector_request(), keyword_results=keyword, limit=3)
    assert [row["id"] for row in output["recipes"]] == [IDS[1], IDS[0], IDS[2]]
    assert output["retrieval"]["rankings"][0]["rrf_score"] == pytest.approx(1/62 + 1/61)
    assert output["retrieval"]["rankings"][0]["semantic_score"] == .9
    assert output["retrieval"]["returned_candidates"] == 3
    assert output["retrieval"]["eligible_candidates"] == 3
    assert all(call[0] != "search" for call in catalog.calls)
    assert len([call for call in catalog.calls if call[0] == "recipe"]) == 3


def test_semantic_can_answer_when_keyword_list_is_empty(backend):
    service, catalog, _, _, _ = backend
    catalog.keyword_ids = []
    output = service.search("a dish involving a steaming step", **vector_request())
    assert output["count"] == 2
    assert output["retrieval"]["effective_mode"] == "hybrid"


@pytest.mark.parametrize("mutation", ["missing", "populating", "manifest_pending", "dependency", "read_error"])
def test_hybrid_fallback_is_explicit_and_semantic_mode_never_falls_back(backend, monkeypatch, tmp_path, mutation):
    service, catalog, driver, retriever, _ = backend
    if mutation == "missing": service.manifest_input = tmp_path / "not-created.json"
    elif mutation == "populating": driver.rows[0]["state"] = "POPULATING"
    elif mutation == "manifest_pending": service.manifest_input["status"] = "building"
    elif mutation == "dependency":
        monkeypatch.setattr(g, "_make_vector_retriever", lambda *_: (_ for _ in ()).throw(g.GraphRAGUnavailable("graphrag_dependency_unavailable")))
    elif mutation == "read_error": retriever.failure = RuntimeError("synthetic-upstream-secret")
    output = service.search("pork", **vector_request())
    assert output["retrieval"]["status"] == "fallback"
    assert output["retrieval"]["effective_mode"] == "keyword"
    assert output["retrieval"]["fallback_reason"]
    assert output["retrieval"]["keyword"]["available"] is True
    assert "synthetic-upstream-secret" not in json.dumps(output)
    assert output["recipes"] == [catalog.details[rid] for rid in IDS[:2]]
    with pytest.raises(g.GraphRAGUnavailable, match="未就绪"):
        service.search("pork", mode="semantic", **vector_request())


def test_candidate_shortfall_is_not_reported_as_no_match_in_whole_graph(backend):
    service, _, _, retriever, _ = backend
    retriever.rows = []
    output = service.search("unmatched query", mode="semantic", **vector_request())
    assert output["count"] == 0
    assert output["retrieval"]["status"] == "ok"
    assert output["retrieval"]["insufficient_candidates"] is True
    assert output["retrieval"]["no_match_scope"] == "retrieved_candidates_only"
    assert output["retrieval"]["exhaustive"] is False


def test_post_read_hard_filters_cannot_be_lost_by_vector_or_hybrid(backend):
    service, catalog, _, _, _ = backend
    catalog.details[IDS[1]] = detail(IDS[1], "egg")
    output = service.search("pork without eggs", ingredients=["pork"], excluded_ingredients=["onion"], **vector_request())
    assert IDS[1] not in [row["id"] for row in output["recipes"]]
    assert output["retrieval"]["filtered_candidates"] == 1
    assert output["retrieval"]["returned_candidates"] == 3
    assert output["retrieval"]["eligible_candidates"] == 2
    assert output["retrieval"]["keyword"]["returned_candidates"] == 2
    assert output["applied_filters"]["ingredients"] == ["pork"]
    assert output["applied_filters"]["excluded_ingredients"] == ["onion", "egg"]
    assert output["applied_filters"]["inventory_sufficient"] is False


@pytest.mark.parametrize("field,value", [("namespace", "other"), ("build_id", "stale"), ("split", "train"),
    ("dataset_revision", "c"*40), ("embedding_model", "other-model"), ("embedding_revision", "c"*40),
    ("index_signature", "f" * 64),
    ("source_recipe_id", IDS[0]), ("text_sha256", "not-a-hash"), ("score", math.nan),
    ("step_ids", ["other:step"]), ("source_ids", ["other:source"]), ("ingredient_ids", ["other:ingredient"])])
def test_scope_model_hash_and_expansion_mismatch_fail_closed_even_in_hybrid(backend, field, value):
    service, _, _, retriever, _ = backend
    retriever.rows[0][field] = value
    with pytest.raises(CatalogError):
        service.search("pork", **vector_request())


@pytest.mark.parametrize("field,value", [("namespace", "train-space"), ("build_id", "other-build"),
    ("signature", None), ("signature", "latest"),
    ("schema_version", 9), ("embedding", {"model": "other", "revision": "a"*40, "dimension": 384}),
    ("dataset", {"id": g.DATASET_ID, "revision": g.REVISION, "split": "train", "is_demo": False})])
def test_manifest_identity_mismatch_is_not_hidden_as_keyword_fallback(backend, field, value):
    service, _, _, _, _ = backend
    service.manifest_input[field] = value
    with pytest.raises(CatalogError):
        service.search("pork", **vector_request())


def test_manifest_model_revision_must_match_fixed_local_encoder_even_without_external_vector(backend):
    service, catalog, driver, retriever, _ = backend
    service.manifest_input["embedding"]["revision"] = "f" * 40
    service.embed_query = lambda _: pytest.fail("wrong-revision index cannot load an encoder")
    with pytest.raises(CatalogError, match="向量空间"):
        service.search("pork", mode="semantic")
    assert not driver.calls and not retriever.calls


@pytest.mark.parametrize("tamper", ["hash", "title", "step_text"])
def test_semantic_hash_recomputed_from_full_original_text_even_when_ids_are_valid(backend, tamper):
    service, catalog, _, retriever, _ = backend
    rid = IDS[1]
    if tamper == "hash":
        retriever.rows[0]["text_sha256"] = "b" * 64
    elif tamper == "title":
        catalog.details[rid]["title"] += " changed"
    else:
        catalog.details[rid]["steps"][0]["text"] += " Changed original instruction."
    with pytest.raises(CatalogError, match="哈希与完整原菜谱文本不一致"):
        service.search("pork", **vector_request())


def test_real_index_shape_must_match_manifest(backend):
    service, _, driver, _, _ = backend
    driver.rows[0]["labelsOrTypes"] = ["VisualObservation"]
    with pytest.raises(CatalogError, match="索引结构"):
        service.search("pork", **vector_request())


def test_native_index_schema_accepts_uppercase_cosine_from_neo4j_2026(backend):
    service, _, driver, _, _ = backend
    driver.rows[0]["options"]["indexConfig"]["vector.similarity_function"] = "COSINE"
    output = service.search("pork", mode="semantic", **vector_request())
    assert output["retrieval"]["effective_mode"] == "semantic"
    assert output["count"] == 2


@pytest.mark.parametrize("similarity", [None, 1, "euclidean", "cosine "])
def test_native_index_schema_does_not_accept_invalid_similarity(backend, similarity):
    service, _, driver, _, _ = backend
    driver.rows[0]["options"]["indexConfig"]["vector.similarity_function"] = similarity
    with pytest.raises(CatalogError, match="索引结构"):
        service.search("pork", mode="semantic", **vector_request())


@pytest.mark.parametrize("kwargs", [{"mode": "invent"}, {"limit": 0}, {"candidate_limit": 2},
    {"mode": []},
    {"candidate_limit": 201}, {"query_vector": [0.]*384}, {"query_vector": [True]*384},
    {"ingredients": ["pork"]*21}, {"excluded_ingredients": ["x\n"]}])
def test_invalid_request_is_rejected_before_graph_reads(backend, kwargs):
    service, catalog, driver, retriever, _ = backend
    with pytest.raises(CatalogRequestError):
        service.search("pork", **kwargs)
    assert not catalog.calls and not driver.calls and not retriever.calls


def test_supplied_query_embedding_identity_must_match_index(backend):
    service, _, _, _, _ = backend
    with pytest.raises(CatalogRequestError, match="模型/revision"):
        service.search("pork", **vector_request(revision="f"*40))


@pytest.mark.parametrize("identity", [{}, {"embedding_model": g.EMBEDDING_MODEL},
    {"embedding_revision": g.EMBEDDING_REVISION}, {"embedding_model": "other", "embedding_revision": g.EMBEDDING_REVISION},
    {"embedding_model": g.EMBEDDING_MODEL, "embedding_revision": "latest"}])
def test_external_vector_requires_declared_model_revision_before_graph_reads(backend, identity):
    service, catalog, driver, retriever, _ = backend
    with pytest.raises(CatalogRequestError, match="外部 query_vector"):
        service.search("pork", query_vector=vector(), **identity)
    assert not catalog.calls and not driver.calls and not retriever.calls


def test_injected_embedder_and_shared_catalog_driver_are_used_only_for_semantic(backend):
    service, catalog, driver, retriever, _ = backend
    seen = []
    service.driver = None
    service.embed_query = lambda query: seen.append(query) or vector()
    output = service.search("pork", mode="semantic")
    assert seen == ["pork"] and ("connect",) in catalog.calls
    assert service.driver is driver and retriever.calls[0]["query_vector"] == vector()
    assert output["retrieval"]["effective_mode"] == "semantic"


def test_embedding_failure_is_safe_and_hybrid_may_fallback(backend):
    service, _, _, _, _ = backend
    service.embed_query = lambda _: (_ for _ in ()).throw(RuntimeError("synthetic-upstream-secret"))
    output = service.search("pork")
    assert output["retrieval"]["fallback_reason"] == "text_embedding_unavailable"
    assert "synthetic-upstream-secret" not in json.dumps(output)


def test_status_reports_keyword_when_semantic_index_is_not_ready(backend):
    service, _, driver, _, _ = backend
    driver.rows = []
    output = service.status()
    assert output["status"] == "unavailable" and output["keyword"]["available"] is True
    assert output["semantic"]["available"] is False and output["read_only"] is True


def test_status_checks_vector_library_initialization_without_loading_embeddings(backend, monkeypatch):
    service, _, _, retriever, _ = backend
    service.embed_query = lambda _: pytest.fail("status cannot embed text")
    assert service.status()["semantic"]["available"] is True
    assert not retriever.calls
    monkeypatch.setattr(g, "_make_vector_retriever", lambda *_: (_ for _ in ()).throw(g.GraphRAGUnavailable("graphrag_dependency_unavailable")))
    status = service.status()
    assert status["status"] == "unavailable" and status["reason"] == "graphrag_dependency_unavailable"


def test_default_factory_calls_real_library_interface_with_fixed_read_query(monkeypatch):
    # Only replace the library constructor, not the module's factory: verifies
    # its import/signature contract without connecting to a database.
    import sys
    from types import ModuleType
    driver = Driver()
    seen = {}
    fake_module = ModuleType("neo4j_graphrag.retrievers")

    def constructor(actual_driver, index_name, retrieval_query, **kwargs):
        seen.update(driver=actual_driver, index=index_name, query=retrieval_query, **kwargs)
        return "contract-retriever"

    fake_module.VectorCypherRetriever = constructor
    monkeypatch.setitem(sys.modules, "neo4j_graphrag.retrievers", fake_module)
    assert g._make_vector_retriever(driver, MANIFEST, "test-db") == "contract-retriever"
    assert seen["driver"] is driver and seen["index"] == MANIFEST["index"]["name"]
    assert seen["query"] == g.GRAPH_EXPANSION_QUERY and seen["embedder"] is None
    assert seen["neo4j_database"] == "test-db"
