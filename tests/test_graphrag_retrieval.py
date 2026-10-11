"""手工 stub 验证图谱先筛、精确排序与范围校验合同，不代表真实检索效果。"""
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
from recipegen.production_hybrid import corpus_identity_hash

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


def hybrid_hit(rid, cosine=.8, **changes):
    row = {key: value for key, value in hit(rid).items()
           if key not in {"score", "step_ids", "source_ids", "ingredient_ids"}}
    row.update(embedding=[cosine, math.sqrt(1 - cosine * cosine)] + [0.] * 382,
               ingredient_names=[{"name": "pork", "zh_name": "猪肉"}])
    row.update(changes)
    return row


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
        self.envelope = {"graph_candidate_count": 2, "scored_recipe_count": 2,
                         "matches": [hit(IDS[1], .9), hit(IDS[2], .8)]}
        self.hybrid_envelope = {"graph_candidate_count": 2, "scored_recipe_count": 2,
                                "matches": [hybrid_hit(IDS[1], .9), hybrid_hit(IDS[2], .8)]}
        self.evidence_changes = {}
        self.calls = []
        self.failure = None
        self.search_failure = None

    def execute_query(self, query, parameters, **kwargs):
        assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b", query, re.I)
        assert kwargs["routing_"] == "r"
        self.calls.append((query, parameters, kwargs))
        if self.failure:
            raise self.failure
        if query == g.INDEX_STATUS_QUERY:
            return [Record(row) for row in self.rows], None, None
        if self.search_failure:
            raise self.search_failure
        if query == g.GRAPH_FIRST_QUERY:
            return [Record(self.envelope)], None, None
        if query == g.GRAPH_HYBRID_CANDIDATES_QUERY:
            return [Record(self.hybrid_envelope)], None, None
        assert query == g.HYBRID_EVIDENCE_QUERY
        return [Record({**{key: value for key, value in hit(rid).items() if key != "score"},
                        **self.evidence_changes.get(rid, {})}) for rid in parameters["recipe_ids"]], None, None


def set_corpus(service, catalog, path):
    docs = []
    for rid, recipe in catalog.details.items():
        text = recipe["title"] + "\n" + "\n".join(step["text"] for step in recipe["steps"])
        docs.append({"recipe_id": rid, "title": recipe["title"], "steps": recipe["steps"],
                     "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
    service.corpus_path = path
    service.manifest_input["corpus_sha256"] = corpus_identity_hash(docs)
    service.manifest_input["index"]["count"] = len(docs)


@pytest.fixture
def backend(tmp_path):
    catalog, driver = Catalog(), Driver()
    catalog._driver = driver
    service = g.RecipeGraphRAGRetriever(catalog, driver, copy.deepcopy(MANIFEST))
    set_corpus(service, catalog, tmp_path / "corpus.jsonl")
    return service, catalog, driver


def vector():
    return [1.] + [0.] * 383  # 手工合同输入，不是真实模型输出。


def vector_request(revision=g.EMBEDDING_REVISION):
    return {"query_vector": vector(), "embedding_model": g.EMBEDDING_MODEL, "embedding_revision": revision}


def test_default_semantic_executes_fixed_graph_first_query_and_verified_details(backend):
    service, catalog, driver = backend
    output = service.search("请找猪肉，不要鸡蛋", **vector_request(), limit=2)
    call = driver.calls[-1]
    assert call[0] == g.GRAPH_FIRST_QUERY and call[2]["database_"] == "actual-fixture-db"
    params = call[1]
    assert params["query_vector"] == vector() and params["candidate_limit"] == 50
    assert params["owner"] == g.NAMESPACE and params["text_owner"] == g.TEXT_NAMESPACE
    assert params["embedding_revision"] == MANIFEST["embedding"]["revision"]
    assert params["index_signature"] == MANIFEST["signature"]
    assert params["ingredients"][0]["names"] == ["pork"]
    assert params["excluded"][0]["names"] == ["egg"]
    assert output["recipes"] == [catalog.details[IDS[1]], catalog.details[IDS[2]]]
    assert output["graph_evidence"][0]["source_ids"] == [IDS[1] + ":source"]
    assert output["applied_filters"]["ingredients"] == ["pork"]
    assert output["applied_filters"]["excluded_ingredients"] == ["egg"]
    retrieval = output["retrieval"]
    assert retrieval["method"] == "graph_first_e5_exact_cosine"
    assert retrieval["score_semantics"] == "neo4j_cosine_similarity_0_1"
    assert retrieval["graph_candidate_count"] == retrieval["scored_recipe_count"] == 2
    assert retrieval["coverage"] == "all_graph_eligible_recipes_scored"
    assert retrieval["graph_constraint_scan_exhaustive"] is True
    assert retrieval["index_used_for_ann"] is False and retrieval["approximate_search"] is False
    assert retrieval["llm_called"] is False
    assert all(call[0] != "search" for call in catalog.calls)


def test_chat_structured_filters_do_not_promote_optional_or_negative_prose(backend):
    service, _, driver = backend
    query = "找猪肉食谱，不使用花生，可加鸡蛋但不必需"
    seen = []
    service.embed_query = lambda value: seen.append(value) or vector()
    output = service.search(query, ingredients=["猪肉"], excluded_ingredients=["花生"],
                            infer_query_filters=False)
    params = driver.calls[-1][1]
    assert len(params["ingredients"]) == 1 and params["ingredients"][0]["names"][0] == "pork"
    assert len(params["excluded"]) == 1 and params["excluded"][0]["names"][0] == "peanut"
    assert seen == [query]  # E5 still encodes the complete semantic question.
    assert output["count"] == 2
    assert output["retrieval"]["query_filter_inference"] is False
    assert output["applied_filters"]["ingredients"] == ["pork"]
    assert output["applied_filters"]["positive_mentions"] == []


@pytest.mark.parametrize("value,mode", [("false", "semantic"), (0, "semantic"),
                                       (False, "keyword")])
def test_query_inference_switch_rejects_invalid_or_ambiguous_modes(backend, value, mode):
    service, catalog, driver = backend
    with pytest.raises(CatalogRequestError):
        service.search("pork", mode=mode, infer_query_filters=value, **vector_request())
    assert not driver.calls and not catalog.calls


def test_fixed_query_filters_all_graph_candidates_before_scoring_or_limiting():
    query = g.GRAPH_FIRST_QUERY
    assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b", query, re.I)
    assert "*" not in query and "node:RecipeGenText" in query
    assert "node.source_recipe_id = r.kg_csv_id" in query
    assert "node.index_signature = $index_signature" in query
    assert "link.embedding_model = $embedding_model" in query
    assert "link.embedding_revision = $embedding_revision" in query
    assert "all(term IN $ingredients" in query and "none(term IN $excluded" in query
    assert query.index("all(term IN $ingredients") < query.index("collect(r) AS graph_candidates")
    assert query.index("none(term IN $excluded") < query.index("vector.similarity.cosine")
    assert query.index("vector.similarity.cosine") < query.index("LIMIT $candidate_limit")
    assert query.count("LIMIT") == 1 and "db.index.vector.queryNodes" not in query
    for name in ("node", "link", "r", "h", "s", "i"):
        assert f"{name}.kg_import_namespace" in query
        assert f"{name}.build_id = $build_id" in query
        assert f"{name}.dataset_revision = $dataset_revision" in query
        assert f"{name}.split = 'test'" in query
    assert "HAS_STEP" in query and "HAS_SOURCE" in query and "HAS_INGREDIENT" in query


def test_graph_candidate_coverage_is_not_limited_to_top50(backend):
    service, catalog, driver = backend
    ids = [f"recipegen:test:fixture:{n:020}" for n in range(50)]
    catalog.details.update({rid: detail(rid) for rid in ids})
    driver.envelope = {"graph_candidate_count": 125, "scored_recipe_count": 125,
                       "matches": [hit(rid, 1 - n / 100) for n, rid in enumerate(ids)]}
    output = service.search("pork", **vector_request(), limit=10)
    assert output["count"] == 10
    assert output["retrieval"]["returned_candidates"] == 50
    assert output["retrieval"]["graph_candidate_count"] == 125
    assert output["retrieval"]["scored_recipe_count"] == 125
    assert output["retrieval"]["semantic"]["returned_after_graph_filter"] == 50


def test_hybrid_executes_three_routes_and_two_rank_rrf_before_top_evidence_expansion(backend):
    service, catalog, _ = backend
    output = service.search("pork", mode="hybrid", keyword_results={"invalid": "unused"}, **vector_request())
    assert [row["id"] for row in output["recipes"]] == IDS[1:]
    assert output["retrieval"]["mode"] == output["retrieval"]["effective_mode"] == "hybrid"
    assert output["retrieval"]["method"] == "graph_first_bm25_e5_rrf"
    assert output["retrieval"]["rrf_k"] == 60
    assert output["retrieval"]["graph_has_rank"] is False
    assert all(rank["rrf_score"] > 0 and rank["bm25_rank"] is None for rank in output["retrieval"]["rankings"])
    assert all(call[0] != "search" for call in catalog.calls)
    service, _, driver = backend
    assert [call[0] for call in driver.calls] == [g.INDEX_STATUS_QUERY, g.GRAPH_HYBRID_CANDIDATES_QUERY, g.HYBRID_EVIDENCE_QUERY]
    assert "query_vector" not in driver.calls[1][1]  # E5 dot runs only after C is materialized.
    assert [row["stage"] for row in output["trace"]] == ["bm25_retrieval", "graph_condition_retrieval",
        "graph_prefiltered_e5", "candidate_condition_check", "rrf_fusion", "graph_evidence_expansion"]
    assert driver.calls[-1][1]["recipe_ids"] == output["retrieval"]["seed_ids"]["dense"]


def test_unconstrained_question_scores_all_recipes_without_invented_filters(backend):
    service, _, driver = backend
    service.search("a dish involving a steaming step", **vector_request())
    assert driver.calls[-1][1]["ingredients"] == []
    assert driver.calls[-1][1]["excluded"] == []


def test_keyword_does_not_need_manifest_vector_library_or_model(backend):
    service, catalog, driver = backend
    service.manifest_input = {"not": "an index"}
    service.embed_query = lambda _: pytest.fail("keyword cannot invoke embedding")
    output = service.search("pork", mode="keyword", limit=1)
    assert output["recipes"] == [catalog.details[IDS[0]]]
    assert output["retrieval"]["effective_mode"] == "keyword" and not driver.calls
    assert output["retrieval"]["returned_candidates"] == 2
    assert output["retrieval"]["graph_candidate_count"] is None
    assert output["retrieval"]["graph_constraint_scan_exhaustive"] is False


def test_keyword_term_budget_rejects_before_catalog_status_or_connection(monkeypatch):
    catalog = RecipeCatalog()
    monkeypatch.setattr(catalog, "status", lambda: pytest.fail("invalid request cannot inspect database status"))
    monkeypatch.setattr(catalog, "_connect", lambda: pytest.fail("invalid request cannot connect"))
    service = g.RecipeGraphRAGRetriever(catalog)
    with pytest.raises(CatalogRequestError, match="12 个关键词"):
        service.search(" ".join(f"unknown{i}" for i in range(13)), mode="keyword")


@pytest.mark.parametrize("mutation", ["missing", "populating", "manifest_pending", "read_error"])
def test_only_legacy_hybrid_may_explicitly_fallback_while_default_fails(backend, tmp_path, mutation):
    service, catalog, driver = backend
    if mutation == "missing": service.manifest_input = tmp_path / "not-created.json"
    elif mutation == "populating": driver.rows[0]["state"] = "POPULATING"
    elif mutation == "manifest_pending": service.manifest_input["status"] = "building"
    elif mutation == "read_error": driver.search_failure = RuntimeError("synthetic-upstream-secret")
    output = service.search("pork", mode="hybrid", **vector_request())
    assert output["retrieval"]["status"] == "fallback"
    assert output["retrieval"]["effective_mode"] == "keyword"
    assert output["retrieval"]["fallback_reason"]
    assert output["retrieval"]["graph_constraint_scan_exhaustive"] is False
    assert "synthetic-upstream-secret" not in json.dumps(output)
    assert output["recipes"] == [catalog.details[rid] for rid in IDS[:2]]
    with pytest.raises(g.GraphRAGUnavailable, match="未就绪"):
        service.search("pork", **vector_request())


def test_hybrid_neither_keyword_nor_semantic_available_is_an_error(backend, monkeypatch, tmp_path):
    service, catalog, _ = backend
    monkeypatch.setattr(catalog, "search", RecipeCatalog().search)
    service.manifest_input = tmp_path / "not-created.json"
    with pytest.raises(g.GraphRAGUnavailable) as captured:
        service.search(" ".join(f"unknown{i}" for i in range(13)), mode="hybrid", **vector_request())
    assert captured.value.reason == "keyword_and_semantic_unavailable"


def test_no_candidate_means_zero_ingredient_mentions_in_verified_graph(backend):
    service, _, driver = backend
    driver.envelope = {"graph_candidate_count": 0, "scored_recipe_count": 0, "matches": []}
    output = service.search("pork", **vector_request())
    assert output["count"] == 0 and output["retrieval"]["status"] == "ok"
    assert output["retrieval"]["insufficient_candidates"] is True
    assert output["retrieval"]["no_match_scope"] == "verified_graph_ingredient_mentions"
    assert output["retrieval"]["graph_constraint_scan_exhaustive"] is True
    assert output["retrieval"]["exhaustive"] is False


@pytest.mark.parametrize("change", [
    {"graph_candidate_count": 3}, {"scored_recipe_count": 1}, {"graph_candidate_count": -1},
    {"graph_candidate_count": True}, {"scored_recipe_count": 2.0}, {"matches": None},
    {"matches": []}, {"matches": [None]}])
def test_partial_or_invalid_candidate_coverage_fails_closed(backend, change):
    service, _, driver = backend
    driver.hybrid_envelope.update(change)
    with pytest.raises(CatalogError, match="覆盖不完整"):
        service.search("pork", mode="hybrid", **vector_request())


def test_duplicate_recipe_output_fails_closed_instead_of_claiming_full_coverage(backend):
    service, _, driver = backend
    driver.envelope["matches"][1] = hit(IDS[1], .7, embedding_id="text:duplicate")
    with pytest.raises(CatalogError, match="去重"):
        service.search("pork", **vector_request())


def test_post_read_graph_filter_mismatch_fails_closed(backend):
    service, catalog, _ = backend
    catalog.details[IDS[1]] = detail(IDS[1], "egg")
    with pytest.raises(CatalogError, match="图谱先筛条件"):
        service.search("pork without eggs", **vector_request())


@pytest.mark.parametrize("field,value", [("namespace", "other"), ("build_id", "stale"), ("split", "train"),
    ("dataset_revision", "c"*40), ("embedding_model", "other-model"), ("embedding_revision", "c"*40),
    ("index_signature", "f" * 64), ("source_recipe_id", IDS[0]), ("text_sha256", "not-a-hash"),
    ("score", math.nan), ("score", -0.5), ("score", 1.5), ("step_ids", ["other:step"]),
    ("source_ids", ["other:source"]), ("ingredient_ids", ["other:ingredient"])])
def test_semantic_scope_model_hash_and_expansion_mismatch_fail_closed(backend, field, value):
    service, _, driver = backend
    driver.envelope["matches"][0][field] = value
    with pytest.raises(CatalogError):
        service.search("pork", mode="semantic", **vector_request())


@pytest.mark.parametrize("field,value", [("namespace", "train-space"), ("build_id", "other-build"),
    ("signature", None), ("signature", "latest"), ("schema_version", 9),
    ("embedding", {"model": "other", "revision": "a"*40, "dimension": 384}),
    ("dataset", {"id": g.DATASET_ID, "revision": g.REVISION, "split": "train", "is_demo": False})])
def test_manifest_identity_mismatch_is_not_hidden_as_keyword_fallback(backend, field, value):
    service, _, _ = backend
    service.manifest_input[field] = value
    with pytest.raises(CatalogError):
        service.search("pork", mode="hybrid", **vector_request())


def test_manifest_revision_must_match_fixed_local_encoder_even_without_external_vector(backend):
    service, _, driver = backend
    service.manifest_input["embedding"]["revision"] = "f" * 40
    service.embed_query = lambda _: pytest.fail("wrong-revision index cannot load an encoder")
    with pytest.raises(CatalogError, match="向量空间"):
        service.search("pork")
    assert not driver.calls


@pytest.mark.parametrize("tamper", ["hash", "title", "step_text"])
def test_hash_recomputed_from_full_original_text_even_when_ids_are_valid(backend, tamper):
    service, catalog, driver = backend
    rid = IDS[1]
    if tamper == "hash": driver.envelope["matches"][0]["text_sha256"] = "b" * 64
    elif tamper == "title": catalog.details[rid]["title"] += " changed"
    else: catalog.details[rid]["steps"][0]["text"] += " Changed original instruction."
    with pytest.raises(CatalogError, match="哈希与完整原菜谱文本不一致"):
        service.search("pork", **vector_request())


def test_real_index_shape_must_match_manifest(backend):
    service, _, driver = backend
    driver.rows[0]["labelsOrTypes"] = ["VisualObservation"]
    with pytest.raises(CatalogError, match="索引结构"):
        service.search("pork", **vector_request())


def test_native_index_schema_accepts_uppercase_cosine_from_neo4j_2026(backend):
    service, _, driver = backend
    driver.rows[0]["options"]["indexConfig"]["vector.similarity_function"] = "COSINE"
    assert service.search("pork", **vector_request())["count"] == 2


@pytest.mark.parametrize("similarity", [None, 1, "euclidean", "cosine "])
def test_native_index_schema_rejects_invalid_similarity(backend, similarity):
    service, _, driver = backend
    driver.rows[0]["options"]["indexConfig"]["vector.similarity_function"] = similarity
    with pytest.raises(CatalogError, match="索引结构"):
        service.search("pork", **vector_request())


@pytest.mark.parametrize("kwargs", [{"mode": "invent"}, {"limit": 0}, {"candidate_limit": 2},
    {"mode": []}, {"candidate_limit": 201}, {"query_vector": [0.]*384}, {"query_vector": [True]*384},
    {"ingredients": ["pork"]*21}, {"excluded_ingredients": ["x\n"]}])
def test_invalid_request_rejected_before_graph_reads(backend, kwargs):
    service, catalog, driver = backend
    with pytest.raises(CatalogRequestError):
        service.search("pork", **kwargs)
    assert not catalog.calls and not driver.calls


def test_supplied_query_embedding_identity_must_match_index(backend):
    service, _, _ = backend
    with pytest.raises(CatalogRequestError, match="模型/revision"):
        service.search("pork", **vector_request(revision="f"*40))


@pytest.mark.parametrize("identity", [{}, {"embedding_model": g.EMBEDDING_MODEL},
    {"embedding_revision": g.EMBEDDING_REVISION}, {"embedding_model": "other", "embedding_revision": g.EMBEDDING_REVISION},
    {"embedding_model": g.EMBEDDING_MODEL, "embedding_revision": "latest"}])
def test_external_vector_requires_declared_identity_before_graph_reads(backend, identity):
    service, catalog, driver = backend
    with pytest.raises(CatalogRequestError, match="外部 query_vector"):
        service.search("pork", query_vector=vector(), **identity)
    assert not catalog.calls and not driver.calls


def test_injected_embedder_and_shared_catalog_driver_used_only_for_semantic(backend):
    service, catalog, driver = backend
    seen = []
    service.driver = None
    service.embed_query = lambda query: seen.append(query) or vector()
    output = service.search("pork")
    assert seen == ["pork"] and ("connect",) in catalog.calls
    assert service.driver is driver and driver.calls[-1][1]["query_vector"] == vector()
    assert output["retrieval"]["effective_mode"] == "semantic"


def test_embedding_failure_is_safe_and_default_never_silently_falls_back(backend):
    service, _, _ = backend
    service.embed_query = lambda _: (_ for _ in ()).throw(RuntimeError("synthetic-upstream-secret"))
    with pytest.raises(g.GraphRAGUnavailable) as captured:
        service.search("pork")
    assert captured.value.reason == "text_embedding_unavailable"
    output = service.search("pork", mode="hybrid")
    assert output["retrieval"]["fallback_reason"] == "text_embedding_unavailable"
    assert "synthetic-upstream-secret" not in json.dumps(output)


def test_status_reports_keyword_when_semantic_index_not_ready(backend):
    service, _, driver = backend
    driver.rows = []
    output = service.status()
    assert output["status"] == "unavailable" and output["keyword"]["available"] is True
    assert output["semantic"]["available"] is False and output["read_only"] is True


def test_status_does_not_embed_or_perform_approximate_retrieval(backend):
    service, _, driver = backend
    service.embed_query = lambda _: pytest.fail("status cannot embed text")
    status = service.status()
    assert status["semantic"]["available"] is True and status["approximate_search"] is False
    assert [call[0] for call in driver.calls] == [g.INDEX_STATUS_QUERY]


def test_hybrid_queries_are_fixed_read_only_exhaustive_candidates_then_bounded_evidence():
    query = g.GRAPH_HYBRID_CANDIDATES_QUERY
    assert "all(term IN $ingredients" in query and "none(term IN $excluded" in query
    assert "embedding:node.embedding" in query
    assert "LIMIT" not in query and "vector.similarity" not in query and "queryNodes" not in query
    assert query.index("all(term IN $ingredients") < query.index("embedding:node.embedding")
    for query in (query, g.HYBRID_EVIDENCE_QUERY):
        assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b", query, re.I)
        assert "*" not in query
        for name in ("node", "link", "r", "h", "i"):
            assert f"{name}.kg_import_namespace" in query
            assert f"{name}.build_id = $build_id" in query
            assert f"{name}.dataset_revision = $dataset_revision" in query
            assert f"{name}.split = 'test'" in query
    assert "UNWIND $recipe_ids" in g.HYBRID_EVIDENCE_QUERY
    assert "HAS_STEP" in g.HYBRID_EVIDENCE_QUERY and "HAS_SOURCE" in g.HYBRID_EVIDENCE_QUERY


def test_hybrid_keyword_illegal_high_score_does_not_enter_graph_qualified_dense_or_final_pool(backend):
    service, catalog, driver = backend
    catalog.details[IDS[0]]["title"] = "pork pork pork"
    set_corpus(service, catalog, service.corpus_path)
    output = service.search("pork", mode="hybrid", **vector_request())
    retrieval = output["retrieval"]
    assert retrieval["seed_ids"]["bm25"] == [IDS[0]]
    assert set(retrieval["seed_ids"]["dense"]) == set(IDS[1:])
    assert retrieval["candidate_pool"] == IDS[1:]
    assert retrieval["candidate_counts"]["filtered_after_seed"] == 1
    assert all(row["cosine_score"] is not None for row in retrieval["rankings"])
    assert all(call != ("recipe", IDS[0]) for call in catalog.calls)


def test_hybrid_scores_all_graph_candidates_including_beyond_top50_then_expands_only_top_rrf(backend):
    service, catalog, driver = backend
    ids = [f"recipegen:test:fixture:{number:020d}" for number in range(60)]
    catalog.details = {rid: detail(rid) for rid in ids}
    set_corpus(service, catalog, service.corpus_path)
    driver.hybrid_envelope = {"graph_candidate_count": 60, "scored_recipe_count": 60,
        "matches": [hybrid_hit(rid, .99 - number / 100) for number, rid in enumerate(ids)]}
    output = service.search("pork", mode="hybrid", **vector_request(), limit=10, candidate_limit=10)
    retrieval = output["retrieval"]
    assert retrieval["graph_candidate_count"] == retrieval["scored_recipe_count"] == 60
    assert len(retrieval["seed_ids"]["dense"]) == 50 and len(retrieval["seed_ids"]["graph"]) == 60
    assert len(retrieval["candidate_pool"]) == 60
    assert retrieval["candidate_counts"]["graph_only_added"] == 10
    assert len(driver.calls[-1][1]["recipe_ids"]) == 10
    assert len([call for call in catalog.calls if call[0] == "recipe"]) == 10
    assert len(retrieval["candidate_rankings"]) == output["count"] == 10


def test_hybrid_rrf_uses_real_positive_bm25_and_can_change_pure_e5_order(backend):
    service, catalog, driver = backend
    catalog.details[IDS[1]]["title"] = "pork original recipe"
    set_corpus(service, catalog, service.corpus_path)
    text = catalog.details[IDS[1]]["title"] + "\nOriginal step."
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    driver.hybrid_envelope["matches"] = [hybrid_hit(IDS[1], .2, text_sha256=text_hash), hybrid_hit(IDS[2], .9)]
    driver.evidence_changes[IDS[1]] = {"text_sha256": text_hash}
    output = service.search("pork", mode="hybrid", **vector_request())
    retrieval = output["retrieval"]
    assert retrieval["seed_ids"]["dense"] == [IDS[2], IDS[1]]
    assert [row["recipe_id"] for row in retrieval["rankings"]] == [IDS[1], IDS[2]]
    first, second = retrieval["rankings"]
    assert first["bm25_rank"] == 1 and first["dense_rank"] == 2
    assert first["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert second["bm25_rank"] is None and second["dense_rank"] == 1
    assert second["rrf_score"] == pytest.approx(1 / 61)
    assert all("graph_rank" not in row for row in retrieval["rankings"])


def test_hybrid_structured_route_filters_do_not_reparse_optional_or_negative_prose(backend):
    service, _, driver = backend
    output = service.search("找猪肉食谱，不使用花生，可加鸡蛋但不必需", mode="hybrid",
                            ingredients=["猪肉"], excluded_ingredients=["花生"],
                            infer_query_filters=False, **vector_request())
    params = driver.calls[1][1]
    assert len(params["ingredients"]) == 1 and params["ingredients"][0]["names"][0] == "pork"
    assert len(params["excluded"]) == 1 and params["excluded"][0]["names"][0] == "peanut"
    assert output["retrieval"]["query_filter_inference"] is False
    assert output["retrieval"]["keyword"]["lexical_query"] == "pork"
    assert output["applied_filters"]["positive_mentions"] == []


def test_hybrid_empty_graph_pool_is_valid_and_never_expands_unqualified_keyword_seed(backend):
    service, catalog, driver = backend
    catalog.details[IDS[0]]["title"] = "pork"
    set_corpus(service, catalog, service.corpus_path)
    driver.hybrid_envelope = {"graph_candidate_count": 0, "scored_recipe_count": 0, "matches": []}
    output = service.search("pork", mode="hybrid", **vector_request())
    assert output["count"] == 0 and output["retrieval"]["candidate_pool"] == []
    assert output["retrieval"]["seed_ids"]["bm25"] == [IDS[0]]
    assert [call[0] for call in driver.calls] == [g.INDEX_STATUS_QUERY, g.GRAPH_HYBRID_CANDIDATES_QUERY]
    assert all(call[0] != "recipe" for call in catalog.calls)


def test_hybrid_no_conditions_requires_entire_corpus_and_reuses_verified_bm25_cache(backend):
    service, _, driver = backend
    driver.hybrid_envelope = {"graph_candidate_count": 3, "scored_recipe_count": 3,
        "matches": [hybrid_hit(rid, .9 - number / 10) for number, rid in enumerate(IDS)]}
    service.search("a steaming recipe", mode="hybrid", **vector_request())
    cached = service._bm25_cache
    service.search("another steaming recipe", mode="hybrid", **vector_request())
    assert service._bm25_cache is cached
    driver.hybrid_envelope["matches"].pop()
    driver.hybrid_envelope.update(graph_candidate_count=2, scored_recipe_count=2)
    with pytest.raises(CatalogError, match="完整原食谱语料"):
        service.search("a steaming recipe", mode="hybrid", **vector_request())


def test_hybrid_corpus_tampering_after_cache_fill_fails_closed(backend):
    service, _, _ = backend
    service.search("pork", mode="hybrid", **vector_request())
    docs = [json.loads(line) for line in service.corpus_path.read_text().splitlines() if line.strip()]
    docs[0]["recipe_id"] = "recipegen:test:test:" + "0" * 20
    service.corpus_path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
    with pytest.raises(CatalogError, match="manifest 哈希不一致"):
        service.search("pork", mode="hybrid", **vector_request())


def test_hybrid_corpus_identity_matches_original_builder_and_accepts_json_formatting_changes(backend):
    from scripts.build_text_index import fingerprint

    service, _, _ = backend
    docs = [json.loads(line) for line in service.corpus_path.read_text().splitlines() if line.strip()]
    expected = fingerprint([{key: doc[key] for key in ("recipe_id", "text_sha256")} for doc in docs])
    assert expected == service.manifest_input["corpus_sha256"] == corpus_identity_hash(docs)
    assert expected != hashlib.sha256(service.corpus_path.read_bytes()).hexdigest()
    service.corpus_path.write_text("\n" + "".join(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n\n" for doc in docs), encoding="utf-8")
    assert service.search("pork", mode="hybrid", **vector_request())["count"] == 2


@pytest.mark.parametrize("mutation", ["text", "order"])
def test_hybrid_original_corpus_text_and_identity_row_order_are_verified(backend, mutation):
    service, _, _ = backend
    docs = [json.loads(line) for line in service.corpus_path.read_text().splitlines() if line.strip()]
    if mutation == "text":
        docs[0]["text"] += " changed"
    else:
        docs.reverse()
    service.corpus_path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
    with pytest.raises(CatalogError):
        service.search("pork", mode="hybrid", **vector_request())


@pytest.mark.parametrize("field,value", [("namespace", "other"), ("build_id", "stale"),
    ("split", "train"), ("dataset_revision", "c" * 40), ("embedding_model", "other"),
    ("embedding_revision", "c" * 40), ("index_signature", "f" * 64),
    ("source_recipe_id", IDS[0]), ("text_sha256", "f" * 64),
    ("embedding", [0.] * 384), ("embedding", [2.] + [0.] * 383),
    ("ingredient_names", [{"name": "egg", "zh_name": "鸡蛋"}])])
def test_hybrid_graph_candidate_scope_vector_and_condition_mismatch_fail_closed(backend, field, value):
    service, _, driver = backend
    driver.hybrid_envelope["matches"][0][field] = value
    with pytest.raises(CatalogError):
        service.search("pork", mode="hybrid", **vector_request())


@pytest.mark.parametrize("field,value", [("step_ids", ["other:step"]),
    ("source_ids", ["other:source"]), ("ingredient_ids", ["other:ingredient"]),
    ("embedding_id", "text:changed"), ("text_sha256", "f" * 64)])
def test_hybrid_expanded_evidence_must_match_candidate_and_original_details(backend, field, value):
    service, _, driver = backend
    driver.evidence_changes[IDS[1]] = {field: value}
    with pytest.raises(CatalogError):
        service.search("pork", mode="hybrid", **vector_request())
