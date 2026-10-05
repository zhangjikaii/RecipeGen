"""Neo4j GraphRAG 文本召回；只读原事实，不调用生成模型或处理媒体。

派生向量只用于找到候选。完整菜谱仍由 RecipeCatalog 读取、验证，
检索不足仅描述有限候选集合，不宣称全库不存在匹配。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

from .catalog import (CatalogError, CatalogRequestError, DATASET_ID, NAMESPACE,
                      REVISION, normalize_ingredient, parse_query_filters)
from .config import PROJECT_ROOT
from .text_embeddings import MODEL as EMBEDDING_MODEL, REVISION as EMBEDDING_REVISION

TEXT_NAMESPACE = "recipegen-text-v1"
TEXT_LABEL = "RecipeGenText"
DIMENSION = 384
RRF_K = 60
DEFAULT_MANIFEST = PROJECT_ROOT / "data/graphrag/index-manifest.json"


class GraphRAGUnavailable(CatalogError):
    """语义索引/运行依赖未就绪；安全消息不包含上游配置或异常内容。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("GraphRAG 文本检索未就绪；检查公开索引状态与本地依赖。")


INDEX_STATUS_QUERY = """// recipegen-graphrag:index-status
SHOW VECTOR INDEXES YIELD name,state,labelsOrTypes,properties,options
WHERE name = $index_name
RETURN name,state,labelsOrTypes,properties,options
"""

# 只接受固定 TEXT_OF、HAS_STEP、HAS_INGREDIENT、HAS_SOURCE 一阶关系。
# VectorCypherRetriever 提供 node/score；索引名字并不能替代逐节点范围校验。
GRAPH_EXPANSION_QUERY = """// recipegen-graphrag:bounded-expansion
WITH node,score
WHERE node:RecipeGenText AND node.kg_import_namespace = $text_owner
  AND node.build_id = $build_id AND node.dataset_revision = $dataset_revision
  AND node.split = 'test' AND node.embedding_model = $embedding_model
  AND node.embedding_revision = $embedding_revision
  AND node.index_signature = $index_signature
  AND node.text_sha256 IS NOT NULL
MATCH (node)-[link:TEXT_OF]->(r:RecipeGen:Recipe)
WHERE link.kg_import_namespace = $text_owner AND link.build_id = $build_id
  AND link.dataset_revision = $dataset_revision AND link.split = 'test'
  AND link.embedding_model = $embedding_model AND link.embedding_revision = $embedding_revision
  AND r.kg_import_namespace = $owner AND r.build_id = $build_id
  AND r.dataset_revision = $dataset_revision AND r.split = 'test'
  AND node.source_recipe_id = r.kg_csv_id
  AND all(term IN $ingredients WHERE EXISTS {
    MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
    WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
      AND h.dataset_revision = $dataset_revision AND h.split = 'test'
      AND i.kg_import_namespace = $owner AND i.build_id = $build_id
      AND i.dataset_revision = $dataset_revision AND i.split = 'test'
      AND (toLower(coalesce(i.normalized_name,'')) IN term.names
           OR toLower(coalesce(i.zh_name,'')) IN term.names) })
  AND none(term IN $excluded WHERE EXISTS {
    MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
    WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
      AND h.dataset_revision = $dataset_revision AND h.split = 'test'
      AND i.kg_import_namespace = $owner AND i.build_id = $build_id
      AND i.dataset_revision = $dataset_revision AND i.split = 'test'
      AND (toLower(coalesce(i.normalized_name,'')) IN term.names
           OR toLower(coalesce(i.zh_name,'')) IN term.names) })
CALL (r) {
 MATCH (r)-[h:HAS_STEP]->(s:RecipeGen:Step)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.dataset_revision = $dataset_revision AND h.split = 'test'
   AND s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.dataset_revision = $dataset_revision AND s.split = 'test'
 WITH DISTINCT s ORDER BY s.order,s.kg_csv_id
 RETURN collect(s.kg_csv_id) AS step_ids
}
CALL (r) {
 MATCH (r)-[h:HAS_SOURCE]->(s:RecipeGen:Source)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.dataset_revision = $dataset_revision AND h.split = 'test'
   AND s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.dataset_revision = $dataset_revision AND s.split = 'test'
 RETURN collect(DISTINCT s.kg_csv_id) AS source_ids
}
CALL (r) {
 MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.dataset_revision = $dataset_revision AND h.split = 'test'
   AND i.kg_import_namespace = $owner AND i.build_id = $build_id
   AND i.dataset_revision = $dataset_revision AND i.split = 'test'
 RETURN collect(DISTINCT i.kg_csv_id) AS ingredient_ids
}
RETURN r.kg_csv_id AS recipe_id,score,
 coalesce(node.kg_csv_id,node.id) AS embedding_id,node.source_recipe_id AS source_recipe_id,
 node.kg_import_namespace AS namespace,node.build_id AS build_id,
 node.dataset_revision AS dataset_revision,node.split AS split,
 node.embedding_model AS embedding_model,node.embedding_revision AS embedding_revision,
 node.index_signature AS index_signature,
 node.text_sha256 AS text_sha256,step_ids,source_ids,ingredient_ids
ORDER BY score DESC,recipe_id,embedding_id
"""


def validate_manifest(value: Any) -> dict[str, Any]:
    """公开 manifest 固定数据与模型身份，避免本地编码器混用向量空间。"""
    if not isinstance(value, dict):
        raise CatalogError("GraphRAG manifest 结构无效。")
    if value.get("status") != "verified":
        raise GraphRAGUnavailable("manifest_not_verified")
    dataset, index, embedding = (value.get(k) for k in ("dataset", "index", "embedding"))
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("namespace") != TEXT_NAMESPACE
            or not isinstance(dataset, dict) or dataset.get("id") != DATASET_ID
            or dataset.get("revision") != REVISION or dataset.get("split") != "test"
            or dataset.get("is_demo") is not False
            or not isinstance(value.get("build_id"), str) or not value["build_id"]
            or not isinstance(value.get("signature"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["signature"])
            or not isinstance(index, dict) or not isinstance(embedding, dict)):
        raise CatalogError("GraphRAG manifest 不属于固定真实测试构建。")
    if (not isinstance(index.get("name"), str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,119}", index["name"])
            or index.get("label") != TEXT_LABEL or index.get("property") != "embedding"
            or type(index.get("dimensions")) is not int or index["dimensions"] != DIMENSION
            or index.get("similarity_function") != "cosine"
            or embedding.get("model") != EMBEDDING_MODEL
            or type(embedding.get("dimension")) is not int or embedding["dimension"] != DIMENSION
            or embedding.get("revision") != EMBEDDING_REVISION):
        raise CatalogError("GraphRAG 文本索引或模型身份不符合固定向量空间。")
    return copy.deepcopy(value)


def _filters(values: Any, label: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    if (not isinstance(values, list) or len(values) > 20
            or any(not isinstance(v, str) or not v.strip() or len(v) > 80
                   or any(ord(c) < 32 for c in v) for v in values)):
        raise CatalogRequestError(f"{label} 必须至多 20 项，每项为 1～80 个可打印字符。")
    return [{"raw": value.strip(), "names": list(dict.fromkeys([normalize_ingredient(value), value.strip().lower()]))}
            for value in dict.fromkeys(values)]


def _vector(value: Any) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != DIMENSION
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or not any(value)):
        raise CatalogRequestError("query_vector 必须为 384 维有限、非零文本向量。")
    return [float(v) for v in value]


def _make_vector_retriever(driver, manifest, database):
    try:
        from neo4j_graphrag.retrievers import VectorCypherRetriever
    except ImportError:
        raise GraphRAGUnavailable("graphrag_dependency_unavailable") from None
    return VectorCypherRetriever(driver, manifest["index"]["name"], GRAPH_EXPANSION_QUERY,
                                 embedder=None, neo4j_database=database)


def _same_build(value, build):
    return (isinstance(value, dict) and all(value.get(k) == build.get(k)
            for k in ("build_id", "dataset_revision", "split"))
            and value.get("active") is True and value.get("status") == "verified")


def _eligible(detail, include, exclude):
    names = {normalize_ingredient(m["name"]) for m in detail["ingredient_mentions"]}
    names.update(m.get("zh_name", "").lower() for m in detail["ingredient_mentions"] if m.get("zh_name"))
    return (all(set(term["names"]) & names for term in include)
            and not any(set(term["names"]) & names for term in exclude))


class RecipeGraphRAGRetriever:
    def __init__(self, catalog, driver=None, manifest: dict | Path | str = DEFAULT_MANIFEST,
                 *, database: str | None = None, embed_query: Callable | None = None):
        self.catalog, self.driver = catalog, driver
        self.manifest_input, self.database, self.embed_query = manifest, database, embed_query

    def _manifest(self):
        if isinstance(self.manifest_input, dict):
            return validate_manifest(self.manifest_input)
        try:
            value = json.loads(Path(self.manifest_input).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise GraphRAGUnavailable("manifest_missing") from None
        except (OSError, TypeError, ValueError):
            raise CatalogError("GraphRAG 公开 manifest 无法读取。") from None
        return validate_manifest(value)

    def _connection(self):
        if self.driver is None:
            self.driver = self.catalog._connect()
        connection = getattr(self.catalog, "_connection", None)
        settings = getattr(self.catalog, "_settings", None)
        database = self.database or (connection.database if connection is not None else
                                     getattr(settings, "neo4j_database", "neo4j"))
        return self.driver, database

    def _ready(self, build):
        manifest = self._manifest()
        if manifest["build_id"] != build["build_id"]:
            raise CatalogError("GraphRAG manifest 与当前已验证图谱 build_id 不一致。")
        driver, database = self._connection()
        try:
            from neo4j import RoutingControl
            records, _, _ = driver.execute_query(INDEX_STATUS_QUERY,
                {"index_name": manifest["index"]["name"]}, database_=database, routing_=RoutingControl.READ)
        except Exception:
            raise GraphRAGUnavailable("index_status_unavailable") from None
        rows = [record.data() for record in records]
        if len(rows) != 1 or rows[0].get("state") != "ONLINE":
            raise GraphRAGUnavailable("index_not_online")
        index = rows[0]
        options = index.get("options")
        config = options.get("indexConfig") if isinstance(options, dict) else None
        if not isinstance(config, dict):
            raise CatalogError("真实 Neo4j 文本索引缺少可核验配置。")
        if (index.get("name") != manifest["index"]["name"] or index.get("labelsOrTypes") != [TEXT_LABEL]
                or index.get("properties") != ["embedding"]
                or config.get("vector.dimensions") != DIMENSION
                or not isinstance(config.get("vector.similarity_function"), str)
                or config["vector.similarity_function"].casefold() != "cosine"):
            raise CatalogError("真实 Neo4j 文本索引结构不符合 manifest。")
        return manifest, driver, database

    def status(self):
        status = self.catalog.status()
        try:
            manifest, driver, database = self._ready(status["build"])
            try:
                _make_vector_retriever(driver, manifest, database)
            except GraphRAGUnavailable:
                raise
            except Exception:
                raise GraphRAGUnavailable("vector_initialization_unavailable") from None
        except GraphRAGUnavailable as error:
            return {"status": "unavailable", "keyword": {"available": True},
                    "semantic": {"available": False}, "reason": error.reason,
                    "build": status["build"], "read_only": True}
        return {"status": "ok", "keyword": {"available": True}, "semantic": {"available": True},
                "index": manifest["index"], "embedding": manifest["embedding"],
                "namespace": TEXT_NAMESPACE, "build": status["build"], "read_only": True}

    def _semantic(self, query, vector, build, include, exclude, candidate_limit,
                  embedding_model, embedding_revision):
        manifest, driver, database = self._ready(build)
        model = manifest["embedding"]
        if ((embedding_model is not None and embedding_model != model["model"])
                or (embedding_revision is not None and embedding_revision != model["revision"])):
            raise CatalogRequestError("查询向量模型/revision 与文本索引不同。")
        if vector is None:
            if not query.strip():
                raise GraphRAGUnavailable("semantic_query_empty")
            if self.embed_query is None:
                raise GraphRAGUnavailable("text_embedder_unavailable")
            try:
                vector = self.embed_query(query)
            except Exception:
                raise GraphRAGUnavailable("text_embedding_unavailable") from None
        vector = _vector(vector)
        parameters = {"owner": NAMESPACE, "text_owner": TEXT_NAMESPACE, "build_id": build["build_id"],
                      "dataset_revision": REVISION, "embedding_model": model["model"],
                      "embedding_revision": model["revision"], "index_signature": manifest["signature"],
                      "ingredients": include, "excluded": exclude}
        try:
            retriever = _make_vector_retriever(driver, manifest, database)
            raw = retriever.get_search_results(query_vector=vector, top_k=candidate_limit,
                                              effective_search_ratio=1, query_params=parameters)
        except GraphRAGUnavailable:
            raise
        except Exception:
            raise GraphRAGUnavailable("vector_read_unavailable") from None
        rows = [record.data() for record in raw.records]
        if len(rows) > candidate_limit:
            raise CatalogError("GraphRAG 返回数量超出受限候选预算。")
        hits = {}
        for row in rows:
            if (row.get("namespace") != TEXT_NAMESPACE
                    or any(row.get(k) != parameters[k] for k in
                           ("build_id", "dataset_revision", "embedding_model", "embedding_revision", "index_signature"))
                    or row.get("split") != "test" or not isinstance(row.get("recipe_id"), str)
                    or not row["recipe_id"].startswith("recipegen:test:")
                    or row.get("source_recipe_id") != row["recipe_id"]
                    or not isinstance(row.get("embedding_id"), str) or not row["embedding_id"]
                    or not isinstance(row.get("text_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["text_sha256"])
                    or type(row.get("score")) not in (int, float) or not math.isfinite(row["score"])
                    or not 0 <= row["score"] <= 1
                    or any(not isinstance(row.get(k), list) or any(not isinstance(i, str) or not i for i in row[k])
                           or len(set(row[k])) != len(row[k]) for k in ("step_ids", "source_ids", "ingredient_ids"))):
                raise CatalogError("GraphRAG 候选的范围、模型、哈希或来源结构不一致。")
            previous = hits.get(row["recipe_id"])
            if previous is None or (row["score"], row["embedding_id"]) > (previous["score"], previous["embedding_id"]):
                hits[row["recipe_id"]] = row
        return sorted(hits.values(), key=lambda h: (-h["score"], h["recipe_id"])), manifest, len(rows)

    def search(self, query: str, ingredients=None, excluded_ingredients=None, limit: int = 6,
               mode: str = "hybrid", query_vector=None, candidate_limit: int = 50,
               keyword_results=None, *, embedding_model=None, embedding_revision=None):
        if (not isinstance(query, str) or len(query) > 500 or not isinstance(mode, str)
                or mode not in {"keyword", "semantic", "hybrid"}):
            raise CatalogRequestError("query 至多 500 字符；mode 必须为 keyword、semantic 或 hybrid。")
        if type(limit) is not int or not 1 <= limit <= 20 or type(candidate_limit) is not int or not limit <= candidate_limit <= 200:
            raise CatalogRequestError("limit 为 1～20，candidate_limit 须在 limit～200 范围。")
        if query_vector is not None:
            _vector(query_vector)
            if (embedding_model != EMBEDDING_MODEL or not isinstance(embedding_revision, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", embedding_revision)):
                raise CatalogRequestError("外部 query_vector 必须声明文本 embedding_model 与固定 embedding_revision。")
        parsed = parse_query_filters(query)
        include = _filters(ingredients, "ingredients")
        exclude = _filters(excluded_ingredients, "excluded_ingredients") + _filters(parsed["excluded"], "excluded_ingredients")
        keyword = None
        keyword_available, keyword_failure = True, None
        if mode in {"keyword", "hybrid"}:
            # 目录先做本地请求校验。keyword 模式的无效输入不能先连接数据库；
            # hybrid 的自然语言问题可能超过关键词预算，仍可独立走文本向量。
            try:
                keyword = keyword_results if keyword_results is not None else self.catalog.search(
                    query, ingredients=ingredients, excluded_ingredients=excluded_ingredients, limit=20)
            except CatalogRequestError:
                if mode == "keyword":
                    raise
                keyword_available, keyword_failure = False, "keyword_query_not_supported"
        build = self.catalog.status()["build"]
        if keyword is not None:
            if (not isinstance(keyword, dict) or keyword.get("backend") != "neo4j"
                    or not _same_build(keyword.get("build"), build) or not isinstance(keyword.get("recipes"), list)):
                raise CatalogError("关键词候选不是同一已验证真实图谱的目录结果。")
        semantic, manifest, raw_count, fallback = [], None, 0, None
        if mode in {"semantic", "hybrid"}:
            try:
                semantic, manifest, raw_count = self._semantic(query, query_vector, build, include, exclude,
                                                             candidate_limit, embedding_model, embedding_revision)
            except GraphRAGUnavailable as error:
                if mode == "semantic":
                    raise
                if not keyword_available:
                    raise GraphRAGUnavailable("keyword_and_semantic_unavailable") from None
                fallback = error.reason
        keyword_ids = list(dict.fromkeys(row["id"] for row in keyword["recipes"])) if keyword else []
        semantic_ids = [row["recipe_id"] for row in semantic]
        candidate_ids = list(dict.fromkeys(keyword_ids + semantic_ids))
        keyword_count = len(keyword_ids)
        details, filtered = {}, 0
        hits = {row["recipe_id"]: row for row in semantic}
        for rid in candidate_ids:
            detail = self.catalog.recipe(rid)
            if detail.get("id") != rid or not _same_build(detail.get("build"), build):
                raise CatalogError("GraphRAG 原菜谱详情与已验证构建不一致。")
            hit = hits.get(rid)
            if hit is not None and (set(hit["step_ids"]) != {s["id"] for s in detail["steps"]}
                    or set(hit["source_ids"]) != {s["source_id"] for s in detail["sources"]}
                    or set(hit["ingredient_ids"]) != {i["id"] for i in detail["ingredient_mentions"]}):
                raise CatalogError("GraphRAG 一阶扩展 ID 与原图谱步骤/来源/食材不一致。")
            if hit is not None:
                # 与离线 corpus 构建保持逐字相同；向量不能继续代表已变化的原文。
                steps = sorted(detail["steps"], key=lambda step: (step["order"], step["id"]))
                source_text = detail["title"] + "\n" + "\n".join(step["text"] for step in steps)
                if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != hit["text_sha256"]:
                    raise CatalogError("GraphRAG 文本向量哈希与完整原菜谱文本不一致。")
            if _eligible(detail, include, exclude):
                details[rid] = detail
            else:
                filtered += 1
        keyword_ids = [rid for rid in keyword_ids if rid in details]
        semantic_ids = [rid for rid in semantic_ids if rid in details]
        keyword_ranks = {rid: rank for rank, rid in enumerate(keyword_ids, 1)}
        semantic_ranks = {rid: rank for rank, rid in enumerate(semantic_ids, 1)}
        effective_mode = "keyword" if fallback else "semantic" if keyword_failure else mode
        fallback = fallback or keyword_failure
        fallback_message = ("语义检索未就绪，已使用关键词结果。" if effective_mode == "keyword" else
                            "关键词查询超出支持范围，已使用语义结果。") if fallback else None
        rankings = []
        for rid in details:
            score = ((1 / (RRF_K + keyword_ranks[rid]) if rid in keyword_ranks else 0)
                     + (1 / (RRF_K + semantic_ranks[rid]) if rid in semantic_ranks else 0))
            rankings.append({"recipe_id": rid, "keyword_rank": keyword_ranks.get(rid),
                             "semantic_rank": semantic_ranks.get(rid), "rrf_score": score,
                             "semantic_score": hits[rid]["score"] if rid in hits else None})
        if effective_mode == "hybrid":
            rankings.sort(key=lambda row: (-row["rrf_score"], row["recipe_id"]))
        else:
            key = "keyword_rank" if effective_mode == "keyword" else "semantic_rank"
            rankings.sort(key=lambda row: (row[key], row["recipe_id"]))
        rankings = rankings[:limit]
        selected = {row["recipe_id"] for row in rankings}
        applied = {"ingredients": list(dict.fromkeys(normalize_ingredient(t["raw"]) for t in include)),
                   "excluded_ingredients": list(dict.fromkeys(normalize_ingredient(t["raw"]) for t in exclude)),
                   "positive_query": parsed["positive_query"], "positive_mentions": parsed["positive"],
                   "semantics": "dictionary_text_mentions_only", "all_requested_mentions_required": True,
                   "inventory_sufficient": False, "allergy_safe": False}
        return {"query": query.strip(), "count": len(rankings),
                "recipes": [details[row["recipe_id"]] for row in rankings], "build": build,
                "backend": "neo4j", "read_only": True, "filters": applied, "applied_filters": applied,
                "limitations": ["文本向量只提供有限候选，不保证穷尽全库或提高语义准确率。",
                                "食材为规则提及，不能证明完整配方、库存充足或忌口安全。"],
                "retrieval": {"status": "fallback" if fallback else "ok", "mode": mode,
                    "effective_mode": effective_mode, "fallback_reason": fallback,
                    "fallback_message": fallback_message,
                    "method": "rrf" if effective_mode == "hybrid" else effective_mode,
                    "rrf_k": RRF_K if effective_mode == "hybrid" else None,
                    "retriever": "neo4j_graphrag.VectorCypherRetriever" if manifest else None,
                    "index": manifest["index"]["name"] if manifest else None,
                    "embedding": manifest["embedding"] if manifest else None,
                    "keyword": {"available": keyword_available, "returned_candidates": keyword_count},
                    "semantic": {"available": bool(manifest), "returned_after_graph_filter": raw_count},
                    "candidate_limit": candidate_limit, "returned_candidates": len(candidate_ids),
                    "eligible_candidates": len(details), "filtered_candidates": filtered,
                    "coverage": "bounded_candidates", "insufficient_candidates": len(rankings) < limit,
                    "no_match_scope": "retrieved_candidates_only", "exhaustive": False,
                    "rankings": rankings, "llm_called": False},
                "graph_evidence": [{**hits[rid], "semantics": "vector_candidate"}
                                   for rid in semantic_ids if rid in selected]}
