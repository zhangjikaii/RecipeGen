"""Neo4j GraphRAG 文本召回；只读原事实，不调用生成模型或处理媒体。

semantic 先在图谱筛选全部合法食谱，再用 E5 精确排序。
hybrid 执行 BM25 / 图谱先筛 E5 / 图谱三路候选与双排名 RRF。
完整菜谱仍由 RecipeCatalog 读取、校验；食材提及不代表完整配方或忌口安全。
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
from .production_hybrid import VerifiedBM25Corpus, lexical_query_text, score_graph_first_pool

TEXT_NAMESPACE = "recipegen-text-v1"
TEXT_LABEL = "RecipeGenText"
DIMENSION = 384
DEFAULT_MANIFEST = PROJECT_ROOT / "data/graphrag/index-manifest.json"
DEFAULT_CORPUS = PROJECT_ROOT / "data/graphrag/full/corpus.jsonl"
HYBRID_METHOD = "graph_first_bm25_e5_rrf"


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

# 固定只读 Cypher：食材条件先于向量评分，LIMIT 只限制排序后的输出。
# Neo4j vector.similarity.cosine 返回 [0,1] 分数，不能当作 [-1,1] 原始余弦。
GRAPH_FIRST_QUERY = """// recipegen-graphrag:graph-first-exact-e5
MATCH (r:RecipeGen:Recipe)
WHERE r.kg_import_namespace = $owner AND r.build_id = $build_id
  AND r.dataset_revision = $dataset_revision AND r.split = 'test'
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
WITH collect(r) AS graph_candidates
WITH graph_candidates,size(graph_candidates) AS graph_candidate_count
CALL (graph_candidates) {
 UNWIND graph_candidates AS r
 MATCH (node:RecipeGenText)-[link:TEXT_OF]->(r)
 WHERE node.kg_import_namespace = $text_owner AND node.build_id = $build_id
   AND node.dataset_revision = $dataset_revision AND node.split = 'test'
   AND node.embedding_model = $embedding_model AND node.embedding_revision = $embedding_revision
   AND node.index_signature = $index_signature AND node.text_sha256 IS NOT NULL
   AND node.source_recipe_id = r.kg_csv_id AND size(node.embedding) = $dimensions
   AND link.kg_import_namespace = $text_owner AND link.build_id = $build_id
   AND link.dataset_revision = $dataset_revision AND link.split = 'test'
   AND link.embedding_model = $embedding_model AND link.embedding_revision = $embedding_revision
 WITH r,node,vector.similarity.cosine(node.embedding,$query_vector) AS score
 WHERE score IS NOT NULL
 ORDER BY score DESC,r.kg_csv_id,coalesce(node.kg_csv_id,node.id)
 WITH r,head(collect({node:node,score:score})) AS best
 RETURN collect({recipe:r,node:best.node,score:best.score}) AS scoring_candidates
}
WITH graph_candidate_count,scoring_candidates,size(scoring_candidates) AS scored_recipe_count
CALL (scoring_candidates) {
 UNWIND scoring_candidates AS hit
 WITH hit.recipe AS r,hit.node AS node,hit.score AS score
 ORDER BY score DESC,r.kg_csv_id,coalesce(node.kg_csv_id,node.id)
 LIMIT $candidate_limit
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
 WITH r,node,score,step_ids,source_ids,ingredient_ids
 ORDER BY score DESC,r.kg_csv_id,coalesce(node.kg_csv_id,node.id)
 RETURN collect({recipe_id:r.kg_csv_id,score:score,
  embedding_id:coalesce(node.kg_csv_id,node.id),source_recipe_id:node.source_recipe_id,
  namespace:node.kg_import_namespace,build_id:node.build_id,
  dataset_revision:node.dataset_revision,split:node.split,
  embedding_model:node.embedding_model,embedding_revision:node.embedding_revision,
  index_signature:node.index_signature,text_sha256:node.text_sha256,
  step_ids:step_ids,source_ids:source_ids,ingredient_ids:ingredient_ids}) AS matches
}
RETURN graph_candidate_count,scored_recipe_count,matches
"""

# Reuse only the fixed graph predicate, before any semantic score or LIMIT.
# Each live text vector is returned for a graph-qualified Recipe; the local
# exact dot product cannot see vectors outside this complete candidate set.
GRAPH_HYBRID_CANDIDATES_QUERY = GRAPH_FIRST_QUERY.split("WITH collect(r) AS graph_candidates", 1)[0].replace(
    "recipegen-graphrag:graph-first-exact-e5", "recipegen-graphrag:hybrid-graph-qualified-vectors") + """
WITH collect(r) AS graph_candidates
WITH graph_candidates,size(graph_candidates) AS graph_candidate_count
CALL (graph_candidates) {
 UNWIND graph_candidates AS r
 MATCH (node:RecipeGenText)-[link:TEXT_OF]->(r)
 WHERE node.kg_import_namespace = $text_owner AND node.build_id = $build_id
   AND node.dataset_revision = $dataset_revision AND node.split = 'test'
   AND node.embedding_model = $embedding_model AND node.embedding_revision = $embedding_revision
   AND node.index_signature = $index_signature AND node.text_sha256 IS NOT NULL
   AND node.source_recipe_id = r.kg_csv_id AND size(node.embedding) = $dimensions
   AND link.kg_import_namespace = $text_owner AND link.build_id = $build_id
   AND link.dataset_revision = $dataset_revision AND link.split = 'test'
   AND link.embedding_model = $embedding_model AND link.embedding_revision = $embedding_revision
 CALL (r) {
  MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
  WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
    AND h.dataset_revision = $dataset_revision AND h.split = 'test'
    AND i.kg_import_namespace = $owner AND i.build_id = $build_id
    AND i.dataset_revision = $dataset_revision AND i.split = 'test'
  RETURN collect(DISTINCT {name:i.normalized_name,zh_name:i.zh_name}) AS ingredient_names
 }
 RETURN collect({recipe_id:r.kg_csv_id,embedding_id:coalesce(node.kg_csv_id,node.id),
   source_recipe_id:node.source_recipe_id,namespace:node.kg_import_namespace,
   build_id:node.build_id,dataset_revision:node.dataset_revision,split:node.split,
   embedding_model:node.embedding_model,embedding_revision:node.embedding_revision,
   index_signature:node.index_signature,text_sha256:node.text_sha256,
   embedding:node.embedding,ingredient_names:ingredient_names}) AS matches
}
RETURN graph_candidate_count,size(matches) AS scored_recipe_count,matches
"""

HYBRID_EVIDENCE_QUERY = """// recipegen-graphrag:hybrid-top-rrf-evidence
UNWIND $recipe_ids AS recipe_id
MATCH (r:RecipeGen:Recipe)
WHERE r.kg_csv_id = recipe_id AND r.kg_import_namespace = $owner AND r.build_id = $build_id
 AND r.dataset_revision = $dataset_revision AND r.split = 'test'
MATCH (node:RecipeGenText)-[link:TEXT_OF]->(r)
WHERE node.kg_import_namespace = $text_owner AND node.build_id = $build_id
 AND node.dataset_revision = $dataset_revision AND node.split = 'test'
 AND node.embedding_model = $embedding_model AND node.embedding_revision = $embedding_revision
 AND node.index_signature = $index_signature AND node.source_recipe_id = recipe_id
 AND link.kg_import_namespace = $text_owner AND link.build_id = $build_id
 AND link.dataset_revision = $dataset_revision AND link.split = 'test'
 AND link.embedding_model = $embedding_model AND link.embedding_revision = $embedding_revision
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
RETURN r.kg_csv_id AS recipe_id,node.source_recipe_id AS source_recipe_id,
 coalesce(node.kg_csv_id,node.id) AS embedding_id,node.kg_import_namespace AS namespace,
 node.build_id AS build_id,node.dataset_revision AS dataset_revision,node.split AS split,
 node.embedding_model AS embedding_model,node.embedding_revision AS embedding_revision,
 node.index_signature AS index_signature,node.text_sha256 AS text_sha256,
 step_ids,source_ids,ingredient_ids
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
                 *, database: str | None = None, embed_query: Callable | None = None,
                 corpus_path: Path | str = DEFAULT_CORPUS):
        self.catalog, self.driver = catalog, driver
        self.manifest_input, self.database, self.embed_query = manifest, database, embed_query
        self.corpus_path = Path(corpus_path)
        self._bm25_cache = None
        self._bm25_cache_key = None

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
            manifest, _, _ = self._ready(status["build"])
        except GraphRAGUnavailable as error:
            return {"status": "unavailable", "keyword": {"available": True},
                    "semantic": {"available": False}, "reason": error.reason,
                    "build": status["build"], "read_only": True}
        return {"status": "ok", "keyword": {"available": True}, "semantic": {"available": True},
                "index": manifest["index"], "embedding": manifest["embedding"],
                "namespace": TEXT_NAMESPACE, "build": status["build"], "read_only": True,
                "method": "graph_first_e5_exact_cosine", "approximate_search": False}

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
                      "ingredients": include, "excluded": exclude, "query_vector": vector,
                      "candidate_limit": candidate_limit, "dimensions": DIMENSION}
        try:
            from neo4j import RoutingControl
            records, _, _ = driver.execute_query(GRAPH_FIRST_QUERY, parameters,
                database_=database, routing_=RoutingControl.READ)
        except Exception:
            raise GraphRAGUnavailable("graph_first_vector_read_unavailable") from None
        envelopes = [record.data() for record in records]
        if len(envelopes) != 1:
            raise CatalogError("GraphRAG 精确排序缺少唯一候选覆盖记录。")
        envelope = envelopes[0]
        total, scored = envelope.get("graph_candidate_count"), envelope.get("scored_recipe_count")
        rows = envelope.get("matches")
        if (type(total) is not int or total < 0 or type(scored) is not int or scored < 0
                or not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or total != scored or len(rows) != min(scored, candidate_limit)):
            raise CatalogError("GraphRAG 食材候选的有效文本向量覆盖不完整或排序输出数量不一致。")
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
        if len(hits) != len(rows):
            raise CatalogError("GraphRAG 精确排序应按食谱去重后再限制输出。")
        return (sorted(hits.values(), key=lambda h: (-h["score"], h["recipe_id"])), manifest,
                {"graph_candidate_count": total, "scored_recipe_count": scored,
                 "returned_after_graph_filter": len(rows)})

    def _bm25(self, manifest):
        try:
            stat = self.corpus_path.stat()
        except OSError:
            raise GraphRAGUnavailable("hybrid_corpus_unavailable") from None
        key = (str(self.corpus_path.resolve()), stat.st_mtime_ns, stat.st_size,
               manifest["build_id"], manifest["signature"], manifest.get("corpus_sha256"))
        if key != self._bm25_cache_key:
            self._bm25_cache = VerifiedBM25Corpus(self.corpus_path, manifest)
            self._bm25_cache_key = key
        return self._bm25_cache

    @staticmethod
    def _hybrid_identity(row, parameters, corpus):
        rid = row.get("recipe_id")
        if (row.get("namespace") != TEXT_NAMESPACE
                or any(row.get(key) != parameters[key] for key in
                       ("build_id", "dataset_revision", "embedding_model", "embedding_revision", "index_signature"))
                or row.get("split") != "test" or not isinstance(rid, str)
                or rid not in corpus.documents or row.get("source_recipe_id") != rid
                or not isinstance(row.get("embedding_id"), str) or not row["embedding_id"]
                or row.get("text_sha256") != corpus.documents[rid]["text_sha256"]):
            raise CatalogError("三路混合候选的图谱范围、模型、来源或原文哈希不一致。")

    def _hybrid(self, query, vector, build, include, exclude, candidate_limit,
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
        corpus = self._bm25(manifest)
        lexical_query = lexical_query_text(query, include)
        bm25_scores = corpus.scores(lexical_query)
        trace = [{"stage": "bm25_retrieval", "status": "ok", "documents": len(bm25_scores),
                  "lexical_query": lexical_query, "seed_limit": 50,
                  "source": "manifest_hash_verified_original_recipe_corpus"}]
        parameters = {"owner": NAMESPACE, "text_owner": TEXT_NAMESPACE, "build_id": build["build_id"],
                      "dataset_revision": REVISION, "embedding_model": model["model"],
                      "embedding_revision": model["revision"], "index_signature": manifest["signature"],
                      "ingredients": include, "excluded": exclude, "dimensions": DIMENSION}
        try:
            from neo4j import RoutingControl
            records, _, _ = driver.execute_query(GRAPH_HYBRID_CANDIDATES_QUERY, parameters,
                database_=database, routing_=RoutingControl.READ)
        except Exception:
            raise GraphRAGUnavailable("hybrid_graph_vector_read_unavailable") from None
        envelopes = [record.data() for record in records]
        if len(envelopes) != 1:
            raise CatalogError("三路混合检索缺少唯一候选覆盖记录。")
        envelope = envelopes[0]
        total, scored, rows = (envelope.get(key) for key in ("graph_candidate_count", "scored_recipe_count", "matches"))
        if (type(total) is not int or total < 0 or type(scored) is not int or scored < 0
                or total != scored or not isinstance(rows, list) or len(rows) != total
                or any(not isinstance(row, dict) for row in rows)):
            raise CatalogError("三路混合检索图条件候选向量覆盖不完整。")
        trace.append({"stage": "graph_condition_retrieval", "status": "ok", "candidate_count": total,
                      "exhaustive": True, "candidate_limit_applied": False})
        vectors, qualified, prehits = {}, set(), {}
        for row in rows:
            self._hybrid_identity(row, parameters, corpus)
            rid = row["recipe_id"]
            if rid in vectors:
                raise CatalogError("三路混合检索图条件候选未按食谱去重。")
            names = row.get("ingredient_names")
            if (not isinstance(names, list) or any(not isinstance(item, dict)
                    or not isinstance(item.get("name"), str) or not item["name"]
                    or (item.get("zh_name") is not None and not isinstance(item["zh_name"], str)) for item in names)):
                raise CatalogError("三路混合检索候选食材提及核对字段无效。")
            try:
                vectors[rid] = _vector(row.get("embedding"))
            except CatalogRequestError:
                raise CatalogError("三路混合检索图条件候选的原文本向量无效。") from None
            prehits[rid] = {key: value for key, value in row.items() if key not in {"embedding", "ingredient_names"}}
            if _eligible({"ingredient_mentions": names}, include, exclude):
                qualified.add(rid)
        if qualified != set(vectors):
            # Fixed Cypher and hard recheck share the same mandatory/excluded
            # predicates. Mismatch is stale/invalid data, never a quiet refill.
            raise CatalogError("三路混合检索图谱条件与候选食材提及核对不一致。")
        if not include and not exclude and len(vectors) != len(corpus.documents):
            raise CatalogError("三路混合检索无条件候选没有覆盖完整原食谱语料。")
        fused = score_graph_first_pool(bm25_scores, vectors, vector, qualified)
        trace += [{"stage": "graph_prefiltered_e5", "status": "ok", "scored_recipe_count": scored,
                   "dense_seed_count": len(fused["seed_ids"]["dense"]), "score_domain": "complete_graph_qualified_pool",
                   "full_corpus_topN_before_filter": False},
                  {"stage": "candidate_condition_check", "status": "ok", "seed_counts": fused["counts"],
                   "final_pool_count": len(fused["candidate_pool"]), "all_final_candidates_have_dense_scores": True},
                  {"stage": "rrf_fusion", "status": "ok", "rrf_k": 60,
                   "ranked_routes": ["bm25_positive_scores", "e5_exact_cosine"], "graph_has_rank": False}]
        top = fused["rankings"][:candidate_limit]
        recipe_ids = [row["recipe_id"] for row in top]
        expansion = []
        if recipe_ids:
            try:
                records, _, _ = driver.execute_query(HYBRID_EVIDENCE_QUERY,
                    {**parameters, "recipe_ids": recipe_ids}, database_=database, routing_=RoutingControl.READ)
            except Exception:
                raise GraphRAGUnavailable("hybrid_graph_evidence_read_unavailable") from None
            expansion = [record.data() for record in records]
        by_id = {}
        for row in expansion:
            self._hybrid_identity(row, parameters, corpus)
            rid = row["recipe_id"]
            if rid not in recipe_ids or rid in by_id:
                raise CatalogError("三路混合检索一阶证据扩展越界或重复。")
            if (any(row.get(key) != prehits[rid].get(key) for key in prehits[rid])
                    or any(not isinstance(row.get(key), list) or any(not isinstance(value, str) or not value for value in row[key])
                           or len(row[key]) != len(set(row[key])) for key in ("step_ids", "source_ids", "ingredient_ids"))):
                raise CatalogError("三路混合检索一阶证据扩展身份或来源字段不一致。")
            by_id[rid] = row
        if set(by_id) != set(recipe_ids):
            raise CatalogError("三路混合检索 RRF 候选一阶证据覆盖不完整。")
        score_rows = {row["recipe_id"]: row for row in top}
        hits = [{**by_id[rid], "score": score_rows[rid]["cosine_score"]} for rid in recipe_ids]
        trace.append({"stage": "graph_evidence_expansion", "status": "ok", "expanded_recipe_ids": recipe_ids,
                      "expanded_recipe_count": len(recipe_ids), "after_rrf": True, "one_hop_only": True})
        return hits, manifest, {
            "graph_candidate_count": total, "scored_recipe_count": scored,
            "returned_after_graph_filter": len(fused["seed_ids"]["dense"]),
            "hybrid_rankings": top, "hybrid_candidate_pool": fused["candidate_pool"],
            "hybrid_seed_ids": fused["seed_ids"], "hybrid_counts": fused["counts"],
            "bm25_metadata": corpus.index.metadata, "bm25_corpus_sha256": corpus.sha256,
            "lexical_query": lexical_query, "trace": trace,
        }

    def search(self, query: str, ingredients=None, excluded_ingredients=None, limit: int = 6,
               mode: str = "semantic", query_vector=None, candidate_limit: int = 50,
               keyword_results=None, *, embedding_model=None, embedding_revision=None,
               infer_query_filters: bool = True):
        if (not isinstance(query, str) or len(query) > 500 or not isinstance(mode, str)
                or mode not in {"keyword", "semantic", "hybrid"}):
            raise CatalogRequestError("query 至多 500 字符；mode 必须为 keyword、semantic 或 hybrid。")
        if type(infer_query_filters) is not bool or (not infer_query_filters and mode == "keyword"):
            raise CatalogRequestError("query 食材推断开关必须为布尔值；semantic/hybrid 支持结构化条件独立筛选。")
        if type(limit) is not int or not 1 <= limit <= 20 or type(candidate_limit) is not int or not limit <= candidate_limit <= 200:
            raise CatalogRequestError("limit 为 1～20，candidate_limit 须在 limit～200 范围。")
        if query_vector is not None:
            _vector(query_vector)
            if (embedding_model != EMBEDDING_MODEL or not isinstance(embedding_revision, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", embedding_revision)):
                raise CatalogRequestError("外部 query_vector 必须声明文本 embedding_model 与固定 embedding_revision。")
        # Chat routing has already separated mandatory/excluded ingredients
        # from optional mentions. Re-parsing prose would turn optional food
        # or an unrecognized negative phrase into an unintended hard filter.
        parsed = parse_query_filters(query) if infer_query_filters else {
            "positive_query": query.strip(), "positive": [], "excluded": []}
        include = _filters(ingredients, "ingredients") + _filters(parsed["positive"], "ingredients")
        exclude = _filters(excluded_ingredients, "excluded_ingredients") + _filters(parsed["excluded"], "excluded_ingredients")
        keyword, fallback = None, None
        # Explicit keyword keeps the original catalog interface. Hybrid has
        # its own verified BM25 / graph-qualified E5 / graph-set RRF pipeline.
        if mode == "keyword":
            keyword = keyword_results if keyword_results is not None else self.catalog.search(
                query, ingredients=ingredients, excluded_ingredients=excluded_ingredients, limit=20)
        build = self.catalog.status()["build"]
        semantic, manifest = [], None
        coverage = {"graph_candidate_count": None, "scored_recipe_count": None,
                    "returned_after_graph_filter": 0}
        if mode in {"semantic", "hybrid"}:
            try:
                retrieve = self._hybrid if mode == "hybrid" else self._semantic
                semantic, manifest, coverage = retrieve(query, query_vector, build, include, exclude,
                                                        candidate_limit, embedding_model, embedding_revision)
            except GraphRAGUnavailable as error:
                if mode == "semantic":
                    raise
                fallback = error.reason
                try:
                    keyword = keyword_results if keyword_results is not None else self.catalog.search(
                        query, ingredients=list(dict.fromkeys(t["raw"] for t in include)),
                        excluded_ingredients=list(dict.fromkeys(t["raw"] for t in exclude)), limit=20)
                except CatalogRequestError:
                    raise GraphRAGUnavailable("keyword_and_semantic_unavailable") from None
        if keyword is not None:
            if (not isinstance(keyword, dict) or keyword.get("backend") != "neo4j"
                    or not _same_build(keyword.get("build"), build) or not isinstance(keyword.get("recipes"), list)):
                raise CatalogError("关键词候选不是同一已验证真实图谱的目录结果。")
        keyword_ids = list(dict.fromkeys(row["id"] for row in keyword["recipes"])) if keyword else []
        semantic_ids = [row["recipe_id"] for row in semantic]
        candidate_ids = semantic_ids if manifest else keyword_ids
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
                steps = sorted(detail["steps"], key=lambda step: (step["order"], step["id"]))
                source_text = detail["title"] + "\n" + "\n".join(step["text"] for step in steps)
                if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != hit["text_sha256"]:
                    raise CatalogError("GraphRAG 文本向量哈希与完整原菜谱文本不一致。")
            if _eligible(detail, include, exclude):
                details[rid] = detail
            elif hit is not None:
                # Cypher 已在打分前保证条件；图谱与详情不一致时不能悄悄过滤，
                # 否则“全部合法候选已评分”的公开覆盖声明会失真。
                raise CatalogError("GraphRAG 图谱先筛条件与原菜谱食材提及不一致。")
            else:
                filtered += 1
        hybrid = bool(manifest) and mode == "hybrid"
        effective_mode = "hybrid" if hybrid else "semantic" if manifest else "keyword"
        if hybrid:
            rankings = [{**row, "keyword_rank": row["bm25_rank"], "semantic_rank": row["dense_rank"],
                         "semantic_score": row["cosine_score"]}
                        for row in coverage["hybrid_rankings"] if row["recipe_id"] in details][:limit]
        else:
            rankings = [{"recipe_id": rid, "keyword_rank": rank if not manifest else None,
                         "semantic_rank": rank if manifest else None, "rrf_score": None,
                         "semantic_score": hits[rid]["score"] if manifest else None}
                        for rank, rid in enumerate((rid for rid in candidate_ids if rid in details), 1)][:limit]
        selected = {row["recipe_id"] for row in rankings}
        applied = {"ingredients": list(dict.fromkeys(normalize_ingredient(t["raw"]) for t in include)),
                   "excluded_ingredients": list(dict.fromkeys(normalize_ingredient(t["raw"]) for t in exclude)),
                   "positive_query": parsed["positive_query"], "positive_mentions": parsed["positive"],
                   "query_filter_inference": infer_query_filters,
                   "semantics": "dictionary_text_mentions_only", "all_requested_mentions_required": True,
                   "inventory_sufficient": False, "allergy_safe": False}
        return {"query": query.strip(), "count": len(rankings),
                "recipes": [details[row["recipe_id"]] for row in rankings], "build": build,
                "backend": "neo4j", "read_only": True, "filters": applied, "applied_filters": applied,
                "limitations": ["三路候选经条件复核后，以 BM25/E5 双排名 RRF 融合；图谱无第三排名。" if hybrid else
                                "食材条件先筛选同一已验证图谱的全部候选，E5 精确排序不保证语义相关性。",
                                "食材为规则提及，不能证明完整配方、库存充足或忌口安全。"],
                "trace": coverage.get("trace", []),
                "retrieval": {"status": "fallback" if fallback else "ok", "mode": mode,
                    "effective_mode": effective_mode, "fallback_reason": fallback,
                    "fallback_message": "语义检索未就绪，已使用关键词结果。" if fallback else None,
                    "method": HYBRID_METHOD if hybrid else "graph_first_e5_exact_cosine" if manifest else "keyword",
                    "query_filter_inference": infer_query_filters,
                    "rrf_k": 60 if hybrid else None,
                    "retriever": "neo4j.FixedGraphConditions+OriginalCorpusBM25+ExactE5+TwoRankRRF" if hybrid else
                                 "neo4j.FixedCypherGraphFirstExactE5" if manifest else None,
                    "index": manifest["index"]["name"] if manifest else None,
                    "index_used_for_ann": False, "approximate_search": False,
                    "score_semantics": "original_L2_float32_cosine_minus1_1" if hybrid else
                                       "neo4j_cosine_similarity_0_1" if manifest else None,
                    "embedding": manifest["embedding"] if manifest else None,
                    "keyword": {"available": True, "returned_candidates": len(coverage["hybrid_seed_ids"]["bm25"]) if hybrid else len(keyword_ids),
                                **({"implementation": "BM25", "metadata": coverage["bm25_metadata"],
                                    "corpus_sha256": coverage["bm25_corpus_sha256"], "lexical_query": coverage["lexical_query"]} if hybrid else {})},
                    "semantic": {"available": bool(manifest), **{key: value for key, value in coverage.items()
                                if key in {"graph_candidate_count", "scored_recipe_count", "returned_after_graph_filter"}}},
                    **({"seed_ids": coverage["hybrid_seed_ids"], "candidate_pool": coverage["hybrid_candidate_pool"],
                        "candidate_counts": coverage["hybrid_counts"], "candidate_rankings": coverage["hybrid_rankings"],
                        "graph_has_rank": False, "fusion": "whole_legal_pool_BM25_positive_rank_plus_E5_rank_RRF",
                        "semantic_scoring_domain": "complete_graph_qualified_recipe_pool",
                        "all_final_candidates_have_dense_scores": True} if hybrid else {}),
                    "graph_candidate_count": coverage["graph_candidate_count"],
                    "scored_recipe_count": coverage["scored_recipe_count"],
                    "graph_constraint_scan_exhaustive": bool(manifest),
                    "candidate_limit": candidate_limit, "returned_candidates": len(candidate_ids),
                    "eligible_candidates": len(details), "filtered_candidates": filtered,
                    "coverage": "all_graph_eligible_recipes_scored" if manifest else "bounded_candidates",
                    "insufficient_candidates": len(rankings) < limit,
                    "no_match_scope": "verified_graph_ingredient_mentions" if manifest else "retrieved_candidates_only",
                    "exhaustive": False, "rankings": rankings, "llm_called": False},
                "graph_evidence": [{**hits[rid], "semantics": "graph_eligible_two_rank_rrf_selected" if hybrid else "graph_eligible_exact_e5_ranked"}
                                   for rid in semantic_ids if rid in selected]}
