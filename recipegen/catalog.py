"""Real, read-only RecipeGen catalog; no demo fallback or model invocation."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from scripts.query_test_graph import Connection, QueryError, load_connection, validate_result
from .config import PROJECT_ROOT
from .kg_extract import DEFAULT_LEXICON

NAMESPACE = "recipegen-test-v1"
MM_NAMESPACE = "recipegen-mm-v1"
REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
DATASET_ID = "RUOXUAN123/RecipeGen"
ARCHIVES = ("test.zip", "test-video.zip")
LIMITATIONS = [
    "食材是原步骤中的词典规则提及，原料列表不完整；筛选不证明库存充足、忌口或过敏安全。",
    "总烹饪时长未知；视频时长和步骤时间提及不等于总烹饪时长。",
    "步骤保留 steps.txt 原文顺序，媒体清单表示归档目录关联。",
    "图像描述和步骤对齐是已有模型候选；最新运行不代表更高语义准确率。",
]


class CatalogError(ValueError):
    """Configuration, connectivity or verified native evidence is unavailable."""


class CatalogRequestError(ValueError):
    """User query/filters are invalid; this does not imply database failure."""


class RecipeNotFound(CatalogError):
    """The requested ID is absent from the active verified test build."""


ACTIVE_BUILD_QUERY = """// recipegen-catalog:active-build
MATCH (b:RecipeGenBuild {split:'test', active:true, status:'verified'})
WHERE b.kg_import_namespace = $owner
RETURN b.build_id AS build_id, b.dataset_revision AS dataset_revision,
       b.split AS split, b.active AS active, b.status AS status,
       b.content_sha256 AS content_sha256
ORDER BY b.verified_at DESC, b.build_id
LIMIT 2
"""

STATUS_QUERY = """// recipegen-catalog:status
CALL () {
 MATCH (n:RecipeGen {build_id:$build_id, split:'test', dataset_revision:$dataset_revision})
 WHERE n.kg_import_namespace = $owner
 RETURN count(CASE WHEN n:Recipe THEN n END) AS recipes,
   count(CASE WHEN n:Step THEN n END) AS steps,
   count(CASE WHEN n:Ingredient THEN n END) AS ingredient_entities,
   count(CASE WHEN n:Image THEN n END) AS images,
   count(CASE WHEN n:Video THEN n END) AS videos
}
CALL () {
 MATCH (image:RecipeGen:Image)-[h:HAS_OBSERVATION]->(o:RecipeGen:VisualObservation)
 WHERE image.kg_import_namespace = $owner AND image.build_id = $build_id
   AND image.split = 'test' AND image.dataset_revision = $dataset_revision
   AND o.kg_import_namespace = $mm_owner AND o.build_id = $build_id
   AND o.split = 'test' AND o.dataset_revision = $dataset_revision
   AND o.source_media_id = image.kg_csv_id AND o.source_id = image.source_id
   AND o.verified = false AND o.semantics = 'model_candidate'
   AND h.kg_import_namespace = $mm_owner AND h.run_id = o.run_id
   AND h.build_id = $build_id AND h.split = 'test' AND h.dataset_revision = $dataset_revision
 RETURN count(DISTINCT o) AS image_observations,
   count(DISTINCT image) AS distinct_recognized_images
}
RETURN recipes, steps, ingredient_entities, images, videos,
 image_observations, distinct_recognized_images
"""

# Each branch aggregates independently before the next branch, avoiding inflated
# Step/Ingredient/media counts. All scope checks apply to both nodes and edges.
DETAIL_BODY = """
CALL (r) {
 MATCH (r)-[h:HAS_STEP]->(s:RecipeGen:Step)
 WHERE s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.split = 'test' AND s.dataset_revision = $dataset_revision
   AND h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.split = 'test' AND h.dataset_revision = $dataset_revision
 WITH DISTINCT s ORDER BY s.order, s.kg_csv_id
 RETURN collect({id:s.kg_csv_id,order:s.order,text:s.text,source_id:s.source_id}) AS steps
}
CALL (r) {
 MATCH (r)-[h:HAS_SOURCE]->(s:RecipeGen:Source)
 WHERE s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.split = 'test' AND s.dataset_revision = $dataset_revision
   AND h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.split = 'test' AND h.dataset_revision = $dataset_revision
 WITH DISTINCT s,h.role AS role ORDER BY role,s.kg_csv_id
 RETURN collect({id:s.kg_csv_id,source_id:s.source_id,role:role,
   source_type:s.source_type,dataset:s.dataset,split:s.split,build_id:s.build_id,
   dataset_revision:s.dataset_revision,archive:s.archive,member:s.member,
   artifact_path:s.artifact_path,url:s.url,sha256:s.sha256}) AS sources
}
CALL (r) {
 MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
 WHERE i.kg_import_namespace = $owner AND i.build_id = $build_id
   AND i.split = 'test' AND i.dataset_revision = $dataset_revision
   AND h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.split = 'test' AND h.dataset_revision = $dataset_revision
 WITH DISTINCT i,h ORDER BY i.normalized_name,i.kg_csv_id
 RETURN collect({id:i.kg_csv_id,normalized_name:i.normalized_name,zh_name:i.zh_name,
   extraction_method:h.extraction_method,semantics:h.semantics,verified:h.verified,
   complete_ingredient_list:h.complete_ingredient_list,evidence_json:h.evidence_json}) AS ingredient_mentions
}
CALL (r) {
 MATCH (r)-[h:HAS_IMAGE|HAS_VIDEO]->(m:RecipeGen)
 WHERE m.kg_import_namespace = $owner AND m.build_id = $build_id
   AND m.split = 'test' AND m.dataset_revision = $dataset_revision AND (m:Image OR m:Video)
   AND h.kg_import_namespace = $owner AND h.build_id = $build_id
   AND h.split = 'test' AND h.dataset_revision = $dataset_revision
 WITH DISTINCT m
 OPTIONAL MATCH (m)-[ms:HAS_SOURCE]->(s:RecipeGen:Source)
 WHERE s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.split = 'test' AND s.dataset_revision = $dataset_revision
   AND ms.kg_import_namespace = $owner AND ms.build_id = $build_id
   AND ms.split = 'test' AND ms.dataset_revision = $dataset_revision
 WITH m,s ORDER BY m.kg_csv_id
 RETURN count(CASE WHEN m:Image THEN m END) AS images,
   count(CASE WHEN m:Video THEN m END) AS videos,
   count(CASE WHEN m:Image AND m.downloaded = true THEN m END) AS downloaded_images,
   count(CASE WHEN m:Video AND m.downloaded = true THEN m END) AS downloaded_videos,
   count(CASE WHEN m.recognition_status = 'not_run' THEN m END) AS recognition_not_run,
   collect(DISTINCT m.storage) AS storage_modes,
   collect({id:m.kg_csv_id,kind:CASE WHEN m:Image THEN 'image' ELSE 'video' END,
     source_id:m.source_id,archive:m.archive,member:m.member,display_path:m.display_path,
     size_bytes:m.size_bytes,crc32:m.crc32,storage:m.storage,
     source:{id:s.kg_csv_id,source_id:s.source_id,source_type:s.source_type,
       dataset:s.dataset,split:s.split,build_id:s.build_id,dataset_revision:s.dataset_revision,
       archive:s.archive,member:s.member,artifact_path:s.artifact_path,url:s.url,sha256:s.sha256}}) AS inventory
}
CALL (r) {
 MATCH (r)-[ri:HAS_IMAGE]->(image:RecipeGen:Image)
 WHERE image.kg_import_namespace = $owner AND image.build_id = $build_id
   AND image.split = 'test' AND image.dataset_revision = $dataset_revision
   AND ri.kg_import_namespace = $owner AND ri.build_id = $build_id
   AND ri.split = 'test' AND ri.dataset_revision = $dataset_revision
 WITH DISTINCT r,image
 CALL (image) {
   OPTIONAL MATCH (image)-[h:HAS_OBSERVATION]->(o:RecipeGen:VisualObservation)
   WHERE o.kg_import_namespace = $mm_owner AND o.build_id = $build_id
     AND o.split = 'test' AND o.dataset_revision = $dataset_revision
     AND o.source_media_id = image.kg_csv_id AND o.source_id = image.source_id
     AND o.verified = false AND o.semantics = 'model_candidate'
     AND h.kg_import_namespace = $mm_owner AND h.run_id = o.run_id
     AND h.build_id = $build_id AND h.split = 'test' AND h.dataset_revision = $dataset_revision
   WITH o,head([p IN $run_priorities WHERE p.run_id = o.run_id | p.finished_at]) AS finished_at
   ORDER BY finished_at IS NULL ASC,finished_at DESC,o.kg_csv_id DESC
   LIMIT 1
   RETURN o,finished_at
 }
 OPTIONAL MATCH (image)-[alignment:ALIGNED_WITH]->(s:RecipeGen:Step)
 WHERE o IS NOT NULL AND alignment.kg_import_namespace = $mm_owner
   AND alignment.run_id = o.run_id AND alignment.build_id = $build_id
   AND alignment.split = 'test' AND alignment.dataset_revision = $dataset_revision
   AND alignment.alignment_method = 'clip_cosine_temporal_candidate'
   AND s.kg_import_namespace = $owner AND s.build_id = $build_id
   AND s.split = 'test' AND s.dataset_revision = $dataset_revision
   AND EXISTS { MATCH (r)-[hs:HAS_STEP]->(s)
     WHERE hs.kg_import_namespace = $owner AND hs.build_id = $build_id
       AND hs.split = 'test' AND hs.dataset_revision = $dataset_revision }
 WITH o,finished_at,s,alignment ORDER BY o.source_media_id,o.kg_csv_id,s.order
 RETURN collect(CASE WHEN o IS NULL THEN null ELSE
   {observation_id:o.kg_csv_id,caption:o.caption,source_media_id:o.source_media_id,
    source_id:o.source_id,run_id:o.run_id,finished_at:finished_at,
    step_id:s.kg_csv_id,step_order:s.order,score:alignment.similarity_score,
    evidence_json:o.evidence_json,raw_json:o.raw_json,
    build_id:o.build_id,dataset_revision:o.dataset_revision,split:o.split} END) AS visual_evidence
}
RETURN {id:r.kg_csv_id,recipe_id:r.recipe_id,title:r.title,origin_archive:r.origin_archive,
  steps_count:r.steps_count,text_complete:r.text_complete,quality_flags:r.quality_flags,
  is_demo:r.is_demo,build_id:r.build_id,dataset_revision:r.dataset_revision,split:r.split} AS recipe,
 steps,sources,ingredient_mentions,
 {images:images,videos:videos,downloaded_images:downloaded_images,downloaded_videos:downloaded_videos,
  recognition_not_run:recognition_not_run,storage_modes:storage_modes,inventory:inventory} AS media,
 visual_evidence
"""

DETAIL_QUERY = """// recipegen-catalog:detail
MATCH (r:RecipeGen:Recipe {build_id:$build_id, split:'test', dataset_revision:$dataset_revision})
WHERE r.kg_import_namespace = $owner AND (r.kg_csv_id = $recipe_id OR r.recipe_id = $recipe_id)
WITH r
""" + DETAIL_BODY

SEARCH_QUERY = """// recipegen-catalog:search
MATCH (r:RecipeGen:Recipe {build_id:$build_id, split:'test', dataset_revision:$dataset_revision})
WHERE r.kg_import_namespace = $owner
  AND all(term IN $terms WHERE
    any(name IN term.names WHERE toLower(coalesce(r.title,'')) CONTAINS name)
    OR EXISTS { MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
      WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
        AND h.split = 'test' AND h.dataset_revision = $dataset_revision
        AND i.kg_import_namespace = $owner AND i.build_id = $build_id
        AND i.split = 'test' AND i.dataset_revision = $dataset_revision
        AND (toLower(coalesce(i.normalized_name,'')) IN term.names OR toLower(coalesce(i.zh_name,'')) IN term.names) })
  AND all(term IN $ingredients WHERE EXISTS {
    MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
    WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
      AND h.split = 'test' AND h.dataset_revision = $dataset_revision
      AND i.kg_import_namespace = $owner AND i.build_id = $build_id
      AND i.split = 'test' AND i.dataset_revision = $dataset_revision
      AND (toLower(coalesce(i.normalized_name,'')) IN term.names OR toLower(coalesce(i.zh_name,'')) IN term.names) })
  AND none(term IN $excluded WHERE EXISTS {
    MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
    WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
      AND h.split = 'test' AND h.dataset_revision = $dataset_revision
      AND i.kg_import_namespace = $owner AND i.build_id = $build_id
      AND i.split = 'test' AND i.dataset_revision = $dataset_revision
      AND (toLower(coalesce(i.normalized_name,'')) IN term.names OR toLower(coalesce(i.zh_name,'')) IN term.names) })
WITH r,toLower(trim(coalesce(r.title,''))) AS normalized_title
ORDER BY CASE
  WHEN $title_query <> '' AND normalized_title = $title_query THEN 0
  WHEN $title_query <> '' AND normalized_title CONTAINS $title_query THEN 1
  ELSE 2 END,
  toLower(coalesce(r.title,'')),r.kg_csv_id
LIMIT $limit
""" + DETAIL_BODY

GRAPH_NODES_QUERY = """// recipegen-catalog:graph-nodes
MATCH (n:RecipeGen {build_id:$build_id,split:'test',dataset_revision:$dataset_revision})
WHERE n.kg_csv_id IN $node_ids AND n.kg_import_namespace IN [$owner,$mm_owner]
RETURN n.kg_csv_id AS id,
 CASE WHEN n:Recipe THEN 'Recipe' WHEN n:Step THEN 'Step' WHEN n:Ingredient THEN 'Ingredient'
 WHEN n:Source THEN 'Source' WHEN n:Image THEN 'Image' WHEN n:Video THEN 'Video'
 WHEN n:VisualObservation THEN 'VisualObservation' ELSE 'Unknown' END AS label,
 coalesce(n.title,n.text,n.normalized_name,n.caption,n.member,n.kg_csv_id) AS name
ORDER BY label,id
"""
GRAPH_EDGES_QUERY = """// recipegen-catalog:graph-edges
MATCH (a:RecipeGen)-[h]->(b:RecipeGen)
WHERE a.kg_csv_id IN $node_ids AND b.kg_csv_id IN $node_ids
  AND a.build_id = $build_id AND b.build_id = $build_id
  AND a.split = 'test' AND b.split = 'test'
  AND a.dataset_revision = $dataset_revision AND b.dataset_revision = $dataset_revision
  AND a.kg_import_namespace IN [$owner,$mm_owner] AND b.kg_import_namespace IN [$owner,$mm_owner]
  AND h.kg_import_namespace IN [$owner,$mm_owner] AND h.build_id = $build_id
  AND h.split = 'test' AND h.dataset_revision = $dataset_revision
  AND ((h.kg_import_namespace = $owner AND a.kg_import_namespace = $owner AND b.kg_import_namespace = $owner)
    OR (h.kg_import_namespace = $mm_owner AND h.run_id IN $visual_run_ids
      AND any(selected IN $visual_selection WHERE selected.media_id = h.source_media_id AND selected.run_id = h.run_id)
      AND (a.kg_import_namespace = $owner OR a.run_id = h.run_id)
      AND (b.kg_import_namespace = $owner OR b.run_id = h.run_id)))
RETURN DISTINCT a.kg_csv_id AS source,b.kg_csv_id AS target,type(h) AS type,
 h.kg_import_namespace = $mm_owner AS candidate
ORDER BY source,type,target
"""


def _key(value: str) -> str:
    return re.sub(r"[\s\-‐‑–—]+", " ",value.lower()).strip()


_ALIASES = { _key(alias):entry.normalized_name for entry in DEFAULT_LEXICON.ingredients
             for alias in (*entry.aliases,entry.normalized_name,entry.zh_name or "") if alias }
_ALIASES.update({"西红柿":"tomato","马铃薯":"potato","青瓜":"cucumber"})
_ALIAS_PATTERNS = tuple((re.compile((r"(?<![a-z0-9])"+re.escape(alias)+r"(?![a-z0-9])")
                                    if re.search(r"[a-z0-9]",alias) else re.escape(alias),re.I),canonical)
                        for alias,canonical in _ALIASES.items())


def _term(value: str) -> dict[str, Any]:
    raw = _key(value)
    return {"raw":value,"names":list(dict.fromkeys([raw,_ALIASES.get(raw,raw)]))}


def normalize_ingredient(name: str) -> str:
    if not isinstance(name,str) or not name.strip() or len(name)>80:
        raise CatalogRequestError("食材名称必须为 1～80 字符的非空字符串。")
    normalized = _key(name)
    return _ALIASES.get(normalized,normalized)


def _recognized(text: str) -> list[str]:
    matches = []
    for pattern,canonical in _ALIAS_PATTERNS:
        matches.extend((match.start(),match.end(),canonical) for match in pattern.finditer(text))
    # Prefer the longest known alias, e.g. 'olive oil' over 'oil'. Chinese
    # contiguous ingredient names work without introducing a broad '蛋' alias.
    chosen = []
    for start,end,name in sorted(matches,key=lambda row:(-(row[1]-row[0]),row[0],row[2])):
        if not any(start<other_end and end>other_start for other_start,other_end,_ in chosen):
            chosen.append((start,end,name))
    return list(dict.fromkeys(row[2] for row in sorted(chosen)))


def parse_query_filters(question: str) -> dict[str,Any]:
    """小型确定性查询规则；提及排除不构成完整配方/过敏安全判断。"""
    if not isinstance(question,str) or len(question)>500:
        raise CatalogRequestError("question 必须为至多 500 字符的字符串。")
    excluded = []
    pattern = r"(?:不要|不吃|不加|不放|排除|without\b)\s*(.*?)(?=但是|但|可以|\bbut\b|[，,；;。.!?]|$)"
    def remove(match):
        clause = match.group(1).strip()
        for part in re.split(r"\s+and\s+|\s*&\s*|、|和",clause,flags=re.I):
            part = part.strip()
            found = _recognized(part)
            if found:
                excluded.extend(found)
            elif part:
            # Unknown explicit names are visible in applied_filters rather than
            # being silently treated as a known ingredient or positive keyword.
                excluded.append(normalize_ingredient(part))
        return " "
    positive_query = re.sub(pattern,remove,question,flags=re.I).strip(" ，,；;")
    excluded = list(dict.fromkeys(excluded))
    positive = _recognized(positive_query)
    return {"positive_query":positive_query,"positive":positive,"excluded":excluded,
            "excluded_ingredients":list(excluded),"semantics":"dictionary_text_mentions_only"}


def _terms(query: str) -> list[dict[str, Any]]:
    if _key(query) in _ALIASES:
        return [_term(query)]
    recognized = _recognized(query)
    if recognized and (re.search(r"[\u4e00-\u9fff]",query)
                       or re.search(r"\b(what|can|could|with|recipe|please|make|cook|give|want)\b",query,re.I)):
        return [_term(name) for name in recognized]
    tokens = [token for token in re.split(r"[\s,，、;；]+",query.strip()) if token]
    if len(tokens) > 12:
        raise CatalogRequestError("查询最多包含 12 个关键词。")
    return [_term(token) for token in dict.fromkeys(tokens)]


def _filters(values: Any, label: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values,list) or len(values) > 20 or any(not isinstance(v,str) or not v.strip() or len(v)>80 for v in values):
        raise CatalogRequestError(f"{label} 必须为至多 20 个非空食材名称的列表，每项至多 80 字符。")
    return [_term(v.strip()) for v in dict.fromkeys(values)]


def _rows(tx, query, parameters):
    return [record.data() for record in tx.run(query,parameters)]


class RecipeCatalog:
    def __init__(self, connection: Connection | None = None, *, settings=None,
                 config_path: Path | None = None, driver=None, project_root: Path = PROJECT_ROOT):
        self._connection = connection
        self._settings = settings
        self._config_path = config_path
        self._driver = driver
        self._owns_driver = driver is None
        self._root = Path(project_root)
        self._priorities: list[dict[str,Any]] = []
        self._priorities_at = -math.inf

    def close(self):
        if self._driver is not None and self._owns_driver:
            self._driver.close()
        self._driver = None

    def _connect(self):
        if self._driver is not None:
            return self._driver
        try:
            if self._connection is None:
                settings = self._settings
                if settings is not None and settings.neo4j_password:
                    environment = {"NEO4J_"+key.upper():getattr(settings,"neo4j_"+key)
                                   for key in ("uri","user","password","database")}
                    self._connection = load_connection(environment=environment)
                else:
                    self._connection = load_connection(self._config_path) if self._config_path else load_connection()
            else:
                self._connection = load_connection(environment={"NEO4J_"+key.upper():getattr(self._connection,key)
                                                               for key in ("uri","user","password","database")})
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._connection.uri,auth=(self._connection.user,self._connection.password),
                                                 connection_timeout=10,max_transaction_retry_time=10)
            return self._driver
        except Exception:
            raise CatalogError("无法建立真实 Neo4j 只读连接；检查私密连接配置和数据库服务。") from None

    def _read(self,callback):
        try:
            from neo4j import READ_ACCESS
            driver = self._connect()
            database = self._connection.database if self._connection else getattr(self._settings,"neo4j_database","neo4j")
            with driver.session(database=database,default_access_mode=READ_ACCESS) as session:
                return session.execute_read(callback)
        except (CatalogError, CatalogRequestError):
            raise
        except Exception:
            raise CatalogError("真实 Neo4j 读取未完成；不会切换示例数据。") from None

    def _build(self,tx):
        rows = _rows(tx,ACTIVE_BUILD_QUERY,{"owner":NAMESPACE})
        if len(rows) != 1:
            raise CatalogError("没有唯一的 active+verified RecipeGen 测试构建。")
        build = rows[0]
        if (build.get("dataset_revision") != REVISION or build.get("split") != "test"
                or build.get("active") is not True or build.get("status") != "verified"
                or not isinstance(build.get("build_id"),str) or not build["build_id"]):
            raise CatalogError("活动构建不符合固定 Hugging Face 测试 revision/verified 范围。")
        return build

    def _run_priorities(self):
        if time.monotonic()-self._priorities_at < 60:
            return self._priorities
        priorities = {}
        # 公开成功检查点只提供排序时间；菜谱和视觉事实仍读取 Neo4j。
        for path in (self._root/"data/multimodal/runs").glob("*/state.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                signature = state.get("signature",{})
                if signature.get("dataset_revision") != REVISION:
                    continue
                for item in state.get("records",{}).values():
                    if item.get("status") != "verified" or not isinstance(item.get("run_id"),str):
                        continue
                    finished = datetime.fromisoformat(item["finished_at"])
                    if finished.tzinfo is None:
                        continue
                    value = finished.timestamp()
                    if math.isfinite(value):
                        priorities[item["run_id"]] = max(value,priorities.get(item["run_id"],-math.inf))
            except (OSError,ValueError,TypeError,KeyError,AttributeError):
                continue
        self._priorities = [{"run_id":key,"finished_at":value} for key,value in sorted(priorities.items())]
        self._priorities_at = time.monotonic()
        return self._priorities

    def _parameters(self,build):
        return {"owner":NAMESPACE,"mm_owner":MM_NAMESPACE,"build_id":build["build_id"],
                "dataset_revision":REVISION,"run_priorities":self._run_priorities()}

    def status(self):
        def read(tx):
            build = self._build(tx)
            rows = _rows(tx,STATUS_QUERY,self._parameters(build))
            keys = ("recipes","steps","ingredient_entities","images","videos","image_observations","distinct_recognized_images")
            if len(rows)!=1 or any(type(rows[0].get(k)) is not int or rows[0][k]<0 for k in keys):
                raise CatalogError("图谱统计缺少真实非负整数计数。")
            counts = rows[0]
            if counts["distinct_recognized_images"]>counts["images"] or counts["distinct_recognized_images"]>counts["image_observations"]:
                raise CatalogError("图像观察去重数量与实际媒体总数不一致。")
            return {"status":"ok","backend":"neo4j","read_only":True,"dataset":{
                    "name":"RecipeGen","id":DATASET_ID,"revision":REVISION,"split":"test","is_demo":False},
                    "graph":counts,"build":build,"capabilities":{"recipe_search":True,"recipe_details":True,
                    "recipe_graph":True,"visual_evidence":counts["image_observations"]>0,"inventory_guarantee":False,
                    "exclusion_safety_guarantee":False,"duration_filter":False,"llm_called":False},
                    "limitations":list(LIMITATIONS)}
        return self._read(read)

    def _detail(self,row,build):
        try:
            value = validate_result(row,build)
        except QueryError as error:
            raise CatalogError(str(error)) from None
        recipe = row["recipe"]
        if recipe.get("is_demo") is not False or any(recipe.get(k)!=build[k] for k in ("build_id","dataset_revision","split")):
            raise CatalogError("菜谱不属于当前真实测试构建。")
        for source in value["sources"]:
            if source.get("archive") not in ARCHIVES or source.get("dataset") != DATASET_ID:
                raise CatalogError("文本来源超出官方测试归档。")
        mentions = []
        for mention in value["ingredient_mentions"]:
            if (mention.get("extraction_method")!="dictionary_rule" or mention.get("verified") is not False
                    or mention.get("complete_ingredient_list") is not False or mention.get("semantics")!="text_mention_candidate"):
                raise CatalogError("食材提及来源或候选属性不符合原图谱合同。")
            try:
                evidence = json.loads(mention["evidence_json"])
            except (ValueError,TypeError,KeyError):
                raise CatalogError("食材提及缺少原始抽取证据。") from None
            if not isinstance(evidence,list):
                raise CatalogError("食材提及证据必须为列表。")
            source_ids = sorted({hit.get("source",{}).get("id") for hit in evidence
                                 if isinstance(hit,dict) and isinstance(hit.get("source"),dict)
                                 and isinstance(hit["source"].get("id"),str)})
            step_sources = {s["source_id"] for s in value["sources"] if s.get("role")=="steps"}
            if not evidence or not source_ids or not set(source_ids)<=step_sources:
                raise CatalogError("食材规则提及缺少该菜谱的原始步骤 Source。")
            mentions.append({**mention,"name":mention["normalized_name"],"evidence":evidence,
                             "source_id":source_ids[0] if len(source_ids)==1 else None,"source_ids":source_ids,
                             "candidate":True,"ingredients_complete":False})
        media = value["media"]
        inventory = media.get("inventory")
        if not isinstance(inventory,list) or len(inventory)!=media["images"]+media["videos"] or len({m.get("id") for m in inventory})!=len(inventory):
            raise CatalogError("媒体清单与原图谱数量不一致。")
        for item in inventory:
            source = item.get("source",{})
            if (item.get("kind") not in {"image","video"} or item.get("archive") not in ARCHIVES
                    or not item.get("id") or source.get("source_id") != item.get("source_id")
                    or source.get("source_type") != "huggingface_zip_media"
                    or source.get("dataset") != DATASET_ID
                    or any(source.get(k)!=build[k] for k in ("build_id","dataset_revision","split"))
                    or source.get("archive")!=item.get("archive") or source.get("member")!=item.get("member")
                    or not source.get("artifact_path") or not source.get("url")):
                raise CatalogError("媒体缺少同一构建的真实 Source/归档成员定位。")
        visual = []
        seen = set()
        images = {m["id"]:m for m in inventory if m["kind"]=="image"}
        step_ids = {s["id"]:s for s in value["steps"]}
        for obs in row.get("visual_evidence",[]):
            mid = obs.get("source_media_id")
            if (mid not in images or mid in seen or obs.get("source_id")!=images[mid]["source_id"]
                    or not obs.get("observation_id") or not isinstance(obs.get("caption"),str)
                    or any(obs.get(k)!=build[k] for k in ("build_id","dataset_revision","split"))):
                raise CatalogError("图像候选证据重复或来源/构建不一致。")
            try:
                evidence = json.loads(obs["evidence_json"])
                raw = json.loads(obs["raw_json"])
            except (ValueError,TypeError,KeyError):
                raise CatalogError("图像候选缺少原始模型输出或输入来源证据。") from None
            if (not isinstance(evidence,dict) or not isinstance(raw,dict)
                    or any(evidence.get(k)!=obs.get(k) for k in
                           ("source_media_id","source_id","run_id","build_id","dataset_revision","split"))):
                raise CatalogError("图像候选 JSON 来源与当前图谱节点不一致。")
            if obs.get("step_id") is not None and (obs["step_id"] not in step_ids or obs.get("step_order")!=step_ids[obs["step_id"]]["order"]):
                raise CatalogError("图像对齐候选不属于该菜谱的原始步骤。")
            score = obs.get("score")
            if score is not None and (type(score) not in (int,float) or not math.isfinite(score) or not -1<=score<=1):
                raise CatalogError("图像对齐候选分数不是有效 cosine 值。")
            seen.add(mid)
            visual.append({**obs,"candidate":True,"latest_unverified":obs.get("finished_at") is None,
                           "selection_method":"verified_state_finished_at" if obs.get("finished_at") is not None else "stable_id_latest_unverified"})
        return {"id":value["id"],"title":value["title"],"steps":value["steps"],
                "ingredient_mentions":mentions,"sources":value["sources"],
                "media":{**media,"recognized_images":len(visual)},"visual_evidence":visual,
                "text_complete":value.get("text_complete"),"quality_flags":value.get("quality_flags"),
                "semantics":{"ingredients_complete":False,"duration_known":False,
                             "inventory_verified":False,"exclusion_safety_verified":False},
                "build":build,"limitations":list(LIMITATIONS)}

    @staticmethod
    def _validate_id(recipe_id):
        if not isinstance(recipe_id,str) or not recipe_id.strip() or len(recipe_id)>200:
            raise CatalogRequestError("recipe_id 必须为非空原图谱 ID。")
        return recipe_id.strip()

    def _recipe_tx(self,tx,recipe_id,build):
        rows = _rows(tx,DETAIL_QUERY,{**self._parameters(build),"recipe_id":recipe_id})
        if not rows:
            raise RecipeNotFound("当前已验证 RecipeGen 测试构建中没有该菜谱。")
        if len(rows)!=1:
            raise CatalogError("原菜谱 ID 返回多个节点，需核查图谱身份。")
        return self._detail(rows[0],build)

    def recipe(self,recipe_id):
        recipe_id = self._validate_id(recipe_id)
        return self._read(lambda tx:self._recipe_tx(tx,recipe_id,self._build(tx)))

    def search(self,query: str,ingredients=None,excluded_ingredients=None,limit: int=10):
        if not isinstance(query,str) or len(query)>500:
            raise CatalogRequestError("query 必须为至多 500 字符的字符串。")
        if type(limit) is not int or not 1<=limit<=30:
            raise CatalogRequestError("limit 必须为 1～30。")
        parsed = parse_query_filters(query)
        terms,include = _terms(parsed["positive_query"]),_filters(ingredients,"ingredients")
        # 完整正向菜名只参与排序，不能放宽食材或排除条件。
        title_query = parsed["positive_query"].strip().lower()
        exclude = _filters(excluded_ingredients,"excluded_ingredients")+_filters(parsed["excluded"],"excluded_ingredients")
        def read(tx):
            build = self._build(tx)
            if query.strip().startswith("recipegen:test:"):
                detail = self._recipe_tx(tx,self._validate_id(query),build)
                names = {_key(m["name"]) for m in detail["ingredient_mentions"]} | {_key(m.get("zh_name") or "") for m in detail["ingredient_mentions"]}
                recipes = [detail] if all(set(t["names"]) & names for t in include) and not any(set(t["names"]) & names for t in exclude) else []
            else:
                rows = _rows(tx,SEARCH_QUERY,{**self._parameters(build),"terms":terms,"title_query":title_query,"ingredients":include,
                            "excluded":exclude,"limit":limit})
                if len(rows)>limit:
                    raise CatalogError("数据库搜索返回数量超过 limit。")
                recipes = [self._detail(row,build) for row in rows]
                if len({r["id"] for r in recipes})!=len(recipes):
                    raise CatalogError("搜索结果有重复 Recipe ID。")
            applied = {"ingredients":[normalize_ingredient(t["raw"]) for t in include],
                       "excluded_ingredients":list(dict.fromkeys(normalize_ingredient(t["raw"]) for t in exclude)),
                       "positive_query":parsed["positive_query"],"positive_mentions":parsed["positive"],
                       "semantics":"dictionary_text_mentions_only","all_requested_mentions_required":True,
                       "inventory_sufficient":False,"allergy_safe":False}
            return {"query":query.strip(),"count":len(recipes),"recipes":recipes,"build":build,
                    "backend":"neo4j","read_only":True,"filters":applied,"applied_filters":applied,
                    "limitations":list(LIMITATIONS)}
        return self._read(read)

    def graph(self,recipe_id):
        recipe_id = self._validate_id(recipe_id)
        def read(tx):
            build = self._build(tx)
            detail = self._recipe_tx(tx,recipe_id,build)
            node_ids = {detail["id"]} | {r["id"] for r in detail["steps"]+detail["ingredient_mentions"]}
            node_ids.update(s["source_id"] for s in detail["sources"])
            node_ids.update(m["id"] for m in detail["media"]["inventory"])
            node_ids.update(m["source_id"] for m in detail["media"]["inventory"])
            node_ids.update(v["observation_id"] for v in detail["visual_evidence"])
            parameters = {**self._parameters(build),"node_ids":sorted(node_ids),
                          "visual_run_ids":sorted({v["run_id"] for v in detail["visual_evidence"]}),
                          "visual_selection":[{"media_id":v["source_media_id"],"run_id":v["run_id"]}
                                              for v in detail["visual_evidence"]]}
            nodes = _rows(tx,GRAPH_NODES_QUERY,parameters)
            if {n.get("id") for n in nodes}!=node_ids or len(nodes)!=len(node_ids) or any(n.get("label")=="Unknown" for n in nodes):
                raise CatalogError("菜谱子图节点缺失、重复或标签超出范围。")
            edges = _rows(tx,GRAPH_EDGES_QUERY,parameters)
            if any(e.get("source") not in node_ids or e.get("target") not in node_ids or not isinstance(e.get("type"),str) for e in edges):
                raise CatalogError("菜谱子图关系端点不属于已验证节点。")
            return {"recipe_id":detail["id"],"title":detail["title"],"nodes":nodes,"edges":edges,
                    "build":build,"backend":"neo4j","read_only":True,"limitations":list(LIMITATIONS)}
        return self._read(read)
