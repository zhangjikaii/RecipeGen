"""Read-only CLIP candidate retrieval with original graph/source evidence.

Visual captions and ALIGNED_WITH links are model hypotheses. This module never
writes Neo4j, translates original steps, or substitutes model prose for facts.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import time
from typing import Any

BASE_NAMESPACE = "recipegen-test-v1"
NAMESPACE = "recipegen-mm-v1"
VECTOR_INDEX = "recipegen_visual_embedding"
ARCHIVES = ("test.zip", "test-video.zip")
LIMITATIONS = [
    "结果是 CLIP 近似向量候选；固定 candidate_limit 后再过滤，可能少于 limit，不能视为穷尽匹配。",
    "caption、视觉对象、动作与 ALIGNED_WITH 是模型候选，未经人工语义准确率评测，不证明精确动作、时间对应或安全。",
    "步骤文字直接来自当前已验证测试构建；可选本地模型只选择已有 observation_id，不修改原始事实。",
    "视频时间窗相对首个实际解码帧，时间来源保留原 PTS；无步骤对齐或无菜谱的媒体保留空值。",
]


class MultimodalQueryError(ValueError):
    """A read-only request or returned evidence violates the query contract."""


ACTIVE_BUILD_QUERY = """// recipegen-mm:read-active-build
MATCH (b:RecipeGenBuild {split: 'test', active: true, status: 'verified'})
WHERE b.kg_import_namespace = $base_owner
RETURN b.build_id AS build_id, b.split AS split, b.status AS status,
       b.active AS active, b.dataset_revision AS dataset_revision,
       b.content_sha256 AS content_sha256
ORDER BY b.verified_at DESC, b.build_id
LIMIT 2
"""

# 索引名固定；输入向量/范围/limit 都通过参数传入。可选步骤和菜谱保留孤媒体。
SEARCH_QUERY = """// recipegen-mm:read-vector-candidates
CALL db.index.vector.queryNodes('recipegen_visual_embedding', $candidate_limit, $embedding)
YIELD node AS observation, score
WHERE observation:RecipeGen AND observation:VisualObservation
  AND observation.kg_import_namespace = $owner
  AND observation.build_id = $build_id AND observation.split = 'test'
  AND observation.dataset_revision = $dataset_revision
  AND observation.embedding_model = $embedding_model
  AND observation.embedding_revision = $embedding_revision
MATCH (parent:RecipeGen)-[h:HAS_OBSERVATION]->(observation)
WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
  AND h.split = 'test' AND h.dataset_revision = $dataset_revision
  AND h.run_id = observation.run_id
  AND ((parent:Image AND parent.kg_import_namespace = $base_owner
        AND parent.kg_csv_id = observation.source_media_id)
    OR (parent:VideoClip AND parent.kg_import_namespace = $owner
        AND parent.run_id = observation.run_id
        AND parent.source_media_id = observation.source_media_id))
  AND parent.build_id = $build_id AND parent.split = 'test'
  AND parent.dataset_revision = $dataset_revision
MATCH (media:RecipeGen {kg_csv_id: observation.source_media_id,
                       build_id: $build_id, split: 'test'})-[ms:HAS_SOURCE]->
      (source:RecipeGen:Source {kg_csv_id: observation.source_id,
                               build_id: $build_id, split: 'test'})
WHERE media.kg_import_namespace = $base_owner
  AND media.dataset_revision = $dataset_revision
  AND media.source_id = observation.source_id
  AND ((parent:Image AND media:Image) OR (parent:VideoClip AND media:Video))
  AND source.kg_import_namespace = $base_owner
  AND source.dataset_revision = $dataset_revision AND source.archive IN $archives
  AND ms.kg_import_namespace = $base_owner AND ms.build_id = $build_id
  AND ms.split = 'test' AND ms.dataset_revision = $dataset_revision
OPTIONAL MATCH (parent)-[alignment:ALIGNED_WITH]->(step:RecipeGen:Step)
WHERE alignment.kg_import_namespace = $owner AND alignment.build_id = $build_id
  AND alignment.split = 'test' AND alignment.dataset_revision = $dataset_revision
  AND alignment.run_id = observation.run_id
  AND step.kg_import_namespace = $base_owner AND step.build_id = $build_id
  AND step.split = 'test' AND step.dataset_revision = $dataset_revision
OPTIONAL MATCH (recipe:RecipeGen:Recipe)-[rm:HAS_IMAGE|HAS_VIDEO]->(media)
WHERE recipe.kg_import_namespace = $base_owner AND recipe.build_id = $build_id
  AND recipe.split = 'test' AND recipe.dataset_revision = $dataset_revision
  AND rm.kg_import_namespace = $base_owner AND rm.build_id = $build_id
  AND rm.split = 'test' AND rm.dataset_revision = $dataset_revision
RETURN observation.kg_csv_id AS observation_id, observation.caption AS caption,
  observation.source_media_id AS source_media_id, observation.source_id AS source_id,
  CASE WHEN media:Image THEN 'image' ELSE 'video' END AS media_kind,
  CASE WHEN parent:Image THEN 'Image' ELSE 'VideoClip' END AS parent_kind,
  parent.start_seconds AS start_sec, parent.end_seconds AS end_sec,
  parent.timeline_origin_seconds AS timeline_origin_sec,
  parent.time_reference AS time_reference,
  step.kg_csv_id AS step_id, step.text AS step_text, step.order AS step_order,
  step.source_id AS step_source_id, recipe.kg_csv_id AS recipe_id,
  recipe.title AS recipe_title, observation.evidence_json AS evidence_json,
  observation.raw_json AS raw_json, score,
  observation.run_id AS run_id, observation.build_id AS build_id,
  observation.split AS split, observation.dataset_revision AS dataset_revision,
  observation.kg_import_namespace AS namespace,
  observation.embedding_model AS embedding_model,
  observation.embedding_revision AS embedding_revision,
  h.run_id AS association_run_id, alignment.run_id AS alignment_run_id,
  source.archive AS source_archive, source.member AS source_member
ORDER BY score DESC, observation_id, step_order, recipe_id
LIMIT $limit
"""


def validate_request(embedding: Any, *, query: str = "", embedding_model: str,
                     embedding_revision: str, limit: int = 5,
                     candidate_limit: int = 100) -> list[float]:
    if not isinstance(query, str) or len(query.strip()) > 500:
        raise MultimodalQueryError("query 必须为至多 500 字符的字符串。")
    for value, name in ((embedding_model, "embedding_model"), (embedding_revision, "embedding_revision")):
        if not isinstance(value, str) or not value.strip():
            raise MultimodalQueryError(f"{name} 必须明确，不能混用不同 embedding 空间。")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise MultimodalQueryError("limit 必须为 1～20。")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or not limit <= candidate_limit <= 1000:
        raise MultimodalQueryError("candidate_limit 必须介于 limit 与 1000。")
    if not isinstance(embedding, (list, tuple)) or len(embedding) != 512:
        raise MultimodalQueryError("查询必须提供真实 CLIP 512 维向量。")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in embedding):
        raise MultimodalQueryError("查询向量必须全部为有限数值。")
    vector = [float(v) for v in embedding]
    if not any(vector):
        raise MultimodalQueryError("查询向量不能为零向量。")
    return vector


def _rows(tx: Any, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run(query, parameters)]


def _json_object(value: Any, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else None
    except (json.JSONDecodeError, ValueError) as error:
        raise MultimodalQueryError(f"{label} 不是有效的 JSON 来源对象。") from error
    if not isinstance(parsed, dict):
        raise MultimodalQueryError(f"{label} 必须保留 JSON 对象。")
    return parsed


def validate_result(row: dict[str, Any], build: dict[str, Any], *,
                    embedding_model: str, embedding_revision: str) -> dict[str, Any]:
    expected = {"namespace":NAMESPACE, "build_id":build["build_id"], "split":"test",
                "dataset_revision":build["dataset_revision"], "embedding_model":embedding_model,
                "embedding_revision":embedding_revision}
    if any(row.get(key) != value for key, value in expected.items()):
        raise MultimodalQueryError("候选的构建/测试范围/embedding 空间不一致。")
    for key in ("observation_id", "caption", "source_media_id", "source_id", "run_id", "source_member"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise MultimodalQueryError(f"候选缺少 {key}，不能补造证据。")
    if row.get("association_run_id") != row["run_id"]:
        raise MultimodalQueryError("HAS_OBSERVATION 与观察不属于同一 run_id。")
    if row.get("source_archive") not in ARCHIVES:
        raise MultimodalQueryError("只允许两个官方测试归档的媒体。")
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise MultimodalQueryError("向量候选 score 必须为 0～1 的有限数值。")
    evidence = _json_object(row.get("evidence_json"), "evidence_json")
    _json_object(row.get("raw_json"), "raw_json")
    for key in ("source_media_id", "source_id", "run_id", "build_id", "split", "dataset_revision"):
        if evidence.get(key) != row.get(key):
            raise MultimodalQueryError("候选 evidence_json 的来源/构建/run_id 不一致。")
    if row.get("step_id") is not None:
        order = row.get("step_order")
        if (not isinstance(row["step_id"], str) or not row["step_id"]
                or not isinstance(row.get("step_text"), str) or not row["step_text"]
                or isinstance(order, bool) or not isinstance(order, int) or order <= 0
                or not isinstance(row.get("step_source_id"), str) or not row["step_source_id"]
                or row.get("alignment_run_id") != row["run_id"]):
            raise MultimodalQueryError("候选缺少同一 run_id 的步骤原文、order 或来源。")
    elif any(row.get(key) is not None for key in ("step_text", "step_order", "step_source_id", "alignment_run_id")):
        raise MultimodalQueryError("无对齐步骤时不能返回虚构步骤字段。")
    if row.get("recipe_id") is None:
        if row.get("recipe_title") is not None:
            raise MultimodalQueryError("无 Recipe 的媒体不能补造菜谱标题。")
    elif not isinstance(row["recipe_id"], str) or not isinstance(row.get("recipe_title"), str):
        raise MultimodalQueryError("菜谱必须保留原 Recipe ID/title。")
    if row.get("media_kind") == "video" and row.get("parent_kind") == "VideoClip":
        start, end, origin = (row.get(key) for key in ("start_sec", "end_sec", "timeline_origin_sec"))
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start,end,origin))
                or start < 0 or end <= start or row.get("time_reference") != "relative_to_first_decoded_frame"):
            raise MultimodalQueryError("视频候选必须保留真实有效的时间窗/时间参照。")
    elif row.get("media_kind") == "image" and row.get("parent_kind") == "Image":
        if any(row.get(key) is not None for key in ("start_sec", "end_sec", "timeline_origin_sec", "time_reference")):
            raise MultimodalQueryError("图像候选不能带虚构视频时间窗。")
    else:
        raise MultimodalQueryError("仅检索原 Image 或真实 VideoClip 的观察。")
    return {**row, "score":float(score), "semantics":"model_candidate", "verified":False}


def search_session(session: Any, embedding: list[float], *, query: str = "",
                   embedding_model: str, embedding_revision: str, limit: int = 5,
                   candidate_limit: int = 100) -> dict[str, Any]:
    vector = validate_request(embedding, query=query, embedding_model=embedding_model,
                              embedding_revision=embedding_revision, limit=limit, candidate_limit=candidate_limit)
    started = time.perf_counter()

    def read(tx: Any) -> dict[str, Any]:
        builds = _rows(tx, ACTIVE_BUILD_QUERY, {"base_owner":BASE_NAMESPACE})
        if len(builds) != 1:
            raise MultimodalQueryError("没有唯一的 active+verified 测试构建。")
        build = builds[0]
        if (build.get("split") != "test" or build.get("active") is not True
                or build.get("status") != "verified" or not build.get("build_id") or not build.get("dataset_revision")):
            raise MultimodalQueryError("活动构建不满足 test/active/verified 范围。")
        rows = _rows(tx, SEARCH_QUERY, {"owner":NAMESPACE, "base_owner":BASE_NAMESPACE,
                     "build_id":build["build_id"], "dataset_revision":build["dataset_revision"],
                     "embedding_model":embedding_model, "embedding_revision":embedding_revision,
                     "candidate_limit":candidate_limit, "embedding":vector, "limit":limit,
                     "archives":list(ARCHIVES)})
        if len(rows) > limit:
            raise MultimodalQueryError("候选数量超出 limit。")
        results = [validate_result(row, build, embedding_model=embedding_model,
                                    embedding_revision=embedding_revision) for row in rows]
        if len({row["observation_id"] for row in results}) != len(results):
            raise MultimodalQueryError("同一 observation_id 出现重复来源或步骤，需先核查图谱关联。")
        return {"status":"ok" if results else "no_match", "backend":"neo4j",
                "read_only":True, "llm_called":False, "query":query.strip(), "build":build,
                "embedding_space":{"model":embedding_model, "revision":embedding_revision, "dimension":512},
                "matching":{"method":"clip_approximate_vector_candidates", "index":VECTOR_INDEX,
                            "post_filter":True, "candidate_limit":candidate_limit},
                "limit":limit, "count":len(results), "results":results, "limitations":list(LIMITATIONS)}

    report = session.execute_read(read)
    report.update(elapsed_ms=round((time.perf_counter()-started)*1000, 2),
                  finished_at=datetime.now(timezone.utc).isoformat())
    return report


def apply_evidence_selection(report: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """只接受已返回 ID 的选择；所有事实从原候选复制，不接受模型替换。"""
    candidates = report.get("results", [])
    if not candidates:
        raise MultimodalQueryError("无候选时不应调用证据选择模型。")
    if report.get("llm_called") is True or not isinstance(selection, dict):
        raise MultimodalQueryError("证据选择只能应用一次且必须是 JSON 对象。")
    ids, reason = selection.get("selected_observation_ids"), selection.get("reason")
    known = {row["observation_id"]:row for row in candidates}
    if (not isinstance(ids, list) or any(not isinstance(oid,str) or oid not in known for oid in ids)
            or len(set(ids)) != len(ids) or not isinstance(reason,str)):
        raise MultimodalQueryError("模型只能选择唯一的已有 observation_id，并给出字符串 reason。")
    result = copy.deepcopy(report)
    result["candidates"] = copy.deepcopy(candidates)
    result["results"] = [copy.deepcopy(known[oid]) for oid in ids]
    result.update(status="ok" if ids else "no_match", count=len(ids), llm_called=True)
    result["selection"] = {key:copy.deepcopy(selection[key]) for key in
                           ("selected_observation_ids", "reason", "raw_text", "model", "responses") if key in selection}
    result["selection"]["semantics"] = "model_evidence_selection"
    return result
