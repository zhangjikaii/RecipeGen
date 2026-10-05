#!/usr/bin/env python3
"""Read-only native Neo4j search over the verified active RecipeGen test build.

Title matching is a substring search. Ingredient matching uses exact normalized
or Chinese names from dictionary-rule text mentions. Neither proves a complete
ingredient list, inventory sufficiency, allergies, cooking time, or nutrition.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG = PROJECT_ROOT / ".runtime" / "active-neo4j.json"
PRIVATE_CONFIG = ACTIVE_CONFIG if ACTIVE_CONFIG.is_file() else PROJECT_ROOT / ".runtime" / "local-neo4j.json"
NAMESPACE = "recipegen-test-v1"
LIMITATIONS = [
    "Ingredient 是 dictionary_rule 从原始步骤中提取的 text mention candidate，原料列表不完整且未经人工逐条确认。",
    "查询不验证已有库存、忌口、过敏、营养或总烹饪用时，不应据此声称满足这些条件。",
    "Step 顺序保留原 steps.txt 非空行顺序；不添加、翻译或改写原文步骤。",
    "Image/Video 数量指图谱中的关联媒体节点；目录关联不证明视觉语义或步骤对应，媒体存在不等于已下载、解码或识别。",
    "此脚本只执行固定参数化 Cypher 读取，不调用大模型。",
]


class QueryError(ValueError):
    """A query/configuration/native-data contract cannot be satisfied."""


@dataclass(frozen=True)
class Connection:
    uri: str
    user: str
    password: str = field(repr=False)
    database: str = "neo4j"


def load_connection(config_path: Path = PRIVATE_CONFIG, environment: Mapping[str, str] | None = None) -> Connection:
    env = os.environ if environment is None else environment
    config: dict[str, Any] = {}
    # Complete explicit env credentials do not require reading a private file.
    explicit = all(env.get(key) for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
    if not explicit and config_path.is_file():
        if os.name == "posix" and config_path.stat().st_mode & 0o077:
            raise QueryError("本地连接配置必须仅当前用户可读写（权限 600）；凭据不会打印。")
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QueryError("无法读取私密 Neo4j JSON 连接配置。") from error
        if not isinstance(loaded, dict):
            raise QueryError("私密 Neo4j 连接配置必须是 JSON 对象。")
        config = loaded
    values = {key:env.get(variable) or config.get(key) for key, variable in
              [("uri", "NEO4J_URI"), ("user", "NEO4J_USER"), ("password", "NEO4J_PASSWORD"), ("database", "NEO4J_DATABASE")]}
    values["database"] = values["database"] or "neo4j"
    for key in ("uri", "user", "password", "database"):
        if not isinstance(values[key], str) or not values[key]:
            raise QueryError(f"缺少有效的 Neo4j {key} 配置；使用私密本地文件或对应 NEO4J 环境变量。")
    parsed = urlsplit(values["uri"])
    if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"} or not parsed.hostname or parsed.username or parsed.password:
        raise QueryError("Neo4j URI 必须是 bolt/neo4j 地址且不含内嵌凭据。")
    return Connection(**values)


ACTIVE_BUILD_QUERY = """// recipegen-test:read-active-build
MATCH (b:RecipeGenBuild {split: 'test', active: true, status: 'verified'})
WHERE b.kg_import_namespace = $owner
RETURN b.build_id AS build_id, b.split AS split, b.status AS status, b.active AS active,
       b.dataset_revision AS dataset_revision, b.content_sha256 AS content_sha256,
       b.nodes AS nodes, b.relationships AS relationships
ORDER BY b.verified_at DESC, b.build_id
LIMIT 2
"""

# Each independent CALL collapses its branch before the next branch runs. No
# joins between Step, Source, Ingredient and media lists, and no per-recipe calls.
SEARCH_QUERY = """// recipegen-test:read-search
MATCH (r:RecipeGen:Recipe {build_id: $build_id, split: 'test'})
WHERE r.dataset_revision = $dataset_revision
WITH r, toLower(coalesce(r.title, '')) CONTAINS $normalized_query AS title_match,
  EXISTS {
    MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient {build_id: $build_id, split: 'test'})
    WHERE h.build_id = $build_id AND h.split = 'test'
      AND i.dataset_revision = $dataset_revision
      AND (toLower(coalesce(i.normalized_name, '')) = $normalized_query OR coalesce(i.zh_name, '') = $query)
  } AS ingredient_match
WHERE title_match OR ingredient_match
WITH r, title_match, ingredient_match
ORDER BY CASE WHEN toLower(coalesce(r.title, '')) = $normalized_query THEN 0 WHEN ingredient_match THEN 1 ELSE 2 END, r.kg_csv_id
LIMIT $limit
CALL (r) {
  MATCH (r)-[h:HAS_STEP]->(s:RecipeGen:Step {build_id: $build_id, split: 'test'})
  WHERE h.build_id = $build_id AND h.split = 'test' AND s.dataset_revision = $dataset_revision
  WITH DISTINCT s ORDER BY s.order, s.kg_csv_id
  RETURN collect({id: s.kg_csv_id, order: s.order, text: s.text, source_id: s.source_id, order_method: s.order_method}) AS steps
}
CALL (r) {
  MATCH (r)-[h:HAS_SOURCE]->(s:RecipeGen:Source {build_id: $build_id, split: 'test'})
  WHERE h.build_id = $build_id AND h.split = 'test' AND s.dataset_revision = $dataset_revision
  WITH DISTINCT s, h.role AS role ORDER BY role, s.kg_csv_id
  RETURN collect({id: s.kg_csv_id, source_id: s.source_id, role: role,
    source_type: s.source_type, dataset: s.dataset, split: s.split,
    dataset_revision: s.dataset_revision, build_id: s.build_id, archive: s.archive,
    member: s.member, artifact_path: s.artifact_path, url: s.url, sha256: s.sha256}) AS sources
}
CALL (r) {
  MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient {build_id: $build_id, split: 'test'})
  WHERE h.build_id = $build_id AND h.split = 'test' AND i.dataset_revision = $dataset_revision
  WITH DISTINCT i, h ORDER BY i.normalized_name, i.kg_csv_id
  RETURN collect({id: i.kg_csv_id, normalized_name: i.normalized_name, zh_name: i.zh_name,
    extraction_method: h.extraction_method, semantics: h.semantics,
    verified: h.verified, complete_ingredient_list: h.complete_ingredient_list,
    evidence_json: h.evidence_json}) AS ingredient_mentions
}
CALL (r) {
  MATCH (r)-[h:HAS_IMAGE|HAS_VIDEO]->(m:RecipeGen {build_id: $build_id, split: 'test'})
  WHERE h.build_id = $build_id AND h.split = 'test' AND m.dataset_revision = $dataset_revision
    AND (m:Image OR m:Video)
  WITH DISTINCT m
  RETURN count(CASE WHEN m:Image THEN m END) AS images,
    count(CASE WHEN m:Video THEN m END) AS videos,
    count(CASE WHEN m:Image AND m.downloaded = true THEN m END) AS downloaded_images,
    count(CASE WHEN m:Video AND m.downloaded = true THEN m END) AS downloaded_videos,
    count(CASE WHEN m.recognition_status = 'not_run' THEN m END) AS recognition_not_run,
    collect(DISTINCT m.storage) AS storage_modes
}
RETURN {id: r.kg_csv_id, recipe_id: r.recipe_id, title: r.title, origin_archive: r.origin_archive,
        display_directory: r.display_directory, steps_count: r.steps_count,
        ingredient_status: r.ingredient_status, total_cooking_time_status: r.total_cooking_time_status,
        media_state: r.media_state, text_complete: r.text_complete, quality_flags: r.quality_flags} AS recipe,
  title_match, ingredient_match, steps, sources, ingredient_mentions,
  {images: images, videos: videos, downloaded_images: downloaded_images,
   downloaded_videos: downloaded_videos, recognition_not_run: recognition_not_run,
   storage_modes: storage_modes, association_semantics: 'directory_membership_only'} AS media
"""


def fetch_rows(tx: Any, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    # Pass a map: a user-facing parameter named query must not collide with
    # Transaction.run(query, ...) itself.
    return [record.data() for record in tx.run(query, parameters)]


def validate_result(row: dict[str, Any], build: dict[str, Any]) -> dict[str, Any]:
    recipe = row.get("recipe")
    if not isinstance(recipe, dict) or not recipe.get("id"):
        raise QueryError("原生检索结果缺少 Recipe.kg_csv_id。")
    sources, steps, mentions = row.get("sources"), row.get("steps"), row.get("ingredient_mentions")
    if not all(isinstance(value, list) for value in (sources, steps, mentions)):
        raise QueryError("原生检索结果未返回独立的步骤、来源和提及列表。")
    if not sources or not any(source.get("source_type") == "huggingface_zip_text" for source in sources):
        raise QueryError("检索到的菜谱没有真实文本 Source，不能生成无来源结果。")
    for source in sources:
        if source.get("split") != "test" or source.get("build_id") != build["build_id"] or source.get("dataset_revision") != build["dataset_revision"]:
            raise QueryError("返回来源与活动测试构建范围不一致。")
        if not all(isinstance(source.get(key), str) and source[key] for key in ("source_id", "artifact_path", "url")):
            raise QueryError("返回来源缺少 source_id/artifact_path/url，不能自动补齐。")
    step_sources = {source["source_id"] for source in sources if source.get("role") == "steps" and source.get("source_type") == "huggingface_zip_text"}
    orders = [step.get("order") for step in steps]
    if any(isinstance(order, bool) or not isinstance(order, int) or order <= 0 for order in orders) or orders != sorted(set(orders)):
        raise QueryError("原始步骤未按唯一的正整数 order 返回，不能猜测顺序。")
    if recipe.get("steps_count") is not None and recipe["steps_count"] != len(steps):
        raise QueryError("原生步骤数量与 Recipe.steps_count 不一致。")
    for step in steps:
        if not isinstance(step.get("text"), str) or not step["text"] or step.get("source_id") not in step_sources:
            raise QueryError("原始步骤缺少原文或对应 steps Source。")
    media = row.get("media", {})
    for key in ("images", "videos", "downloaded_images", "downloaded_videos", "recognition_not_run"):
        if isinstance(media.get(key), bool) or not isinstance(media.get(key), int) or media[key] < 0:
            raise QueryError("原生媒体统计不是有效的非负整数。")
    matched_by = [name for name, flag in [("title_substring", row.get("title_match")), ("ingredient_exact", row.get("ingredient_match"))] if flag]
    return {**recipe, "matched_by":matched_by, "steps":steps, "sources":sources,
            "ingredient_mentions":mentions, "media":media,
            "ingredient_semantics":"rule_text_mention_incomplete", "inventory_or_exclusion_verified":False}


def search_session(session: Any, query: str, limit: int = 3) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 500:
        raise QueryError("query 必须为 1～500 个字符的标题或食材名称。")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise QueryError("limit 必须为 1～20。")
    query = query.strip()
    started = time.perf_counter()

    def read(tx: Any) -> dict[str, Any]:
        builds = fetch_rows(tx, ACTIVE_BUILD_QUERY, {"owner":NAMESPACE})
        if len(builds) != 1:
            raise QueryError("没有唯一的 active+verified 测试构建；请先完成导入验证并激活。")
        build = builds[0]
        if build.get("split") != "test" or build.get("active") is not True or build.get("status") != "verified" or not build.get("build_id") or not build.get("dataset_revision"):
            raise QueryError("活动构建未满足 test/active/verified 范围约束。")
        parameters = {"build_id":build["build_id"], "dataset_revision":build["dataset_revision"],
                      "query":query, "normalized_query":query.lower(), "limit":limit}
        records = fetch_rows(tx, SEARCH_QUERY, parameters)
        if len(records) > limit:
            raise QueryError("原生检索返回数量超出 limit。")
        results = [validate_result(row, build) for row in records]
        return {"status":"ok" if results else "no_match", "backend":"neo4j", "read_only":True,
                "llm_called":False, "query":query, "limit":limit, "build":build, "count":len(results),
                "matching":{"title":"case_insensitive_substring", "ingredient":"exact_normalized_name_or_zh_name"},
                "results":results, "limitations":list(LIMITATIONS)}
    report = session.execute_read(read)
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def connect_and_search(connection: Connection, query: str, limit: int) -> dict[str, Any]:
    from neo4j import GraphDatabase, READ_ACCESS
    with GraphDatabase.driver(connection.uri, auth=(connection.user, connection.password), connection_timeout=10, max_transaction_retry_time=10) as driver:
        driver.verify_connectivity()
        with driver.session(database=connection.database, default_access_mode=READ_ACCESS) as session:
            return search_session(session, query, limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="标题子串或规则食材的 normalized_name/zh_name 精确值")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=PRIVATE_CONFIG, help="私密本地 Neo4j 配置；显式 NEO4J 环境变量优先")
    args = parser.parse_args(argv)
    try:
        # Validate before reading credentials or initiating network access.
        if not args.query.strip() or len(args.query.strip()) > 500 or not 1 <= args.limit <= 20:
            raise QueryError("query 必须为 1～500 个字符，limit 必须为 1～20。")
        report = connect_and_search(load_connection(args.config), args.query, args.limit)
        return_code = 0
    except QueryError as error:
        report = {"status":"failed", "read_only":True, "llm_called":False, "error":str(error)}
        return_code = 1
    except Exception as error:
        # Never serialize driver errors: a URI or connection payload may be secret.
        report = {"status":"failed", "read_only":True, "llm_called":False, "error_type":type(error).__name__,
                  "error":"Neo4j 读取未完成；检查本地服务、私密连接配置和已验证活动构建。凭据不会输出。"}
        return_code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
