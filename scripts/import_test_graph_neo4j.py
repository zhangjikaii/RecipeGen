#!/usr/bin/env python3
"""Validate and version-import the test-only RecipeGen CSV graph into Neo4j.

The importer never deletes graph data. Every imported node has RecipeGen,
and every read/write is scoped to the validated build and split. Text labels
and relationship identifiers are taken exclusively from fixed allowlists.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "recipegen-test-v1"
ALLOWED_LABELS = frozenset({"Recipe", "Ingredient", "Step", "Action", "Tool", "Image", "Video", "Source"})
ALLOWED_RELATIONSHIPS = frozenset({"HAS_INGREDIENT", "HAS_STEP", "NEXT_STEP", "HAS_ACTION", "USES_INGREDIENT", "USES_TOOL", "HAS_IMAGE", "HAS_VIDEO", "STEP_IMAGE", "HAS_SOURCE"})
SOURCE_TYPES = frozenset({"huggingface_zip_text", "huggingface_zip_media"})
NODE_COLUMNS = {"id", "label", "properties_json"}
RELATIONSHIP_COLUMNS = {"id", "start_id", "end_id", "type", "properties_json"}
RESERVED_PROPERTIES = {"kg_id", "kg_csv_id", "kg_import_namespace"}


class ImportValidationError(ValueError):
    """Input is incomplete, mixed-split, or would change an immutable build."""


class ImportVerificationError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ImportValidationError(f"{field} 必须是非空字符串，且不能含首尾空白")
    return value


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ImportValidationError("properties_json 含重复属性键")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ImportValidationError("properties_json 含非有限数值")


def parse_properties(value: str, where: str) -> dict[str, Any]:
    try:
        properties = json.loads(value, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise ImportValidationError(f"{where} properties_json 不是合法 JSON 对象") from error
    if not isinstance(properties, dict):
        raise ImportValidationError(f"{where} properties_json 必须是对象")
    for key, item in properties.items():
        required_string(key, f"{where} 属性名")
        if key in RESERVED_PROPERTIES:
            raise ImportValidationError(f"{where} 不得预设导入器保留属性 {key}")
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, int) and -(2**63) <= item < 2**63:
            continue
        if isinstance(item, float) and math.isfinite(item):
            continue
        if isinstance(item, list) and all(isinstance(part, str) for part in item):
            continue
        raise ImportValidationError(f"{where} 属性 {key} 只能是 Neo4j 标量、null 或字符串列表，不能是嵌套对象或混合数组")
    for key in ("build_id", "dataset_revision"):
        required_string(properties.get(key), f"{where}.{key}")
    if not re.fullmatch(r"[0-9a-f]{40}", properties["dataset_revision"]):
        raise ImportValidationError(f"{where}.dataset_revision 必须是固定的 40 位 commit SHA")
    if properties.get("split") != "test":
        raise ImportValidationError(f"{where}.split 必须严格为 test；禁止混入 train/validation")
    for key in ("source_split", "dataset_split", "partition"):
        if key in properties and properties[key] != "test":
            raise ImportValidationError(f"{where}.{key} 与 test split 不一致")
    return properties


def read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or set(names) != columns:
            raise ImportValidationError(f"{path.name} 的 CSV 列必须为 {', '.join(sorted(columns))}")
        result = []
        for index, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ImportValidationError(f"{path.name}:{index} CSV 列数不一致")
            row = dict(row)
            row["_row"] = str(index)
            result.append(row)
    if not result:
        raise ImportValidationError(f"{path.name} 不能是空图文件")
    return result


def qualified_id(build_id: str, local_id: str, *, relationship: bool = False) -> str:
    # JSON tuple encoding prevents delimiter collisions between arbitrary IDs.
    digest = hashlib.sha256(json.dumps([NAMESPACE, build_id, "test", local_id], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return f"rgtest:{'edge' if relationship else 'node'}:{digest}"


def validate_source(properties: dict[str, Any], where: str) -> None:
    if properties.get("source_type") not in SOURCE_TYPES:
        raise ImportValidationError(f"{where}.source_type 必须明确为 huggingface_zip_text 或 huggingface_zip_media")
    for key in ("source_id", "artifact_path", "url"):
        required_string(properties.get(key), f"{where}.{key}")
    artifact = properties["artifact_path"]
    if artifact.startswith(("/", "\\")) or ".." in artifact.replace("\\", "/").split("/"):
        raise ImportValidationError(f"{where}.artifact_path 应为 repo/archive/raw_member，不得冒充本地路径")
    if len(artifact.split("/")) < 3:
        raise ImportValidationError(f"{where}.artifact_path 缺少 repo/archive/raw_member 来源信息")
    if re.search(r"(?:^|[/_.-])(?:train|training|val|validation)(?:$|[/_.-])", artifact, re.IGNORECASE):
        raise ImportValidationError(f"{where}.artifact_path 指向非 test 数据")
    url = urlsplit(properties["url"])
    if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.password:
        raise ImportValidationError(f"{where}.url 必须是公开的 Hugging Face HTTPS 归档地址")
    marker = f"/resolve/{properties['dataset_revision']}/"
    if marker not in url.path:
        raise ImportValidationError(f"{where}.url 必须固定到同一 dataset_revision 的 resolve 地址")
    repository_path, archive = url.path.split(marker, 1)
    if not repository_path.startswith("/datasets/") or not archive.lower().endswith(".zip"):
        raise ImportValidationError(f"{where}.url 必须明确指向数据集的 ZIP 归档")
    repository = unquote(repository_path.removeprefix("/datasets/"))
    archive = unquote(archive)
    prefix = f"{repository}/{archive}/"
    if not artifact.startswith(prefix) or not artifact[len(prefix):]:
        raise ImportValidationError(f"{where}.artifact_path 的 repo/archive/raw_member 与 resolve URL 不一致")


@dataclass(frozen=True)
class CSVBundle:
    nodes_path: Path
    relationships_path: Path
    build_id: str
    dataset_revision: str
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    content_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id, "split": "test", "dataset_revision": self.dataset_revision,
            "content_sha256": self.content_sha256, "nodes": len(self.nodes), "relationships": len(self.relationships),
            "nodes_by_label": dict(sorted(Counter(node["label"] for node in self.nodes).items())),
            "relationships_by_type": dict(sorted(Counter(edge["type"] for edge in self.relationships).items())),
            "sources_by_type": dict(sorted(Counter(node["properties"]["source_type"] for node in self.nodes if node["label"] == "Source").items())),
            "input_orphans": 0, "nodes_file": str(self.nodes_path.resolve()), "relationships_file": str(self.relationships_path.resolve()),
        }


def load_bundle(nodes_path: str | Path, relationships_path: str | Path) -> CSVBundle:
    nodes_path, relationships_path = Path(nodes_path), Path(relationships_path)
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    edge_ids: set[str] = set()
    scopes: set[tuple[str, str]] = set()
    sources: dict[str, dict[str, Any]] = {}
    for row in read_csv(nodes_path, NODE_COLUMNS):
        where = f"{nodes_path.name}:{row['_row']}"
        identifier = required_string(row["id"], f"{where}.id")
        if identifier in node_by_id:
            raise ImportValidationError(f"{where} 节点 ID 重复")
        if row["label"] not in ALLOWED_LABELS:
            raise ImportValidationError(f"{where} 节点 label 不在固定白名单")
        properties = parse_properties(row["properties_json"], where)
        if row["label"] == "Source":
            validate_source(properties, where)
            source_id = properties["source_id"]
            if source_id in sources:
                raise ImportValidationError(f"{where} Source.source_id 重复")
            sources[source_id] = properties
        node = {"id": identifier, "label": row["label"], "properties": properties}
        node_by_id[identifier] = node
        nodes.append(node)
        scopes.add((properties["build_id"], properties["dataset_revision"]))
    incident: Counter[str] = Counter()
    recipe_sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in read_csv(relationships_path, RELATIONSHIP_COLUMNS):
        where = f"{relationships_path.name}:{row['_row']}"
        identifier = required_string(row["id"], f"{where}.id")
        if identifier in edge_ids:
            raise ImportValidationError(f"{where} 关系 ID 重复")
        edge_ids.add(identifier)
        if row["type"] not in ALLOWED_RELATIONSHIPS:
            raise ImportValidationError(f"{where} relationship type 不在固定白名单")
        start = required_string(row["start_id"], f"{where}.start_id")
        end = required_string(row["end_id"], f"{where}.end_id")
        if start not in node_by_id or end not in node_by_id:
            raise ImportValidationError(f"{where} 关系引用了不存在的节点（dangling endpoint）")
        properties = parse_properties(row["properties_json"], where)
        scopes.add((properties["build_id"], properties["dataset_revision"]))
        if row["type"] == "HAS_SOURCE":
            if node_by_id[end]["label"] != "Source" or node_by_id[start]["label"] == "Source":
                raise ImportValidationError(f"{where} HAS_SOURCE 必须从内容节点指向 Source")
            recipe_sources[start].add(node_by_id[end]["properties"]["source_type"])
        relationships.append({"id": identifier, "start_id": start, "end_id": end, "type": row["type"], "properties": properties})
        incident[start] += 1
        incident[end] += 1
    if len(scopes) != 1:
        raise ImportValidationError("全部节点/关系必须属于同一个 build_id 和 dataset_revision")
    if not any(node["label"] == "Recipe" for node in nodes):
        raise ImportValidationError("测试图谱必须含 Recipe 节点")
    for node in nodes:
        if not incident[node["id"]]:
            raise ImportValidationError("输入含无任何关系的孤点；未关联媒体仍应保留 HAS_SOURCE 来源边")
        if node["label"] == "Recipe" and "huggingface_zip_text" not in recipe_sources[node["id"]]:
            raise ImportValidationError("每个 Recipe 必须通过 HAS_SOURCE 关联至少一个真实文本 Source")
        if node["label"] == "Step":
            source_id = required_string(node["properties"].get("source_id"), "Step.source_id")
            if source_id not in sources or sources[source_id]["source_type"] != "huggingface_zip_text":
                raise ImportValidationError("Step.source_id 必须引用当前构建提供的文本 Source")
    build_id, revision = next(iter(scopes))
    canonical = {"nodes": sorted(nodes, key=lambda row: row["id"]), "relationships": sorted(relationships, key=lambda row: row["id"])}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return CSVBundle(nodes_path, relationships_path, build_id, revision, nodes, relationships, digest)


CONSTRAINT_NODE = "CREATE CONSTRAINT recipegen_kg_id IF NOT EXISTS FOR (n:RecipeGen) REQUIRE n.kg_id IS UNIQUE"
CONSTRAINT_BUILD = "CREATE CONSTRAINT recipegen_build_id IF NOT EXISTS FOR (b:RecipeGenBuild) REQUIRE b.build_id IS UNIQUE"
RESERVE_BUILD = """// recipegen-test:reserve-build
MERGE (b:RecipeGenBuild {build_id: $build_id})
ON CREATE SET b.split = 'test', b.dataset_revision = $dataset_revision,
  b.content_sha256 = $content_sha256, b.kg_import_namespace = $owner,
  b.expected_nodes = $expected_nodes, b.expected_relationships = $expected_relationships,
  b.status = 'importing', b.active = false, b.created_at = datetime()
RETURN b.content_sha256 AS content_sha256, b.dataset_revision AS dataset_revision,
       b.split AS split, b.kg_import_namespace AS owner
"""
MARK_IMPORTING = """// recipegen-test:mark-importing
MATCH (b:RecipeGenBuild {build_id: $build_id, split: $split})
SET b.status = 'importing', b.active = false, b.last_attempt_at = datetime()
RETURN count(b) AS count
"""
NODE_COUNTS = """// recipegen-test:node-counts
MATCH (n:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})
UNWIND labels(n) AS label
WITH label WHERE label IN $allowed_labels
RETURN label, count(*) AS count ORDER BY label
"""
RELATIONSHIP_COUNTS = """// recipegen-test:relationship-counts
MATCH (a:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})-[r]->(b:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})
WHERE r.build_id = $build_id AND r.split = $split AND r.kg_import_namespace = $owner
RETURN type(r) AS type, count(r) AS count ORDER BY type
"""
VERIFY_ANOMALIES = """// recipegen-test:verify-anomalies
CALL () {
  MATCH (n:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})
  WHERE NOT EXISTS { MATCH (n)-[r]-(m:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})
    WHERE r.build_id = $build_id AND r.split = $split AND r.kg_import_namespace = $owner }
  RETURN count(n) AS orphans
}
CALL () {
  MATCH (a)-[r]->(b) WHERE r.build_id = $build_id AND r.split = $split AND r.kg_import_namespace = $owner
  AND (NOT a:RecipeGen OR NOT b:RecipeGen OR coalesce(a.build_id, '') <> $build_id
    OR coalesce(b.build_id, '') <> $build_id OR coalesce(a.split, '') <> $split OR coalesce(b.split, '') <> $split
    OR coalesce(a.kg_import_namespace, '') <> $owner OR coalesce(b.kg_import_namespace, '') <> $owner)
  RETURN count(r) AS out_of_scope_endpoints
}
CALL () {
  MATCH (n:RecipeGen {build_id: $build_id, split: $split, kg_import_namespace: $owner})
  WITH n.kg_id AS id, count(n) AS occurrences WHERE id IS NULL OR occurrences > 1
  RETURN coalesce(sum(occurrences), 0) AS invalid_node_ids
}
CALL () {
  MATCH ()-[r]->() WHERE r.build_id = $build_id AND r.split = $split AND r.kg_import_namespace = $owner
  WITH r.kg_id AS id, count(r) AS occurrences WHERE id IS NULL OR occurrences > 1
  RETURN coalesce(sum(occurrences), 0) AS invalid_relationship_ids
}
CALL () {
  MATCH (recipe:RecipeGen:Recipe {build_id: $build_id, split: $split, kg_import_namespace: $owner})
  WHERE NOT EXISTS { MATCH (recipe)-[r:HAS_SOURCE]->(s:RecipeGen:Source {build_id: $build_id, split: $split, kg_import_namespace: $owner})
    WHERE r.build_id = $build_id AND r.split = $split AND s.source_type = 'huggingface_zip_text' }
  RETURN count(recipe) AS recipes_without_text_source
}
RETURN orphans, out_of_scope_endpoints, invalid_node_ids, invalid_relationship_ids, recipes_without_text_source
"""
MARK_VERIFIED = """// recipegen-test:mark-verified
MATCH (b:RecipeGenBuild {build_id: $build_id, split: $split})
SET b.status = 'verified', b.active = $activate, b.nodes = $nodes,
    b.relationships = $relationships, b.verified_at = datetime()
RETURN count(b) AS count
"""
DEACTIVATE_OTHERS = """// recipegen-test:deactivate-others
MATCH (b:RecipeGenBuild {split: $split})
WHERE b.build_id <> $build_id AND b.kg_import_namespace = $owner
SET b.active = false
RETURN count(b) AS count
"""
MARK_FAILED = """// recipegen-test:mark-failed
MATCH (b:RecipeGenBuild {build_id: $build_id, split: $split})
SET b.status = 'failed', b.active = false, b.failure_type = $failure_type, b.failed_at = datetime()
RETURN count(b) AS count
"""


def rows(tx: Any, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run(query, **parameters)]


def chunks(values: list[dict[str, Any]], size: int):
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def native_stats(session: Any, bundle: CSVBundle) -> dict[str, Any]:
    parameters = {"build_id": bundle.build_id, "split": "test", "owner": NAMESPACE, "allowed_labels": sorted(ALLOWED_LABELS)}
    by_label = session.execute_read(lambda tx: rows(tx, NODE_COUNTS, parameters))
    by_type = session.execute_read(lambda tx: rows(tx, RELATIONSHIP_COUNTS, parameters))
    anomalies = session.execute_read(lambda tx: rows(tx, VERIFY_ANOMALIES, parameters))
    if len(anomalies) != 1:
        raise ImportValidationError("Neo4j 未返回完整的原生校验统计")
    result = {"nodes_by_label": {row["label"]: row["count"] for row in by_label},
              "relationships_by_type": {row["type"]: row["count"] for row in by_type}, **anomalies[0]}
    result["nodes"] = sum(result["nodes_by_label"].values())
    result["relationships"] = sum(result["relationships_by_type"].values())
    return result


def import_bundle(session: Any, bundle: CSVBundle, *, batch_size: int = 1000, activate: bool = True) -> dict[str, Any]:
    if not 1 <= batch_size <= 10_000:
        raise ImportValidationError("batch_size 必须为 1～10000")
    if any(node["label"] not in ALLOWED_LABELS for node in bundle.nodes) or any(edge["type"] not in ALLOWED_RELATIONSHIPS for edge in bundle.relationships):
        raise ImportValidationError("导入标识符不在固定白名单")
    canonical = {"nodes": sorted(bundle.nodes, key=lambda row: row["id"]), "relationships": sorted(bundle.relationships, key=lambda row: row["id"])}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != bundle.content_sha256:
        raise ImportValidationError("验证后的输入已被修改，必须重新读取 CSV 并完整验证")
    expected = bundle.summary()
    parameters = {"build_id": bundle.build_id, "split": "test", "dataset_revision": bundle.dataset_revision,
                  "content_sha256": bundle.content_sha256, "owner": NAMESPACE,
                  "expected_nodes": len(bundle.nodes), "expected_relationships": len(bundle.relationships)}
    for query in (CONSTRAINT_NODE, CONSTRAINT_BUILD):
        session.execute_write(lambda tx, query=query: rows(tx, query, {}))

    def reserve(tx: Any) -> None:
        records = rows(tx, RESERVE_BUILD, parameters)
        if len(records) != 1 or any(records[0].get(key) != parameters.get(key) for key in ("content_sha256", "dataset_revision", "split", "owner")):
            raise ImportValidationError("同一 build_id 已对应不同输入哈希或来源范围；请生成新的 build_id，不能覆盖旧构建")
        rows(tx, MARK_IMPORTING, parameters)
    session.execute_write(reserve)

    batches = {"nodes": 0, "relationships": 0}
    try:
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in bundle.nodes:
            kg_id = qualified_id(bundle.build_id, node["id"])
            groups[node["label"]].append({"kg_id": kg_id, "properties": {**node["properties"], "kg_id": kg_id, "kg_csv_id": node["id"], "kg_import_namespace": NAMESPACE}})
        for label in sorted(groups):
            query = f"UNWIND $rows AS row MERGE (n:RecipeGen:`{label}` {{kg_id: row.kg_id}}) SET n += row.properties RETURN count(n) AS count"
            for batch in chunks(groups[label], batch_size):
                records = session.execute_write(lambda tx, batch=batch, query=query: rows(tx, query, {"rows": batch}))
                if len(records) != 1 or records[0]["count"] != len(batch):
                    raise ImportValidationError("节点批次写入数量不一致")
                batches["nodes"] += 1
        groups = defaultdict(list)
        for edge in bundle.relationships:
            kg_id = qualified_id(bundle.build_id, edge["id"], relationship=True)
            groups[edge["type"]].append({"kg_id": kg_id, "start_kg_id": qualified_id(bundle.build_id, edge["start_id"]),
                "end_kg_id": qualified_id(bundle.build_id, edge["end_id"]),
                "properties": {**edge["properties"], "kg_id": kg_id, "kg_csv_id": edge["id"], "kg_import_namespace": NAMESPACE}})
        for relation in sorted(groups):
            query = f"UNWIND $rows AS row MATCH (a:RecipeGen {{kg_id: row.start_kg_id, build_id: $build_id, split: $split}}) MATCH (b:RecipeGen {{kg_id: row.end_kg_id, build_id: $build_id, split: $split}}) MERGE (a)-[r:`{relation}` {{kg_id: row.kg_id}}]->(b) SET r += row.properties RETURN count(r) AS count"
            for batch in chunks(groups[relation], batch_size):
                records = session.execute_write(lambda tx, batch=batch, query=query: rows(tx, query, {"rows": batch, "build_id": bundle.build_id, "split": "test"}))
                if len(records) != 1 or records[0]["count"] != len(batch):
                    raise ImportValidationError("关系批次端点缺失或写入数量不一致")
                batches["relationships"] += 1
        actual = native_stats(session, bundle)
        checks = {"node_counts": actual["nodes_by_label"] == expected["nodes_by_label"],
                  "relationship_counts": actual["relationships_by_type"] == expected["relationships_by_type"],
                  **{key: actual[key] == 0 for key in ("orphans", "out_of_scope_endpoints", "invalid_node_ids", "invalid_relationship_ids", "recipes_without_text_source")}}
        report = {"status": "verified" if all(checks.values()) else "failed", "validation_only": False,
                  "input": expected, "native_neo4j": actual, "checks": checks, "batches": batches, "activated": bool(activate and all(checks.values())),
                  "note": "原生 Neo4j 统计限定 RecipeGen + build_id + split；不同 build 的数据保留。"}
        if not all(checks.values()):
            raise ImportVerificationError("Neo4j 原生数量/来源/端点校验未通过，未激活该构建", report)

        def finish(tx: Any) -> None:
            if activate:
                rows(tx, DEACTIVATE_OTHERS, parameters)
            rows(tx, MARK_VERIFIED, {**parameters, "activate": activate, "nodes": actual["nodes"], "relationships": actual["relationships"]})
        session.execute_write(finish)
        return report
    except Exception as error:
        try:
            session.execute_write(lambda tx: rows(tx, MARK_FAILED, {**parameters, "failure_type": type(error).__name__}))
        except Exception:
            pass  # Preserve the original failure. Same immutable input may resume.
        raise


def load_environment(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key.replace("_", "").isalnum():
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    folder = PROJECT_ROOT / "data" / "test_graph" / "neo4j_import"
    parser.add_argument("--nodes", type=Path, default=folder / "nodes.csv")
    parser.add_argument("--relationships", type=Path, default=folder / "relationships.csv")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true", help="验证完整 CSV，不读取 Neo4j 凭据或连接数据库")
    parser.add_argument("--no-activate", action="store_true", help="导入并验证，但不修改其他构建的 active 状态")
    parser.add_argument("--database", default=None, help="默认使用 NEO4J_DATABASE 或 neo4j")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args(argv)
    report: dict[str, Any]
    try:
        if not 1 <= args.batch_size <= 10_000:
            raise ImportValidationError("batch-size 必须为 1～10000")
        bundle = load_bundle(args.nodes, args.relationships)
        if args.validate_only:
            report = {"status": "validated", "validation_only": True, "connected_to_neo4j": False, "input": bundle.summary()}
        else:
            load_environment(args.env_file)
            uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
            parsed = urlsplit(uri)
            if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"} or not parsed.hostname or parsed.username or parsed.password:
                raise ImportValidationError("NEO4J_URI 必须是有效的 bolt/neo4j 地址且不含凭据")
            user, password = os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")
            if not password:
                raise ImportValidationError("请通过 NEO4J_PASSWORD 环境变量或本地 .env 配置密码；不接受命令行密码")
            from neo4j import GraphDatabase
            with GraphDatabase.driver(uri, auth=(user, password)) as driver:
                driver.verify_connectivity()
                with driver.session(database=args.database or os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                    report = import_bundle(session, bundle, batch_size=args.batch_size, activate=not args.no_activate)
            report["connected_to_neo4j"] = True
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except ImportVerificationError as error:
        report = {**error.report, "error": str(error)}
    except (ImportValidationError, OSError) as error:
        report = {"status": "failed", "validation_only": args.validate_only, "error_type": type(error).__name__, "error": str(error)}
    except Exception as error:
        report = {"status": "failed", "validation_only": False, "error_type": type(error).__name__,
                  "error": "Neo4j 导入未完成；检查数据库连接/权限和版本。既有图数据未删除，可使用相同构建输入续导。"}
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
