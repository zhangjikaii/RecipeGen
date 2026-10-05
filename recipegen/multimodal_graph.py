"""Evidence-bound, append-only Neo4j extensions for an existing test graph.

Offline validation requires the validated original CSV bundle. It cannot attest
that a model actually ran or that a visual hypothesis is semantically correct.
Existing graph nodes are MATCHed, never created or updated by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "recipegen-mm-v1"
NAMESPACE = SCHEMA_VERSION
BASE_NAMESPACE = "recipegen-test-v1"
EXTENSION_LABELS = frozenset({"VisualObject", "VisualObservation", "VideoClip", "Frame", "SemanticRun"})
REFERENCE_LABELS = frozenset({"Image", "Video", "Step", "Ingredient", "Tool", "Action", "Recipe"})
RELATIONSHIP_ENDPOINTS = {
    "DEPICTS": ({"Image", "Frame", "VisualObservation"}, {"VisualObject"}),
    "DEPICTS_INGREDIENT": ({"Image", "Frame", "VisualObject", "VisualObservation"}, {"Ingredient"}),
    "DEPICTS_TOOL": ({"Image", "Frame", "VisualObject", "VisualObservation"}, {"Tool"}),
    "HAS_OBSERVATION": ({"Image", "Frame", "VideoClip"}, {"VisualObservation"}),
    "HAS_CLIP": ({"Video"}, {"VideoClip"}),
    "HAS_FRAME": ({"Video", "VideoClip"}, {"Frame"}),
    "SHOWS_ACTION": ({"Image", "Frame", "VideoClip", "VisualObservation"}, {"Action"}),
    "CONTAINS_OBJECT": ({"Image", "Frame", "VideoClip"}, {"VisualObject"}),
    "CONTAINS_INGREDIENT": ({"Image", "Frame", "VideoClip", "VisualObject"}, {"Ingredient"}),
    "BEFORE": ({"Frame", "VideoClip"}, {"Frame", "VideoClip"}),
    "ALIGNED_WITH": ({"Image", "Video", "Frame", "VideoClip", "VisualObject", "VisualObservation", "Step", "Recipe", "Ingredient", "Tool", "Action"},
                     {"Image", "Video", "Frame", "VideoClip", "VisualObject", "Step", "Recipe", "Ingredient", "Tool", "Action"}),
    "ON": ({"VisualObject"}, {"VisualObject"}),
    "BESIDE": ({"VisualObject"}, {"VisualObject"}),
    "MIXED_WITH": ({"VisualObject"}, {"VisualObject"}),
    "IN": ({"VisualObject"}, {"VisualObject"}),
    "HOLDING": ({"VisualObject"}, {"VisualObject"}),
    "USING": ({"VisualObject"}, {"VisualObject"}),
}
RESERVED = frozenset({"kg_id", "kg_csv_id", "kg_import_namespace", "payload_sha256", "content_sha256", "schema_version"})
SCOPE_KEYS = ("build_id", "dataset_revision", "split", "run_id")


class MultimodalValidationError(ValueError):
    """A bundle, original source, or native reference violates the contract."""


class MultimodalVerificationError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def required_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
        raise MultimodalValidationError(f"{where} 必须是无首尾空白或控制字符的非空字符串")
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise MultimodalValidationError(f"JSON 重复键：{key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise MultimodalValidationError(f"JSON 不允许非有限数值：{value}")


def parse_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise MultimodalValidationError("不是有效的严格 JSON") from error


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def qualified_id(build_id: str, run_id: str, identifier: str, *, relationship: bool = False) -> str:
    # Tuple hashing binds identities to both run and build without delimiter collisions.
    identity = [NAMESPACE, build_id, "test", run_id, "relationship" if relationship else "node", identifier]
    return f"rgmm:{'edge' if relationship else 'node'}:{canonical_hash(identity)}"


def _flat(properties: Any, where: str) -> dict[str, Any]:
    if not isinstance(properties, dict):
        raise MultimodalValidationError(f"{where} properties 必须为平坦对象")
    for key, value in properties.items():
        required_string(key, f"{where}.属性名")
        if key in RESERVED:
            raise MultimodalValidationError(f"{where} 不得设置保留属性 {key}")
        if value is None or isinstance(value, (str, bool)):
            continue
        if isinstance(value, int) and -(2**63) <= value < 2**63:
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        if isinstance(value, list):
            # Flat homogeneous arrays are Neo4j property values; nested data goes in evidence_json.
            if not value or all(isinstance(v, str) for v in value) or all(type(v) is bool for v in value):
                continue
            if all(type(v) in (int, float) and math.isfinite(v) and (not isinstance(v, int) or -(2**63) <= v < 2**63) for v in value):
                continue
        raise MultimodalValidationError(f"{where}.{key} 必须为有限标量或同类标量数组，禁止嵌套属性")
    return dict(properties)


def _scope(properties: Mapping[str, Any], scope: Mapping[str, Any], where: str) -> None:
    for key in SCOPE_KEYS:
        if key in properties and properties[key] != scope[key]:
            raise MultimodalValidationError(f"{where}.{key} 与 bundle 范围不一致")
    for key in ("source_split", "dataset_split", "partition"):
        if key in properties and properties[key] != "test":
            raise MultimodalValidationError(f"{where}.{key} 必须为 test")


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise MultimodalValidationError(f"{where} 必须是有限非负秒数，不能是 bool")
    return float(value)


@dataclass(frozen=True)
class MultimodalBundle:
    run_id: str
    build_id: str
    dataset_revision: str
    nodes: list[dict[str, Any]]
    references: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    source_references: list[dict[str, Any]]
    base_content_sha256: str
    content_sha256: str
    base_validation_method: str = "full_csv_canonical_contract"

    def summary(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "build_id": self.build_id,
                "split": "test", "dataset_revision": self.dataset_revision, "content_sha256": self.content_sha256,
                "base_content_sha256": self.base_content_sha256,
                "base_validation_method": self.base_validation_method,
                "extension_nodes": len(self.nodes), "reference_nodes": len(self.references),
                "extension_relationships": len(self.relationships),
                "nodes_by_label": dict(sorted(Counter(n["label"] for n in self.nodes).items())),
                "relationships_by_type": dict(sorted(Counter(e["type"] for e in self.relationships).items())),
                "semantic_accuracy_verified": False, "existing_facts_rewritten": False}

    @property
    def scope(self) -> dict[str, str]:
        return {"build_id": self.build_id, "split": "test", "dataset_revision": self.dataset_revision, "run_id": self.run_id}


@dataclass(frozen=True)
class BaseGraphIndex:
    """Reusable original-CSV indexes; build once for an entire recipe pipeline."""
    build_id: str
    dataset_revision: str
    content_sha256: str
    nodes_by_id: dict[str, dict[str, Any]]
    source_links: dict[str, set[str]]
    provenance: dict[str, Any] = field(default_factory=dict)


def prepare_base_graph(base_graph: Any) -> BaseGraphIndex:
    if isinstance(base_graph, BaseGraphIndex):
        return base_graph
    sources: dict[str, set[str]] = defaultdict(set)
    for edge in base_graph.relationships:
        if edge["type"] == "HAS_SOURCE":
            sources[edge["start_id"]].add(edge["end_id"])
    return BaseGraphIndex(base_graph.build_id, base_graph.dataset_revision, base_graph.content_sha256,
                          {n["id"]: n for n in base_graph.nodes}, dict(sources))


def load_base_index(project_root: str | Path) -> BaseGraphIndex:
    """Stream trusted original CSVs without materializing their large edge bundle.

    Trust is explicit: file SHA-256 values must match the local source manifest
    that recorded the prior successful full canonical/native validation. That
    prior deployment report supplies content_sha256. This helper does NOT
    recompute the canonical graph hash or re-validate every non-source edge.
    It checks byte identity, row shape, scope, counts, and HAS_SOURCE links, then
    retains all original node properties for immutable native-reference checks.
    No environment/private connection files or network are accessed. Call once
    and reuse the returned index for the entire multimodal pipeline.
    """
    from scripts.import_test_graph_neo4j import ALLOWED_LABELS, ImportValidationError, parse_properties, validate_source
    root = Path(project_root)
    manifest = parse_json((root / "reports/source-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise MultimodalValidationError("source-manifest 必须为文件 SHA-256 映射")
    required_paths = ["data/test_graph/neo4j_import/nodes.csv", "data/test_graph/neo4j_import/relationships.csv",
                      "reports/user-neo4j-deployment.json", "data/test_graph/build-report.json", "data/test_graph/manifest.json"]
    snapshots: dict[str, tuple[int, int, int]] = {}
    verified_hashes: dict[str, str] = {}
    for name in required_paths:
        expected = manifest.get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise MultimodalValidationError(f"source-manifest 缺少有效的 {name} SHA-256")
        path = root / name
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        state = (before.st_ino, before.st_size, before.st_mtime_ns)
        if state != (after.st_ino, after.st_size, after.st_mtime_ns) or digest.hexdigest() != expected:
            raise MultimodalValidationError(f"{name} 与先前验证文件 SHA-256 不一致或读取时发生变化")
        snapshots[name] = state
        verified_hashes[name] = expected
    deployment = parse_json((root / required_paths[2]).read_text(encoding="utf-8"))
    build = parse_json((root / required_paths[3]).read_text(encoding="utf-8"))
    corpus = parse_json((root / required_paths[4]).read_text(encoding="utf-8"))
    check_names = {"node_counts", "relationship_counts", "orphans", "out_of_scope_endpoints", "invalid_node_ids", "invalid_relationship_ids", "recipes_without_text_source"}
    if not isinstance(deployment, dict) or deployment.get("status") != "verified" or deployment.get("validation_only") is not False or deployment.get("connected_to_neo4j") is not True or deployment.get("activated") is not True or any(deployment.get("checks", {}).get(k) is not True for k in check_names):
        raise MultimodalValidationError("缺少先前真实导入/全量回读 verified 证据，不能跳过 canonical 验证")
    original = deployment.get("input")
    if not isinstance(original, dict) or not isinstance(build, dict) or not isinstance(corpus, dict):
        raise MultimodalValidationError("原部署、构建或来源报告格式无效")
    build_id = required_string(original.get("build_id"), "deployment.input.build_id")
    revision = required_string(original.get("dataset_revision"), "deployment.input.dataset_revision")
    content_hash = required_string(original.get("content_sha256"), "deployment.input.content_sha256")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        raise MultimodalValidationError("先前验证报告缺少固定提交或 canonical SHA-256")
    if original.get("split") != "test" or build.get("split") != "test" or build.get("build_id") != build_id or build.get("dataset_revision") != revision or corpus.get("split") != "test" or corpus.get("revision") != revision or corpus.get("status") != "complete" or corpus.get("complete_test_scope") is not True:
        raise MultimodalValidationError("部署/build/source manifest 必须同 build、固定 test revision 且已完整读取")
    archives = corpus.get("archives")
    if not isinstance(archives, list) or len(archives) != 2 or {a.get("archive") for a in archives if isinstance(a, dict)} != {"test.zip", "test-video.zip"} or any(not isinstance(a, dict) or a.get("split") != "test" or a.get("revision") != revision or a.get("status") != "complete" for a in archives):
        raise MultimodalValidationError("来源范围必须完整且仅为 test.zip/test-video.zip")
    scope = {"build_id": build_id, "split": "test", "dataset_revision": revision}
    node_path, edge_path = root / required_paths[0], root / required_paths[1]
    nodes: dict[str, dict[str, Any]] = {}
    source_links: dict[str, set[str]] = defaultdict(set)
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))

    def rows(path: Path, columns: set[str]):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != columns or len(reader.fieldnames or []) != len(columns):
                raise MultimodalValidationError(f"{path.name} CSV 列不符合原图谱合同")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise MultimodalValidationError(f"{path.name} CSV 行列数不一致")
                yield row

    try:
        for row in rows(node_path, {"id", "label", "properties_json"}):
            identifier = required_string(row["id"], "original CSV node.id")
            if identifier in nodes or row["label"] not in ALLOWED_LABELS:
                raise MultimodalValidationError("原 CSV 节点 ID 重复或 label 非法")
            properties = parse_properties(row["properties_json"], identifier)
            if any(properties.get(k) != v for k, v in scope.items()):
                raise MultimodalValidationError("原 CSV 节点 scope 与验证记录不一致")
            if row["label"] == "Source":
                validate_source(properties, identifier)
            nodes[identifier] = {"id": identifier, "label": row["label"], "properties": properties}
            node_counts[row["label"]] += 1
        for row in rows(edge_path, {"id", "start_id", "end_id", "type", "properties_json"}):
            edge_counts[row["type"]] += 1
            if row["type"] != "HAS_SOURCE":
                continue  # Byte identity to the previously validated CSV is the trust boundary.
            properties = parse_properties(row["properties_json"], row["id"])
            if any(properties.get(k) != v for k, v in scope.items()):
                raise MultimodalValidationError("HAS_SOURCE CSV scope 与验证记录不一致")
            start, end = row["start_id"], row["end_id"]
            if start not in nodes or end not in nodes or nodes[end]["label"] != "Source" or nodes[start]["label"] == "Source":
                raise MultimodalValidationError("原 CSV HAS_SOURCE 端点不符合来源合同")
            if end in source_links[start]:
                raise MultimodalValidationError("原 CSV 同一 HAS_SOURCE 来源关联重复")
            source_links[start].add(end)
    except (ImportValidationError, csv.Error) as error:
        raise MultimodalValidationError("原图谱流式 CSV/来源合同无效") from error
    if dict(node_counts) != original.get("nodes_by_label") or dict(node_counts) != build.get("node_counts") or sum(node_counts.values()) != original.get("nodes") or sum(node_counts.values()) != build.get("node_count"):
        raise MultimodalValidationError("原节点数量与先前实际验证报告不一致")
    if dict(edge_counts) != original.get("relationships_by_type") or dict(edge_counts) != build.get("relationship_counts") or sum(edge_counts.values()) != original.get("relationships") or sum(edge_counts.values()) != build.get("relationship_count"):
        raise MultimodalValidationError("原关系数量与先前实际验证报告不一致")
    for name, snapshot in snapshots.items():
        state = (root / name).stat()
        if snapshot != (state.st_ino, state.st_size, state.st_mtime_ns):
            raise MultimodalValidationError("索引读取期间原验证文件发生变化，拒绝使用混合快照")
    return BaseGraphIndex(build_id, revision, content_hash, nodes, dict(source_links),
                          {"validation_method": "previous_full_validation_plus_streamed_csv_sha256",
                           "canonical_hash_recomputed": False, "verified_file_sha256": verified_hashes,
                           "node_count": sum(node_counts.values()), "relationship_count": sum(edge_counts.values())})


def validate_bundle(data: Any, base_graph: Any) -> MultimodalBundle:
    """Validate against an original CSVBundle produced by import_test_graph_neo4j.load_bundle."""
    fields = {"schema_version", "run_id", "build_id", "split", "dataset_revision", "nodes", "relationships"}
    if not isinstance(data, dict) or set(data) != fields:
        raise MultimodalValidationError("bundle 必须包含且仅包含 schema_version/run_id/build_id/split/dataset_revision/nodes/relationships")
    if data["schema_version"] != SCHEMA_VERSION or data["split"] != "test":
        raise MultimodalValidationError("schema_version 必须为 recipegen-mm-v1，split 必须为 test")
    scope = {key: required_string(data[key], key) for key in SCOPE_KEYS}
    if not re.fullmatch(r"[0-9a-f]{40}", scope["dataset_revision"]):
        raise MultimodalValidationError("dataset_revision 必须为固定 40 位 commit SHA")
    if (base_graph.build_id, base_graph.dataset_revision) != (scope["build_id"], scope["dataset_revision"]):
        raise MultimodalValidationError("原图谱 CSV 的 build/revision 与扩展 bundle 不一致")
    if not isinstance(data["nodes"], list) or not isinstance(data["relationships"], list):
        raise MultimodalValidationError("nodes 与 relationships 必须为数组")
    base_graph = prepare_base_graph(base_graph)
    base = base_graph.nodes_by_id
    base_sources = base_graph.source_links
    declared: dict[str, dict[str, Any]] = {}
    extensions, references = [], []
    for raw in data["nodes"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "label", "properties"}:
            raise MultimodalValidationError("节点必须为 {id,label,properties}")
        identifier = required_string(raw["id"], "node.id")
        if identifier in declared:
            raise MultimodalValidationError("节点 ID 重复")
        label = raw["label"]
        if not isinstance(label, str) or label not in EXTENSION_LABELS | REFERENCE_LABELS:
            raise MultimodalValidationError("节点 label 不在固定白名单")
        properties = _flat(raw["properties"], identifier)
        _scope(properties, scope, identifier)
        if label in REFERENCE_LABELS:
            if properties.get("reference") is not True or set(properties) - {"reference", *SCOPE_KEYS}:
                raise MultimodalValidationError("已有节点引用只能声明 reference=true 及同值 scope，禁止重写原事实")
            original = base.get(identifier)
            if not original or original["label"] != label:
                raise MultimodalValidationError("已有引用 kg_csv_id/label 在原图谱 CSV 中不存在")
            if any(original["properties"].get(key) != scope[key] for key in ("build_id", "split", "dataset_revision")):
                raise MultimodalValidationError("已有引用的原 CSV scope 不一致")
            node = {"id": identifier, "label": label, "properties": properties, "original": original}
            references.append(node)
        else:
            if "reference" in properties or identifier in base:
                raise MultimodalValidationError("扩展节点不能冒充已有节点引用或复用原 CSV ID")
            node = {"id": identifier, "label": label, "properties": {**properties, **scope}}
            extensions.append(node)
        declared[identifier] = node
    runs = [n for n in extensions if n["label"] == "SemanticRun"]
    if len(runs) != 1:
        raise MultimodalValidationError("bundle 必须有且仅有 1 个 SemanticRun")

    used_sources: dict[str, dict[str, Any]] = {}

    def source_contract(properties: dict[str, Any], where: str) -> None:
        media_id = required_string(properties.get("source_media_id"), f"{where}.source_media_id")
        source_id = required_string(properties.get("source_id"), f"{where}.source_id")
        media = declared.get(media_id)
        if not media or media["label"] not in {"Image", "Video"}:
            raise MultimodalValidationError(f"{where} source_media_id 必须为声明的已有 Image/Video 引用")
        original = media["original"]
        source = base.get(source_id)
        if original["properties"].get("source_id") != source_id or source_id not in base_sources.get(media_id, set()) or not source or source["label"] != "Source":
            raise MultimodalValidationError(f"{where} source_id 与原媒体 CSV/HAS_SOURCE 不一致")
        sp = source["properties"]
        if sp.get("source_type") != "huggingface_zip_media" or any(sp.get(k) != scope[k] for k in ("build_id", "split", "dataset_revision")):
            raise MultimodalValidationError(f"{where} 原媒体 Source scope/type 不一致")
        if sp.get("archive") not in {"test.zip", "test-video.zip"} or original["properties"].get("archive") != sp.get("archive") or original["properties"].get("member") != sp.get("member"):
            raise MultimodalValidationError(f"{where} 原媒体与 Source archive/member 不一致")
        evidence = parse_json(required_string(properties.get("evidence_json"), f"{where}.evidence_json"))
        if not isinstance(evidence, dict) or evidence.get("source_media_id") != media_id or evidence.get("source_id") != source_id:
            raise MultimodalValidationError(f"{where} evidence_json 必须绑定相同 source_media_id/source_id")
        _scope(evidence, scope, f"{where}.evidence_json")
        for key in ("archive", "member", "artifact_path", "url"):
            if key in evidence and evidence[key] != sp.get(key):
                raise MultimodalValidationError(f"{where}.evidence_json.{key} 与真实 Source 不一致")
        if properties.get("verified") is not False:
            raise MultimodalValidationError(f"{where}.verified 必须为 false，模型候选不能提升为原事实")
        semantics = properties.get("semantics")
        if semantics not in {"model_candidate", "media_structure"}:
            raise MultimodalValidationError(f"{where}.semantics 必须为 model_candidate 或 media_structure")
        required_string(properties.get("model"), f"{where}.model")
        if "confidence" not in properties or properties["confidence"] is not None:
            raise MultimodalValidationError(f"{where}.confidence 必须显式为 null；生成模型没有校准概率")
        if "similarity_score" in properties and (type(properties["similarity_score"]) not in (int, float) or not math.isfinite(properties["similarity_score"]) or not -1 <= properties["similarity_score"] <= 1):
            raise MultimodalValidationError(f"{where}.similarity_score 必须为 [-1,1] 有限数值，不能当 confidence")
        used_sources[source_id] = source

    for node in extensions:
        p, label = node["properties"], node["label"]
        if label == "SemanticRun":
            if "source_media_id" in p or "source_id" in p:
                raise MultimodalValidationError("SemanticRun 是运行元数据，来源应写在具体扩展节点/关系")
            continue
        source_contract(p, node["id"])
        if label == "VisualObservation":
            embedding = p.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != 512 or any(type(v) not in (int, float) or not math.isfinite(v) for v in embedding) or not any(v != 0 for v in embedding):
                raise MultimodalValidationError("VisualObservation.embedding 必须为 512 维有限非零数值向量，不能含 bool")
            required_string(p.get("embedding_model"), "VisualObservation.embedding_model")
            if not isinstance(parse_json(required_string(p.get("raw_json"), "VisualObservation.raw_json")), dict):
                raise MultimodalValidationError("VisualObservation.raw_json 必须是保留模型原输出的 JSON 对象字符串")
        if label in {"Frame", "VideoClip"}:
            if declared[p["source_media_id"]]["label"] != "Video":
                raise MultimodalValidationError("Frame/VideoClip 必须来源于原 Video")
            if label == "Frame":
                _number(p.get("timestamp_seconds"), "Frame.timestamp_seconds")
            else:
                start = _number(p.get("start_seconds"), "VideoClip.start_seconds")
                end = _number(p.get("end_seconds"), "VideoClip.end_seconds")
                if end <= start:
                    raise MultimodalValidationError("VideoClip.end_seconds 必须大于 start_seconds")
    edges, edge_ids = [], set()
    for raw in data["relationships"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "start_id", "end_id", "type", "properties"}:
            raise MultimodalValidationError("关系必须为 {id,start_id,end_id,type,properties}")
        identifier = required_string(raw["id"], "relationship.id")
        if identifier in edge_ids:
            raise MultimodalValidationError("关系 ID 重复")
        edge_ids.add(identifier)
        kind = raw["type"]
        if not isinstance(kind, str) or kind not in RELATIONSHIP_ENDPOINTS:
            raise MultimodalValidationError("关系 type 不在固定白名单")
        start = declared.get(raw["start_id"]) if isinstance(raw["start_id"], str) else None
        end = declared.get(raw["end_id"]) if isinstance(raw["end_id"], str) else None
        if not start or not end:
            raise MultimodalValidationError("关系含 dangling endpoint，端点必须先声明")
        if start["id"] == end["id"]:
            raise MultimodalValidationError("扩展关系不允许自环")
        labels = RELATIONSHIP_ENDPOINTS[kind]
        if start["label"] not in labels[0] or end["label"] not in labels[1]:
            raise MultimodalValidationError(f"{kind} 端点 label 与扩展引用合同不一致")
        properties = _flat(raw["properties"], identifier)
        _scope(properties, scope, identifier)
        properties.update(scope)
        source_contract(properties, identifier)
        media_id = properties["source_media_id"]
        for endpoint in (start, end):
            if endpoint["label"] in EXTENSION_LABELS and endpoint["properties"].get("source_media_id") != media_id:
                raise MultimodalValidationError("关系与扩展端点必须绑定相同来源媒体")
        if kind in {"HAS_CLIP", "HAS_FRAME", "HAS_OBSERVATION", "BEFORE"}:
            if properties["semantics"] != "media_structure":
                raise MultimodalValidationError("HAS_CLIP/HAS_FRAME/HAS_OBSERVATION/BEFORE 必须明确为 media_structure")
            if start["label"] == "Video" and start["id"] != media_id:
                raise MultimodalValidationError("结构关系源 Video 与 source_media_id 不一致")
            if kind == "HAS_OBSERVATION" and start["label"] == "Image" and start["id"] != media_id:
                raise MultimodalValidationError("观察关系源 Image 与 source_media_id 不一致")
            if kind == "HAS_FRAME" and start["label"] == "VideoClip":
                moment = end["properties"]["timestamp_seconds"]
                if not start["properties"]["start_seconds"] <= moment <= start["properties"]["end_seconds"]:
                    raise MultimodalValidationError("Frame 时间必须位于关联 VideoClip 范围")
            if kind == "BEFORE":
                if start["label"] != end["label"]:
                    raise MultimodalValidationError("BEFORE 只能连接同类 Frame 或 VideoClip")
                a, b = start["properties"], end["properties"]
                if start["label"] == "Frame":
                    ordered = a["timestamp_seconds"] < b["timestamp_seconds"]
                else:
                    ordered = a["end_seconds"] <= b["start_seconds"]
                if not ordered:
                    raise MultimodalValidationError("BEFORE 必须符合原视频时间顺序，片段不能重叠")
        elif properties["semantics"] != "model_candidate":
            raise MultimodalValidationError("视觉/语义关系必须为 model_candidate")
        if start["label"] in {"Image", "Video"} and kind != "ALIGNED_WITH" and start["id"] != media_id:
            raise MultimodalValidationError("关系媒体端点与 source_media_id 不一致")
        edges.append({"id": identifier, "start_id": start["id"], "end_id": end["id"], "type": kind, "properties": properties})
    # Do not import unattached visual hypotheses; the run metadata node is exempt.
    incident = {e[k] for e in edges for k in ("start_id", "end_id")}
    if any(n["label"] != "SemanticRun" and n["id"] not in incident for n in extensions):
        raise MultimodalValidationError("扩展内容节点必须参与至少一条有来源的关系")
    canonical = {**scope, "schema_version": SCHEMA_VERSION,
                 "nodes": sorted([{k: n[k] for k in ("id", "label", "properties")} for n in declared.values()], key=lambda n: n["id"]),
                 "relationships": sorted(edges, key=lambda e: e["id"]), "base_content_sha256": base_graph.content_sha256}
    return MultimodalBundle(scope["run_id"], scope["build_id"], scope["dataset_revision"], extensions, references, edges,
                            list(used_sources.values()), base_graph.content_sha256, canonical_hash(canonical),
                            base_graph.provenance.get("validation_method", "full_csv_canonical_contract"))


ACTIVE_BUILD = """// recipegen-mm:active-build
MATCH (b:RecipeGenBuild {build_id: $build_id, split: 'test', active: true, status: 'verified'})
WHERE b.kg_import_namespace = $base_owner AND b.dataset_revision = $dataset_revision
RETURN b.content_sha256 AS content_sha256
"""
REFERENCES = """// recipegen-mm:references
UNWIND $rows AS row
MATCH (n:RecipeGen {kg_csv_id: row.id, build_id: $build_id, split: 'test', dataset_revision: $dataset_revision})
WHERE n.kg_import_namespace = $base_owner AND row.label IN labels(n)
RETURN row.id AS id, labels(n) AS labels, properties(n) AS properties
"""
SOURCE_LINKS = """// recipegen-mm:source-links
UNWIND $rows AS row
MATCH (m:RecipeGen {kg_csv_id: row.media_id, build_id: $build_id, split: 'test', dataset_revision: $dataset_revision})
  -[h:HAS_SOURCE]->(s:RecipeGen:Source {kg_csv_id: row.source_id, build_id: $build_id, split: 'test', dataset_revision: $dataset_revision})
WHERE m.kg_import_namespace = $base_owner AND s.kg_import_namespace = $base_owner
  AND h.kg_import_namespace = $base_owner AND h.build_id = $build_id AND h.split = 'test' AND h.dataset_revision = $dataset_revision
RETURN row.media_id AS media_id, row.source_id AS source_id, count(h) AS count
"""
EXISTING_NODES = """// recipegen-mm:existing-nodes
MATCH (n:RecipeGen) WHERE n.kg_id IN $ids
RETURN n.kg_id AS kg_id, labels(n) AS labels, properties(n) AS properties
"""
EXISTING_EDGES = """// recipegen-mm:existing-edges
MATCH (a)-[e]->(b) WHERE e.kg_id IN $ids
RETURN e.kg_id AS kg_id, type(e) AS type, a.kg_id AS start_kg_id, b.kg_id AS end_kg_id, properties(e) AS properties
"""
RUN_NODES = """// recipegen-mm:run-nodes
MATCH (n:RecipeGen {kg_import_namespace: $owner, run_id: $run_id, build_id: $build_id, split: 'test'})
RETURN n.kg_id AS kg_id, labels(n) AS labels, properties(n) AS properties
"""
RUN_EDGES = """// recipegen-mm:run-edges
MATCH (a)-[e]->(b) WHERE e.kg_import_namespace = $owner AND e.run_id = $run_id
  AND e.build_id = $build_id AND e.split = 'test'
RETURN e.kg_id AS kg_id, type(e) AS type, a.kg_id AS start_kg_id, b.kg_id AS end_kg_id, properties(e) AS properties
"""


def _rows(tx: Any, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [row.data() for row in tx.run(query, parameters)]


def _native_properties(properties: Mapping[str, Any]) -> dict[str, Any]:
    # Neo4j removes null property values; null stays explicit in the bundle/evidence file.
    return {k: v for k, v in properties.items() if v is not None}


def prepared_rows(bundle: MultimodalBundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes, edges = [], []
    ref_ids = {n["id"] for n in bundle.references}
    def endpoint(identifier: str) -> dict[str, Any]:
        return {"id": identifier, "reference": identifier in ref_ids,
                "kg_id": None if identifier in ref_ids else qualified_id(bundle.build_id, bundle.run_id, identifier)}
    for n in bundle.nodes:
        p = {**n["properties"], "kg_id": qualified_id(bundle.build_id, bundle.run_id, n["id"]),
             "kg_csv_id": n["id"], "kg_import_namespace": NAMESPACE, "schema_version": SCHEMA_VERSION,
             "payload_sha256": canonical_hash(n)}
        if n["label"] == "SemanticRun":
            p["content_sha256"] = bundle.content_sha256
        nodes.append({"kg_id": p["kg_id"], "label": n["label"], "properties": p})
    for e in bundle.relationships:
        p = {**e["properties"], "kg_id": qualified_id(bundle.build_id, bundle.run_id, e["id"], relationship=True),
             "kg_csv_id": e["id"], "kg_import_namespace": NAMESPACE, "schema_version": SCHEMA_VERSION,
             "payload_sha256": canonical_hash(e)}
        edges.append({"kg_id": p["kg_id"], "type": e["type"], "start": endpoint(e["start_id"]), "end": endpoint(e["end_id"]), "properties": p})
    return nodes, edges


def _parameters(bundle: MultimodalBundle) -> dict[str, Any]:
    return {**bundle.scope, "owner": NAMESPACE, "base_owner": BASE_NAMESPACE}


def _check_references(tx: Any, bundle: MultimodalBundle, batch_size: int) -> dict[str, str]:
    parameters = _parameters(bundle)
    builds = _rows(tx, ACTIVE_BUILD, parameters)
    if len(builds) != 1 or builds[0].get("content_sha256") != bundle.base_content_sha256:
        raise MultimodalValidationError("数据库没有匹配当前 CSV 哈希的唯一 active+verified 原构建")
    wanted = {n["id"]: n["original"] for n in bundle.references}
    wanted.update({n["id"]: n for n in bundle.source_references})
    ids: dict[str, str] = {}
    records = list(wanted.values())
    for offset in range(0, len(records), batch_size):
        items = records[offset:offset + batch_size]
        native = _rows(tx, REFERENCES, {**parameters, "rows": [{"id": n["id"], "label": n["label"]} for n in items]})
        received: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in native:
            received[row.get("id")].append(row)
        if set(received) != {n["id"] for n in items} or any(len(v) != 1 for v in received.values()):
            raise MultimodalValidationError("已有节点/Source 原生引用缺失、重复或范围不符")
        for n in items:
            row = received[n["id"]][0]
            p = row.get("properties", {})
            if n["label"] not in row.get("labels", []) or any(p.get(k) != v for k, v in _native_properties(n["properties"]).items()):
                raise MultimodalValidationError("原生引用事实/Source 与原 CSV 不一致，拒绝覆盖或猜测")
            ids[n["id"]] = required_string(p.get("kg_id"), "native.kg_id")
    source_ids = {n["id"] for n in bundle.source_references}
    links = [{"media_id": n["id"], "source_id": n["original"]["properties"].get("source_id")}
             for n in bundle.references if n["label"] in {"Image", "Video"} and n["original"]["properties"].get("source_id") in source_ids]
    for offset in range(0, len(links), batch_size):
        batch = links[offset:offset + batch_size]
        native = _rows(tx, SOURCE_LINKS, {**parameters, "rows": batch})
        expected = {(r["media_id"], r["source_id"]) for r in batch}
        actual = [(r.get("media_id"), r.get("source_id")) for r in native]
        if len(actual) != len(expected) or set(actual) != expected or any(r.get("count") != 1 for r in native):
            raise MultimodalValidationError("原生媒体 HAS_SOURCE 来源关系缺失、重复或范围不一致")
    return ids


def _check_native_nodes(native: list[dict[str, Any]], expected: dict[str, dict[str, Any]], *, complete: bool) -> bool:
    observed = [r.get("kg_id") for r in native]
    if len(observed) != len(set(observed)) or not set(observed) <= set(expected) or (complete and set(observed) != set(expected)):
        return False
    return all(expected[r["kg_id"]]["label"] in r.get("labels", []) and r.get("properties") == _native_properties(expected[r["kg_id"]]["properties"]) for r in native)


def _check_native_edges(native: list[dict[str, Any]], expected: dict[str, dict[str, Any]], *, complete: bool) -> bool:
    observed = [r.get("kg_id") for r in native]
    if len(observed) != len(set(observed)) or not set(observed) <= set(expected) or (complete and set(observed) != set(expected)):
        return False
    return all(r.get("type") == expected[r["kg_id"]]["type"] and r.get("start_kg_id") == expected[r["kg_id"]]["start_kg_id"]
               and r.get("end_kg_id") == expected[r["kg_id"]]["end_kg_id"]
               and r.get("properties") == _native_properties(expected[r["kg_id"]]["properties"]) for r in native)


def import_bundle(session: Any, bundle: MultimodalBundle, *, batch_size: int = 500) -> dict[str, Any]:
    """Import in one transaction; poststats is read-only and reports verified/unverified."""
    if type(batch_size) is not int or not 1 <= batch_size <= 5000:
        raise MultimodalValidationError("batch_size 必须为 1～5000 的整数")
    nodes, edges = prepared_rows(bundle)
    parameters = _parameters(bundle)
    node_map = {n["kg_id"]: n for n in nodes}

    def write(tx: Any) -> dict[str, Any]:
        reference_ids = _check_references(tx, bundle, batch_size)
        for e in edges:
            e["start_kg_id"] = reference_ids[e["start"]["id"]] if e["start"]["reference"] else e["start"]["kg_id"]
            e["end_kg_id"] = reference_ids[e["end"]["id"]] if e["end"]["reference"] else e["end"]["kg_id"]
        edge_map = {e["kg_id"]: e for e in edges}
        prior_nodes = _rows(tx, EXISTING_NODES, {"ids": list(node_map)})
        prior_edges = _rows(tx, EXISTING_EDGES, {"ids": list(edge_map)})
        if not _check_native_nodes(prior_nodes, node_map, complete=False) or not _check_native_edges(prior_edges, edge_map, complete=False):
            raise MultimodalValidationError("相同 run/扩展 ID 已有不同内容、label 或端点；必须使用新 run_id，拒绝覆盖")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for n in nodes:
            groups[n["label"]].append(n)
        for label in sorted(groups, key=lambda item: (item != "SemanticRun", item)):
            query = f"// recipegen-mm:merge-nodes\nUNWIND $rows AS row MERGE (n:RecipeGen:`{label}` {{kg_id: row.kg_id}}) ON CREATE SET n += row.properties RETURN n.kg_id AS kg_id, labels(n) AS labels, properties(n) AS properties"
            records = groups[label]
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                returned = _rows(tx, query, {"rows": batch})
                if not _check_native_nodes(returned, {n["kg_id"]: n for n in batch}, complete=True):
                    raise MultimodalValidationError("扩展节点 MERGE 回读不一致；事务回滚，原事实不覆盖")
        groups.clear()
        for e in edges:
            groups[e["type"]].append(e)
        for kind, records in sorted(groups.items()):
            query = f"// recipegen-mm:merge-edges\nUNWIND $rows AS row MATCH (a:RecipeGen {{kg_id: row.start_kg_id, build_id: $build_id, split: 'test', dataset_revision: $dataset_revision}}) MATCH (b:RecipeGen {{kg_id: row.end_kg_id, build_id: $build_id, split: 'test', dataset_revision: $dataset_revision}}) MERGE (a)-[e:`{kind}` {{kg_id: row.kg_id}}]->(b) ON CREATE SET e += row.properties RETURN e.kg_id AS kg_id, type(e) AS type, a.kg_id AS start_kg_id, b.kg_id AS end_kg_id, properties(e) AS properties"
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                returned = _rows(tx, query, {**parameters, "rows": batch})
                if not _check_native_edges(returned, {e["kg_id"]: e for e in batch}, complete=True):
                    raise MultimodalValidationError("扩展关系 MERGE 端点/内容回读不一致；事务回滚")
        return {"references": reference_ids, "edge_map": edge_map}

    written = session.execute_write(write)

    def read(tx: Any) -> dict[str, Any]:
        # Re-read reference facts plus every extension ID/property/endpoint. Exact
        # structure timestamps match the already validated temporal contracts.
        refs = _check_references(tx, bundle, batch_size)
        native_nodes = _rows(tx, RUN_NODES, parameters)
        native_edges = _rows(tx, RUN_EDGES, parameters)
        checks = {"references_unchanged": refs == written["references"],
                  "extension_node_ids_properties_and_scope": _check_native_nodes(native_nodes, node_map, complete=True),
                  "extension_edge_ids_endpoints_properties_and_time": _check_native_edges(native_edges, written["edge_map"], complete=True)}
        return {"status": "verified" if all(checks.values()) else "unverified", "connected_to_neo4j": True,
                "validation_only": False, "input": bundle.summary(), "checks": checks,
                "native_neo4j": {"extension_nodes": len(native_nodes), "extension_relationships": len(native_edges)},
                "existing_facts_rewritten": False, "semantic_accuracy_verified": False,
                "note": "模型关系保持候选；此报告核验 ID、来源、范围、属性、端点和时间结构，不证明模型语义正确。"}

    try:
        report = session.execute_read(read)
    except MultimodalValidationError as error:
        report = {"status": "unverified", "connected_to_neo4j": True, "validation_only": False,
                  "input": bundle.summary(), "checks": {"post_import_reference_contract": False}, "error": str(error)}
        raise MultimodalVerificationError("写入后只读验证失败，不报告已验证成功", report) from error
    if report["status"] != "verified":
        raise MultimodalVerificationError("写入后只读验证失败，不报告已验证成功", report)
    return report
