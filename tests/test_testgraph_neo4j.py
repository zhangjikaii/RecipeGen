"""Offline importer contracts; explicit opt-in permits a dedicated Neo4j check."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recipegen_testgraph_import_contract", PROJECT / "scripts" / "import_test_graph_neo4j.py")
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
assert SPEC.loader is not None
SPEC.loader.exec_module(IMPORTER)
REVISION = "a" * 40


def graph_rows(build_id="contract-build"):
    common = {"split": "test", "dataset_revision": REVISION, "build_id": build_id}
    def node(identifier, label, **values):
        return {"id": identifier, "label": label, "properties": {**common, **values}}
    def source(identifier, source_id, member):
        return node(identifier, "Source", source_type="huggingface_zip_text", source_id=source_id,
                    artifact_path=f"test-owner/RecipeGen/test.zip/{member}",
                    url=f"https://huggingface.co/datasets/test-owner/RecipeGen/resolve/{REVISION}/test.zip")
    nodes = [node("recipe:r1", "Recipe", name="土豆示例", minutes=None),
             node("ingredient:potato", "Ingredient", name="土豆", aliases=["马铃薯", "洋芋"]),
             node("step:r1:1", "Step", text="将土豆洗净", order=1, source_id="text:steps:r1"),
             source("source:goals:r1", "text:goals:r1", "goals/r1.txt"),
             source("source:steps:r1", "text:steps:r1", "steps/r1.txt")]
    relationships = [{"id": f"edge:{index}", "start_id": start, "end_id": end, "type": relation, "properties": dict(common)}
                     for index, (start, end, relation) in enumerate([
                         ("recipe:r1", "ingredient:potato", "HAS_INGREDIENT"),
                         ("recipe:r1", "step:r1:1", "HAS_STEP"),
                         ("step:r1:1", "ingredient:potato", "USES_INGREDIENT"),
                         ("recipe:r1", "source:goals:r1", "HAS_SOURCE"),
                         ("recipe:r1", "source:steps:r1", "HAS_SOURCE"),
                     ], 1)]
    return nodes, relationships


def write_csv_graph(folder, nodes, relationships):
    folder.mkdir(parents=True, exist_ok=True)
    node_path, edge_path = folder / "nodes.csv", folder / "relationships.csv"
    for path, records, names in [(node_path, nodes, ["id", "label", "properties_json"]),
                                  (edge_path, relationships, ["id", "start_id", "end_id", "type", "properties_json"])]:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            for record in records:
                writer.writerow({**{key: value for key, value in record.items() if key != "properties"},
                                 "properties_json": json.dumps(record["properties"], ensure_ascii=False)})
    return node_path, edge_path


def bundle(tmp_path, build_id="contract-build"):
    return IMPORTER.load_bundle(*write_csv_graph(tmp_path, *graph_rows(build_id)))


class Record(dict):
    def data(self):
        return dict(self)


class NativeNeo4jMock:
    """Stateful versioned store; native count queries inspect imported records."""
    def __init__(self):
        self.nodes, self.relationships, self.builds = {}, {}, {}
        self.queries = []
        self.force_orphan_count = 0

    def execute_write(self, callback):
        before = copy.deepcopy((self.nodes, self.relationships, self.builds))
        try:
            return callback(self)
        except Exception:
            self.nodes, self.relationships, self.builds = before
            raise

    def execute_read(self, callback):
        return callback(self)

    def scoped_nodes(self, params):
        return {identifier: value for identifier, value in self.nodes.items()
                if value["properties"].get("build_id") == params["build_id"] and value["properties"].get("split") == params["split"]}

    def scoped_edges(self, params):
        return {identifier: value for identifier, value in self.relationships.items()
                if value["properties"].get("build_id") == params["build_id"] and value["properties"].get("split") == params["split"]
                and value["properties"].get("kg_import_namespace") == params["owner"]}

    def run(self, query, **params):
        self.queries.append((query, copy.deepcopy(params)))
        if query in {IMPORTER.CONSTRAINT_NODE, IMPORTER.CONSTRAINT_BUILD}:
            return []
        if query == IMPORTER.RESERVE_BUILD:
            metadata = self.builds.setdefault(params["build_id"], {**params, "status":"importing", "active":False})
            return [Record(**{key:metadata[key] for key in ["content_sha256", "dataset_revision", "split", "owner"]})]
        if query == IMPORTER.MARK_IMPORTING:
            self.builds[params["build_id"]].update(status="importing", active=False)
            return [Record(count=1)]
        if query == IMPORTER.DEACTIVATE_OTHERS:
            count = 0
            for identifier, metadata in self.builds.items():
                if identifier != params["build_id"] and metadata["split"] == params["split"]:
                    metadata["active"] = False
                    count += 1
            return [Record(count=count)]
        if query == IMPORTER.MARK_VERIFIED:
            self.builds[params["build_id"]].update(status="verified", active=params["activate"], nodes=params["nodes"], relationships=params["relationships"])
            return [Record(count=1)]
        if query == IMPORTER.MARK_FAILED:
            self.builds[params["build_id"]].update(status="failed", active=False)
            return [Record(count=1)]
        if query.startswith("UNWIND $rows AS row MERGE (n:"):
            label = re.search(r"RecipeGen:`([^`]+)`", query)[1]
            for row in params["rows"]:
                self.nodes[row["kg_id"]] = {"label":label, "properties":copy.deepcopy(row["properties"])}
            return [Record(count=len(params["rows"]))]
        if query.startswith("UNWIND $rows AS row MATCH (a:"):
            relation = re.search(r"\[r:`([^`]+)`", query)[1]
            count = 0
            for row in params["rows"]:
                if row["start_kg_id"] not in self.nodes or row["end_kg_id"] not in self.nodes:
                    continue
                self.relationships[row["kg_id"]] = {"type":relation, "start":row["start_kg_id"], "end":row["end_kg_id"], "properties":copy.deepcopy(row["properties"])}
                count += 1
            return [Record(count=count)]
        if query == IMPORTER.NODE_COUNTS:
            counts = Counter(node["label"] for node in self.scoped_nodes(params).values())
            return [Record(label=label, count=count) for label, count in sorted(counts.items())]
        if query == IMPORTER.RELATIONSHIP_COUNTS:
            nodes = self.scoped_nodes(params)
            counts = Counter(edge["type"] for edge in self.scoped_edges(params).values() if edge["start"] in nodes and edge["end"] in nodes)
            return [Record(type=label, count=count) for label, count in sorted(counts.items())]
        if query == IMPORTER.VERIFY_ANOMALIES:
            nodes, edges = self.scoped_nodes(params), self.scoped_edges(params)
            connected = {identifier for edge in edges.values() for identifier in [edge["start"], edge["end"]]
                         if edge["start"] in nodes and edge["end"] in nodes}
            recipes = {identifier for identifier, node in nodes.items() if node["label"] == "Recipe"}
            source_linked = {edge["start"] for edge in edges.values() if edge["type"] == "HAS_SOURCE" and edge["end"] in nodes
                             and nodes[edge["end"]]["label"] == "Source" and nodes[edge["end"]]["properties"]["source_type"] == "huggingface_zip_text"}
            return [Record(orphans=len(set(nodes)-connected)+self.force_orphan_count,
                           out_of_scope_endpoints=sum(edge["start"] not in nodes or edge["end"] not in nodes for edge in edges.values()),
                           invalid_node_ids=0, invalid_relationship_ids=0, recipes_without_text_source=len(recipes-source_linked))]
        raise AssertionError(f"Unexpected importer query: {query}")


def test_complete_csv_validation_and_canonical_hash(tmp_path):
    nodes, relationships = graph_rows()
    first = IMPORTER.load_bundle(*write_csv_graph(tmp_path / "first", nodes, relationships))
    second = IMPORTER.load_bundle(*write_csv_graph(tmp_path / "second", list(reversed(nodes)), list(reversed(relationships))))
    assert first.content_sha256 == second.content_sha256
    assert first.summary()["nodes_by_label"] == {"Ingredient":1, "Recipe":1, "Source":2, "Step":1}
    assert first.summary()["sources_by_type"] == {"huggingface_zip_text":2}


@pytest.mark.parametrize("change,match", [
    ("duplicate_node", "节点 ID 重复"), ("duplicate_edge", "关系 ID 重复"),
    ("dangling", "dangling"), ("node_train", "split"), ("edge_train", "split"),
    ("different_build", "同一个 build_id"), ("different_revision", "同一个 build_id"),
    ("label_injection", "白名单"), ("relationship_injection", "白名单"),
    ("nested_property", "嵌套"), ("mixed_list", "混合"), ("integer_overflow", "Neo4j 标量"),
    ("missing_source_id", "source_id"), ("missing_artifact", "artifact_path"),
    ("wrong_source_revision", "resolve"), ("train_artifact", "非 test"),
    ("wrong_archive", "不一致"),
    ("missing_step_source", "Step.source_id"), ("unknown_step_source", "文本 Source"),
    ("orphan", "孤点"), ("no_recipe_source", "真实文本 Source"),
])
def test_invalid_input_rejected_before_any_database_access(tmp_path, change, match):
    nodes, relationships = graph_rows()
    if change == "duplicate_node": nodes.append(copy.deepcopy(nodes[0]))
    if change == "duplicate_edge": relationships.append(copy.deepcopy(relationships[0]))
    if change == "dangling": relationships[0]["end_id"] = "missing-node"
    if change == "node_train": nodes[0]["properties"]["split"] = "train"
    if change == "edge_train": relationships[0]["properties"]["split"] = "train"
    if change == "different_build": relationships[0]["properties"]["build_id"] = "another-build"
    if change == "different_revision": nodes[0]["properties"]["dataset_revision"] = "b"*40
    if change == "label_injection": nodes[0]["label"] = "Recipe`) DETACH DELETE n //"
    if change == "relationship_injection": relationships[0]["type"] = "HAS_STEP`] DELETE r //"
    if change == "nested_property": nodes[0]["properties"]["unsupported"] = {"nested":True}
    if change == "mixed_list": nodes[0]["properties"]["unsupported"] = ["a", 1]
    if change == "integer_overflow": nodes[0]["properties"]["unsupported"] = 2**64
    if change == "missing_source_id": del nodes[3]["properties"]["source_id"]
    if change == "missing_artifact": del nodes[3]["properties"]["artifact_path"]
    if change == "wrong_source_revision": nodes[3]["properties"]["url"] = nodes[3]["properties"]["url"].replace(REVISION, "main")
    if change == "train_artifact": nodes[3]["properties"]["artifact_path"] = "owner/repo/train.zip/goals/r1.txt"
    if change == "wrong_archive": nodes[3]["properties"]["artifact_path"] = "test-owner/RecipeGen/another.zip/goals/r1.txt"
    if change == "missing_step_source": del nodes[2]["properties"]["source_id"]
    if change == "unknown_step_source": nodes[2]["properties"]["source_id"] = "not-provided"
    if change == "orphan": nodes.append({"id":"orphan", "label":"Image", "properties":dict(nodes[0]["properties"])})
    if change == "no_recipe_source":
        for edge in relationships:
            if edge["type"] == "HAS_SOURCE": edge["start_id"] = "ingredient:potato"
    with pytest.raises(IMPORTER.ImportValidationError, match=match):
        IMPORTER.load_bundle(*write_csv_graph(tmp_path, nodes, relationships))


def test_unassociated_media_with_its_own_source_is_not_an_orphan(tmp_path):
    nodes, relationships = graph_rows()
    metadata = {key:nodes[0]["properties"][key] for key in ["split", "dataset_revision", "build_id"]}
    nodes.append({"id":"video:unmatched", "label":"Video", "properties":dict(metadata)})
    nodes.append({"id":"source:media", "label":"Source", "properties":{**nodes[3]["properties"], "source_id":"media:unmatched", "source_type":"huggingface_zip_media", "artifact_path":"test-owner/RecipeGen/test.zip/videos/unmatched.mp4"}})
    relationships.append({"id":"edge:media", "start_id":"video:unmatched", "end_id":"source:media", "type":"HAS_SOURCE", "properties":metadata})
    validated = IMPORTER.load_bundle(*write_csv_graph(tmp_path, nodes, relationships))
    assert validated.summary()["sources_by_type"]["huggingface_zip_media"] == 1


def test_duplicate_json_keys_and_nonfinite_values_are_rejected():
    for value in ['{"split":"test","split":"train"}', '{"bad":NaN}', '{"bad":Infinity}']:
        with pytest.raises(IMPORTER.ImportValidationError): IMPORTER.parse_properties(value, "input")


def test_validate_only_never_reads_credentials_or_connects(tmp_path, monkeypatch, capsys):
    node_path, edge_path = write_csv_graph(tmp_path, *graph_rows())
    def forbidden(*args): raise AssertionError("validate-only must not access environment/Neo4j")
    monkeypatch.setattr(IMPORTER, "load_environment", forbidden)
    assert IMPORTER.main(["--nodes",str(node_path),"--relationships",str(edge_path),"--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["connected_to_neo4j"] is False
    assert result["validation_only"] is True


def test_parameterized_batch_import_is_idempotent_and_scoped(tmp_path):
    validated = bundle(tmp_path)
    session = NativeNeo4jMock()
    first = IMPORTER.import_bundle(session, validated, batch_size=2)
    second = IMPORTER.import_bundle(session, validated, batch_size=2)
    assert first["status"] == second["status"] == "verified"
    assert len(session.nodes) == 5 and len(session.relationships) == 5
    assert first["native_neo4j"]["orphans"] == 0
    assert all(node["properties"]["split"] == "test" for node in session.nodes.values())
    assert all("DELETE" not in query and "DROP" not in query for query, _ in session.queries)
    assert all(len(params["rows"]) <= 2 for query, params in session.queries if "rows" in params)
    assert session.builds[validated.build_id]["active"] is True


def test_property_payload_cannot_become_cypher(tmp_path):
    nodes, relationships = graph_rows()
    attack = "x') DETACH DELETE n //"
    nodes[1]["properties"]["name"] = attack
    validated = IMPORTER.load_bundle(*write_csv_graph(tmp_path, nodes, relationships))
    session = NativeNeo4jMock()
    IMPORTER.import_bundle(session, validated)
    assert all(attack not in query for query, _ in session.queries)
    assert any(node["properties"]["name"] == attack for node in session.nodes.values() if "name" in node["properties"])


def test_versions_keep_old_data_and_activate_only_verified_build(tmp_path):
    session = NativeNeo4jMock()
    original = bundle(tmp_path / "first", "build-a")
    later = bundle(tmp_path / "second", "build-b")
    IMPORTER.import_bundle(session, original)
    IMPORTER.import_bundle(session, later)
    assert len(session.nodes) == 10 and len(session.relationships) == 10
    assert session.builds["build-a"]["active"] is False
    assert session.builds["build-b"]["active"] is True
    assert IMPORTER.qualified_id("build-a", "recipe:r1") != IMPORTER.qualified_id("build-b", "recipe:r1")


def test_same_build_different_hash_is_rejected_without_changing_old_graph(tmp_path):
    session = NativeNeo4jMock()
    original = bundle(tmp_path / "first")
    IMPORTER.import_bundle(session, original)
    nodes, relationships = graph_rows()
    nodes[0]["properties"]["name"] = "changed-input"
    changed = IMPORTER.load_bundle(*write_csv_graph(tmp_path / "changed", nodes, relationships))
    before = copy.deepcopy((session.nodes, session.relationships, session.builds))
    with pytest.raises(IMPORTER.ImportValidationError, match="不同输入哈希"):
        IMPORTER.import_bundle(session, changed)
    assert (session.nodes, session.relationships, session.builds) == before


def test_modified_validated_bundle_fails_before_any_write(tmp_path):
    validated = bundle(tmp_path)
    validated.nodes[0]["properties"]["name"] = "mutated"
    session = NativeNeo4jMock()
    with pytest.raises(IMPORTER.ImportValidationError, match="输入已被修改"):
        IMPORTER.import_bundle(session, validated)
    assert not session.queries


def test_native_anomaly_blocks_activation_and_keeps_prior_build_active(tmp_path):
    session = NativeNeo4jMock()
    IMPORTER.import_bundle(session, bundle(tmp_path / "original", "original"))
    session.force_orphan_count = 1
    with pytest.raises(IMPORTER.ImportVerificationError) as caught:
        IMPORTER.import_bundle(session, bundle(tmp_path / "failed", "failed-build"))
    assert caught.value.report["activated"] is False
    assert caught.value.report["native_neo4j"]["orphans"] == 1
    assert session.builds["original"]["active"] is True
    assert session.builds["failed-build"]["active"] is False


def test_no_activate_preserves_active_build(tmp_path):
    session = NativeNeo4jMock()
    IMPORTER.import_bundle(session, bundle(tmp_path / "original", "original"))
    report = IMPORTER.import_bundle(session, bundle(tmp_path / "additional", "additional"), activate=False)
    assert report["activated"] is False
    assert session.builds["original"]["active"] is True
    assert session.builds["additional"]["status"] == "verified"


@pytest.mark.skipif(os.getenv("RECIPEGEN_RUN_NEO4J_TESTS") != "1", reason="Set RECIPEGEN_RUN_NEO4J_TESTS=1 only for an authorized dedicated Neo4j instance")
def test_optional_real_neo4j_native_counts_and_idempotence(tmp_path):
    from neo4j import GraphDatabase
    IMPORTER.load_environment(PROJECT / ".env")
    uri = os.environ.get("RECIPEGEN_NEO4J_TEST_URI") or os.environ["NEO4J_URI"]
    user = os.environ.get("RECIPEGEN_NEO4J_TEST_USER") or os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("RECIPEGEN_NEO4J_TEST_PASSWORD") or os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("RECIPEGEN_NEO4J_TEST_DATABASE") or os.environ.get("NEO4J_DATABASE", "neo4j")
    validated = bundle(tmp_path, "contract-" + uuid.uuid4().hex)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            first = IMPORTER.import_bundle(session, validated, batch_size=2, activate=False)
            second = IMPORTER.import_bundle(session, validated, batch_size=2, activate=False)
    assert first["native_neo4j"] == second["native_neo4j"]
    assert first["native_neo4j"]["nodes"] == 5
    assert first["native_neo4j"]["relationships"] == 5
    assert all(first["checks"].values())
    # Versioned contract fixture remains in its own build; it never becomes active.
