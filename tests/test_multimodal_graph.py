"""Offline provenance, adversarial, idempotence and immutable-base contracts."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import re

import pytest

from recipegen import multimodal_graph as mm
from scripts import import_multimodal_graph as cli
from scripts import import_test_graph_neo4j as base_importer

REVISION = "a" * 40
BUILD = "base-test-build"
RUN = "offline-contract-run"


def base_csv(tmp_path):
    common = {"build_id": BUILD, "split": "test", "dataset_revision": REVISION}
    def node(identifier, label, **p):
        return {"id": identifier, "label": label, "properties": {**common, **p}}
    def source(identifier, member, media=False):
        return node(identifier, "Source", source_id=identifier, source_type="huggingface_zip_media" if media else "huggingface_zip_text",
                    archive="test.zip", member=member, artifact_path=f"owner/RecipeGen/test.zip/{member}",
                    url=f"https://huggingface.co/datasets/owner/RecipeGen/resolve/{REVISION}/test.zip")
    nodes = [node("recipe:r1", "Recipe", title="Synthetic fixture only"),
             node("step:s1", "Step", text="Mix tomatoes in a pan.", order=1, source_id="source:text"),
             node("ingredient:tomato", "Ingredient", normalized_name="tomato"),
             node("tool:pan", "Tool", normalized_name="pan"),
             node("action:mix", "Action", normalized_name="mix"),
             node("image:i1", "Image", archive="test.zip", member="r1/1.jpg", source_id="source:image"),
             node("video:v1", "Video", archive="test.zip", member="r1/video.mp4", source_id="source:video"),
             source("source:text", "r1/steps.txt"), source("source:image", "r1/1.jpg", True), source("source:video", "r1/video.mp4", True)]
    pairs = [("recipe:r1", "HAS_SOURCE", "source:text"), ("recipe:r1", "HAS_STEP", "step:s1"),
             ("recipe:r1", "HAS_INGREDIENT", "ingredient:tomato"), ("step:s1", "USES_TOOL", "tool:pan"),
             ("step:s1", "HAS_ACTION", "action:mix"), ("recipe:r1", "HAS_IMAGE", "image:i1"),
             ("recipe:r1", "HAS_VIDEO", "video:v1"), ("image:i1", "HAS_SOURCE", "source:image"),
             ("video:v1", "HAS_SOURCE", "source:video")]
    edges = [{"id": f"edge:{i}", "start_id": a, "end_id": b, "type": k, "properties": dict(common)} for i, (a, k, b) in enumerate(pairs)]
    paths = []
    for name, rows, fields in [("nodes.csv", nodes, ["id", "label", "properties_json"]),
                               ("relationships.csv", edges, ["id", "start_id", "end_id", "type", "properties_json"])]:
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{k: v for k, v in row.items() if k != "properties"}, "properties_json": json.dumps(row["properties"])})
        paths.append(path)
    return base_importer.load_bundle(*paths)


def evidence(video=False, structure=False, **extra):
    media = "video:v1" if video else "image:i1"
    source = "source:video" if video else "source:image"
    return {"source_media_id": media, "source_id": source, "verified": False, "confidence": None,
            "semantics": "media_structure" if structure else "model_candidate",
            "model": "deterministic" if structure else "offline-mock-model",
            "evidence_json": json.dumps({"source_media_id": media, "source_id": source, "fixture_only": True}), **extra}


def payload():
    refs = [{"id": i, "label": k, "properties": {"reference": True}} for i, k in
            [("image:i1", "Image"), ("video:v1", "Video"), ("step:s1", "Step"),
             ("ingredient:tomato", "Ingredient"), ("tool:pan", "Tool"), ("action:mix", "Action")]]
    nodes = refs + [
        {"id": RUN, "label": "SemanticRun", "properties": {"model": "offline-mock-model", "verified": False}},
        {"id": "object:tomato", "label": "VisualObject", "properties": evidence(name="tomato")},
        {"id": "clip:1", "label": "VideoClip", "properties": evidence(True, True, start_seconds=0, end_seconds=2)},
        {"id": "clip:2", "label": "VideoClip", "properties": evidence(True, True, start_seconds=3, end_seconds=4)},
        {"id": "frame:1", "label": "Frame", "properties": evidence(True, True, timestamp_seconds=0.5)},
        {"id": "frame:2", "label": "Frame", "properties": evidence(True, True, timestamp_seconds=1.5)},
    ]
    triples = [("image:i1", "DEPICTS", "object:tomato", False, False),
               ("object:tomato", "DEPICTS_INGREDIENT", "ingredient:tomato", False, False),
               ("image:i1", "DEPICTS_TOOL", "tool:pan", False, False),
               ("video:v1", "HAS_CLIP", "clip:1", True, True), ("video:v1", "HAS_CLIP", "clip:2", True, True),
               ("clip:1", "HAS_FRAME", "frame:1", True, True), ("clip:1", "HAS_FRAME", "frame:2", True, True),
               ("clip:1", "BEFORE", "clip:2", True, True), ("frame:1", "BEFORE", "frame:2", True, True),
               ("frame:1", "SHOWS_ACTION", "action:mix", True, False),
               ("frame:1", "ALIGNED_WITH", "step:s1", True, False)]
    edges = [{"id": f"mm-edge:{i}", "start_id": a, "end_id": b, "type": k, "properties": evidence(v, s)} for i, (a, k, b, v, s) in enumerate(triples)]
    return {"schema_version": mm.SCHEMA_VERSION, "run_id": RUN, "build_id": BUILD, "split": "test",
            "dataset_revision": REVISION, "nodes": nodes, "relationships": edges}


class Record(dict):
    def data(self):
        return dict(self)


class Neo4jMock:
    def __init__(self, base):
        self.base = base
        self.nodes = {}
        self.edges = {}
        self.queries = []
        self.corrupt_post_edge = False
        self.duplicate_reference = False
        self.source_links = {(e["start_id"], e["end_id"]) for e in base.relationships if e["type"] == "HAS_SOURCE"}
        for n in base.nodes:
            identifier = base_importer.qualified_id(BUILD, n["id"])
            self.nodes[identifier] = {"kg_id": identifier, "labels": ["RecipeGen", n["label"]],
                                      "properties": mm._native_properties({**n["properties"], "kg_id": identifier, "kg_csv_id": n["id"], "kg_import_namespace": mm.BASE_NAMESPACE})}
        self.base_snapshot = copy.deepcopy(self.nodes)

    def execute_read(self, callback):
        return callback(self)

    def execute_write(self, callback):
        before = copy.deepcopy((self.nodes, self.edges))
        try:
            return callback(self)
        except Exception:
            self.nodes, self.edges = before
            raise

    def run(self, query, params):
        self.queries.append((query, copy.deepcopy(params)))
        if query == mm.ACTIVE_BUILD:
            return [Record(content_sha256=self.base.content_sha256)]
        if query == mm.REFERENCES:
            found = []
            for row in params["rows"]:
                for n in self.nodes.values():
                    p = n["properties"]
                    if p.get("kg_csv_id") == row["id"] and row["label"] in n["labels"] and p.get("kg_import_namespace") == mm.BASE_NAMESPACE and p.get("build_id") == params["build_id"] and p.get("split") == "test" and p.get("dataset_revision") == params["dataset_revision"]:
                        found.append(Record(id=row["id"], labels=n["labels"], properties=copy.deepcopy(p)))
            if self.duplicate_reference and found:
                found.append(copy.deepcopy(found[0]))
            return found
        if query == mm.SOURCE_LINKS:
            return [Record(media_id=r["media_id"], source_id=r["source_id"], count=1)
                    for r in params["rows"] if (r["media_id"], r["source_id"]) in self.source_links]
        if query == mm.EXISTING_NODES:
            return [Record(copy.deepcopy(n)) for k, n in self.nodes.items() if k in params["ids"]]
        if query == mm.EXISTING_EDGES:
            return [Record(copy.deepcopy(e)) for k, e in self.edges.items() if k in params["ids"]]
        if query.startswith("// recipegen-mm:merge-nodes"):
            label = re.search(r"RecipeGen:`([^`]+)`", query)[1]
            result = []
            for row in params["rows"]:
                n = self.nodes.setdefault(row["kg_id"], {"kg_id": row["kg_id"], "labels": ["RecipeGen", label], "properties": mm._native_properties(copy.deepcopy(row["properties"]))})
                result.append(Record(copy.deepcopy(n)))
            return result
        if query.startswith("// recipegen-mm:merge-edges"):
            kind = re.search(r"\[e:`([^`]+)`", query)[1]
            result = []
            for row in params["rows"]:
                if row["start_kg_id"] not in self.nodes or row["end_kg_id"] not in self.nodes:
                    continue
                e = self.edges.setdefault(row["kg_id"], {"kg_id": row["kg_id"], "type": kind,
                    "start_kg_id": row["start_kg_id"], "end_kg_id": row["end_kg_id"], "properties": mm._native_properties(copy.deepcopy(row["properties"]))})
                result.append(Record(copy.deepcopy(e)))
            return result
        if query == mm.RUN_NODES:
            return [Record(copy.deepcopy(n)) for n in self.nodes.values() if n["properties"].get("kg_import_namespace") == mm.NAMESPACE and n["properties"].get("run_id") == params["run_id"] and n["properties"].get("build_id") == params["build_id"] and n["properties"].get("split") == "test"]
        if query == mm.RUN_EDGES:
            result = [Record(copy.deepcopy(e)) for e in self.edges.values() if e["properties"].get("run_id") == params["run_id"]]
            if self.corrupt_post_edge and result:
                result[0]["end_kg_id"] = "nonexistent-native-endpoint"
            return result
        raise AssertionError(query)


def test_valid_bundle_hash_order_invariant_scope_and_source(tmp_path):
    base = base_csv(tmp_path)
    data = payload()
    first = mm.validate_bundle(data, base)
    reversed_data = copy.deepcopy(data)
    reversed_data["nodes"].reverse()
    reversed_data["relationships"].reverse()
    second = mm.validate_bundle(reversed_data, base)
    assert first.content_sha256 == second.content_sha256
    assert first.summary()["extension_nodes"] == 6
    assert len(first.source_references) == 2
    assert all(n["properties"]["split"] == "test" for n in first.nodes)
    assert mm.qualified_id(BUILD, RUN, "object:tomato") != mm.qualified_id(BUILD, "another-run", "object:tomato")
    assert mm.qualified_id("a:b", "c", "d") != mm.qualified_id("a", "b:c", "d")


@pytest.mark.parametrize("mutation,match", [
    ("wrong_split", "split"), ("wrong_build", "CSV"), ("wrong_revision", "CSV"),
    ("unknown_label", "白名单"), ("relation_injection", "白名单"), ("dangling", "dangling"),
    ("duplicate_node", "重复"), ("duplicate_edge", "重复"), ("missing_run", "SemanticRun"), ("two_runs", "SemanticRun"),
    ("reference_rewrite", "重写"), ("reference_missing_flag", "reference"), ("unknown_reference", "CSV"),
    ("nested_property", "嵌套"), ("reserved_property", "保留"), ("node_scope", "范围"),
    ("edge_scope", "范围"), ("source_fake", "source_id"), ("media_fake", "source_media_id"),
    ("evidence_fake", "evidence_json"), ("evidence_train", "范围"), ("evidence_member", "member"),
    ("verified_fact", "false"), ("confidence_claim", "confidence"), ("confidence_bool", "confidence"),
    ("wrong_end_label", "端点"), ("clip_reversed", "end_seconds"), ("frame_bool", "秒数"),
    ("frame_outside", "范围"), ("before_reversed", "时间顺序"), ("before_overlap", "时间顺序"),
    ("clip_from_image", "Video"), ("edge_wrong_media", "来源媒体"), ("similarity_bool", "similarity_score"),
])
def test_adversarial_bundle_rejected_offline(tmp_path, mutation, match):
    base, data = base_csv(tmp_path), payload()
    objects = {n["id"]: n for n in data["nodes"]}
    obj, clip, frame = objects["object:tomato"], objects["clip:1"], objects["frame:1"]
    if mutation == "wrong_split": data["split"] = "train"
    if mutation == "wrong_build": data["build_id"] = "other"
    if mutation == "wrong_revision": data["dataset_revision"] = "b" * 40
    if mutation == "unknown_label": obj["label"] = "VisualObject`) DELETE n //"
    if mutation == "relation_injection": data["relationships"][0]["type"] = 'HAS_STEP"] DELETE e //'
    if mutation == "dangling": data["relationships"][0]["end_id"] = "missing"
    if mutation == "duplicate_node": data["nodes"].append(copy.deepcopy(obj))
    if mutation == "duplicate_edge": data["relationships"].append(copy.deepcopy(data["relationships"][0]))
    if mutation == "missing_run": data["nodes"] = [n for n in data["nodes"] if n["label"] != "SemanticRun"]
    if mutation == "two_runs": data["nodes"].append({"id": "another-run", "label": "SemanticRun", "properties": {}})
    if mutation == "reference_rewrite": data["nodes"][0]["properties"]["recognition_status"] = "verified"
    if mutation == "reference_missing_flag": data["nodes"][0]["properties"] = {}
    if mutation == "unknown_reference": data["nodes"][0]["id"] = "image:unknown"
    if mutation == "nested_property": obj["properties"]["object"] = {"name": "tomato"}
    if mutation == "reserved_property": obj["properties"]["kg_id"] = "original"
    if mutation == "node_scope": obj["properties"]["split"] = "validation"
    if mutation == "edge_scope": data["relationships"][0]["properties"]["run_id"] = "other"
    if mutation == "source_fake": obj["properties"]["source_id"] = "source:text"
    if mutation == "media_fake": obj["properties"]["source_media_id"] = "step:s1"
    if mutation == "evidence_fake": obj["properties"]["evidence_json"] = '{}'
    if mutation == "evidence_train": obj["properties"]["evidence_json"] = json.dumps({"source_media_id": "image:i1", "source_id": "source:image", "split": "train"})
    if mutation == "evidence_member": obj["properties"]["evidence_json"] = json.dumps({"source_media_id": "image:i1", "source_id": "source:image", "member": "train/r1.jpg"})
    if mutation == "verified_fact": obj["properties"]["verified"] = True
    if mutation == "confidence_claim": obj["properties"]["confidence"] = 0.99
    if mutation == "confidence_bool": obj["properties"]["confidence"] = False
    if mutation == "wrong_end_label": data["relationships"][0]["end_id"] = "ingredient:tomato"
    if mutation == "clip_reversed": clip["properties"]["end_seconds"] = 0
    if mutation == "frame_bool": frame["properties"]["timestamp_seconds"] = True
    if mutation == "frame_outside": frame["properties"]["timestamp_seconds"] = 10
    if mutation == "before_reversed": objects["frame:2"]["properties"]["timestamp_seconds"] = 0.25
    if mutation == "before_overlap": objects["clip:2"]["properties"]["start_seconds"] = 1
    if mutation == "clip_from_image": clip["properties"].update(evidence(False, True))
    if mutation == "edge_wrong_media": data["relationships"][0]["properties"].update(evidence(True))
    if mutation == "similarity_bool": obj["properties"]["similarity_score"] = True
    with pytest.raises(mm.MultimodalValidationError, match=match):
        mm.validate_bundle(data, base)


def test_strict_json_nonfinite_and_duplicate_keys():
    for text in ['{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}']:
        with pytest.raises(mm.MultimodalValidationError):
            mm.parse_json(text)


def test_idempotent_import_never_changes_existing_facts(tmp_path):
    base = base_csv(tmp_path)
    bundle = mm.validate_bundle(payload(), base)
    session = Neo4jMock(base)
    first = mm.import_bundle(session, bundle, batch_size=2)
    counts = len(session.nodes), len(session.edges)
    second = mm.import_bundle(session, bundle, batch_size=3)
    assert first["status"] == second["status"] == "verified"
    assert counts == (len(session.nodes), len(session.edges)) == (16, 11)
    assert {k: session.nodes[k] for k in session.base_snapshot} == session.base_snapshot
    assert all("DELETE" not in q and "DROP" not in q and "ON MATCH SET" not in q for q, _ in session.queries)
    assert all(p["properties"].get("confidence") is None for p in session.edges.values())
    read_queries = {mm.ACTIVE_BUILD, mm.REFERENCES, mm.SOURCE_LINKS, mm.EXISTING_NODES, mm.EXISTING_EDGES, mm.RUN_NODES, mm.RUN_EDGES}
    assert all("MERGE" not in q and "SET" not in q for q in read_queries)


def test_same_run_changed_content_rejected_without_updates(tmp_path):
    base, data = base_csv(tmp_path), payload()
    session = Neo4jMock(base)
    mm.import_bundle(session, mm.validate_bundle(data, base))
    snapshot = copy.deepcopy((session.nodes, session.edges))
    next(n for n in data["nodes"] if n["label"] == "VisualObject")["properties"]["name"] = "invented changed object"
    with pytest.raises(mm.MultimodalValidationError, match="不同内容"):
        mm.import_bundle(session, mm.validate_bundle(data, base))
    assert (session.nodes, session.edges) == snapshot


@pytest.mark.parametrize("failure", ["missing_native_reference", "changed_original_text", "source_changed", "duplicate_native_reference", "wrong_active_hash", "missing_native_source_link"])
def test_native_preflight_failure_has_no_extension_writes(tmp_path, failure):
    base = base_csv(tmp_path)
    bundle = mm.validate_bundle(payload(), base)
    session = Neo4jMock(base)
    if failure == "missing_native_reference":
        del session.nodes[base_importer.qualified_id(BUILD, "image:i1")]
    if failure == "changed_original_text":
        session.nodes[base_importer.qualified_id(BUILD, "step:s1")]["properties"]["text"] = "Changed original fact"
    if failure == "source_changed":
        session.nodes[base_importer.qualified_id(BUILD, "source:image")]["properties"]["member"] = "wrong.jpg"
    if failure == "duplicate_native_reference": session.duplicate_reference = True
    if failure == "wrong_active_hash": session.base = copy.copy(base); object.__setattr__(session.base, "content_sha256", "wrong")
    if failure == "missing_native_source_link": session.source_links.remove(("image:i1", "source:image"))
    snapshot = copy.deepcopy((session.nodes, session.edges))
    with pytest.raises(mm.MultimodalValidationError):
        mm.import_bundle(session, bundle)
    assert (session.nodes, session.edges) == snapshot
    assert not any(q.startswith("// recipegen-mm:merge") for q, _ in session.queries)


def test_poststats_endpoint_mismatch_reported_unverified(tmp_path):
    base = base_csv(tmp_path)
    session = Neo4jMock(base)
    session.corrupt_post_edge = True
    with pytest.raises(mm.MultimodalVerificationError) as raised:
        mm.import_bundle(session, mm.validate_bundle(payload(), base))
    assert raised.value.report["status"] == "unverified"
    assert raised.value.report["checks"]["extension_edge_ids_endpoints_properties_and_time"] is False


def test_cli_default_and_validate_only_do_not_access_connection(tmp_path, monkeypatch, capsys):
    base = base_csv(tmp_path)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("offline validation must never read credentials or connect")
    monkeypatch.setattr(cli, "connect_and_import", forbidden)
    args = ["--bundle", str(path), "--nodes-csv", str(base.nodes_path), "--relationships-csv", str(base.relationships_path)]
    for suffix in ([], ["--validate-only"]):
        assert cli.main(args + suffix) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "validated"
        assert report["connected_to_neo4j"] is False
        assert report["validation_only"] is True


def test_cli_driver_exception_redacted(tmp_path, monkeypatch, capsys):
    base = base_csv(tmp_path)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")
    def broken(*args, **kwargs):
        raise RuntimeError("password=do-not-print-this-test-secret")
    monkeypatch.setattr(cli, "connect_and_import", broken)
    assert cli.main(["--bundle", str(path), "--nodes-csv", str(base.nodes_path), "--relationships-csv", str(base.relationships_path), "--import"]) == 1
    output = capsys.readouterr().out
    assert "do-not-print-this-test-secret" not in output
    assert json.loads(output)["status"] == "unverified"


def observation_payload():
    data = payload()
    data["nodes"].append({"id": "observation:1", "label": "VisualObservation", "properties": evidence(
        caption="Synthetic fixture: a tomato in a pan", raw_json=json.dumps({"objects": ["tomato", "pan"], "relations": [{"predicate": "unknown-raw-predicate"}]}),
        embedding=[0.125] * 512, embedding_model="offline-fixture-clip")})
    data["nodes"].append({"id": "object:pan", "label": "VisualObject", "properties": evidence(name="pan")})
    data["relationships"].extend([
        {"id": "observation-link", "start_id": "image:i1", "end_id": "observation:1", "type": "HAS_OBSERVATION", "properties": evidence(structure=True)},
        {"id": "observation-object", "start_id": "observation:1", "end_id": "object:pan", "type": "DEPICTS", "properties": evidence()},
        {"id": "observation-step", "start_id": "observation:1", "end_id": "step:s1", "type": "ALIGNED_WITH", "properties": evidence(similarity_score=0.25)},
    ])
    for kind in ("IN", "HOLDING", "USING", "ON", "BESIDE", "MIXED_WITH"):
        data["relationships"].append({"id": f"spatial:{kind}", "start_id": "object:tomato", "end_id": "object:pan", "type": kind, "properties": evidence()})
    return data


def test_visual_observation_embeddings_raw_output_and_spatial_whitelist(tmp_path):
    base = base_csv(tmp_path)
    indexed = mm.prepare_base_graph(base)
    assert mm.prepare_base_graph(indexed) is indexed
    bundle = mm.validate_bundle(observation_payload(), indexed)
    assert bundle.summary()["nodes_by_label"]["VisualObservation"] == 1
    session = Neo4jMock(base)
    report = mm.import_bundle(session, bundle)
    assert report["status"] == "verified"
    observation = next(n for n in session.nodes.values() if "VisualObservation" in n["labels"])
    assert len(observation["properties"]["embedding"]) == 512
    assert "unknown-raw-predicate" in observation["properties"]["raw_json"]
    assert {k: session.nodes[k] for k in session.base_snapshot} == session.base_snapshot


@pytest.mark.parametrize("change", ["dimension", "zero", "bool", "model", "raw_array", "raw_duplicates"])
def test_invalid_observation_embedding_or_raw_json_rejected(tmp_path, change):
    base, data = base_csv(tmp_path), observation_payload()
    p = next(n for n in data["nodes"] if n["label"] == "VisualObservation")["properties"]
    if change == "dimension": p["embedding"] = [0.125] * 511
    if change == "zero": p["embedding"] = [0.0] * 512
    if change == "bool": p["embedding"] = [True] * 512
    if change == "model": del p["embedding_model"]
    if change == "raw_array": p["raw_json"] = '["invented", "schema"]'
    if change == "raw_duplicates": p["raw_json"] = '{"objects":[],"objects":["tomato"]}'
    with pytest.raises(mm.MultimodalValidationError):
        mm.validate_bundle(data, base)


def streaming_project(tmp_path):
    project = tmp_path / "project"
    csv_dir = project / "data/test_graph/neo4j_import"
    csv_dir.mkdir(parents=True)
    (project / "reports").mkdir()
    base = base_csv(csv_dir)
    counts = base.summary()
    build = {"build_id": BUILD, "split": "test", "dataset_revision": REVISION,
             "node_count": counts["nodes"], "relationship_count": counts["relationships"],
             "node_counts": counts["nodes_by_label"], "relationship_counts": counts["relationships_by_type"]}
    deployment = {"status": "verified", "validation_only": False, "connected_to_neo4j": True, "activated": True,
                  "input": counts, "checks": {k: True for k in ("node_counts", "relationship_counts", "orphans", "out_of_scope_endpoints", "invalid_node_ids", "invalid_relationship_ids", "recipes_without_text_source")}}
    corpus = {"status": "complete", "complete_test_scope": True, "revision": REVISION, "split": "test",
              "archives": [{"archive": name, "status": "complete", "revision": REVISION, "split": "test"} for name in ("test.zip", "test-video.zip")]}
    for name, data in [("data/test_graph/build-report.json", build), ("reports/user-neo4j-deployment.json", deployment), ("data/test_graph/manifest.json", corpus)]:
        (project / name).write_text(json.dumps(data), encoding="utf-8")
    refresh_manifest(project)
    return project, base


def refresh_manifest(project):
    files = ["data/test_graph/neo4j_import/nodes.csv", "data/test_graph/neo4j_import/relationships.csv",
             "reports/user-neo4j-deployment.json", "data/test_graph/build-report.json", "data/test_graph/manifest.json"]
    checks = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in files}
    (project / "reports/source-manifest.json").write_text(json.dumps(checks), encoding="utf-8")


def test_streaming_index_reuses_prior_verified_hash_without_canonical_edges(tmp_path, monkeypatch):
    project, base = streaming_project(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("streaming index must not load the full edge bundle or credentials")
    monkeypatch.setattr(base_importer, "load_bundle", forbidden)
    index = mm.load_base_index(project)
    assert index.content_sha256 == base.content_sha256
    assert index.provenance["canonical_hash_recomputed"] is False
    assert index.nodes_by_id == {n["id"]: n for n in base.nodes}
    assert index.source_links["image:i1"] == {"source:image"}
    assert mm.validate_bundle(payload(), index).summary()["base_validation_method"] == "previous_full_validation_plus_streamed_csv_sha256"
    assert mm.import_bundle(Neo4jMock(base), mm.validate_bundle(payload(), index))["status"] == "verified"


@pytest.mark.parametrize("change", ["csv_byte_tamper", "deployment_unverified", "wrong_revision", "wrong_split", "incomplete_archive", "wrong_count", "missing_hash"])
def test_streaming_trust_boundary_rejected(tmp_path, change):
    project, base = streaming_project(tmp_path)
    if change == "csv_byte_tamper":
        with (project / "data/test_graph/neo4j_import/nodes.csv").open("a") as stream:
            stream.write("unaudited change\n")
    elif change == "missing_hash":
        (project / "reports/source-manifest.json").write_text('{}')
    else:
        name = "reports/user-neo4j-deployment.json" if change == "deployment_unverified" else "data/test_graph/build-report.json" if change in {"wrong_revision", "wrong_split", "wrong_count"} else "data/test_graph/manifest.json"
        p = project / name
        d = json.loads(p.read_text())
        if change == "deployment_unverified": d["status"] = "failed"
        if change == "wrong_revision": d["dataset_revision"] = "b" * 40
        if change == "wrong_split": d["split"] = "train"
        if change == "incomplete_archive": d["archives"][1]["status"] = "incomplete"
        if change == "wrong_count": d["node_count"] += 1
        p.write_text(json.dumps(d))
        refresh_manifest(project)
    with pytest.raises(mm.MultimodalValidationError):
        mm.load_base_index(project)


def test_cli_standard_csvs_use_streaming_index_without_full_bundle(tmp_path, monkeypatch, capsys):
    project, base = streaming_project(tmp_path)
    path = project / "mm.json"
    path.write_text(json.dumps(payload()))
    monkeypatch.setattr(cli, "PROJECT_ROOT", project)
    def forbidden(*args, **kwargs):
        raise AssertionError("standard CSV mode must not load full graph or connect")
    monkeypatch.setattr(cli, "load_base_graph", forbidden)
    monkeypatch.setattr(cli, "connect_and_import", forbidden)
    assert cli.main(["--bundle", str(path), "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["input"]["base_validation_method"] == "previous_full_validation_plus_streamed_csv_sha256"
