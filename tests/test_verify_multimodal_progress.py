"""Audit acceptance tests use public synthetic artifacts and never contact Neo4j."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import socket

import pytest

from scripts import verify_multimodal_progress as audit
from scripts import import_test_graph_neo4j as base_importer
from scripts import query_test_graph as base_query


REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
BUILD = "synthetic-base-build"
PIPELINE = "synthetic-pipeline"
BASE_HASH = "a" * 64
COUNT_KEYS = ("images", "videos", "clips", "frames", "observations", "aligned_media",
              "extension_nodes", "extension_relationships")
BASE_CHECKS = ("node_counts", "relationship_counts", "orphans", "out_of_scope_endpoints",
               "invalid_node_ids", "invalid_relationship_ids", "recipes_without_text_source")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sum_counts(rows):
    return {key: sum(row["counts"][key] for row in rows.values()) for key in COUNT_KEYS}


@pytest.fixture
def verified_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "EXPECTED_FULL_COUNTS",
                        {"recipes": 2, "orphan_media": 1, "images": 2, "videos": 1})
    graph = tmp_path / "data/test_graph"
    records = [{"id": "recipe:r1", "split": "test", "source": {"archive": "test.zip", "split": "test", "revision": REVISION}},
               {"id": "recipe:r2", "split": "test", "source": {"archive": "test-video.zip", "split": "test", "revision": REVISION}}]
    associations = [{"media_id": "image:i1", "recipe_id": "recipe:r1", "archive": "test.zip", "member": "r1/1.jpg"},
                    {"media_id": "video:v1", "recipe_id": "recipe:r1", "archive": "test-video.zip", "member": "r1/1.mp4"},
                    {"media_id": "image:orphan", "recipe_id": None, "archive": "test.zip", "member": "orphan/1.jpg"}]
    write_jsonl(graph / "records.jsonl", records)
    write_jsonl(graph / "media_associations.jsonl", associations)
    labels = {"Recipe": 2, "Image": 2, "Video": 1, "Source": 3}
    rels = {"HAS_SOURCE": 3, "HAS_IMAGE": 1, "HAS_VIDEO": 1}
    scope = {"build_id": BUILD, "split": "test", "dataset_revision": REVISION}
    build = {**scope, "archives": ["test.zip", "test-video.zip"], "records_seen": 2,
             "node_counts": labels, "relationship_counts": rels,
             "media_counts": {"image": 2, "video": 1}, "unassociated_media": [associations[-1]]}
    deployment = {"status": "verified", "activated": True, "connected_to_neo4j": True,
                  "validation_only": False, "checks": dict.fromkeys(BASE_CHECKS, True),
                  "input": {**scope, "content_sha256": BASE_HASH,
                            "nodes_by_label": labels, "relationships_by_type": rels}}
    write_json(graph / "build-report.json", build)
    write_json(tmp_path / "reports/user-neo4j-deployment.json", deployment)
    protected = ["data/test_graph/records.jsonl", "data/test_graph/media_associations.jsonl",
                 "data/test_graph/build-report.json", "reports/user-neo4j-deployment.json"]
    write_json(tmp_path / "reports/source-manifest.json",
               {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in protected})
    signature = {"schema": "recipegen-mm-v1", **scope, "models": {},
                 "graph_target": {"uri": "neo4j://fixture.invalid:7687", "database": "neo4j"}}
    run_dir = tmp_path / "data/multimodal/runs" / PIPELINE
    write_json(run_dir / "signature.json", signature)
    specs = {
        "recipe:r1": ([1, 1, 1, 1, 2, 2, 6, 6],
                      {"SemanticRun": 1, "VisualObservation": 2, "VideoClip": 1, "Frame": 1, "VisualObject": 1},
                      {"HAS_CLIP": 1, "HAS_FRAME": 1, "HAS_OBSERVATION": 2, "ALIGNED_WITH": 2}),
        "recipe:r2": ([0, 0, 0, 0, 0, 0, 1, 0], {"SemanticRun": 1}, {}),
        "image:orphan": ([1, 0, 0, 0, 1, 0, 2, 1], {"SemanticRun": 1, "VisualObservation": 1}, {"HAS_OBSERVATION": 1}),
    }
    state_records, reports, native_runs = {}, {}, {}
    for index, (identifier, (numbers, node_counts, edge_counts)) in enumerate(specs.items()):
        run_id = f"semantic-{index}"
        counts = dict(zip(COUNT_KEYS, numbers))
        report_path = run_dir / f"item-{index}/import-report.json"
        summary = {**scope, "schema_version": "recipegen-mm-v1", "run_id": run_id, "content_sha256": str(index + 1) * 64,
                   "base_content_sha256": BASE_HASH, "nodes_by_label": node_counts,
                   "relationships_by_type": edge_counts, "extension_nodes": counts["extension_nodes"],
                   "extension_relationships": counts["extension_relationships"]}
        report = {"status": "verified", "connected_to_neo4j": True, "validation_only": False,
                  "input": summary, "checks": {"references_unchanged": True,
                    "extension_node_ids_properties_and_scope": True,
                    "extension_edge_ids_endpoints_properties_and_time": True},
                  "native_neo4j": {"extension_nodes": counts["extension_nodes"],
                                   "extension_relationships": counts["extension_relationships"]}}
        write_json(report_path, report)
        reports[identifier] = (report_path, report)
        state_records[identifier] = {"status": "verified", "run_id": run_id, "counts": counts,
                                     "report": str(report_path.relative_to(tmp_path)),
                                     "observation_ids": [f"observation:{index}:{n}" for n in range(counts["observations"])]}
        native_runs[run_id] = copy.deepcopy(summary)
        native_runs[run_id]["source_media_ids"] = [row["media_id"] for row in associations
                                                  if row["recipe_id"] == identifier or row["media_id"] == identifier]
        native_runs[run_id]["aligned_media"] = counts["aligned_media"]
    state = {"pipeline_run_id": PIPELINE, "signature": signature, "records": state_records}
    state_path = run_dir / "state.json"
    write_json(state_path, state)
    progress = {"pipeline_run_id": PIPELINE, "status": "completed", "scope": "full_test",
                "expected_original_recipe_records": 2, "expected_orphan_media": 1,
                "expected_images": 2, "expected_videos": 1, "completed_work_items": 3,
                "total_work_items": 3, "failed_work_items": 0,
                "processed_counts": sum_counts(state_records), "completed_full_scope": True,
                "neo4j_import_requested": True}
    progress_path = tmp_path / "reports/multimodal-progress.json"
    write_json(progress_path, progress)
    native = {"connected_to_neo4j": True,
              "active_build": {**scope, "content_sha256": BASE_HASH, "status": "verified", "active": True},
              "base": {"nodes_by_label": labels, "relationships_by_type": rels,
                       **dict.fromkeys(BASE_CHECKS[2:], 0)},
              "runs": native_runs,
              "anomalies": {"invalid_scope": 0, "invalid_ids": 0, "invalid_endpoints": 0, "invalid_candidates": 0}}
    return {"root": tmp_path, "state": state, "state_path": state_path,
            "progress": progress, "progress_path": progress_path, "reports": reports, "native": native}


def test_complete_scope_requires_native_recheck_and_offline_never_reads_credentials(verified_artifacts, monkeypatch):
    f = verified_artifacts
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != ".env" and ".runtime" not in path.parts, "Offline audit must not read private connection config"
        return original_open(path, *args, **kwargs)

    def no_network(*args, **kwargs):
        pytest.fail("Offline audit must not open a network connection")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(audit, "_module", no_network)
    offline = audit.audit_progress(f["root"])
    assert offline["status"] == "incomplete"
    assert offline["completed_full_scope"] is False
    assert offline["current_scope_verified"] is False
    actual = audit.audit_progress(f["root"], native=f["native"])
    assert actual["status"] == "verified"
    assert actual["completed_full_scope"] is True
    assert actual["current_scope_verified"] is True


@pytest.mark.parametrize("defect", ["missing_state", "missing_report", "failed_item", "bool_count", "wrong_scope", "unsafe_report", "failed_report_check"])
def test_forged_completed_progress_cannot_replace_per_item_evidence(verified_artifacts, defect):
    f = verified_artifacts
    item = f["state"]["records"]["recipe:r1"]
    report_path, report = f["reports"]["recipe:r1"]
    if defect == "missing_state":
        f["state_path"].unlink()
    elif defect == "missing_report":
        report_path.unlink()
    elif defect == "failed_item":
        item["status"] = "failed"
    elif defect == "bool_count":
        item["counts"]["images"] = True
    elif defect == "wrong_scope":
        report["input"]["split"] = "train"
        write_json(report_path, report)
    elif defect == "unsafe_report":
        outside = f["root"].parent / "outside-import-report.json"
        write_json(outside, report)
        item["report"] = str(outside)
    elif defect == "failed_report_check":
        report["checks"]["references_unchanged"] = False
        write_json(report_path, report)
    if defect != "missing_state":
        write_json(f["state_path"], f["state"])
    result = audit.audit_progress(f["root"], native=f["native"])
    assert result["status"] == "incomplete"
    assert result["completed_full_scope"] is False
    assert result["current_scope_verified"] is False


def test_native_statistics_use_only_parameterized_read_queries(verified_artifacts, monkeypatch):
    native = verified_artifacts["native"]
    queries = {base_query.ACTIVE_BUILD_QUERY: [native["active_build"]],
               base_importer.NODE_COUNTS: [{"label": label, "count": count} for label, count in native["base"]["nodes_by_label"].items()],
               base_importer.RELATIONSHIP_COUNTS: [{"type": kind, "count": count} for kind, count in native["base"]["relationships_by_type"].items()],
               base_importer.VERIFY_ANOMALIES: [{key: native["base"][key] for key in BASE_CHECKS[2:]}],
               audit.RUN_QUERY: [{"run_id": run_id, **{key: value[key] for key in ("build_id", "split", "dataset_revision", "content_sha256")}}
                                 for run_id, value in native["runs"].items()],
               audit.NODE_QUERY: [{"run_id": run_id, "label": label, "count": count}
                                 for run_id, value in native["runs"].items() for label, count in value["nodes_by_label"].items()],
               audit.EDGE_QUERY: [{"run_id": run_id, "type": kind, "count": count}
                                 for run_id, value in native["runs"].items() for kind, count in value["relationships_by_type"].items()],
               audit.MEDIA_QUERY: [{"run_id": run_id, "source_media_ids": value["source_media_ids"]}
                                  for run_id, value in native["runs"].items()],
               audit.ALIGNMENT_QUERY: [{"run_id": run_id, "aligned_media": value["aligned_media"]}
                                      for run_id, value in native["runs"].items()],
               audit.EXTENSION_ANOMALIES: [native["anomalies"]]}
    seen = []

    class Record:
        def __init__(self, value):
            self.value = value

        def data(self):
            return self.value

    class Transaction:
        def run(self, cypher, **parameters):
            assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV)\b", cypher, re.I)
            assert parameters["pipeline_id"] == PIPELINE
            assert parameters["build_id"] == BUILD
            assert parameters["dataset_revision"] == REVISION
            if cypher in (audit.NODE_QUERY, audit.EDGE_QUERY, audit.MEDIA_QUERY, audit.ALIGNMENT_QUERY, audit.EXTENSION_ANOMALIES):
                assert set(parameters["run_ids"]) == set(native["runs"])
            seen.append(cypher)
            return [Record(value) for value in queries[cypher]]

    class Session:
        def execute_read(self, callback):
            return callback(Transaction())

        def execute_write(self, *args, **kwargs):
            pytest.fail("Audit must never execute a write transaction")

    def public_query_module(root, filename, name):
        return base_importer if filename == "import_test_graph_neo4j.py" else base_query

    monkeypatch.setattr(audit, "_module", public_query_module)
    result = audit.read_native_stats(Session(), verified_artifacts["root"], PIPELINE, BUILD)
    assert len(seen) == len(queries) == 10
    assert set(audit.READ_ONLY_QUERIES) <= set(seen)
    assert result["connected_to_neo4j"] is True
    assert result["base"] == native["base"]
    assert result["anomalies"] == native["anomalies"]
    for run_id, value in result["runs"].items():
        assert value["extension_nodes"] == native["runs"][run_id]["extension_nodes"]
        assert value["extension_relationships"] == native["runs"][run_id]["extension_relationships"]
        assert value["source_media_ids"] == native["runs"][run_id]["source_media_ids"]
        assert value["aligned_media"] == native["runs"][run_id]["aligned_media"]


def test_original_source_hash_change_blocks_completion(verified_artifacts):
    f = verified_artifacts
    path = f["root"] / "data/test_graph/records.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    result = audit.audit_progress(f["root"], native=f["native"])
    assert result["status"] == "incomplete"
    assert result["completed_full_scope"] is False
    assert result["audit_status"] == "inconsistent"


def test_bounded_recipe_can_be_verified_without_claiming_full_completion(verified_artifacts):
    f = verified_artifacts
    item = f["state"]["records"]["recipe:r1"]
    f["state"]["records"] = {"recipe:r1": item}
    write_json(f["state_path"], f["state"])
    f["native"]["runs"] = {item["run_id"]: f["native"]["runs"][item["run_id"]]}
    f["progress"].update(scope="bounded_review", expected_original_recipe_records=1, expected_orphan_media=0,
                         expected_images=1, expected_videos=1, completed_work_items=1,
                         total_work_items=1, processed_counts=item["counts"], completed_full_scope=False)
    write_json(f["progress_path"], f["progress"])
    result = audit.audit_progress(f["root"], recipe_id="recipe:r1", native=f["native"])
    assert result["status"] == "incomplete"
    assert result["completed_full_scope"] is False
    assert result["current_scope_verified"] is True


@pytest.mark.parametrize("defect", ["native_scope", "native_count", "native_anomaly", "native_media", "progress_count"])
def test_native_and_progress_counts_and_scope_must_agree(verified_artifacts, defect):
    f = verified_artifacts
    run_id = f["state"]["records"]["recipe:r1"]["run_id"]
    if defect == "native_scope":
        f["native"]["runs"][run_id]["build_id"] = "another-build"
    elif defect == "native_count":
        f["native"]["runs"][run_id]["nodes_by_label"]["VisualObservation"] += 1
    elif defect == "native_anomaly":
        f["native"]["anomalies"]["invalid_endpoints"] = 1
    elif defect == "native_media":
        f["native"]["runs"][run_id]["source_media_ids"] = ["image:another", "video:another"]
    else:
        f["progress"]["processed_counts"]["images"] += 1
        write_json(f["progress_path"], f["progress"])
    result = audit.audit_progress(f["root"], native=f["native"])
    assert result["status"] == "incomplete"
    assert result["completed_full_scope"] is False


@pytest.mark.parametrize("key, forged_count", [("frames", 100), ("aligned_media", 0)])
def test_claimed_structure_and_alignment_counts_must_equal_native_counts(verified_artifacts, key, forged_count):
    f = verified_artifacts
    f["state"]["records"]["recipe:r1"]["counts"][key] = forged_count
    f["progress"]["processed_counts"] = sum_counts(f["state"]["records"])
    write_json(f["state_path"], f["state"])
    write_json(f["progress_path"], f["progress"])
    result = audit.audit_progress(f["root"], native=f["native"])
    assert result["status"] == "incomplete"
    assert result["completed_full_scope"] is False


def image_artifacts(f):
    """同一完整原始来源，只将新批次的执行与原生媒体覆盖投影为 Image。"""
    f["state"]["signature"]["processing_modalities"] = ["image"]
    write_json(f["state_path"].parent/"signature.json", f["state"]["signature"])
    for identifier, item in f["state"]["records"].items():
        native = f["native"]["runs"][item["run_id"]]
        image_ids = [mid for mid in native["source_media_ids"] if mid.startswith("image:")]
        item["observation_ids"] = image_ids
        if item["counts"]["videos"]:
            item["counts"].update(videos=0,clips=0,frames=0,observations=len(image_ids),
                                  aligned_media=len(image_ids),extension_nodes=2,extension_relationships=2)
            path, report = f["reports"][identifier]
            report["input"].update(nodes_by_label={"SemanticRun":1,"VisualObservation":1},
                                   relationships_by_type={"HAS_OBSERVATION":1,"ALIGNED_WITH":1},
                                   extension_nodes=2,extension_relationships=2)
            report["native_neo4j"] = {"extension_nodes":2,"extension_relationships":2}
            write_json(path, report)
            native.update(copy.deepcopy(report["input"]),aligned_media=len(image_ids))
        native["source_media_ids"] = image_ids
    write_json(f["state_path"],f["state"])
    f["progress"].update(scope="full_test_images",processing_modalities=["image"],expected_videos=0,
                         processed_counts=sum_counts(f["state"]["records"]),completed_full_scope=False,
                         completed_image_scope=True)
    write_json(f["progress_path"],f["progress"])
    return f


def test_image_full_scope_retains_every_recipe_and_native_base_videos(verified_artifacts):
    f = image_artifacts(verified_artifacts)
    offline = audit.audit_progress(f["root"])
    assert offline["audit_status"] == "verified"
    assert offline["completed_image_scope"] is False and offline["completed_full_scope"] is False
    actual = audit.audit_progress(f["root"],native=f["native"])
    assert actual["status"] == "verified" and actual["completed_image_scope"] is True
    assert actual["completed_full_scope"] is False and actual["current_scope_verified"] is True
    assert actual["expected_full"]["videos"] == 1  # Original complete source inventory is unchanged.
    assert actual["current_scope_expected"] == {"recipes":2,"orphan_media":1,"images":2,"videos":0}
    assert actual["total_work_items"] == 3  # Zero-image recipe:r2 is still required.
    assert actual["checks"]["base_node_counts"] is True and actual["checks"]["database_run_coverage"] is True
    assert actual["processed_counts"]["clips"] == actual["processed_counts"]["frames"] == 0


@pytest.mark.parametrize("defect",["missing_empty_recipe","missing_orphan","video_count","video_native_media",
                                  "video_observation_id","full_completion_claim","image_completion_claim"])
def test_image_completion_requires_exact_image_only_evidence(verified_artifacts,defect):
    f = image_artifacts(verified_artifacts)
    item = f["state"]["records"]["recipe:r1"]
    if defect == "missing_empty_recipe":
        del f["state"]["records"]["recipe:r2"]
    elif defect == "missing_orphan":
        del f["state"]["records"]["image:orphan"]
    elif defect == "video_count":
        item["counts"].update(videos=1,clips=1,frames=3)
    elif defect == "video_native_media":
        f["native"]["runs"][item["run_id"]]["source_media_ids"].append("video:v1")
    elif defect == "video_observation_id":
        item["observation_ids"] = ["video:v1"]
    elif defect == "full_completion_claim":
        f["progress"]["completed_full_scope"] = True
    else:
        f["progress"]["completed_image_scope"] = False
    write_json(f["state_path"],f["state"])
    write_json(f["progress_path"],f["progress"])
    result = audit.audit_progress(f["root"],native=f["native"])
    assert result["audit_status"] == "inconsistent" and result["status"] == "incomplete"
    assert result["completed_image_scope"] is False and result["completed_full_scope"] is False


@pytest.mark.parametrize("defect",["full_scope_label","all_signature","invalid_modalities","progress_modalities"])
def test_image_scope_and_declared_modalities_cannot_disagree(verified_artifacts,defect):
    f = image_artifacts(verified_artifacts)
    if defect == "full_scope_label": f["progress"]["scope"] = "full_test"
    elif defect == "all_signature": f["state"]["signature"]["processing_modalities"] = ["image","video"]
    elif defect == "invalid_modalities": f["state"]["signature"]["processing_modalities"] = ["video"]
    else: f["progress"]["processing_modalities"] = ["image","video"]
    write_json(f["state_path"],f["state"])
    write_json(f["state_path"].parent/"signature.json",f["state"]["signature"])
    write_json(f["progress_path"],f["progress"])
    result = audit.audit_progress(f["root"],native=f["native"])
    assert result["audit_status"] == "inconsistent" and result["completed_image_scope"] is False


def test_progress_can_identify_image_scope_without_legacy_signature_field(verified_artifacts):
    f = image_artifacts(verified_artifacts)
    del f["state"]["signature"]["processing_modalities"]
    write_json(f["state_path"],f["state"])
    write_json(f["state_path"].parent/"signature.json",f["state"]["signature"])
    result = audit.audit_progress(f["root"],native=f["native"])
    assert result["processing_modalities"] == ["image"] and result["completed_image_scope"] is True


@pytest.mark.parametrize("signature_declares_mode",[True,False])
def test_bounded_image_success_cannot_claim_all_images(verified_artifacts,signature_declares_mode):
    f = image_artifacts(verified_artifacts)
    item = f["state"]["records"]["recipe:r1"]
    f["state"]["records"] = {"recipe:r1":item}
    f["native"]["runs"] = {item["run_id"]:f["native"]["runs"][item["run_id"]]}
    f["progress"].update(scope="bounded_review",expected_original_recipe_records=1,expected_orphan_media=0,
                         expected_images=1,total_work_items=1,completed_work_items=1,completed_image_scope=False,
                         processed_counts=item["counts"])
    if not signature_declares_mode:
        del f["state"]["signature"]["processing_modalities"]
        write_json(f["state_path"].parent/"signature.json",f["state"]["signature"])
    write_json(f["state_path"],f["state"])
    write_json(f["progress_path"],f["progress"])
    result = audit.audit_progress(f["root"],recipe_id="recipe:r1",native=f["native"])
    assert result["current_scope_verified"] is True
    assert result["completed_image_scope"] is False and result["completed_full_scope"] is False


def test_image_full_scope_excludes_video_orphan_from_work_items(verified_artifacts,monkeypatch):
    f = image_artifacts(verified_artifacts)
    monkeypatch.setattr(audit,"EXPECTED_FULL_COUNTS",{"recipes":2,"orphan_media":2,"images":2,"videos":2})
    media_path = f["root"]/"data/test_graph/media_associations.jsonl"
    rows = [json.loads(line) for line in media_path.read_text().splitlines()]
    rows.append({"media_id":"video:orphan","recipe_id":None,"archive":"test-video.zip","member":"orphan/1.mp4"})
    write_jsonl(media_path,rows)
    build_path = f["root"]/"data/test_graph/build-report.json"
    deployment_path = f["root"]/"reports/user-neo4j-deployment.json"
    for path in (build_path,deployment_path):
        value = json.loads(path.read_text())
        labels = value["node_counts"] if path == build_path else value["input"]["nodes_by_label"]
        labels.update(Video=2,Source=4)
        write_json(path,value)
    f["native"]["base"]["nodes_by_label"].update(Video=2,Source=4)
    manifest_path = f["root"]/"reports/source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative in manifest: manifest[relative] = hashlib.sha256((f["root"]/relative).read_bytes()).hexdigest()
    write_json(manifest_path,manifest)
    result = audit.audit_progress(f["root"],native=f["native"])
    assert result["audit_status"] == "verified" and result["completed_image_scope"] is True
    assert result["expected_full"]["orphan_media"] == 2
    assert result["expected_image_scope"]["orphan_media"] == 1 and result["total_work_items"] == 3
