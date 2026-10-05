#!/usr/bin/env python3
"""Audit saved multimodal coverage and optionally read native Neo4j statistics.

Offline mode never opens a credential file or treats old import reports as a
fresh database measurement. The four input hashes reuse the verified original
test snapshot; this audit does not repeat its complete CSV canonical validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
BASE_OWNER = "recipegen-test-v1"
MM_OWNER = "recipegen-mm-v1"
EXPECTED_FULL_COUNTS = {"recipes": 5898, "orphan_media": 9, "images": 46553, "videos": 1512}
COUNT_KEYS = {"images", "videos", "clips", "frames", "observations", "aligned_media", "extension_nodes", "extension_relationships"}
BASE_CHECK_KEYS = {"node_counts", "relationship_counts", "orphans", "out_of_scope_endpoints", "invalid_node_ids", "invalid_relationship_ids", "recipes_without_text_source"}
ANOMALY_KEYS = {"invalid_scope", "invalid_ids", "invalid_endpoints", "invalid_candidates"}
INPUT_FILES = ("data/test_graph/records.jsonl", "data/test_graph/media_associations.jsonl", "data/test_graph/build-report.json", "reports/user-neo4j-deployment.json")


class AuditError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError("JSON duplicate key")
        result[key] = value
    return result


def _json(text):
    def constant(value):
        raise AuditError("JSON non-finite number")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=constant)


def _read(path: Path):
    return _json(path.read_text(encoding="utf-8"))


def _sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _count(value):
    return type(value) is int and value >= 0


def _counter(value):
    return isinstance(value, dict) and all(isinstance(k, str) and _count(v) for k, v in value.items())


def _zero(value):
    return type(value) is int and value == 0


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _scope(value, build_id):
    return isinstance(value, dict) and value.get("split") == "test" and value.get("dataset_revision") == REVISION and value.get("build_id") == build_id


def _load_inputs(root: Path):
    manifest = _read(root / "reports/source-manifest.json")
    if not isinstance(manifest, dict):
        raise AuditError("source_manifest_not_object")
    hashes = {}
    for relative in INPUT_FILES:
        path = root / relative
        actual = _sha(path)
        if not _hash(manifest.get(relative)) or actual != manifest[relative]:
            raise AuditError(f"source_hash_mismatch:{relative}")
        hashes[relative] = actual
    build = _read(root / INPUT_FILES[2])
    deployment = _read(root / INPUT_FILES[3])
    build_id = build.get("build_id")
    if not isinstance(build_id, str) or not build_id or not _scope(build, build_id) or set(build.get("archives", [])) != {"test.zip", "test-video.zip"} or build.get("training_archives_accessed", []) != []:
        raise AuditError("original_build_scope_invalid")
    checks = deployment.get("checks", {})
    if deployment.get("status") != "verified" or deployment.get("validation_only") is not False or deployment.get("activated") is not True or deployment.get("connected_to_neo4j") is not True or not BASE_CHECK_KEYS <= checks.keys() or any(checks[k] is not True for k in BASE_CHECK_KEYS):
        raise AuditError("original_deployment_not_verified")
    original = deployment.get("input", {})
    if not _scope(original, build_id) or not _hash(original.get("content_sha256")) or not _counter(original.get("nodes_by_label")) or not _counter(original.get("relationships_by_type")):
        raise AuditError("original_deployment_input_invalid")
    if build.get("input_hashes", {}).get("records.jsonl") not in (None, hashes[INPUT_FILES[0]]):
        raise AuditError("original_records_hash_mismatch")
    expected = {}
    with (root / INPUT_FILES[0]).open(encoding="utf-8") as stream:
        for line in stream:
            record = _json(line)
            identifier = record.get("id")
            source = record.get("source", {})
            if not isinstance(identifier, str) or not identifier or identifier in expected or record.get("split") != "test" or source.get("split") != "test" or source.get("revision") != REVISION or source.get("archive") not in {"test.zip", "test-video.zip"}:
                raise AuditError("original_record_scope_or_id_invalid")
            expected[identifier] = {"images": 0, "videos": 0, "kind": "recipe", "media_ids": set()}
    seen = set()
    with (root / INPUT_FILES[1]).open(encoding="utf-8") as stream:
        for line in stream:
            media = _json(line)
            identifier, recipe = media.get("media_id"), media.get("recipe_id")
            if not isinstance(identifier, str) or identifier in seen or not identifier.startswith(("image:", "video:")) or media.get("archive") not in {"test.zip", "test-video.zip"} or not isinstance(media.get("member"), str) or not media["member"]:
                raise AuditError("original_media_scope_or_id_invalid")
            seen.add(identifier)
            kind = "images" if identifier.startswith("image:") else "videos"
            if recipe is None:
                expected[identifier] = {"images": 0, "videos": 0, "kind": "orphan_media", "media_ids": set()}
                recipe = identifier
            if recipe not in expected:
                raise AuditError("original_media_recipe_missing")
            expected[recipe][kind] += 1
            expected[recipe]["media_ids"].add(identifier)
    totals = {"recipes": sum(v["kind"] == "recipe" for v in expected.values()),
              "orphan_media": sum(v["kind"] == "orphan_media" for v in expected.values()),
              "images": sum(v["images"] for v in expected.values()), "videos": sum(v["videos"] for v in expected.values())}
    if totals != EXPECTED_FULL_COUNTS or original["nodes_by_label"].get("Recipe") != totals["recipes"] or original["nodes_by_label"].get("Image") != totals["images"] or original["nodes_by_label"].get("Video") != totals["videos"]:
        raise AuditError("full_test_source_counts_mismatch")
    # Re-hash after streaming to fail closed if an input changed during reading.
    if any(_sha(root / relative) != digest for relative, digest in hashes.items()):
        raise AuditError("source_changed_during_audit")
    return expected, build_id, deployment, hashes


def _processing_modalities(signature, progress):
    """旧批次默认图像+视频；图像范围必须与显式签名和进度相符。"""
    inferred = ["image"] if progress.get("scope") == "full_test_images" else ["image", "video"]
    modalities = signature.get("processing_modalities", progress.get("processing_modalities", inferred))
    if modalities not in (["image"], ["image", "video"]):
        raise AuditError("processing_modalities_invalid")
    if (progress.get("scope") == "full_test_images" and modalities != ["image"]
            or progress.get("scope") == "full_test" and modalities != ["image", "video"]):
        raise AuditError("processing_modalities_scope_mismatch")
    if "processing_modalities" in progress and progress["processing_modalities"] != modalities:
        raise AuditError("progress_processing_modalities_mismatch")
    return list(modalities)


def _image_expected(expected):
    # 原始两归档已完整核验，再投影图像工作范围；零图 Recipe 也必须遍历。
    return {identifier: {**item, "videos": 0,
                         "media_ids": {mid for mid in item["media_ids"] if mid.startswith("image:")}}
            for identifier, item in expected.items()
            if item["kind"] == "recipe" or item["images"] > 0}


def _item_report(root, run_dir, identifier, item, expected, build_id, base_sha):
    counts = item.get("counts")
    if not _counter(counts) or not COUNT_KEYS <= counts.keys():
        raise AuditError("counts_missing_or_not_nonnegative_integer")
    if any(counts[k] != expected[k] for k in ("images", "videos")) or counts["observations"] != counts["images"] + counts["clips"] or counts["aligned_media"] > counts["observations"]:
        raise AuditError("media_or_observation_counts_mismatch")
    if counts["clips"] < counts["videos"] or counts["frames"] < counts["clips"] or (counts["videos"] == 0 and (counts["clips"] or counts["frames"])):
        raise AuditError("video_window_counts_invalid")
    relative = item.get("report")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AuditError("unsafe_report_path")
    path = (root / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()) or path.name != "import-report.json":
        raise AuditError("report_outside_pipeline_run")
    report = _read(path)
    checks = report.get("checks", {})
    if report.get("status") != "verified" or report.get("connected_to_neo4j") is not True or report.get("validation_only") is not False or not isinstance(checks, dict) or not checks or any(v is not True for v in checks.values()):
        raise AuditError("import_report_not_verified")
    data = report.get("input", {})
    if not _scope(data, build_id) or data.get("schema_version") != MM_OWNER or data.get("run_id") != item.get("run_id") or not isinstance(item.get("run_id"), str) or not item["run_id"] or not _hash(data.get("content_sha256")) or data.get("base_content_sha256") != base_sha:
        raise AuditError("import_report_scope_or_hash_invalid")
    native = report.get("native_neo4j", {})
    if not _counter(data.get("nodes_by_label")) or not _counter(data.get("relationships_by_type")) or data["nodes_by_label"].get("SemanticRun") != 1 or data["nodes_by_label"].get("VisualObservation", 0) != counts["observations"]:
        raise AuditError("import_report_labels_invalid")
    if data["nodes_by_label"].get("Frame", 0) != counts["frames"] or data["nodes_by_label"].get("VideoClip", 0) != counts["clips"]:
        raise AuditError("import_report_video_window_counts_mismatch")
    for key, mapping in (("extension_nodes", "nodes_by_label"), ("extension_relationships", "relationships_by_type")):
        if not _count(data.get(key)) or not _count(native.get(key)) or data[key] != counts[key] or native[key] != counts[key] or sum(data[mapping].values()) != counts[key]:
            raise AuditError("import_report_extension_counts_mismatch")
    observation_ids = item.get("observation_ids")
    if not isinstance(observation_ids, list) or any(not isinstance(v, str) or not v for v in observation_ids) or len(observation_ids) != counts["observations"] or len(set(observation_ids)) != len(observation_ids):
        raise AuditError("observation_ids_missing_or_duplicate")
    return {**data, "aligned_media": counts["aligned_media"]}


def _native_checks(native, original, valid_reports, expected_media):
    checks = {}
    active, base = native.get("active_build", {}), native.get("base", {})
    checks["active_original_build"] = (_scope(active, original["build_id"]) and active.get("active") is True and active.get("status") == "verified" and active.get("content_sha256") == original["content_sha256"])
    checks["base_node_counts"] = _counter(base.get("nodes_by_label")) and base["nodes_by_label"] == original["nodes_by_label"]
    checks["base_relationship_counts"] = _counter(base.get("relationships_by_type")) and base["relationships_by_type"] == original["relationships_by_type"]
    for key in BASE_CHECK_KEYS - {"node_counts", "relationship_counts"}:
        checks["base_" + key] = _zero(base.get(key))
    anomalies = native.get("anomalies", {})
    for key in ANOMALY_KEYS:
        checks["extensions_" + key] = _zero(anomalies.get(key))
    observed = native.get("runs", {})
    checks["native_connected"] = native.get("connected_to_neo4j") is True
    for run_id, report in valid_reports.items():
        value = observed.get(run_id, {})
        media = value.get("source_media_ids")
        checks["run:" + run_id] = (_scope(value, report["build_id"]) and value.get("content_sha256") == report["content_sha256"]
            and _counter(value.get("nodes_by_label")) and value["nodes_by_label"] == report["nodes_by_label"]
            and _counter(value.get("relationships_by_type")) and value["relationships_by_type"] == report["relationships_by_type"]
            and all(_count(value.get(k)) and value[k] == report[k] for k in ("extension_nodes", "extension_relationships"))
            and _count(value.get("aligned_media")) and value["aligned_media"] == report["aligned_media"]
            and isinstance(media, list) and all(isinstance(v, str) for v in media) and len(set(media)) == len(media)
            and set(media) == expected_media[run_id])
    return checks


def audit_progress(project_root: Path, *, progress_path: Path | None = None, recipe_id: str | None = None, native: dict | None = None) -> dict[str, Any]:
    """Verify coverage contracts; completion requires an explicit fresh DB snapshot.

    ``native`` is supplied by ``read_native_stats`` or a test double. Current
    scope can be verified for a bounded sample while full-scope status remains
    incomplete. Saved failure messages are not copied into this public report.
    """
    root = Path(project_root).resolve()
    result = {"status": "incomplete", "audit_status": "inconsistent", "completed_full_scope": False,
              "completed_image_scope": False,
              "current_scope_verified": False, "read_only": True, "connected_to_neo4j": native is not None and native.get("connected_to_neo4j") is True,
              "semantic_accuracy_verified": False, "expected_full": dict(EXPECTED_FULL_COUNTS), "errors": [], "checks": {},
              "checked_at": datetime.now(timezone.utc).isoformat()}
    errors, checks = result["errors"], result["checks"]
    try:
        expected, build_id, deployment, hashes = _load_inputs(root)
        result.update(build_id=build_id, dataset_revision=REVISION, split="test", source_sha256=hashes,
                      source_validation="previous_verified_snapshot_plus_current_source_hashes")
        progress = _read(Path(progress_path) if progress_path else root / "reports/multimodal-progress.json")
        pipeline = progress.get("pipeline_run_id")
        if not isinstance(pipeline, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", pipeline):
            raise AuditError("invalid_pipeline_run_id")
        result.update(pipeline_run_id=pipeline, reported_status=progress.get("status"), scope=progress.get("scope"))
        run_dir = root / "data/multimodal/runs" / pipeline
        state = _read(run_dir / "state.json")
        signature = _read(run_dir / "signature.json")
        if state.get("pipeline_run_id") != pipeline or state.get("signature") != signature or signature.get("build_id") != build_id or signature.get("dataset_revision") != REVISION or signature.get("schema") != MM_OWNER:
            raise AuditError("state_or_signature_scope_mismatch")
        modalities = _processing_modalities(signature, progress)
        image_only = modalities == ["image"]
        result["processing_modalities"] = modalities
        if image_only:
            expected = _image_expected(expected)
            result["expected_image_scope"] = {"recipes": EXPECTED_FULL_COUNTS["recipes"],
                "orphan_media": sum(v["kind"] == "orphan_media" for v in expected.values()),
                "images": EXPECTED_FULL_COUNTS["images"], "videos": 0}
        records = state.get("records")
        if not isinstance(records, dict) or any(k not in expected or not isinstance(v, dict) for k, v in records.items()):
            raise AuditError("state_records_out_of_scope")
        full = progress.get("scope") in {"full_test", "full_test_images"}
        if full:
            if recipe_id is not None:
                raise AuditError("recipe_id_requires_bounded_scope")
            selected = set(expected)
        elif progress.get("scope") == "bounded_review":
            selected = {recipe_id} if recipe_id is not None else set(records)
            if not selected or any(k not in expected for k in selected):
                raise AuditError("bounded_selection_unknown")
        else:
            raise AuditError("progress_scope_invalid")
        totals = {"recipes": sum(expected[k]["kind"] == "recipe" for k in selected),
                  "orphan_media": sum(expected[k]["kind"] == "orphan_media" for k in selected),
                  "images": sum(expected[k]["images"] for k in selected), "videos": sum(expected[k]["videos"] for k in selected)}
        result["current_scope_expected"] = totals
        for field, value in (("expected_original_recipe_records", totals["recipes"]), ("expected_orphan_media", totals["orphan_media"]), ("expected_images", totals["images"]), ("expected_videos", totals["videos"]), ("total_work_items", len(selected))):
            checks["progress_" + field] = _count(progress.get(field)) and progress[field] == value
        saved_valid, saved_media, processed, failed, pending = {}, {}, Counter(), [], []
        completed_count = 0
        for identifier in sorted(selected):
            item = records.get(identifier, {})
            status = item.get("status", "pending")
            if status == "verified":
                try:
                    data = _item_report(root, run_dir, identifier, item, expected[identifier], build_id, deployment["input"]["content_sha256"])
                    if image_only and set(item["observation_ids"]) != expected[identifier]["media_ids"]:
                        raise AuditError("image_observation_ids_do_not_cover_original_images")
                    if data["run_id"] in saved_valid:
                        raise AuditError("duplicate_semantic_run_id")
                    saved_valid[data["run_id"]] = data
                    saved_media[data["run_id"]] = expected[identifier]["media_ids"]
                    completed_count += 1
                    processed.update(item["counts"])
                except (AuditError, OSError, ValueError, TypeError, KeyError) as error:
                    errors.append({"work_item": identifier, "reason": str(error) if isinstance(error, AuditError) else "import_report_missing_or_invalid"})
            elif status == "failed":
                failed.append({"work_item": identifier, "error_type": item.get("error_type")})
            else:
                pending.append({"work_item": identifier, "status": status})
        # Count only native-verified items; offline processed records stay pending.
        checks["progress_completed_work_items"] = _count(progress.get("completed_work_items")) and progress["completed_work_items"] == completed_count
        checks["progress_failed_work_items"] = _count(progress.get("failed_work_items")) and progress["failed_work_items"] == len(failed)
        checks["progress_processed_counts"] = _counter(progress.get("processed_counts")) and dict(Counter(progress["processed_counts"])) == dict(processed)
        result.update(verified_saved_work_items=completed_count, total_work_items=len(selected), processed_counts=dict(processed),
                      failed_work_items=len(failed), pending_work_items=len(pending), failures=failed, pending=pending,
                      saved_import_reports_verified=not errors)
        saved_scope = full and completed_count == len(expected) and not failed and not pending
        saved_full = saved_scope and not image_only and all(processed[k] == EXPECTED_FULL_COUNTS[k] for k in ("images", "videos"))
        saved_images = saved_scope and image_only and processed["images"] == EXPECTED_FULL_COUNTS["images"] and processed["videos"] == 0
        checks["progress_completion_claim"] = type(progress.get("completed_full_scope")) is bool and progress["completed_full_scope"] == saved_full
        if image_only:
            checks["progress_image_completion_claim"] = type(progress.get("completed_image_scope")) is bool and progress["completed_image_scope"] == saved_images
        if native is not None:
            checks.update(_native_checks(native, deployment["input"], saved_valid, saved_media))
            unexpected = sorted(set(native.get("runs", {})) - set(saved_valid))
            result["database_runs_without_current_verified_report"] = unexpected
            if full:
                checks["database_run_coverage"] = not unexpected
        result["checks"] = checks
        result["audit_status"] = "verified" if not errors and all(checks.values()) else "inconsistent"
        result["current_scope_verified"] = (native is not None and result["audit_status"] == "verified" and completed_count == len(selected) and not failed and not pending)
        result["completed_full_scope"] = saved_full and result["current_scope_verified"]
        result["completed_image_scope"] = saved_images and result["current_scope_verified"]
        result["status"] = "verified" if result["completed_full_scope"] or result["completed_image_scope"] else "incomplete"
        result["note"] = "Verified refers to coverage, source/structure and native counts; no semantic accuracy or held-out performance is claimed. Offline or bounded success is never full completion; image-only completion never claims full multimodal completion."
    except (AuditError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append({"reason": str(error) if isinstance(error, AuditError) else "required_artifact_missing_or_invalid"})
    return result


RUN_QUERY = """// recipegen-mm:audit-runs
MATCH (r:RecipeGen:SemanticRun {kg_import_namespace: $mm_owner, pipeline_run_id: $pipeline_id})
RETURN r.run_id AS run_id, r.build_id AS build_id, r.split AS split,
 r.dataset_revision AS dataset_revision, r.content_sha256 AS content_sha256
"""
NODE_QUERY = """// recipegen-mm:audit-nodes
MATCH (n:RecipeGen {kg_import_namespace: $mm_owner}) WHERE n.run_id IN $run_ids
UNWIND labels(n) AS label WITH n.run_id AS run_id, label WHERE label IN $extension_labels
RETURN run_id, label, count(*) AS count
"""
EDGE_QUERY = """// recipegen-mm:audit-edges
MATCH ()-[r]->() WHERE r.kg_import_namespace = $mm_owner AND r.run_id IN $run_ids
RETURN r.run_id AS run_id, type(r) AS type, count(*) AS count
"""
MEDIA_QUERY = """// recipegen-mm:audit-media-coverage
MATCH (n:RecipeGen {kg_import_namespace: $mm_owner}) WHERE n.run_id IN $run_ids
 AND (n:VisualObservation OR n:VideoClip)
RETURN n.run_id AS run_id, collect(DISTINCT n.source_media_id) AS source_media_ids
"""
ALIGNMENT_QUERY = """// recipegen-mm:audit-main-step-alignments
MATCH (a:RecipeGen)-[r:ALIGNED_WITH]->(s:RecipeGen:Step {kg_import_namespace: $base_owner})
WHERE r.kg_import_namespace = $mm_owner AND r.run_id IN $run_ids
 AND r.alignment_method = 'clip_cosine_temporal_candidate' AND (a:Image OR a:VideoClip)
RETURN r.run_id AS run_id, count(DISTINCT a) AS aligned_media
"""
EXTENSION_ANOMALIES = """// recipegen-mm:audit-anomalies
CALL () {
 MATCH (n:RecipeGen {kg_import_namespace: $mm_owner}) WHERE n.run_id IN $run_ids
 RETURN sum(CASE WHEN coalesce(n.build_id, '') <> $build_id OR coalesce(n.split, '') <> 'test'
   OR coalesce(n.dataset_revision, '') <> $dataset_revision THEN 1 ELSE 0 END) AS invalid_node_scope,
 sum(CASE WHEN NOT n:SemanticRun AND (n.verified IS NULL OR n.verified <> false
   OR n.confidence IS NOT NULL OR NOT coalesce(n.semantics, '') IN ['model_candidate','media_structure']
   OR n.source_media_id IS NULL OR n.source_id IS NULL OR n.evidence_json IS NULL) THEN 1 ELSE 0 END) AS invalid_node_candidates
}
CALL () {
 MATCH (a)-[r]->(b) WHERE r.kg_import_namespace = $mm_owner AND r.run_id IN $run_ids
 RETURN sum(CASE WHEN coalesce(r.build_id, '') <> $build_id OR coalesce(r.split, '') <> 'test'
   OR coalesce(r.dataset_revision, '') <> $dataset_revision THEN 1 ELSE 0 END) AS invalid_edge_scope,
 sum(CASE WHEN NOT a:RecipeGen OR NOT b:RecipeGen OR coalesce(a.build_id, '') <> $build_id
   OR coalesce(b.build_id, '') <> $build_id OR coalesce(a.split, '') <> 'test' OR coalesce(b.split, '') <> 'test'
   OR coalesce(a.dataset_revision, '') <> $dataset_revision OR coalesce(b.dataset_revision, '') <> $dataset_revision
   OR NOT coalesce(a.kg_import_namespace, '') IN [$base_owner,$mm_owner]
   OR NOT coalesce(b.kg_import_namespace, '') IN [$base_owner,$mm_owner]
   OR (a.kg_import_namespace = $mm_owner AND coalesce(a.run_id, '') <> r.run_id)
   OR (b.kg_import_namespace = $mm_owner AND coalesce(b.run_id, '') <> r.run_id)
   THEN 1 ELSE 0 END) AS invalid_endpoints,
 sum(CASE WHEN r.verified IS NULL OR r.verified <> false OR r.confidence IS NOT NULL
   OR NOT coalesce(r.semantics, '') IN ['model_candidate','media_structure']
   OR r.source_media_id IS NULL OR r.source_id IS NULL OR r.evidence_json IS NULL
   THEN 1 ELSE 0 END) AS invalid_edge_candidates
}
CALL () {
 MATCH (n:RecipeGen {kg_import_namespace: $mm_owner}) WHERE n.run_id IN $run_ids
 WITH n.kg_id AS id, count(*) AS occurrences WHERE id IS NULL OR occurrences > 1
 RETURN coalesce(sum(occurrences),0) AS invalid_node_ids
}
CALL () {
 MATCH ()-[r]->() WHERE r.kg_import_namespace = $mm_owner AND r.run_id IN $run_ids
 WITH r.kg_id AS id, count(*) AS occurrences WHERE id IS NULL OR occurrences > 1
 RETURN coalesce(sum(occurrences),0) AS invalid_edge_ids
}
RETURN coalesce(invalid_node_scope,0)+coalesce(invalid_edge_scope,0) AS invalid_scope,
 invalid_node_ids+invalid_edge_ids AS invalid_ids, coalesce(invalid_endpoints,0) AS invalid_endpoints,
 coalesce(invalid_node_candidates,0)+coalesce(invalid_edge_candidates,0) AS invalid_candidates
"""
READ_ONLY_QUERIES = (RUN_QUERY, NODE_QUERY, EDGE_QUERY, MEDIA_QUERY, ALIGNMENT_QUERY, EXTENSION_ANOMALIES)


def _module(root, filename, name):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_native_stats(session, project_root: Path, pipeline_id: str, build_id: str):
    """All native queries are fixed parameterized reads in one read transaction."""
    importer = _module(Path(project_root), "import_test_graph_neo4j.py", "mm_audit_base")
    query = _module(Path(project_root), "query_test_graph.py", "mm_audit_query")
    parameters = {"build_id": build_id, "dataset_revision": REVISION, "split": "test", "owner": BASE_OWNER,
                  "base_owner": BASE_OWNER, "mm_owner": MM_OWNER, "pipeline_id": pipeline_id,
                  "allowed_labels": sorted(importer.ALLOWED_LABELS), "extension_labels": ["SemanticRun", "VisualObservation", "VisualObject", "VideoClip", "Frame"]}
    def read(tx):
        def rows(cypher, p=parameters):
            return [r.data() for r in tx.run(cypher, **p)]
        active = rows(query.ACTIVE_BUILD_QUERY)
        if len(active) != 1:
            raise AuditError("native_active_build_not_unique")
        base = {"nodes_by_label": {r["label"]: r["count"] for r in rows(importer.NODE_COUNTS)},
                "relationships_by_type": {r["type"]: r["count"] for r in rows(importer.RELATIONSHIP_COUNTS)}}
        anomalies = rows(importer.VERIFY_ANOMALIES)
        if len(anomalies) != 1:
            raise AuditError("native_base_checks_missing")
        base.update(anomalies[0])
        run_rows = rows(RUN_QUERY)
        runs = {}
        for r in run_rows:
            if r["run_id"] in runs:
                raise AuditError("native_duplicate_semantic_run")
            runs[r["run_id"]] = {k: v for k, v in r.items() if k != "run_id"}
            runs[r["run_id"]].update(nodes_by_label={}, relationships_by_type={}, source_media_ids=[], aligned_media=0)
        p = {**parameters, "run_ids": list(runs)}
        for r in rows(NODE_QUERY, p):
            runs[r["run_id"]]["nodes_by_label"][r["label"]] = r["count"]
        for r in rows(EDGE_QUERY, p):
            runs[r["run_id"]]["relationships_by_type"][r["type"]] = r["count"]
        for r in rows(MEDIA_QUERY, p):
            runs[r["run_id"]]["source_media_ids"] = r["source_media_ids"]
        for r in rows(ALIGNMENT_QUERY, p):
            runs[r["run_id"]]["aligned_media"] = r["aligned_media"]
        for r in runs.values():
            r.update(extension_nodes=sum(r["nodes_by_label"].values()), extension_relationships=sum(r["relationships_by_type"].values()))
        anomalies = rows(EXTENSION_ANOMALIES, p)
        if len(anomalies) != 1:
            raise AuditError("native_extension_checks_missing")
        return {"connected_to_neo4j": True, "active_build": active[0], "base": base, "runs": runs, "anomalies": anomalies[0]}
    return session.execute_read(read)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--recipe-id", help="Disambiguate a bounded sample; full scope must omit it")
    parser.add_argument("--neo4j", action="store_true", help="Explicit fresh native read; offline default never reads credentials")
    parser.add_argument("--config", type=Path, help="Optional private connection config, only read with --neo4j")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    preliminary = audit_progress(root, progress_path=args.progress, recipe_id=args.recipe_id)
    native = None
    if args.neo4j and preliminary.get("pipeline_run_id") and preliminary.get("build_id"):
        sys.path.insert(0, str(root))
        query = _module(root, "query_test_graph.py", "mm_audit_connection")
        try:
            connection = query.load_connection(args.config) if args.config else query.load_connection()
            from neo4j import GraphDatabase, READ_ACCESS
            with GraphDatabase.driver(connection.uri, auth=(connection.user, connection.password)) as driver:
                driver.verify_connectivity()
                with driver.session(database=connection.database, default_access_mode=READ_ACCESS) as session:
                    native = read_native_stats(session, root, preliminary["pipeline_run_id"], preliminary["build_id"])
        except Exception as error:
            # Keep secret-bearing URI/driver messages out of public artifacts.
            preliminary["errors"].append({"reason": "native_read_failed", "error_type": type(error).__name__})
            preliminary["audit_status"] = "inconsistent"
    result = audit_progress(root, progress_path=args.progress, recipe_id=args.recipe_id, native=native) if native is not None else preliminary
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["current_scope_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
