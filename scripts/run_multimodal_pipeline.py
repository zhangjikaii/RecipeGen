#!/usr/bin/env python3
"""Checkpointed test-only V3/V4/V5 inference, alignment and append-only import."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.kg_extract import DEFAULT_LEXICON
from recipegen.multimodal_align import align_multimodal
from recipegen.multimodal_bundle import build_multimodal_bundle
from recipegen.multimodal_graph import import_bundle, load_base_index, validate_bundle
from recipegen.multimodal_media import MediaPreparer, select_records
from recipegen.multimodal_models import LocalMultimodalModels, sha_file
from recipegen.multimodal_scope import verify_full_scope
from recipegen.multimodal_reuse import seed_observations


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def read_checkpoint(path: Path):
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()))


def compress_checkpoints(folder: Path):
    """压缩已核验结果；保留原始响应与向量，减少全量磁盘占用。"""
    for name in ("prepared.json", "observations.json", "alignment.json", "bundle.json"):
        path = folder / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        target = path.with_suffix(path.suffix + ".gz")
        part = target.with_suffix(target.suffix + ".part")
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        if gzip.decompress(compressed) != raw:
            raise ValueError("检查点压缩回读不一致")
        part.write_bytes(compressed)
        part.replace(target)
        path.unlink()


def read_config():
    spec = importlib.util.spec_from_file_location("mm_connection", ROOT / "scripts/query_test_graph.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_connection()


def mean_unit(vectors):
    values = [math.fsum(v[i] for v in vectors) / len(vectors) for i in range(512)]
    norm = math.sqrt(math.fsum(v * v for v in values))
    if norm == 0:
        raise ValueError("真实帧向量均值为零")
    return [v / norm for v in values]


def base_metadata(base):
    aliases = {}
    for kind, entries in (("Ingredient", DEFAULT_LEXICON.ingredients), ("Action", DEFAULT_LEXICON.actions), ("Tool", DEFAULT_LEXICON.tools)):
        for entry in entries:
            aliases[(kind, entry.normalized_name)] = list(entry.aliases)
    entities = []
    for node in base.nodes_by_id.values():
        kind = node["label"]
        if kind in {"Ingredient", "Action", "Tool"}:
            name = node["properties"]["normalized_name"]
            entities.append({"id": node["id"], "type": kind, "name": name, "aliases": aliases.get((kind, name), [name])})
    return {"build_id": base.build_id, "dataset_revision": base.dataset_revision, "entities": entities}


def observe_record(prepared, folder, models, update):
    cache_path = folder / "observations.json"
    observations = read_checkpoint(cache_path) if cache_path.is_file() or cache_path.with_suffix(".json.gz").is_file() else []
    done = {row["id"] for row in observations}
    pending = []
    for image in prepared["images"]:
        pending.append((image["media_id"], "image", image, None))
    for video in prepared["videos"]:
        for clip in video["clips"]:
            pending.append((clip["clip_id"], "clip", video, clip))
    for number, (identifier, kind, medium, clip) in enumerate(pending, 1):
        if identifier in done:
            continue
        update(stage="vision_inference", current_observation=identifier, observation_number=number,
               expected_observations=len(pending))
        frames = clip["frames"] if clip is not None else []
        paths = [ROOT / frame["path"] for frame in frames] if clip else [ROOT / medium["path"]]
        timestamps = [frame["timestamp_sec"] for frame in frames] if clip else None
        result = models.observe(paths, timestamps=timestamps)
        embeddings = models.image_embeddings(paths)
        embedding = mean_unit(embeddings) if clip else embeddings[0]
        evidence = {"source_media_id": medium["media_id"], "source_id": medium["source_id"],
                    "archive": medium["archive"], "member": medium["member"], "revision": medium["revision"],
                    "split": "test", "media_sha256": medium["sha256"], "zip_crc_verified": medium["zip_crc_verified"],
                    "model_input_sha256": result["input_sha256"], "model_input_paths": result["input_paths"],
                    "raw_model_response": result["raw_text"], "local_inference": True,
                    "temporary_input_retained": prepared.get("retain_media", False)}
        if clip:
            evidence.update(frame_evidence=[{"sha256": f["sha256"], "timestamp_sec": f["timestamp_sec"],
                                             "offset_sec": f["offset_sec"], "pts": f["pts"], "time_base": f["time_base"]}
                                            for f in frames],
                            start_sec=clip["start_sec"], end_sec=clip["end_sec"])
        output = result["output"]
        row = {"id": identifier, "kind": kind, "source_media_id": medium["media_id"], "source_id": medium["source_id"],
               "caption": output["caption"], "objects": output["objects"], "actions": output["actions"],
               "raw_output": output, "raw_text": result["raw_text"], "model": result["model"], "evidence": evidence,
               "embedding": embedding, "embedding_space": models.embedding_space,
               "inference_seconds": result["inference_seconds"], "generation_attempts": result["attempts"],
               "prior_invalid_outputs": result["prior_invalid_outputs"]}
        if clip:
            row.update(timeline_id=medium["media_id"], start_seconds=clip["start_sec"], end_seconds=clip["end_sec"])
        observations.append(row)
        done.add(identifier)
        write_json(cache_path, observations)
    return observations


def ensure_vector_index(session):
    # The separate retrieval command remains read-only.
    tick = chr(96)
    query = ("CREATE VECTOR INDEX recipegen_visual_embedding IF NOT EXISTS "
             "FOR (n:VisualObservation) ON (n.embedding) OPTIONS {indexConfig: {"
             + tick + "vector.dimensions" + tick + ": 512, "
             + tick + "vector.similarity_function" + tick + ": 'cosine'}}")
    session.execute_write(lambda tx: tx.run(query).consume())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 means every original test record plus all orphan media")
    parser.add_argument("--recipe-id")
    parser.add_argument("--import-neo4j", action="store_true")
    parser.add_argument("--keep-media", action="store_true", help="Retain inputs for bounded manual review only")
    parser.add_argument("--run-id")
    parser.add_argument("--reuse-run", help="复用兼容旧批次的真实观测；保留原签名与来源，不复制成功状态")
    parser.add_argument("--modality", choices=("all", "images"), default="all", help="images 只下载、识别和对齐图片，保留原文本步骤")
    parser.add_argument("--min-free-gb", type=float, default=1.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_attempts < 1 or args.min_free_gb <= 0:
        parser.error("Invalid limit, retry count or free-space bound")
    if args.keep_media and not args.recipe_id and args.limit == 0:
        parser.error("Full scope must clean temporary media; --keep-media is for bounded review only")
    lock = (ROOT / ".runtime/multimodal-pipeline.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another RecipeGen multimodal worker already holds the project lock")
    manifest = json.loads((ROOT / ".runtime/multimodal-models.json").read_text())
    base = load_base_index(ROOT)
    metadata = base_metadata(base)
    preparer = MediaPreparer(ROOT)
    selected = list(select_records(ROOT / "data/test_graph/records.jsonl", limit=args.limit, recipe_id=args.recipe_id))
    full_scope = args.limit == 0 and args.recipe_id is None
    orphans = list(preparer.iter_orphan_media()) if full_scope else []
    coverage = verify_full_scope(selected, orphans, base) if full_scope else None
    processing_modalities = ["image"] if args.modality == "images" else ["image", "video"]
    image_coverage = None
    if args.modality == "images":
        # 先核验完整原始来源，再投影图片执行范围；视频成员不会交给下载/解码器。
        selected = [dict(record, media=[medium for medium in record["media"] if medium["kind"] == "image"])
                    for record in selected]
        orphans = [medium for medium in orphans if medium["kind"] == "image"]
        if coverage:
            image_coverage = {**coverage, "processing_modalities": ["image"],
                              "counts": {**coverage["counts"], "Video": 0}, "orphan_media": len(orphans),
                              "id_set_sha256": {**coverage["id_set_sha256"],
                                                "Video": hashlib.sha256(b"[]").hexdigest()}}
    code_files = [Path(__file__), *(ROOT / "recipegen" / f"multimodal_{name}.py" for name in ("align", "bundle", "graph", "media", "models", "scope", "reuse"))]
    signature = {"schema": "recipegen-mm-v1", "build_id": base.build_id, "dataset_revision": base.dataset_revision,
                 "models": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in manifest.items()},
                 "code_hashes": {str(p.relative_to(ROOT)): sha_file(p) for p in code_files},
                 "image_side": 448, "clip_seconds": 5.0, "frames_per_clip": 3, "similarity_threshold": args.similarity_threshold,
                 "temporary_media_retention": args.keep_media,
                 "processing_modalities": processing_modalities,
                 "image_scope_coverage": image_coverage,
                 "graph_target": {"uri": read_config().uri, "database": read_config().database} if args.import_neo4j else None,
                 "full_scope_coverage": coverage, "records_sha256": sha_file(ROOT / "data/test_graph/records.jsonl")}
    pipeline_id = args.run_id or "mm-" + digest(signature)[:20]
    if not pipeline_id.replace("-", "").replace("_", "").isalnum() or len(pipeline_id) > 80:
        parser.error("run-id must use alphanumeric, hyphen or underscore characters")
    run_dir = ROOT / "data/multimodal/runs" / pipeline_id
    run_dir.mkdir(parents=True, exist_ok=True)
    signature_path = run_dir / "signature.json"
    if signature_path.is_file() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Same pipeline_id has different models/code/input; use a new run ID")
    write_json(signature_path, signature)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "pipeline_run_id": pipeline_id, "signature": signature, "started_at": stamp(), "records": {}, "model_calls": {}}
    items = [(r["id"], r, False) for r in selected] + [(r["media_id"], r, True) for r in orphans]
    if args.reuse_run:
        reused = seed_observations(ROOT, args.reuse_run, run_dir, signature, {identifier for identifier, _, _ in items})
        if "observation_reuse" not in state:
            state["observation_reuse"] = {key: value for key, value in reused.items() if key != "items"}
            write_json(run_dir / "reuse-report.json", reused)
    # 中断后先清理本批次未完成项的自有临时目录，避免遗留原件使低盘续跑永久卡住。
    if not args.keep_media:
        for identifier, _, orphan in items:
            if state["records"].get(identifier, {}).get("status") in {"processing", "failed"}:
                preparer.cleanup_record("orphan:" + identifier if orphan else identifier)
    expected_media = {kind: sum(m["kind"] == kind for r in selected for m in r["media"]) +
                     sum(r["kind"] == kind for r in orphans) for kind in ("image", "video")}
    progress_path = ROOT / "reports/multimodal-progress.json"
    live = {}

    def update(**extra):
        live.update(extra)
        rows = [state["records"].get(identifier, {}) for identifier, _, _ in items]
        successes = [r for r in rows if r.get("status") == ("verified" if args.import_neo4j else "processed")]
        counts = Counter()
        for row in successes:
            counts.update(row.get("counts", {}))
        report = {"pipeline_run_id": pipeline_id, "status": live.get("status", "running"),
                  "scope": ("full_test_images" if args.modality == "images" else "full_test") if full_scope else "bounded_review",
                  "processing_modalities": processing_modalities,
                  "expected_original_recipe_records": len(selected), "expected_orphan_media": len(orphans),
                  "expected_images": expected_media["image"], "expected_videos": expected_media["video"],
                  "completed_work_items": len(successes), "total_work_items": len(items),
                  "failed_work_items": sum(r.get("status") == "failed" for r in rows),
                  "processed_counts": dict(counts), "completed_full_scope": False, "model_calls_this_process": state.get("model_calls", {}),
                  "neo4j_import_requested": args.import_neo4j, "local_inference": True,
                  "full_scope_coverage": coverage,
                  "image_scope_coverage": image_coverage,
                  "observation_reuse": state.get("observation_reuse"),
                  "semantic_accuracy_verified": False, "updated_at": stamp(),
                  "free_disk_gb": shutil.disk_usage(ROOT).free / 1024**3, **live}
        execution_complete = (full_scope and args.import_neo4j and len(successes) == len(items) and
                              counts["images"] == expected_media["image"] and counts["videos"] == expected_media["video"])
        report["completed_full_scope"] = execution_complete and args.modality == "all"
        report["completed_image_scope"] = execution_complete and args.modality == "images"
        write_json(state_path, state)
        write_json(progress_path, report)
        return report

    update(stage="loading_local_models", pid=os.getpid())
    write_json(ROOT / ".runtime/multimodal-process.json", {"pid": os.getpid(), "pipeline_run_id": pipeline_id, "started_at": stamp()})
    models = LocalMultimodalModels(ROOT)
    driver = session = None
    if args.import_neo4j:
        from neo4j import GraphDatabase
        connection = read_config()
        driver = GraphDatabase.driver(connection.uri, auth=(connection.user, connection.password))
        driver.verify_connectivity()
        session = driver.session(database=connection.database)
    try:
        for attempt in range(args.max_attempts):
            remaining = [(identifier, source, orphan) for identifier, source, orphan in items
                         if state["records"].get(identifier, {}).get("status") != ("verified" if args.import_neo4j else "processed")]
            if not remaining:
                break
            for identifier, source, orphan in remaining:
                if shutil.disk_usage(ROOT).free < args.min_free_gb * 1024**3:
                    update(status="stopped_low_disk", stage="resource_check", minimum_free_gb=args.min_free_gb)
                    return 10
                folder = run_dir / digest(identifier)[:24]
                folder.mkdir(exist_ok=True)
                state["records"][identifier] = {"status": "processing", "attempts": attempt + 1, "started_at": stamp()}
                update(stage="preparing_media", current_work_item=identifier, attempt=attempt + 1)
                started = time.monotonic()
                prepared = None
                try:
                    run_id = pipeline_id + "-" + digest(identifier)[:16]
                    bundle_path = folder / "bundle.json"
                    if bundle_path.is_file() or bundle_path.with_suffix(".json.gz").is_file():
                        # 重试使用完全相同的已保存载荷，避免下载复用状态改变不可变入库内容。
                        prepared = read_checkpoint(folder / "prepared.json")
                        observations = read_checkpoint(folder / "observations.json")
                        aligned = read_checkpoint(folder / "alignment.json")
                        bundle = read_checkpoint(bundle_path)
                    else:
                        prepared = preparer.prepare_orphan_media(source) if orphan else preparer.prepare_record(source)
                        if prepared["status"] != "complete":
                            raise ValueError(json.dumps(prepared["errors"], ensure_ascii=False))
                        if args.modality == "images" and prepared["videos"]:
                            raise ValueError("图片执行范围中出现视频，拒绝继续识别")
                        prepared["retain_media"] = args.keep_media
                        write_json(folder / "prepared.json", prepared)
                        observations = observe_record(prepared, folder, models, update)
                        vectors = models.text_embeddings([step["text"] for step in prepared["steps"]])
                        if len(vectors) != len(prepared["steps"]):
                            raise ValueError("原步骤数量与真实文本向量数量不一致")
                        steps = [dict(step, embedding=embedding) for step, embedding in zip(prepared["steps"], vectors)]
                        aligned = align_multimodal({"schema_version": 1, "embedding_space": models.embedding_space,
                                                   "steps": steps, "media": observations, "entities": metadata["entities"],
                                                   "config": {"similarity_threshold": args.similarity_threshold, "top_k": 3}})
                        write_json(folder / "alignment.json", aligned)
                        bundle = build_multimodal_bundle(prepared, observations, aligned, metadata, run_id,
                                                         {**models.model_info, "pipeline_run_id": pipeline_id, "embedding_space": models.embedding_space})
                        write_json(bundle_path, bundle)
                    if args.modality == "images" and (prepared["videos"] or any(row["kind"] != "image" for row in observations)):
                        raise ValueError("图片检查点中混入视频或窗口观察，拒绝入库")
                    expected_observations = {image["media_id"] for image in prepared["images"]} | {
                        clip["clip_id"] for video in prepared["videos"] for clip in video["clips"]}
                    actual_observations = {observation["id"] for observation in observations}
                    if actual_observations != expected_observations or len(observations) != len(expected_observations):
                        raise ValueError("真实识别未覆盖全部图片和视频窗口")
                    validated = validate_bundle(bundle, base)
                    imported = import_bundle(session, validated) if session else {"status": "processed", "input": validated.summary()}
                    if session:
                        ensure_vector_index(session)
                        if imported.get("status") != "verified":
                            raise ValueError("Neo4j 写入后回读未通过")
                    write_json(folder / "import-report.json", imported)
                    counts = {**prepared["counts"], "observations": len(observations),
                              "aligned_media": sum(a["status"] == "candidate" for a in aligned["alignments"]),
                              "extension_nodes": len(validated.nodes), "extension_relationships": len(validated.relationships)}
                    state["records"][identifier] = {"status": "verified" if session else "processed", "counts": counts,
                                                    "run_id": run_id, "finished_at": stamp(), "seconds": time.monotonic() - started,
                                                    "report": str((folder / "import-report.json").relative_to(ROOT)),
                                                    "observation_ids": [o["id"] for o in observations]}
                    print(json.dumps({"work_item": identifier, "status": state["records"][identifier]["status"], "counts": counts,
                                      "seconds": state["records"][identifier]["seconds"]}, ensure_ascii=False), flush=True)
                except Exception as error:
                    state["records"][identifier] = {"status": "failed", "attempts": attempt + 1, "error_type": type(error).__name__,
                                                    "error": str(error), "updated_at": stamp()}
                    write_json(folder / "failure.json", state["records"][identifier])
                    print(json.dumps({"work_item": identifier, "status": "failed", "error_type": type(error).__name__,
                                      "error": str(error)[:1200]}, ensure_ascii=False), flush=True)
                finally:
                    try:
                        if prepared is not None and not args.keep_media and prepared.get("storage_state") != "cleaned":
                            cleaned = preparer.cleanup_record(prepared.get("work_id") or prepared["recipe_id"])
                            write_json(folder / "prepared.json", cleaned)
                        if state["records"].get(identifier, {}).get("status") in {"verified", "processed"}:
                            compress_checkpoints(folder)
                    except Exception as error:
                        prior = state["records"].get(identifier, {})
                        state["records"][identifier] = {**prior, "status": "failed", "attempts": attempt + 1,
                                                        "error_type": type(error).__name__, "error": "检查点或临时媒体清理失败: " + str(error),
                                                        "updated_at": stamp()}
                        write_json(folder / "failure.json", state["records"][identifier])
                    finally:
                        state["model_calls"] = dict(models.calls)
                        update(stage="checkpoint_saved")
        finished = update(status="completed" if all(state["records"].get(i, {}).get("status") ==
                                                   ("verified" if session else "processed") for i, _, _ in items) else "partial_failed",
                          stage="finished", finished_at=stamp())
        print(json.dumps(finished, ensure_ascii=False), flush=True)
        return 0 if finished["status"] == "completed" else 1
    finally:
        if session:
            session.close()
        if driver:
            driver.close()
        preparer.close()


if __name__ == "__main__":
    raise SystemExit(main())
