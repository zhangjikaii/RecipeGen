"""Reuse evidence-bound observations across compatible test pipeline runs.

Only observations are copied. A new controller must prepare its own inputs,
recompute alignments/bundle, and perform a fresh append-only import and readback.
Failed source work items may contain useful *valid* partial observation caches.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .test_graph import ARCHIVES, REPO, uid


class ObservationReuseError(ValueError):
    """A run, input identity, or saved observation cannot be reused safely."""


REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
MATCH_FIELDS = ("schema", "build_id", "dataset_revision", "records_sha256", "image_side", "clip_seconds", "frames_per_clip", "temporary_media_retention")
MAX_OBSERVATION_BYTES = 256 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _safe_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and not any(ord(c) < 32 for c in value)


def _number(value, *, nonnegative=False):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (not nonnegative or value >= 0)
    except OverflowError:
        return False


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ObservationReuseError("duplicate_json_key")
        result[key] = value
    return result


def _parse(text):
    def invalid(value):
        raise ObservationReuseError("nonfinite_json_value")
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)
    except (ValueError, TypeError) as error:
        raise ObservationReuseError("invalid_strict_json") from error


def _read(path, maximum=MAX_OBSERVATION_BYTES):
    # Limit decompression too: a damaged gzip must not expand without bounds.
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ObservationReuseError("checkpoint_too_large")
        return _parse(raw.decode("utf-8"))
    except (OSError, EOFError, UnicodeError) as error:
        raise ObservationReuseError("checkpoint_unreadable") from error


def _inside(path, parent):
    # Every path is derived from a safe ID. Reject symlink aliases, including
    # aliases to a different item inside the same run, before reading/writing.
    absolute = path.absolute()
    if absolute.resolve() != absolute or not absolute.is_relative_to(parent.resolve()):
        raise ObservationReuseError("unsafe_checkpoint_path")
    return absolute


def _modalities(signature):
    # Older mixed runs did not spell out modalities in their signatures.
    value = signature.get("processing_modalities", ["image", "video"])
    if not isinstance(value, list) or value not in (["image"], ["image", "video"]):
        raise ObservationReuseError("signature_processing_modalities_invalid")
    return tuple(value)


def _signature(signature):
    if not isinstance(signature, dict) or any(k not in signature for k in MATCH_FIELDS):
        raise ObservationReuseError("signature_required_fields_missing")
    if signature["schema"] != "recipegen-mm-v1" or signature["dataset_revision"] != REVISION or not _text(signature["build_id"]) or not _hash(signature["records_sha256"]):
        raise ObservationReuseError("signature_scope_invalid")
    _modalities(signature)
    if type(signature["image_side"]) is not int or signature["image_side"] <= 0 or type(signature["frames_per_clip"]) is not int or signature["frames_per_clip"] <= 0 or not _number(signature["clip_seconds"]) or signature["clip_seconds"] <= 0 or type(signature["temporary_media_retention"]) is not bool:
        raise ObservationReuseError("signature_sampling_invalid")
    models = signature.get("models")
    if not isinstance(models, dict):
        raise ObservationReuseError("signature_models_missing")
    for kind in ("vision", "embedding"):
        model = models.get(kind)
        if not isinstance(model, dict) or not _text(model.get("repo")) or not isinstance(model.get("revision"), str) or re.fullmatch(r"[a-f0-9]{40}", model["revision"]) is None:
            raise ObservationReuseError("signature_model_revision_invalid")
    code_hashes = signature.get("code_hashes")
    media_hash = code_hashes.get("recipegen/multimodal_media.py") if isinstance(code_hashes, dict) else None
    if not _hash(media_hash):
        raise ObservationReuseError("signature_media_code_hash_missing")
    # Validate nested JSON in a caller-supplied target signature as well.
    try:
        json.dumps(signature, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ObservationReuseError("signature_invalid_json") from error


def _dataset_items(root, records_sha):
    path = _inside(root / "data/test_graph/records.jsonl", root)
    if _sha(path) != records_sha:
        raise ObservationReuseError("current_records_sha256_mismatch")
    items, owners = {}, {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = _parse(line)
            source, identifier = record.get("source", {}), record.get("id")
            if not _text(identifier) or identifier in items or record.get("split") != "test" or source.get("split") != "test" or source.get("revision") != REVISION or source.get("archive") not in ARCHIVES or not isinstance(record.get("media"), list):
                raise ObservationReuseError("dataset_record_invalid")
            media = {}
            for entry in record["media"]:
                provenance = entry.get("source", {})
                archive, member, kind = provenance.get("archive"), entry.get("member"), entry.get("kind")
                if archive not in ARCHIVES or provenance.get("revision") != REVISION or provenance.get("split") != "test" or not _text(member) or kind not in {"image", "video"}:
                    raise ObservationReuseError("dataset_media_invalid")
                mid = uid(kind, f"{archive}\n{member}")
                if mid in owners:
                    raise ObservationReuseError("dataset_media_duplicate")
                media[mid] = {"source_id": uid("source", f"{REPO}\n{archive}\n{member}"), "archive": archive, "member": member, "kind": kind}
                owners[mid] = identifier
            items[identifier] = media
    associations = root / "data/test_graph/media_associations.jsonl"
    if associations.is_file():
        _inside(associations, root)
        with associations.open(encoding="utf-8") as stream:
            for line in stream:
                row = _parse(line)
                if row.get("recipe_id") is not None:
                    continue
                mid, archive, member = row.get("media_id"), row.get("archive"), row.get("member")
                if not isinstance(mid, str) or not mid.startswith(("image:", "video:")) or archive not in ARCHIVES or not _text(member):
                    raise ObservationReuseError("orphan_media_invalid")
                kind = mid.split(":", 1)[0]
                if mid != uid(kind, f"{archive}\n{member}") or mid in items or mid in owners:
                    raise ObservationReuseError("orphan_media_identity_mismatch")
                items[mid] = {mid: {"source_id": uid("source", f"{REPO}\n{archive}\n{member}"), "archive": archive, "member": member, "kind": kind}}
    if _sha(path) != records_sha:
        raise ObservationReuseError("records_changed_during_reuse")
    return items


def _observation(row, allowed, signature):
    if not isinstance(row, dict) or not _text(row.get("id")):
        raise ObservationReuseError("observation_id_missing")
    mid = row.get("source_media_id")
    source = allowed.get(mid) if isinstance(mid, str) else None
    if source is None or row.get("source_id") != source["source_id"]:
        raise ObservationReuseError("observation_media_source_mismatch")
    kind = row.get("kind")
    if kind == "image":
        if source["kind"] != "image" or row["id"] != mid:
            raise ObservationReuseError("observation_image_identity_invalid")
    elif kind == "clip":
        start, end = row.get("start_seconds"), row.get("end_seconds")
        if source["kind"] != "video" or row.get("timeline_id") != mid or not _number(start, nonnegative=True) or not _number(end, nonnegative=True) or end <= start or row["id"] != uid("video_clip", f"{mid}\n{start:.9f}\n{end:.9f}"):
            raise ObservationReuseError("observation_clip_identity_invalid")
    else:
        raise ObservationReuseError("observation_kind_invalid")
    model = row.get("model", {})
    vision, embedding = signature["models"]["vision"], signature["models"]["embedding"]
    if not isinstance(model, dict) or model.get("name") != vision["repo"] or model.get("revision") != vision["revision"] or model.get("image_side") != signature["image_side"]:
        raise ObservationReuseError("observation_vision_model_mismatch")
    space = row.get("embedding_space", {})
    if not isinstance(space, dict) or space.get("model") != embedding["repo"] or space.get("revision") != embedding["revision"] or type(space.get("dimension")) is not int or space["dimension"] != 512:
        raise ObservationReuseError("observation_embedding_model_mismatch")
    vector = row.get("embedding")
    if not isinstance(vector, list) or len(vector) != 512 or any(not _number(v) for v in vector) or not any(v != 0 for v in vector):
        raise ObservationReuseError("observation_embedding_invalid")
    output, caption, objects, actions = row.get("raw_output"), row.get("caption"), row.get("objects"), row.get("actions")
    if not isinstance(output, dict) or not isinstance(caption, str) or not caption.strip() or not isinstance(objects, list) or any(not isinstance(o, dict) or not _text(o.get("name")) for o in objects) or not isinstance(actions, list) or any(not _text(a) for a in actions) or output.get("caption") != caption or output.get("objects") != objects or output.get("actions") != actions:
        raise ObservationReuseError("observation_output_incomplete_or_inconsistent")
    if not isinstance(row.get("raw_text"), str) or not row["raw_text"].strip():
        raise ObservationReuseError("observation_raw_text_missing")
    evidence = row.get("evidence")
    if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in {"source_media_id": mid, "source_id": source["source_id"], "archive": source["archive"], "member": source["member"], "revision": REVISION, "split": "test"}.items()) or evidence.get("zip_crc_verified") is not True or not _hash(evidence.get("media_sha256")) or evidence.get("local_inference") is not True or evidence.get("raw_model_response") != row["raw_text"]:
        raise ObservationReuseError("observation_evidence_invalid")
    hashes = evidence.get("model_input_sha256")
    if not isinstance(hashes, list) or not hashes or any(not _hash(v) for v in hashes) or len(hashes) > signature["frames_per_clip"] or (kind == "image" and len(hashes) != 1):
        raise ObservationReuseError("observation_input_hash_invalid")
    # Reject Infinity from exponent overflow (e.g. 1e400), not just NaN tokens.
    try:
        json.dumps(row, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ObservationReuseError("observation_nonfinite_or_nonjson") from error


def _atomic_create(path, rows):
    payload = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".observations-reuse-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        # link is an atomic no-clobber publication on the same filesystem.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_observations(root, source_run_id, target_run_dir, target_signature, selected_ids) -> dict:
    """Copy only valid observations, attaching immutable old-file provenance.

    Global scope/path defects raise ``ObservationReuseError`` before copying.
    Invalid item caches return ``rejected`` and remain untouched. A valid cache
    may be partial; its count is explicit and never becomes a verified state.
    Existing target observations (plain or gzip) always win over copied data.
    An image-only target drops clip rows explicitly; their saved observations
    cannot establish success for an image work item or enter its new bundle.
    """
    root = Path(root).resolve()
    if not _safe_id(source_run_id):
        raise ObservationReuseError("unsafe_source_run_id")
    runs = _inside(root / "data/multimodal/runs", root)
    source_run = _inside(runs / source_run_id, runs)
    target_run = Path(target_run_dir)
    if not target_run.is_absolute():
        target_run = root / target_run
    target_run = _inside(target_run, runs)
    if target_run.parent != runs or not _safe_id(target_run.name) or target_run == source_run:
        raise ObservationReuseError("unsafe_or_same_target_run")
    source_path = _inside(source_run / "signature.json", source_run)
    source_signature = _read(source_path, 2 * 1024 * 1024)
    _signature(source_signature)
    _signature(target_signature)
    image_only = _modalities(target_signature) == ("image",)
    if any(source_signature[k] != target_signature[k] for k in MATCH_FIELDS) or any(source_signature["models"][kind] != target_signature["models"][kind] for kind in ("vision", "embedding")) or source_signature["code_hashes"]["recipegen/multimodal_media.py"] != target_signature["code_hashes"]["recipegen/multimodal_media.py"]:
        raise ObservationReuseError("incompatible_source_target_signatures")
    target_signature_path = _inside(target_run / "signature.json", target_run)
    if target_signature_path.is_file() and _read(target_signature_path, 2 * 1024 * 1024) != target_signature:
        raise ObservationReuseError("target_signature_file_mismatch")
    state_path = _inside(source_run / "state.json", source_run)
    state = _read(state_path, 64 * 1024 * 1024)
    if not isinstance(state, dict) or state.get("pipeline_run_id") != source_run_id or state.get("signature") != source_signature or not isinstance(state.get("records"), dict):
        raise ObservationReuseError("source_state_or_signature_mismatch")
    if isinstance(selected_ids, (str, bytes)):
        raise ObservationReuseError("selection_must_be_ids_iterable")
    try:
        selected = list(selected_ids)
    except TypeError as error:
        raise ObservationReuseError("selection_must_be_ids_iterable") from error
    if any(not _text(v) for v in selected) or len(set(selected)) != len(selected):
        raise ObservationReuseError("selection_invalid_or_duplicate")
    dataset = _dataset_items(root, target_signature["records_sha256"])
    if any(identifier not in dataset for identifier in selected):
        raise ObservationReuseError("selection_outside_original_dataset")
    signature_sha = _sha(source_path)
    if _read(source_path, 2 * 1024 * 1024) != source_signature:
        raise ObservationReuseError("source_signature_changed_during_reuse")
    result = {"status": "skipped", "source_run_id": source_run_id, "target_run_id": target_run.name,
              "copied": 0, "skipped": 0, "rejected": 0, "observations": 0, "filtered_video_observations": 0,
              "processing_modalities": list(_modalities(target_signature)), "items": [], "success_states_copied": False}
    for identifier in selected:
        item = {"work_item": identifier, "filtered_video_observations": 0}
        result["items"].append(item)
        try:
            folder = _digest(identifier)[:24]
            destination_dir = _inside(target_run / folder, target_run)
            destination = _inside(destination_dir / "observations.json", destination_dir)
            if destination.is_file() or destination.with_suffix(".json.gz").exists():
                item.update(status="skipped", reason="target_checkpoint_exists")
                result["skipped"] += 1
                continue
            if identifier not in state["records"]:
                item.update(status="skipped", reason="not_in_source_state")
                result["skipped"] += 1
                continue
            origin_dir = _inside(source_run / folder, source_run)
            plain = _inside(origin_dir / "observations.json", origin_dir)
            compressed = _inside(origin_dir / "observations.json.gz", origin_dir)
            if plain.is_file() and compressed.is_file():
                raise ObservationReuseError("ambiguous_source_checkpoints")
            origin = plain if plain.is_file() else compressed
            if not origin.is_file():
                item.update(status="skipped", reason="source_checkpoint_missing")
                result["skipped"] += 1
                continue
            source_sha = _sha(origin)
            rows = _read(origin)
            if not isinstance(rows, list) or not rows:
                raise ObservationReuseError("source_observations_empty_or_not_list")
            if image_only:
                if any(not isinstance(row, dict) or row.get("kind") not in {"image", "clip"} for row in rows):
                    raise ObservationReuseError("observation_kind_invalid")
                filtered = sum(row["kind"] == "clip" for row in rows)
                item["filtered_video_observations"] = filtered
                result["filtered_video_observations"] += filtered
                rows = [row for row in rows if row["kind"] == "image"]
                if not rows:
                    item.update(status="skipped", reason="no_image_observations", observations=0)
                    result["skipped"] += 1
                    continue
            ids = set()
            for row in rows:
                _observation(row, dataset[identifier], source_signature)
                if row["id"] in ids:
                    raise ObservationReuseError("duplicate_observation_id")
                ids.add(row["id"])
            if _sha(origin) != source_sha or _sha(source_path) != signature_sha:
                raise ObservationReuseError("source_changed_during_copy")
            copied_rows = copy.deepcopy(rows)
            provenance = {"pipeline_run_id": source_run_id, "signature_sha256": signature_sha, "observation_file_sha256": source_sha}
            for row in copied_rows:
                previous = row["evidence"].get("reused_observation_from")
                if previous is not None:
                    history = row["evidence"].get("reused_observation_history", [])
                    if not isinstance(history, list):
                        raise ObservationReuseError("invalid_reuse_history")
                    row["evidence"]["reused_observation_history"] = [*history, previous]
                row["evidence"]["reused_observation_from"] = dict(provenance)
            destination_dir.mkdir(parents=True, exist_ok=True)
            _atomic_create(destination, copied_rows)
            item.update(status="copied", observations=len(rows), source_file=str(origin.relative_to(root)), target_file=str(destination.relative_to(root)))
            result["copied"] += 1
            result["observations"] += len(rows)
        except FileExistsError:
            item.update(status="skipped", reason="target_checkpoint_exists")
            result["skipped"] += 1
        except (ObservationReuseError, OSError, ValueError, TypeError, KeyError) as error:
            item.update(status="rejected", reason=str(error) if isinstance(error, ObservationReuseError) else "checkpoint_invalid_or_copy_failed")
            result["rejected"] += 1
    if result["rejected"]:
        result["status"] = "rejected" if not result["copied"] else "partial"
    elif result["copied"]:
        result["status"] = "copied"
    return result
