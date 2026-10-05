"""Reuse only validated observation checkpoints; never reuse graph artifacts."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from recipegen import multimodal_reuse as reuse
from recipegen.test_graph import uid


REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
RECIPE = "recipe:fixture"
SOURCE_RUN = "source-run"
TARGET_RUN = "target-run"
VISION = {"repo": "fixture/vision", "revision": "b" * 40}
CLIP = {"repo": "openai/clip-vit-base-patch32", "revision": "c" * 40}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_folder(identifier):
    return hashlib.sha256(json.dumps(identifier, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()[:24]


def write_observations(fixture, observations=None, compressed=False):
    path = fixture["source_dir"] / item_folder(RECIPE) / "observations.json"
    raw = json.dumps(observations if observations is not None else fixture["observations"],
                     ensure_ascii=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        path = path.with_suffix(".json.gz")
        path.write_bytes(gzip.compress(raw, mtime=0))
    else:
        path.write_bytes(raw)
    return path


@pytest.fixture
def observation_fixture(tmp_path):
    graph = tmp_path / "data/test_graph"
    archive = "test-video.zip"
    media = [{"kind": "image", "member": "fixture/image.jpg", "source": {"archive": archive, "revision": REVISION, "split": "test"}},
             {"kind": "video", "member": "fixture/video.mp4", "source": {"archive": archive, "revision": REVISION, "split": "test"}}]
    record = {"id": RECIPE, "split": "test", "source": {"archive": archive, "revision": REVISION, "split": "test"}, "media": media}
    graph.mkdir(parents=True)
    (graph / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    signature = {"schema": "recipegen-mm-v1", "build_id": "fixture-build", "dataset_revision": REVISION,
                 "records_sha256": sha(graph / "records.jsonl"), "models": copy.deepcopy({"vision": VISION, "embedding": CLIP}),
                 "image_side": 448, "clip_seconds": 5.0, "frames_per_clip": 3,
                 "temporary_media_retention": False, "similarity_threshold": 0.25,
                 "code_hashes": {"old.py": "d" * 64, "recipegen/multimodal_media.py": "a" * 64},
                 "graph_target": {"database": "old"}}
    target_signature = copy.deepcopy(signature)
    target_signature.update(code_hashes={"new.py": "e" * 64, "recipegen/multimodal_media.py": "a" * 64},
                            graph_target={"database": "new"}, similarity_threshold=0.6)
    source_dir = tmp_path / "data/multimodal/runs" / SOURCE_RUN
    target_dir = source_dir.parent / TARGET_RUN
    write_json(source_dir / "signature.json", signature)
    write_json(source_dir / "state.json", {"pipeline_run_id": SOURCE_RUN, "signature": signature,
                                          "records": {RECIPE: {"status": "failed"}}, "must_not_be_copied": True})
    write_json(target_dir / "signature.json", target_signature)
    observations = []
    for item in media:
        kind, member = item["kind"], item["member"]
        media_id = uid(kind, archive + "\n" + member)
        source_id = uid("source", "RUOXUAN123/RecipeGen\n" + archive + "\n" + member)
        identifier = media_id if kind == "image" else uid("video_clip", f"{media_id}\n0.000000000\n5.000000000")
        raw_output = {"caption": "Fixture observation", "objects": [], "actions": [], "relations": []}
        evidence = {"source_media_id": media_id, "source_id": source_id, "archive": archive, "member": member,
                    "revision": REVISION, "split": "test", "media_sha256": "f" * 64,
                    "temporary_input_retained": False, "zip_crc_verified": True, "local_inference": True,
                    "raw_model_response": json.dumps(raw_output),
                    "model_input_sha256": ["1" * 64] * (1 if kind == "image" else 3)}
        observation = {"id": identifier, "kind": "image" if kind == "image" else "clip",
                       "source_media_id": media_id, "source_id": source_id,
                       "caption": raw_output["caption"], "objects": [], "actions": [],
                       "raw_output": raw_output, "raw_text": json.dumps(raw_output),
                       "model": {"name": VISION["repo"], "revision": VISION["revision"], "image_side": 448},
                       "embedding_space": {"model": CLIP["repo"], "revision": CLIP["revision"], "dimension": 512},
                       "embedding": [1.0] + [0.0] * 511, "evidence": evidence}
        if kind == "video":
            observation.update(timeline_id=media_id, start_seconds=0.0, end_seconds=5.0)
            evidence.update(start_sec=0.0, end_sec=5.0)
        observations.append(observation)
    return {"root": tmp_path, "source_dir": source_dir, "target_dir": target_dir,
            "signature": signature, "target_signature": target_signature, "observations": observations}


def seed(fixture, selected=None):
    return reuse.seed_observations(fixture["root"], SOURCE_RUN, fixture["target_dir"],
                                   fixture["target_signature"], selected if selected is not None else [RECIPE])


@pytest.mark.parametrize("compressed", [False, True])
def test_plain_and_gzip_reuse_preserves_raw_observations_and_only_adds_provenance(observation_fixture, compressed):
    f = observation_fixture
    source_file = write_observations(f, compressed=compressed)
    source_bytes = source_file.read_bytes()
    old_folder = source_file.parent
    for name in ("bundle.json", "import-report.json", "alignment.json", "state.json"):
        write_json(old_folder / name, {"must_not_be_copied": True})
    result = seed(f)
    assert result["status"] == "copied"
    assert result["success_states_copied"] is False
    assert (result["copied"], result["skipped"], result["rejected"], result["observations"]) == (1, 0, 0, 2)
    target_file = f["target_dir"] / item_folder(RECIPE) / "observations.json"
    actual = json.loads(target_file.read_text())
    for received, original in zip(actual, f["observations"]):
        provenance = received["evidence"].pop("reused_observation_from")
        assert provenance == {"pipeline_run_id": SOURCE_RUN,
                              "signature_sha256": sha(f["source_dir"] / "signature.json"),
                              "observation_file_sha256": sha(source_file)}
        assert received == original
    assert source_file.read_bytes() == source_bytes
    assert sorted(path.name for path in target_file.parent.iterdir()) == ["observations.json"]
    assert sorted(path.name for path in f["target_dir"].iterdir()) == [item_folder(RECIPE), "signature.json"]


@pytest.mark.parametrize("compressed", [False, True])
def test_existing_target_checkpoint_is_never_overwritten(observation_fixture, compressed):
    f = observation_fixture
    write_observations(f)
    target_file = f["target_dir"] / item_folder(RECIPE) / ("observations.json.gz" if compressed else "observations.json")
    target_file.parent.mkdir(parents=True)
    existing = b"existing-resume-checkpoint"
    target_file.write_bytes(existing)
    result = seed(f)
    assert result["status"] == "skipped"
    assert result["copied"] == 0 and result["skipped"] == 1
    assert target_file.read_bytes() == existing
    assert sorted(path.name for path in target_file.parent.iterdir()) == [target_file.name]


def test_missing_old_observation_is_an_explicit_skip(observation_fixture):
    result = seed(observation_fixture)
    assert result["status"] == "skipped"
    assert result["skipped"] == 1 and result["rejected"] == 0
    assert result["items"][0]["status"] == "skipped"


@pytest.mark.parametrize("defect", ["model", "embedding_model", "duplicate_id", "bool_vector", "nan_vector", "huge_vector", "short_vector", "zero_vector", "raw_list", "raw_nan", "raw_text", "source_identity"])
def test_invalid_old_observations_are_explicitly_rejected_without_target_write(observation_fixture, defect):
    f = observation_fixture
    rows = copy.deepcopy(f["observations"])
    if defect == "model":
        rows[0]["model"]["revision"] = "0" * 40
    elif defect == "embedding_model":
        rows[0]["embedding_space"]["model"] = "other/embedding"
    elif defect == "duplicate_id":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "bool_vector":
        rows[0]["embedding"][1] = False
    elif defect == "nan_vector":
        rows[0]["embedding"][1] = float("nan")
    elif defect == "huge_vector":
        rows[0]["embedding"][1] = 1 << 4096
    elif defect == "short_vector":
        rows[0]["embedding"].pop()
    elif defect == "zero_vector":
        rows[0]["embedding"] = [0.0] * 512
    elif defect == "raw_list":
        rows[0]["raw_output"] = []
    elif defect == "raw_nan":
        rows[0]["raw_output"]["score"] = float("inf")
    elif defect == "raw_text":
        rows[0]["raw_text"] = 42
    elif defect == "source_identity":
        rows[0]["source_id"] = "source:foreign"
    write_observations(f, rows)
    result = seed(f)
    assert result["status"] == "rejected"
    assert result["rejected"] == 1 and result["copied"] == 0
    assert result["items"][0]["status"] == "rejected" and result["items"][0]["reason"]
    assert not (f["target_dir"] / item_folder(RECIPE) / "observations.json").exists()


@pytest.mark.parametrize("defect", ["source_run_path", "target_path", "unknown_selected", "signature_model", "signature_records", "target_signature", "media_code"])
def test_unsafe_or_incompatible_global_inputs_fail_closed(observation_fixture, defect):
    f = observation_fixture
    write_observations(f)
    source_run, target_dir, selected = SOURCE_RUN, f["target_dir"], [RECIPE]
    if defect == "source_run_path":
        source_run = "../source-run"
    elif defect == "target_path":
        target_dir = f["root"].parent / "unsafe-target"
    elif defect == "unknown_selected":
        selected = ["recipe:unknown"]
    elif defect == "signature_model":
        f["signature"]["models"]["vision"]["revision"] = "1" * 40
        write_json(f["source_dir"] / "signature.json", f["signature"])
    elif defect == "signature_records":
        f["signature"]["records_sha256"] = "1" * 64
        write_json(f["source_dir"] / "signature.json", f["signature"])
    elif defect == "target_signature":
        disk_signature = copy.deepcopy(f["target_signature"])
        disk_signature["image_side"] = 224
        write_json(f["target_dir"] / "signature.json", disk_signature)
    elif defect == "media_code":
        f["signature"]["code_hashes"]["recipegen/multimodal_media.py"] = "2" * 64
        write_json(f["source_dir"] / "signature.json", f["signature"])
    with pytest.raises(reuse.ObservationReuseError):
        reuse.seed_observations(f["root"], source_run, target_dir, f["target_signature"], selected)
    assert not (f["target_dir"] / item_folder(RECIPE) / "observations.json").exists()


def enable_image_only(fixture):
    fixture["target_signature"]["processing_modalities"] = ["image"]
    write_json(fixture["target_dir"] / "signature.json", fixture["target_signature"])


def test_image_only_reuse_filters_clips_and_records_counts(observation_fixture):
    f = observation_fixture
    enable_image_only(f)
    source_file = write_observations(f)
    original_bytes = source_file.read_bytes()
    result = seed(f)
    assert result["status"] == "copied"
    assert result["copied"] == 1 and result["observations"] == 1
    assert result["processing_modalities"] == ["image"]
    assert type(result["filtered_video_observations"]) is int
    assert result["filtered_video_observations"] == 1
    item = result["items"][0]
    assert type(item["filtered_video_observations"]) is int
    assert item["filtered_video_observations"] == 1 and item["observations"] == 1
    target = f["target_dir"] / item_folder(RECIPE) / "observations.json"
    observations = json.loads(target.read_text())
    assert len(observations) == 1 and observations[0]["kind"] == "image"
    provenance = observations[0]["evidence"].pop("reused_observation_from")
    assert provenance["observation_file_sha256"] == sha(source_file)
    assert observations[0] == f["observations"][0]
    assert source_file.read_bytes() == original_bytes


def test_image_only_all_clip_cache_is_a_skip_without_target_write(observation_fixture):
    f = observation_fixture
    enable_image_only(f)
    write_observations(f, [f["observations"][1]])
    result = seed(f)
    assert result["status"] == "skipped"
    assert (result["copied"], result["skipped"], result["rejected"], result["observations"]) == (0, 1, 0, 0)
    assert result["filtered_video_observations"] == 1
    assert result["items"][0]["reason"] == "no_image_observations"
    assert result["items"][0]["observations"] == 0
    assert result["items"][0]["filtered_video_observations"] == 1
    assert not (f["target_dir"] / item_folder(RECIPE) / "observations.json").exists()


def test_image_only_discards_damaged_clip_without_counting_it_as_an_image(observation_fixture):
    f = observation_fixture
    enable_image_only(f)
    rows = copy.deepcopy(f["observations"])
    rows[1] = {"kind": "clip", "source_media_id": "video:foreign", "embedding": [False], "evidence": None}
    write_observations(f, rows)
    result = seed(f)
    assert result["status"] == "copied"
    assert result["observations"] == 1 and result["filtered_video_observations"] == 1
    target = f["target_dir"] / item_folder(RECIPE) / "observations.json"
    copied = json.loads(target.read_text())
    assert [row["id"] for row in copied] == [f["observations"][0]["id"]]


@pytest.mark.parametrize("defect", ["missing_kind", "unknown_kind", "image_source", "image_embedding"])
def test_image_only_rejects_unknown_kinds_and_preserves_image_validation(observation_fixture, defect):
    f = observation_fixture
    enable_image_only(f)
    rows = copy.deepcopy(f["observations"])
    if defect == "missing_kind":
        rows[1].pop("kind")
    elif defect == "unknown_kind":
        rows[1]["kind"] = "video"
    elif defect == "image_source":
        rows[0].pop("source_id")
    else:
        rows[0]["embedding"] = [False] + [0.0] * 511
    write_observations(f, rows)
    result = seed(f)
    assert result["status"] == "rejected"
    assert result["rejected"] == 1 and result["copied"] == 0 and result["observations"] == 0
    assert result["items"][0]["status"] == "rejected" and result["items"][0]["reason"]
    assert not (f["target_dir"] / item_folder(RECIPE) / "observations.json").exists()


def test_image_only_existing_target_skips_source_filtering(observation_fixture):
    f = observation_fixture
    enable_image_only(f)
    old_source = write_observations(f)
    old_source.write_bytes(b"invalid old JSON that must not be read")
    target = f["target_dir"] / item_folder(RECIPE) / "observations.json"
    target.parent.mkdir(parents=True)
    existing = b"existing target resume checkpoint"
    target.write_bytes(existing)
    result = seed(f)
    assert result["status"] == "skipped" and result["skipped"] == 1
    assert result["copied"] == 0 and result["filtered_video_observations"] == 0
    assert result["items"][0]["reason"] == "target_checkpoint_exists"
    assert result["items"][0]["filtered_video_observations"] == 0
    assert target.read_bytes() == existing


def test_image_only_still_requires_matching_media_code_hash(observation_fixture):
    f = observation_fixture
    enable_image_only(f)
    f["signature"]["code_hashes"]["recipegen/multimodal_media.py"] = "2" * 64
    write_json(f["source_dir"] / "signature.json", f["signature"])
    write_observations(f)
    with pytest.raises(reuse.ObservationReuseError):
        seed(f)
    assert not (f["target_dir"] / item_folder(RECIPE) / "observations.json").exists()
