import json
import hashlib
from pathlib import Path

import pytest

from recipegen.test_graph import build_graph, read_jsonl

REVISION = "a" * 40

def fixture_corpus(tmp_path: Path, *, archive="test.zip", steps_source=True, title="Tomato bowl"):
    root = tmp_path / "corpus"
    source = root / "source"
    source.mkdir(parents=True)
    for name in ("test.zip", "test-video.zip"):
        media = [{"member": f"{name[:-4]}/dish/{'0.jpg' if name == 'test.zip' else 'video.mp4'}", "display_path": f"{name[:-4]}/dish/{'0.jpg' if name == 'test.zip' else 'video.mp4'}",
                  "kind": "image" if name == "test.zip" else "video", "size": 100, "compressed_size": 80, "crc": "00000000",
                  "source": {"archive": name, "revision": REVISION, "split": "test"}}]
        (source / f"{name[:-4]}.members.jsonl").write_text("\n".join(json.dumps(m) for m in media) + "\n")
        (source / f"{name[:-4]}.texts.jsonl").write_text("\n")
        (source / f"{name[:-4]}.manifest.json").write_text(json.dumps({"archive": name, "split": "test", "revision": REVISION, "status": "complete",
            "recipe_records": int(name == "test.zip"),
            "index_sha256": hashlib.sha256((source / f"{name[:-4]}.members.jsonl").read_bytes()).hexdigest(),
            "texts_sha256": hashlib.sha256((source / f"{name[:-4]}.texts.jsonl").read_bytes()).hexdigest()}))
    record = {"id": "recipe:test:dish", "split": "test", "title": title, "steps": [{"order": 1, "text": "Chop tomatoes and cook in a pan for 5 minutes."}],
              "source": {"archive": archive, "split": "test", "revision": REVISION, "recipe_dir": "test/dish", "display_dir": "test/dish", "goal_member": "test/dish/goal.txt",
                         "steps_member": "test/dish/steps.txt" if steps_source else None, "goal_sha256": "b" * 64, "steps_sha256": "c" * 64}, "media": [], "quality_flags": []}
    (root / "records.jsonl").write_text(json.dumps(record) + "\n")
    return root, record

def test_source_evidence_media_and_unknown_total_time(tmp_path):
    root, _ = fixture_corpus(tmp_path)
    report = build_graph(root / "records.jsonl", root)
    nodes = read_jsonl(root / "nodes.jsonl")
    edges = read_jsonl(root / "relationships.jsonl")
    assert report["media_counts"] == {"image": 1, "video": 1}
    assert report["associated_media_counts"] == {"image": 1, "video": 1}
    recipe = next(n for n in nodes if n["label"] == "Recipe")
    assert recipe["properties"]["total_cooking_time_status"] == "unknown"
    assert "minutes" not in recipe["properties"]
    step = next(n for n in nodes if n["label"] == "Step")
    assert step["properties"]["duration_mentions"] == ["5 minutes"]
    assert not any(n["label"] in ("Frame", "VideoClip", "VisualObject") for n in nodes)
    assert not any(e["type"] == "STEP_IMAGE" for e in edges)
    inferred = [e for e in edges if e["type"] == "HAS_INGREDIENT"]
    assert inferred and all(e["properties"]["verified"] is False and e["properties"]["complete_ingredient_list"] is False for e in inferred)
    sources = {n["id"] for n in nodes if n["label"] == "Source"}
    assert step["properties"]["source_id"] in sources
    mentions = read_jsonl(root / "extraction_evidence.jsonl")
    for m in mentions:
        evidence = m["evidence"]
        assert evidence["text"][evidence["span_start"]:evidence["span_end"]] == m["name"]
    assert build_graph(root / "records.jsonl", root)["build_id"] == report["build_id"]

@pytest.mark.parametrize("archive", ["train-image1.zip", "video1.zip", "../test.zip"])
def test_non_test_sources_are_rejected(tmp_path, archive):
    root, _ = fixture_corpus(tmp_path, archive=archive)
    with pytest.raises(ValueError, match="Non-test"):
        build_graph(root / "records.jsonl", root)

def test_nonempty_steps_without_source_rejected(tmp_path):
    root, _ = fixture_corpus(tmp_path, steps_source=False)
    with pytest.raises(ValueError, match="steps.txt"):
        build_graph(root / "records.jsonl", root)

def test_missing_title_remains_missing(tmp_path):
    root, _ = fixture_corpus(tmp_path, title=None)
    report = build_graph(root / "records.jsonl", root)
    recipe = next(n for n in read_jsonl(root / "nodes.jsonl") if n["label"] == "Recipe")
    assert recipe["properties"]["title"] is None
    assert recipe["properties"]["text_complete"] is False
    assert len(report["incomplete_records"]) == 1

def test_changed_source_artifact_is_rejected(tmp_path):
    root, _ = fixture_corpus(tmp_path)
    (root / "source/test.texts.jsonl").write_text("altered source bytes")
    with pytest.raises(ValueError, match="manifest hashes"):
        build_graph(root / "records.jsonl", root)
