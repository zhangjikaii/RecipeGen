"""Full-scope ID checks over synthetic original records, with no IO or models."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from recipegen.multimodal_scope import verify_full_scope
from recipegen.test_graph import uid


@pytest.fixture
def scope():
    revision = "a" * 40

    def medium(kind, member, *, orphan=False):
        archive = "test-video.zip" if kind == "video" else "test.zip"
        row = {"kind": kind, "member": member,
               "source": {"archive": archive, "split": "test", "revision": revision}}
        if orphan:
            row["media_id"] = uid(kind, f"{archive}\n{member}")
        return row

    image = medium("image", "recipe-a/frame.jpg")
    video = medium("video", "recipe-a/video.mp4")
    orphan = medium("image", "unassociated/image.jpg", orphan=True)
    records = [{"id": "recipe:fixture-a", "media": [image, video],
                "source": {"archive": "test.zip", "split": "test", "revision": revision}}]
    nodes = {"recipe:fixture-a": {"label": "Recipe"}}
    for item in [image, video, orphan]:
        identifier = uid(item["kind"], f'{item["source"]["archive"]}\n{item["member"]}')
        nodes[identifier] = {"label": "Image" if item["kind"] == "image" else "Video"}
    # Ingredient and source nodes are outside the original media coverage scope.
    nodes["ingredient:fixture"] = {"label": "Ingredient"}
    base = SimpleNamespace(nodes_by_id=nodes, dataset_revision=revision, content_sha256="b" * 64)
    return SimpleNamespace(records=records, orphans=[orphan], base=base)


def test_orphan_image_is_counted_with_exact_original_node_coverage(scope):
    result = verify_full_scope(scope.records, scope.orphans, scope.base)
    assert result["verified"] is True
    assert result["counts"] == {"Recipe": 1, "Image": 2, "Video": 1}
    assert result["orphan_media"] == 1
    assert result["base_content_sha256"] == scope.base.content_sha256
    for kind, expected_hash in result["id_set_sha256"].items():
        ids = sorted(nid for nid, row in scope.base.nodes_by_id.items() if row["label"] == kind)
        assert expected_hash == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def test_id_hashes_do_not_depend_on_record_or_media_file_order(scope):
    first = verify_full_scope(scope.records, scope.orphans, scope.base)
    scope.records[0]["media"].reverse()
    scope.base.nodes_by_id = dict(reversed(list(scope.base.nodes_by_id.items())))
    assert verify_full_scope(scope.records, scope.orphans, scope.base) == first


@pytest.mark.parametrize("missing", ["recipe", "image", "video", "orphan"])
def test_missing_original_recipe_or_media_prevents_full_scope(scope, missing):
    if missing == "recipe":
        scope.records = []
    elif missing == "orphan":
        scope.orphans = []
    else:
        scope.records[0]["media"] = [m for m in scope.records[0]["media"] if m["kind"] != missing]
    with pytest.raises(ValueError, match="覆盖与原图谱不一致"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


@pytest.mark.parametrize("duplicate", ["recipe", "owned_media", "orphan", "owned_as_orphan"])
def test_duplicate_records_and_owned_orphan_media_are_rejected(scope, duplicate):
    if duplicate == "recipe":
        scope.records.append(copy.deepcopy(scope.records[0]))
    elif duplicate == "owned_media":
        scope.records[0]["media"].append(copy.deepcopy(scope.records[0]["media"][0]))
    elif duplicate == "orphan":
        scope.orphans.append(copy.deepcopy(scope.orphans[0]))
    else:
        owned = copy.deepcopy(scope.records[0]["media"][0])
        owned["media_id"] = uid(owned["kind"], f'{owned["source"]["archive"]}\n{owned["member"]}')
        scope.orphans.append(owned)
    with pytest.raises(ValueError, match="重复"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


@pytest.mark.parametrize("target", ["recipe", "owned_media", "orphan"])
@pytest.mark.parametrize("field,value", [("split", "train"), ("archive", "train.zip"), ("revision", "c" * 40)])
def test_wrong_split_archive_or_revision_is_rejected_at_every_source(scope, target, field, value):
    item = scope.records[0] if target == "recipe" else scope.records[0]["media"][0] if target == "owned_media" else scope.orphans[0]
    item["source"][field] = value
    with pytest.raises(ValueError, match="非固定 test 来源"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


def test_wrong_orphan_id_cannot_substitute_a_real_original_media_node(scope):
    scope.orphans[0]["media_id"] = "image:made-up"
    with pytest.raises(ValueError, match="身份不一致"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


def test_different_equal_count_original_ids_do_not_pass_full_scope(scope):
    scope.records[0]["media"][0]["member"] = "different/member.jpg"
    with pytest.raises(ValueError, match="Image ID 覆盖"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


def test_unknown_media_kind_is_rejected(scope):
    scope.records[0]["media"][0]["kind"] = "audio"
    with pytest.raises(ValueError, match="未知媒体类型"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


@pytest.mark.parametrize("kind", ["Recipe", "Image", "Video", "all"])
def test_empty_original_node_set_cannot_claim_full_completion(scope, kind):
    scope.base.nodes_by_id = {nid: row for nid, row in scope.base.nodes_by_id.items()
                             if kind != "all" and row["label"] != kind}
    with pytest.raises(ValueError, match="覆盖与原图谱不一致"):
        verify_full_scope(scope.records, scope.orphans, scope.base)


def test_zero_selected_and_zero_original_graph_cannot_verify_as_full(scope):
    scope.base.nodes_by_id = {}
    with pytest.raises(ValueError, match="expected=0, selected=0"):
        verify_full_scope([], [], scope.base)
