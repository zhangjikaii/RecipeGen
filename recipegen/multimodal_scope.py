"""Require exact original node-ID coverage before declaring a full test run."""
from __future__ import annotations

import hashlib
import json
from .test_graph import ARCHIVES, uid


def verify_full_scope(records, orphans, base):
    expected = {kind: {nid for nid, row in base.nodes_by_id.items() if row["label"] == kind}
                for kind in ("Recipe", "Image", "Video")}
    actual = {kind: set() for kind in expected}

    def check_source(source):
        if (source.get("archive") not in ARCHIVES or source.get("split") != "test"
                or source.get("revision") != base.dataset_revision):
            raise ValueError("全量选择出现非固定 test 来源")

    def add_medium(medium, *, orphan=False):
        check_source(medium["source"])
        kind = {"image": "Image", "video": "Video"}.get(medium["kind"])
        if kind is None:
            raise ValueError("未知媒体类型")
        nid = uid(medium["kind"], f'{medium["source"]["archive"]}\n{medium["member"]}')
        if orphan and medium.get("media_id") != nid:
            raise ValueError("孤立媒体 ID 与原始成员身份不一致")
        if nid in actual[kind]:
            raise ValueError("原媒体被重复计入全量覆盖")
        actual[kind].add(nid)

    for record in records:
        check_source(record["source"])
        if record["id"] in actual["Recipe"]:
            raise ValueError("原 Recipe ID 重复")
        actual["Recipe"].add(record["id"])
        for medium in record["media"]:
            add_medium(medium)
    for medium in orphans:
        add_medium(medium, orphan=True)
    for kind in expected:
        if not expected[kind] or actual[kind] != expected[kind]:
            raise ValueError(f"全量 {kind} ID 覆盖与原图谱不一致: expected={len(expected[kind])}, selected={len(actual[kind])}")
    return {"verified": True, "base_content_sha256": base.content_sha256,
            "counts": {kind: len(ids) for kind, ids in actual.items()}, "orphan_media": len(orphans),
            "id_set_sha256": {kind: hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
                              for kind, ids in actual.items()}}
