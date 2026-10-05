"""Build a provenance-bearing graph from the official RecipeGen test split.

Media nodes refer to verified ZIP directory entries. No visual recognition or
step/image alignment is inferred from filenames. Ingredient mentions are an
incomplete, rule-extracted view of the source steps, not a complete recipe list.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .kg_extract import extract_steps

REPO = "RUOXUAN123/RecipeGen"
ARCHIVES = ("test.zip", "test-video.zip")
BUILD_FORMAT = "recipegen-test-graph-v1"


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def uid(kind: str, value: str) -> str:
    return f"{kind.lower()}:{digest(value)[:24]}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def folder_key(path: str, archive: str) -> str:
    """Remove only the archive's literal root; preserve the remaining identity."""
    parts = PurePosixPath(path).parts
    if parts and parts[0] == archive[:-4]:
        parts = parts[1:]
    return str(PurePosixPath(*parts))


def build_graph(records_path: Path, output_dir: Path) -> dict:
    source_dir = records_path.parent / "source"
    records = read_jsonl(records_path)
    manifests = [json.loads((source_dir / f"{name[:-4]}.manifest.json").read_text()) for name in ARCHIVES]
    revisions = {m["revision"] for m in manifests}
    if len(revisions) != 1:
        raise ValueError("Archives must share one immutable dataset revision")
    revision = next(iter(revisions))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    for manifest, archive in zip(manifests, ARCHIVES):
        if manifest["archive"] != archive or manifest["split"] != "test" or manifest["status"] != "complete":
            raise ValueError("Both test archive inventories and text imports must be complete")
    manifest_hashes = {f"{name[:-4]}.manifest.json": digest((source_dir / f"{name[:-4]}.manifest.json").read_bytes()) for name in ARCHIVES}
    index_files = [source_dir / f"{name[:-4]}.members.jsonl" for name in ARCHIVES]
    for manifest, index_path, archive in zip(manifests, index_files, ARCHIVES):
        text_path = source_dir / f"{archive[:-4]}.texts.jsonl"
        if digest(index_path.read_bytes()) != manifest.get("index_sha256") or digest(text_path.read_bytes()) != manifest.get("texts_sha256"):
            raise ValueError("Source artifacts do not agree with their complete manifest hashes")
        if manifest.get("recipe_records") != sum(r["source"]["archive"] == archive for r in records):
            # Validate archive identities separately below to give useful errors.
            if all(r["source"]["archive"] in ARCHIVES for r in records):
                raise ValueError("Recipe record count does not agree with source manifest")
    code_hashes = {name: digest((Path(__file__).parent / name).read_bytes()) for name in ("test_graph.py", "kg_extract.py")}
    input_hashes = {"records.jsonl": digest(records_path.read_bytes()), **{p.name: digest(p.read_bytes()) for p in index_files}}
    sample_map = {}
    sample_report = records_path.parent / "media_samples/materialization-report.json"
    if sample_report.is_file():
        for sample in json.loads(sample_report.read_text())["media"]:
            if sample["archive"] not in ARCHIVES or sample["revision"] != revision or sample["split"] != "test" or sample["zip_crc_verified"] is not True:
                raise ValueError("Invalid local test media sample provenance")
            filename = Path(sample["path"]).name
            local = records_path.parent / "media_samples" / filename
            if digest(local.read_bytes()) != sample["sha256"]:
                raise ValueError("Local media sample hash mismatch")
            key = (sample["archive"], sample["member"])
            sample_map[key] = {"sha256": sample["sha256"], "path": f"media_samples/{filename}", "crc": sample["crc32"]}
        input_hashes["verified_media_samples"] = digest(json.dumps([[*key, val["sha256"]] for key, val in sorted(sample_map.items())]))
    build_id = "test-" + digest(json.dumps({"format": BUILD_FORMAT, "revision": revision, "inputs": input_hashes, "code": code_hashes}, sort_keys=True))[:20]
    common = {"split": "test", "dataset_revision": revision, "build_id": build_id}
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    evidence_rows = []
    warnings = []

    def node(node_id: str, label: str, **properties):
        candidate = {"id": node_id, "label": label, "properties": {**common, **properties}}
        if node_id in nodes and nodes[node_id] != candidate:
            raise ValueError(f"Conflicting node identity: {node_id}")
        nodes[node_id] = candidate

    def edge(start: str, relation: str, end: str, **properties):
        edge_id = uid("relationship", f"{start}\n{relation}\n{end}")
        if edge_id in edges:
            raise ValueError(f"Duplicate relationship: {start} {relation} {end}")
        edges[edge_id] = {"id": edge_id, "start_id": start, "end_id": end, "type": relation, "properties": {**common, **properties}}

    def source_node(archive: str, member: str, kind: str, *, sha256: str | None = None):
        sid = uid("source", f"{REPO}\n{archive}\n{member}")
        props = {"source_id": sid, "source_type": f"huggingface_zip_{kind}", "dataset": REPO,
                 "archive": archive, "member": member, "artifact_path": f"{REPO}/{archive}/{member}",
                 "url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(archive, safe='')}"}
        if sha256:
            props["sha256"] = sha256
        node(sid, "Source", **props)
        return sid

    recipe_by_folder: dict[str, list[str]] = defaultdict(list)
    original_media_owner = {}
    recipe_count = Counter()
    extraction_counts = Counter()
    for record in records:
        src = record["source"]
        if record.get("split") != "test" or src["archive"] not in ARCHIVES or src["revision"] != revision or src.get("split") != "test":
            raise ValueError("Non-test or mixed-revision record rejected")
        rid = record["id"]
        if not src.get("goal_member") and not src.get("steps_member"):
            raise ValueError("Recipe record has no original text source")
        if record.get("steps") and not src.get("steps_member"):
            raise ValueError("Non-empty steps require an original steps.txt source")
        incomplete = not record.get("title") or not record.get("steps")
        if incomplete:
            warnings.append({"record_id": rid, "reason": "incomplete_text_retained", "source": src})
        goal_source = source_node(src["archive"], src["goal_member"], "text", sha256=src.get("goal_sha256")) if src.get("goal_member") else None
        step_source = source_node(src["archive"], src["steps_member"], "text", sha256=src.get("steps_sha256")) if src.get("steps_member") else None
        node(rid, "Recipe", recipe_id=rid, title=record["title"], origin_archive=src["archive"],
             recipe_directory=src["recipe_dir"], display_directory=src["display_dir"],
             steps_count=len(record["steps"]), ingredient_status="rule_extracted_incomplete",
             total_cooking_time_status="unknown", media_state="metadata_only", is_demo=False,
             local_media_samples_count=0,
             text_complete=not incomplete, quality_flags=[f for f in record.get("quality_flags", []) if f != "ingredients_not_extracted"] + ["rule_ingredients_are_incomplete"])
        if goal_source:
            edge(rid, "HAS_SOURCE", goal_source, role="title")
        if step_source:
            edge(rid, "HAS_SOURCE", step_source, role="steps")
        recipe_by_folder[folder_key(src["display_dir"], src["archive"])].append(rid)
        recipe_count[src["archive"]] += 1
        for media in record.get("media", []):
            original_media_owner[(media["source"]["archive"], media["member"])] = rid
        grouped = defaultdict(list)
        previous = None
        mentions = extract_steps(record["steps"], source={"id": step_source, "archive": src["archive"], "member": src["steps_member"], "revision": revision})
        by_order = defaultdict(list)
        for mention in mentions:
            by_order[mention["step_order"]].append(mention)
        for position, step in enumerate(record["steps"], 1):
            if step["order"] != position:
                raise ValueError("Steps must have contiguous source-line ordering")
            step_id = uid("step", f"{rid}\n{position}")
            step_mentions = by_order[position]
            node(step_id, "Step", step_id=step_id, recipe_id=rid, order=position, text=step["text"], source_id=step_source,
                 order_method="nonempty_original_lines",
                 duration_mentions=[m["name"] for m in step_mentions if m["kind"] == "Duration"],
                 temperature_mentions=[m["name"] for m in step_mentions if m["kind"] == "Temperature"])
            edge(rid, "HAS_STEP", step_id, order=position)
            if previous:
                edge(previous, "NEXT_STEP", step_id)
            previous = step_id
            step_groups = defaultdict(list)
            for mention in step_mentions:
                evidence_rows.append({"recipe_id": rid, "step_id": step_id, **mention})
                extraction_counts[mention["kind"]] += 1
                if mention["kind"] not in {"Ingredient", "Action", "Tool"}:
                    continue
                key = (mention["kind"], mention["normalized_name"])
                step_groups[key].append(mention)
                if mention["kind"] == "Ingredient":
                    grouped[key].append(mention)
            for (kind, name), hits in step_groups.items():
                eid = uid(kind, name)
                node(eid, kind, name=name, normalized_name=name, zh_name=hits[0].get("zh_name") or "", extraction_method="dictionary_rule")
                rel = {"Ingredient": "USES_INGREDIENT", "Action": "HAS_ACTION", "Tool": "USES_TOOL"}[kind]
                edge(step_id, rel, eid, extraction_method="dictionary_rule", semantics="text_mention_candidate", evidence_json=json.dumps(hits, ensure_ascii=False), verified=False)
        for (kind, name), hits in grouped.items():
            edge(rid, "HAS_INGREDIENT", uid(kind, name), extraction_method="dictionary_rule",
                 evidence_json=json.dumps(hits, ensure_ascii=False), semantics="text_mention_candidate", complete_ingredient_list=False, verified=False)

    media_count = Counter()
    media_link_counts = Counter()
    unmatched = []
    association_audit = []
    for archive, index_path in zip(ARCHIVES, index_files):
        for entry in read_jsonl(index_path):
            if entry["source"] != {"archive": archive, "revision": revision, "split": "test"}:
                raise ValueError("Media inventory source mismatch")
            if entry["kind"] not in {"image", "video"}:
                continue
            mid = uid(entry["kind"], f"{archive}\n{entry['member']}")
            source_id = source_node(archive, entry["member"], "media")
            sample = sample_map.get((archive, entry["member"]))
            if sample and sample["crc"] != entry["crc"]:
                raise ValueError("Local sample CRC identity mismatch")
            node(mid, entry["kind"].title(), media_id=mid, archive=archive, member=entry["member"], display_path=entry["display_path"],
                 size_bytes=entry["size"], compressed_size_bytes=entry["compressed_size"], crc32=entry["crc"],
                 storage="local_verified_sample" if sample else "remote_archive_member", downloaded=bool(sample),
                 local_relative_path=sample["path"] if sample else None, local_sha256=sample["sha256"] if sample else None,
                 recognition_status="not_run", source_id=source_id)
            edge(mid, "HAS_SOURCE", source_id)
            owner = original_media_owner.get((archive, entry["member"]))
            method = "same_exact_archive_directory"
            if not owner:
                key = folder_key(str(PurePosixPath(entry["display_path"]).parent), archive)
                candidates = recipe_by_folder.get(key, [])
                if len(candidates) == 1:
                    owner = candidates[0]
                    method = "exact_cross_archive_relative_directory"
                else:
                    method = "unmatched_or_ambiguous_directory"
            if owner:
                edge(owner, "HAS_IMAGE" if entry["kind"] == "image" else "HAS_VIDEO", mid, association_method=method,
                     association_semantics="directory_membership_only", visual_content_verified=False)
                media_link_counts[entry["kind"]] += 1
                if sample:
                    nodes[owner]["properties"]["local_media_samples_count"] += 1
                    nodes[owner]["properties"]["media_state"] = "metadata_with_verified_local_samples"
            else:
                unmatched.append({"media_id": mid, "archive": archive, "member": entry["member"], "display_path": entry["display_path"], "reason": method})
            media_count[entry["kind"]] += 1
            association_audit.append({"media_id": mid, "recipe_id": owner, "association_method": method, "archive": archive, "member": entry["member"]})

    if any(e["start_id"] not in nodes or e["end_id"] not in nodes for e in edges.values()):
        raise ValueError("Dangling graph relationship")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("nodes.jsonl", list(nodes.values())), ("relationships.jsonl", list(edges.values())), ("extraction_evidence.jsonl", evidence_rows), ("media_associations.jsonl", association_audit)):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_dir = output_dir / "neo4j_import"
    csv_dir.mkdir(exist_ok=True)
    for name, fields, rows in (("nodes.csv", ("id", "label", "properties_json"), nodes.values()),
                              ("relationships.csv", ("id", "start_id", "end_id", "type", "properties_json"), edges.values())):
        with (csv_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{k: row[k] for k in fields if k != "properties_json"}, "properties_json": json.dumps(row["properties"], ensure_ascii=False)})
    report = {"format": BUILD_FORMAT, **common, "dataset": REPO, "is_demo": False,
              "archives": list(ARCHIVES), "training_archives_accessed": [], "input_hashes": input_hashes, "builder_hashes": code_hashes,
              "source_manifest_hashes": manifest_hashes,
              "records_seen": len(records), "recipe_counts_by_archive": dict(recipe_count),
              "node_count": len(nodes), "relationship_count": len(edges),
              "node_counts": dict(Counter(n["label"] for n in nodes.values())),
              "relationship_counts": dict(Counter(e["type"] for e in edges.values())),
              "extraction_mention_counts": dict(extraction_counts), "media_counts": dict(media_count), "associated_media_counts": dict(media_link_counts),
              "verified_local_media_samples": len(sample_map),
              "unassociated_media": unmatched, "incomplete_records": warnings,
              "limits": ["full_media_payload_not_downloaded", "no_visual_semantic_extraction", "no_frame_or_video_action_extraction", "no_verified_step_image_alignment", "rule_ingredients_are_incomplete", "recipe_total_time_unknown", "not_a_held_out_model_evaluation"],
              "structural_validation": {"dangling_relationships": 0, "split": "test", "source_archive_whitelist": True}}
    (output_dir / "build-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
