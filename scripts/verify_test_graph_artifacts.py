#!/usr/bin/env python3
"""Streaming provenance/structure audit, NOT semantic extraction accuracy.

Uses a temporary disk SQLite index (8 MiB page cache) and never changes graph
artifacts. Every failure is streamed to the final JSON report without keeping
the corpus or the failure list in RAM.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import resource
import sqlite3
import sys
import tempfile
import time


ALLOWED = {"test.zip", "test-video.zip"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


class Audit:
    def __init__(self, root: Path, db_path: Path):
        self.root = root.resolve()
        self.db = sqlite3.connect(db_path)
        self.db.executescript("""
            PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE; PRAGMA cache_size=-8192;
            CREATE TABLE failures(category TEXT,details TEXT);
            CREATE TABLE raw_sources(archive TEXT,member TEXT,sha TEXT,text TEXT,PRIMARY KEY(archive,member));
            CREATE TABLE raw_lines(archive TEXT,member TEXT,ord INTEGER,text TEXT,raw_line TEXT,PRIMARY KEY(archive,member,ord));
            CREATE TABLE raw_members(archive TEXT,member TEXT,kind TEXT,size INTEGER,compressed INTEGER,crc TEXT,PRIMARY KEY(archive,member));
            CREATE TABLE raw_recipes(id TEXT PRIMARY KEY,archive TEXT,dir TEXT,title TEXT,goal_member TEXT,steps_member TEXT,steps_count INTEGER);
            CREATE TABLE raw_media_assoc(archive TEXT,member TEXT,recipe TEXT,PRIMARY KEY(archive,member));
            CREATE TABLE nodes(id TEXT PRIMARY KEY,label TEXT);
            CREATE TABLE sources(id TEXT PRIMARY KEY,archive TEXT,member TEXT,sha TEXT,revision TEXT,stype TEXT);
            CREATE INDEX source_member ON sources(archive,member);
            CREATE TABLE steps(id TEXT PRIMARY KEY,recipe TEXT,ord INTEGER,text TEXT,source TEXT);
            CREATE INDEX step_origin ON steps(source,ord);
            CREATE TABLE recipes(id TEXT PRIMARY KEY,archive TEXT,dir TEXT,title TEXT,steps_count INTEGER,text_complete INTEGER);
            CREATE TABLE media(id TEXT PRIMARY KEY,kind TEXT,archive TEXT,member TEXT,source TEXT,size INTEGER,compressed INTEGER,crc TEXT);
            CREATE INDEX media_origin ON media(archive,member);
            CREATE TABLE links(start TEXT,end TEXT,type TEXT);
            CREATE INDEX link_end ON links(type,end);
            CREATE INDEX link_start ON links(type,start,end);
            CREATE TABLE rel_ids(id TEXT PRIMARY KEY);
        """)
        self.counts = Counter()
        self.failures = Counter()
        self.node_counts = Counter()
        self.relationship_counts = Counter()
        self.evidence_counts = Counter()
        self.file_hashes = {}
        self.snapshots = {}
        self.normalization_examples = []
        self.media_samples = []
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.revision = self.manifest.get("revision")

    def fail(self, category: str, **details) -> None:
        self.failures[category] += 1
        self.db.execute("INSERT INTO failures VALUES(?,?)", (category, json.dumps(details, ensure_ascii=False)))

    def check(self, condition: bool, category: str, **details) -> bool:
        if not condition:
            self.fail(category, **details)
        return bool(condition)

    def watch(self, path: Path) -> None:
        stat = path.stat()
        self.snapshots[str(path)] = (stat.st_size, stat.st_mtime_ns)

    def rows(self, path: Path):
        self.watch(path)
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    self.fail("invalid_jsonl", file=str(path), line=number, error=str(error))
                    continue
                yield number, row

    def insert(self, sql: str, values: tuple, category: str, **details):
        try:
            self.db.execute(sql, values)
        except sqlite3.IntegrityError as error:
            self.fail(category, error=str(error), **details)

    def source_scope(self, archive, revision, *, identity):
        self.check(archive in ALLOWED, "non_test_archive", identity=identity, archive=archive)
        self.check(revision == self.revision, "revision_mismatch", identity=identity, revision=revision, expected=self.revision)

    def load_originals(self):
        self.watch(self.root / "manifest.json")
        self.check(self.manifest.get("status") == "complete" and self.manifest.get("complete_test_scope") is True,
                   "incomplete_corpus_manifest")
        self.check(set(self.manifest.get("requested_archives", [])) == ALLOWED, "corpus_archive_scope")
        self.check(digest(self.root / "records.jsonl") == self.manifest.get("records_sha256"), "records_hash_mismatch")
        for archive in self.manifest.get("archives", []):
            self.source_scope(archive.get("archive"), archive.get("revision"), identity="archive_manifest")
            self.check(archive.get("status") == "complete", "incomplete_archive_manifest", archive=archive.get("archive"))
            for key, checksum, kind in (("texts_file", "texts_sha256", "texts"), ("index_file", "index_sha256", "members")):
                path = Path(archive[key]).resolve()
                if not self.check(self.root in path.parents, "source_path_outside_corpus", path=str(path)):
                    continue
                actual_hash = digest(path)
                self.file_hashes[str(path)] = actual_hash
                self.check(actual_hash == archive[checksum], "source_file_hash_mismatch", file=str(path))
                for number, row in self.rows(path):
                    self.source_scope(row["source"].get("archive"), row["source"].get("revision"), identity=f"{path.name}:{number}")
                    self.check(row["source"].get("split") == "test", "source_split_mismatch", member=row.get("member"))
                    origin = row["source"]["archive"]
                    if kind == "texts":
                        self.counts["original_text_members"] += 1
                        self.insert("INSERT INTO raw_sources VALUES(?,?,?,?)", (origin, row["member"], row["sha256"], row["text"]),
                                    "duplicate_original_text", member=row["member"])
                        enc = row.get("encoding", "utf-8")
                        candidates = [row["text"].encode("utf-8"), row["text"].encode("utf-8-sig")] if enc == "utf-8-sig" else [row["text"].encode(enc)]
                        self.check(any(hashlib.sha256(value).hexdigest() == row["sha256"] for value in candidates),
                                   "original_text_hash_not_reconstructible", member=row["member"], encoding=enc)
                        if PurePosixPath(row["member"]).name == "steps.txt":
                            order = 0
                            for original_line in row["text"].splitlines():
                                if not original_line.strip():
                                    continue
                                order += 1
                                self.db.execute("INSERT INTO raw_lines VALUES(?,?,?,?,?)", (origin, row["member"], order, original_line.strip(), original_line))
                                self.counts["original_nonempty_step_lines"] += 1
                                if original_line != original_line.strip():
                                    self.counts["original_step_lines_with_outer_whitespace"] += 1
                    else:
                        self.counts["original_zip_members"] += 1
                        self.insert("INSERT INTO raw_members VALUES(?,?,?,?,?,?)", (origin, row["member"], row["kind"], row["size"], row["compressed_size"], row["crc"]),
                                    "duplicate_original_member", member=row["member"])
                self.db.commit()
        for number, row in self.rows(self.root / "records.jsonl"):
            source = row["source"]
            self.source_scope(source.get("archive"), source.get("revision"), identity=row["id"])
            self.counts["original_recipe_records"] += 1
            self.insert("INSERT INTO raw_recipes VALUES(?,?,?,?,?,?,?)", (row["id"], source["archive"], source["recipe_dir"], row["title"], source["goal_member"], source["steps_member"], len(row["steps"])),
                        "duplicate_original_recipe", recipe=row["id"])
            for media in row["media"]:
                self.db.execute("INSERT INTO raw_media_assoc VALUES(?,?,?)", (media["source"]["archive"], media["member"], row["id"]))
        self.db.commit()

    def load_nodes(self):
        path = self.root / "nodes.jsonl"
        for number, row in self.rows(path):
            identity, label, props = row.get("id"), row.get("label"), row.get("properties", {})
            self.counts["graph_nodes"] += 1
            self.node_counts[label] += 1
            if not self.check(isinstance(identity, str) and bool(identity), "missing_node_id", line=number):
                continue
            self.insert("INSERT INTO nodes VALUES(?,?)", (identity, label), "duplicate_node_id", id=identity)
            self.check(props.get("split") == "test", "node_split_mismatch", id=identity)
            self.check(props.get("dataset_revision") == self.revision, "node_revision_mismatch", id=identity)
            if label == "Source":
                self.source_scope(props.get("archive"), props.get("dataset_revision"), identity=identity)
                self.insert("INSERT INTO sources VALUES(?,?,?,?,?,?)", (identity, props.get("archive"), props.get("member"), props.get("sha256"), props.get("dataset_revision"), props.get("source_type")),
                            "duplicate_source_id", id=identity)
            elif label == "Step":
                self.check(type(props.get("order")) is int, "step_order_not_integer", id=identity)
                self.insert("INSERT INTO steps VALUES(?,?,?,?,?)", (identity, props.get("recipe_id"), props.get("order"), props.get("text"), props.get("source_id")),
                            "duplicate_step_id", id=identity)
            elif label == "Recipe":
                self.check(props.get("origin_archive") in ALLOWED, "recipe_archive_mismatch", id=identity)
                self.insert("INSERT INTO recipes VALUES(?,?,?,?,?,?)", (identity, props.get("origin_archive"), props.get("recipe_directory"), props.get("title"), props.get("steps_count"), bool(props.get("text_complete"))),
                            "duplicate_recipe_id", id=identity)
            elif label in {"Image", "Video"}:
                self.source_scope(props.get("archive"), props.get("dataset_revision"), identity=identity)
                self.insert("INSERT INTO media VALUES(?,?,?,?,?,?,?,?)", (identity, label.lower(), props.get("archive"), props.get("member"), props.get("source_id"), props.get("size_bytes"), props.get("compressed_size_bytes"), props.get("crc32")),
                            "duplicate_media_id", id=identity)
                if props.get("local_sha256") is not None or props.get("downloaded"):
                    self.verify_local_media(identity, props)
            if number % 10000 == 0:
                self.db.commit()
        self.db.commit()

    def verify_local_media(self, identity, props):
        self.counts["registered_local_media_samples"] += 1
        relative = props.get("local_relative_path")
        expected = props.get("local_sha256")
        if not self.check(isinstance(relative, str) and bool(relative) and isinstance(expected, str), "incomplete_local_media_registration", id=identity):
            return
        path = (self.root / relative).resolve()
        if not self.check(self.root in path.parents, "local_media_path_outside_corpus", id=identity, path=str(path)):
            return
        if not self.check(path.is_file(), "local_media_file_missing", id=identity, path=str(path)):
            return
        self.watch(path)
        actual = digest(path)
        ok = self.check(actual == expected, "local_media_hash_mismatch", id=identity, expected=expected, actual=actual)
        self.check(path.stat().st_size == props.get("size_bytes"), "local_media_size_mismatch", id=identity)
        if ok:
            self.counts["local_media_hashes_verified"] += 1
        self.media_samples.append({"id": identity, "path": str(path), "sha256": actual, "matches_registered_hash": ok})

    def verify_steps_and_sources(self):
        for identity, archive, member, expected_hash, stype in self.db.execute("SELECT id,archive,member,sha,stype FROM sources"):
            original = self.db.execute("SELECT sha FROM raw_sources WHERE archive=? AND member=?", (archive, member)).fetchone()
            if stype == "huggingface_zip_text":
                self.counts["text_source_nodes_checked"] += 1
                if self.check(original is not None, "text_source_member_missing", source=identity, archive=archive, member=member):
                    if self.check(expected_hash == original[0], "text_source_hash_mismatch", source=identity):
                        self.counts["text_source_hashes_matched"] += 1
            raw_member = self.db.execute("SELECT kind FROM raw_members WHERE archive=? AND member=?", (archive, member)).fetchone()
            self.check(raw_member is not None, "source_not_an_original_zip_member", source=identity)
        for archive, member, count in self.db.execute("""SELECT r.archive,r.member,COUNT(s.id) FROM raw_sources r
                 LEFT JOIN sources s ON s.archive=r.archive AND s.member=r.member AND s.stype='huggingface_zip_text'
                 GROUP BY r.archive,r.member HAVING COUNT(s.id)!=1"""):
            self.fail("original_text_source_coverage_not_exactly_one", archive=archive, member=member, count=count)
        query = """SELECT st.id,st.recipe,st.ord,st.text,st.source,s.archive,s.member,
                          r.text,r.raw_line,orig.archive,orig.steps_member
                   FROM steps st LEFT JOIN sources s ON s.id=st.source
                   LEFT JOIN raw_lines r ON r.archive=s.archive AND r.member=s.member AND r.ord=st.ord
                   LEFT JOIN raw_recipes orig ON orig.id=st.recipe"""
        for identity, recipe, order, text, source, archive, member, expected, raw_line, recipe_archive, recipe_member in self.db.execute(query):
            self.counts["step_nodes_checked"] += 1
            ok = self.check(expected is not None, "step_original_line_missing", id=identity, source=source, order=order)
            ok = self.check(text == expected, "step_text_changed", id=identity, expected=expected, actual=text) and ok
            ok = self.check(archive == recipe_archive and member == recipe_member, "step_source_not_its_recipe_source", id=identity, source=source) and ok
            if ok:
                self.counts["steps_matching_recorded_source_normalization"] += 1
            if text == raw_line:
                self.counts["steps_matching_raw_line_exactly"] += 1
            elif raw_line is not None and text == expected:
                self.counts["steps_with_documented_outer_whitespace_normalization"] += 1
                if len(self.normalization_examples) < 10:
                    self.normalization_examples.append({"step_id": identity, "raw_line": raw_line, "graph_text": text})
        for archive, member, order, count in self.db.execute("""SELECT r.archive,r.member,r.ord,COUNT(st.id) FROM raw_lines r
              LEFT JOIN sources s ON s.archive=r.archive AND s.member=r.member
              LEFT JOIN steps st ON st.source=s.id AND st.ord=r.ord GROUP BY r.archive,r.member,r.ord HAVING COUNT(st.id)!=1"""):
            self.fail("original_step_coverage_not_exactly_one", archive=archive, member=member, order=order, count=count)
        for row in self.db.execute("""SELECT orig.id,orig.archive,orig.dir,orig.title,orig.steps_count,orig.steps_member,
                  r.id,r.archive,r.dir,r.title,r.steps_count,r.text_complete FROM raw_recipes orig LEFT JOIN recipes r ON r.id=orig.id"""):
            identity, archive, directory, title, steps_count, steps_member, node, a, d, t, count, complete = row
            self.check(node is not None, "recipe_record_dropped", id=identity)
            self.check((archive, directory, title, steps_count) == (a, d, t, count), "recipe_origin_or_text_changed", id=identity)
            if steps_member is None:
                self.counts["incomplete_recipe_records_expected"] += 1
                retained = node is not None and count == 0 and not complete
                self.check(retained, "incomplete_recipe_not_preserved", id=identity)
                if retained:
                    self.counts["incomplete_recipe_records_retained"] += 1
        self.db.commit()

    def verify_relationships(self):
        for number, row in self.rows(self.root / "relationships.jsonl"):
            self.counts["graph_relationships"] += 1
            kind, start, end = row.get("type"), row.get("start_id"), row.get("end_id")
            self.relationship_counts[kind] += 1
            self.insert("INSERT INTO rel_ids VALUES(?)", (row.get("id"),), "duplicate_relationship_id", id=row.get("id"))
            props = row.get("properties", {})
            self.check(props.get("split") == "test" and props.get("dataset_revision") == self.revision, "relationship_scope_mismatch", id=row.get("id"))
            for endpoint in (start, end):
                self.check(self.db.execute("SELECT 1 FROM nodes WHERE id=?", (endpoint,)).fetchone() is not None,
                           "dangling_relationship", id=row.get("id"), endpoint=endpoint)
            if kind in {"HAS_SOURCE", "HAS_STEP", "HAS_IMAGE", "HAS_VIDEO", "NEXT_STEP"}:
                self.db.execute("INSERT INTO links VALUES(?,?,?)", (start, end, kind))
            if kind == "HAS_STEP":
                linked = self.db.execute("SELECT recipe FROM steps WHERE id=?", (end,)).fetchone()
                self.check(linked is not None and linked[0] == start, "has_step_recipe_mismatch", id=row.get("id"))
            if number % 10000 == 0:
                self.db.commit()
        self.db.commit()

    def verify_evidence(self):
        for number, row in self.rows(self.root / "extraction_evidence.jsonl"):
            self.counts["extraction_evidence_checked"] += 1
            self.evidence_counts[row.get("kind")] += 1
            step = self.db.execute("""SELECT st.recipe,st.ord,st.text,st.source,s.archive,s.member,s.revision
                                      FROM steps st LEFT JOIN sources s ON s.id=st.source WHERE st.id=?""", (row.get("step_id"),)).fetchone()
            if not self.check(step is not None, "evidence_step_missing", line=number, step_id=row.get("step_id")):
                continue
            recipe, order, text, source_id, archive, member, revision = step
            evidence, source = row.get("evidence", {}), row.get("source", {})
            start, end = evidence.get("span_start"), evidence.get("span_end")
            valid_span = type(start) is int and type(end) is int and 0 <= start < end <= len(text)
            checks = [
                self.check(valid_span, "evidence_span_invalid", line=number),
                self.check(evidence.get("text") == text, "evidence_text_not_step_text", line=number),
                self.check(source.get("id") == source_id, "evidence_source_id_mismatch", line=number),
                self.check((source.get("archive"), source.get("member"), source.get("revision")) == (archive, member, revision), "evidence_source_origin_mismatch", line=number),
                self.check(row.get("recipe_id") == recipe and row.get("step_order") == order, "evidence_recipe_or_order_mismatch", line=number),
                self.check(valid_span and text[start:end] == row.get("name"), "evidence_span_not_name", line=number, name=row.get("name")),
            ]
            self.source_scope(source.get("archive"), source.get("revision"), identity=f"evidence:{number}")
            if all(checks):
                self.counts["extraction_evidence_literal_alignment_passed"] += 1
            if number % 10000 == 0:
                self.db.commit()
        self.db.commit()

    def verify_media(self):
        for identity, kind, archive, member, source_id, size, compressed, crc in self.db.execute("SELECT * FROM media"):
            self.counts["media_nodes_checked"] += 1
            original = self.db.execute("SELECT kind,size,compressed,crc FROM raw_members WHERE archive=? AND member=?", (archive, member)).fetchone()
            self.check(original == (kind, size, compressed, crc), "media_zip_metadata_changed", id=identity)
            source = self.db.execute("SELECT archive,member FROM sources WHERE id=?", (source_id,)).fetchone()
            self.check(source == (archive, member), "media_source_origin_mismatch", id=identity)
            self.check(self.db.execute("SELECT 1 FROM links WHERE type='HAS_SOURCE' AND start=? AND end=?", (identity, source_id)).fetchone() is not None,
                       "media_source_link_missing", id=identity)
        for archive, member, count in self.db.execute("""SELECT r.archive,r.member,COUNT(m.id) FROM raw_members r
                 LEFT JOIN media m ON m.archive=r.archive AND m.member=r.member
                 WHERE r.kind IN ('image','video') GROUP BY r.archive,r.member HAVING COUNT(m.id)!=1"""):
            self.fail("original_media_coverage_not_exactly_one", archive=archive, member=member, count=count)
        for number, row in self.rows(self.root / "media_associations.jsonl"):
            self.counts["media_associations_checked"] += 1
            media = self.db.execute("SELECT kind,archive,member FROM media WHERE id=?", (row.get("media_id"),)).fetchone()
            if not self.check(media is not None, "association_media_node_missing", line=number):
                continue
            kind, archive, member = media
            self.check((archive, member) == (row.get("archive"), row.get("member")), "association_origin_mismatch", line=number)
            self.check(archive in ALLOWED, "association_non_test_archive", line=number)
            link = "HAS_IMAGE" if kind == "image" else "HAS_VIDEO"
            if row.get("recipe_id") is None:
                self.counts["explicit_unassociated_media_rows_checked"] += 1
                self.check(row.get("association_method") == "unmatched_or_ambiguous_directory", "unassociated_media_reason_missing", line=number)
                expected_association = self.db.execute("SELECT recipe FROM raw_media_assoc WHERE archive=? AND member=?", (archive, member)).fetchone()
                self.check(expected_association is None, "associated_media_incorrectly_registered_as_orphan", line=number)
                self.check(self.db.execute("SELECT 1 FROM links WHERE type=? AND end=?", (link, row.get("media_id"))).fetchone() is None,
                           "orphan_media_has_recipe_relationship", line=number)
                continue
            self.counts["associated_media_rows_checked"] += 1
            recipe = self.db.execute("SELECT archive,dir FROM recipes WHERE id=?", (row.get("recipe_id"),)).fetchone()
            self.check(recipe is not None, "association_recipe_missing", line=number)
            if row.get("association_method") == "same_exact_archive_directory":
                self.check(recipe == (archive, str(PurePosixPath(member).parent)), "association_exact_directory_mismatch", line=number)
            self.check(self.db.execute("SELECT 1 FROM links WHERE type=? AND start=? AND end=?", (link, row.get("recipe_id"), row.get("media_id"))).fetchone() is not None,
                       "association_relationship_missing", line=number)
        orphan_images = []
        for archive, member in self.db.execute("""SELECT m.archive,m.member FROM raw_members m LEFT JOIN raw_media_assoc a
                 ON a.archive=m.archive AND a.member=m.member WHERE m.kind='image' AND a.member IS NULL"""):
            self.counts["orphan_images_expected"] += 1
            image = self.db.execute("SELECT id,source FROM media WHERE archive=? AND member=? AND kind='image'", (archive, member)).fetchone()
            retained = image is not None
            self.check(retained, "orphan_image_dropped", archive=archive, member=member)
            if image:
                identity, source = image
                attached = self.db.execute("SELECT 1 FROM links WHERE type='HAS_SOURCE' AND start=? AND end=?", (identity, source)).fetchone() is not None
                linked_recipe = self.db.execute("SELECT 1 FROM links WHERE type='HAS_IMAGE' AND end=?", (identity,)).fetchone()
                self.check(attached and linked_recipe is None, "orphan_image_source_or_association_invalid", id=identity)
                if attached and linked_recipe is None:
                    self.counts["orphan_images_retained_with_source"] += 1
                orphan_images.append({"id": identity, "archive": archive, "member": member, "source_id": source})
        self.check(self.counts["orphan_images_expected"] == 9, "unexpected_orphan_image_count", actual=self.counts["orphan_images_expected"])
        self.check(self.counts["incomplete_recipe_records_expected"] == 2, "unexpected_incomplete_recipe_count", actual=self.counts["incomplete_recipe_records_expected"])
        self.db.commit()
        return orphan_images

    def finish(self, report_path: Path, started_at: str, elapsed: float, orphan_images: list[dict]) -> dict:
        for path, expected in self.snapshots.items():
            stat = Path(path).stat()
            self.check((stat.st_size, stat.st_mtime_ns) == expected, "artifact_changed_during_audit", file=path)
        for name in ("nodes.jsonl", "relationships.jsonl", "extraction_evidence.jsonl", "media_associations.jsonl", "records.jsonl", "manifest.json"):
            self.file_hashes[str(self.root / name)] = digest(self.root / name)
        self.db.commit()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
        report = {
            "audit_type": "full_source_and_structure_alignment", "semantic_extraction_accuracy_evaluated": False,
            "scope_note": "Literal spans, exact source identity, hashes and structural retention are checked. This is not precision/recall of extracted Ingredient/Action/Tool semantics.",
            "status": "passed" if not self.failures else "failed", "corpus_root": str(self.root), "split": "test",
            "dataset_revision": self.revision, "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3), "peak_rss_mib": round(peak_mib, 2),
            "method": {"input": "streamed JSONL", "joins": "temporary disk SQLite", "sqlite_page_cache_mib": 8,
                       "step_source_normalization": "splitlines; remove empty lines; strip outer whitespace, as recorded by source reader",
                       "source_graph_artifacts_modified": False},
            "actual_counts": dict(self.counts), "node_counts": dict(self.node_counts), "relationship_counts": dict(self.relationship_counts),
            "evidence_kind_counts": dict(self.evidence_counts), "failure_count": sum(self.failures.values()), "failure_counts_by_category": dict(self.failures),
            "outer_whitespace_normalization_examples": self.normalization_examples,
            "orphan_images": orphan_images, "verified_local_media_samples": self.media_samples, "input_sha256": self.file_hashes,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".part")
        with temporary.open("w", encoding="utf-8") as handle:
            prefix = json.dumps(report, ensure_ascii=False, indent=2).rstrip()
            handle.write(prefix[:-1] + ',\n  "failures": [')
            first = True
            for category, detail in self.db.execute("SELECT category,details FROM failures"):
                if not first:
                    handle.write(",")
                handle.write("\n    " + json.dumps({"category": category, **json.loads(detail)}, ensure_ascii=False))
                first = False
            handle.write("\n  ]\n}\n")
        temporary.replace(report_path)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "test_graph")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "test-graph-artifact-verification.json")
    args = parser.parse_args()
    started_at, start = datetime.now(timezone.utc).isoformat(), time.monotonic()
    with tempfile.TemporaryDirectory(prefix="recipegen-artifact-audit-") as temporary:
        audit = Audit(args.corpus_root, Path(temporary) / "audit.sqlite")
        for name in ("load_originals", "load_nodes", "verify_steps_and_sources", "verify_relationships", "verify_evidence"):
            getattr(audit, name)()
            print(json.dumps({"phase": name, "counts": dict(audit.counts), "failures": sum(audit.failures.values())}, ensure_ascii=False), flush=True)
        orphan_images = audit.verify_media()
        report = audit.finish(args.output, started_at, time.monotonic() - start, orphan_images)
        audit.db.close()
    print(json.dumps({"report": str(args.output.resolve()), "status": report["status"], "failure_count": report["failure_count"], "counts": report["actual_counts"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
