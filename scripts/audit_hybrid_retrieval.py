#!/usr/bin/env python3
"""Independent v2 source-label, evidence, metric, and graph-candidate audit.

Never imports the evaluated retrieval or evaluation modules. Does not read
.env, call an API, or run a model. Only the frozen local tokenizer is loaded.
Use --data-only while embeddings are being produced; run again after results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation_hybrid_v2"
OLD = ROOT / "data/evaluation_real_v1"
METHODS = ("A_recipe_e5", "B_bm25", "C_recipe_hybrid", "D_recipe_graph",
           "E_chunk_e5", "F_chunk_hybrid", "G_chunk_graph")
VARIANTS = ("without_structured_filter", "with_structured_filter")
KS = (1, 5, 10)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    return " ".join(text.casefold().split())


def name_key(text):
    return re.sub(r"[\s\-‐‑–—]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def verify_hashes(manifest):
    checks = 0
    for base, key in ((ROOT, "input_sha256"), (DATA, "artifact_sha256")):
        for relative, expected in manifest[key].items():
            check(sha(base / relative) == expected, "Frozen hash mismatch: " + relative)
            checks += 1
    original = read_json(OLD / "manifest.json")
    for relative, expected in original["artifact_sha256"].items():
        check(sha(OLD / relative) == expected, "Original v1 artifact changed: " + relative)
        checks += 1
    return checks


def graph_index():
    labels, ingredients, sources, recipe_ids = {}, {}, {}, set()
    for node in rows(ROOT / "data/test_graph/nodes.jsonl"):
        nid, label, props = node["id"], node["label"], node["properties"]
        labels[nid] = label
        if label == "Recipe":
            recipe_ids.add(nid)
        elif label == "Ingredient":
            ingredients[nid] = {name_key(props[k]) for k in ("name", "normalized_name", "zh_name")
                                if isinstance(props.get(k), str) and props[k]}
        elif label == "Source":
            sources[nid] = props
    owners, ingredient_edges, direct = {}, [], []
    for edge in rows(ROOT / "data/test_graph/relationships.jsonl"):
        start, end, kind = edge["start_id"], edge["end_id"], edge["type"]
        if kind == "HAS_STEP":
            check(start in recipe_ids and labels[end] == "Step", "Invalid HAS_STEP")
            check(end not in owners or owners[end] == start, "Step has multiple owners")
            owners[end] = start
        elif kind == "USES_INGREDIENT":
            ingredient_edges.append((start, end))
        elif kind == "HAS_INGREDIENT":
            direct.append((start, end))
    by_name = defaultdict(set)
    for step, entity in ingredient_edges:
        check(step in owners and entity in ingredients, "Invalid graph ingredient provenance")
        for name in ingredients[entity]:
            by_name[name].add(owners[step])
    for rid, entity in direct:
        check(rid in recipe_ids and entity in ingredients, "Invalid direct ingredient provenance")
        for name in ingredients[entity]:
            by_name[name].add(rid)
    return recipe_ids, by_name, sources


def graph_candidates(query, recipe_ids, by_name):
    included, excluded = query["ingredients"], query["excluded_ingredients"]
    if not included and not excluded:
        return set()
    selected = set(recipe_ids)
    for term in included:
        selected &= by_name.get(name_key(term), set())
    for term in excluded:
        selected -= by_name.get(name_key(term), set())
    return selected


def allowed_ids(query, variant, features):
    # All current explicit entities are canonical English benchmark terms.
    # This reproduces name membership without importing catalog or its aliases.
    if variant == "without_structured_filter":
        return set(features)
    included = [name_key(s) for s in query["ingredients"]]
    excluded = [name_key(s) for s in query["excluded_ingredients"]]
    return {rid for rid, names in features.items()
            if all(term in names for term in included) and not any(term in names for term in excluded)}


def source_audit(manifest):
    raw = {r["id"]: r for r in rows(ROOT / "data/test_graph/records.jsonl")}
    corpus = {r["recipe_id"]: r for r in rows(ROOT / "data/graphrag/full/corpus.jsonl")}
    queries = list(rows(DATA / "queries.jsonl"))
    qrels = list(rows(DATA / "qrels.jsonl"))
    old_queries, old_qrels = list(rows(OLD / "queries.jsonl")), list(rows(OLD / "qrels.jsonl"))
    check(len(raw) == len(corpus) == 5898 and set(raw) == set(corpus), "Corpus identity changed")
    check(len(queries) == 100 and len({q["query_id"] for q in queries}) == 100, "Query count/ID mismatch")
    check(queries[:60] == [{**q, "query_source": "previously_viewed_frozen_regression_v1"}
                          for q in old_queries], "Old 60 queries changed")
    check(qrels[:len(old_qrels)] == old_qrels, "Old qrels prefix changed")
    old_ids = {q["query_id"] for q in old_queries}
    check([r for r in qrels if r["query_id"] in old_ids] == old_qrels, "Old qrels subset changed")
    pairs = [(r["query_id"], r["recipe_id"]) for r in qrels]
    check(len(pairs) == len(set(pairs)), "Duplicate recipe qrels")
    grouped = defaultdict(set)
    for r in qrels:
        check(r["relevance"] == 1 and r["human_annotated"] is False, "Invalid automatic qrel")
        grouped[r["query_id"]].add(r["recipe_id"])
    original_embeddings = {}
    for row in rows(ROOT / "data/graphrag/full/embeddings.jsonl"):
        original_embeddings[row["recipe_id"]] = {k: v for k, v in row.items() if k != "embedding"}
    original_steps = {}
    for rid, record in raw.items():
        item = corpus[rid]
        expected_text = record["title"] + "\n" + "\n".join(s["text"] for s in record["steps"])
        check(expected_text == item["text"] and text_sha(expected_text) == item["text_sha256"],
              "Full original text mismatch")
        check(len(record["steps"]) == len(item["steps"]), "Original step count mismatch")
        for original, projected in zip(record["steps"], item["steps"]):
            check(original["order"] == projected["order"] and original["text"] == projected["text"],
                  "Original step text/order changed")
            expected_id = "step:" + text_sha(rid + "\n" + str(original["order"]))[:24]
            check(projected["id"] == expected_id, "Graph Step ID mismatch")
            original_steps[(rid, projected["id"])] = projected
    check(len(original_steps) == 44802, "Step count mismatch")

    # Independently reconstruct the original literal condition qrels.
    lexicon = read_json(OLD / "gold_lexicon.json")["entries"]
    rules = {e["name"]: re.compile(r"(?<![a-z0-9_])(?:"
             + "|".join(map(re.escape, e["literal_aliases"])) + r")(?![a-z0-9_])", re.I)
             for e in lexicon}
    mention_sets = {name: set() for name in rules}
    for rid, record in raw.items():
        fields = [record["title"]] + [s["text"] for s in record["steps"]]
        for name, pattern in rules.items():
            if any(pattern.search(text) for text in fields):
                mention_sets[name].add(rid)
    for query in queries[:60]:
        if query["category"] == "known_item":
            expected = {query["target_recipe_id"]}
            title = raw[query["target_recipe_id"]]["title"]
            check(title == query["query"] and sum(r["title"] == title for r in raw.values()) == 1,
                  "Known-item title no longer unique")
        else:
            expected = set(raw)
            for term in query["ingredients"]:
                expected &= mention_sets[term]
            for term in query["excluded_ingredients"]:
                expected -= mention_sets[term]
        check(expected == grouped[query["query_id"]], "Independent old gold mismatch")
    for qrel in qrels[:len(old_qrels)]:
        rid = qrel["recipe_id"]
        check(qrel["corpus_text_sha256"] == corpus[rid]["text_sha256"], "Old qrel text hash mismatch")
        for proof in qrel["included_match_evidence"]:
            text = (raw[rid]["title"] if proof["field"] == "title" else
                    raw[rid]["steps"][proof["step_order"] - 1]["text"])
            check(text == proof["original_text"]
                  and text[proof["span_start"]:proof["span_end"]] == proof["matched_text"],
                  "Original gold span mismatch")
            source_key = "goal_sha256" if proof["field"] == "title" else "steps_sha256"
            check(proof["source_text_sha256"] == raw[rid]["source"][source_key], "Old gold source SHA mismatch")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1")
    def denied(*args, **kwargs):
        raise RuntimeError("Network forbidden by independent audit")
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    info = read_json(ROOT / ".runtime/graphrag-model.json")
    tokenizer = AutoTokenizer.from_pretrained(info["path"], local_files_only=True, trust_remote_code=False)
    quoted = queries[60:]
    needles = [normalize(q["target_span"]["text"]) for q in quoted]
    check(len(set(needles)) == 40, "Quoted phrase duplicates across splits/categories")
    normalized_steps = {rid: [normalize(s["text"]) for s in record["steps"]]
                        for rid, record in raw.items()}
    token_ranges, source_spans = [], 0
    for query in quoted:
        rid = query["target_recipe_id"]
        step = original_steps[(rid, query["target_step_id"])]
        span = query["target_span"]
        check(step["text"][span["start"]:span["end"]] == span["text"]
              and step["order"] == query["target_step_order"], "Target original span mismatch")
        passage = "passage: " + corpus[rid]["text"]
        start, end = query["target_passage_char_span"]
        check(passage[start:end] == span["text"], "Passage absolute span mismatch")
        offsets = tokenizer(passage, truncation=False, return_offsets_mapping=True)["offset_mapping"]
        positions = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
        actual_range = [min(positions), max(positions) + 1]
        check(actual_range == query["target_passage_token_span"], "Independent token span mismatch")
        truncated = original_embeddings[rid]["truncated"]
        check(truncated == query["target_recipe_original_embedding_truncated"], "Truncation source mismatch")
        if query["category"] == "late_step_passage":
            check(truncated and actual_range[0] > 530, "Late quote visible before frozen boundary")
            token_ranges.append(actual_range)
        else:
            check(query["category"] == "short_step_passage" and not truncated and len(offsets) <= 512,
                  "Short control source is actually truncated")
        needle = normalize(span["text"])
        expected = {other_id for other_id, texts in normalized_steps.items()
                    if any(needle in text for text in texts)}
        check(expected == grouped[query["query_id"]], "Independent all-corpus phrase gold mismatch")
        check(len(expected) == query["gold_relevant_count"] and 1 <= len(expected) <= 50,
              "Frozen phrase count mismatch")
    by_qid = {q["query_id"]: q for q in queries}
    for qrel in qrels[len(old_qrels):]:
        rid, query = qrel["recipe_id"], by_qid[qrel["query_id"]]
        step = original_steps[(rid, qrel["matched_step_id"])]
        proof = qrel["matched_span"]
        check(step["text"] == qrel["original_step_text"] and step["order"] == qrel["matched_step_order"],
              "New qrel original step mismatch")
        check(step["text"][proof["start"]:proof["end"]] == proof["text"]
              and normalize(proof["text"]) == normalize(query["target_span"]["text"]), "New qrel span mismatch")
        check(qrel["source"] == raw[rid]["source"] and qrel["source"]["steps_sha256"],
              "New qrel source/SHA mismatch")
        check(qrel["corpus_text_sha256"] == corpus[rid]["text_sha256"], "New qrel corpus hash mismatch")
        source_spans += 1

    # Check source ownership, exact text reconstruction and real token budgets.
    recipe_ids, graph_names, graph_sources = graph_index()
    check(recipe_ids == set(raw), "Graph Recipe identities differ from corpus")
    fragments, title_only, chunk_ids = defaultdict(list), [], set()
    chunks = list(rows(DATA / "chunks.jsonl"))
    max_tokens = 0
    for chunk in chunks:
        rid = chunk["recipe_id"]
        check(chunk["id"] == chunk["chunk_id"] and chunk["id"] not in chunk_ids, "Chunk ID duplicate")
        chunk_ids.add(chunk["id"])
        check(chunk["embedding_text"] == "passage: " + chunk["text"], "Double/missing passage prefix")
        actual_tokens = len(tokenizer.encode(chunk["embedding_text"], add_special_tokens=True, truncation=False))
        check(actual_tokens == chunk["token_count"] and actual_tokens <= 480, "Actual chunk token budget mismatch")
        max_tokens = max(max_tokens, actual_tokens)
        if not chunk["fragments"]:
            check(chunk["unit"] == "title_only" and not raw[rid]["steps"], "Unexpected empty chunk")
            check(chunk["text"] == chunk["title_context"] and chunk["source_ids"], "Title-only evidence missing")
            for sid in chunk["source_ids"]:
                check(graph_sources[sid]["sha256"] == raw[rid]["source"]["goal_sha256"], "Title Source SHA mismatch")
            title_only.append(rid)
        for part in chunk["fragments"]:
            key = (rid, part["step_id"])
            step = original_steps[key]
            a, b = part["span_start"], part["span_end"]
            check(0 <= a <= b <= len(step["text"]) and part["text"] == step["text"][a:b],
                  "Chunk source character span mismatch")
            check(part["source_id"] == step["source_id"]
                  and graph_sources[part["source_id"]]["sha256"] == raw[rid]["source"]["steps_sha256"],
                  "Chunk Step Source ownership/hash mismatch")
            check(part["original_text_sha256"] == text_sha(step["text"]), "Chunk original step hash mismatch")
            fragments[key].append((a, b))
    for key, step in original_steps.items():
        cursor = 0
        for start, end in sorted(fragments[key]):
            check(start == cursor, "Chunk fragments overlap or leave a gap")
            cursor = end
        check(cursor == len(step["text"]), "Incomplete original step chunk coverage")
    check(len(title_only) == 2 and len(chunks) == 16897, "Chunk/title-only count mismatch")
    counts = Counter((q["category"], q["split"]) for q in queries)
    check(all(n in (4, 16) for n in counts.values()) and len(counts) == 10, "Category split mismatch")
    features = {r["recipe_id"]: set(r["production_mention_names"])
                for r in rows(OLD / "production_flat_metadata.jsonl")}
    result = {"passed": True, "old_60_queries_preserved": True, "old_qrels_preserved": True,
              "queries": len(queries), "qrels": len(qrels), "original_steps": len(original_steps),
              "independently_rebuilt_query_gold_sets": len(queries), "new_source_span_checks": source_spans,
              "quoted_phrases_unique_across_splits": True, "actual_token_spans_verified": 40,
              "late_token_start_min": min(p[0] for p in token_ranges),
              "late_token_start_max": max(p[0] for p in token_ranges), "chunks": len(chunks),
              "all_chunk_token_counts_verified": True, "maximum_actual_chunk_tokens": max_tokens,
              "complete_step_span_coverage": True, "title_only_units": len(title_only),
              "new_query_gold_counts": dict(sorted(Counter(q["gold_relevant_count"] for q in quoted).items())),
              "category_counts": [{"category": c, "split": s, "queries": n} for (c, s), n in sorted(counts.items())]}
    return result, queries, grouped, features, recipe_ids, graph_names


def compute_metrics(query, ranking, relevant):
    ids = [r["recipe_id"] for r in ranking]
    check(len(ids) == len(set(ids)), "Recipe rankings not deduplicated")
    first = next((i for i, rid in enumerate(ids[:10], 1) if rid in relevant), None)
    known = query["category"] == "known_item"
    values = {"target_mrr_at_10" if known else "mrr_at_10": 0.0 if first is None else 1.0 / first}
    for k in KS:
        hit = len(set(ids[:k]) & relevant)
        if known:
            values["target_hit_at_" + str(k)] = float(hit > 0)
        else:
            values["precision_at_" + str(k)] = hit / k
            values["recall_at_" + str(k)] = hit / len(relevant)
            values["precision_returned_at_" + str(k)] = hit / len(ids[:k]) if ids[:k] else 0.0
            values["hit_at_" + str(k)] = float(hit > 0)
    return values


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), message)


def audit_vector_artifacts(queries):
    manifest = read_json(DATA / "vector-manifest.json")
    check(manifest["benchmark_manifest_sha256"] == sha(DATA / "manifest.json"),
          "Vector build benchmark changed")
    for relative, expected in manifest["artifact_sha256"].items():
        check(sha(DATA / relative) == expected, "Encoded vector hash mismatch")
    query_vectors = {r["query_id"]: r for r in rows(DATA / "query_embeddings.jsonl")}
    original_vectors = {r["query_id"]: r for r in rows(OLD / "query_embeddings.jsonl")}
    check(len(query_vectors) == len(queries), "Encoded query count differs")
    for query in queries:
        row = query_vectors[query["query_id"]]
        check(row["query"] == query["query"], "Encoded query text changed")
        vector = row["embedding"]
        check(len(vector) == 384 and all(math.isfinite(v) for v in vector),
              "Invalid query vector values")
        check(abs(sum(v * v for v in vector) - 1) <= .01, "Query vector not normalized")
        if query["query_id"] in original_vectors:
            check(vector == original_vectors[query["query_id"]]["embedding"],
                  "Frozen original query embedding changed")
    import numpy as np
    matrix = np.load(DATA / "chunk_vectors.npy", mmap_mode="r")
    check(matrix.shape == (16897, 384) and matrix.dtype == np.float32,
          "Encoded chunk shape/dtype differs")
    check(np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01),
          "Chunk vector values/norms invalid")
    return {"passed": True, "encoded_artifact_hashes_verified": len(manifest["artifact_sha256"]),
            "original_query_embeddings_preserved": len(original_vectors),
            "all_query_vectors_valid": len(query_vectors), "chunk_vectors_valid": len(matrix),
            "dimension": 384, "dtype": "float32",
            "scope": "vector_artifact_identity_and_norms_no_model_inference"}


def audit_results(queries, gold, features, recipe_ids, by_name):
    rankings = list(rows(DATA / "rankings.jsonl"))
    report = read_json(ROOT / "reports/hybrid-retrieval-v2.json")
    check(report["rankings_sha256"] == sha(DATA / "rankings.jsonl"), "Ranking report hash mismatch")
    check(report["manifest_sha256"] == sha(DATA / "manifest.json"), "Report benchmark hash mismatch")
    check(len(rankings) == 1400, "Incomplete 100 x 2 x 7 rankings")
    queries_by_id = {q["query_id"]: q for q in queries}
    indexed, grouped, metric_checks = {}, defaultdict(list), 0
    for row in rankings:
        key = (row["query_id"], row["variant"], row["method"])
        check(key not in indexed and row["variant"] in VARIANTS and row["method"] in METHODS, "Ranking key invalid")
        indexed[key] = row
        query = queries_by_id[row["query_id"]]
        check(row["query"] == query["query"] and row["category"] == query["category"]
              and row["split"] == query["split"], "Ranking query identity changed")
        relevant, allowed = gold[row["query_id"]], allowed_ids(query, row["variant"], features)
        check(row["qrels_count"] == len(relevant), "Wrong Recall denominator")
        check(row["eligible_records"] == len(allowed), "Production eligibility mismatch")
        check(row["gold_filter_reachable_count"] == len(relevant & allowed), "Gold filter diagnostic mismatch")
        close(row["gold_filter_reachable_fraction"], len(relevant & allowed) / len(relevant), "Reachable fraction mismatch")
        seed = row["seed_ids"]
        check(len(seed["bm25"]) <= 50 and len(seed["dense"]) <= 50, "Ranked route budget exceeded")
        expected_graph = (graph_candidates(query, recipe_ids, by_name)
                          if row["method"] in ("D_recipe_graph", "G_chunk_graph") else set())
        check(set(seed["graph"]) == expected_graph, "Graph candidates differ from original exported relations")
        regular_seed = set(seed["bm25"]) | set(seed["dense"])
        union = regular_seed | expected_graph
        pool = union & allowed
        counts = {"bm25_seed": len(seed["bm25"]), "dense_seed": len(seed["dense"]),
                  "graph": len(expected_graph), "graph_only_added": len((expected_graph - regular_seed) & pool),
                  "filtered_after_seed": len(union - pool), "pool": len(pool)}
        check(row["candidate_counts"] == counts, "Candidate/graph-only pool counts mismatch")
        check(len(row["ranking"]) == min(50, len(pool)), "Returned candidate count differs from pool")
        check(row["returned_candidates"] == len(row["ranking"]), "Returned count mismatch")
        check(row["empty_result"] == (not row["ranking"]), "Empty-result accounting mismatch")
        check(row["fewer_than_10"] == (len(row["ranking"]) < 10), "Short-return accounting mismatch")
        previous_order = None
        for item in row["ranking"]:
            rid = item["recipe_id"]
            check(rid in pool and item["from_graph"] == (rid in expected_graph), "Ranking outside allowed pool")
            check(item["graph_only_added"] == (rid in expected_graph - regular_seed), "Graph-only flag mismatch")
            lexical_rank, dense_rank = item["bm25_rank"], item["dense_rank"]
            expected_rrf = ((1 / (60 + lexical_rank)) if lexical_rank is not None else 0)
            expected_rrf += ((1 / (60 + dense_rank)) if dense_rank is not None else 0)
            close(item["rrf_score"], expected_rrf, "RRF formula mismatch")
            if item["bm25_score"] is not None and item["bm25_score"] == 0:
                check(lexical_rank is None, "Zero BM25 received lexical rank")
            for score in ("bm25_score", "cosine_score"):
                check(item[score] is None or math.isfinite(item[score]), "Nonfinite ranking score")
            current_order = (-item["rrf_score"], rid)
            check(previous_order is None or current_order >= previous_order, "RRF ranking order mismatch")
            previous_order = current_order
        expected = compute_metrics(query, row["ranking"], relevant)
        check(set(expected) == set(row["metrics"]), "Metric schema mismatch")
        for name, value in expected.items():
            close(row["metrics"][name], value, "Per-query metric mismatch: " + name)
            metric_checks += 1
        check(row["missed_gold_at_10"] == sorted(relevant - {r["recipe_id"] for r in row["ranking"][:10]}),
              "Missed-gold listing mismatch")
        group_key = (row["split"], row["category"], row["variant"], row["method"])
        grouped[group_key].append(row)
        if row["category"] in ("multi_ingredient", "ingredient_exclusion"):
            grouped[(row["split"], "conditions_combined", row["variant"], row["method"])].append(row)
        elif row["category"] in ("late_step_passage", "short_step_passage"):
            grouped[(row["split"], "step_passages_combined", row["variant"], row["method"])].append(row)
    check(len(indexed) == len(queries) * len(VARIANTS) * len(METHODS), "Missing method/filter conditions")
    summary_checks, verified_summary = 0, []
    check(len(report["summary"]) == len(grouped), "Summary group count mismatch")
    for summary in report["summary"]:
        key = (summary["split"], summary["category"], summary["variant"], summary["method"])
        selected = grouped[key]
        check(summary["queries"] == len(selected), "Summary query count mismatch")
        for name in selected[0]["metrics"]:
            expected = sum(compute_metrics(queries_by_id[r["query_id"]], r["ranking"], gold[r["query_id"]])[name]
                           for r in selected) / len(selected)
            close(summary["macro_metrics"][name], expected, "Macro metric mismatch: " + name)
            summary_checks += 1
        verified_summary.append({
            "split": key[0], "category": key[1], "variant": key[2], "method": key[3],
            "queries": len(selected),
            "independently_recomputed_macro_metrics": {
                name: sum(compute_metrics(queries_by_id[r["query_id"]], r["ranking"],
                                         gold[r["query_id"]])[name] for r in selected) / len(selected)
                for name in selected[0]["metrics"]}})
        for field, expected in {
            "mean_relevant_count": sum(r["qrels_count"] for r in selected) / len(selected),
            "mean_returned_at_10": sum(min(len(r["ranking"]), 10) for r in selected) / len(selected),
            "empty_results": sum(not r["ranking"] for r in selected),
            "queries_returning_fewer_than_10": sum(len(r["ranking"]) < 10 for r in selected),
            "mean_gold_filter_reachable_fraction": sum(r["gold_filter_reachable_fraction"] for r in selected) / len(selected),
            "mean_graph_only_added": sum(r["candidate_counts"]["graph_only_added"] for r in selected) / len(selected)
        }.items():
            close(summary[field], expected, "Summary count/diagnostic mismatch")
    contrasts = []
    for base, expanded in (("C_recipe_hybrid", "D_recipe_graph"), ("F_chunk_hybrid", "G_chunk_graph")):
        for variant in VARIANTS:
            changed = added = top10_new = metric_changed = 0
            for query in queries:
                a, b = indexed[(query["query_id"], variant, base)], indexed[(query["query_id"], variant, expanded)]
                check(a["seed_ids"]["bm25"] == b["seed_ids"]["bm25"]
                      and a["seed_ids"]["dense"] == b["seed_ids"]["dense"], "Graph changed original ranked seeds")
                added += b["candidate_counts"]["graph_only_added"]
                changed += [x["recipe_id"] for x in a["ranking"]] != [x["recipe_id"] for x in b["ranking"]]
                top10_new += len({r["recipe_id"] for r in b["ranking"][:10]}
                                 - {r["recipe_id"] for r in a["ranking"][:10]})
                metric_changed += a["metrics"] != b["metrics"]
            contrasts.append({"base": base, "graph_method": expanded, "variant": variant,
                              "graph_only_candidates_added_total": added, "changed_recipe_rankings": changed,
                              "new_top10_members_total": top10_new, "queries_with_metric_changes": metric_changed})
    return {"passed": True, "ranking_rows_independently_recomputed": len(rankings),
            "individual_metric_values_checked": metric_checks, "summary_groups_checked": len(grouped),
            "summary_metric_values_checked": summary_checks, "exported_graph_candidates_independently_verified": True,
            "graph_contrasts": contrasts, "verified_summary": verified_summary,
            "ranking_algorithms_reexecuted": False,
            "scope": "labels_metrics_graph_sets_pool_counts_and_saved_RRF_not_new_model_inference",
            "rankings_sha256": sha(DATA / "rankings.jsonl"), "reported_results_sha256": sha(ROOT / "reports/hybrid-retrieval-v2.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = read_json(DATA / "manifest.json")
    verified_hashes = verify_hashes(manifest)
    data_result, queries, gold, features, recipe_ids, by_name = source_audit(manifest)
    report = {"schema": "recipegen-hybrid-v2-independent-audit-v1", "status": "data_verified_results_pending",
              "checked_at": datetime.now(timezone.utc).isoformat(), "frozen_file_hashes_checked": verified_hashes,
              "benchmark_manifest_sha256": sha(DATA / "manifest.json"),
              "audit_script_sha256": sha(__file__), "data": data_result, "results": None,
              "dotenv_read": False, "network_calls": 0, "llm_calls": 0, "model_inference_calls": 0}
    if not args.data_only:
        report["vectors"] = audit_vector_artifacts(queries)
        report["results"] = audit_results(queries, gold, features, recipe_ids, by_name)
        report["status"] = "passed"
    output = args.output or ROOT / "reports" / ("hybrid-retrieval-v2-data-audit.json" if args.data_only else
                                               "hybrid-retrieval-v2-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit_report": str(output.resolve()),
                      "queries": len(queries), "data_passed": data_result["passed"],
                      "results_passed": bool(report["results"] and report["results"]["passed"])}))


if __name__ == "__main__":
    main()
