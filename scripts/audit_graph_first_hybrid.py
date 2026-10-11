#!/usr/bin/env python3
"""Independent local reconstruction of graph-first three-route retrieval.

No evaluated runner/helper imports, models, dotenv, API, or network. Source
qrels are independently rebuilt from original text and never enter ranking.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation_graph_first_hybrid_v1"
V2 = ROOT / "data/evaluation_hybrid_v2"
REPORT = ROOT / "reports/graph-first-hybrid-v1.json"
AUDIT = ROOT / "reports/graph-first-hybrid-v1-audit.json"
CONDITIONS = {"multi_ingredient", "ingredient_exclusion"}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def name(text):
    return re.sub(r"[\s\-‐‑–—]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*|[\u3400-\u9fff]|[^\W\d_]+",
                      unicodedata.normalize("NFKC", text).lower(), flags=re.UNICODE)


def graph_data():
    labels, ingredients, recipe_ids = {}, {}, set()
    for node in rows(ROOT / "data/test_graph/nodes.jsonl"):
        nid, kind, props = node["id"], node["label"], node["properties"]
        check(nid not in labels, "Graph node IDs duplicated")
        labels[nid] = kind
        if kind == "Recipe":
            recipe_ids.add(nid)
        elif kind == "Ingredient":
            ingredients[nid] = {name(props[k]) for k in ("name", "normalized_name", "zh_name")
                                if isinstance(props.get(k), str) and props[k]}
    owners, indirect, direct = {}, [], []
    for edge in rows(ROOT / "data/test_graph/relationships.jsonl"):
        start, end, kind = edge["start_id"], edge["end_id"], edge["type"]
        if kind == "HAS_STEP":
            check(start in recipe_ids and labels[end] == "Step", "Invalid HAS_STEP relation")
            check(end not in owners or owners[end] == start, "Step has multiple owners")
            owners[end] = start
        elif kind == "USES_INGREDIENT":
            indirect.append((start, end))
        elif kind == "HAS_INGREDIENT":
            direct.append((start, end))
    by_name = defaultdict(set)
    for step, entity in indirect:
        check(step in owners and entity in ingredients, "Invalid graph ingredient provenance")
        for value in ingredients[entity]:
            by_name[value].add(owners[step])
    for rid, entity in direct:
        check(rid in recipe_ids and entity in ingredients, "Invalid direct ingredient provenance")
        for value in ingredients[entity]:
            by_name[value].add(rid)
    return recipe_ids, by_name


def graph_pool(query, recipe_ids, by_name):
    selected = set(recipe_ids)
    for term in query["ingredients"]:
        selected &= by_name.get(name(term), set())
    for term in query["excluded_ingredients"]:
        selected -= by_name.get(name(term), set())
    return selected


def independent_qrels(queries, raw):
    """Ground truth is separate from graph, BM25, vectors, and ranking."""
    lexicon = read(ROOT / "data/evaluation_real_v1/gold_lexicon.json")["entries"]
    rules = {e["name"]: re.compile(r"(?<![a-z0-9_])(?:"
             + "|".join(map(re.escape, e["literal_aliases"])) + r")(?![a-z0-9_])", re.I)
             for e in lexicon}
    mentions = {key: set() for key in rules}
    for rid, record in raw.items():
        fields = [record["title"]] + [step["text"] for step in record["steps"]]
        for key, pattern in rules.items():
            if any(pattern.search(text) for text in fields):
                mentions[key].add(rid)
    saved = defaultdict(set)
    for row in rows(V2 / "qrels.jsonl"):
        check(row["human_annotated"] is False and row["relevance"] == 1,
              "Invalid source automatic label")
        saved[row["query_id"]].add(row["recipe_id"])
    for query in queries:
        gold = set(raw)
        for term in query["ingredients"]:
            gold &= mentions[term]
        for term in query["excluded_ingredients"]:
            gold -= mentions[term]
        check(gold == saved[query["query_id"]], "Full-source literal qrels disagree")
        check(len(gold) == query["gold_relevant_count"] and gold, "Qrel count mismatch")
    return saved


class IndependentBM25:
    def __init__(self, documents):
        self.counters = {d["recipe_id"]: Counter(tokens(d["text"])) for d in documents}
        check(len(self.counters) == len(documents), "Duplicate BM25 documents")
        self.lengths = {rid: sum(counts.values()) for rid, counts in self.counters.items()}
        avg = sum(self.lengths.values()) / len(self.lengths)
        self.norm = {rid: 1.5 * (1 - .75 + .75 * length / avg) for rid, length in self.lengths.items()}
        document_frequency = Counter(term for counts in self.counters.values() for term in counts)
        self.idf = {term: math.log1p((len(self.counters) - df + .5) / (df + .5))
                    for term, df in document_frequency.items()}

    def scores(self, query):
        # Every explicit condition term in this regression set is canonical.
        # Therefore visible positive terms reproduce the production keyword
        # expansion without importing catalog, aliases, or gold lexicon.
        term_set = set(tokens(" ".join(dict.fromkeys(query["ingredients"]))))
        return {rid: sum(self.idf[t] * counts[t] * 2.5 / (counts[t] + self.norm[rid])
                         for t in term_set if counts[t])
                for rid, counts in self.counters.items()}


def independent_retrieve(query, recipe_ids, by_name, bm25, pvec, qvec, features=None):
    """Only visible input, graph, lexical documents, and vectors enter here."""
    import numpy as np
    eligible = graph_pool(query, recipe_ids, by_name)
    selected = sorted(eligible)
    vector = np.asarray(qvec[query["query_id"]]["embedding"], dtype=np.float32)
    # No full-corpus E5 scoring: gather only graph-qualified vectors.
    matrix = np.asarray([pvec[rid]["embedding"] for rid in selected], dtype=np.float32)
    dense = dict(zip(selected, map(float, matrix @ vector))) if selected else {}
    lexical = bm25.scores(query)
    bm25_seed = sorted((rid for rid, value in lexical.items() if value > 0),
                       key=lambda rid: (-lexical[rid], rid))[:50]
    dense_seed = sorted(selected, key=lambda rid: (-dense[rid], rid))[:50]
    graph_seed = set(eligible)
    union = set(bm25_seed) | set(dense_seed) | graph_seed
    production = eligible if features is None else {
        rid for rid, names in features.items()
        if all(name(t) in names for t in query["ingredients"])
        and not any(name(t) in names for t in query["excluded_ingredients"])}
    pool = union & eligible & production
    lexical_order = sorted((rid for rid in pool if lexical[rid] > 0),
                           key=lambda rid: (-lexical[rid], rid))
    dense_order = sorted(pool, key=lambda rid: (-dense[rid], rid))
    lexical_rank = {rid: i for i, rid in enumerate(lexical_order, 1)}
    dense_rank = {rid: i for i, rid in enumerate(dense_order, 1)}
    output = []
    for rid in pool:
        lr, dr = lexical_rank.get(rid), dense_rank.get(rid)
        output.append({"recipe_id": rid, "bm25_score": lexical[rid], "cosine_score": dense[rid],
                       "bm25_rank": lr, "dense_rank": dr,
                       "rrf_score": (1 / (60 + lr) if lr else 0) + (1 / (60 + dr) if dr else 0)})
    output.sort(key=lambda row: (-row["rrf_score"], row["recipe_id"]))
    return {"ranking": output[:50], "bm25_seed": bm25_seed, "dense_seed": dense_seed,
            "graph_ids": sorted(graph_seed), "candidate_pool": sorted(pool),
            "dense_scored_ids": selected, "production_ids": sorted(production),
            "union_ids": sorted(union), "lexical_scores": lexical,
            "dense_scores": dense,
            "graph_only_added_ids": sorted((graph_seed - set(bm25_seed) - set(dense_seed)) & pool)}


def independent_metrics(ranking, gold):
    ids = [r["recipe_id"] for r in ranking]
    first = next((i for i, rid in enumerate(ids[:10], 1) if rid in gold), None)
    result = {"mrr_at_10": 1 / first if first else 0.0}
    for k in (1, 5, 10):
        returned = ids[:k]
        hits = len(set(returned) & gold)
        result.update({f"precision_at_{k}": hits / k, f"recall_at_{k}": hits / len(gold),
                       f"precision_returned_at_{k}": hits / len(returned) if returned else 0.0,
                       f"hit_at_{k}": float(bool(hits))})
    return result


def main():
    def denied(*args, **kwargs):
        raise RuntimeError("Network forbidden by independent graph-first hybrid audit")
    socket.socket.connect = denied
    socket.create_connection = denied
    manifest, report = read(DATA / "manifest.json"), read(REPORT)
    for relative, expected in manifest["input_sha256"].items():
        check(Path(relative).name != ".env", "Audit must never read credentials")
        check(sha(ROOT / relative) == expected, "Frozen input mismatch: " + relative)
    check(sha(DATA / "PROTOCOL.md") == manifest["protocol_sha256"], "Frozen protocol changed")
    queries = [q for q in rows(V2 / "queries.jsonl") if q["category"] in CONDITIONS]
    check(len(queries) == 40 and sum(q["split"] == "test" for q in queries) == 32,
          "Condition split must be 8 dev and 32 test")
    raw = {r["id"]: r for r in rows(ROOT / "data/test_graph/records.jsonl")}
    docs = rows(V2 / "recipe_documents.jsonl")
    check(len(raw) == len(docs) == 5898, "Recipe domain size changed")
    for doc in docs:
        record = raw[doc["recipe_id"]]
        expected = record["title"] + "\n" + "\n".join(s["text"] for s in record["steps"])
        check(doc["text"] == expected, "BM25 does not preserve full original text")
    recipe_ids, by_name = graph_data()
    pvec = {r["recipe_id"]: r for r in rows(ROOT / "data/graphrag/full/embeddings.jsonl")}
    qvec = {r["query_id"]: r for r in rows(V2 / "query_embeddings.jsonl")}
    check(set(raw) == recipe_ids == set(pvec), "Graph/source/vector domains differ")
    import numpy as np
    matrix = np.asarray([r["embedding"] for r in pvec.values()], dtype=np.float32)
    check(matrix.shape == (5898, 384) and np.isfinite(matrix).all()
          and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01), "Invalid recipe vectors")
    for row in pvec.values():
        check(row["model_revision"] == manifest["model_revision"], "Recipe model revision drift")
    for query in queries:
        cached = qvec[query["query_id"]]
        vector = np.asarray(cached["embedding"], dtype=np.float32)
        check(cached["query"] == query["query"] and cached["model"] == manifest["model"]
              and cached["model_revision"] == manifest["model_revision"], "Query model/text drift")
        check(vector.shape == (384,) and np.isfinite(vector).all()
              and np.isclose(np.linalg.norm(vector), 1, atol=.01), "Invalid query vector")
    features = {r["recipe_id"]: {name(t) for t in r["production_mention_names"]}
                for r in rows(ROOT / "data/evaluation_real_v1/production_flat_metadata.jsonl")}
    check(set(features) == recipe_ids, "Production condition domain differs")
    bm25 = IndependentBM25(docs)
    expected = {q["query_id"]: independent_retrieve(q, recipe_ids, by_name, bm25, pvec, qvec, features)
                for q in queries}
    gold = independent_qrels(queries, raw)
    # Output schema verification is completed against the new frozen runner
    # report below; it deliberately has no evaluated retrieval imports.
    validate_outputs(manifest, report, queries, expected, gold)


def validate_outputs(manifest, report, queries, expected, gold):
    check(manifest["status"] == "frozen_before_new_rankings", "Inputs were not frozen before rankings")
    check(set(manifest["query_ids"]) == {q["query_id"] for q in queries}, "Query IDs differ")
    check(manifest["human_annotated"] is False and manifest["new_unseen_queries"] is False,
          "Benchmark claim boundary changed")
    check(report["manifest_sha256"] == sha(DATA / "manifest.json"), "Report manifest hash differs")
    for relative, expected_hash in report["output_sha256"].items():
        check(sha(DATA / relative) == expected_hash, "Report output hash differs: " + relative)
    for relative, expected_hash in report["code_sha256"].items():
        check(sha(ROOT / relative) == expected_hash, "Report code hash differs: " + relative)
    legacy = read(V2 / "manifest.json")
    for relative, expected_hash in legacy["input_sha256"].items():
        if not relative.endswith(".py"):
            check(sha(ROOT / relative) == expected_hash, "Historical non-code input changed: " + relative)
    for relative, expected_hash in legacy["artifact_sha256"].items():
        check(sha(V2 / relative) == expected_hash, "Historical benchmark artifact changed: " + relative)
    vector_manifest = read(V2 / "vector-manifest.json")
    check(vector_manifest["benchmark_manifest_sha256"] == sha(V2 / "manifest.json"),
          "Historical vector benchmark identity changed")
    for relative, expected_hash in vector_manifest["artifact_sha256"].items():
        check(sha(V2 / relative) == expected_hash, "Historical vector artifact changed: " + relative)
    old_D = {r["query_id"]: r for r in rows(V2 / "rankings.jsonl")
             if r["method"] == "D_recipe_graph" and r["variant"] == "with_structured_filter"}
    candidate_rows = rows(DATA / "candidates.jsonl")
    candidates = {r["query_id"]: r for r in candidate_rows}
    check(len(candidate_rows) == len(candidates) == 40, "Candidate record count differs")
    results = rows(DATA / "rankings.jsonl")
    index = {(r["query_id"], r["method"]): r for r in results}
    methods = ("D_recipe_graph_saved", "C_graph_then_e5", "H_graph_first_three_route")
    check(len(results) == len(index) == 120, "Ranking row count differs")
    groups = defaultdict(list)
    changed_order = []
    max_score_error = 0.0
    for query in queries:
        qid = query["query_id"]
        wanted, candidate = expected[qid], candidates[qid]
        check(candidate["lexical_query"] == " ".join(dict.fromkeys(query["ingredients"])),
              "Lexical expansion differs from canonical visible positive terms")
        check(candidate["graph_candidate_ids"] == wanted["graph_ids"], "Raw graph predicate disagrees")
        check(candidate["production_eligible_ids"] == wanted["production_ids"], "Production predicate disagrees")
        check(candidate["semantic_scored_ids"] == wanted["dense_scored_ids"], "E5 domain not graph-qualified")
        check(candidate["candidate_pool"] == wanted["candidate_pool"], "Final pool differs")
        check(candidate["union_before_condition_recheck"] == wanted["union_ids"], "Candidate union differs")
        check(candidate["seed_ids"] == {"bm25": wanted["bm25_seed"],
                                       "dense": wanted["dense_seed"], "graph": wanted["graph_ids"]},
              "Candidate route seeds differ")
        check(candidate["graph_has_rank"] is False, "Graph must not have a third rank")
        counts = candidate["counts"]
        check(counts == {"bm25_seed": len(wanted["bm25_seed"]), "dense_seed": len(wanted["dense_seed"]),
                         "graph": len(wanted["graph_ids"]),
                         "graph_only_added": len(wanted["graph_only_added_ids"]),
                         "filtered_after_seed": len(set(wanted["union_ids"]) - set(wanted["candidate_pool"])),
                         "pool": len(wanted["candidate_pool"])}, "Candidate accounting differs")
        for method in methods:
            row = index[qid, method]
            ranking = row["ranking"]
            check(row["split"] == query["split"] and row["category"] == query["category"], "Split/category drift")
            check(len({r["recipe_id"] for r in ranking}) == len(ranking), "Duplicate ranked recipe")
            if method == "D_recipe_graph_saved":
                check(ranking == old_D[qid]["ranking"], "Historical saved D ranking changed")
                expected_count = old_D[qid]["candidate_counts"]["pool"]
            elif method == "C_graph_then_e5":
                check([r["recipe_id"] for r in ranking] == wanted["dense_seed"], "Pure E5 ranking differs")
                for actual in ranking:
                    check(math.isclose(actual["cosine_score"], wanted["dense_scores"][actual["recipe_id"]],
                                       rel_tol=1e-7, abs_tol=1e-7), "Pure E5 score differs")
                expected_count = len(wanted["graph_ids"])
            else:
                check([r["recipe_id"] for r in ranking] == [r["recipe_id"] for r in wanted["ranking"]],
                      "Mixed complete-pool RRF ranking differs")
                expected_count = len(wanted["candidate_pool"])
                for actual, reference in zip(ranking, wanted["ranking"]):
                    check(actual["bm25_rank"] == reference["bm25_rank"]
                          and actual["dense_rank"] == reference["dense_rank"], "Complete-pool ranks differ")
                    check("graph_rank" not in actual, "Graph ranking was fabricated")
                    check(actual["cosine_score"] is not None, "Final candidate missing actual E5 score")
                    for key in ("bm25_score", "cosine_score", "rrf_score"):
                        error = abs(actual[key] - reference[key])
                        max_score_error = max(max_score_error, error)
                        check(math.isclose(actual[key], reference[key], rel_tol=1e-7, abs_tol=1e-7),
                              "Independent score differs: " + key)
                    check(actual["from_graph"] is True, "Final candidate fails graph conditions")
                    check(actual["graph_only_added"] == (actual["recipe_id"] in wanted["graph_only_added_ids"]),
                          "Graph-only provenance differs")
                    if actual["bm25_score"] == 0:
                        check(actual["bm25_rank"] is None, "Zero BM25 acquired a lexical rank")
            check(row["candidate_count"] == expected_count, "Candidate count differs")
            check(row["qrels_count"] == len(gold[qid]), "Reported qrels count differs")
            metrics = independent_metrics(ranking, gold[qid])
            check(set(row["metrics"]) == set(metrics), "Metric key set differs")
            for key, value in metrics.items():
                check(math.isclose(row["metrics"][key], value, abs_tol=1e-12), "Per-query metric differs: " + key)
            check(row["gold_hits_at_10"] == len({r["recipe_id"] for r in ranking[:10]} & gold[qid]),
                  "Top10 relevance hit count differs")
            groups[query["split"], query["category"], method].append((row, metrics))
            groups[query["split"], "conditions_combined", method].append((row, metrics))
        if [r["recipe_id"] for r in wanted["ranking"][:10]] != wanted["dense_seed"][:10]:
            changed_order.append(qid)
    summary_index = {(r["split"], r["category"], r["method"]): r for r in report["summary"]}
    check(set(summary_index) == set(groups), "Summary groups differ")
    verified = []
    for key, batch in sorted(groups.items()):
        actual = summary_index[key]
        average = {metric: sum(metrics[metric] for _, metrics in batch) / len(batch)
                   for metric in batch[0][1]}
        check(actual["queries"] == len(batch), "Summary query count differs")
        for metric, value in average.items():
            check(math.isclose(actual["macro_metrics"][metric], value, abs_tol=1e-12),
                  "Macro metric differs: " + metric)
        check(actual["empty_results"] == sum(not row["ranking"] for row, _ in batch), "Empty summary differs")
        check(math.isclose(actual["mean_returned_at_10"],
                           sum(min(len(row["ranking"]), 10) for row, _ in batch) / len(batch), abs_tol=1e-12),
              "Returned-count summary differs")
        if key[1] == "conditions_combined" and key[2] == "H_graph_first_three_route":
            verified.append({"split": key[0], "queries": len(batch),
                             "precision_at_10": average["precision_at_10"],
                             "recall_at_10": average["recall_at_10"]})
    check(report["llm_calls"] == report["network_calls"] == report["model_encoding_calls"] == 0,
          "Metric scope includes unrequested model/API execution")
    audit = {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
             "benchmark": "graph-first-hybrid-v1",
             "independent_implementation": "source graph/lexical BM25/graph-only E5 float32/RRF/qrels/metrics; no evaluated module imports",
             "verified_three_route_macro_metrics": verified,
             "hybrid_top10_order_different_from_pure_e5_queries": len(changed_order),
             "hybrid_top10_order_different_query_ids": changed_order,
             "maximum_independent_score_absolute_error": max_score_error,
             "whole_graph_candidate_pool_scored_before_E5_top50": True,
             "final_pool_matches_graph_and_production_intersection": True,
             "zero_BM25_has_no_lexical_rank": True, "graph_has_no_third_rank": True,
             "original_source_literal_qrels_reconstructed_independently": True,
             "frozen_inputs_and_outputs_verified": True,
             "manifest_sha256": sha(DATA / "manifest.json"), "report_sha256": sha(REPORT),
             "audit_script_sha256": sha(Path(__file__)), "network_calls": 0, "credentials_read": False,
             "human_annotated": False, "semantic_accuracy_verified": False,
             "new_unseen_queries": False,
             "limitations": ["32 test are fixed condition regression queries; 8 dev reported separately",
                             "literal automatic relevance labels, not human semantic labels",
                             "retrieval metrics precede DeepSeek reranking and QA",
                             "frozen graph export and recipe-level exact vectors, not live Neo4j or latency evidence",
                             "structured entities supplied by benchmark; natural-language parsing not evaluated"]}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "verified": verified,
                      "hybrid_top10_order_changed_queries": len(changed_order)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
