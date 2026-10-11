#!/usr/bin/env python3
"""Read-only live retrieval checks plus frozen, causal ranking ablations.

Never changes production, historical evidence, labels, embeddings, or graphs.
No LLM/API calls or model encoding. Secrets are used only by local Neo4j.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.catalog import RecipeCatalog
from recipegen.config import Settings
from recipegen.graphrag_retrieval import RecipeGraphRAGRetriever
from recipegen.production_hybrid import VerifiedBM25Corpus, lexical_query_text
from recipegen.catalog import _filters
from recipegen.text_embeddings import MODEL, REVISION

DATA = ROOT / "data/diagnostic_hybrid_rrf_v1"
REPORT = ROOT / "reports/hybrid-rrf-diagnostic-v1.json"
OLD = ROOT / "data/evaluation_hybrid_v2"
INPUTS = [
    "data/evaluation_hybrid_v2/queries.jsonl", "data/evaluation_hybrid_v2/qrels.jsonl",
    "data/evaluation_hybrid_v2/query_embeddings.jsonl",
    "data/graphrag/index-manifest.json", "data/graphrag/full/corpus.jsonl",
    "data/graphrag/full/embeddings.jsonl",
    "data/evaluation_serial_hybrid_v1/rankings.jsonl",
    "data/evaluation_graph_first_hybrid_v1/rankings.jsonl",
    "recipegen/graphrag_retrieval.py", "recipegen/production_hybrid.py",
    "recipegen/experimental_retrieval.py", "recipegen/catalog.py",
    "recipegen/config.py", "scripts/diagnose_hybrid_rrf.py",
]
METHODS = {
    "live_hybrid": "生产三路+BM25/E5 RRF（真实Neo4j）",
    "live_semantic": "图谱先筛+E5（真实Neo4j，无BM25/RRF）",
    "graph_bm25": "同一图候选池仅BM25（离线，正分优先，零分按ID）",
    "graph_id_control": "同一图候选池按ID（离线，无词面/语义排序）",
    "remove_graph_seed": "移除C显式并集路，保留图预筛/硬条件，A+B RRF（离线）",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def queries():
    selected = [q for q in rows(OLD / "queries.jsonl")
                if q["category"] in {"multi_ingredient", "ingredient_exclusion"}]
    assert len(selected) == 40 and sum(q["split"] == "test" for q in selected) == 32
    return selected


def prepare():
    assert not DATA.exists(), "Do not overwrite a previous diagnostic"
    DATA.mkdir(parents=True)
    protocol = """# Hybrid/RRF diagnosis v1

Freeze protocol, source/data SHA256 and ablations before new execution. Original
40 previously viewed condition-regression queries, 8 dev/32 test, automatic
literal source-field labels. No new labels, tuning, query rewriting, API calls,
or embeddings. Production and historical artifacts are read-only.

Run live hybrid and graph-first semantic with identical explicit query filters,
cached original E5 vectors and real Neo4j. Independently reconstruct BM25 ranks,
complete-C E5 ranks, RRF k60 and Top50 from live C and original corpus/vectors.
Record every route's seed, C, final pool, graph-only additions, positive lexical
ranks, E5 ranks and both RRF summands. Compare old/new live hybrid ranks and
same-run pure E5 Top10 sequence, set and hits separately.

Ablate BM25 via live semantic. Ablate E5 via BM25-only on the same C (zero-score
recipes follow positive ones, ties by ID). Use C sorted by ID as a deterministic
negative ranking control. Remove ONLY the explicit C union route via
(A Top50 union graph-prefiltered B Top50) intersection C, then rerank remaining
pool using positive BM25 and E5 RRF. This last control retains mandatory graph
conditions and semantic prefilter; it is not a fully graph-free system.

Gold is read after all live/ablation rankings are complete. Macro P@10 uses
hits/10 even if fewer returned; R@10 uses hits/all original qrels. Report only
P@10/R@10 effects plus candidate/sequence/set counts and condition-label bounds.
These labels do not rate flavor, cooking method, semantic intent, evidence or
LLM QA. Any saturated control result is diagnostic, not deployment advice.
"""
    (DATA / "PROTOCOL.md").write_text(protocol)
    write(DATA / "manifest.json", {
        "status": "frozen_before_live_and_ablation_rankings",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS, "queries": 40, "test": 32, "dev": 8,
        "query_ids": [q["query_id"] for q in queries()],
        "input_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "protocol_sha256": digest(DATA / "PROTOCOL.md"),
        "rrf_k": 60, "seeds": 50, "model": MODEL, "revision": REVISION,
        "human_annotated": False, "new_unseen_queries": False, "llm_calls": 0,
    })
    print(json.dumps({"status": "protocol_frozen", "queries": 40}))


def ranked(scores, positive=False):
    return sorted((rid for rid in scores if not positive or scores[rid] > 0),
                  key=lambda rid: (-scores[rid], rid))


def fuse(pool, lexical_scores, dense_scores):
    lexical = ranked({rid: lexical_scores[rid] for rid in pool}, positive=True)
    dense = ranked({rid: dense_scores[rid] for rid in pool})
    lr = {rid: pos for pos, rid in enumerate(lexical, 1)}
    dr = {rid: pos for pos, rid in enumerate(dense, 1)}
    output = []
    for rid in pool:
        first = 1 / (60 + lr[rid]) if rid in lr else 0
        second = 1 / (60 + dr[rid])
        output.append({"recipe_id": rid, "bm25_score": lexical_scores[rid],
                       "cosine_score": dense_scores[rid], "bm25_rank": lr.get(rid),
                       "dense_rank": dr[rid], "bm25_rrf_term": first,
                       "e5_rrf_term": second, "rrf_score": first + second})
    return sorted(output, key=lambda row: (-row["rrf_score"], row["recipe_id"]))


def run():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        assert digest(ROOT / name) == expected, "Frozen input changed: " + name
    assert digest(DATA / "PROTOCOL.md") == manifest["protocol_sha256"]
    assert not REPORT.exists() and not (DATA / "rankings.jsonl").exists()
    original = {r["query_id"]: r for r in rows(ROOT / "data/evaluation_serial_hybrid_v1/rankings.jsonl")}
    prior = {r["query_id"]: r for r in rows(ROOT / "data/evaluation_graph_first_hybrid_v1/rankings.jsonl")
             if r["method"] == "H_graph_first_three_route"}
    qvec = {r["query_id"]: r for r in rows(OLD / "query_embeddings.jsonl")}
    pvec = {r["recipe_id"]: np.asarray(r["embedding"], dtype=np.float32)
            for r in rows(ROOT / "data/graphrag/full/embeddings.jsonl")}
    corpus = VerifiedBM25Corpus(ROOT / "data/graphrag/full/corpus.jsonl",
                               json.loads((ROOT / "data/graphrag/index-manifest.json").read_text()))
    catalog = RecipeCatalog(settings=Settings.from_env())
    output, checks = [], {}

    def check(name, value):
        checks[name] = bool(value)
        assert value, name

    try:
        state = catalog.status()
        check("real_neo4j", state["backend"] == "neo4j" and state["graph"]["recipes"] == 5898)
        retriever = RecipeGraphRAGRetriever(catalog)
        for pos, query in enumerate(queries(), 1):
            qid = query["query_id"]
            v = np.asarray(qvec[qid]["embedding"], dtype=np.float32)
            v = v / np.linalg.norm(v)
            common = {"ingredients": query["ingredients"], "excluded_ingredients": query["excluded_ingredients"],
                      "limit": 10, "query_vector": qvec[qid]["embedding"], "embedding_model": MODEL,
                      "embedding_revision": REVISION, "infer_query_filters": False}
            hybrid = retriever.search(query["query"], mode="hybrid", **common)
            semantic = retriever.search(query["query"], mode="semantic", **common)
            h, s = hybrid["retrieval"], semantic["retrieval"]
            check(qid + "_live_modes", h["effective_mode"] == "hybrid" and s["effective_mode"] == "semantic"
                  and not h["fallback_reason"] and not s["fallback_reason"])
            c = set(h["seed_ids"]["graph"])
            check(qid + "_complete_C_pool", c == set(h["candidate_pool"])
                  and len(c) == h["scored_recipe_count"] == s["scored_recipe_count"])
            ids = sorted(c)
            matrix = np.asarray([pvec[rid] for rid in ids], dtype=np.float32)
            dense = dict(zip(ids, map(float, matrix @ v))) if ids else {}
            text = lexical_query_text(query["query"], _filters(query["ingredients"], "ingredients"))
            lexical = corpus.scores(text)
            rebuilt = fuse(c, lexical, dense)
            actual = h["candidate_rankings"]
            check(qid + "_independent_RRF_order", [r["recipe_id"] for r in rebuilt[:50]]
                  == [r["recipe_id"] for r in actual])
            check(qid + "_independent_RRF_terms", all(a["bm25_rank"] == r["bm25_rank"]
                and a["dense_rank"] == r["dense_rank"] and abs(a["rrf_score"] - r["rrf_score"]) < 1e-14
                and abs(a["bm25_score"] - r["bm25_score"]) < 1e-10
                and abs(a["cosine_score"] - r["cosine_score"]) < 1e-5
                for a, r in zip(actual, rebuilt)))
            live_h = [r["id"] for r in hybrid["recipes"]]
            live_s = [r["id"] for r in semantic["recipes"]]
            check(qid + "_live_semantic_order", live_s == ranked(dense)[:10])
            check(qid + "_current_original_hybrid", live_h == original[qid]["returned_ids"])
            check(qid + "_prior_hybrid_order", live_h == [r["recipe_id"] for r in prior[qid]["ranking"][:10]])
            check(qid + "_seeds_real", h["seed_ids"]["bm25"] == ranked(lexical, positive=True)[:50]
                  and h["seed_ids"]["dense"] == ranked(dense)[:50])
            ab_pool = (set(h["seed_ids"]["bm25"]) | set(h["seed_ids"]["dense"])) & c
            ablated = fuse(ab_pool, lexical, dense)
            methods = {"live_hybrid": live_h, "live_semantic": live_s,
                       "graph_bm25": ranked({rid: lexical[rid] for rid in c})[:10],
                       "graph_id_control": sorted(c)[:10],
                       "remove_graph_seed": [r["recipe_id"] for r in ablated[:10]]}
            output.append({"query_id": qid, "query": query["query"], "split": query["split"],
                "category": query["category"], "methods": methods, "seed_ids": h["seed_ids"],
                "graph_candidate_ids": sorted(c), "candidate_pool": h["candidate_pool"],
                "candidate_counts": h["candidate_counts"], "independent_complete_RRF": rebuilt,
                "remove_graph_seed_pool": sorted(ab_pool), "live_trace": hybrid["trace"],
                "hybrid_vs_semantic_sequence_changed": live_h != live_s,
                "hybrid_vs_semantic_set_changed": set(live_h) != set(live_s),
                "graph_only_final_top10_ids": [rid for rid in live_h if rid not in ab_pool]})
            if pos % 10 == 0:
                print(json.dumps({"event": "live_hybrid_and_semantic", "queries_complete": pos}), flush=True)
        after = catalog.status()
        check("graph_build_and_counts_unchanged", state["build"] == after["build"] and state["graph"] == after["graph"])
    finally:
        catalog.close()
    # Relevance labels do not enter any preceding live or ablated retrieval.
    gold = defaultdict(set)
    for row in rows(OLD / "qrels.jsonl"):
        gold[row["query_id"]].add(row["recipe_id"])
    for row in output:
        relevant = gold[row["query_id"]]
        row["metrics"] = {}
        for name, ids in row["methods"].items():
            hits = len(set(ids) & relevant)
            row["metrics"][name] = {"hits_at_10": hits, "returned_at_10": len(ids),
                                      "precision_at_10": hits/10, "recall_at_10": hits/len(relevant)}
        row["qrels_count"] = len(relevant)
        c = set(row["graph_candidate_ids"])
        reachable = len(c & relevant)
        row["graph_pool_all_relevant"] = c <= relevant
        row["graph_reachable_qrels"] = reachable
        row["precision_at_10_ceiling"] = min(10, reachable) / 10
        row["recall_at_10_ceiling"] = min(10, reachable) / len(relevant)
        row["hits_same_hybrid_semantic"] = row["metrics"]["live_hybrid"]["hits_at_10"] == row["metrics"]["live_semantic"]["hits_at_10"]
    summaries, diagnostics = [], []
    for split in ("dev", "test"):
        group = [r for r in output if r["split"] == split]
        for name in METHODS:
            summaries.append({"split": split, "method": name, "name": METHODS[name], "queries": len(group),
                "precision_at_10": sum(r["metrics"][name]["precision_at_10"] for r in group)/len(group),
                "recall_at_10": sum(r["metrics"][name]["recall_at_10"] for r in group)/len(group)})
        diagnostics.append({"split": split, "queries": len(group),
            "hybrid_E5_sequence_changed_queries": sum(r["hybrid_vs_semantic_sequence_changed"] for r in group),
            "hybrid_E5_set_changed_queries": sum(r["hybrid_vs_semantic_set_changed"] for r in group),
            "hybrid_E5_same_hit_count_queries": sum(r["hits_same_hybrid_semantic"] for r in group),
            "entire_graph_pool_relevant_queries": sum(r["graph_pool_all_relevant"] for r in group),
            "fewer_than_10_candidates_queries": sum(len(r["candidate_pool"]) < 10 for r in group),
            "C_added_candidates_queries": sum(r["candidate_counts"]["graph_only_added"] > 0 for r in group),
            "C_added_final_Top10_queries": sum(bool(r["graph_only_final_top10_ids"]) for r in group),
            "removed_C_seed_Top10_changed_queries": sum(r["methods"]["remove_graph_seed"] != r["methods"]["live_hybrid"] for r in group),
            "mean_precision_at_10_ceiling": sum(r["precision_at_10_ceiling"] for r in group)/len(group),
            "mean_recall_at_10_ceiling": sum(r["recall_at_10_ceiling"] for r in group)/len(group)})
    for name, expected in manifest["input_sha256"].items():
        check("unchanged_" + name, digest(ROOT / name) == expected)
    ranking_path = DATA / "rankings.jsonl"
    ranking_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    report = {"status": "passed", "scope": "live_neo4j_and_offline_causal_ablations_before_LLM",
        "completed_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "summary": summaries, "diagnostics": diagnostics, "llm_calls": 0, "encoding_calls": 0,
        "human_annotated": False, "new_unseen_queries": False,
        "manifest_sha256": digest(DATA / "manifest.json"), "rankings_sha256": digest(ranking_path)}
    write(REPORT, report)
    md = ["# 三路检索与 RRF 核查\n", "40 条原条件回归查询；真实 Neo4j 重跑混合/E5，两路 RRF 独立重算。",
          "32 test 主结果，8 dev 单列；原字段自动标签，DeepSeek 重排前。\n",
          "| 对照 | Precision@10 | Recall@10 |", "|---|---:|---:|"]
    for item in summaries:
        if item["split"] == "test":
            md.append(f"| {item['name']} | {item['precision_at_10']*100:.2f}% | {item['recall_at_10']*100:.2f}% |")
    md += ["\n逐题完整候选、独立 RRF 两项分数与消融见 `data/diagnostic_hybrid_rrf_v1/rankings.jsonl`。",
           "移除 C 显式候选路仍保留图预筛与硬条件，不能称为完全关闭图谱。ID对照仅诊断标签敏感性。",
           "完整 C 使候选池固定，排序和 Top10 命中仍可能改变；本次 test 逐题命中恰好相同。",
           "原标签只判断食材字面条件；不能据此证明混合排序比单路好，或问答效果改善。\n"]
    REPORT.with_suffix(".md").write_text("\n".join(md))
    print(json.dumps({"status": "passed", "summary": summaries, "diagnostics": diagnostics}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
