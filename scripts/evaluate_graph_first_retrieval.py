#!/usr/bin/env python3
"""Frozen graph-before-dense comparison; cached E5, offline, no LLM/.env.

Keep the existing v1/v2 benchmarks untouched. Label IDs enter scoring only.
All arms share float32 dense scores, so only candidate selection order changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_real_retrieval import (
    deny_network, digest, eligible, json_read, lines, log, write_json, write_lines,
)
from scripts.evaluate_hybrid_retrieval import metric_values, verify as verify_v2
from recipegen.catalog import _filters
from recipegen.experimental_retrieval import GraphCandidateIndex
from recipegen.text_embeddings import MODEL, REVISION

DATA = ROOT / "data/evaluation_graph_first_v1"
V2 = ROOT / "data/evaluation_hybrid_v2"
REPORT = ROOT / "reports/graph-first-retrieval-v1.json"
CATEGORIES = {"multi_ingredient", "ingredient_exclusion"}
METHODS = {
    "A_current_e5_then_mentions": "E5全库Top50→原生产提及过滤",
    "B_control_e5_then_graph": "E5全库Top50→图谱过滤（同谓词控制）",
    "C_graph_then_e5": "图谱筛全部合法候选→E5排序",
    "D_control_mentions_then_e5": "原生产提及筛全部候选→E5排序（顺序控制）",
}
PARAMETERS = {
    "ks": [1, 5, 10], "global_dense_limit": 50, "saved_ranking_limit": 50,
    "embedding_granularity": "original_one_document_per_recipe",
    "dense_scoring": "shared_cached_float32_dot_of_original_L2_normalized_vectors",
    "tie_break": "recipe_id_ascending", "query_rewrite": False,
    "graph_predicate": "all_positive_Ingredient_mentions_AND_NOT_any_excluded_mention",
    "graph_candidate_limit_before_semantic_ranking": None,
    "tuning": "none", "bm25": False, "rrf": False, "llm": False,
    "reuse_scores_for_order_ablation_not_latency_measurement": True,
}
INPUTS = [
    "data/evaluation_hybrid_v2/manifest.json",
    "data/evaluation_hybrid_v2/vector-manifest.json",
    "data/evaluation_hybrid_v2/queries.jsonl",
    "data/evaluation_hybrid_v2/qrels.jsonl",
    "data/evaluation_hybrid_v2/query_embeddings.jsonl",
    "data/evaluation_hybrid_v2/rankings.jsonl",
    "data/graphrag/full/embeddings.jsonl",
    "data/evaluation_real_v1/production_flat_metadata.jsonl",
    "data/test_graph/nodes.jsonl", "data/test_graph/relationships.jsonl",
    "recipegen/catalog.py", "recipegen/experimental_retrieval.py",
    "scripts/evaluate_real_retrieval.py", "scripts/evaluate_hybrid_retrieval.py",
    "scripts/evaluate_graph_first_retrieval.py",
]


def protocol() -> str:
    return """# RecipeGen 图谱先筛选与向量后过滤对比 v1

在新排名及指标计算前冻结本协议与输入SHA256。不改变原v1/v2标签、查询、向量、图谱或代码。

## 范围
5898道食谱，复用原食谱级multilingual-e5-small同revision的384维向量与原问题向量；不重新编码，不改写中文问题。沿用v2的40道条件回归题：8 dev、32 test（多食材16 test、排除16 test），两类各4 dev。此前结果已经查看，本轮属于固定回归比较，不能称新未见测试。所有qrels来自独立扫描原始标题和步骤的字面自动标签，非人工语义标签。原数据集split=test与查询dev/test是不同概念。

## 四组
A：复现旧A_recipe_e5：全库E5 Top50，再由冻结production_flat_metadata及catalog._filters执行提及条件过滤，不补搜。
B：全库同一E5 Top50，再与图谱全部合法候选取交集，不补搜；与C使用完全相同图谓词。
C：从冻结实际Ingredient节点与HAS_INGREDIENT/USES_INGREDIENT关系筛出全部合法食谱，再在其中按同一E5分数排序。所有正向食材集合求交，排除食材集合求差；不限图候选数。
D：先用与A完全相同的生产提及谓词筛全集，再按同一E5分数排序；作为仅改变旧链路执行顺序的附加控制。

全组按(-float32 cosine, recipe_id)稳定排序。离线对照统一预计算全库cosine以复用完全相同分数，仅在候选池内排序；不据此报告线上效率或声称实际减少了向量计算。没有BM25/RRF/重排/LLM；不读.env，不访问网络。E5负责相似度，图谱负责字典提及关系约束；图谱候选不读取gold词表或qrel ID。基准显式提供食材条件，没有验证自然问题解析。图谱提及不是实际使用或完整食材库存证明。

## 验证与指标
检查图谱Recipe ID、原5898食谱向量ID与生产提及缓存ID完全相同；向量模型revision一致、维度384、有限值、L2归一化。旧A逐题Top50排名和全部k指标必须与冻结v2吻合。B必须是C全部排序的前缀，A必须是D全部排序的前缀；同谓词下固定k相关命中数不得下降。图谱/生产候选差异单列，不修订自动标签。

主报32条test条件题的宏Precision@10与Recall@10，附k=1/5/10、每类与dev独立结果、实际返回Top10数量、空题数、MRR@10、候选数、图谱可达率与Top10可达Recall上限。P@k=hits/k，不足k仍固定分母；R@k=hits/全部qrels数；空结果为0。逐题排名与图候选ID保存以便复核。

本轮使用冻结图导出和NumPy精确相似度，不是实时Neo4j ANN/Cypher测试；不部署新方案，不修改网页默认检索或简历。
"""


def prepare() -> None:
    deny_network()
    if (DATA / "manifest.json").exists():
        raise FileExistsError("Comparison already frozen; use --run")
    original = verify_v2()
    vm = json_read(V2 / "vector-manifest.json")
    if vm["benchmark_manifest_sha256"] != digest(V2 / "manifest.json"):
        raise ValueError("Vector manifest differs from original benchmark")
    if digest(V2 / "query_embeddings.jsonl") != vm["artifact_sha256"]["query_embeddings.jsonl"]:
        raise ValueError("Original query vectors changed")
    queries = [q for q in lines(V2 / "queries.jsonl") if q["category"] in CATEGORIES]
    if len(queries) != 40 or sum(q["split"] == "test" for q in queries) != 32:
        raise ValueError("Expected 40 frozen condition queries and 32 test queries")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "PROTOCOL.md").write_text(protocol(), encoding="utf-8")
    write_json(DATA / "manifest.json", {
        "status": "frozen_before_new_rankings", "benchmark": "graph-first-retrieval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "parameters": PARAMETERS,
        "model": MODEL, "model_revision": REVISION, "source_recipes": original["source_recipes"],
        "queries": len(queries), "test_queries": 32, "dev_queries": 8,
        "query_ids": [q["query_id"] for q in queries], "methods": METHODS,
        "input_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "protocol_sha256": digest(DATA / "PROTOCOL.md"),
        "human_annotated": False, "new_unseen_queries": False,
        "label_semantics": "original_source_literal_mentions_not_human_semantic_relevance",
    })
    log("comparison_frozen", queries=40, test_queries=32, protocol=str(DATA / "PROTOCOL.md"))


def verify() -> dict:
    manifest = json_read(DATA / "manifest.json")
    if manifest["parameters"] != PARAMETERS or manifest["methods"] != METHODS:
        raise ValueError("Frozen experiment configuration changed")
    if digest(DATA / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise ValueError("Frozen comparison protocol changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    return manifest


def retrieve(query, ids, scores, features, graph):
    """No relevance labels, target IDs or gold counts used in retrieval."""
    includes = _filters(query["ingredients"], "ingredients")
    excludes = _filters(query["excluded_ingredients"], "excluded_ingredients")
    mention_ids = {rid for rid in ids if eligible(features[rid], includes, excludes)}
    graph_ids = graph.candidates(query["ingredients"], query["excluded_ingredients"])
    if not graph_ids <= set(ids):
        raise ValueError("Graph candidate outside frozen vector domain")
    ordered = sorted(ids, key=lambda rid: (-scores[rid], rid))
    global_seed = ordered[:PARAMETERS["global_dense_limit"]]
    # The new method sorts only the complete graph-qualified candidate pool.
    graph_ranked = sorted(graph_ids, key=lambda rid: (-scores[rid], rid))
    mentions_ranked = sorted(mention_ids, key=lambda rid: (-scores[rid], rid))
    rankings = {
        "A_current_e5_then_mentions": [rid for rid in global_seed if rid in mention_ids],
        "B_control_e5_then_graph": [rid for rid in global_seed if rid in graph_ids],
        "C_graph_then_e5": graph_ranked,
        "D_control_mentions_then_e5": mentions_ranked,
    }
    for post, pre in [("A_current_e5_then_mentions", "D_control_mentions_then_e5"),
                      ("B_control_e5_then_graph", "C_graph_then_e5")]:
        if rankings[post] != rankings[pre][:len(rankings[post])]:
            raise ValueError("Same-predicate postfilter ranking is not a prefix of prefilter")
    return rankings, global_seed, mention_ids, graph_ids


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["category"], row["method"])].append(row)
        groups[(row["split"], "conditions_combined", row["method"])].append(row)
    summaries = []
    for (split, category, method), values in sorted(groups.items()):
        summaries.append({
            "split": split, "category": category, "method": method, "queries": len(values),
            "macro_metrics": {k: sum(v["metrics"][k] for v in values) / len(values)
                              for k in values[0]["metrics"]},
            "empty_results": sum(v["returned_at_10"] == 0 for v in values),
            "queries_returning_fewer_than_10": sum(v["returned_at_10"] < 10 for v in values),
            "mean_returned_at_10": sum(v["returned_at_10"] for v in values) / len(values),
            "mean_candidate_count_before_final_cut": sum(v["candidate_count"] for v in values) / len(values),
            "mean_qrels_count": sum(v["qrels_count"] for v in values) / len(values),
            "mean_predicate_gold_reachable_fraction": sum(v["predicate_gold_reachable_fraction"] for v in values) / len(values),
            "macro_top10_recall_ceiling_under_predicate": sum(v["top10_recall_ceiling_under_predicate"] for v in values) / len(values),
        })
    return summaries


def run() -> None:
    deny_network()
    tick = time.monotonic()
    manifest = verify()
    import numpy as np
    queries = [q for q in lines(V2 / "queries.jsonl") if q["query_id"] in manifest["query_ids"]]
    pvec_rows = list(lines(ROOT / "data/graphrag/full/embeddings.jsonl"))
    pvec = {r["recipe_id"]: r["embedding"] for r in pvec_rows}
    if len(pvec_rows) != len(pvec) or len(pvec) != 5898:
        raise ValueError("Recipe vectors are duplicate or incomplete")
    ids = sorted(pvec)
    features = {r["recipe_id"]: set(r["production_mention_names"])
                for r in lines(ROOT / "data/evaluation_real_v1/production_flat_metadata.jsonl")}
    graph = GraphCandidateIndex(lines(ROOT / "data/test_graph/nodes.jsonl"),
                                lines(ROOT / "data/test_graph/relationships.jsonl"))
    if graph.recipe_ids != set(ids) or set(features) != set(ids):
        raise ValueError("Graph, vectors and production metadata domains differ")
    qvec = {r["query_id"]: r for r in lines(V2 / "query_embeddings.jsonl")}
    for query in queries:
        cached = qvec[query["query_id"]]
        if cached["query"] != query["query"] or cached["model"] != MODEL or cached["model_revision"] != REVISION:
            raise ValueError("Query vector text/model/revision mismatch")
    matrix = np.asarray([pvec[rid] for rid in ids], dtype=np.float32)
    if matrix.shape != (5898, 384) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01):
        raise ValueError("Recipe matrix invalid or not L2 normalized")
    golden = defaultdict(set)
    for row in lines(V2 / "qrels.jsonl"):
        golden[row["query_id"]].add(row["recipe_id"])
    baseline = {r["query_id"]: r for r in lines(V2 / "rankings.jsonl")
                if r["method"] == "A_recipe_e5" and r["variant"] == "with_structured_filter"}
    results, candidate_rows = [], []
    for query in queries:
        qid = query["query_id"]
        vector = np.asarray(qvec[qid]["embedding"], dtype=np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=.01):
            raise ValueError("Query vector invalid or not L2 normalized")
        scores = dict(zip(ids, map(float, matrix @ vector)))
        # Only visible query text and explicit conditions go to retrieval.
        visible = {k: query[k] for k in ["query", "ingredients", "excluded_ingredients"]}
        ranked, global_seed, mention_ids, graph_ids = retrieve(visible, ids, scores, features, graph)
        relevant = golden[qid]
        if not relevant <= set(ids) or len(relevant) != query["gold_relevant_count"]:
            raise ValueError("Frozen labels/domain mismatch")
        old_ids = [r["recipe_id"] for r in baseline[qid]["ranking"]]
        if ranked["A_current_e5_then_mentions"] != old_ids:
            raise ValueError("Original A ranking not exactly reproduced: " + qid)
        candidate_rows.append({"query_id": qid, "split": query["split"], "category": query["category"],
            "global_seed_ids": global_seed, "graph_candidate_ids": sorted(graph_ids),
            "production_candidate_ids": sorted(mention_ids),
            "graph_only_ids": sorted(graph_ids - mention_ids),
            "production_only_ids": sorted(mention_ids - graph_ids),
            "same_predicate_prefix_invariants_passed": True})
        for method, all_ids in ranked.items():
            kept = all_ids[:PARAMETERS["saved_ranking_limit"]]
            ranking = [{"recipe_id": rid, "cosine_score": scores[rid]} for rid in kept]
            metrics = metric_values(query["category"], ranking, relevant)
            if method == "A_current_e5_then_mentions" and metrics != baseline[qid]["metrics"]:
                raise ValueError("Original A metrics not exactly reproduced: " + qid)
            predicate = graph_ids if method in {"B_control_e5_then_graph", "C_graph_then_e5"} else mention_ids
            reachable = relevant & predicate
            results.append({"query_id": qid, "query": query["query"], "split": query["split"],
                "category": query["category"], "method": method, "metrics": metrics,
                "qrels_count": len(relevant), "ranking": ranking, "candidate_count": len(all_ids),
                "predicate_candidate_count": len(predicate), "returned_at_10": min(len(kept), 10),
                "predicate_gold_reachable_count": len(reachable),
                "predicate_gold_reachable_fraction": len(reachable) / len(relevant),
                "top10_recall_ceiling_under_predicate": min(10, len(reachable)) / len(relevant),
                "gold_hits_at_10": len(set(kept[:10]) & relevant),
                "missed_gold_at_10": sorted(relevant - set(kept[:10]))})
        values = {r["method"]: r for r in results[-4:]}
        for post, pre in [("A_current_e5_then_mentions", "D_control_mentions_then_e5"),
                          ("B_control_e5_then_graph", "C_graph_then_e5")]:
            for k in PARAMETERS["ks"]:
                if values[pre]["metrics"][f"precision_at_{k}"] < values[post]["metrics"][f"precision_at_{k}"]:
                    raise ValueError("Same-predicate prefilter hits unexpectedly decreased")
    write_lines(DATA / "rankings.jsonl", results)
    write_lines(DATA / "candidates.jsonl", candidate_rows)
    report = {
        "status": "completed", "benchmark": manifest["benchmark"], "parameters": PARAMETERS,
        "model": MODEL, "model_revision": REVISION, "dimension": 384,
        "source_recipes": 5898, "queries": len(queries), "test_queries": 32, "dev_queries": 8,
        "ranking_rows": len(results), "methods": METHODS,
        "scope": "frozen_graph_export_and_exact_cached_cosine_not_live_Neo4j_ANN",
        "human_annotated": False, "semantic_accuracy_verified": False,
        "new_unseen_queries": False, "llm_calls": 0, "network_calls": 0,
        "dotenv_read": False, "model_encoding_calls": 0, "model_changed": False,
        "baseline_exact_ranking_and_metric_reproduction_queries": len(queries),
        "prefix_and_nondecreasing_hit_checks_queries": len(queries),
        "same_graph_and_production_candidate_queries": sum(r["graph_candidate_ids"] == r["production_candidate_ids"] for r in candidate_rows),
        "summary": summarize(results), "elapsed_seconds": time.monotonic() - tick,
        "manifest_sha256": digest(DATA / "manifest.json"),
        "output_sha256": {name: digest(DATA / name) for name in ["rankings.jsonl", "candidates.jsonl"]},
    }
    write_json(REPORT, report)
    write_report_md(report, results)
    log("comparison_complete", report=str(REPORT),
        test_summary=[r for r in report["summary"] if r["split"] == "test" and r["category"] == "conditions_combined"])


def write_report_md(report, rows):
    out = ["# RecipeGen 图谱先筛选对比 v1", "",
        "**冻结回归题、原始字段自动标签；不是人工语义评测。**", "",
        "复用5898条食谱级E5向量、同一中文问题向量；32 test条件题（多食材16、排除16），8 dev单列。",
        "精确float32 cosine离线对照；没有重编码、查询改写、BM25、RRF或LLM，也没有部署网页改动。",
        "P@k固定分母k；R@k使用完整原qrels；按查询宏平均，空结果计0。", ""]
    for category in ["conditions_combined", "multi_ingredient", "ingredient_exclusion"]:
        out += [f"## test / {category}", "",
            "| 方法 | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 | 空题数 | 平均返回@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            value = next(s for s in report["summary"] if s["split"] == "test" and s["category"] == category and s["method"] == method)
            m = value["macro_metrics"]
            out.append(f"| {METHODS[method]} | {m['precision_at_1']:.2%} | {m['recall_at_1']:.2%} | {m['precision_at_5']:.2%} | {m['recall_at_5']:.2%} | {m['precision_at_10']:.2%} | {m['recall_at_10']:.2%} | {value['empty_results']}/{value['queries']} | {value['mean_returned_at_10']:.3f} |")
        out.append("")
    out += ["## 可复核实例：西兰花与胡萝卜", "",
        "multi_ingredient-11，原始字段自动标签共40条：", "",
        "| 方法 | 筛选后候选数 | 返回@10 | 命中@10 | P@10 | R@10 |",
        "|---|---:|---:|---:|---:|---:|"]
    for value in rows:
        if value["query_id"] == "multi_ingredient-11":
            m = value["metrics"]
            out.append(f"| {METHODS[value['method']]} | {value['candidate_count']} | {value['returned_at_10']} | {value['gold_hits_at_10']} | {m['precision_at_10']:.2%} | {m['recall_at_10']:.2%} |")
    out += ["", "## 复核与边界", "",
        f"- 旧A逐题排名和所有指标完全复现：{report['baseline_exact_ranking_and_metric_reproduction_queries']}/40；同谓词前缀与命中不下降检查：{report['prefix_and_nondecreasing_hit_checks_queries']}/40。",
        f"- 图谱与原生产提及过滤的候选集合完全相同：{report['same_graph_and_production_candidate_queries']}/40；集合差异逐题保存。",
        "- 图谱筛选穷举全部合法候选，旧方法全库Top50后不补搜，搜索覆盖范围不同；本轮测效果，不报告效率。",
        "- 字面标签与图实体提及存在复合词/标题差异；排除关系不能证明完整食材清单或过敏安全。",
        "- 食材条件由基准显式提供，不证明自然问题解析、LLM回答质量或语义排序质量。",
        "- 已查看的回归测试结果；不当作新的未见查询泛化结果。", "",
        "输入冻结与方法定义见 data/evaluation_graph_first_v1/PROTOCOL.md 和 manifest.json；逐题排名与图候选分别见 rankings.jsonl、candidates.jsonl。JSON报告还保存dev结果、候选数、图谱可达率与Recall上限。", ""]
    REPORT.with_suffix(".md").write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == "__main__":
    main()
