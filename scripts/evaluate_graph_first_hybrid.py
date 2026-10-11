#!/usr/bin/env python3
"""Frozen three-route BM25/graph-prefiltered-E5/graph regression evaluation.

No API, network, .env, model loading, re-encoding, query rewriting, or tuning.
Previous experiment files are read only; each new run verifies frozen hashes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_real_retrieval import (
    deny_network, digest, eligible, json_read, lines, log, write_json, write_lines,
)
from scripts.evaluate_hybrid_retrieval import lexical_text, metric_values
from recipegen.catalog import _filters
from recipegen.experimental_retrieval import BM25Index, GraphCandidateIndex, lexical_tokens
from recipegen.graph_first_hybrid import graph_condition_candidates, graph_first_hybrid
from recipegen.text_embeddings import MODEL, REVISION

DATA = ROOT / "data/evaluation_graph_first_hybrid_v1"
V2 = ROOT / "data/evaluation_hybrid_v2"
GRAPH_FIRST = ROOT / "data/evaluation_graph_first_v1"
REPORT = ROOT / "reports/graph-first-hybrid-v1.json"
CATEGORIES = {"multi_ingredient", "ingredient_exclusion"}
METHODS = {
    "D_recipe_graph_saved": "旧三路混合（冻结v2保存排名，不重跑旧代码）",
    "C_graph_then_e5": "单路：图谱全部条件筛选→E5排序（本轮实际重算）",
    "H_graph_first_three_route": "三路：BM25＋图谱先筛E5＋图谱候选→条件核对→双排名RRF",
}
PARAMETERS = {
    "ks": [1, 5, 10], "lexical_seed_limit": 50, "dense_seed_limit": 50,
    "saved_ranking_limit": 50, "rrf_k": 60, "bm25_k1": 1.5, "bm25_b": .75,
    "embedding_granularity": "original_one_document_per_recipe",
    "dense_scoring": "float32_dot_only_on_complete_graph_qualified_recipe_matrix",
    "graph_candidate_limit_before_semantic_ranking": None,
    "graph_predicate": "production_alias_OR_per_term_then_positive_AND_then_excluded_set_difference",
    "condition_recheck": "graph_qualified_membership_AND_original_production_mention_names_with_catalog._filters",
    "keyword_seed": "whole_corpus_positive_BM25_Top50_no_refill_after_filter",
    "lexical_query": "unchanged_v2_lexical_text_from_visible_conditions_and_production_aliases",
    "semantic_seed": "complete_graph_qualified_recipe_pool_E5_Top50",
    "graph_seed": "complete_unordered_graph_qualified_recipe_set",
    "fusion": "entire_filtered_union_BM25_positive_rank_and_E5_rank_RRF_k60",
    "graph_has_rank": False, "tie_break": "recipe_id_ascending",
    "query_rewrite": False, "tuning": "none", "llm": False,
    "metric_stage": "retrieval_before_DeepSeek_reranking_and_QA",
}
INPUTS = [
    "data/evaluation_hybrid_v2/manifest.json",
    "data/evaluation_hybrid_v2/vector-manifest.json",
    "data/evaluation_hybrid_v2/queries.jsonl", "data/evaluation_hybrid_v2/qrels.jsonl",
    "data/evaluation_hybrid_v2/query_embeddings.jsonl",
    "data/evaluation_hybrid_v2/rankings.jsonl",
    "data/evaluation_hybrid_v2/recipe_documents.jsonl",
    "data/evaluation_graph_first_v1/manifest.json",
    "data/evaluation_graph_first_v1/rankings.jsonl",
    "data/graphrag/full/embeddings.jsonl",
    "data/evaluation_real_v1/production_flat_metadata.jsonl",
    "data/test_graph/nodes.jsonl", "data/test_graph/relationships.jsonl",
    "recipegen/catalog.py", "recipegen/experimental_retrieval.py",
    "recipegen/graph_first_hybrid.py", "recipegen/text_embeddings.py",
    "scripts/evaluate_real_retrieval.py", "scripts/evaluate_hybrid_retrieval.py",
    "scripts/evaluate_graph_first_hybrid.py", "tests/test_graph_first_hybrid.py",
]


def protocol():
    return """# RecipeGen 三路混合检索：语义路图谱先筛 v1

本协议、全部使用的输入SHA256和参数在新结果计算前冻结；不改变历史数据、查询、qrels、向量、参数、排名或manifest。

## 数据与边界
沿用5898道食谱、原食谱级intfloat/multilingual-e5-small（同revision、384维、原L2向量），原中文问题向量不重新编码。沿用原40道食材条件回归题，8 dev/32 test：多食材、食材排除各4 dev/16 test。已有结果曾被查看，不是新的未见泛化测试。不加新题、不重新抽样、不按本轮结果选择参数。

原qrels独立扫描原始标题/步骤构造字面自动标签；非人工语义相关性标签。图候选只从原图Ingredient节点及HAS_INGREDIENT、USES_INGREDIENT关系得到，禁止从qrels、目标recipe ID或gold词表构造。基准显式提供结构化食材条件，未评测DeepSeek自然语言解析。检索函数不接收gold；gold只在排名完成后计算指标。

## 新三路方法（固定参数）
1. BM25：对原标题和完整步骤食谱文档计算正IDF Okapi BM25，k1=1.5/b=.75，NFKC小写英文词＋中文单字分词；词面输入与旧v2完全一致，由可见正向食材经生产别名表展开英文词，无gold扩词；全库正分Top50，之后过滤不补搜。
2. 语义：图谱先筛全部满足条件的Recipe ID（每个正向食材生产别名OR，多食材AND，排除食材集合差）；仅在这些ID对应的recipe向量矩阵进行float32 dot(query, recipe)，然后E5 Top50。严格禁止先全库E5 Top50再过滤，也不预先全库计算cosine。
3. 图谱：把同一完整合法Recipe集合作为独立、无序候选路加入并集，不截断，不给第三个rank或分数。
4. 三路候选并集后，保留图谱条件合格集合，并用冻结production_flat_metadata中的原提及字段与catalog._filters再次核对正向AND、排除条件。最终pool为graph合格集合与生产条件合格集合的交集，所有最终候选均有实际E5分数。对全部剩余候选重新生成真实BM25正分排名和E5排名，RRF=1/(60+BM25rank)+1/(60+E5rank)。零BM25无词面rank；图候选只扩集合，不人为贡献rank。按(-RRF,recipe_id)稳定取Top10，保存Top50。

图谱路提供完整合法池，因此本轮语义Top50改变通常不会再次扩大最终pool；这一点须如实报告，不能声称新三路必然优于旧三路。只有语义路先筛真实执行，不能声称实时Neo4j/网页默认已经改为本离线方法。

## 对照
旧D_recipe_graph：使用原v2 with_structured_filter保存排名及指标，核验原manifest对数据、模型、向量、原查询与标签的hash；旧生产源码因之前图谱先筛改造已漂移，不再重跑，不修订旧manifest。完整记录旧源码hash差异。
单路图谱→E5：本轮在同一图合格候选矩阵实际计算，再按(-E5,recipe_id)排序；记录与历史图谱先筛保存排名的差异，避免把历史分数当作新计算。
新三路：按上述固定流程实际计算。

## 指标
主结果为32 test条件题的宏平均Precision@10、Recall@10，8 dev独立报告。每题P@k=相关命中/k（不足k仍固定分母），R@k=相关命中/原qrels全部相关数，空结果计0，k=1/5/10。报告逐题seed、graph候选、condition核对前后pool、BM25/E5分数及rank、RRF分数和代码hash。图提及不等于完整食材清单或过敏安全；指标是字面自动标签条件回归，不证明自由语义问答质量。

DeepSeek用于产品意图解析、候选重排与中文问答，但本轮仅测检索重排前指标；LLM/API/网络/.env读取/重编码次数全为0，不修改网页默认。
"""


def legacy_integrity():
    """Verify immutable historical data, preserve and disclose code drift."""
    legacy = json_read(V2 / "manifest.json")
    drift, checked = [], []
    for name, expected in legacy["input_sha256"].items():
        actual = digest(ROOT / name)
        if actual != expected:
            if not name.endswith(".py"):
                raise ValueError("Historical non-code input changed: " + name)
            drift.append({"path": name, "historical_sha256": expected, "current_sha256": actual})
        else:
            checked.append(name)
    for name, expected in legacy["artifact_sha256"].items():
        if digest(V2 / name) != expected:
            raise ValueError("Historical benchmark artifact changed: " + name)
    vectors = json_read(V2 / "vector-manifest.json")
    if vectors["benchmark_manifest_sha256"] != digest(V2 / "manifest.json"):
        raise ValueError("Historical vector manifest differs from benchmark")
    for name, expected in vectors["artifact_sha256"].items():
        if digest(V2 / name) != expected:
            raise ValueError("Historical vector artifact changed: " + name)
    if vectors["model"] != MODEL or vectors["model_revision"] != REVISION or vectors["dimension"] != 384:
        raise ValueError("Historical model revision or vector dimension drift")
    return {"historical_non_code_inputs_and_artifacts_verified": True,
            "unchanged_input_paths": checked, "historical_code_drift": drift,
            "old_D_used_as_saved_reference_not_reexecuted": True,
            "legacy_artifact_sha256": legacy["artifact_sha256"],
            "legacy_vector_artifact_sha256": vectors["artifact_sha256"]}


def prepare():
    deny_network()
    if (DATA / "manifest.json").exists():
        raise FileExistsError("Already frozen; use --run")
    legacy = legacy_integrity()
    queries = [q for q in lines(V2 / "queries.jsonl") if q["category"] in CATEGORIES]
    if len(queries) != 40 or sum(q["split"] == "test" for q in queries) != 32:
        raise ValueError("Expected original 40 conditions: 8 dev, 32 test")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "PROTOCOL.md").write_text(protocol(), encoding="utf-8")
    write_json(DATA / "manifest.json", {
        "status": "frozen_before_new_rankings", "benchmark": "graph-first-hybrid-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": PARAMETERS, "methods": METHODS,
        "model": MODEL, "model_revision": REVISION, "dimension": 384,
        "source_recipes": 5898, "queries": 40, "test_queries": 32, "dev_queries": 8,
        "query_ids": [q["query_id"] for q in queries],
        "protocol_sha256": digest(DATA / "PROTOCOL.md"),
        "input_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "legacy_integrity": legacy, "human_annotated": False,
        "new_unseen_queries": False, "tuning": False,
        "metric_stage": PARAMETERS["metric_stage"],
    })
    log("three_route_protocol_frozen", test_queries=32, dev_queries=8,
        historical_code_drift=legacy["historical_code_drift"], protocol=str(DATA / "PROTOCOL.md"))


def verify():
    manifest = json_read(DATA / "manifest.json")
    if manifest["parameters"] != PARAMETERS or manifest["methods"] != METHODS:
        raise ValueError("New frozen configuration changed")
    if digest(DATA / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise ValueError("New frozen protocol changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("New frozen input changed: " + name)
    if legacy_integrity() != manifest["legacy_integrity"]:
        raise ValueError("Historical integrity changed after new freeze")
    return manifest


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["category"], row["method"])].append(row)
        groups[(row["split"], "conditions_combined", row["method"])].append(row)
    return [{"split": split, "category": category, "method": method,
             "queries": len(values),
             "macro_metrics": {key: sum(row["metrics"][key] for row in values) / len(values)
                               for key in values[0]["metrics"]},
             "empty_results": sum(not row["ranking"] for row in values),
             "mean_returned_at_10": sum(min(10, len(row["ranking"])) for row in values) / len(values),
             "mean_candidate_count": sum(row["candidate_count"] for row in values) / len(values)}
            for (split, category, method), values in sorted(groups.items())]


def run():
    deny_network()
    started = time.monotonic()
    manifest = verify()
    import numpy as np

    queries = [q for q in lines(V2 / "queries.jsonl") if q["query_id"] in manifest["query_ids"]]
    vector_rows = list(lines(ROOT / "data/graphrag/full/embeddings.jsonl"))
    pvec = {row["recipe_id"]: row["embedding"] for row in vector_rows}
    if len(pvec) != 5898 or len(vector_rows) != len(pvec):
        raise ValueError("Recipe vector domain incomplete or duplicated")
    ids = sorted(pvec)
    domain = set(ids)
    matrix = np.asarray([pvec[rid] for rid in ids], dtype=np.float32)
    if matrix.shape != (5898, 384) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01):
        raise ValueError("Original vectors invalid or not L2 normalized")
    qvec = {row["query_id"]: row for row in lines(V2 / "query_embeddings.jsonl")}
    docs = list(lines(V2 / "recipe_documents.jsonl"))
    bm25 = BM25Index(docs, k1=PARAMETERS["bm25_k1"], b=PARAMETERS["bm25_b"])
    graph = GraphCandidateIndex(lines(ROOT / "data/test_graph/nodes.jsonl"),
                                lines(ROOT / "data/test_graph/relationships.jsonl"))
    features = {row["recipe_id"]: set(row["production_mention_names"])
                for row in lines(ROOT / "data/evaluation_real_v1/production_flat_metadata.jsonl")}
    if graph.recipe_ids != domain or set(bm25.doc_ids) != domain or set(features) != domain:
        raise ValueError("Graph, BM25, production and vector domains differ")
    # Gold remains outside the retrieval function and is used only after ranking.
    golden = defaultdict(set)
    for row in lines(V2 / "qrels.jsonl"):
        golden[row["query_id"]].add(row["recipe_id"])
    old_D = {row["query_id"]: row for row in lines(V2 / "rankings.jsonl")
             if row["method"] == "D_recipe_graph" and row["variant"] == "with_structured_filter"}
    old_C = {row["query_id"]: row for row in lines(GRAPH_FIRST / "rankings.jsonl")
             if row["method"] == "C_graph_then_e5"}
    results, candidates = [], []
    for query in queries:
        qid = query["query_id"]
        cached = qvec[qid]
        if cached["query"] != query["query"] or cached["model"] != MODEL or cached["model_revision"] != REVISION:
            raise ValueError("Query text/model/revision differs from cache")
        vector = np.asarray(cached["embedding"], dtype=np.float32)
        if vector.shape != (384,):
            raise ValueError("Query vector dimension mismatch")
        visible = {key: query[key] for key in ["query", "ingredients", "excluded_ingredients"]}
        graph_ids = graph_condition_candidates(graph, visible["ingredients"], visible["excluded_ingredients"])
        includes = _filters(visible["ingredients"], "ingredients")
        excludes = _filters(visible["excluded_ingredients"], "excluded_ingredients")
        allowed = {rid for rid in ids if eligible(features[rid], includes, excludes)}
        lexical_query = lexical_text(visible)
        lexical_scores = bm25.scores(lexical_tokens(lexical_query))
        hybrid = graph_first_hybrid(lexical_scores, ids, matrix, vector, graph_ids,
                                    eligible_ids=allowed,
                                    lexical_limit=PARAMETERS["lexical_seed_limit"],
                                    dense_limit=PARAMETERS["dense_seed_limit"], rrf_k=PARAMETERS["rrf_k"])
        if set(hybrid["semantic_scored_ids"]) != graph_ids:
            raise ValueError("Semantic score domain is not exactly graph qualified")
        # Original graph-qualified dots from the new run, no extra full-domain dot.
        single = [{"recipe_id": rid, "cosine_score": score} for rid, score in hybrid["semantic_scores"].items()]
        single.sort(key=lambda row: (-row["cosine_score"], row["recipe_id"]))
        ranked = {"D_recipe_graph_saved": old_D[qid]["ranking"],
                  "C_graph_then_e5": single,
                  "H_graph_first_three_route": hybrid["rankings"]}
        before_recheck = set().union(*map(set, hybrid["seed_ids"].values()))
        candidates.append({"query_id": qid, "split": query["split"], "category": query["category"],
            "lexical_query": lexical_query, "seed_ids": hybrid["seed_ids"],
            "graph_candidate_ids": sorted(graph_ids), "production_eligible_ids": sorted(allowed),
            "semantic_scored_ids": hybrid["semantic_scored_ids"],
            "union_before_condition_recheck": sorted(before_recheck),
            "candidate_pool": hybrid["candidate_pool"], "counts": hybrid["counts"],
            "graph_only_ids": sorted(graph_ids - allowed), "production_only_ids": sorted(allowed - graph_ids),
            "graph_has_rank": False,
            "single_top50_matches_historical_graph_first": [row["recipe_id"] for row in single[:50]] == [row["recipe_id"] for row in old_C[qid]["ranking"]],
            "new_hybrid_top50_matches_saved_old_D": [row["recipe_id"] for row in hybrid["rankings"][:50]] == [row["recipe_id"] for row in old_D[qid]["ranking"]]})
        # Ranking is finished before accessing labels for any method.
        relevant = golden[qid]
        if len(relevant) != query["gold_relevant_count"] or not relevant <= domain:
            raise ValueError("Original label count/domain mismatch")
        for method, full_ranking in ranked.items():
            ranking = full_ranking[:PARAMETERS["saved_ranking_limit"]]
            metrics = metric_values(query["category"], ranking, relevant)
            if method == "D_recipe_graph_saved" and metrics != old_D[qid]["metrics"]:
                raise ValueError("Original D saved metrics differ from recomputation")
            results.append({"query_id": qid, "query": query["query"], "split": query["split"],
                "category": query["category"], "method": method,
                "ranking_origin": "historical_saved_not_reexecuted" if method == "D_recipe_graph_saved" else "actually_recomputed_graph_pool_only",
                "qrels_count": len(relevant), "metrics": metrics,
                "candidate_count": old_D[qid]["candidate_counts"]["pool"] if method == "D_recipe_graph_saved" else len(full_ranking),
                "ranking": ranking, "gold_hits_at_10": len({row["recipe_id"] for row in ranking[:10]} & relevant)})
    write_lines(DATA / "rankings.jsonl", results)
    write_lines(DATA / "candidates.jsonl", candidates)
    report = {
        "status": "completed", "benchmark": manifest["benchmark"],
        "parameters": PARAMETERS, "methods": METHODS, "summary": summarize(results),
        "source_recipes": 5898, "queries": 40, "test_queries": 32, "dev_queries": 8,
        "model": MODEL, "model_revision": REVISION, "dimension": 384,
        "ranking_rows": len(results), "bm25_metadata": bm25.metadata,
        "scope": "offline_frozen_graph_export_recipe_level_exact_scoring_before_LLM_reranking",
        "human_annotated": False, "semantic_accuracy_verified": False,
        "new_unseen_queries": False, "llm_calls": 0, "network_calls": 0,
        "network_socket_connections_denied": True, "dotenv_read": False,
        "credentials_read": False, "model_loads": 0, "model_encoding_calls": 0,
        "query_rewrite": False, "parameters_selected_using_test": False,
        "web_default_changed": False, "historical_saved_old_D_metric_recheck_queries": len(queries),
        "same_graph_and_production_candidate_queries": sum(row["graph_candidate_ids"] == row["production_eligible_ids"] for row in candidates),
        "single_top50_matches_historical_graph_first_queries": sum(row["single_top50_matches_historical_graph_first"] for row in candidates),
        "new_hybrid_top50_matches_saved_old_D_queries": sum(row["new_hybrid_top50_matches_saved_old_D"] for row in candidates),
        "graph_candidate_count_equals_semantic_scored_count_all_queries": all(len(row["graph_candidate_ids"]) == len(row["semantic_scored_ids"]) for row in candidates),
        "legacy_integrity": manifest["legacy_integrity"],
        "manifest_sha256": digest(DATA / "manifest.json"),
        "code_sha256": {name: digest(ROOT / name) for name in INPUTS if name.endswith(".py")},
        "output_sha256": {name: digest(DATA / name) for name in ["rankings.jsonl", "candidates.jsonl"]},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(REPORT, report)
    write_report_md(report)
    log("three_route_evaluation_complete", report=str(REPORT),
        main_test=[row for row in report["summary"] if row["split"] == "test" and row["category"] == "conditions_combined"])


def write_report_md(report):
    text = ["# RecipeGen 三路混合检索：图谱先筛语义路 v1", "",
        "**5898道食谱，原字段字面自动标签；32 test条件回归题，8 dev单列。**", "",
        "BM25全库正分Top50；语义路先从冻结图筛全部合法候选，仅在这些recipe向量上计算E5 float32 dot并取Top50；图谱无序完整候选独立加入并集。条件复核后，对完整pool生成BM25/E5双排名，RRF k=60；没有第三个图谱rank。", "",
        "指标为检索阶段、DeepSeek重排之前。P@10固定分母10，R@10使用原qrels全相关数，按查询宏平均。未验证LLM回答质量，未改变网页默认。", ""]
    for split in ("test", "dev"):
        for category in ("conditions_combined", "multi_ingredient", "ingredient_exclusion"):
            text += [f"## {split} / {category}", "", "| 方法 | 查询数 | Precision@10 | Recall@10 |", "|---|---:|---:|---:|"]
            for method in METHODS:
                row = next(value for value in report["summary"] if value["split"] == split and value["category"] == category and value["method"] == method)
                metrics = row["macro_metrics"]
                text.append(f"| {METHODS[method]} | {row['queries']} | {metrics['precision_at_10']:.2%} | {metrics['recall_at_10']:.2%} |")
            text.append("")
    text += ["## 复核与限制", "",
        f"- 图谱候选与原生产条件核对全集一致：{report['same_graph_and_production_candidate_queries']}/40。每题实际语义计算ID完全等于图合格ID。",
        f"- 本轮单路Top50与历史图谱先筛Top50一致：{report['single_top50_matches_historical_graph_first_queries']}/40；新三路Top50与历史旧D一致：{report['new_hybrid_top50_matches_saved_old_D_queries']}/40。旧D只读取保存结果并重新计算指标，不声称旧代码重跑。",
        "- 图谱完整池已经加入旧/新三路，因此语义先筛可能改变执行方式而不改变混合排名；不能把单路顺序消融的提升移作三路的提升。",
        "- 没有调参、网络、LLM、.env/凭据读取、模型加载、重编码或查询改写。Neo4j节点/边来源为冻结导出，本轮不是实时Cypher/ANN或效率测试。",
        "- qrels基于原始标题/步骤字面扫描。字面标签与图实体提及存在复合词/标题差异，图筛选不是完整食材库存或过敏安全证明。",
        "- 原查询是已经查看过结果的回归条件题，显式给出食材条件；不代表未见自然查询、自由语义理解或问答效果。", "",
        "## 历史源码漂移", "",
        "历史数据、模型、查询、qrels及artifact hash均通过。原v2生产代码已在先前迭代修改，历史manifest保持原样。当前新代码另行冻结：", ""]
    for drift in report["legacy_integrity"]["historical_code_drift"]:
        text.append(f"- `{drift['path']}`：历史 `{drift['historical_sha256']}`；当前 `{drift['current_sha256']}`。")
    text += ["", "协议/输入hash：`data/evaluation_graph_first_hybrid_v1/PROTOCOL.md`、`manifest.json`；逐题候选/种子/pool：`candidates.jsonl`；逐题排名/分数/rank/指标：`rankings.jsonl`。", ""]
    REPORT.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == "__main__":
    main()
