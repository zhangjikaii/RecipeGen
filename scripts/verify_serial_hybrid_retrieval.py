#!/usr/bin/env python3
"""Freeze then evaluate the real read-only Neo4j hybrid path, before LLM reranking.

The original condition queries, E5 query vectors and literal automatic qrels
remain unchanged. This script never calls an LLM or prints connection settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.catalog import RecipeCatalog
from recipegen.config import Settings
from recipegen.graphrag_retrieval import RecipeGraphRAGRetriever
from recipegen.text_embeddings import MODEL, REVISION

DATA = ROOT / "data/evaluation_serial_hybrid_v1"
OLD = ROOT / "data/evaluation_hybrid_v2"
REPORT = ROOT / "reports/serial-hybrid-retrieval-v1.json"
INPUTS = [
    "data/evaluation_hybrid_v2/queries.jsonl",
    "data/evaluation_hybrid_v2/qrels.jsonl",
    "data/evaluation_hybrid_v2/query_embeddings.jsonl",
    "data/graphrag/index-manifest.json",
    "data/graphrag/full/corpus.jsonl",
    "recipegen/graphrag_retrieval.py",
    "recipegen/production_hybrid.py",
    "recipegen/experimental_retrieval.py",
    "recipegen/graph_first_hybrid.py",
    "recipegen/catalog.py", "recipegen/config.py", "recipegen/text_embeddings.py",
    "scripts/verify_serial_hybrid_retrieval.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def selected_queries():
    rows = [q for q in lines(OLD / "queries.jsonl")
            if q["category"] in {"multi_ingredient", "ingredient_exclusion"}]
    assert len(rows) == 40 and sum(q["split"] == "test" for q in rows) == 32
    return rows


def prepare():
    assert not (DATA / "manifest.json").exists(), "This protocol has already been frozen"
    catalog = RecipeCatalog(settings=Settings.from_env())
    try:
        public = catalog.status()
    finally:
        catalog.close()
    assert public["backend"] == "neo4j" and public["graph"]["recipes"] == 5898
    assert public["graph"]["steps"] == 44802
    protocol = """# Production serial hybrid retrieval regression v1

Protocol and hashes are frozen before new production rankings. Reuse the original
40 condition regression queries: 8 dev reported separately, 32 test (16 multi-
ingredient and 16 ingredient-exclusion). These queries were already viewed;
this is not an unseen natural-language or human-semantic evaluation. Qrels are
unchanged automatic literal matches against original recipe fields.

Production contract: fixed parameterized read-only Cypher returns the complete
condition-qualified Recipe/Text set C; no candidate limit before E5 scoring.
BM25 scores the original title/all-step recipe corpus, global positive Top50 A.
E5 uses the original frozen query vectors and only the live C matrix, Top50 B.
C is an unordered full candidate route, not a third relevance ranking. Union
then hard-condition recheck; actual BM25 positive rank and E5 rank over the
complete legal pool form two-rank RRF k60, with stable recipe-ID ties. Only
after fusion does production read back ordered Step, Ingredient and Source
evidence for selected recipes. Search returns Top10; API reranking/QA is not
called by this evaluation. Production lexical rewriting is fixed in the hashed
implementation and uses only visible query/condition aliases, never qrels.

No model re-encoding, new labels, parameter selection, or graph mutation.
Query filters are explicit benchmark fields (infer_query_filters=False); the
natural-language DeepSeek routing stage is verified separately. P@10 uses
hits/10, including results shorter than ten; R@10 uses hits/all original qrels.
Macro average per query, dev and test separate. Gold is read only after all
production rankings have finished. Live Neo4j build/counts are checked before
and after execution. Scope is retrieval before DeepSeek reranking and QA.
"""
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "PROTOCOL.md").write_text(protocol)
    write(DATA / "manifest.json", {
        "status": "frozen_before_production_rankings", "benchmark": "serial-hybrid-retrieval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_ids": [q["query_id"] for q in selected_queries()],
        "queries": 40, "test_queries": 32, "dev_queries": 8,
        "model": MODEL, "model_revision": REVISION, "embedding_dimension": 384,
        "metric_stage": "production_retrieval_before_DeepSeek_reranking",
        "human_annotated": False, "new_unseen_queries": False, "llm_calls": 0,
        "graph_build": public["build"], "graph_counts": public["graph"],
        "input_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "protocol_sha256": digest(DATA / "PROTOCOL.md"),
    })
    print(json.dumps({"status": "protocol_frozen", "queries": 40, "test_queries": 32}))


def run():
    start = time.monotonic()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        assert digest(ROOT / name) == expected, "Frozen input changed: " + name
    assert digest(DATA / "PROTOCOL.md") == manifest["protocol_sha256"]
    query_rows = selected_queries()
    vectors = {q["query_id"]: q for q in lines(OLD / "query_embeddings.jsonl")}
    catalog = RecipeCatalog(settings=Settings.from_env())
    checks, rows = {}, []

    def check(name, value):
        checks[name] = bool(value)
        assert value, name

    try:
        state = catalog.status()
        check("frozen_live_build", state["build"] == manifest["graph_build"])
        check("frozen_live_counts", state["graph"] == manifest["graph_counts"])
        retriever = RecipeGraphRAGRetriever(catalog)
        for number, query in enumerate(query_rows, 1):
            qid = query["query_id"]
            vector = vectors[qid]
            check(qid + "_original_query_vector", vector["query"] == query["query"]
                  and vector["model"] == MODEL and vector["model_revision"] == REVISION)
            result = retriever.search(query["query"], ingredients=query["ingredients"],
                excluded_ingredients=query["excluded_ingredients"], limit=10, mode="hybrid",
                query_vector=vector["embedding"], embedding_model=MODEL,
                embedding_revision=REVISION, infer_query_filters=False)
            metadata = result["retrieval"]
            identifiers = [recipe["id"] for recipe in result["recipes"]]
            check(qid + "_actual_hybrid", metadata["effective_mode"] == "hybrid"
                  and metadata.get("rrf_k") == 60 and metadata["status"] == "ok")
            check(qid + "_complete_graph_scoring", metadata["graph_candidate_count"] == metadata["scored_recipe_count"])
            check(qid + "_unique_top10", len(identifiers) <= 10 and len(identifiers) == len(set(identifiers)))
            trace = result.get("trace", [])
            stages = [item["stage"] for item in trace]
            check(qid + "_fusion_then_evidence", "rrf_fusion" in stages and
                  any(stage in stages and stages.index("rrf_fusion") < stages.index(stage)
                      for stage in ["graph_evidence_expansion", "graph_evidence_readback"]))
            rows.append({"query_id": qid, "query": query["query"], "split": query["split"],
                         "category": query["category"], "returned_ids": identifiers,
                         "retrieval": metadata, "trace": trace})
            if number % 10 == 0:
                print(json.dumps({"event": "production_hybrid_retrieval", "completed": number, "total": 40}), flush=True)
        after = catalog.status()
        check("live_build_unchanged", after["build"] == manifest["graph_build"])
        check("live_counts_unchanged", after["graph"] == manifest["graph_counts"])
    finally:
        catalog.close()
    # Relevance labels are never supplied to the production retriever.
    gold = defaultdict(set)
    for qrel in lines(OLD / "qrels.jsonl"):
        gold[qrel["query_id"]].add(qrel["recipe_id"])
    queries_by_id = {q["query_id"]: q for q in query_rows}
    for row in rows:
        relevant = gold[row["query_id"]]
        assert len(relevant) == queries_by_id[row["query_id"]]["gold_relevant_count"]
        hits = len(set(row["returned_ids"]) & relevant)
        row.update(qrels_count=len(relevant), hits_at_10=hits,
                   precision_at_10=hits/10, recall_at_10=hits/len(relevant))
    summaries = []
    for split in ["dev", "test"]:
        group = [r for r in rows if r["split"] == split]
        summaries.append({"split": split, "queries": len(group),
            "precision_at_10": sum(r["precision_at_10"] for r in group)/len(group),
            "recall_at_10": sum(r["recall_at_10"] for r in group)/len(group)})
    ranking_path = DATA / "rankings.jsonl"
    ranking_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"status": "passed", "benchmark": manifest["benchmark"], "scope": manifest["metric_stage"],
        "completed_at": datetime.now(timezone.utc).isoformat(), "summary": summaries,
        "checks": checks, "queries": 40, "human_annotated": False, "new_unseen_queries": False,
        "llm_calls": 0, "model_encoding_calls": 0, "graph_build": manifest["graph_build"],
        "manifest_sha256": digest(DATA / "manifest.json"), "rankings_sha256": digest(ranking_path),
        "elapsed_seconds": round(time.monotonic()-start, 3)}
    write(REPORT, report)
    main_test = next(row for row in summaries if row["split"] == "test")
    (ROOT / "reports/serial-hybrid-retrieval-v1.md").write_text(
        "# 生产三路检索回归 v1\n\n三路候选合并、条件核验、双排名 RRF，之后才回读图谱证据。\n\n"
        f"32 条原字段自动标注条件回归 test 查询：宏 Precision@10 **{main_test['precision_at_10']*100:.2f}%**，"
        f"Recall@10 **{main_test['recall_at_10']*100:.2f}%**。\n\n"
        "本轮实际调用生产检索器与真实 Neo4j；使用原 E5 查询向量，不重编码。"
        "8 条 dev 单列。指标在 DeepSeek 重排前，不是人工语义或问答准确率。"
        "查询已查看过，不作为未见查询泛化证明。协议、输入哈希及逐题候选与排名见 "
        "`data/evaluation_serial_hybrid_v1/`；真实 API 完整流程单独验证。\n")
    print(json.dumps({"status": report["status"], "test": main_test, "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare == arguments.run:
        parser.error("Choose exactly one of --prepare or --run")
    prepare() if arguments.prepare else run()
