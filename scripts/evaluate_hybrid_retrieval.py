#!/usr/bin/env python3
"""Frozen v2 retrieval experiment: real E5, BM25, graph candidates, step chunks.

No .env, network, or LLM in retrieval evaluation. The API QA experiment is
separate. Old 60 source-derived queries are a regression benchmark; 40 new
quoted-step queries are literal known-passage tests, never human relevance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_real_retrieval import (deny_network, digest, eligible,
    json_read, lines, log, read_sources, verify_freeze, write_json, write_lines)
from recipegen.text_embeddings import MODEL, REVISION, model_info

DATA = ROOT / "data/evaluation_hybrid_v2"
REPORT = ROOT / "reports/hybrid-retrieval-v2.json"
PARAMETERS = {
    "seed": 20261010, "ks": [1, 5, 10], "candidate_limit_per_ranked_route": 50,
    "rrf_k": 60, "chunk_max_tokens_including_prefix": 480,
    "chunk_max_steps": 3, "chunk_recipe_aggregation": "max_cosine",
    "bm25_k1": 1.5, "bm25_b": .75,
    "bm25_idf": "log(1+(N-df+0.5)/(df+0.5))",
    "device": "cpu", "batch_size": 32,
    "filter_order": "each_ranked_route_global_top50_then_same_production_mention_filter_no_refill",
    "graph": "all_matching_recipe_ids_from_frozen_graph_edges_then_same_filter",
    "fusion": "rescore_entire_union_with_BM25_and_cosine_then_two_rank_RRF",
    "e5_query_rewrite": False, "late_step_start_after_token": 530,
    "new_step_queries_per_category": 20, "new_step_dev_per_category": 4,
    "tuning": "none_parameters_fixed_before_results",
}
METHODS = ["A_recipe_e5", "B_bm25", "C_recipe_hybrid", "D_recipe_graph",
           "E_chunk_e5", "F_chunk_hybrid", "G_chunk_graph"]


def normalize(text):
    return " ".join(text.casefold().split())


def normalized_span(text, needle):
    """Map casefold/whitespace-normalized literal matching back to raw chars."""
    chars, origins = [], []
    for word in re.finditer(r"\S+", text):
        if chars:
            chars.append(" ")
            origins.append((word.start(), word.start()))
        for offset, original in enumerate(word.group()):
            for lowered in original.casefold():
                chars.append(lowered)
                origins.append((word.start() + offset, word.start() + offset + 1))
    start = "".join(chars).find(needle)
    if start < 0:
        return None
    return origins[start][0], origins[start + len(needle) - 1][1]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_info(ROOT)["path"], local_files_only=True,
                                        trust_remote_code=False)


def step_query_candidates(raw, corpus, tokenizer):
    """Select source spans before retrieval. Gold scans original text only."""
    original = {r["recipe_id"]: r for r in lines(ROOT / "data/graphrag/full/embeddings.jsonl")}
    rng = random.Random(PARAMETERS["seed"])
    result, qrels, seen_phrases = [], [], set()
    normalized_records = {rid: [(s, normalize(s["text"])) for s in r["steps"]]
                          for rid, r in corpus.items()}
    for category, truncated in [("late_step_passage", True), ("short_step_passage", False)]:
        pool = sorted(rid for rid, r in original.items()
                      if bool(r["truncated"]) == truncated and len(raw[rid]["steps"]) >= 3)
        rng.shuffle(pool)
        selected = 0
        for rid in pool:
            offsets = tokenizer("passage: " + corpus[rid]["text"], truncation=False,
                                return_offsets_mapping=True)["offset_mapping"]
            boundary = PARAMETERS["late_step_start_after_token"]
            cutoff = (offsets[boundary][1] - len("passage: ")
                      if truncated and len(offsets) > boundary and offsets[boundary][1] > offsets[boundary][0]
                      else None)
            if truncated and cutoff is None:
                continue
            cursor = len(raw[rid]["title"]) + 1
            possibilities = []
            for step in corpus[rid]["steps"]:
                words = list(re.finditer(r"\S+", step["text"]))
                if len(words) >= 8 and (not truncated or cursor >= cutoff):
                    end = words[min(16, len(words)) - 1].end()
                    possibilities.append((step, 0, end, cursor))
                cursor += len(step["text"]) + 1
            if not possibilities:
                continue
            step, span_start, span_end, absolute_start = possibilities[0]
            phrase = step["text"][span_start:span_end]
            needle = normalize(phrase)
            if needle in seen_phrases:
                continue
            passage_start = len("passage: ") + absolute_start + span_start
            passage_end = len("passage: ") + absolute_start + span_end
            token_positions = [n for n, (a, b) in enumerate(offsets)
                               if b > a and b > passage_start and a < passage_end]
            if not token_positions or (truncated and min(token_positions) <= boundary):
                raise ValueError("Late-step source span is not beyond the frozen token boundary")
            matches = []
            for other_id, original_steps in normalized_records.items():
                for other_step, normalized_text in original_steps:
                    if needle in normalized_text:
                        match_span = normalized_span(other_step["text"], needle)
                        if match_span is None:
                            raise ValueError("Normalized match cannot map back to raw source")
                        matches.append((other_id, other_step, match_span))
                        break
            if not 1 <= len(matches) <= 50:
                continue
            qid = f"{category}-{selected + 1:02d}"
            query = f'Find recipes whose recorded instructions contain this exact passage: "{phrase}"'
            result.append({"query_id": qid, "split": "dev" if selected < 4 else "test",
                           "category": category, "query": query, "language": "en",
                           "ingredients": [], "excluded_ingredients": [],
                           "target_recipe_id": rid, "gold_relevant_count": len(matches),
                           "human_annotated": False, "query_source": "new_source_quoted_step",
                           "label_semantics": "normalized_literal_original_step_passage",
                           "original_step_quote_copy_bias": True,
                           "natural_user_query_generalization": False,
                           "target_step_id": step["id"], "target_step_order": step["order"],
                           "target_span": {"start": span_start, "end": span_end, "text": phrase},
                           "target_passage_char_span": [passage_start, passage_end],
                           "target_passage_token_span": [min(token_positions), max(token_positions) + 1],
                           "target_recipe_original_embedding_truncated": truncated})
            seen_phrases.add(needle)
            for other_id, other_step, matched_span in matches:
                qrels.append({"query_id": qid, "recipe_id": other_id, "relevance": 1,
                              "label_type": "normalized_literal_original_step_passage",
                              "human_annotated": False, "matched_step_id": other_step["id"],
                              "matched_step_order": other_step["order"],
                              "original_step_text": other_step["text"], "query_quote": phrase,
                              "matched_span": {"start": matched_span[0], "end": matched_span[1],
                                               "text": other_step["text"][matched_span[0]:matched_span[1]]},
                              "recipe_original_embedding_truncated": original[other_id]["truncated"],
                              "corpus_text_sha256": corpus[other_id]["text_sha256"],
                              "source": raw[other_id]["source"]})
            selected += 1
            if selected == PARAMETERS["new_step_queries_per_category"]:
                break
        if selected != PARAMETERS["new_step_queries_per_category"]:
            raise ValueError(f"Insufficient predeclared query sources for {category}: {selected}")
    return result, qrels


def chunk_integrity(chunks, raw):
    expected = {(rid, s["id"]): s for rid, r in raw.items() for s in r["steps"]}
    coverage = defaultdict(list)
    for chunk in chunks:
        if chunk["token_count"] > PARAMETERS["chunk_max_tokens_including_prefix"]:
            raise ValueError("Chunk exceeds actual tokenizer budget")
        for part in chunk["fragments"]:
            key = (chunk["recipe_id"], part["step_id"])
            step = expected[key]
            a, b = part["span_start"], part["span_end"]
            if part["text"] != step["text"][a:b] or part["source_id"] != step["source_id"]:
                raise ValueError("Chunk original span/source changed")
            coverage[key].append((a, b))
    for key, step in expected.items():
        cursor = 0
        for a, b in sorted(coverage[key]):
            if a > cursor:
                raise ValueError("Original step content lost during chunking")
            cursor = max(cursor, b)
        if cursor != len(step["text"]):
            raise ValueError("Original step coverage incomplete")
    return {"original_steps": len(expected), "full_original_step_span_coverage": True,
            "source_bindings_valid": True, "chunks": len(chunks),
            "maximum_token_count": max(c["token_count"] for c in chunks)}


def prepare():
    if (DATA / "manifest.json").exists():
        raise FileExistsError("v2 already frozen; do not regenerate based on results")
    deny_network()
    baseline = verify_freeze()
    raw, corpus = read_sources()
    from recipegen.experimental_retrieval import build_recipe_documents, build_step_chunks
    tokenizer = load_tokenizer()
    documents = build_recipe_documents(list(raw.values()))
    log("prepare_chunks", recipes=len(raw))
    chunks = build_step_chunks(list(raw.values()), tokenizer, max_tokens=480, max_steps=3)
    integrity = chunk_integrity(chunks, corpus)
    queries = list(lines(ROOT / "data/evaluation_real_v1/queries.jsonl"))
    for query in queries:
        query["query_source"] = "previously_viewed_frozen_regression_v1"
    qrels = list(lines(ROOT / "data/evaluation_real_v1/qrels.jsonl"))
    extra, extra_qrels = step_query_candidates(raw, corpus, tokenizer)
    queries.extend(extra)
    qrels.extend(extra_qrels)
    DATA.mkdir(parents=True, exist_ok=True)
    write_lines(DATA / "recipe_documents.jsonl", documents)
    write_lines(DATA / "chunks.jsonl", chunks)
    write_lines(DATA / "queries.jsonl", queries)
    write_lines(DATA / "qrels.jsonl", qrels)
    write_json(DATA / "chunk-integrity.json", integrity)
    files = ["recipe_documents.jsonl", "chunks.jsonl", "queries.jsonl", "qrels.jsonl", "chunk-integrity.json"]
    inputs = dict(baseline["input_sha256"])
    inputs["data/evaluation_real_v1/manifest.json"] = digest(ROOT / "data/evaluation_real_v1/manifest.json")
    for path in ["scripts/evaluate_hybrid_retrieval.py", "recipegen/experimental_retrieval.py",
                 "data/evaluation_real_v1/query_embeddings.jsonl", "data/evaluation_real_v1/production_flat_metadata.jsonl"]:
        inputs[path] = digest(ROOT / path)
    manifest = {"status": "frozen_before_retrieval", "benchmark": "hybrid-retrieval-v2",
                "parameters": PARAMETERS, "input_sha256": inputs,
                "artifact_sha256": {p: digest(DATA / p) for p in files},
                "model": MODEL, "model_revision": REVISION, "source_recipes": len(raw),
                "source_steps": sum(len(r["steps"]) for r in raw.values()),
                "queries": len(queries), "dev_queries": sum(q["split"] == "dev" for q in queries),
                "test_queries": sum(q["split"] == "test" for q in queries),
                "old_regression_queries": 60, "new_source_quote_queries": 40,
                "human_annotated": False, "semantic_accuracy_verified": False,
                "query_holdout_only_not_dataset_train_test": True,
                "raw_literal_labels_not_derived_from_graph_or_retriever": True,
                "dataset_revision": baseline["dataset_revision"],
                "structured_entities_provided_by_benchmark_not_LLM_parsed": True}
    write_json(DATA / "manifest.json", manifest)
    (DATA / "PROTOCOL.md").write_text(protocol(), encoding="utf-8")
    log("v2_frozen", queries=len(queries), **integrity)


def protocol():
    return """# RecipeGen 检索对比实验 v2

## 数据与标签
使用完整 5898 条食谱与 44802 个步骤。原60题为已经查看结果的v1固定回归基准，不称新的未见测试：20原始唯一标题、20多食材共同提及、20排除提及。追加40条新步骤原文片段查询：20来自原E5编码截断后的步骤（步骤开始在完整输入token530之后），20来自未截断食谱；每类4 dev/16 test。总100题20 dev/80 test。无参数搜索；所有方法参数在查看v2结果前固定。原数据集split=test与查询dev/test没有训练测试关系。

所有标签自动生成，不是人工语义相关性。原条件标签复用独立字面词表全库穷举；新步骤标签独立扫描全库原始步骤，以忽略大小写/空白后的完整摘录匹配构造qrels，保存原文、步骤和source SHA。摘录直接来自原步骤，存在词面复制偏易；仅衡量原文片段找回，不证明自然语言问答或语义泛化。三种标签类别单列，不混成总Recall。

## 检索实现与公平性
E5固定为multilingual-e5-small同revision、384维、CPU实际编码、query/passsage前缀与L2归一化。原60题复用已有query向量，新40题重新编码原问题；没有API语言改写。BM25为项目实现正IDF Okapi：idf=log(1+(N-df+.5)/(df+.5))，k1=1.5,b=.75，统一英文分词。条件问题只从生产词典/显式查询实体扩展英文关键词，剔除负向项；不读取gold词表。步骤问题的检索词从问题中引号摘录提取，标题题使用问题原文。

所有ranked通道先全库Top50，再使用同一冻结production_flat_metadata与生产catalog._filters检查显式食材提及，不补搜。没有条件时允许全库。图候选集合来自冻结原始Ingredient关系，不用gold或标注ID；图无序集合并入候选，整池补算BM25与cosine各自排名后执行两路RRF(k=60)。zero BM25不获取词面排名；图条件路不伪造第三路排名，记录独有候选和最终命中变化。不声称计算预算相同，图路可穷举更多候选。max cosine把步骤块归并到Recipe；每条食谱最终只出现一次。gold仍可包含被生产过滤遗漏的复合词/标题提及，单列过滤可达上限，不根据结果改标签。

A整食谱E5，B BM25，C整食谱E5+BM25，D C+图谱独立候选；E步骤E5，F步骤E5+同一BM25，G F+图谱候选。保持BM25文档、查询、过滤规则、Top50/k/融合参数相同，C/F和D/G用于比较检索文本粒度。先保留原索引，派生步骤块最多3相邻步骤、总480 tokenizer tokens含passage/title前缀；超长单步按原文字符跨度细分，source/step ID及全文覆盖校验。2条无步骤记录保留真实标题单元。

## 指标与范围
标题题仅报告Target Hit@k/MRR，不当作完整语义相关性Recall。条件题和步骤字面相关集：Precision@k=相关返回数/k（不足k仍固定分母），Recall@k=相关返回数/完整qrels数量；另报Precision-returned@k、实际返回数、空结果率、MRR@10、过滤可达率。k=1/5/10；按dev/test、类别和两种过滤设置分别宏平均。空结果P/R/MRR为0。检索过程只获取问题/显式条件，标签ID、target ID、gold计数只在指标函数使用。

本实验为全量原数据的离线精确cosine与冻结图导出遍历，不是实时Neo4j ANN/Cypher端到端；不读.env，不调用LLM/API。DeepSeek API重排/问答另存独立真实调用记录，引用有效率不等同答案支持率。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-mm/bin/python scripts/evaluate_hybrid_retrieval.py --prepare
PYTHONDONTWRITEBYTECODE=1 .venv-mm/bin/python scripts/evaluate_hybrid_retrieval.py --encode
PYTHONDONTWRITEBYTECODE=1 .venv-mm/bin/python scripts/evaluate_hybrid_retrieval.py --run
```
"""


def verify():
    manifest = json_read(DATA / "manifest.json")
    if manifest["parameters"] != PARAMETERS:
        raise ValueError("Frozen v2 parameters changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    for name, expected in manifest["artifact_sha256"].items():
        if digest(DATA / name) != expected:
            raise ValueError("Frozen v2 artifact changed: " + name)
    return manifest


def encode_vectors():
    deny_network()
    manifest = verify()
    import numpy as np
    queries, chunks = list(lines(DATA / "queries.jsonl")), list(lines(DATA / "chunks.jsonl"))
    from scripts.local_text_embedding import load_encoder, encode
    tokenizer, model = load_encoder()
    old = {r["query_id"]: r for r in lines(ROOT / "data/evaluation_real_v1/query_embeddings.jsonl")}
    qvectors = []
    new = [q for q in queries if q["query_id"] not in old]
    new_values = encode([q["query"] for q in new], tokenizer, model, prefix="query: ")
    encoded = dict(zip([q["query_id"] for q in new], new_values))
    for q in queries:
        if q["query_id"] in old:
            if old[q["query_id"]]["query"] != q["query"]:
                raise ValueError("Old query cache mismatch")
            vector = old[q["query_id"]]["embedding"]
        else:
            vector = encoded[q["query_id"]]
        qvectors.append({"query_id": q["query_id"], "query": q["query"], "embedding": vector,
                         "model": MODEL, "model_revision": REVISION})
    write_lines(DATA / "query_embeddings.jsonl", qvectors)
    target = DATA / "chunk_vectors.npy"
    if target.exists():
        raise FileExistsError("Chunk vectors exist; use --run or validate cached vector manifest")
    matrix = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(chunks), 384))
    tick = time.monotonic()
    batch = PARAMETERS["batch_size"]
    for start in range(0, len(chunks), batch):
        selected = chunks[start:start + batch]
        values = encode([c["text"] for c in selected], tokenizer, model, prefix="passage: ")
        matrix[start:start + len(selected)] = values
        if start % (batch * 32) == 0 or start + batch >= len(chunks):
            matrix.flush()
            log("chunk_encoding", completed=min(start + batch, len(chunks)), total=len(chunks),
                elapsed_seconds=round(time.monotonic() - tick, 2))
    matrix.flush()
    write_json(DATA / "vector-manifest.json", {"status": "completed", "model": MODEL,
        "model_revision": REVISION, "queries": len(queries), "chunks": len(chunks),
        "dimension": 384, "device": "cpu", "model_loads": 1,
        "reused_query_vectors": len(old), "new_query_vectors": len(new),
        "encoding_seconds": time.monotonic() - tick,
        "benchmark_manifest_sha256": digest(DATA / "manifest.json"),
        "artifact_sha256": {p: digest(DATA / p) for p in ["query_embeddings.jsonl", "chunk_vectors.npy"]}})
    log("vectors_complete", chunks=len(chunks))


def lexical_text(query):
    # Both branches derive terms only from visible query/explicit user conditions.
    if query["ingredients"]:
        from recipegen.catalog import _filters
        filters = _filters(query["ingredients"], "ingredients")
        return " ".join(dict.fromkeys(name for item in filters for name in item["names"]
                                      if re.search(r"[a-zA-Z]", name)))
    quoted = re.search(r'contain this exact passage: "(.*)"$', query["query"], re.S)
    return quoted.group(1) if quoted else query["query"]


def metric_values(category, ranking, relevant):
    ids = [r["recipe_id"] for r in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError("Recipe ranking contains duplicates")
    first = next((n for n, rid in enumerate(ids[:10], 1) if rid in relevant), None)
    out = {"target_mrr_at_10" if category == "known_item" else "mrr_at_10": 1 / first if first else 0.0}
    for k in PARAMETERS["ks"]:
        hits, count = len(set(ids[:k]) & relevant), len(ids[:k])
        if category == "known_item":
            out[f"target_hit_at_{k}"] = float(bool(hits))
        else:
            out.update({f"precision_at_{k}": hits / k, f"recall_at_{k}": hits / len(relevant),
                        f"precision_returned_at_{k}": hits / count if count else 0.0,
                        f"hit_at_{k}": float(bool(hits))})
    return out


def run():
    deny_network()
    tick = time.monotonic()
    manifest = verify()
    vm = json_read(DATA / "vector-manifest.json")
    if vm["benchmark_manifest_sha256"] != digest(DATA / "manifest.json"):
        raise ValueError("Vector build benchmark differs")
    for p, expected in vm["artifact_sha256"].items():
        if digest(DATA / p) != expected:
            raise ValueError("Vector artifact changed")
    import numpy as np
    from recipegen.experimental_retrieval import BM25Index, GraphCandidateIndex, pool_rrf, lexical_tokens as tokenize
    from recipegen.catalog import _filters
    raw, corpus = read_sources()
    ids = sorted(raw)
    index_of = {rid: n for n, rid in enumerate(ids)}
    queries = list(lines(DATA / "queries.jsonl"))
    chunks = list(lines(DATA / "chunks.jsonl"))
    docs = list(lines(DATA / "recipe_documents.jsonl"))
    bm25 = BM25Index(docs, k1=1.5, b=.75)
    graph = GraphCandidateIndex(lines(ROOT / "data/test_graph/nodes.jsonl"),
                                lines(ROOT / "data/test_graph/relationships.jsonl"))
    features = {r["recipe_id"]: set(r["production_mention_names"])
                for r in lines(ROOT / "data/evaluation_real_v1/production_flat_metadata.jsonl")}
    golden = defaultdict(set)
    for r in lines(DATA / "qrels.jsonl"):
        golden[r["query_id"]].add(r["recipe_id"])
    pvec = {r["recipe_id"]: r["embedding"] for r in lines(ROOT / "data/graphrag/full/embeddings.jsonl")}
    qvec = {r["query_id"]: r["embedding"] for r in lines(DATA / "query_embeddings.jsonl")}
    pmatrix = np.asarray([pvec[rid] for rid in ids], dtype=np.float32)
    cmatrix = np.load(DATA / "chunk_vectors.npy", mmap_mode="r")
    if cmatrix.shape != (len(chunks), 384) or not np.isfinite(cmatrix).all():
        raise ValueError("Invalid chunk vector shape or values")
    if not np.allclose(np.linalg.norm(cmatrix, axis=1), 1, atol=.01):
        raise ValueError("Chunk vectors not L2 normalized")
    chunk_owner = np.asarray([index_of[c["recipe_id"]] for c in chunks], dtype=np.int64)
    rows = []
    for q in queries:
        vector = np.asarray(qvec[q["query_id"]], dtype=np.float32)
        recipe_scores = dict(zip(ids, map(float, pmatrix @ vector)))
        cscores = cmatrix @ vector
        aggregate = np.full(len(ids), -1.0, dtype=np.float32)
        np.maximum.at(aggregate, chunk_owner, cscores)
        chunk_scores = dict(zip(ids, map(float, aggregate)))
        bscore = bm25.scores(tokenize(lexical_text(q)))
        for variant in ["without_structured_filter", "with_structured_filter"]:
            includes = _filters(q["ingredients"], "ingredients") if variant == "with_structured_filter" else []
            excludes = _filters(q["excluded_ingredients"], "excluded_ingredients") if variant == "with_structured_filter" else []
            allowed = {rid for rid in ids if eligible(features[rid], includes, excludes)}
            gid = graph.candidates(ingredients=q["ingredients"], excluded=q["excluded_ingredients"])
            zero = {rid: 0.0 for rid in ids}
            configs = {
                "A_recipe_e5": (zero, recipe_scores, set()),
                "B_bm25": (bscore, {}, set()),
                "C_recipe_hybrid": (bscore, recipe_scores, set()),
                "D_recipe_graph": (bscore, recipe_scores, gid),
                "E_chunk_e5": (zero, chunk_scores, set()),
                "F_chunk_hybrid": (bscore, chunk_scores, set()),
                "G_chunk_graph": (bscore, chunk_scores, gid),
            }
            for method in METHODS:
                bm, dense, candidates = configs[method]
                result = pool_rrf(bm, dense, candidates, eligible_ids=allowed,
                                  lexical_limit=50, dense_limit=50, rrf_k=60)
                ranking = result["rankings"][:50]
                relevant = golden[q["query_id"]]
                if len(relevant) != q["gold_relevant_count"]:
                    raise ValueError("Qrels count drift")
                rows.append({"query_id": q["query_id"], "query": q["query"], "category": q["category"],
                    "split": q["split"], "variant": variant, "method": method,
                    "label_scope": q["label_semantics"], "qrels_count": len(relevant),
                    "metrics": metric_values(q["category"], ranking, relevant), "ranking": ranking,
                    "candidate_counts": result["counts"], "seed_ids": result["seed_ids"],
                    "returned_candidates": len(ranking), "fewer_than_10": len(ranking) < 10,
                    "empty_result": not ranking, "eligible_records": len(allowed),
                    "gold_filter_reachable_count": len(relevant & allowed),
                    "gold_filter_reachable_fraction": len(relevant & allowed) / len(relevant),
                    "missed_gold_at_10": sorted(relevant - {r["recipe_id"] for r in ranking[:10]})})
        if len(rows) % 140 == 0:
            log("retrieval", completed=len(rows)//14, total=len(queries))
    write_lines(DATA / "rankings.jsonl", rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["category"], r["variant"], r["method"])].append(r)
        if r["category"] in {"multi_ingredient", "ingredient_exclusion"}:
            grouped[(r["split"], "conditions_combined", r["variant"], r["method"])].append(r)
        if r["category"] in {"late_step_passage", "short_step_passage"}:
            grouped[(r["split"], "step_passages_combined", r["variant"], r["method"])].append(r)
    summaries = []
    for (split, category, variant, method), values in sorted(grouped.items()):
        summaries.append({"split": split, "category": category, "variant": variant, "method": method,
            "queries": len(values), "macro_metrics": {k: sum(v["metrics"][k] for v in values)/len(values)
                                                      for k in values[0]["metrics"]},
            "mean_relevant_count": sum(v["qrels_count"] for v in values)/len(values),
            "mean_returned_at_10": sum(min(v["returned_candidates"], 10) for v in values)/len(values),
            "empty_results": sum(v["empty_result"] for v in values),
            "queries_returning_fewer_than_10": sum(v["fewer_than_10"] for v in values),
            "mean_gold_filter_reachable_fraction": sum(v["gold_filter_reachable_fraction"] for v in values)/len(values),
            "mean_graph_only_added": sum(v["candidate_counts"].get("graph_only_added", 0) for v in values)/len(values)})
    report = {"status": "completed", "benchmark": "hybrid-retrieval-v2", "scope": "offline_exact_full_source_not_live_Neo4j_ANN",
        "queries": len(queries), "ranking_rows": len(rows), "source_recipes": len(raw),
        "source_steps": manifest["source_steps"], "chunks": len(chunks), "parameters": PARAMETERS,
        "human_annotated": False, "semantic_accuracy_verified": False, "llm_calls": 0,
        "network_calls": 0, "dotenv_read": False, "old_regression_queries": 60,
        "new_source_quote_queries": 40, "encoding": vm, "chunk_integrity": json_read(DATA / "chunk-integrity.json"),
        "manifest_sha256": digest(DATA / "manifest.json"), "rankings_sha256": digest(DATA / "rankings.jsonl"),
        "summary": summaries, "elapsed_seconds": time.monotonic() - tick}
    write_json(REPORT, report)
    write_report_md(report)
    log("report_complete", rows=len(rows), report=str(REPORT))


def write_report_md(report):
    content = ["# RecipeGen 检索实验 v2", "", "**自动来源字段标注；不是人工语义评测。**", "",
        f"语料：{report['source_recipes']} 道食谱、{report['source_steps']} 个步骤、{report['chunks']} 个派生步骤块。",
        "100题：60题已查看的固定回归基准，40题新原文步骤摘录找回；总20 dev/80 test，无调参。",
        "精确cosine离线实验，未运行实时Neo4j ANN；LLM问答另存报告。", "",
        "P@k固定分母k；R@k完整qrels分母；宏平均。标题题单列Target Hit/MRR。", ""]
    for category, variant in [("conditions_combined", "with_structured_filter"),
                              ("late_step_passage", "without_structured_filter"),
                              ("short_step_passage", "without_structured_filter")]:
        selected = [s for s in report["summary"] if s["split"] == "test" and s["category"] == category and s["variant"] == variant]
        content += [f"## {category} / {variant}", "", "| 方法 | P@5 | R@5 | P@10 | R@10 | MRR@10 | 返回不足10的题数 |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
        for s in selected:
            m = s["macro_metrics"]
            content.append(f"| {s['method']} | {m['precision_at_5']:.2%} | {m['recall_at_5']:.2%} | {m['precision_at_10']:.2%} | {m['recall_at_10']:.2%} | {m['mrr_at_10']:.4f} | {s['queries_returning_fewer_than_10']} |")
        content += [""]
    content += ["## 限制", "", "- 条件标签按原文字面提及，复合词与图实体语义可能不同；排除不代表过敏安全或完整配方。",
        "- 步骤题直接引用原文，偏易，只测片段找回；未标注自然问答相关性或因果/顺序理解。",
        "- 图集合可扩大候选预算，所有方法相同过滤但计算量不同；原图谱不重建。",
        "- 结构化食材条件由基准显式提供；没有验证API问题解析。",
        "- 原标题唯一目标题未混入条件Recall；60题旧test已查看。",
        "", "完整dev/test、各类别、无过滤/统一过滤的全部指标见同名JSON；逐题排名与源标签见 data/evaluation_hybrid_v2。", ""]
    REPORT.with_suffix(".md").write_text("\n".join(content), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--encode", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.encode: encode_vectors()
    else: run()


if __name__ == "__main__":
    main()
