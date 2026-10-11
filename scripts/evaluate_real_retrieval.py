#!/usr/bin/env python3
"""Frozen, source-derived weak-label retrieval evaluation. Offline; no .env/API.

Prepare once, then run with the existing .venv-mm Python. Gold labels scan raw
records with a separate literal lexicon, never graph Ingredient relationships.
This is exact-cosine replay of production retrieval rules, not live Neo4j ANN.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import socket
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/evaluation_real_v1"
REPORT = ROOT / "reports/real-retrieval-v1.json"
SEED = 20261010
PARAMETERS = {
    "seed": SEED, "queries_per_category": 20, "dev_per_category": 4,
    "ks": [1, 5, 10], "dense_candidate_limit": 50,
    "keyword_candidate_limit": 20, "final_candidate_limit": 50,
    "rrf_k": 60, "query_batch_size": 12, "device": "cpu",
    "gold_min_relevant": 3, "gold_max_relevant": 150,
    "dense_search": "exact_cosine_float32", "tie_break": "recipe_id_ascending",
    "keyword_search": "catalog._terms + SEARCH_QUERY literal ranking replay",
    "structured_filter": "production Step ingredient mentions for all methods",
    "semantic_filter_order": "full_corpus_top50_then_mention_filter_no_refill",
    "keyword_filter_order": "all_matching_records_then_mention_filter_then_top20",
    "graph_changes_recipe_ranking": False,
}

# Deliberately independent, inspectable *literal* weak labels. No production
# extractor/lexicon is used to construct these rules or the gold qrels.
# A phrase inside a longer compound (rice flour, garlic powder) is a raw mention;
# this can differ from production canonical-entity extraction, and is reported.
GOLD_LEXICON = [
    ("tomato", "番茄", ["tomato", "tomatoes"]),
    ("potato", "马铃薯", ["potato", "potatoes"]),
    ("carrot", "胡萝卜", ["carrot", "carrots"]),
    ("onion", "洋葱", ["onion", "onions"]),
    ("garlic", "大蒜", ["garlic"]),
    ("ginger", "姜", ["ginger"]),
    ("green onion", "葱", ["green onion", "green onions", "scallion", "scallions", "spring onion", "spring onions"]),
    ("broccoli", "西兰花", ["broccoli"]),
    ("cabbage", "卷心菜", ["cabbage", "cabbages"]),
    ("spinach", "菠菜", ["spinach"]),
    ("cucumber", "黄瓜", ["cucumber", "cucumbers"]),
    ("eggplant", "茄子", ["eggplant", "eggplants", "aubergine", "aubergines"]),
    ("mushroom", "蘑菇", ["mushroom", "mushrooms"]),
    ("corn", "玉米", ["corn", "sweetcorn"]),
    ("pea", "豌豆", ["pea", "peas"]),
    ("green bean", "四季豆", ["green bean", "green beans", "string bean", "string beans"]),
    ("lemon", "柠檬", ["lemon", "lemons"]),
    ("apple", "苹果", ["apple", "apples"]),
    ("banana", "香蕉", ["banana", "bananas"]),
    ("rice", "大米或米饭", ["rice"]),
    ("pasta", "意大利面", ["pasta"]),
    ("flour", "面粉", ["flour"]),
    ("egg", "鸡蛋", ["egg", "eggs"]),
    ("milk", "牛奶", ["milk"]),
    ("butter", "黄油", ["butter"]),
    ("cheese", "奶酪", ["cheese"]),
    ("chicken", "鸡肉", ["chicken", "chickens"]),
    ("pork", "猪肉", ["pork"]),
    ("beef", "牛肉", ["beef"]),
    ("shrimp", "虾", ["shrimp", "shrimps"]),
    ("fish", "鱼", ["fish"]),
    ("tofu", "豆腐", ["tofu"]),
]
GOLD_RULES = {name: re.compile(r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, aliases)) + r")(?![a-z0-9_])", re.I)
              for name, _, aliases in GOLD_LEXICON}
ZH = {name: zh for name, zh, _ in GOLD_LEXICON}


def log(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_lines(path, values):
    with Path(path).open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def source_evidence(record, name):
    fields = [("title", None, record["title"])] + [("step", step["order"], step["text"]) for step in record["steps"]]
    for field, order, text in fields:
        match = GOLD_RULES[name].search(text)
        if match:
            src = record["source"]
            role = "goal" if field == "title" else "steps"
            return {"canonical_term": name, "field": field, "step_order": order,
                    "span_start": match.start(), "span_end": match.end(), "matched_text": match.group(),
                    "original_text": text, "archive": src["archive"], "member": src[role + "_member"],
                    "source_text_sha256": src[role + "_sha256"], "dataset_revision": src["revision"]}
    return None


def read_sources():
    raw = {row["id"]: row for row in lines(ROOT / "data/test_graph/records.jsonl")}
    corpus = {row["recipe_id"]: row for row in lines(ROOT / "data/graphrag/full/corpus.jsonl")}
    if len(raw) != 5898 or set(raw) != set(corpus):
        raise ValueError("Source and embedding corpus must have the frozen 5898 IDs")
    for rid, record in raw.items():
        item = corpus[rid]
        source_text = record["title"] + "\n" + "\n".join(step["text"] for step in record["steps"])
        if (item["text"] != source_text or item["title"] != record["title"]
                or [s["text"] for s in item["steps"]] != [s["text"] for s in record["steps"]]
                or hashlib.sha256(source_text.encode()).hexdigest() != item["text_sha256"]):
            raise ValueError("Original source and corpus text/hash disagree: " + rid)
    return raw, corpus


def input_hashes():
    relative = [
        "data/test_graph/records.jsonl", "data/test_graph/source/test.texts.jsonl",
        "data/test_graph/source/test-video.texts.jsonl", "data/test_graph/nodes.jsonl",
        "data/test_graph/relationships.jsonl", "data/graphrag/full/corpus.jsonl",
        "data/graphrag/full/embeddings.jsonl", "data/graphrag/full/manifest.json",
        "data/graphrag/index-manifest.json", ".runtime/graphrag-model.json",
        "recipegen/catalog.py", "recipegen/kg_extract.py", "recipegen/graphrag_retrieval.py",
        "recipegen/system_generation.py", "recipegen/graphrag_answer.py",
        "scripts/local_text_embedding.py", "scripts/evaluate_real_retrieval.py",
    ]
    info = json_read(ROOT / ".runtime/graphrag-model.json")
    model_path = Path(info["path"]).resolve()
    if not model_path.is_relative_to(ROOT / ".runtime/text-model-cache"):
        raise ValueError("Unexpected local model path")
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"):
        if (model_path / name).is_file():
            relative.append(str((model_path / name).relative_to(ROOT)))
    return {name: digest(ROOT / name) for name in relative}


def prepare():
    if (DATA / "manifest.json").exists():
        raise FileExistsError("Benchmark already frozen; use --run. No result-based regeneration.")
    DATA.mkdir(parents=True, exist_ok=True)
    raw, corpus = read_sources()
    log("source_read", recipes=len(raw), steps=sum(len(r["steps"]) for r in raw.values()))
    evidence = {}
    by_term = defaultdict(set)
    for rid in sorted(raw):
        evidence[rid] = {}
        for name in GOLD_RULES:
            proof = source_evidence(raw[rid], name)
            if proof:
                evidence[rid][name] = proof
                by_term[name].add(rid)
    rng = random.Random(SEED)
    title_counts = Counter(r["title"].strip().lower() for r in raw.values())
    unique = [rid for rid in sorted(raw) if title_counts[raw[rid]["title"].strip().lower()] == 1
              and raw[rid]["steps"] and len(raw[rid]["title"].split()) <= 12]
    rng.shuffle(unique)
    chosen = [("known_item", rid, [], []) for rid in unique[:20]]
    pairs = []
    for a, b in itertools.combinations(sorted(GOLD_RULES), 2):
        positives = by_term[a] & by_term[b]
        if PARAMETERS["gold_min_relevant"] <= len(positives) <= PARAMETERS["gold_max_relevant"]:
            pairs.append((a, b))
    rng.shuffle(pairs)
    if len(pairs) < 40:
        raise ValueError("Insufficient frozen query candidates")
    for pair in pairs[:20]:
        chosen.append(("multi_ingredient", None, list(pair), []))
    exclusion_candidates = []
    for pair in pairs[20:]:
        base = by_term[pair[0]] & by_term[pair[1]]
        excluded_terms = sorted(set(GOLD_RULES) - set(pair))
        rng.shuffle(excluded_terms)
        for term in excluded_terms:
            positive = base - by_term[term]
            if PARAMETERS["gold_min_relevant"] <= len(positive) <= PARAMETERS["gold_max_relevant"] and len(positive) < len(base):
                exclusion_candidates.append((pair, term))
                break
        if len(exclusion_candidates) == 20:
            break
    if len(exclusion_candidates) != 20:
        raise ValueError("Insufficient non-vacuous exclusion queries")
    for pair, exclude in exclusion_candidates:
        chosen.append(("ingredient_exclusion", None, list(pair), [exclude]))
    queries, qrels = [], []
    per_category = Counter()
    for category, target, include, exclude in chosen:
        n = per_category[category]
        per_category[category] += 1
        qid = f"{category}-{n + 1:02d}"
        if target:
            query = raw[target]["title"]
            relevant = {target}
        else:
            query = f"请找原始步骤或标题中同时提到{ZH[include[0]]}和{ZH[include[1]]}的食谱"
            if exclude:
                query += f"，排除提到{ZH[exclude[0]]}的记录"
            query += "。"
            relevant = set.intersection(*(by_term[name] for name in include))
            for name in exclude:
                relevant -= by_term[name]
        row = {"query_id": qid, "split": "dev" if n < 4 else "test", "category": category,
               "query": query, "language": "en" if target else "zh",
               "template": "copy_original_unique_title" if target else "literal_mention_request_zh_v1",
               "target_recipe_id": target, "ingredients": include, "excluded_ingredients": exclude,
               "gold_relevant_count": len(relevant), "human_annotated": False,
               "label_semantics": "known_source_target" if target else "all_included_literal_mentions_and_no_excluded_literal_mention",
               "title_copy_leakage_risk": bool(target), "natural_user_query_generalization": False}
        queries.append(row)
        for rid in sorted(relevant):
            proof = ({"field": "title", "original_text": raw[rid]["title"],
                      "archive": raw[rid]["source"]["archive"],
                      "member": raw[rid]["source"]["goal_member"],
                      "source_text_sha256": raw[rid]["source"]["goal_sha256"]} if target else None)
            qrels.append({"query_id": qid, "recipe_id": rid, "relevance": 1,
                          "label_type": row["label_semantics"], "human_annotated": False,
                          "corpus_text_sha256": corpus[rid]["text_sha256"],
                          "target_title_evidence": proof,
                          "included_match_evidence": [evidence[rid][name] for name in include],
                          "excluded_terms_absent_from_original_title_and_steps": exclude,
                          "not_an_allergy_or_inventory_label": True})
    write_json(DATA / "gold_lexicon.json", {"version": 1, "independent_of_production_extractor": True,
               "boundary": "case-insensitive ASCII word boundaries; compounds count as literal phrase mentions",
               "entries": [{"name": name, "zh_query_label": zh, "literal_aliases": aliases} for name, zh, aliases in GOLD_LEXICON]})
    write_lines(DATA / "queries.jsonl", queries)
    write_lines(DATA / "qrels.jsonl", qrels)
    # Production features are a separate derived cache, never gold qrels.
    from recipegen.kg_extract import RuleExtractor
    extractor = RuleExtractor()
    features = []
    tick = time.monotonic()
    for n, rid in enumerate(sorted(raw), 1):
        mentions = [m for m in extractor.extract_steps(raw[rid]["steps"]) if m["kind"] == "Ingredient"]
        names = sorted({m["normalized_name"] for m in mentions} | {m.get("zh_name", "").lower() for m in mentions if m.get("zh_name")})
        features.append({"recipe_id": rid, "production_mention_names": names,
                         "source_text_sha256": corpus[rid]["text_sha256"]})
        if n % 1000 == 0:
            log("production_flat_features", processed=n, total=len(raw), elapsed_seconds=round(time.monotonic() - tick, 2))
    write_lines(DATA / "production_flat_metadata.jsonl", features)
    manifest = {"schema_version": 1, "status": "frozen_before_retrieval", "benchmark": "real-retrieval-v1",
                "parameters": PARAMETERS, "input_sha256": input_hashes(),
                "artifact_sha256": {name: digest(DATA / name) for name in
                   ["queries.jsonl", "qrels.jsonl", "gold_lexicon.json", "production_flat_metadata.jsonl"]},
                "source_recipes": len(raw), "source_steps": sum(len(r["steps"]) for r in raw.values()),
                "queries": len(queries), "dev_queries": 12, "test_queries": 48,
                "source_split_name": "original_dataset_test", "eval_split_is_query_holdout_only": True,
                "dataset_revision": json_read(ROOT / "data/graphrag/full/manifest.json")["dataset"]["revision"],
                "model": json_read(ROOT / ".runtime/graphrag-model.json")["model"],
                "model_revision": json_read(ROOT / ".runtime/graphrag-model.json")["revision"],
                "human_annotated": False, "semantic_accuracy_verified": False,
                "qrels_source": "raw records.jsonl title/steps only; graph never consulted",
                "title_duplicate_groups": sum(v > 1 for v in title_counts.values()),
                "gold_relation_count": len(qrels)}
    write_json(DATA / "manifest.json", manifest)
    (DATA / "PROTOCOL.md").write_text(protocol_text(manifest), encoding="utf-8")
    log("prepared_frozen", queries=len(queries), qrel_rows=len(qrels), data=str(DATA))


def protocol_text(manifest):
    return """# RecipeGen 真实数据离线检索基准 v1

这是自动构造的来源字段弱标注评测，不是人工语义相关性评测。源数据为原始 test/test-video 归档的 5898 条食谱、44802 个步骤；数据库的原始 split 名称 test 与本基准的 dev/test 查询划分是两回事。

## 冻结

随机种子 20261010。60 条查询：唯一原英文标题 known-item、多食材共同提及、多食材＋排除提及各 20 条，每类前 4 条 dev、其余 16 条 test。没有训练或参数搜索。检索语料仍包含目标文档。模板在 dev/test 共享，不主张自然用户查询泛化。原题复制的 20 条只能衡量已知来源找回，存在词面复制偏易风险。

manifest.json 固定源记录、完整 corpus、已有 passage 向量、原始 graph 导出、相关生产源码、模型权重/tokenizer、查询、qrels、规则词表和平面过滤元数据的 SHA-256。准备后执行不能重建查询或 qrels。后续运行验证所有哈希；不根据结果换查询、词表、参数或源数据。

## Gold 与指标

known-item 只采原始数据中英文标题唯一且有步骤的记录，qrels 为该来源 ID，报告 Target Hit@1/5/10 与 Target MRR@10，不称完整语义 Recall。

其余 40 条使用独立固定的 32 项字面词表，大小写无关、ASCII 单词边界，扫描全部原始 title/steps 穷举 qrels，禁止从 graph Ingredient 或被测检索器生成标签。rice 在 rice flour、garlic 在 garlic powder 也构成字面提及；与生产的复合实体抽取不同并不自动说明语义错误。每个正例保存匹配原文、字符偏移、步骤序号、归档 member 和原源 SHA。排除仅表示这些字面词未出现在已记录 title/steps；缺失配方不允许推导过敏安全。

多正例报告规则相关集 Recall@1/5/10、Hit@1/5/10、Precision@1/5/10（固定 k 作分母，不足 k 不补零记录）和 MRR@10；相关集较大时 Hit/Precision 用于解释低 Recall，不能替代宏平均 Recall。空 qrels 不计这些指标。本基准查询生成保证至少 3 个正例，最多 150 个；这属于预先规定的构造分布，不是自然查询分布。dev/test、类别分别报告，禁止将两种 gold 语义混成一个总 Recall。

## 三路与过滤消融

A：已有 E5 passage 向量与一次批量离线编码的 query 向量，float32 精确 cosine，recipe_id 打破并列。B：相同 dense 候选＋忠实复现 catalog._terms、SEARCH_QUERY 的关键词检索与 RRF(k=60)。C：B 的同一排名＋只读 graph 原步骤/Source/Ingredient 证据装配，不增加候选、不造图谱权重。B/C 的 ID 排名必须一致。图谱贡献另看原步骤完整性、来源归属、引用可解析率；不是语义正确率。

dense 一律先全语料 top50，再生产食材提及过滤，不补搜；keyword 忠实生产全库词面搜索、生产过滤后取 20。B/C 内部最多 50+20 个候选，RRF 后最多保留 50，最终 k=1/5/10；记录实际候选数。不同算法内部通道不同，固定的是公开通道预算、最终上下文预算和过滤规则，并不声称总计算量相同。

所有方法对同一查询分别运行 without_structured_filter 与 with_structured_filter。过滤元数据独立由生产 RuleExtractor 从原始 steps 生成，不用 gold 词表；include 必须全部命中规范食材/中文显示名，exclude 不得命中。graphrag_retrieval._eligible 只做提及过滤；回答阶段 system_generation 的全文排除是后续另一层，不混入检索排名。查询的正向食材仅在本评测显式传入结构化 ingredients，不自动视为库存充足。缺料许可、总用时、食材替代不属于本基准 gold。

## 执行范围与产物

使用现有 .venv-mm 的 E5，只加载模型一次，CPU 批量编码，设置 HF/Transformers offline 并拒绝 socket connect，不读 .env、不调用 API。输出全部 60 条 query 向量、六组方法/过滤排名、逐条指标与失败案例、图谱装配检查。结果是离线精确 cosine 和生产规则重放，不是 Neo4j ANN 在线端到端效果或延迟。没有 LLM 调用，不能报告模型改写忠实度或真实 API 回退率。引用拒绝/异常回退由其他真实调用或明确故障注入报告单独证明。

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-mm/bin/python scripts/evaluate_real_retrieval.py --prepare
PYTHONDONTWRITEBYTECODE=1 .venv-mm/bin/python scripts/evaluate_real_retrieval.py --run
```
"""


def verify_freeze():
    manifest = json_read(DATA / "manifest.json")
    if manifest["parameters"] != PARAMETERS:
        raise ValueError("Parameters changed after freeze")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    for name, expected in manifest["artifact_sha256"].items():
        if digest(DATA / name) != expected:
            raise ValueError("Frozen benchmark changed: " + name)
    return manifest


def deny_network():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    def denied(*args, **kwargs):
        raise RuntimeError("Network forbidden in this offline benchmark")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def query_vectors(queries, manifest):
    from scripts.local_text_embedding import load_encoder, encode
    path = DATA / "query_embeddings.jsonl"
    if path.exists():
        rows = list(lines(path))
        if len(rows) != len(queries) or any(r["query_id"] != q["query_id"] or r["query"] != q["query"]
                 or r["model_revision"] != manifest["model_revision"] for r, q in zip(rows, queries)):
            raise ValueError("Query embedding cache differs from frozen queries")
        log("reuse_frozen_query_vectors", queries=len(rows))
        return rows, {"cache_reused": True, "model_loads_this_run": 0}
    log("load_local_e5_once", model=manifest["model"], device="cpu")
    tokenizer, model = load_encoder()
    result = []
    batch = PARAMETERS["query_batch_size"]
    tick = time.monotonic()
    for start in range(0, len(queries), batch):
        selected = queries[start:start + batch]
        values = encode([q["query"] for q in selected], tokenizer, model, prefix="query: ", device="cpu")
        for q, vector in zip(selected, values):
            result.append({"query_id": q["query_id"], "query": q["query"], "model": manifest["model"],
                           "model_revision": manifest["model_revision"], "prefix": "query: ", "embedding": vector})
        log("query_encoding", completed=len(result), total=len(queries), elapsed_seconds=round(time.monotonic() - tick, 2))
    write_lines(path, result)
    return result, {"cache_reused": False, "model_loads_this_run": 1,
                    "encoding_seconds": round(time.monotonic() - tick, 4)}


def eligible(names, include, exclude):
    return all(set(term["names"]) & names for term in include) and not any(set(term["names"]) & names for term in exclude)


def keyword_rank(query, ids, corpus, feature_names, include, exclude):
    from recipegen.catalog import _terms, parse_query_filters
    parsed = parse_query_filters(query)
    terms = _terms(parsed["positive_query"])
    title_query = parsed["positive_query"].strip().lower()
    candidates = []
    for rid in ids:
        title = corpus[rid]["title"].lower()
        names = feature_names[rid]
        if (all(any(name in title or name in names for name in term["names"]) for term in terms)
                and eligible(names, include, exclude)):
            normalized_title = title.strip()
            level = 0 if title_query and normalized_title == title_query else 1 if title_query and title_query in normalized_title else 2
            candidates.append((level, title, rid))
    candidates.sort()
    return [row[2] for row in candidates[:PARAMETERS["keyword_candidate_limit"]]]


def ranks(dense_ids, keyword_ids, scores, method):
    dense = {rid: n for n, rid in enumerate(dense_ids, 1)}
    keyword = {rid: n for n, rid in enumerate(keyword_ids, 1)}
    if method == "A_vector":
        selected = dense_ids
    else:
        selected = sorted(set(dense) | set(keyword), key=lambda rid:
            (-(1 / (60 + dense[rid]) if rid in dense else 0)
             -(1 / (60 + keyword[rid]) if rid in keyword else 0), rid))
    selected = selected[:PARAMETERS["final_candidate_limit"]]
    return [{"rank": n, "recipe_id": rid, "semantic_rank": dense.get(rid),
             "keyword_rank": keyword.get(rid), "cosine_score": scores.get(rid),
             "rrf_score": ((1 / (60 + dense[rid]) if rid in dense else 0)
                           + (1 / (60 + keyword[rid]) if rid in keyword else 0)) if method != "A_vector" else None}
            for n, rid in enumerate(selected, 1)]


def metrics(query, ranking, relevant):
    ids = [r["recipe_id"] for r in ranking]
    first = next((n for n, rid in enumerate(ids[:10], 1) if rid in relevant), None)
    result = {"target_mrr_at_10" if query["category"] == "known_item" else "mrr_at_10": 1 / first if first else 0.0}
    for k in PARAMETERS["ks"]:
        hits = len(set(ids[:k]) & relevant)
        if query["category"] == "known_item":
            result[f"target_hit_at_{k}"] = float(bool(hits))
        else:
            result.update({f"recall_at_{k}": hits / len(relevant), f"hit_at_{k}": float(bool(hits)), f"precision_at_{k}": hits / k})
    return result


def graph_evidence(selected, raw, corpus, manifest):
    related = defaultdict(lambda: defaultdict(list))
    graph_file = ROOT / "data/test_graph/relationships.jsonl"
    for edge in lines(graph_file):
        if edge["start_id"] in selected and edge["type"] in {"HAS_STEP", "HAS_SOURCE", "HAS_INGREDIENT"}:
            p = edge["properties"]
            if p.get("split") != "test" or p.get("dataset_revision") != manifest["dataset_revision"]:
                raise ValueError("Graph edge scope mismatch")
            related[edge["start_id"]][edge["type"]].append(edge)
    needed = set(selected)
    for links in related.values():
        needed.update(e["end_id"] for values in links.values() for e in values)
    nodes = {}
    for node in lines(ROOT / "data/test_graph/nodes.jsonl"):
        if node["id"] in needed:
            p = node["properties"]
            if p.get("split") != "test" or p.get("dataset_revision") != manifest["dataset_revision"]:
                raise ValueError("Graph node scope mismatch")
            nodes[node["id"]] = node
    evidence = []
    for rid in sorted(selected):
        failures = []
        links = related[rid]
        step_ids = sorted({edge["end_id"] for edge in links["HAS_STEP"]})
        source_ids = sorted({edge["end_id"] for edge in links["HAS_SOURCE"]})
        ingredient_ids = sorted({edge["end_id"] for edge in links["HAS_INGREDIENT"]})
        all_ids = [rid, *step_ids, *source_ids, *ingredient_ids]
        if any(value not in nodes for value in all_ids):
            failures.append("unresolvable_graph_id")
        steps = [dict(id=value, **nodes[value]["properties"]) for value in step_ids if value in nodes]
        steps.sort(key=lambda value: (value["order"], value["id"]))
        sources = [dict(id=value, **nodes[value]["properties"]) for value in source_ids if value in nodes]
        if [(s["id"], s["text"], s["source_id"], s["order"]) for s in steps] != [
                (s["id"], s["text"], s["source_id"], s["order"]) for s in corpus[rid]["steps"]]:
            failures.append("original_steps_incomplete_or_changed")
        if not steps:
            failures.append("no_original_steps")
        if set(source_ids) != set(corpus[rid]["source_ids"]):
            failures.append("source_set_changed")
        source_set = set(source_ids)
        if any(step["source_id"] not in source_set for step in steps):
            failures.append("step_source_not_owned_by_recipe")
        source_record = raw[rid]["source"]
        expected_hashes = {source_record.get(k + "_member"): source_record.get(k + "_sha256") for k in ["goal", "steps"]}
        if any(s.get("sha256") != expected_hashes.get(s.get("member")) for s in sources):
            failures.append("source_sha_different_from_raw_record")
        if rid not in nodes or nodes[rid]["properties"].get("title") != raw[rid]["title"]:
            failures.append("recipe_title_changed")
        evidence.append({"recipe_id": rid, "steps": steps, "sources": sources,
                         "ingredient_ids": ingredient_ids, "citation_bindings": [
                         {"step_id": s["id"], "source_id": s["source_id"]} for s in steps],
                         "passed": not failures, "failures": failures,
                         "scope": "frozen_graph_source_binding_only_not_semantic_correctness"})
    write_lines(DATA / "graph_evidence.jsonl", evidence)
    log("graph_source_bindings", unique_recipes=len(evidence), passed=sum(e["passed"] for e in evidence))
    return {item["recipe_id"]: item for item in evidence}


def run():
    deny_network()
    start = time.monotonic()
    manifest = verify_freeze()
    raw, corpus = read_sources()
    ids = sorted(corpus)
    queries = list(lines(DATA / "queries.jsonl"))
    gold = defaultdict(set)
    for row in lines(DATA / "qrels.jsonl"):
        gold[row["query_id"]].add(row["recipe_id"])
    feature_names = {row["recipe_id"]: set(row["production_mention_names"]) for row in lines(DATA / "production_flat_metadata.jsonl")}
    if set(feature_names) != set(ids) or len(queries) != 60:
        raise ValueError("Frozen feature/query count mismatch")
    for q in queries:
        if len(gold[q["query_id"]]) != q["gold_relevant_count"]:
            raise ValueError("Qrels count mismatch")
    log("frozen_inputs_verified", queries=len(queries), recipes=len(ids))
    vectors, encoding = query_vectors(queries, manifest)
    import numpy as np
    embeddings = {row["recipe_id"]: row for row in lines(ROOT / "data/graphrag/full/embeddings.jsonl")}
    if set(embeddings) != set(ids):
        raise ValueError("Passage vector ID coverage mismatch")
    for rid in ids:
        row = embeddings[rid]
        if row["text_sha256"] != corpus[rid]["text_sha256"] or row["model_revision"] != manifest["model_revision"]:
            raise ValueError("Passage vector provenance mismatch")
    passages = np.asarray([embeddings[rid]["embedding"] for rid in ids], dtype=np.float32)
    query_matrix = np.asarray([row["embedding"] for row in vectors], dtype=np.float32)
    if passages.shape != (5898, 384) or query_matrix.shape != (60, 384):
        raise ValueError("E5 vector shape mismatch")
    if not np.isfinite(passages).all() or not np.isfinite(query_matrix).all() or not np.allclose(np.linalg.norm(passages, axis=1), 1, atol=.01):
        raise ValueError("Invalid normalized passage/query vector")
    tick = time.monotonic()
    similarities = query_matrix @ passages.T
    vector_seconds = time.monotonic() - tick
    from recipegen.catalog import _filters, parse_query_filters
    rows = []
    for index, query in enumerate(queries):
        scores = {rid: float(similarities[index, n]) for n, rid in enumerate(ids)}
        base_dense = [ids[n] for n in np.argsort(-similarities[index], kind="stable")[:50]]
        for variant in ["without_structured_filter", "with_structured_filter"]:
            included = _filters(query["ingredients"], "ingredients") if variant == "with_structured_filter" else []
            excluded = _filters(query["excluded_ingredients"], "excluded_ingredients") if variant == "with_structured_filter" else []
            # Query text has '排除提到...' rather than an extra unsupported
            # negative-clause syntax; structured filters are explicit and equal.
            parsed = parse_query_filters(query["query"])
            if variant == "with_structured_filter":
                excluded += _filters(parsed["excluded"], "excluded_ingredients")
            dense = [rid for rid in base_dense if eligible(feature_names[rid], included, excluded)]
            try:
                keyword = keyword_rank(query["query"], ids, corpus, feature_names, included, excluded)
                keyword_error = None
            except ValueError as error:
                keyword, keyword_error = [], type(error).__name__
            for method in ["A_vector", "B_hybrid", "C_hybrid_graph_evidence"]:
                ranking = ranks(dense, keyword if method != "A_vector" else [], scores, method)
                actual_ids = [r["recipe_id"] for r in ranking[:10]]
                violations = [rid for rid in actual_ids if query["category"] != "known_item" and rid not in gold[query["query_id"]]]
                result = {**{k: query[k] for k in ["query_id", "split", "category", "query", "language"]},
                          "variant": variant, "method": method, "qrels_count": len(gold[query["query_id"]]),
                          "metrics": metrics(query, ranking, gold[query["query_id"]]), "ranking": ranking,
                          "dense_raw_candidate_count": len(base_dense), "dense_after_filter_count": len(dense),
                          "keyword_candidate_count": len(keyword) if method != "A_vector" else 0,
                          "union_candidate_count": len(set(dense) | set(keyword)) if method != "A_vector" else len(dense),
                          "returned_candidate_count": len(ranking), "keyword_error": keyword_error if method != "A_vector" else None,
                          "gold_predicate_violations_at_10": violations,
                          "missed_gold_at_10": sorted(gold[query["query_id"]] - set(actual_ids)),
                          "insufficient_top10": len(ranking) < 10,
                          "evidence_scope": "offline_exact_cosine_production_rule_replay"}
                rows.append(result)
        if (index + 1) % 10 == 0:
            log("retrieval_replay", completed=index + 1, total=len(queries))
    for q in queries:
        for variant in ["without_structured_filter", "with_structured_filter"]:
            selected = [r for r in rows if r["query_id"] == q["query_id"] and r["variant"] == variant]
            if [r["recipe_id"] for r in selected[1]["ranking"]] != [r["recipe_id"] for r in selected[2]["ranking"]]:
                raise ValueError("Graph evidence must not change recipe ranking")
    graph_selected = {item["recipe_id"] for row in rows if row["method"] == "C_hybrid_graph_evidence" for item in row["ranking"][:10]}
    graph = graph_evidence(graph_selected, raw, corpus, manifest)
    for row in rows:
        if row["method"] == "C_hybrid_graph_evidence":
            selected = [item["recipe_id"] for item in row["ranking"][:10]]
            row["graph_source_binding"] = {"checked_recipes": len(selected),
                "passed_recipes": sum(graph[rid]["passed"] for rid in selected),
                "failed_recipe_ids": [rid for rid in selected if not graph[rid]["passed"]]}
    write_lines(DATA / "rankings.jsonl", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["category"], row["variant"], row["method"])].append(row)
    summary = []
    for (split, category, variant, method), values in sorted(grouped.items()):
        names = sorted(values[0]["metrics"])
        summary.append({"split": split, "category": category, "variant": variant, "method": method,
                        "queries": len(values), "macro_metrics": {name: sum(row["metrics"][name] for row in values) / len(values) for name in names},
                        "mean_relevant_count": sum(row["qrels_count"] for row in values) / len(values),
                        "min_relevant_count": min(row["qrels_count"] for row in values),
                        "median_relevant_count": median(row["qrels_count"] for row in values),
                        "max_relevant_count": max(row["qrels_count"] for row in values),
                        "query_languages": dict(Counter(row["language"] for row in values)),
                        "queries_with_no_relevant_top10": sum(not any("hit_at_10" in name and row["metrics"][name] for name in names) for row in values),
                        "queries_with_gold_predicate_violation_top10": sum(bool(row["gold_predicate_violations_at_10"]) for row in values),
                        "gold_predicate_violation_records_top10": sum(len(row["gold_predicate_violations_at_10"]) for row in values),
                        "queries_returning_fewer_than_10": sum(row["insufficient_top10"] for row in values)})
    report = {"schema_version": 1, "status": "completed", "benchmark": "real-retrieval-v1",
              "scope": "real_source_data_offline_exact_cosine_and_production_rule_replay_not_Neo4j_ANN",
              "human_annotated": False, "semantic_accuracy_verified": False, "llm_calls": 0,
              "network_calls": 0, "dotenv_read": False, "queries": len(queries), "dev_queries": 12, "test_queries": 48,
              "source_recipes": len(ids), "source_steps": manifest["source_steps"],
              "parameters": PARAMETERS, "manifest_sha256": digest(DATA / "manifest.json"),
              "query_sha256": digest(DATA / "queries.jsonl"), "qrels_sha256": digest(DATA / "qrels.jsonl"),
              "query_embeddings_sha256": digest(DATA / "query_embeddings.jsonl"),
              "rankings_sha256": digest(DATA / "rankings.jsonl"), "graph_evidence_sha256": digest(DATA / "graph_evidence.jsonl"),
              "encoding": encoding, "exact_dense_matrix_seconds": vector_seconds,
              "elapsed_seconds": time.monotonic() - start, "summary": summary,
              "graph_ranking_identity_check": "B_equals_C_all_60_queries_both_filter_variants",
              "graph_source_binding": {"distinct_top10_recipes": len(graph), "passed": sum(item["passed"] for item in graph.values()),
                  "failed": [{"recipe_id": rid, "reasons": item["failures"]} for rid, item in graph.items() if not item["passed"]],
                  "checked_steps": sum(len(item["steps"]) for item in graph.values()),
                  "scope": "source_identifiers_hashes_and_complete_original_steps_not_semantic_faithfulness"},
              "limitations": ["Automatically generated source-field weak labels; no human relevance judgments.",
                  "Known-item copies unique original titles and measures target lookup only, not broad semantic recall.",
                  "Multi-ingredient qrels are exhaustive only for the frozen literal-mention predicate.",
                  "Gold and production extractors intentionally differ; compound-mention mismatches are not human semantic errors.",
                  "Dev/test hold out queries, not corpus documents; templates and vocabulary are shared; no natural-query generalization claim.",
                  "All internal channel budgets and final k are fixed, but hybrid has an extra keyword channel; total compute is not equal.",
                  "Structured exclusion means recorded mention filtering, not inventory completeness or allergy safety.",
                  "Graph supplies evidence within each retrieved recipe, no new candidate or reranking.",
                  "Latency is offline replay; this run does not test Neo4j ANN, external generation, citation paraphrase truth or real API fallback."]}
    write_json(REPORT, report)
    REPORT.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    log("completed", report=str(REPORT), elapsed_seconds=round(report["elapsed_seconds"], 2))


def render_report(report):
    text = ["# RecipeGen 真实数据离线检索结果 v1", "",
            "**已完成 60 条源字段自动标注查询（12 dev／48 test）；5898 条真实来源食谱、44802 个原始步骤。没有人工相关性标注。**",
            "", "E5 真实本地编码＋已有 passage 向量精确 cosine；生产关键词规则、RRF(k=60) 和结构化过滤离线重放。不是 Neo4j ANN 在线端到端效果。没有 LLM/API 调用。",
            "", "known-item 的 20 条英文问题直接复制唯一原标题，只报告 Target Hit/MRR。其余 40 条中文问题的 qrels 来自独立字面词表穷举扫描原始 title/steps；Recall 的范围仅为该固定谓词。dev/test 是查询划分，语料包含被检索目标。", "",
            "## 冻结 test 结果", "",
            "| 条件 | 类别 | 方法 | n | Target Hit@10 / Hit@10 | Target MRR@10 / MRR@10 | Recall@10 | Precision@10 | 平均相关集大小 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"A_vector": "E5", "B_hybrid": "关键词＋E5", "C_hybrid_graph_evidence": "混合＋图谱证据"}
    for row in report["summary"]:
        if row["split"] != "test":
            continue
        m = row["macro_metrics"]
        known = row["category"] == "known_item"
        hit = m["target_hit_at_10" if known else "hit_at_10"]
        mrr = m["target_mrr_at_10" if known else "mrr_at_10"]
        recall = "—" if known else f"{m['recall_at_10']:.4f}"
        precision = "—" if known else f"{m['precision_at_10']:.4f}"
        text.append(f"| {'统一提及过滤' if row['variant'] == 'with_structured_filter' else '无结构化过滤'} | {row['category']} | {labels[row['method']]} | {row['queries']} | {hit:.4f} | {mrr:.4f} | {recall} | {precision} | {row['mean_relevant_count']:.2f} |")
    graph = report["graph_source_binding"]
    text += ["", "## 图谱装配与边界", "",
             f"B/C 在全部 60 条查询、两种过滤条件下 ID 排名一致。C 对 {graph['distinct_top10_recipes']} 个不同 top10 食谱装配原步骤与 Source，来源归属／原步骤完整性检查通过 {graph['passed']} 个；核对 {graph['checked_steps']} 个步骤。该检查不证明语义正确、完整配方、翻译忠实或忌口安全。", "",
             "dense 先取全语料 top50 再提及过滤、不补搜；keyword 先全库词面匹配及过滤、再取 20。混合内部候选最多 70，RRF 后保留最多 50。所有方法使用同一生产结构化过滤和最终 k，另保留无过滤消融。提及标签与生产实体抽取对复合词的处理不同，逐条违反 gold 谓词的记录已全部保存，不能自动视为人工确认的语义错误。", "",
             "低 Recall 可能源于正例集合远大于 k；同时给出 Hit/Precision，并保留每条的相关集大小，不能把 Hit 当 Recall。Known-item 的高分存在原标题复制偏易风险。图谱没有扩展新食谱，不能声称图谱改善了召回。", "",
             "缺料许可、替代食材、总用时、过敏安全无可靠 gold，不纳入效果指标。检索不足只表示有限候选不足。API 回退／引用拒绝要单独用真实调用或明确故障注入验证，本报告未伪造该证据。", "",
             "## 可复核文件", "",
             "- `data/evaluation_real_v1/PROTOCOL.md`：执行前冻结协议。",
             "- `manifest.json`：输入、模型、规则、查询与 qrels 的 SHA-256。",
             "- `queries.jsonl`、`qrels.jsonl`、`gold_lexicon.json`：全部查询及匹配原文/偏移/来源 SHA。",
             "- `query_embeddings.jsonl`：真实本地 E5 查询向量；模型只加载一次。",
             "- `rankings.jsonl`：60×2×3 组完整候选排名、指标、漏召回、gold 谓词违规与不足 k。",
             "- `graph_evidence.jsonl`：C 的步骤、Source、引用绑定及失败。",
             "- `reports/real-retrieval-v1.json`：按 dev/test、类别、条件、方法分组的完整 @1/@5/@10 宏平均指标。", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Construct and freeze source-derived queries before retrieval")
    parser.add_argument("--run", action="store_true", help="Verify freeze then perform real E5 offline evaluation")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Select exactly one of --prepare or --run")
    deny_network()
    if args.prepare:
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
