#!/usr/bin/env python3
"""Frozen, offline real-Qwen selection checks; not a semantic quality benchmark.

Prepare inputs before inference, then run exactly once. Only this script and
reports/local-selection-v1.{json,md} are authored; production logic is reused.
No Settings.from_env(), .env reads, network calls, or model downloads occur.
"""
from __future__ import annotations

import argparse
import collections
import copy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.system_generation import (
    MODEL_REPO, MODEL_REVISION, PROTOCOL, RecipeGenerator,
    GenerationValidationError, _answer, _candidate_records, _evidence,
    _selection_payload, validate_local_selection,
)
from scripts.local_recipe_generation import instruction

REPORT = ROOT / "reports/local-selection-v1.json"
MARKDOWN = ROOT / "reports/local-selection-v1.md"
REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
SOURCE_FILES = (
    "data/graphrag/full/corpus.jsonl", "data/test_graph/nodes.jsonl",
    "data/test_graph/relationships.jsonl", "data/test_graph/records.jsonl",
    "data/test_graph/source/test.texts.jsonl",
    "data/test_graph/source/test-video.texts.jsonl",
)
CODE_FILES = (
    "recipegen/system_generation.py", "scripts/local_recipe_generation.py",
    "scripts/evaluate_local_selection.py",
)
IDS = {
    "noodles": "recipegen:test:test:b5f901b80159a018dbf3",
    "dumplings": "recipegen:test:test:fa69085f10804139085a",
    "pork": "recipegen:test:test-video:004de93faeb0fba12069",
    "vegetables": "recipegen:test:test-video:00b10a0fb3c124899675",
    "eggplant": "recipegen:test:test-video:00c263f2e5e89e366ef5",
    "wings": "recipegen:test:test-video:0104202d87d8a1f5b1e9",
    "soup": "recipegen:test:test-video:017c01b26ed320113eba",
    "fish": "recipegen:test:test-video:030dbc5a9e2508c5f5d8",
    "rice_noodles": "recipegen:test:test-video:0312134a4553fa6dedf6",
    "enoki": "recipegen:test:test-video:09d75419c99e1c4573bb",
    "salad": "recipegen:test:test-video:1529fe11612e4f70959e",
    "tofu": "recipegen:test:test-video:221b1f5e154eb81ba7b1",
}
# Authored before any inference. Fixed IDs and orders are never changed based
# on model success. Targets describe input construction, not scored relevance.
CASE_SPECS = (
    ("G01", "unique_title", "请只选择标题完全为 Breakfast Tomato Noodles 的原始食谱，并用已有证据说明选择。", "noodles", ("eggplant", "noodles", "dumplings"), (), ()),
    ("G02", "unique_title", "请只选择标题完全为 Steamed Salted Pork with Winter Melon 的原始食谱，保留来源。", "pork", ("wings", "salad", "pork"), (), ()),
    ("G03", "unique_title", "请只选择标题完全为 Egg Wrapped Dumplings 的原始食谱，不新增做法。", "dumplings", ("dumplings", "tofu", "fish"), (), ()),
    ("G04", "multiple_ingredients", "我想用番茄和鸡蛋做面条，请在候选中选择有这些原始步骤证据的食谱。", "noodles", ("salad", "eggplant", "noodles"), ("番茄", "鸡蛋"), ()),
    ("G05", "multiple_ingredients", "请找出用茄子夹猪肉和虾馅、再裹面糊油炸的原始食谱，仅引用已有证据。", "eggplant", ("rice_noodles", "eggplant", "salad"), ("茄子", "猪肉", "虾"), ()),
    ("G06", "exclusion", "排除记录中出现鸡蛋的候选；选择冬瓜和咸猪肉一起蒸的食谱，不作过敏安全保证。", "pork", ("dumplings", "soup", "pork", "salad", "fish"), ("冬瓜", "猪肉"), ("鸡蛋",)),
    ("G07", "exclusion", "排除记录中出现鸡蛋的候选；我想把黄瓜、长豆和香肠炒在一起，请选对应原始食谱。", "vegetables", ("enoki", "salad", "vegetables", "pork", "fish"), ("黄瓜", "香肠"), ("鸡蛋",)),
    ("G08", "step_only", "选择原始步骤中先把咸猪肉切成约三毫米薄片、再浸泡冬瓜干的食谱，保留全部原步骤。", "pork", ("rice_noodles", "eggplant", "pork"), (), ()),
    ("G09", "step_only", "选择原始步骤中先把黄瓜和长豆切成小丁，再将香肠切小块并翻炒的食谱。", "vegetables", ("salad", "vegetables", "dumplings"), (), ()),
    ("G10", "step_only", "选择原始步骤中先切去金针菇根部，再把鸡蛋与料酒搅拌并准备淀粉的食谱。", "enoki", ("enoki", "wings", "fish"), (), ()),
    ("G11", "no_invented_quantities", "整理鸡肉和黑木耳汤的做法，仅选择已有原记录，不补造食材用量，引用原始步骤。", "soup", ("tofu", "salad", "soup"), ("鸡肉", "木耳"), ()),
    ("G12", "no_invented_total_duration", "整理豆腐、五花肉和鱼一起炖的原始食谱，不推断总烹饪时长，保留已有步骤及来源。", "fish", ("rice_noodles", "fish", "pork"), ("豆腐", "猪肉", "鱼"), ()),
)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_identity():
    return {name: {"sha256": file_hash(ROOT / name), "bytes": (ROOT / name).stat().st_size}
            for name in SOURCE_FILES + CODE_FILES}


def model_identity():
    info = json.loads((ROOT / ".runtime/multimodal-models.json").read_text())["vision"]
    path = Path(info["path"]).resolve()
    if (info.get("repo") != MODEL_REPO or info.get("revision") != MODEL_REVISION
            or not path.is_relative_to((ROOT / ".runtime/model-cache").resolve())):
        raise ValueError("fixed_local_model_identity_mismatch")
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]
    identity = {"repo": MODEL_REPO, "revision": MODEL_REVISION, "backend": "mlx-vlm/MLX",
                "num_images": 0, "max_tokens": 768, "temperature": 0.0,
                "repetition_penalty": 1.1, "repetition_context_size": 128,
                "mlx_cache_limit_bytes": 64 * 1024 * 1024,
                "files": {name: {"sha256": file_hash(path / name), "bytes": (path / name).stat().st_size}
                          for name in files if (path / name).is_file()},
                "quantization_config": json.loads((path / "config.json").read_text()).get("quantization")}
    if "model.safetensors" not in identity["files"]:
        raise ValueError("fixed_local_weights_missing")
    identity["identity_sha256"] = object_hash(identity)
    return identity, path


def frozen_graph_records():
    wanted = set(IDS.values())
    corpus = {row["recipe_id"]: row for row in jsonl(ROOT / SOURCE_FILES[0])}
    title_counts = collections.Counter(row["title"] for row in corpus.values())
    related = collections.defaultdict(list)
    needed = set(wanted)
    for row in jsonl(ROOT / "data/test_graph/relationships.jsonl"):
        if row["start_id"] in wanted and row["type"] in {"HAS_SOURCE", "HAS_STEP", "HAS_INGREDIENT"}:
            related[row["start_id"]].append(row)
            needed.add(row["end_id"])
    nodes = {row["id"]: row for row in jsonl(ROOT / "data/test_graph/nodes.jsonl") if row["id"] in needed}
    originals = {row["id"]: row for row in jsonl(ROOT / "data/test_graph/records.jsonl") if row["id"] in wanted}
    sources_wanted = {(nodes[edge["end_id"]]["properties"]["archive"], nodes[edge["end_id"]]["properties"]["member"])
                      for edges in related.values() for edge in edges if edge["type"] == "HAS_SOURCE"}
    raw_texts = {}
    for name in SOURCE_FILES[-2:]:
        for row in jsonl(ROOT / name):
            key = (row["source"]["archive"], row["member"])
            if key in sources_wanted:
                raw_texts[key] = row
    result = {}
    for recipe_id in sorted(wanted):
        props = nodes[recipe_id]["properties"]
        assert nodes[recipe_id]["label"] == "Recipe" and props["is_demo"] is False
        assert props["dataset_revision"] == REVISION and props["split"] == "test"
        sources, steps, mentions = [], [], []
        for edge in related[recipe_id]:
            node = nodes[edge["end_id"]]
            values = node["properties"]
            assert values["dataset_revision"] == REVISION and values["build_id"] == props["build_id"]
            if edge["type"] == "HAS_SOURCE":
                item = {"id": node["id"], **values, "role": edge["properties"]["role"]}
                raw = raw_texts[(item["archive"], item["member"])]
                assert item["sha256"] == raw["sha256"]
                if item["role"] == "title":
                    assert raw["text"].strip() == props["title"]
                sources.append(item)
            elif edge["type"] == "HAS_STEP":
                steps.append({"id": node["id"], "order": values["order"], "text": values["text"], "source_id": values["source_id"]})
            else:
                evidence = json.loads(edge["properties"]["evidence_json"])
                source_ids = sorted({hit["source"]["id"] for hit in evidence})
                mentions.append({"id": node["id"], "name": values["normalized_name"],
                                 "normalized_name": values["normalized_name"], "zh_name": values.get("zh_name"),
                                 "source_id": source_ids[0] if len(source_ids) == 1 else None,
                                 "source_ids": source_ids, "evidence": evidence,
                                 "semantics": "text_mention_candidate", "verified": False,
                                 "ingredients_complete": False})
        steps.sort(key=lambda x: x["order"])
        sources.sort(key=lambda x: (x["role"], x["source_id"]))
        mentions.sort(key=lambda x: (x["normalized_name"], x["id"]))
        source_ids = {s["source_id"] for s in sources}
        assert all(s["source_id"] in source_ids for s in steps)
        assert all(set(m["source_ids"]) <= source_ids for m in mentions)
        assert steps == corpus[recipe_id]["steps"]
        assert props["title"] == corpus[recipe_id]["title"] == originals[recipe_id]["title"]
        assert [(s["order"], s["text"]) for s in steps] == [(s["order"], s["text"]) for s in originals[recipe_id]["steps"]]
        assert hashlib.sha256(corpus[recipe_id]["text"].encode()).hexdigest() == corpus[recipe_id]["text_sha256"]
        for s in sources:
            if s["role"] == "steps":
                raw = raw_texts[(s["archive"], s["member"])]
                assert [line.strip() for line in raw["text"].splitlines() if line.strip()] == [step["text"] for step in steps]
                assert s["sha256"] == originals[recipe_id]["source"]["steps_sha256"]
            if s["role"] == "title":
                assert s["sha256"] == originals[recipe_id]["source"]["goal_sha256"]
        result[recipe_id] = {"id": recipe_id, "title": props["title"], "steps": steps, "sources": sources,
                             "ingredient_mentions": mentions, "build_id": props["build_id"],
                             "dataset_revision": REVISION, "split": "test", "is_demo": False,
                             "source_verification": {"nodes_relationships_corpus_records_raw_text_agree": True,
                                                     "title_occurrences_in_corpus": title_counts[props["title"]],
                                                     "corpus_text_sha256": corpus[recipe_id]["text_sha256"]}}
    return result, len(corpus)


def fixed_cases(records):
    cases = []
    for case_id, kind, question, target, names, ingredients, excluded in CASE_SPECS:
        snapshots = [copy.deepcopy(records[IDS[name]]) for name in names]
        eligible, rejected = _candidate_records(snapshots, list(excluded))
        payload, aliases, supplied = _selection_payload(question, eligible, _evidence(eligible), list(ingredients), list(excluded), 1)
        assert len(aliases) >= 3 and len(payload["candidates"]) == len(eligible)
        target_alias = next(alias for alias, identifier in aliases.items() if identifier == IDS[target])
        if kind == "unique_title":
            assert records[IDS[target]]["source_verification"]["title_occurrences_in_corpus"] == 1
        item = {"case_id": case_id, "category": kind, "question": question,
                "ingredients": list(ingredients), "excluded_ingredients": list(excluded), "limit": 1,
                "input_recipe_ids": [r["id"] for r in snapshots], "candidate_records": snapshots,
                "eligible_recipe_ids": [r["id"] for r in eligible], "rejected_before_model": rejected,
                "model_payload": payload, "aliases": aliases, "supplied_evidence": supplied,
                "instruction_sha256": hashlib.sha256(instruction(payload).encode()).hexdigest(),
                "intended_target_for_sample_design_only": {"recipe_id": IDS[target], "alias": target_alias},
                "semantic_target_accuracy_scored": False}
        item["input_sha256"] = object_hash(item)
        cases.append(item)
    assert len(cases) == 12 and sum(len(c["aliases"]) > 1 for c in cases) >= 6
    assert {c["intended_target_for_sample_design_only"]["alias"] for c in cases} >= {"R1", "R2", "R3"}
    return cases


def assess(result, case):
    generation = result["generation"]
    try:
        validate_local_selection(generation.get("raw_text"), case["aliases"], case["supplied_evidence"], case["limit"])
        protocol_valid = True
    except (GenerationValidationError, TypeError, ValueError):
        protocol_valid = False
    originals = {r["id"]: r for r in case["candidate_records"]}
    selected = result.get("recipes", [])
    evidence = result.get("evidence", [])
    evidence_by_id = {e["id"]: e for e in evidence}
    citations = re.findall(r"\[(E\d+)\]", result.get("answer", ""))
    selected_ids = {r["id"] for r in selected}
    original_selected = [originals[r["id"]] for r in selected if r["id"] in originals]
    references_resolve = bool(citations) and len(evidence_by_id) == len(evidence) and all(identifier in evidence_by_id for identifier in citations)
    evidence_owned = bool(selected) and selected_ids <= originals.keys() and all(e["recipe_id"] in selected_ids for e in evidence)
    step_evidence = [e for e in evidence if e["kind"] == "step"]
    expected_steps = {(r["id"], s["id"]): s for r in original_selected for s in r["steps"]}
    step_set_complete = {(e["recipe_id"], e["graph_id"]) for e in step_evidence} == set(expected_steps)
    step_identity = all((e["recipe_id"], e["graph_id"]) in expected_steps
                        and e["text"] == expected_steps[(e["recipe_id"], e["graph_id"])]["text"]
                        and e["source_id"] == expected_steps[(e["recipe_id"], e["graph_id"])]["source_id"]
                        and e["source_id"] in {s["source_id"] for s in originals[e["recipe_id"]]["sources"]}
                        for e in step_evidence)
    raw_steps_preserved = bool(expected_steps) and step_set_complete and step_identity and all(f"{e['text']} [{e['id']}]" in result["answer"] for e in step_evidence)
    selected_record_steps_unchanged = bool(selected) and all(r["steps"] == originals[r["id"]]["steps"] for r in selected if r["id"] in originals)
    replay_evidence = _evidence(original_selected)
    production_assembly_exact = bool(original_selected) and result["answer"] == _answer(original_selected, replay_evidence)
    return {"raw_selection_protocol_valid": protocol_valid,
            "final_citation_ids_resolve": references_resolve,
            "final_evidence_recipe_ownership_valid": evidence_owned,
            "final_step_evidence_matches_original_graph_and_source": bool(step_evidence) and step_identity and step_set_complete,
            "final_steps_original_verbatim_and_complete": raw_steps_preserved and selected_record_steps_unchanged,
            "final_answer_matches_production_grounded_assembly": production_assembly_exact,
            "citation_occurrences": len(citations), "step_evidence_count": len(step_evidence),
            "semantic_faithfulness_evaluated": False}


def summary(cases):
    finished = [c for c in cases if c.get("execution_status") == "finished"]
    count = len(finished)
    keys = ("raw_selection_protocol_valid", "final_citation_ids_resolve", "final_evidence_recipe_ownership_valid",
            "final_step_evidence_matches_original_graph_and_source", "final_steps_original_verbatim_and_complete",
            "final_answer_matches_production_grounded_assembly")
    result = {"completed_cases": count, "metrics_are_structural_not_semantic": True,
              "actual_llm_called": sum(c.get("actual_llm_call_attempted", False) for c in finished),
              "fallback": {"numerator": sum(c["result"]["generation"]["fallback"] for c in finished), "denominator": count}}
    for key in keys:
        numerator = sum(c["metrics"][key] for c in finished)
        result[key] = {"numerator": numerator, "denominator": count, "rate": numerator / count if count else None}
    result["fallback"]["rate"] = result["fallback"]["numerator"] / count if count else None
    result["selected_alias_distribution"] = dict(collections.Counter(alias for c in finished for alias in c.get("selected_aliases", [])))
    return result


def write_report(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    real = report.get("real_model_summary", {})
    rows = ["# RecipeGen 本地 Qwen 选择协议小样本评测 v1", "",
            f"状态：`{report['status']}`；输入冻结：{report['frozen_at']}。", "",
            "本评测使用真实 RecipeGen 测试归档的冻结图谱导出、corpus 和原始文本证据；没有在线 Neo4j、HTTP 或检索请求。",
            "12 条中文问题及候选顺序在模型调用前固定，不根据输出替换样本，不重跑取最好结果。每题至少 3 个模型候选；明确目标分布在 R1/R2/R3。目标用于构造样本，不计语义选择准确率。", "",
            f"输入冻结 SHA-256：`{report['frozen_inputs_sha256']}`。",
            f"模型：`{MODEL_REPO}`；revision：`{MODEL_REVISION}`；身份 SHA-256：`{report['model_identity']['identity_sha256']}`。", "",
            "复用生产 RecipeGenerator、上下文构造、instruction、选择 validator 和原步骤回答组装。MLX 模型一次加载，离线 batch 每题独立 generate；参数与生产 worker 相同，温度 0、max_tokens 768、纯文本。",
            "模型加载耗时单列；每题耗时是加载后的 batch 推理耗时，不能当作生产 subprocess 冷启动端到端延迟。", "",
            "## 真实模型结果", "",
            f"已完成：{real.get('completed_cases', 0)}/12；真实推理调用：{real.get('actual_llm_called', 0)}。"]
    if real.get("completed_cases"):
        rows.extend(["", "| 指标 | 通过/总数 |", "|---|---:|"])
        labels = {"raw_selection_protocol_valid": "原始选择协议有效",
                  "final_citation_ids_resolve": "最终回答引用 ID 可解析",
                  "final_evidence_recipe_ownership_valid": "最终证据归属所选食谱",
                  "final_step_evidence_matches_original_graph_and_source": "步骤证据匹配原图谱与 Source",
                  "final_steps_original_verbatim_and_complete": "原始步骤原文完整保留",
                  "final_answer_matches_production_grounded_assembly": "回答等于生产原步骤组装", "fallback": "真实调用回退"}
        for key, label in labels.items():
            value = real[key]
            rows.append(f"| {label} | {value['numerator']}/{value['denominator']} |")
        rows.extend(["", "这些是代码结构和记录一致性指标。最终引用/步骤指标会受程序组装与回退保护，不能用来推导模型语义忠实度、无幻觉效果或检索质量。"])
    rows.extend(["", "| 样本 | 类别 | 中文问题 | 模型候选数 | 选择 | 协议有效 | 回退 | 推理 ms |", "|---|---|---|---:|---|---|---|---:|"])
    outputs = {c["case_id"]: c for c in report.get("real_model_cases", [])}
    for case in report["frozen_cases"]:
        output = outputs.get(case["case_id"], {})
        metrics = output.get("metrics", {})
        generation = output.get("result", {}).get("generation", {})
        rows.append(f"| {case['case_id']} | {case['category']} | {case['question']} | {len(case['aliases'])} | {','.join(output.get('selected_aliases', [])) or '—'} | {metrics.get('raw_selection_protocol_valid', '—')} | {generation.get('fallback', '—')} | {output.get('inference_elapsed_ms', '—')} |")
    rows.extend(["", "## 受控失败注入", "", "以下使用冻结真实上下文，直接注入 runner 返回；无真实模型调用，不与真实回退率混算，也不是操作系统故障/并发压力测试。", "",
                 "| 注入类别 | 响应错误码 | 回退 | 引用 ID 可解析 | 原步骤完整保留 |", "|---|---|---|---|---|"])
    for case in report.get("injected_failure_cases", []):
        generation = case["result"]["generation"]
        rows.append(f"| {case['injected_error']} | {generation['errors'][0]['code']} | {generation['fallback']} | {case['metrics']['final_citation_ids_resolve']} | {case['metrics']['final_steps_original_verbatim_and_complete']} |")
    rows.extend(["", "## 边界", "",
                 "- 固定 12 条工程例，不是随机抽样、保留测试集或语义 benchmark；不报告语义忠实度、幻觉率、翻译质量、选菜相关性、Recall@K。",
                 "- 全部上下文是真实英文原记录，问题用中文；没有人为修复模型输出。",
                 "- 引用 ID 合法、证据归属正确与步骤原文保留，不代表原始食谱正确、食材完整、用量完整或饮食安全。",
                 "- 真正失败仍保留在 12 个结果中；受控失败 6 类单列，不伪装成自然发生的失败。",
                 "- JSON 保存输入快照、提示 hash、原始输出、选择 ID、回退、耗时、所有代码/数据/模型身份 hash，可复核。"])
    MARKDOWN.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare():
    if REPORT.exists():
        raise ValueError("report_already_exists_do_not_overwrite_or_cherry_pick")
    records, corpus_count = frozen_graph_records()
    cases = fixed_cases(records)
    identity, _ = model_identity()
    report = {"schema_version": "recipegen-local-selection-evaluation-v1", "status": "prepared", "frozen_at": now(),
              "data_mode": "offline_verified_graph_exports_not_live_neo4j", "corpus_recipe_count": corpus_count,
              "dataset_revision": REVISION, "source_identity": source_identity(), "model_identity": identity,
              "environment_probe": {"sandboxed_mlx_import": "Metal device unavailable; import aborted before model load or case inference; excluded from case denominator"},
              "protocol": {"case_count": 12, "no_case_replacement_after_inference": True, "no_best_of_reruns": True,
                           "semantic_target_accuracy_scored": False, "real_and_injected_denominators_separate": True,
                           "production_parameters_reused": True, "latency_scope": "offline_batch_warm_inference_not_subprocess_cold_start"},
              "frozen_cases": cases, "frozen_inputs_sha256": object_hash(cases), "real_model_cases": [],
              "injected_failure_cases": [], "real_model_summary": summary([])}
    write_report(report)
    print(json.dumps({"status": "prepared", "cases": len(cases), "frozen_inputs_sha256": report["frozen_inputs_sha256"],
                      "targets": [c["intended_target_for_sample_design_only"]["alias"] for c in cases],
                      "candidates": [len(c["aliases"]) for c in cases]}, ensure_ascii=False), flush=True)


def run():
    report = json.loads(REPORT.read_text())
    if report["status"] != "prepared" or report["real_model_cases"]:
        raise ValueError("run_once_only_prepared_inputs_required")
    assert object_hash(report["frozen_cases"]) == report["frozen_inputs_sha256"]
    assert source_identity() == report["source_identity"]
    identity, model_path = model_identity()
    assert identity == report["model_identity"]
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    report["status"] = "running"
    report["run_started_at"] = now()
    write_report(report)
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    report["runtime"] = {"python": sys.version.split()[0], "mlx": importlib.metadata.version("mlx"),
                         "mlx_vlm": importlib.metadata.version("mlx-vlm"), "default_device": str(mx.default_device()),
                         "model_loads": 1, "network_enabled_by_harness": False, "dotenv_loaded_by_harness": False}
    mx.set_cache_limit(64 * 1024 * 1024)
    tick = time.perf_counter()
    model, processor = load(str(model_path), trust_remote_code=False)
    report["model_load_elapsed_ms"] = round((time.perf_counter() - tick) * 1000, 3)
    write_report(report)
    print(json.dumps({"event": "model_loaded", "load_ms": report["model_load_elapsed_ms"]}), flush=True)
    last = {}

    def runner(payload):
        last.clear()
        last["payload"] = copy.deepcopy(payload)
        prompt = apply_chat_template(processor, model.config, instruction(payload), num_images=0)
        last["formatted_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        started = time.perf_counter()
        last["actual_llm_call_attempted"] = True
        try:
            raw = generate(model, processor, prompt, max_tokens=768, temperature=0.0,
                           repetition_penalty=1.1, repetition_context_size=128, verbose=False).text
            if not isinstance(raw, str):
                raise TypeError("model_output_not_text")
            return {"protocol": PROTOCOL, "status": "ok", "raw_text": raw, "llm_called": True,
                    "model": {"name": MODEL_REPO, "revision": MODEL_REVISION, "backend": "mlx", "num_images": 0,
                              "max_tokens": 768, "temperature": 0.0, "repetition_penalty": 1.1,
                              "repetition_context_size": 128, "mlx_cache_limit_bytes": 64 * 1024 * 1024}}
        except Exception as error:
            last["safe_exception_type"] = type(error).__name__
            return {"protocol": PROTOCOL, "status": "model_error", "llm_called": True}
        finally:
            last["inference_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            mx.clear_cache()

    service = RecipeGenerator(ROOT, local_runner=runner)
    for case in report["frozen_cases"]:
        output = {"case_id": case["case_id"], "execution_status": "running", "started_at": now(),
                  "input_sha256": case["input_sha256"], "actual_llm_call_attempted": False}
        report["real_model_cases"].append(output)
        write_report(report)
        tick = time.perf_counter()
        result = service.generate(case["question"], copy.deepcopy(case["candidate_records"]), mode="local",
                                  ingredients=case["ingredients"], excluded_ingredients=case["excluded_ingredients"], limit=case["limit"])
        assert last["payload"] == case["model_payload"]
        reverse = {identifier: alias for alias, identifier in case["aliases"].items()}
        output.update({k: v for k, v in last.items() if k != "payload"})
        output.update({"execution_status": "finished", "finished_at": now(),
                       "generator_batch_elapsed_ms": round((time.perf_counter() - tick) * 1000, 3),
                       "selected_aliases": [reverse[item["recipe_id"]] for item in result["generation"]["selection"]],
                       "result": result, "metrics": assess(result, case)})
        report["real_model_summary"] = summary(report["real_model_cases"])
        write_report(report)
        print(json.dumps({"event": "case_finished", "case_id": case["case_id"], "selected_aliases": output["selected_aliases"],
                          "protocol_valid": output["metrics"]["raw_selection_protocol_valid"],
                          "fallback": result["generation"]["fallback"], "inference_ms": output["inference_elapsed_ms"]}, ensure_ascii=False), flush=True)

    for error_code, case in zip(("busy", "timeout", "unavailable", "model_error", "invalid_runner_output", "invalid_output"), report["frozen_cases"]):
        captured = {}

        def injected_runner(payload, code=error_code):
            captured["payload"] = copy.deepcopy(payload)
            if code == "invalid_output":
                return {"status": "ok", "llm_called": True, "raw_text": json.dumps({"selections": [{"recipe": "R999", "reason_codes": ["original_steps"], "evidence_ids": ["E999"]}]})}
            return {"status": code, "llm_called": False}

        tick = time.perf_counter()
        result = RecipeGenerator(ROOT, local_runner=injected_runner).generate(case["question"], copy.deepcopy(case["candidate_records"]), mode="local",
                             ingredients=case["ingredients"], excluded_ingredients=case["excluded_ingredients"], limit=case["limit"])
        assert captured["payload"] == case["model_payload"]
        assert result["generation"]["fallback"] and result["generation"]["errors"][0]["code"] == error_code
        report["injected_failure_cases"].append({"case_id": "F-" + error_code, "source_case_id": case["case_id"],
            "input_sha256": case["input_sha256"], "execution_status": "finished", "injected_error": error_code,
            "actual_llm_call_attempted": False, "controlled_injection_not_real_model_failure": True,
            "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3), "selected_aliases": [],
            "result": result, "metrics": assess(result, case)})
    report["injected_failure_summary"] = summary(report["injected_failure_cases"])
    assert source_identity() == report["source_identity"]
    report["source_and_code_unchanged_after_run"] = True
    report["finished_at"] = now()
    report["status"] = "complete"
    write_report(report)
    print(json.dumps({"status": "complete", "real_summary": report["real_model_summary"],
                      "injected_summary": report["injected_failure_summary"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
