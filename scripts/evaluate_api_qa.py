#!/usr/bin/env python3
"""Freeze six original-source QA questions before an explicit API smoke run.

--diagnose/--prepare never call an API. --run attempts six QA calls and at most
two independent reranks, without retries. This is fixed-candidate QA smoke,
not retrieval evaluation or independent semantic accuracy measurement.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.experimental_qa import (
    ExperimentalEvidenceQA, QA_MAX_TOKENS, QA_PROMPT, RERANK_MAX_TOKENS,
    RERANK_PROMPT, build_qa_context, json_hash, load_experimental_settings, safe_configuration,
)

PLAN_SCHEMA = "recipegen-api-qa-smoke-plan-v1"
FIXED_QUESTIONS = (
    ("S01", "G01", "原食谱 Breakfast Tomato Noodles 中，鸡蛋和番茄分别如何处理？", "targeted_actions"),
    ("S02", "G05", "茄盒的馅料和裹糊分别如何处理？请依据原始步骤回答。", "two_processes"),
    ("S03", "G06", "这些候选已排除记录中出现鸡蛋的食谱。咸猪肉和冬瓜应怎样处理？不要作过敏安全保证。", "prefiltered_candidates"),
    ("S04", "G08", "蒸制前，咸猪肉和冬瓜干分别如何切片和处理？请保留来源中的明确数值。", "source_numeric_values"),
    ("S05", "G12", "这些原始步骤是否明确给出了每道菜的总烹饪时长？若证据不足请返回证据不足。", "insufficient_total_duration"),
    ("S06", "G11", "鸡肉黑木耳汤中，原始步骤是否明确给出煮鸡肉的时间？如有请说明对应操作。", "specific_duration"),
)
CODE_FILES = ("recipegen/experimental_qa.py", "scripts/evaluate_api_qa.py", "recipegen/config.py",
              "recipegen/llm.py", "recipegen/graphrag_answer.py", "recipegen/system_generation.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    return {name: file_hash(ROOT / name) for name in CODE_FILES}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(source, output, rerank_count=2):
    if output.exists():
        raise ValueError("output_already_exists")
    original = json.loads(source.read_text(encoding="utf-8"))
    if json_hash(original["frozen_cases"]) != original["frozen_inputs_sha256"]:
        raise ValueError("source_frozen_inputs_hash_mismatch")
    if not original.get("source_and_code_unchanged_after_run"):
        raise ValueError("source_provenance_check_missing")
    frozen = {case["case_id"]: case for case in original["frozen_cases"]}
    cases = []
    for case_id, source_id, question, category in FIXED_QUESTIONS:
        source_case = frozen[source_id]
        ids = list(source_case["eligible_recipe_ids"])
        records = [copy.deepcopy(row) for row in source_case["candidate_records"] if row["id"] in ids]
        context = build_qa_context(question, records, ids)
        cases.append({"case_id": case_id, "source_case_id": source_id, "category": category,
                      "question": question, "candidate_recipe_ids": ids, "raw_records": records,
                      "context_sha256": json_hash(context), "evidence_count": len(context["evidence"]),
                      "candidate_policy": "eligible frozen local-selection candidates, no response-dependent selection",
                      "semantic_review": {"status": "pending", "independently_labeled": False}})
    plan = {"schema": PLAN_SCHEMA, "status": "prepared", "prepared_at": now(),
            "data_mode": "frozen_real_graph_and_original_source_snapshot_not_live_Neo4j",
            "source_report": str(source.resolve()), "source_report_sha256": file_hash(source),
            "source_frozen_inputs_sha256": original["frozen_inputs_sha256"],
            "source_identity": original.get("source_identity"),
            "cases": cases, "cases_sha256": json_hash(cases), "code_identity": code_identity(),
            "qa_prompt_sha256": hashlib.sha256(QA_PROMPT.encode()).hexdigest(),
            "rerank_prompt_sha256": hashlib.sha256(RERANK_PROMPT.encode()).hexdigest(),
            "rerank_case_ids": [case["case_id"] for case in cases[:rerank_count]],
            "maximum_api_attempts": len(cases) + rerank_count,
            "generation_options": {"qa_max_tokens": QA_MAX_TOKENS, "rerank_max_tokens": RERANK_MAX_TOKENS,
                                   "temperature": 0, "stream": False, "response_format": "json_object",
                                   "thinking": "disabled only for api.deepseek.com", "retries": 0},
            "protocol": {"sample_size": 6, "chosen_before_any_api_output": True,
                         "qa_candidate_order": "frozen input order; rerank tested independently",
                         "rerank_failures": "retain frozen candidate order",
                         "qa_failures": "explicit failure status, no fabricated answer",
                         "metrics_scope": "JSON/schema/citation-ID/numeric-literal checks only",
                         "semantic_accuracy_evaluated": False, "retrieval_accuracy_evaluated": False,
                         "local_Qwen_used": False, "new_retrieval_or_embeddings": False}}
    write_json(output, plan)
    return {"status": "prepared", "plan": str(output.resolve()), "qa_cases": 6,
            "rerank_cases": rerank_count, "maximum_api_attempts": 6 + rerank_count,
            "cases_sha256": plan["cases_sha256"]}


def summary(report):
    qa = [case["qa"] for case in report["results"] if case.get("qa")]
    ranks = [case["rerank"] for case in report["results"] if case.get("rerank")]
    return {"qa_completed": len(qa), "qa_requested": 6,
            "qa_api_attempts": sum(x["generation"]["api_call_attempted"] for x in qa),
            "qa_structural_valid": sum(x["validation"]["schema_citation_number_checks_passed"] for x in qa),
            "qa_answered": sum(x["status"] == "answered" for x in qa),
            "qa_insufficient_evidence": sum(x["status"] == "insufficient_evidence" for x in qa),
            "qa_failed": sum(x["status"] not in {"answered", "insufficient_evidence"} for x in qa),
            "rerank_completed": len(ranks), "rerank_api_attempts": sum(x["api_call_attempted"] for x in ranks),
            "rerank_valid_permutations": sum(x["validation"]["candidate_permutation_passed"] for x in ranks),
            "rerank_fallbacks": sum(x["status"] == "fallback" for x in ranks),
            "metrics_are_structural_not_semantic": True, "semantic_review": "pending"}


def write_review(path, report, plan):
    questions = {case["case_id"]: case for case in plan["cases"]}
    lines = ["# 固定候选 API 问答 smoke", "", "6 题原始来源问答及最多 2 次独立重排；候选已冻结。",
             "结构化校验仅覆盖 schema、所给证据 ID 与字面数值，不等于语义支持或准确率。",
             "本次不运行本地 Qwen，不改变 E5 检索。人工语义审查尚未完成。", "",
             "```json", json.dumps(report["summary"], ensure_ascii=False, indent=2), "```", ""]
    for result in report["results"]:
        case = questions[result["case_id"]]
        lines.extend([f"## {case['case_id']} {case['question']}", ""])
        qa = result.get("qa")
        if qa:
            lines.extend([f"状态：`{qa['status']}`；耗时：{qa['elapsed_ms']} ms。", "", qa["answer"] or "未产生通过校验的最终回答。", "",
                          "原始解析输出（校验失败时也保存；上游非 JSON 错误不会回显正文）：", "```json",
                          json.dumps(qa["generation"]["response_json"], ensure_ascii=False, indent=2), "```", "",
                          "来源步骤证据：", ""])
            for evidence in qa["evidence"]:
                lines.append(f"- {evidence['id']} / {evidence['recipe_id']} / {evidence['step_id']} / {evidence['source_id']}：{evidence['original_text']}")
        if result.get("rerank"):
            lines.extend(["", "独立重排：", "```json", json.dumps(result["rerank"], ensure_ascii=False, indent=2), "```"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_plan_budget(plan, max_calls):
    cases, reranks = plan.get("cases"), plan.get("rerank_case_ids")
    if not isinstance(cases, list) or len(cases) != 6 or not isinstance(reranks, list):
        raise ValueError("invalid_case_budget")
    identifiers = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if (len(identifiers) != 6 or any(not isinstance(x, str) or not x for x in identifiers)
            or len(set(identifiers)) != 6 or len(reranks) > 2
            or any(not isinstance(x, str) or x not in identifiers for x in reranks)
            or len(set(reranks)) != len(reranks)):
        raise ValueError("invalid_case_budget")
    actual_maximum = len(cases) + len(reranks)
    if plan.get("maximum_api_attempts") != actual_maximum or actual_maximum > min(max_calls, 8):
        raise ValueError("maximum_api_attempts_exceeded")


def run(plan_path, output, max_calls=8, env_file=None):
    if output.exists() or output.with_suffix(".md").exists():
        raise ValueError("output_already_exists")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "prepared":
        raise ValueError("prepared_plan_required")
    if (json_hash(plan["cases"]) != plan["cases_sha256"] or code_identity() != plan["code_identity"]
            or file_hash(Path(plan["source_report"])) != plan["source_report_sha256"]):
        raise ValueError("frozen_plan_source_or_code_changed")
    validate_plan_budget(plan, max_calls)
    for case in plan["cases"]:
        if json_hash(build_qa_context(case["question"], case["raw_records"], case["candidate_recipe_ids"])) != case["context_sha256"]:
            raise ValueError("case_context_changed")
    settings = load_experimental_settings(env_file)
    service = ExperimentalEvidenceQA(settings=settings, allow_api_calls=True)
    if not settings.llm_configured or settings.llm_provider != "chat_completions":
        print(json.dumps(safe_configuration(settings), ensure_ascii=False), flush=True)
        raise ValueError("configured_chat_completions_API_required")
    report = {"schema": "recipegen-api-qa-smoke-result-v1", "status": "running", "started_at": now(),
              "plan": str(plan_path.resolve()), "plan_sha256": file_hash(plan_path),
              "cases_sha256": plan["cases_sha256"], "configuration": safe_configuration(settings),
              "generation_options": plan["generation_options"], "code_identity": plan["code_identity"],
              "protocol": plan["protocol"], "results": [], "successful_json_api_calls": []}
    report["summary"] = summary(report)
    write_json(output, report)
    for case in plan["cases"]:
        result = {"case_id": case["case_id"], "question": case["question"],
                  "candidate_recipe_ids": case["candidate_recipe_ids"]}
        report["results"].append(result)
        result["qa"] = service.answer(case["question"], case["raw_records"], case["candidate_recipe_ids"])
        if case["case_id"] in plan["rerank_case_ids"]:
            result["rerank"] = service.rerank(case["question"], case["raw_records"], case["candidate_recipe_ids"])
        report["successful_json_api_calls"] = copy.deepcopy(service.llm_client.calls)
        report["summary"] = summary(report)
        write_json(output, report)
        write_review(output.with_suffix(".md"), report, plan)
        print(json.dumps({"case_id": case["case_id"], "qa_status": result["qa"]["status"],
                          "rerank_status": result.get("rerank", {}).get("status"),
                          "qa_elapsed_ms": result["qa"]["elapsed_ms"]}, ensure_ascii=False), flush=True)
    report["status"], report["finished_at"] = "finished", now()
    write_json(output, report)
    write_review(output.with_suffix(".md"), report, plan)
    return {"status": "finished", "result": str(output.resolve()), **report["summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--diagnose", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--source", type=Path, default=ROOT / "reports/local-selection-v1.json")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--env-file", type=Path, help="Explicit configuration file; never searched or printed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rerank-count", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--max-calls", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    try:
        if args.diagnose:
            result = safe_configuration(load_experimental_settings(args.env_file))
        elif args.prepare:
            if not args.output:
                parser.error("--prepare requires --output")
            result = prepare(args.source, args.output, args.rerank_count)
        else:
            if not args.plan or not args.output:
                parser.error("--run requires --plan and --output")
            result = run(args.plan, args.output, args.max_calls, args.env_file)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        # No upstream exception, URL or settings repr may reach terminal logs.
        print(json.dumps({"status": "failed", "error": "inspect_nonsecret_inputs_or_configuration"}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
