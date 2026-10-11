"""Question-answering contracts use mock HTTP only; no credentials/network/model."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from recipegen.config import Settings
from recipegen.experimental_qa import (
    ExperimentalEvidenceQA, ExperimentalLLMClient, ExperimentalQAError,
    QA_MAX_TOKENS, RERANK_MAX_TOKENS, build_qa_context,
    load_experimental_settings, safe_configuration, validate_qa_output,
)

KEY = "PRIVATE-CONTRACT-KEY"
IDS = ["recipe:soup", "recipe:tea"]


@pytest.fixture
def records():
    return [{"id": IDS[0], "title": "Tomato Soup", "sources": [{"source_id": "source:soup"}],
             "steps": [{"id": "step:heat", "order": 1, "text": "1. Heat water for 5 minutes.", "source_id": "source:soup"},
                       {"id": "step:tomato", "order": 2, "text": "2. Add tomatoes and stir.", "source_id": "source:soup"}]},
            {"id": IDS[1], "title": "Tea", "source_ids": ["source:tea"],
             "steps": [{"id": "step:tea", "order": 1, "text": "Steep tea for 10 minutes.", "source_id": "source:tea"}]}]


def output(text="将水加热5分钟。", ids=None):
    return {"status": "answered", "paragraphs": [{"text": text, "evidence_ids": ids or ["E1"]}]}


def response(value, *, finish="stop", raw=False):
    content = value if raw else json.dumps(value, ensure_ascii=False)
    return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}],
                                    "usage": {"prompt_tokens": 17, "completion_tokens": 13, "secret": KEY}})


def service(handler, *, base="https://api.deepseek.com", allowed=True, **options):
    settings = Settings(llm_provider="chat_completions", llm_base_url=base, llm_model="contract-only", llm_api_key=KEY, **options)
    client = ExperimentalLLMClient(settings, transport=httpx.MockTransport(handler))
    return ExperimentalEvidenceQA(settings=settings, llm_client=client, allow_api_calls=allowed), client


def test_partial_question_answer_uses_source_evidence_not_full_recipe_rewrite(records):
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        assert request.headers["authorization"] == "Bearer " + KEY
        assert KEY not in json.dumps(body)
        assert body["max_tokens"] == QA_MAX_TOKENS
        assert body["thinking"] == {"type": "disabled"}
        assert body["temperature"] == 0 and body["stream"] is False
        assert body["response_format"] == {"type": "json_object"}
        context = json.loads(body["messages"][1]["content"])
        assert context["evidence"][0]["original_text"] == records[0]["steps"][0]["text"]
        return response(output())
    original = copy.deepcopy(records)
    qa, client = service(handler)
    result = qa.answer("水加热多久？", records, IDS)
    assert result["status"] == "answered" and result["answer"] == "将水加热5分钟。 [E1]"
    assert len(result["paragraphs"]) == 1 and len(result["evidence"]) == 3
    assert records == original and len(captured) == len(client.calls) == 1
    assert result["validation"] == {"schema_citation_number_checks_passed": True, "semantic_evaluated": False}
    assert client.calls[0]["usage"] == {"prompt_tokens": 17, "completion_tokens": 13}
    assert KEY not in json.dumps(result) + json.dumps(client.calls)


@pytest.mark.parametrize("wording", ["另起锅热油。", "另起一口锅热油。", "另用一口锅热油。", "换锅热油。"])
def test_extra_cookware_needs_explicit_support_in_cited_source(records, wording):
    records[0]["steps"][0]["text"] = "Heat oil in a pot."
    records[0]["steps"][1]["text"] = "Use another pot for tomatoes."
    context = build_qa_context("关键做法？", records, IDS)
    with pytest.raises(ExperimentalQAError, match="unsupported_separate_cookware"):
        validate_qa_output(output(wording, ["E1"]), context)
    # An uncited step mentioning another pot cannot validate the assertion.
    assert validate_qa_output(output("锅中热油。", ["E1"]), context)
    assert validate_qa_output(output("另用一口锅处理番茄。", ["E2"]), context)


def test_unsupported_extra_cookware_is_not_a_successful_api_answer(records):
    records[0]["steps"][0]["text"] = "Heat oil in a pot."
    qa, _ = service(lambda _: response(output("另起锅热油。")))
    result = qa.answer("关键做法？", records, IDS)
    assert result["status"] == "validation_failed"
    assert result["generation"]["error_code"] == "unsupported_separate_cookware"
    assert result["answer"] is None


def test_proxy_does_not_get_provider_specific_thinking_option(records):
    def handler(request):
        assert "thinking" not in json.loads(request.content)
        return response(output())
    qa, _ = service(handler, base="https://proxy.invalid/v1")
    assert qa.answer("加热多久？", records, IDS)["status"] == "answered"


def test_cited_step_ordinals_are_not_invented_cooking_quantities():
    records = [{"id": "recipe:chicken", "title": "Chicken Soup", "source_ids": ["source:chicken"],
                "steps": [{"id": "step:parboil", "order": 4,
                           "text": "4. Parboil chicken for 8 minutes.", "source_id": "source:chicken"},
                          {"id": "step:simmer", "order": 7,
                           "text": "7. Simmer for 40 minutes.", "source_id": "source:chicken"}]}]
    context = build_qa_context("鸡肉煮多久？", records, ["recipe:chicken"])
    answer = {"status": "answered", "paragraphs": [
        {"text": "原文第4步说明鸡肉煮8分钟。", "evidence_ids": ["E1"]},
        {"text": "后续第7步说明再煮40分钟。", "evidence_ids": ["E2"]}]}
    assert validate_qa_output(answer, context) == answer["paragraphs"]


@pytest.mark.parametrize("text,ids,error", [
    ("原文第8步说明鸡肉煮8分钟。", ["E1"], "unsupported_step_reference"),
    ("原文第4步说明再煮40分钟。", ["E2"], "unsupported_step_reference"),
    ("原文第7步说明再煮7分钟。", ["E2"], "unsupported_numeric_literal"),
])
def test_step_reference_does_not_bypass_wrong_order_or_duration(text, ids, error):
    records = [{"id": "recipe:chicken", "title": "Chicken Soup", "source_ids": ["source:chicken"],
                "steps": [{"id": "step:parboil", "order": 4,
                           "text": "4. Parboil chicken for 8 minutes.", "source_id": "source:chicken"},
                          {"id": "step:simmer", "order": 7,
                           "text": "7. Simmer for 40 minutes.", "source_id": "source:chicken"}]}]
    context = build_qa_context("鸡肉煮多久？", records, ["recipe:chicken"])
    with pytest.raises(ExperimentalQAError, match=error):
        validate_qa_output(output(text, ids), context)


@pytest.mark.parametrize("invalid,error", [
    (output(ids=["E99"]), "unknown_or_duplicate_citation"),
    (output(ids=["E1", "E1"]), "unknown_or_duplicate_citation"),
    (output("将水加热10分钟。"), "unsupported_numeric_literal"),
    (output("加入番茄后煮5分钟。", ["E2"]), "unsupported_numeric_literal"),
    (output("10. 将水加热。"), "invalid_chinese_paragraph"),
    (output("将水加热5分钟。[E1]"), "invalid_chinese_paragraph"),
    ({**output(), "extra": "field"}, "invalid_schema"),
    ({"status": [], "paragraphs": []}, "invalid_answer_status"),
    ({"status": "insufficient_evidence", "paragraphs": output()["paragraphs"]}, "insufficient_evidence_must_be_empty"),
])
def test_rejected_outputs_keep_safe_raw_json_for_review_and_do_not_answer(records, invalid, error):
    qa, _ = service(lambda _: response(invalid))
    result = qa.answer("怎么处理？", records, IDS)
    assert result["status"] == "validation_failed" and result["answer"] is None
    assert result["generation"]["error_code"] == error
    assert result["generation"]["response_json"] == invalid
    assert not result["validation"]["schema_citation_number_checks_passed"]


def test_missing_evidence_is_explicit_valid_protocol_not_a_fabricated_answer(records):
    qa, _ = service(lambda _: response({"status": "insufficient_evidence", "paragraphs": []}))
    result = qa.answer("总热量多少？", records, IDS)
    assert result["status"] == "insufficient_evidence" and result["paragraphs"] == []
    assert result["answer"] == "现有步骤证据不足以回答这个问题。"


def test_structural_checks_do_not_claim_semantic_support(records):
    context = build_qa_context("有什么用？", records, IDS)
    # Valid IDs/no new numbers cannot prove this unsupported medical claim.
    assert validate_qa_output(output("这个做法可以治愈疾病。"), context)
    qa, _ = service(lambda _: response(output("这个做法可以治愈疾病。")))
    assert qa.answer("有什么用？", records, IDS)["validation"]["semantic_evaluated"] is False


@pytest.mark.parametrize("base", ["https://user:secret@api.invalid", "https://api.invalid?key=secret", "https://api.invalid#secret"])
def test_credential_url_is_rejected_without_request(records, base):
    qa, client = service(lambda _: pytest.fail("must not request"), base=base)
    result = qa.answer("怎么做？", records, IDS)
    assert result["status"] == "invalid_api_configuration" and not client.calls
    assert "secret" not in json.dumps(result)


def test_explicit_optin_and_config_required(records):
    qa, client = service(lambda _: pytest.fail("must not request"), allowed=False)
    assert qa.answer("怎么做？", records, IDS)["status"] == "dry_run" and not client.calls
    qa = ExperimentalEvidenceQA()
    assert qa.answer("怎么做？", records, IDS)["status"] == "not_configured"
    assert set(safe_configuration(Settings())) == {"configured", "provider", "model"}


@pytest.mark.parametrize("mutation", ["unknown_source", "duplicate_step", "bad_source_list", "missing_candidate", "none_ids"])
def test_bad_provenance_prevents_api(records, mutation):
    candidate_ids = IDS
    if mutation == "unknown_source": records[0]["steps"][0]["source_id"] = "not-owned"
    elif mutation == "duplicate_step": records[0]["steps"][1]["id"] = "step:heat"
    elif mutation == "bad_source_list": records[0]["source_ids"] = [{}]
    elif mutation == "missing_candidate": candidate_ids = ["missing"]
    else: candidate_ids = None
    qa, _ = service(lambda _: pytest.fail("must not request"))
    result = qa.answer("怎么做？", records, candidate_ids)
    assert result["status"] == "context_error" and not result["generation"]["api_call_attempted"]


def test_sensitive_input_and_output_never_persist_credentials(records):
    qa, _ = service(lambda _: pytest.fail("must not request"))
    records[0]["steps"][0]["text"] = KEY
    result = qa.answer("怎么做？", records, IDS)
    assert result["status"] == "sensitive_input" and KEY not in json.dumps(result)
    qa, _ = service(lambda _: response(output("不能泄漏" + KEY)))
    records[0]["steps"][0]["text"] = "Heat water."
    result = qa.answer("怎么做？", records, IDS)
    assert result["status"] == "validation_failed" and result["generation"]["error_code"] == "sensitive_output"
    assert result["generation"]["response_json"] is None and KEY not in json.dumps(result)


def test_secret_in_candidate_id_is_not_returned_even_if_context_is_invalid(records):
    qa, _ = service(lambda _: pytest.fail("must not request"))
    for result in (qa.answer("怎么做？", records, [KEY]), qa.rerank("怎么做？", records, [KEY])):
        assert KEY not in json.dumps(result)


def test_explicit_config_file_and_deepseek_aliases_are_experimental_only(tmp_path, monkeypatch):
    keys = ("RECIPEGEN_LLM_API_KEY", "RECIPEGEN_LLM_PROVIDER", "RECIPEGEN_LLM_BASE_URL", "RECIPEGEN_LLM_MODEL",
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
    for key in keys: monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: Settings()))
    path = tmp_path / "explicit.env"
    path.write_text("DEEPSEEK_API_KEY=" + KEY + "\n")
    # Ensure load_dotenv mutations are reverted by the monkeypatch fixture.
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    settings = load_experimental_settings(path)
    assert settings.llm_configured and settings.llm_model == "deepseek-flash"
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert KEY not in json.dumps(safe_configuration(settings))
    monkeypatch.setenv("RECIPEGEN_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("RECIPEGEN_LLM_MODEL", "explicit-model")
    assert load_experimental_settings(path).llm_model == "explicit-model"
    assert not load_experimental_settings(path).llm_configured


@pytest.mark.parametrize("value,finish", [
    ('{"status":"answered","status":"insufficient_evidence","paragraphs":[]}', "stop"),
    ('{"status":"answered","paragraphs":NaN}', "stop"),
    ('```json\n{"status":"insufficient_evidence","paragraphs":[]}\n```', "stop"),
    ('{"status":"insufficient_evidence","paragraphs":[]}', "length"),
    ("", "stop"),
])
def test_bad_wire_json_or_truncation_is_safe_upstream_error(records, value, finish):
    qa, _ = service(lambda _: response(value, raw=True, finish=finish))
    result = qa.answer("怎么做？", records, IDS)
    assert result["status"] == "upstream_error" and result["generation"]["response_json"] is None


@pytest.mark.parametrize("failure", ["http", "timeout"])
def test_http_failure_does_not_echo_body_url_or_exception(records, failure):
    def handler(request):
        if failure == "timeout": raise httpx.ReadTimeout("https://api.invalid?key=" + KEY, request=request)
        return httpx.Response(401, text="secret=" + KEY)
    qa, _ = service(handler)
    result = qa.answer("怎么做？", records, IDS)
    assert result["status"] == "upstream_error" and KEY not in json.dumps(result)


def test_rerank_is_complete_permutation_with_bounded_options(records):
    def handler(request):
        assert json.loads(request.content)["max_tokens"] == RERANK_MAX_TOKENS
        return response({"ordered_recipe_ids": IDS[::-1]})
    qa, _ = service(handler)
    result = qa.rerank("更想喝茶", records, IDS)
    assert result["status"] == "ok" and result["ordered_recipe_ids"] == IDS[::-1]
    assert not result["retained_input_order"] and result["validation"]["semantic_evaluated"] is False


@pytest.mark.parametrize("invalid", [{"ordered_recipe_ids": [IDS[0], IDS[0]]}, {"ordered_recipe_ids": ["unknown", IDS[0]]},
                                    {"ordered_recipe_ids": [IDS[0]]}, {"ordered_recipe_ids": IDS[::-1], "extra": 1}])
def test_rerank_failure_keeps_rrf_input_order(records, invalid):
    qa, _ = service(lambda _: response(invalid))
    result = qa.rerank("怎么做？", records, IDS)
    assert result["status"] == "fallback" and result["ordered_recipe_ids"] == IDS
    assert result["retained_input_order"] and result["response_json"] == invalid


def test_fixed_smoke_prepare_is_reproducible_and_has_no_api_calls(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("api_qa_smoke", root / "scripts/evaluate_api_qa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the public unit test self-contained; real source snapshots are private.
    frozen_cases = []
    for _, source_id, _, _ in module.FIXED_QUESTIONS:
        candidates = [
            {"id": f"recipe:fixture:{source_id}:{index}", "title": f"Synthetic recipe {index}",
             "source_ids": [f"source:fixture:{source_id}:{index}"],
             "steps": [{"id": f"step:fixture:{source_id}:{index}", "order": 1,
                        "text": "Heat water for 5 minutes.",
                        "source_id": f"source:fixture:{source_id}:{index}"}]}
            for index in range(3)
        ]
        frozen_cases.append({"case_id": source_id, "eligible_recipe_ids": [r["id"] for r in candidates],
                             "candidate_records": candidates})
    source = tmp_path / "synthetic-source.json"
    source.write_text(json.dumps({"frozen_cases": frozen_cases,
                                  "frozen_inputs_sha256": module.json_hash(frozen_cases),
                                  "source_and_code_unchanged_after_run": True,
                                  "source_identity": {"kind": "synthetic_unit_test", "is_demo": True}}),
                      encoding="utf-8")
    source_bytes = source.read_bytes()
    monkeypatch.setattr(module, "ExperimentalEvidenceQA", lambda *args, **kwargs: pytest.fail("must not create API service"))
    monkeypatch.setattr(httpx.Client, "send", lambda *args, **kwargs: pytest.fail("must not send HTTP request"))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    a = module.prepare(source, first)
    b = module.prepare(source, second)
    plan = json.loads(first.read_text())
    assert a["cases_sha256"] == b["cases_sha256"] and a["maximum_api_attempts"] == 8
    assert len(plan["cases"]) == 6 and plan["rerank_case_ids"] == ["S01", "S02"]
    assert all(len(c["candidate_recipe_ids"]) >= 3 for c in plan["cases"])
    assert all(c["evidence_count"] == 3 for c in plan["cases"])
    assert plan["source_frozen_inputs_sha256"] == module.json_hash(frozen_cases)
    assert plan["source_report_sha256"] == module.file_hash(source) and source.read_bytes() == source_bytes
    assert plan["protocol"]["semantic_accuracy_evaluated"] is False
    with pytest.raises(ValueError, match="output_already_exists"):
        module.prepare(source, first)
    for changed in (
        {**plan, "maximum_api_attempts": 6},
        {**plan, "rerank_case_ids": ["unknown"]},
        {**plan, "rerank_case_ids": ["S01", "S01"]},
        {**plan, "cases": [plan["cases"][0]] * 6},
    ):
        with pytest.raises(ValueError):
            module.validate_plan_budget(changed, 8)
