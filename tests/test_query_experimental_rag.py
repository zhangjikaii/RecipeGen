"""Small CLI orchestration contracts; no model or API evaluation."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.query_experimental_rag import (
    FrozenExperimentalIndex, FrozenIndexError, plan_question, run_question,
)


def document(rid):
    return {"id": rid, "recipe_id": rid, "title": "Original " + rid,
            "source_ids": ["source:" + rid], "sources": [],
            "steps": [{"id": "step:" + rid, "source_id": "source:" + rid,
                       "order": 1, "text": "Fry onion for 3 minutes."}]}


class Index:
    root = "/unused"
    summary = {"scope": "fixture_only"}

    def __init__(self):
        self.documents = {rid: document(rid) for rid in ("r1", "r2")}
        self.documents["title"] = {"id": "title", "title": "Actual title", "steps": [],
                                   "source_ids": ["source:title"]}

    def retrieve(self, plan, vector):
        return {"rankings": [{"recipe_id": rid} for rid in ("title", "r1", "r2")]}


class Encoder:
    def __init__(self):
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return [1.] + [0.] * 383


def test_plan_normalizes_conditions_but_preserves_original_English_recipe_action_words():
    q = "How do I fry Onion Soup with 洋葱 without milk?"
    plan = plan_question(q, ingredients=["西红柿"])
    assert plan["ingredients"] == ["tomato", "onion"]
    assert plan["excluded_ingredients"] == ["milk"]
    assert {"fry", "onion", "soup", "tomato"} <= set(plan["lexical_tokens"])
    assert "milk" not in plan["lexical_tokens"]
    assert plan["e5_query"] == q and plan["api_question_parsing"] is False


def test_quote_is_derived_only_from_visible_question_and_negative_is_removed():
    plan = plan_question('Find "Stir onion gently" without milk.')
    assert plan["quoted_passages"] == ["Stir onion gently"]
    assert {"stir", "onion", "gently"} <= set(plan["lexical_tokens"])
    assert "milk" not in plan["lexical_tokens"]


def test_default_retrieval_never_calls_api_and_keeps_full_source_ids():
    encoder = Encoder()
    qa = SimpleNamespace(rerank=lambda *a: pytest.fail("unexpected billed rerank"),
                         answer=lambda *a: pytest.fail("unexpected billed answer"))
    result = run_question(Index(), "请找洋葱食谱", embedder=encoder, qa=qa)
    assert encoder.questions == ["请找洋葱食谱"]
    assert result["status"] == "retrieval_only"
    assert result["selected_recipe_ids"] == ["r1", "r2"]
    assert [e["source_id"] for e in result["evidence"]] == ["source:r1", "source:r2"]
    assert [e["step_id"] for e in result["evidence"]] == ["step:r1", "step:r2"]
    assert result["source_documents"][0]["steps"] == document("r1")["steps"]
    assert result["title_only_materials"][0]["source_ids"] == ["source:title"]
    assert result["rerank"]["api_call_attempted"] is False


def test_explicit_api_rerank_then_answer_failure_is_material_display_not_success():
    calls = []

    class QA:
        def status(self):
            return {"configured": True, "provider": "fixture"}

        def rerank(self, question, docs, ids):
            calls.append(("rerank", ids[:]))
            return {"status": "ok", "ordered_recipe_ids": list(reversed(ids)), "api_call_attempted": True}

        def answer(self, question, docs, ids):
            calls.append(("answer", ids[:]))
            return {"status": "upstream_error", "generation": {"api_call_attempted": True}}

    result = run_question(Index(), "Find onion", run_api=True, embedder=Encoder(), qa=QA())
    assert calls == [("rerank", ["r1", "r2"]), ("answer", ["r2", "r1"])]
    assert result["status"] == "source_materials_only"
    assert result["selected_recipe_ids"] == ["r2", "r1"]
    assert result["evidence"][0]["source_id"] == "source:r2"
    assert result["original_steps_complete"] is True


def test_vector_dimension_dtype_and_normalization_are_checked():
    index = object.__new__(FrozenExperimentalIndex)
    index.np = np
    valid = np.zeros((2, 384), dtype=np.float32)
    valid[:, 0] = 1
    index._validate_matrix(valid, 2)
    with pytest.raises(FrozenIndexError, match="shape"):
        index._validate_matrix(valid[:, :383], 2)
    with pytest.raises(FrozenIndexError, match="shape"):
        index._validate_matrix(valid.astype(np.float64), 2)
    with pytest.raises(FrozenIndexError, match="normalized"):
        index._validate_matrix(valid * 0, 2)
