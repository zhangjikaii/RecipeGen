"""Synthetic unit contracts only; these are not semantic retrieval evaluations."""
from __future__ import annotations

import math

import pytest

from recipegen.experimental_retrieval import (
    BM25Index, GraphCandidateIndex, aggregate_chunk_scores, build_recipe_documents,
    build_step_chunks, cosine_scores, lexical_tokens, pool_rrf,
)
from recipegen.test_graph import REPO, uid


class CharacterTokenizer:
    """Budget fixture with two special tokens and exact Unicode characters."""
    def encode(self, text, *, add_special_tokens=True, truncation=False):
        assert truncation is False
        return [ord(c) for c in text] + ([0, 1] if add_special_tokens else [])


def raw_record(texts, title="Soup", rid="r1"):
    return {"id": rid, "title": title,
            "steps": [{"order": i, "text": text} for i, text in enumerate(texts, 1)],
            "source": {"archive": "test.zip", "goal_member": f"{rid}/goal.txt",
                       "steps_member": f"{rid}/steps.txt", "revision": "fixture",
                       "goal_sha256": "a" * 64, "steps_sha256": "b" * 64}}


def test_documents_keep_original_text_and_use_actual_graph_id_scheme():
    raw = raw_record(["  Chop onion.\t", "Do not add milk.\n"])
    doc = build_recipe_documents({raw["id"]: raw})[0]
    source_id = uid("source", f"{REPO}\ntest.zip\nr1/steps.txt")
    assert doc["text"] == "Soup\n  Chop onion.\t\nDo not add milk.\n"
    assert doc["steps"][0]["id"] == uid("step", "r1\n1")
    assert doc["steps"][0]["source_id"] == source_id
    assert source_id in doc["source_ids"]
    assert build_recipe_documents([doc])[0]["steps"] == doc["steps"]
    assert raw["steps"][0] == {"order": 1, "text": "  Chop onion.\t"}


def test_long_step_spans_reconstruct_unicode_and_whitespace_losslessly():
    tokenizer = CharacterTokenizer()
    original = "  Mix 🥕，加入盐。\t" * 13 + "\nfinish  "
    raw = raw_record([original, "serve"])
    chunks = build_step_chunks([raw], tokenizer, max_tokens=40)
    step_id = uid("step", "r1\n1")
    fragments = [f for chunk in chunks for f in chunk["fragments"] if f["step_id"] == step_id]
    assert len(fragments) > 3
    assert "".join(f["text"] for f in fragments) == original
    assert fragments[0]["span_start"] == 0 and fragments[-1]["span_end"] == len(original)
    assert all(a["span_end"] == b["span_start"] for a, b in zip(fragments, fragments[1:]))
    for chunk in chunks:
        assert chunk["embedding_text"] == "passage: " + chunk["text"]
        assert chunk["token_count"] == len(tokenizer.encode(chunk["embedding_text"])) <= 40
        for fragment in chunk["fragments"]:
            text = raw["steps"][fragment["order"] - 1]["text"]
            assert fragment["text"] == text[fragment["span_start"]:fragment["span_end"]]
            assert fragment["source_id"] in chunk["source_ids"]
    assert chunks == build_step_chunks([raw], tokenizer, max_tokens=40)


def test_chunks_group_adjacent_steps_without_overlaps_or_cross_recipe_sources():
    tokenizer = CharacterTokenizer()
    chunks = build_step_chunks([raw_record(["a", "b", "c", "d"]), raw_record(["z"], rid="r2")],
                               tokenizer, max_tokens=100)
    assert [len(c["fragments"]) for c in chunks] == [3, 1, 1]
    assert [[f["order"] for f in c["fragments"]] for c in chunks] == [[1, 2, 3], [4], [1]]
    assert len({f["step_id"] for c in chunks for f in c["fragments"]}) == 5
    assert chunks[0]["source_ids"] != chunks[-1]["source_ids"]


def test_exceptional_title_is_explicitly_shortened_only_in_projection():
    raw = raw_record(["original step text"], title="long title " * 30)
    chunks = build_step_chunks([raw], CharacterTokenizer(), max_tokens=48)
    assert all(c["title"] == raw["title"] and c["title_context_truncated"] for c in chunks)
    assert "".join(f["text"] for c in chunks for f in c["fragments"]) == "original step text"
    assert all(c["token_count"] <= 48 for c in chunks)


def test_no_steps_retains_real_title_only_unit_without_invented_step_or_source():
    raw = raw_record([], title="Original title")
    raw["source"]["steps_member"] = None
    expected_source = uid("source", f"{REPO}\ntest.zip\nr1/goal.txt")
    chunks = build_step_chunks([raw], CharacterTokenizer(), max_tokens=48)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["unit"] == "title_only" and chunk["text"] == raw["title"]
    assert chunk["fragments"] == [] and chunk["step_ids"] == []
    assert chunk["source_ids"] == [expected_source] and chunk["token_count"] <= 48
    corpus = {"recipe_id": "r1", "title": raw["title"], "steps": [], "source_ids": [expected_source]}
    assert build_step_chunks([corpus], CharacterTokenizer(), max_tokens=48)[0] == chunk
    with pytest.raises(ValueError, match="title Source"):
        build_step_chunks([{**corpus, "source_ids": []}], CharacterTokenizer(), max_tokens=48)
    with pytest.raises(ValueError, match="title Source"):
        build_step_chunks([{**corpus, "source_ids": ["source:a", "source:b"]}], CharacterTokenizer(), max_tokens=48)


def test_source_or_order_errors_are_not_silently_indexed():
    raw = raw_record(["step"])
    raw["steps"][0]["order"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        build_recipe_documents([raw])
    with pytest.raises(ValueError, match="Source"):
        build_recipe_documents([{"id": "r", "title": "t", "steps": [{"order": 1, "text": "x"}]}])
    with pytest.raises(ValueError, match="unique"):
        build_recipe_documents([raw_record([]), raw_record([])])


def test_positive_idf_okapi_has_expected_numeric_scores_and_zero_for_no_match():
    index = BM25Index([{"id": "a", "text": "onion onion"}, {"id": "b", "text": "onion milk"},
                       {"id": "c", "text": "rice"}])
    idf = math.log1p((3 - 2 + .5) / (2 + .5))
    norm = 1.5 * (1 - .75 + .75 * 2 / (5 / 3))
    scores = index.scores("onion")
    assert scores["a"] == pytest.approx(idf * 2 * 2.5 / (2 + norm))
    assert scores["b"] == pytest.approx(idf * 1 * 2.5 / (1 + norm))
    assert scores["c"] == 0 and index.scores("unseen") == {"a": 0, "b": 0, "c": 0}
    assert index.scores("onion onion") == scores  # Explicit binary query term frequency.
    assert index.scores(["onion"], ids={"b"}) == {"b": scores["b"]}
    assert index.metadata["implementation"] == "project_positive_idf_okapi"
    common = BM25Index([{"id": "a", "text": "salt"}, {"id": "b", "text": "salt"}])
    assert all(score > 0 for score in common.scores("salt").values())
    assert BM25Index([]).scores("x") == {}


def test_lexical_rules_are_explicit_and_do_not_stem_or_use_graph_gold_aliases():
    assert lexical_tokens("ＣＡＲＲＯＴＳ, carrots carrot 洋葱") == ["carrots", "carrots", "carrot", "洋", "葱"]
    assert BM25Index([{"id": "x", "text": "carrots"}]).scores("carrot")["x"] == 0


def graph_fixture():
    nodes = [{"id": r, "label": "Recipe", "properties": {}} for r in ("r1", "r2", "r3")]
    for r in ("r1", "r2", "r3"):
        nodes += [{"id": "s" + r, "label": "Step", "properties": {"order": 1, "recipe_id": r, "source_id": "src" + r}},
                  {"id": "src" + r, "label": "Source", "properties": {}}]
    nodes += [{"id": "i1", "label": "Ingredient", "properties": {"normalized_name": "onion", "zh_name": "洋葱"}},
              {"id": "i2", "label": "Ingredient", "properties": {"normalized_name": "milk"}},
              {"id": "a1", "label": "Action", "properties": {"normalized_name": "fry"}},
              {"id": "t1", "label": "Tool", "properties": {"normalized_name": "pan"}}]
    def edge(start, kind, end):
        return {"start_id": start, "end_id": end, "type": kind, "properties": {}}
    edges = [edge(r, "HAS_STEP", "s" + r) for r in ("r1", "r2", "r3")]
    edges += [edge("r1", "HAS_INGREDIENT", "i1"), edge("r2", "HAS_INGREDIENT", "i1"),
              edge("sr2", "USES_INGREDIENT", "i2"), edge("sr1", "HAS_ACTION", "a1"),
              edge("sr1", "USES_TOOL", "t1"), edge("r3", "SUBSTITUTES", "i1")]
    return nodes, edges


def test_graph_candidates_use_actual_relation_generators_and_never_invent_rank():
    nodes, edges = graph_fixture()
    graph = GraphCandidateIndex(iter(nodes), iter(reversed(edges)))  # No edge-order dependence.
    assert graph.candidates() == set() and graph.eligible_ids() == {"r1", "r2", "r3"}
    assert graph.candidates(ingredients=["洋葱"], excluded=["milk"]) == {"r1"}
    assert graph.candidates(actions=["fry"], tools=["pan"]) == {"r1"}
    assert graph.candidates(ingredients=["unknown"]) == set()
    assert graph.candidates(ingredients=["onion"]) == {"r1", "r2"}
    paths = graph.evidence_paths("r1")
    assert any(p["relations"] == ["HAS_STEP", "HAS_ACTION"] and p["source_id"] == "srcr1" for p in paths)
    assert graph.semantics == "dictionary_text_mentions_only_not_verified_use"


def test_graph_rejects_conflicting_recipe_ownership_or_build_scopes():
    nodes, edges = graph_fixture()
    edges.append({"start_id": "r2", "type": "HAS_STEP", "end_id": "sr1"})
    with pytest.raises(ValueError, match="two Recipes"):
        GraphCandidateIndex(iter(nodes), iter(edges))
    nodes[0]["properties"]["build_id"] = "other"
    with pytest.raises(ValueError, match="mix"):
        GraphCandidateIndex(iter(nodes), iter(edges))


def test_true_cosine_and_chunk_max_aggregate_before_recipe_candidate_selection():
    scores = cosine_scores([2, 0], {"c1": [3, 0], "c2": [0, 3], "c3": [-1, 0]})
    assert scores == pytest.approx({"c1": 1, "c2": 0, "c3": -1})
    chunks = [{"id": "c1", "recipe_id": "r1"}, {"id": "c2", "recipe_id": "r1"},
              {"id": "c3", "recipe_id": "r2"}]
    assert aggregate_chunk_scores(chunks, scores) == {"r1": 1, "r2": -1}
    with pytest.raises(ValueError, match="finite"):
        cosine_scores([0, 0], {"c1": [1, 0]})
    with pytest.raises(ValueError, match="dimension"):
        cosine_scores([1, 0], {"c1": [1]})
    with pytest.raises(ValueError, match="unknown"):
        aggregate_chunk_scores(chunks, {**scores, "alien": 1})


def test_pool_rescores_graph_only_candidates_with_two_real_rankings_and_no_zero_bonus():
    bm25 = {"a": 2, "b": 1, "g": 0}
    dense = {"a": .1, "b": .9, "g": .8}
    result = pool_rrf(bm25, dense, graph_ids={"g"}, lexical_limit=1, dense_limit=1)
    rows = {row["recipe_id"]: row for row in result["rankings"]}
    assert rows["g"]["graph_only_added"] and rows["g"]["from_graph"]
    assert rows["g"]["bm25_rank"] is None and rows["g"]["dense_rank"] == 2
    assert rows["g"]["rrf_score"] == pytest.approx(1 / 62)
    assert rows["b"]["bm25_rank"] == 2  # Supplemented from full corpus, not only lexical top1.
    assert result["graph_has_rank"] is False and result["counts"]["graph_only_added"] == 1
    assert result == pool_rrf(bm25, dense, graph_ids=["g", "g"], lexical_limit=1, dense_limit=1)


def test_uniform_filter_runs_after_topN_without_refill_and_graph_can_expand_pool():
    bm25, dense = {"a": 3, "b": 2, "g": 1}, {"a": .9, "b": .8, "g": .7}
    kwargs = {"eligible_ids": {"b", "g"}, "lexical_limit": 1, "dense_limit": 1}
    without = pool_rrf(bm25, dense, **kwargs)
    assert without["candidate_pool"] == [] and without["counts"]["filtered_after_seed"] == 1
    with_graph = pool_rrf(bm25, dense, graph_ids={"g"}, **kwargs)
    assert with_graph["candidate_pool"] == ["g"]
    assert with_graph["rankings"][0]["rrf_score"] == pytest.approx(2 / 61)
    with pytest.raises(ValueError, match="real corpus scores"):
        pool_rrf(bm25, dense, graph_ids={"foreign"})
