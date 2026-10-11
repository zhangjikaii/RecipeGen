import numpy as np
import pytest

from recipegen.experimental_retrieval import GraphCandidateIndex
from recipegen.graph_first_hybrid import graph_condition_candidates, graph_first_hybrid


def test_illegal_high_scores_do_not_truncate_graph_prefiltered_semantic_seed():
    ids = [f"illegal-{i:02d}" for i in range(60)] + ["legal-a", "legal-b"]
    matrix = np.asarray([[1, 0]] * 60 + [[.6, .8], [0, 1]], dtype=np.float32)
    result = graph_first_hybrid(dict.fromkeys(ids, 0), ids, matrix, [1, 0],
                                {"legal-a", "legal-b"}, eligible_ids={"legal-a", "legal-b"},
                                dense_limit=50)
    assert result["semantic_scored_ids"] == ["legal-a", "legal-b"]
    assert result["seed_ids"]["dense"] == ["legal-a", "legal-b"]
    assert result["candidate_pool"] == ["legal-a", "legal-b"]


def test_no_conditions_scores_entire_graph_domain_and_condition_terms_use_production_aliases():
    nodes = [{"id": rid, "label": "Recipe", "properties": {}} for rid in ("a", "b", "c")]
    nodes += [{"id": "ingredient-tomato", "label": "Ingredient", "properties": {"name": "tomato"}},
              {"id": "ingredient-onion", "label": "Ingredient", "properties": {"name": "onion"}}]
    edges = [{"start_id": "a", "end_id": "ingredient-tomato", "type": "HAS_INGREDIENT"},
             {"start_id": "a", "end_id": "ingredient-onion", "type": "HAS_INGREDIENT"},
             {"start_id": "b", "end_id": "ingredient-tomato", "type": "HAS_INGREDIENT"}]
    graph = GraphCandidateIndex(nodes, edges)
    assert graph_condition_candidates(graph) == {"a", "b", "c"}
    assert graph_condition_candidates(graph, ["番茄"]) == {"a", "b"}
    assert graph_condition_candidates(graph, ["番茄", "洋葱"]) == {"a"}
    assert graph_condition_candidates(graph, ["番茄"], ["洋葱"]) == {"b"}
    result = graph_first_hybrid({"a": 0, "b": 0, "c": 0}, ["a", "b", "c"],
                                [[1, 0], [0, 1], [-1, 0]], [1, 0], graph_condition_candidates(graph))
    assert result["semantic_scored_count"] == 3
    assert result["candidate_pool"] == ["a", "b", "c"]


def test_graph_is_unordered_and_zero_bm25_never_gets_lexical_rank_or_graph_bonus():
    kwargs = dict(bm25_scores={"a": 2, "b": 1, "g": 0}, recipe_ids=["a", "b", "g"],
                  recipe_matrix=[[1, 0], [0, 1], [-1, 0]], query_vector=[1, 0],
                  lexical_limit=1, dense_limit=1)
    result = graph_first_hybrid(**kwargs, graph_ids=["g", "a", "b"])
    assert result == graph_first_hybrid(**kwargs, graph_ids=["b", "g", "a", "g"])
    rows = {row["recipe_id"]: row for row in result["rankings"]}
    assert rows["g"]["from_graph"] and rows["g"]["graph_only_added"]
    assert rows["g"]["bm25_rank"] is None and rows["g"]["dense_rank"] == 3
    assert rows["g"]["rrf_score"] == pytest.approx(1 / 63)
    assert all("graph_rank" not in row for row in rows.values())
    assert result["graph_has_rank"] is False


def test_keyword_seed_is_global_then_conditions_recheck_without_refill():
    result = graph_first_hybrid({"illegal": 10, "a": 1, "b": 0}, ["illegal", "a", "b"],
                                [[1, 0], [0, 1], [-1, 0]], [1, 0], {"a", "b"},
                                eligible_ids={"a", "b"}, lexical_limit=1, dense_limit=1)
    assert result["seed_ids"]["bm25"] == ["illegal"]
    assert result["seed_ids"]["dense"] == ["a"]
    assert result["counts"]["filtered_after_seed"] == 1
    assert result["candidate_pool"] == ["a", "b"]
    # A graph-added recipe gets its actual positive lexical rank on the complete pool.
    assert result["rankings"][0]["recipe_id"] == "a"
    assert result["rankings"][0]["bm25_rank"] == 1


def test_scoring_rejects_unknown_graph_ids_and_non_normalized_query():
    with pytest.raises(ValueError, match="outside"):
        graph_first_hybrid({"a": 0}, ["a"], [[1, 0]], [1, 0], {"unknown"})
    with pytest.raises(ValueError, match="L2"):
        graph_first_hybrid({"a": 0}, ["a"], [[1, 0]], [2, 0], {"a"})


@pytest.mark.parametrize("eligible", [None, {"illegal", "a", "b"}])
def test_missing_or_overbroad_final_eligibility_cannot_add_unscored_illegal_keyword_candidate(eligible):
    result = graph_first_hybrid({"illegal": 10, "a": 1, "b": 0}, ["illegal", "a", "b"],
                                [[1, 0], [0, 1], [-1, 0]], [1, 0], {"a", "b"},
                                eligible_ids=eligible, lexical_limit=1, dense_limit=1)
    assert result["seed_ids"]["bm25"] == ["illegal"]
    assert result["candidate_pool"] == ["a", "b"]
    assert all(row["cosine_score"] is not None for row in result["rankings"])
