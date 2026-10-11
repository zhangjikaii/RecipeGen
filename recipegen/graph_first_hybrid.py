"""Offline three-route retrieval with graph-prefiltered E5 scoring.

BM25 contributes a global positive-score Top-N seed. E5 scores only the
complete graph-qualified recipe matrix. The graph contributes an unordered
candidate set; final candidates receive two real ranks, never a graph rank.
No model loading, credentials, network, query rewriting, or relevance labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .catalog import _filters
from .experimental_retrieval import GraphCandidateIndex, pool_rrf


def graph_condition_candidates(graph: GraphCandidateIndex, ingredients=(), excluded=()) -> set[str]:
    """Production alias OR per term, positive AND, exclusion set difference.

Only frozen graph Ingredient nodes and relationships provide membership.
With no conditions, the semantic score domain is every graph Recipe.
"""
    includes = _filters(list(ingredients), "ingredients")
    excludes = _filters(list(excluded), "excluded_ingredients")

    def matches(term):
        found = set()
        for name in term["names"]:
            found.update(graph.eligible_ids(ingredients=[name]))
        return found

    result = set(graph.recipe_ids)
    for term in includes:
        result.intersection_update(matches(term))
    for term in excludes:
        result.difference_update(matches(term))
    return result


def graph_first_hybrid(bm25_scores: Mapping[str, float], recipe_ids: Sequence[str],
                       recipe_matrix, query_vector, graph_ids,
                       eligible_ids=None, lexical_limit=50, dense_limit=50,
                       rrf_k=60, limit=None) -> dict:
    """Execute prefiltered exact E5 + global BM25 + unordered graph -> RRF.

    ``recipe_matrix`` holds original L2-normalized recipe-level vectors. The
    actual float32 dot product reads only rows in ``graph_ids``. The entire
    graph candidate set joins the union, and production conditions are checked
    once more before complete-pool BM25/E5 ranks and two-rank RRF are computed.
    Caller-supplied gold IDs cannot influence this function.
    """
    import numpy as np

    ids = list(recipe_ids)
    if len(ids) != len(set(ids)) or set(ids) != set(bm25_scores):
        raise ValueError("Unique recipe IDs and complete BM25 domain are required")
    graph = set(graph_ids)
    if not graph <= set(ids):
        raise ValueError("Graph candidates outside the recipe vector domain")
    matrix = np.asarray(recipe_matrix, dtype=np.float32)
    vector = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or vector.shape != (matrix.shape[1],):
        raise ValueError("Recipe matrix and query vector dimensions differ")
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=.01):
        raise ValueError("Query vector must be finite and L2 normalized")
    positions = {rid: pos for pos, rid in enumerate(ids)}
    scored_ids = sorted(graph)
    selected = matrix[[positions[rid] for rid in scored_ids]]
    if not np.isfinite(selected).all() or not np.allclose(np.linalg.norm(selected, axis=1), 1, atol=.01):
        raise ValueError("Graph-qualified vectors must be finite and L2 normalized")
    # Deliberately no full-corpus matrix-vector multiplication.
    dense_scores = dict(zip(scored_ids, map(float, selected @ vector)))
    final_eligible = graph if eligible_ids is None else graph & set(eligible_ids)
    result = pool_rrf(bm25_scores, dense_scores, graph_ids=graph,
                      eligible_ids=final_eligible, lexical_limit=lexical_limit,
                      dense_limit=dense_limit, rrf_k=rrf_k, limit=limit)
    result.update({
        "filter_order": "global_positive_BM25_TopN_and_graph_prefiltered_E5_TopN_then_union_hard_condition_recheck",
        "semantic_score_domain": "complete_graph_qualified_recipe_pool",
        "semantic_scored_ids": scored_ids,
        "semantic_scores": dense_scores,
        "semantic_scored_count": len(scored_ids),
        "dense_scoring": "float32_dot_of_original_L2_normalized_recipe_vectors",
        "all_graph_candidates_join_union": True,
        "graph_has_rank": False,
    })
    if any(row["cosine_score"] is None for row in result["rankings"]):
        raise ValueError("Every final candidate needs a graph-qualified E5 score")
    return result
