"""Production BM25 corpus and graph-qualified exact E5/RRF scoring.

This module reads no credentials and invokes no model, API or database. The
caller supplies verified live graph candidates; qrels never enter this layer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .catalog import CatalogError, _filters
from .experimental_retrieval import BM25Index, lexical_tokens, pool_rrf


def corpus_identity_hash(documents):
    """Exact build_text_index contract: ordered recipe-ID/text-hash projection.

    The builder's Cypher orders rows by recipe_id before exporting JSONL. Its
    corpus_sha256 is fingerprint(projection), not the JSONL file byte hash.
    Original row order is preserved; JSON formatting/key order is irrelevant.
    """
    projection = [{"recipe_id": doc["recipe_id"], "text_sha256": doc["text_sha256"]}
                  for doc in documents]
    canonical = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lexical_query_text(query, include):
    """Same visible-condition lexical projection as the frozen experiment."""
    if include:
        terms = _filters([term["raw"] for term in include], "ingredients")
        return " ".join(dict.fromkeys(name for term in terms for name in term["names"]
                                      if re.search(r"[a-zA-Z]", name)))
    quoted = re.search(r'contain this exact passage: "(.*)"$', query, re.S)
    return quoted.group(1) if quoted else query


class VerifiedBM25Corpus:
    def __init__(self, path, manifest):
        self.path = Path(path)
        expected = manifest.get("corpus_sha256")
        expected_count = manifest.get("index", {}).get("count")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise CatalogError("三路混合检索缺少可核验的原食谱语料哈希。")
        if type(expected_count) is not int or expected_count < 1:
            raise CatalogError("三路混合检索缺少可核验的原食谱语料数量。")
        try:
            content = self.path.read_bytes()
        except OSError:
            raise CatalogError("三路混合检索的原食谱语料无法读取。") from None
        try:
            docs = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            self.documents = {}
            for doc in docs:
                rid = doc["recipe_id"]
                steps = doc["steps"]
                if (not isinstance(rid, str) or not rid.startswith("recipegen:test:")
                        or rid in self.documents or not isinstance(doc["title"], str)
                        or not isinstance(steps, list)
                        or any(type(step.get("order")) is not int or step["order"] != pos
                               or not isinstance(step.get("text"), str) for pos, step in enumerate(steps, 1))):
                    raise ValueError("Corpus identity or original step order invalid")
                text = doc["title"] + "\n" + "\n".join(step["text"] for step in steps)
                if (doc["text"] != text or hashlib.sha256(text.encode("utf-8")).hexdigest() != doc["text_sha256"]):
                    raise ValueError("Corpus original text hash mismatch")
                self.documents[rid] = doc
            if len(self.documents) != expected_count:
                raise ValueError("Corpus count mismatch")
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise CatalogError("三路混合检索原食谱语料结构、数量或完整文本哈希无效。") from None
        if corpus_identity_hash(docs) != expected:
            raise CatalogError("三路混合检索原食谱语料与索引 manifest 哈希不一致。")
        self.index = BM25Index(self.documents.values(), k1=1.5, b=.75)
        self.sha256 = expected
        self.file_sha256 = hashlib.sha256(content).hexdigest()

    def scores(self, query):
        return self.index.scores(lexical_tokens(query))


def score_graph_first_pool(bm25_scores, vectors_by_id, query_vector, eligible_ids,
                           lexical_limit=50, dense_limit=50, rrf_k=60):
    """Score only the complete graph C matrix, then complete-pool two-rank RRF."""
    import numpy as np

    graph_ids = set(vectors_by_id)
    if not graph_ids <= set(bm25_scores):
        raise CatalogError("图谱候选不在已验证的原食谱 BM25 语料中。")
    ids = sorted(graph_ids)
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.shape != (384,) or not np.isfinite(vector).all() or not np.any(vector):
        raise CatalogError("三路混合检索问题向量无效。")
    # _vector accepts nonunit external vectors; normalize for genuine cosine.
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or not norm:
        raise CatalogError("三路混合检索问题向量归一化无效。")
    vector = vector / norm
    matrix = np.asarray([vectors_by_id[rid] for rid in ids], dtype=np.float32)
    if not ids:
        matrix = np.empty((0, 384), dtype=np.float32)
    if (matrix.shape != (len(ids), 384) or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01)):
        raise CatalogError("三路混合检索图条件候选缺少有效 L2 文本向量。")
    dense_scores = dict(zip(ids, map(float, matrix @ vector)))
    result = pool_rrf(bm25_scores, dense_scores, graph_ids=graph_ids,
                      eligible_ids=graph_ids & set(eligible_ids),
                      lexical_limit=lexical_limit, dense_limit=dense_limit, rrf_k=rrf_k)
    result.update({"filter_order": "BM25_global_top50_and_graph_prefiltered_E5_top50_then_union_condition_recheck",
                   "semantic_scored_ids": ids, "semantic_scored_count": len(ids),
                   "graph_has_rank": False, "all_final_candidates_have_dense_scores": True})
    if any(row["cosine_score"] is None for row in result["rankings"]):
        raise CatalogError("三路混合检索最终候选缺少真实 E5 分数。")
    return result
