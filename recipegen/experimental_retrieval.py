"""Offline experimental retrieval; never loads a model, .env, gold, or Neo4j.

The original graph is unchanged. Text documents/chunks are derived projections,
and graph matches are unordered candidate sets, never a third RRF ranking.
Callers must supply frozen production eligibility separately from gold labels.
Exact scores in this module do not measure live Neo4j ANN or HTTP performance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import math
import re
import unicodedata
from typing import Any

from .test_graph import REPO, uid

RRF_K = 60
SCOPE = "offline_exact_scoring_not_Neo4j_ANN"
BM25_REFERENCE = "https://lucene.apache.org/core/9_12_2/core/org/apache/lucene/search/similarities/BM25Similarity.html"


def _records(raw_records):
    if isinstance(raw_records, Mapping):
        return [raw_records] if "id" in raw_records or "recipe_id" in raw_records else raw_records.values()
    return raw_records


def build_recipe_documents(raw_records: Iterable[dict] | Mapping) -> list[dict]:
    """Preserve title and every original step; derive IDs using the graph builder.

    Accept raw records.jsonl rows, existing corpus.jsonl rows, or an ID mapping.
    A provided step/source ID is retained. Missing raw IDs are derived using the
    exact test_graph.uid scheme, not invented from text similarity or gold.
    """
    documents, seen = [], set()
    for record in _records(raw_records):
        rid = record.get("recipe_id", record.get("id"))
        title = record.get("title")
        if not isinstance(rid, str) or not rid or rid in seen or not isinstance(title, str):
            raise ValueError("Recipes need unique nonempty IDs and string titles")
        seen.add(rid)
        source = record.get("source", {})
        source_ids = set(record.get("source_ids", []))
        sources = []
        for role, member_key, hash_key in (("title", "goal_member", "goal_sha256"),
                                           ("steps", "steps_member", "steps_sha256")):
            if source.get(member_key):
                if not source.get("archive"):
                    raise ValueError("Raw sources need archive and member identity")
                sid = uid("source", f"{REPO}\n{source['archive']}\n{source[member_key]}")
                source_ids.add(sid)
                sources.append({"id": sid, "role": role, "archive": source["archive"],
                                "member": source[member_key], "sha256": source.get(hash_key),
                                "dataset_revision": source.get("revision"), "url": source.get("url")})
        step_source = next((s["id"] for s in sources if s["role"] == "steps"), None)
        steps = []
        for position, step in enumerate(record.get("steps", []), 1):
            if step.get("order") != position or not isinstance(step.get("text"), str):
                raise ValueError("Steps must preserve contiguous original-line order and text")
            sid = step.get("source_id") or step_source
            if not isinstance(sid, str) or not sid:
                raise ValueError("Every step needs an original Source ID")
            source_ids.add(sid)
            steps.append({"id": step.get("id") or uid("step", f"{rid}\n{position}"),
                          "order": position, "text": step["text"], "source_id": sid})
        text = title + "\n" + "\n".join(s["text"] for s in steps)
        documents.append({"id": rid, "recipe_id": rid, "title": title, "text": text,
                          "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                          "steps": steps, "step_ids": [s["id"] for s in steps],
                          "source_ids": sorted(source_ids), "sources": sources, "unit": "recipe"})
    return sorted(documents, key=lambda d: d["recipe_id"])


def _token_count(tokenizer, text: str) -> int:
    # E5 budget includes prefix, title, separators, and model special tokens.
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def _fitting_end(text: str, start: int, render, tokenizer, max_tokens: int) -> int:
    """Find a safe character boundary; no tokenizer.decode round trip is used."""
    low, high, best = start + 1, len(text), start
    while low <= high:
        end = (low + high) // 2
        if _token_count(tokenizer, render(text[start:end])) <= max_tokens:
            best, low = end, end + 1
        else:
            high = end - 1
    # Prefix-token counts need not be perfectly monotonic (subword merges).
    # Safety is checked on the actual chosen string, even if not maximal.
    if best == start and _token_count(tokenizer, render(text[start:start + 1])) <= max_tokens:
        best = start + 1
    if best == start:
        raise ValueError("Prefix/title budget leaves no room for a source character")
    return best


def build_step_chunks(raw_records, tokenizer, max_tokens: int = 480,
                      max_steps: int = 3, prefix: str = "passage: ") -> list[dict]:
    """Group 1–3 adjacent original steps, splitting oversized steps losslessly.

    Chunks do not overlap. Fragments retain exact Python character spans into
    their original Step, so concatenating its fragments reconstructs it exactly.
    `text` is unprefixed for encode(..., prefix='passage: '); `embedding_text`
    already contains the prefix and must not receive a second prefix.
    An exceptionally long title is shortened only in the search projection;
    the full title remains metadata and title_context_truncated is explicit.
    Records without steps retain a title_only unit with a real title Source,
    empty fragments/step IDs, and no invented cooking-step evidence.
    """
    if type(max_tokens) is not int or max_tokens < 8 or type(max_steps) is not int or not 1 <= max_steps <= 3:
        raise ValueError("max_tokens >= 8 and max_steps in 1..3 are required")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    chunks = []
    for document in build_recipe_documents(raw_records):
        title = document["title"]
        title_context = title
        header_count = _token_count(tokenizer, prefix + title + "\n")
        if header_count > max_tokens - min(8, max_tokens // 4):
            # Keep usable step capacity without changing the original title.
            if _token_count(tokenizer, prefix + "\n") >= max_tokens // 2:
                raise ValueError("Prefix alone exceeds the available chunk budget")
            end = _fitting_end(title, 0, lambda t: prefix + t + "\n", tokenizer, max_tokens // 2)
            title_context = title[:end]

        def render(fragments):
            return title_context + "\n" + "\n".join(f["text"] for f in fragments)

        def fits(fragments):
            return _token_count(tokenizer, prefix + render(fragments)) <= max_tokens

        def append_chunk(fragments, *, title_only=False):
            text = title_context if title_only else render(fragments)
            embedding_text = prefix + text
            token_count = _token_count(tokenizer, embedding_text)
            if token_count > max_tokens:
                raise ValueError("A derived chunk exceeds its complete E5 token budget")
            identity = document["recipe_id"] + "\n" + embedding_text + "\n" + repr(
                [(f["step_id"], f["span_start"], f["span_end"]) for f in fragments])
            cid = uid("experimental_chunk", identity)
            source_ids = sorted({f["source_id"] for f in fragments})
            if title_only:
                source_ids = sorted({s["id"] for s in document["sources"] if s["role"] == "title"})
                # Existing empty-step corpus rows have exactly one Source ID:
                # their original title Source. An ambiguous source list cannot
                # establish title provenance and must not be guessed.
                if not source_ids and len(document["source_ids"]) == 1:
                    source_ids = document["source_ids"][:]
                if not source_ids:
                    raise ValueError("A title-only unit needs an unambiguous original title Source ID")
            chunks.append({"id": cid, "chunk_id": cid, "recipe_id": document["recipe_id"],
                           "title": title, "title_context": title_context,
                           "title_context_truncated": title_context != title,
                           "text": text, "embedding_text": embedding_text, "prefix": prefix,
                           "token_count": token_count, "max_tokens": max_tokens,
                           "fragments": fragments, "step_ids": list(dict.fromkeys(f["step_id"] for f in fragments)),
                           "source_ids": source_ids, "unit": "title_only" if title_only else "step_chunk"})

        if not document["steps"]:
            append_chunk([], title_only=True)
            continue

        pending = []
        for step in document["steps"]:
            original = step["text"]

            def fragment(start, end):
                return {"step_id": step["id"], "source_id": step["source_id"], "order": step["order"],
                        "span_start": start, "span_end": end, "text": original[start:end],
                        "original_text_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest()}

            whole = fragment(0, len(original))
            if fits([whole]):
                if pending and (len(pending) >= max_steps or not fits(pending + [whole])):
                    append_chunk(pending)
                    pending = []
                pending.append(whole)
                continue
            if pending:
                append_chunk(pending)
                pending = []
            start = 0
            while start < len(original):
                end = _fitting_end(original, start, lambda piece: prefix + title_context + "\n" + piece,
                                   tokenizer, max_tokens)
                append_chunk([fragment(start, end)])
                start = end
        if pending:
            append_chunk(pending)
    return chunks


_LEXICAL_PATTERN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*|[\u3400-\u9fff]|[^\W\d_]+", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    """NFKC/lowercase words + CJK character unigrams; no stemming or aliases.

    English plural/inflected words remain distinct. Chinese query translation
    or entity expansion must be supplied and frozen by the caller, not gold.
    """
    if not isinstance(text, str):
        raise ValueError("Lexical input must be text")
    return _LEXICAL_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())


class BM25Index:
    """Project positive-IDF Okapi implementation; not rank_bm25 execution.

    idf(t)=log(1+(N-df(t)+.5)/(df(t)+.5)); score sums unique query terms:
    idf(t)*tf(t,d)*(k1+1)/(tf(t,d)+k1*(1-b+b*len(d)/avgdl)).
    Positive IDF follows Lucene BM25Similarity (reference below); defaults are
    explicitly k1=1.5,b=.75, not a claim to reproduce Lucene byte-length norms.
    """
    def __init__(self, documents: Iterable[dict], k1: float = 1.5, b: float = .75,
                 tokenizer=lexical_tokens):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.k1, self.b, self.tokenizer = float(k1), float(b), tokenizer
        self.doc_ids, lengths, self.postings = [], {}, defaultdict(dict)
        for doc in documents:
            identifier = doc.get("id", doc.get("recipe_id"))
            if not isinstance(identifier, str) or not identifier or identifier in lengths:
                raise ValueError("BM25 document IDs must be unique nonempty strings")
            tokens = list(tokenizer(doc["text"]))
            self.doc_ids.append(identifier)
            lengths[identifier] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self.postings[term][identifier] = frequency
        self.doc_ids.sort()
        self.avgdl = sum(lengths.values()) / len(lengths) if lengths else 0.0
        self.lengths = lengths
        self.idf = {term: math.log1p((len(lengths) - len(postings) + .5) / (len(postings) + .5))
                    for term, postings in self.postings.items()}
        self.norm = {rid: self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
                     for rid, length in lengths.items()}
        self.metadata = {"implementation": "project_positive_idf_okapi", "k1": self.k1, "b": self.b,
                         "idf": "log1p((N-df+0.5)/(df+0.5))", "query_term_frequency": "binary_unique",
                         "tokenization": "NFKC_lowercase_words_CJK_unigrams_no_stemming_no_aliases",
                         "reference": BM25_REFERENCE, "documents": len(lengths), "avgdl": self.avgdl}

    def scores(self, query_tokens: str | Sequence[str], ids=None) -> dict[str, float]:
        selected = set(self.doc_ids) if ids is None else set(ids)
        if not selected <= self.lengths.keys():
            raise ValueError("Cannot score unknown BM25 document IDs")
        tokens = self.tokenizer(query_tokens) if isinstance(query_tokens, str) else query_tokens
        result = {rid: 0.0 for rid in sorted(selected)}
        for term in set(tokens):
            for rid, tf in self.postings.get(term, {}).items():
                if rid in selected:
                    result[rid] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[rid])
        return result


def _name(value):
    return re.sub(r"[\s\-‐‑–—]+", " ", unicodedata.normalize("NFKC", value).lower()).strip()


class GraphCandidateIndex:
    """Compact index of actual exported graph relations; consumes generators.

    Only graph nodes/edges are read: no source rescanning, extractor, or gold.
    Ingredient/action/tool predicates mean dictionary text mentions, not actual
    use, complete inventories, semantic verb arguments, or allergy guarantees.
    """
    def __init__(self, nodes: Iterable[dict], edges: Iterable[dict]):
        labels, entities, self.step_info = {}, {}, {}
        self.recipe_ids, scopes = set(), set()
        for node in nodes:
            label, identifier, props = node["label"], node["id"], node.get("properties", {})
            if label not in {"Recipe", "Step", "Ingredient", "Action", "Tool", "Source"}:
                continue
            if identifier in labels:
                raise ValueError("Graph node IDs must be unique")
            labels[identifier] = label
            if label == "Recipe":
                self.recipe_ids.add(identifier)
                scopes.add(tuple(props.get(k) for k in ("build_id", "dataset_revision", "split")))
            elif label == "Step":
                self.step_info[identifier] = {k: props.get(k) for k in ("order", "source_id", "recipe_id")}
            elif label in {"Ingredient", "Action", "Tool"}:
                entities[identifier] = (label, {_name(props[k]) for k in ("name", "normalized_name", "zh_name")
                                               if isinstance(props.get(k), str) and props[k].strip()})
        if len(scopes) > 1:
            raise ValueError("Graph candidates cannot mix dataset/build scopes")
        self.scope = next(iter(scopes), (None, None, None))
        self.step_owner, self.recipe_sources = {}, defaultdict(set)
        entity_edges, direct_ingredients = [], []
        for edge in edges:
            start, end, kind = edge["start_id"], edge["end_id"], edge["type"]
            if kind not in {"HAS_STEP", "HAS_SOURCE", "HAS_INGREDIENT", "USES_INGREDIENT", "HAS_ACTION", "USES_TOOL"}:
                continue
            props = edge.get("properties", {})
            if any(value is not None for value in self.scope) and tuple(
                    props.get(k) for k in ("build_id", "dataset_revision", "split")) != self.scope:
                raise ValueError("Graph edge scope differs from its Recipe build")
            if kind == "HAS_STEP" and labels.get(start) == "Recipe" and labels.get(end) == "Step":
                if end in self.step_owner and self.step_owner[end] != start:
                    raise ValueError("A Step cannot belong to two Recipes")
                self.step_owner[end] = start
            elif kind == "HAS_SOURCE" and labels.get(start) == "Recipe" and labels.get(end) == "Source":
                self.recipe_sources[start].add(end)
            elif kind == "HAS_INGREDIENT" and labels.get(start) == "Recipe" and labels.get(end) == "Ingredient":
                direct_ingredients.append((start, end))
            elif labels.get(start) == "Step" and end in entities:
                expected = {"USES_INGREDIENT": "Ingredient", "HAS_ACTION": "Action", "USES_TOOL": "Tool"}.get(kind)
                if entities[end][0] == expected:
                    entity_edges.append((start, end, kind))
        self.by_kind = {kind: defaultdict(set) for kind in ("Ingredient", "Action", "Tool")}
        self.paths = defaultdict(list)
        for rid, eid in direct_ingredients:
            for name in entities[eid][1]:
                self.by_kind["Ingredient"][name].add(rid)
            self.paths[rid].append({"node_ids": [rid, eid], "relations": ["HAS_INGREDIENT"], "kind": "Ingredient"})
        for step_id, eid, relation in entity_edges:
            rid = self.step_owner.get(step_id)
            if rid is None:
                raise ValueError("An entity-bearing Step lacks HAS_STEP provenance")
            owner_hint = self.step_info[step_id].get("recipe_id")
            if owner_hint is not None and owner_hint != rid:
                raise ValueError("Step recipe_id differs from HAS_STEP owner")
            kind, names = entities[eid]
            for name in names:
                self.by_kind[kind][name].add(rid)
            self.paths[rid].append({"node_ids": [rid, step_id, eid], "relations": ["HAS_STEP", relation],
                                    "kind": kind, "source_id": self.step_info[step_id]["source_id"]})
        self.semantics = "dictionary_text_mentions_only_not_verified_use"

    def eligible_ids(self, ingredients=(), excluded=(), actions=(), tools=()) -> set[str]:
        """Graph predicate result, not gold; caller chooses uniform eligibility."""
        result = self.recipe_ids.copy()
        for kind, terms in (("Ingredient", ingredients), ("Action", actions), ("Tool", tools)):
            for term in terms:
                result &= self.by_kind[kind].get(_name(term), set())
        for term in excluded:
            result -= self.by_kind["Ingredient"].get(_name(term), set())
        return result

    def candidates(self, ingredients=(), excluded=(), actions=(), tools=()) -> set[str]:
        if not any((ingredients, excluded, actions, tools)):
            return set()
        return self.eligible_ids(ingredients, excluded, actions, tools)

    def evidence_paths(self, recipe_id: str) -> list[dict]:
        return self.paths.get(recipe_id, [])


def cosine_scores(query_vector, vectors_by_id: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """True cosine for caller-supplied vectors; no model calls or truncation."""
    q = [float(v) for v in query_vector]
    qnorm = math.sqrt(sum(v * v for v in q))
    if not q or not all(math.isfinite(v) for v in q) or not qnorm:
        raise ValueError("Query vector must be finite and nonzero")
    ids = sorted(vectors_by_id)
    vectors = [[float(v) for v in vectors_by_id[rid]] for rid in ids]
    if any(len(v) != len(q) or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise ValueError("Candidate vectors must share dimension and be finite/nonzero")
    try:
        import numpy as np
    except ImportError:
        return {rid: sum(a * b for a, b in zip(q, vector)) / (qnorm * math.sqrt(sum(x * x for x in vector)))
                for rid, vector in zip(ids, vectors)}
    if not ids:
        return {}
    matrix = np.asarray(vectors, dtype=np.float64)
    values = matrix @ np.asarray(q, dtype=np.float64) / (np.linalg.norm(matrix, axis=1) * qnorm)
    return {rid: float(score) for rid, score in zip(ids, values)}


def aggregate_chunk_scores(chunks: Iterable[dict], scores: Mapping[str, float]) -> dict[str, float]:
    """Max cosine per Recipe; chunk-count/length bias needs separate evaluation."""
    aggregated = {}
    seen = set()
    for chunk in chunks:
        cid = chunk.get("chunk_id", chunk.get("id"))
        if cid in seen or cid not in scores:
            raise ValueError("Each unique chunk needs a score")
        seen.add(cid)
        score, rid = float(scores[cid]), chunk["recipe_id"]
        if not math.isfinite(score):
            raise ValueError("Chunk scores must be finite")
        aggregated[rid] = max(aggregated.get(rid, -math.inf), score)
    if set(scores) != seen:
        raise ValueError("Scores contain unknown chunk IDs")
    return aggregated


def _ranking(scores, positive_only=False):
    if any(not math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("Scores must be finite")
    return sorted((rid for rid, value in scores.items() if not positive_only or value > 0),
                  key=lambda rid: (-scores[rid], rid))


def pool_rrf(bm25_scores: Mapping[str, float], dense_scores: Mapping[str, float], graph_ids=(),
             eligible_ids=None, lexical_limit: int = 50, dense_limit: int = 50,
             rrf_k: int = RRF_K, limit: int | None = None) -> dict[str, Any]:
    """Whole-corpus top-N -> uniform filter (no refill) -> pool rescoring/RRF.

    Supply full-corpus scores, including BM25 zeros and graph-only candidates.
    Graph IDs add candidates, not ranks or score bonuses. Real BM25/cosine are
    reranked within the filtered union U. A zero BM25 gets no lexical rank.
    Missing dense scores (e.g. no source steps/chunks) get no dense rank.
    """
    if any(type(v) is not int or v < 1 for v in (lexical_limit, dense_limit, rrf_k)):
        raise ValueError("Limits and RRF k must be positive integers")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Final limit must be positive or None")
    lexical_seed = _ranking(bm25_scores, positive_only=True)[:lexical_limit]
    dense_seed = _ranking(dense_scores)[:dense_limit]
    graph = set(graph_ids)
    seed = set(lexical_seed) | set(dense_seed)
    union = seed | graph
    if not union <= bm25_scores.keys() | dense_scores.keys():
        raise ValueError("Every pool candidate needs real corpus scores")
    pool = union if eligible_ids is None else union & set(eligible_ids)
    lexical = _ranking({rid: bm25_scores[rid] for rid in pool if rid in bm25_scores}, positive_only=True)
    dense = _ranking({rid: dense_scores[rid] for rid in pool if rid in dense_scores})
    lexical_ranks, dense_ranks = ({rid: i for i, rid in enumerate(ranking, 1)} for ranking in (lexical, dense))
    rows = []
    for rid in sorted(pool):
        lr, dr = lexical_ranks.get(rid), dense_ranks.get(rid)
        rows.append({"recipe_id": rid, "bm25_score": bm25_scores.get(rid), "cosine_score": dense_scores.get(rid),
                     "bm25_rank": lr, "dense_rank": dr,
                     "rrf_score": (1 / (rrf_k + lr) if lr else 0) + (1 / (rrf_k + dr) if dr else 0),
                     "from_graph": rid in graph, "graph_only_added": rid in graph - seed})
    rows.sort(key=lambda row: (-row["rrf_score"], row["recipe_id"]))
    return {"scope": SCOPE, "rrf_k": rrf_k, "graph_has_rank": False,
            "filter_order": "full_corpus_topN_then_uniform_filter_no_refill",
            "seed_ids": {"bm25": lexical_seed, "dense": dense_seed, "graph": sorted(graph)},
            "candidate_pool": sorted(pool), "counts": {"bm25_seed": len(lexical_seed), "dense_seed": len(dense_seed),
                "graph": len(graph), "graph_only_added": len((graph - seed) & pool),
                "filtered_after_seed": len(union - pool), "pool": len(pool)},
            "rankings": rows if limit is None else rows[:limit]}
