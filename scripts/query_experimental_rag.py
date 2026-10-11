#!/usr/bin/env python3
"""Read-only frozen-index RAG CLI; API billing requires explicit --run-api.

This is offline exact retrieval over the v2 projections, not live Neo4j ANN.
Question parsing is currently the deterministic production dictionary parser,
not an API query planner. No benchmark questions, targets, or qrels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipegen.catalog import _filters, normalize_ingredient, parse_query_filters
from recipegen.experimental_retrieval import BM25Index, GraphCandidateIndex, lexical_tokens, pool_rrf
from recipegen.experimental_qa import ExperimentalEvidenceQA, ExperimentalQAError, build_qa_context
from recipegen.text_embeddings import DIMENSION, MODEL, REVISION, TextEmbedder, validate_vector

DATA_RELATIVE = "data/evaluation_hybrid_v2"
OPERATIONAL_INPUTS = (
    "data/test_graph/nodes.jsonl", "data/test_graph/relationships.jsonl",
    "data/graphrag/full/corpus.jsonl", "data/graphrag/full/embeddings.jsonl",
    "data/graphrag/full/manifest.json", "data/graphrag/index-manifest.json",
    ".runtime/graphrag-model.json", "recipegen/catalog.py", "recipegen/kg_extract.py",
    "recipegen/experimental_retrieval.py", "scripts/local_text_embedding.py",
    "data/evaluation_real_v1/production_flat_metadata.jsonl",
)
MODEL_FILES = {"model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"}


class FrozenIndexError(ValueError):
    pass


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_index_artifacts(root=ROOT, mode="chunk"):
    """Hash only operational files; never read benchmark labels or queries."""
    root, data = Path(root), Path(root) / DATA_RELATIVE
    manifest_path = data / "manifest.json"
    if not manifest_path.is_file():
        raise FrozenIndexError("index_manifest_not_ready")
    manifest = json_read(manifest_path)
    if (manifest.get("status") != "frozen_before_retrieval" or manifest.get("benchmark") != "hybrid-retrieval-v2"
            or manifest.get("model") != MODEL or manifest.get("model_revision") != REVISION):
        raise FrozenIndexError("index_model_or_scope_mismatch")
    parameters = manifest.get("parameters", {})
    expected = {"bm25_k1": 1.5, "bm25_b": .75, "rrf_k": 60,
                "candidate_limit_per_ranked_route": 50, "chunk_recipe_aggregation": "max_cosine"}
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise FrozenIndexError("index_retrieval_parameters_mismatch")
    inputs, artifacts = manifest.get("input_sha256", {}), manifest.get("artifact_sha256", {})
    required = set(OPERATIONAL_INPUTS)
    model_entries = {name for name in inputs if name.startswith(".runtime/text-model-cache/")
                     and Path(name).name in MODEL_FILES}
    if not required <= inputs.keys():
        raise FrozenIndexError("operational_input_hash_missing")
    checked = {}
    for name in sorted(required | model_entries):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != inputs[name]:
            raise FrozenIndexError("operational_input_hash_mismatch:" + name)
        checked[name] = inputs[name]
    for name in ("recipe_documents.jsonl", "chunks.jsonl"):
        if name not in artifacts or not (data / name).is_file() or digest(data / name) != artifacts[name]:
            raise FrozenIndexError("projection_hash_mismatch:" + name)
        checked[DATA_RELATIVE + "/" + name] = artifacts[name]
    vector_manifest = None
    if mode == "chunk":
        path = data / "vector-manifest.json"
        if not path.is_file():
            raise FrozenIndexError("chunk_vectors_not_completed")
        vector_manifest = json_read(path)
        if (vector_manifest.get("status") != "completed" or vector_manifest.get("model") != MODEL
                or vector_manifest.get("model_revision") != REVISION
                or vector_manifest.get("dimension") != DIMENSION
                or vector_manifest.get("benchmark_manifest_sha256") != digest(manifest_path)):
            raise FrozenIndexError("chunk_vector_manifest_mismatch")
        expected_hash = vector_manifest.get("artifact_sha256", {}).get("chunk_vectors.npy")
        if not expected_hash or digest(data / "chunk_vectors.npy") != expected_hash:
            raise FrozenIndexError("chunk_vector_hash_mismatch")
        checked[DATA_RELATIVE + "/chunk_vectors.npy"] = expected_hash
    return manifest, vector_manifest, checked


def plan_question(question, ingredients=None, exclude=None):
    parsed = parse_query_filters(question)
    if not question.strip():
        raise ValueError("question_must_not_be_empty")
    positive = list(dict.fromkeys([*(ingredients or []), *parsed["positive"]]))
    negative = list(dict.fromkeys([*(exclude or []), *parsed["excluded"]]))
    include, excluded = _filters(positive, "ingredients"), _filters(negative, "excluded_ingredients")
    canonical_positive = list(dict.fromkeys(normalize_ingredient(item["raw"]) for item in include))
    canonical_negative = list(dict.fromkeys(normalize_ingredient(item["raw"]) for item in excluded))
    english = list(dict.fromkeys(name for item in include for name in item["names"] if re.search(r"[a-zA-Z]", name)))
    # Preserve visible English recipe/action words after negative-clause
    # removal, alongside canonical English entity expansion. Quoted passages
    # are extracted from the user's question, never a benchmark label/target.
    quoted = re.findall(r'["“]([^"”]+)["”]', parsed["positive_query"])
    body_words = [word for word in lexical_tokens(parsed["positive_query"]) if re.search(r"[a-zA-Z0-9]", word)]
    lexical_query = " ".join([*quoted, *dict.fromkeys([*body_words, *english])])
    if not lexical_query:
        lexical_query = parsed["positive_query"]
    return {"question": question, "parser": "production_deterministic_dictionary_not_API",
            "parsed_query": parsed, "include_terms": include, "exclude_terms": excluded,
            "ingredients": canonical_positive, "excluded_ingredients": canonical_negative,
            "lexical_query": lexical_query, "lexical_tokens": lexical_tokens(lexical_query), "quoted_passages": quoted,
            "e5_query": question, "api_question_parsing": False,
            "conflicting_conditions": sorted(set(canonical_positive) & set(canonical_negative)),
            "constraint_semantics": "dictionary_text_mentions_only", "ingredients_complete": False,
            "inventory_sufficient": False, "allergy_safe": False}


def eligible_recipe_ids(features, include, exclude):
    return {rid for rid, names in features.items()
            if all(set(item["names"]) & names for item in include)
            and not any(set(item["names"]) & names for item in exclude)}


class FrozenExperimentalIndex:
    def __init__(self, root=ROOT, mode="chunk"):
        if mode not in {"recipe", "chunk"}:
            raise ValueError("mode_must_be_recipe_or_chunk")
        import numpy as np
        self.root, self.mode, self.np = Path(root), mode, np
        data = self.root / DATA_RELATIVE
        manifest, vector_manifest, hashes = verify_index_artifacts(self.root, mode)
        self.manifest, self.hashes = manifest, hashes
        documents = list(lines(data / "recipe_documents.jsonl"))
        self.documents = {doc["recipe_id"]: doc for doc in documents}
        self.ids = sorted(self.documents)
        if len(self.ids) != len(documents) or len(self.ids) != manifest["source_recipes"]:
            raise FrozenIndexError("recipe_projection_count_or_ID_mismatch")
        corpus = {row["recipe_id"]: row for row in lines(self.root / "data/graphrag/full/corpus.jsonl")}
        if set(corpus) != set(self.ids):
            raise FrozenIndexError("recipe_projection_not_complete_original_corpus")
        for rid, doc in self.documents.items():
            original = corpus[rid]
            if (doc["text_sha256"] != original["text_sha256"] or doc["text"] != original["text"]
                    or doc["steps"] != original["steps"] or set(doc["source_ids"]) != set(original["source_ids"])):
                raise FrozenIndexError("recipe_projection_original_text_or_source_mismatch")
        self.features = {row["recipe_id"]: set(row["production_mention_names"])
                         for row in lines(self.root / "data/evaluation_real_v1/production_flat_metadata.jsonl")}
        if set(self.features) != set(self.ids):
            raise FrozenIndexError("production_filter_ID_mismatch")
        self.bm25 = BM25Index(documents, k1=1.5, b=.75)
        self.graph = GraphCandidateIndex(lines(self.root / "data/test_graph/nodes.jsonl"),
                                        lines(self.root / "data/test_graph/relationships.jsonl"))
        if self.graph.recipe_ids != set(self.ids):
            raise FrozenIndexError("graph_recipe_ID_mismatch")
        self.chunks, self.chunk_matrix, self.chunk_owner = [], None, None
        if mode == "recipe":
            values = {}
            for row in lines(self.root / "data/graphrag/full/embeddings.jsonl"):
                rid = row["recipe_id"]
                if (rid in values or rid not in self.documents or row.get("model_revision") != REVISION
                        or row["text_sha256"] != self.documents[rid]["text_sha256"]):
                    raise FrozenIndexError("recipe_vector_identity_mismatch")
                values[rid] = row["embedding"]
            if set(values) != set(self.ids):
                raise FrozenIndexError("recipe_vectors_incomplete")
            self.recipe_matrix = np.asarray([values[rid] for rid in self.ids], dtype=np.float32)
            self._validate_matrix(self.recipe_matrix, len(self.ids))
        else:
            self.chunks = list(lines(data / "chunks.jsonl"))
            if len(self.chunks) != vector_manifest["chunks"] or len({c["id"] for c in self.chunks}) != len(self.chunks):
                raise FrozenIndexError("chunk_count_or_ID_mismatch")
            self.chunk_matrix = np.load(data / "chunk_vectors.npy", mmap_mode="r", allow_pickle=False)
            self._validate_matrix(self.chunk_matrix, len(self.chunks))
            positions = {rid: i for i, rid in enumerate(self.ids)}
            try:
                self.chunk_owner = np.asarray([positions[c["recipe_id"]] for c in self.chunks], dtype=np.int64)
            except KeyError:
                raise FrozenIndexError("unknown_chunk_recipe_ID") from None
        self.summary = {"ready": True, "mode": mode, "scope": "offline_exact_not_live_Neo4j_ANN",
                        "recipes": len(self.ids), "source_steps": manifest["source_steps"],
                        "chunks": len(self.chunks) if mode == "chunk" else None,
                        "embedding_model": MODEL, "embedding_revision": REVISION, "dimension": DIMENSION,
                        "index_hashes_verified": True, "verified_operational_files": len(hashes),
                        "benchmark_labels_or_questions_read": False, "bm25": self.bm25.metadata}

    def _validate_matrix(self, matrix, rows):
        np = self.np
        if matrix.shape != (rows, DIMENSION) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise FrozenIndexError("vector_shape_dtype_or_values_invalid")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.01):
            raise FrozenIndexError("vector_not_L2_normalized")

    def retrieve(self, plan, query_vector):
        np = self.np
        vector = np.asarray(validate_vector(query_vector), dtype=np.float32)
        bm25_scores = self.bm25.scores(plan["lexical_tokens"])
        eligible = eligible_recipe_ids(self.features, plan["include_terms"], plan["exclude_terms"])
        graph_ids = self.graph.candidates(ingredients=plan["ingredients"], excluded=plan["excluded_ingredients"])
        if self.mode == "recipe":
            scores = self.recipe_matrix @ vector
            chunk_scores = None
        else:
            chunk_scores = self.chunk_matrix @ vector
            scores = np.full(len(self.ids), -np.inf, dtype=np.float32)
            np.maximum.at(scores, self.chunk_owner, chunk_scores)
            if not np.isfinite(scores).all():
                raise FrozenIndexError("recipe_has_no_derived_text_vector")
        dense_scores = dict(zip(self.ids, map(float, scores)))
        result = pool_rrf(bm25_scores, dense_scores, graph_ids=graph_ids, eligible_ids=eligible,
                          lexical_limit=50, dense_limit=50, rrf_k=60)
        result["mode"] = self.mode
        result["eligible_recipe_count"] = len(eligible)
        result["query_vector_sha256"] = hashlib.sha256(vector.tobytes()).hexdigest()
        result["dense_scoring"] = "float32_dot_of_verified_L2_normalized_E5_vectors"
        result["matched_chunks"] = {}
        if chunk_scores is not None:
            selected = {row["recipe_id"] for row in result["rankings"][:20]}
            best = {}
            for chunk, score in zip(self.chunks, chunk_scores):
                rid = chunk["recipe_id"]
                if rid in selected and (rid not in best or float(score) > best[rid][0]):
                    best[rid] = (float(score), chunk)
            result["matched_chunks"] = {rid: {"score": score, "chunk_id": chunk["id"], "unit": chunk["unit"],
                "step_ids": chunk["step_ids"], "source_ids": chunk["source_ids"], "fragments": chunk["fragments"]}
                for rid, (score, chunk) in best.items()}
        return result


def run_question(index, question, *, ingredients=None, exclude=None, run_api=False, env_file=None,
                 embedder=None, qa=None):
    tick = time.monotonic()
    plan = plan_question(question, ingredients, exclude)
    encoder = embedder or TextEmbedder(index.root)
    retrieval = index.retrieve(plan, encoder.embed_query(question))
    ranked_ids = [row["recipe_id"] for row in retrieval["rankings"]]
    # Missing-step title units stay visible in retrieval. They are not cooking
    # evidence and cannot be promoted into a successful full-step API context.
    rejected = [rid for rid in ranked_ids if not index.documents[rid]["steps"]]
    candidates = [rid for rid in ranked_ids if rid not in rejected][:20]
    result = {"status": "retrieval_only", "question_plan": plan, "index": index.summary,
              "retrieval": retrieval, "rerank": {"status": "not_requested", "api_call_attempted": False},
              "qa": {"status": "not_requested", "generation": {"api_call_attempted": False}},
              "omitted_from_step_QA": [{"recipe_id": rid, "reason": "original_steps_missing"} for rid in rejected],
              "api_question_parsing": False, "semantic_accuracy_verified": False}
    ordered = candidates
    if run_api and candidates:
        if qa is None:
            # The safe loader is imported/called only after explicit opt-in.
            from recipegen.experimental_qa import load_experimental_settings
            qa = ExperimentalEvidenceQA(settings=load_experimental_settings(env_file), allow_api_calls=True)
        result["api_configuration"] = qa.status()
        result["rerank"] = qa.rerank(question, index.documents, candidates)
        ordered = result["rerank"]["ordered_recipe_ids"]
    selected = ordered[:5]
    result["selected_recipe_ids"] = selected
    result["source_documents"] = [index.documents[rid] for rid in selected]
    result["title_only_materials"] = [index.documents[rid] for rid in rejected[:5]]
    if selected:
        try:
            context = build_qa_context(question, index.documents, selected)
            result["evidence"] = context["evidence"]
            result["original_steps_complete"] = context["original_steps_complete"]
        except ExperimentalQAError as error:
            result["evidence"] = []
            result["original_steps_complete"] = False
            result["status"] = "source_materials_only"
            result["qa"] = {"status": "context_error", "error_code": str(error),
                            "generation": {"api_call_attempted": False}}
        if run_api and result["original_steps_complete"]:
            result["qa"] = qa.answer(question, index.documents, selected)
            qa_status = result["qa"]["status"]
            result["status"] = "api_answered" if qa_status == "answered" else (
                "api_insufficient_evidence" if qa_status == "insufficient_evidence" else "source_materials_only")
    else:
        result["evidence"] = []
        result["original_steps_complete"] = False
        result["status"] = "source_materials_only" if rejected else "no_match"
    result["elapsed_seconds"] = round(time.monotonic() - tick, 3)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="原问题；未提供时只检查索引，不编码或调用API")
    parser.add_argument("--mode", choices=["recipe", "chunk"], default="chunk")
    parser.add_argument("--ingredients", nargs="+", default=[], help="显式要求提及的食材")
    parser.add_argument("--exclude", nargs="+", default=[], help="显式排除提及的食材")
    parser.add_argument("--run-api", action="store_true", help="显式授权DeepSeek Top20重排和Top5证据问答（可能收费）")
    parser.add_argument("--env-file", type=Path, help="只在--run-api时读取指定配置；默认诊断不读.env")
    parser.add_argument("--output", type=Path, help="可选新JSON文件，禁止覆盖已有文件")
    args = parser.parse_args(argv)
    if args.env_file and not args.run_api:
        parser.error("--env-file requires --run-api")
    if args.run_api and not args.question:
        parser.error("--run-api requires --question")
    if args.output and args.output.exists():
        parser.error("--output must name a new file")
    try:
        index = FrozenExperimentalIndex(mode=args.mode)
        result = (run_question(index, args.question, ingredients=args.ingredients, exclude=args.exclude,
                               run_api=args.run_api, env_file=args.env_file) if args.question else
                  {"status": "index_ready", "index": index.summary, "api_call_attempted": False})
        exit_code = 0
    except (FrozenIndexError, ValueError) as error:
        # Controlled validation messages contain only public paths/codes.
        result, exit_code = {"status": "validation_error", "error_code": str(error), "api_call_attempted": False}, 2
    except Exception:
        # Never echo configuration objects, upstream bodies, or credentials.
        result, exit_code = {"status": "query_error", "error_code": "query_error", "api_call_attempted": False}, 2
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
