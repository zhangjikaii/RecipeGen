#!/usr/bin/env python3
"""验收本机文本 GraphRAG：真实 HTTP、Neo4j 原文与来源；不处理媒体。

只调用检索、grounded 和未配置 API 的待接入路径。示例结果与耗时不构成
准确率、召回率、质量提升或独立测试集指标；本地文本 embedding 会实际运行。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_REVISION = "2506260d8cb193ecdb18ac31fc725ffec43f6602"
MODEL = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
OWNER = "recipegen-test-v1"
TEXT_OWNER = "recipegen-text-v1"
SOURCE_QUALITY_WARNING = "本次示例的部分原始标题包含拒答或翻译说明文字；来源一致性通过不等于检索相关性或源数据质量通过。"

NATIVE_COUNTS = """
MATCH (n:RecipeGen {build_id:$build_id,split:'test',dataset_revision:$revision})
WHERE n.kg_import_namespace = $owner
RETURN count(CASE WHEN n:Recipe THEN n END) AS recipes,
 count(CASE WHEN n:Step THEN n END) AS steps
"""
NATIVE_VECTORS = """
MATCH (t:RecipeGenText)-[h:TEXT_OF]->(r:RecipeGen:Recipe)
WHERE t.kg_import_namespace = $text_owner AND t.build_id = $build_id
 AND t.dataset_revision = $revision AND t.split = 'test'
 AND t.embedding_model = $model AND t.embedding_revision = $model_revision
 AND t.index_signature = $signature AND size(t.embedding) = 384
 AND h.kg_import_namespace = $text_owner AND h.build_id = $build_id
 AND h.dataset_revision = $revision AND h.split = 'test'
 AND h.embedding_model = $model AND h.embedding_revision = $model_revision
 AND r.kg_import_namespace = $owner AND r.build_id = $build_id
 AND r.dataset_revision = $revision AND r.split = 'test'
 AND t.source_recipe_id = r.kg_csv_id
RETURN count(DISTINCT t) AS vectors,count(DISTINCT r) AS recipes
"""
NATIVE_RECIPES = """
MATCH (r:RecipeGen:Recipe {build_id:$build_id,split:'test',dataset_revision:$revision})
WHERE r.kg_import_namespace = $owner AND r.kg_csv_id IN $recipe_ids
CALL (r) {
 MATCH (r)-[h:HAS_STEP]->(s:RecipeGen:Step)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
  AND h.dataset_revision = $revision AND h.split = 'test'
  AND s.kg_import_namespace = $owner AND s.build_id = $build_id
  AND s.dataset_revision = $revision AND s.split = 'test'
 WITH DISTINCT s ORDER BY s.order,s.kg_csv_id
 RETURN collect({id:s.kg_csv_id,order:s.order,text:s.text,source_id:s.source_id}) AS steps
}
CALL (r) {
 MATCH (r)-[h:HAS_SOURCE]->(s:RecipeGen:Source)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
  AND h.dataset_revision = $revision AND h.split = 'test'
  AND s.kg_import_namespace = $owner AND s.build_id = $build_id
  AND s.dataset_revision = $revision AND s.split = 'test'
 RETURN collect(DISTINCT {id:s.kg_csv_id,source_id:s.source_id,member:s.member,
  archive:s.archive,sha256:s.sha256,url:s.url,build_id:s.build_id,
  split:s.split,dataset_revision:s.dataset_revision}) AS sources
}
CALL (r) {
 MATCH (r)-[h:HAS_INGREDIENT]->(i:RecipeGen:Ingredient)
 WHERE h.kg_import_namespace = $owner AND h.build_id = $build_id
  AND h.dataset_revision = $revision AND h.split = 'test'
  AND i.kg_import_namespace = $owner AND i.build_id = $build_id
  AND i.dataset_revision = $revision AND i.split = 'test'
 RETURN collect(DISTINCT i.kg_csv_id) AS ingredient_ids
}
CALL (r) {
 MATCH (t:RecipeGenText)-[h:TEXT_OF]->(r)
 WHERE t.kg_import_namespace = $text_owner AND t.build_id = $build_id
  AND t.dataset_revision = $revision AND t.split = 'test'
  AND t.embedding_model = $model AND t.embedding_revision = $model_revision
  AND t.index_signature = $signature AND t.source_recipe_id = r.kg_csv_id
  AND h.kg_import_namespace = $text_owner AND h.build_id = $build_id
  AND h.dataset_revision = $revision AND h.split = 'test'
  AND h.embedding_model = $model AND h.embedding_revision = $model_revision
 RETURN collect(DISTINCT {id:t.id,text_sha256:t.text_sha256,text:t.text,
  source_recipe_id:t.source_recipe_id}) AS embeddings
}
RETURN r.kg_csv_id AS recipe_id,r.title AS title,steps,sources,ingredient_ids,embeddings
"""


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_quality_observation(recipes):
    """仅标记已返回标题里的字面话术，保留原文，不充当语义或全量审计。"""
    markers = ("无法", "请提供", "缺少", "英文名称", "应为")
    flagged = []
    for recipe in recipes:
        title = recipe["title"]
        matched = [marker for marker in markers if marker in title]
        if matched:
            flagged.append({"recipe_id": recipe.get("id", recipe.get("recipe_id")), "title": title,
                            "matched_literal_markers": matched})
    return {"scope": "returned_examples_only", "method": "literal_title_markers",
            "complete_dataset_audit": False, "semantic_quality_verified": False,
            "original_graph_data_modified": False, "flagged_original_titles": flagged}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or parsed.path not in {"", "/"}:
        parser.error("验收只访问 127.0.0.1 上已有的 HTTP 服务")
    from neo4j import GraphDatabase, READ_ACCESS
    from recipegen.config import Settings
    from recipegen.catalog import normalize_ingredient

    settings = Settings.from_env()
    report_path = ROOT / "reports/graphrag-system-smoke.json"
    report = {"schema_version": "recipegen-graphrag-system-smoke-v1", "status": "running",
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "base_url": args.url,
              "scope": "real_http_text_graphrag_and_native_source_readback", "checks": {}, "requests": [],
              "retrieval_examples": [], "native_read_only": True,
              "external_llm_call_attempted": False, "local_generation_requested": False,
              "media_processing_requested": False, "semantic_accuracy_verified": False,
              "quality_or_recall_improvement_verified": False, "held_out_evaluation": False,
              "limitations": ["固定少量功能示例，未评估准确率、Recall@K 或质量提升。",
                              "检索为有限候选；食材规则提及不证明完整配方、库存充足或忌口安全。",
                              "真实 LLM API 尚未联调；待配置路径保留原始步骤。"]}
    opener = build_opener(ProxyHandler({}))
    driver = None

    def check(name, condition):
        report["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def request(path, payload=None):
        tick = time.perf_counter()
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        req = Request(args.url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=180) as response:
                code, body = response.status, json.load(response)
        except HTTPError as error:
            code, body = error.code, json.load(error)
        elapsed = round((time.perf_counter() - tick) * 1000, 3)
        report["requests"].append({"method": "POST" if payload is not None else "GET", "path": path,
                                   "http_status": code, "elapsed_ms": elapsed})
        return code, body, elapsed

    def read(query, params):
        with driver.session(database=settings.neo4j_database, default_access_mode=READ_ACCESS) as session:
            return session.execute_read(lambda tx: [record.data() for record in tx.run(query, params)])

    try:
        check("local_configuration_has_no_enabled_llm", not settings.llm_configured)
        manifest = json.loads((ROOT / "data/graphrag/index-manifest.json").read_text())
        check("verified_full_text_manifest", manifest["status"] == "verified" and manifest["completed_full_scope"] is True)
        check("fixed_test_dataset", manifest["dataset"]["revision"] == DATASET_REVISION
              and manifest["dataset"]["split"] == "test" and manifest["dataset"]["is_demo"] is False)
        check("fixed_multilingual_text_model", manifest["embedding"]["model"] == MODEL
              and manifest["embedding"]["revision"] == MODEL_REVISION and manifest["embedding"]["dimension"] == 384)
        check("manifest_index_online_5898", manifest["index"]["status"] == "ONLINE" and manifest["index"]["count"] == 5898)
        report["manifest"] = {key: manifest[key] for key in ("dataset", "build_id", "signature", "index", "embedding", "encoding")}

        code, status, _ = request("/api/system/status")
        check("system_status_http_200", code == 200)
        check("http_real_fixed_test_graph", status["backend"] == "neo4j" and status["dataset"]["revision"] == DATASET_REVISION
              and status["dataset"]["split"] == "test" and status["dataset"]["is_demo"] is False)
        check("http_original_counts_5898_44802", status["graph"]["recipes"] == 5898 and status["graph"]["steps"] == 44802)
        check("http_semantic_ready", status["retrieval"]["semantic_available"] is True
              and status["retrieval"]["index"]["name"] == manifest["index"]["name"]
              and status["retrieval"]["index"]["status"] == "ONLINE" and status["retrieval"]["index"]["count"] == 5898)
        check("http_api_still_pending", status["generation"]["api_configured"] is False)
        check("http_build_matches_manifest", status["build"]["build_id"] == manifest["build_id"])
        report["system"] = status

        progress_path = ROOT / "reports/multimodal-progress.json"
        media_before = progress_path.read_bytes()
        progress = json.loads(media_before)
        check("media_status_stopped_by_user", progress["status"] == "stopped_by_user" and progress["stage"] == "stopped")
        lock_path = ROOT / ".runtime/multimodal-pipeline.lock"
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                check("media_pipeline_lock_free", True)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                check("media_pipeline_lock_free", False)
        processes = subprocess.run(["/bin/ps", "-axo", "pid=,command="], text=True, capture_output=True, check=True).stdout
        media_workers = [line.split(maxsplit=1)[0] for line in processes.splitlines()
                         if str(ROOT) in line and any(name in line for name in
                            ("run_multimodal_pipeline.py", "run_full_multimodal_batch.py"))]
        check("no_owned_media_worker", not media_workers)
        report["media_processing"] = {"status": progress["status"], "completed_full_scope": progress["completed_full_scope"],
                                       "completed_image_scope": progress["completed_image_scope"],
                                       "original_worker_pid": progress["pid"], "owned_active_workers": media_workers}

        cases = [
            ("chinese_keyword_baseline", "keyword", "清爽的沙拉", [], []),
            ("chinese_semantic_example", "semantic", "清爽的沙拉", [], []),
            ("chinese_hybrid_example", "hybrid", "清爽的沙拉", [], []),
            ("semantic_explicit_filters", "semantic", "番茄沙拉", ["番茄"], ["鸡蛋"]),
            ("hybrid_explicit_and_query_negation", "hybrid", "番茄 不要鸡蛋", ["番茄"], ["鸡蛋"]),
            ("known_keyword_mentions", "keyword", "番茄 鸡蛋", [], []),
        ]
        searches, all_recipes = {}, {}
        for name, mode, query, ingredients, excluded in cases:
            payload = {"query": query, "retrieval_mode": mode, "ingredients": ingredients,
                       "excluded_ingredients": excluded, "limit": 3}
            code, result, elapsed = request("/api/search", payload)
            check(name + "_http_200", code == 200)
            retrieval = result["retrieval"]
            check(name + "_real_mode", result["backend"] == "neo4j" and retrieval["status"] == "ok"
                  and retrieval["mode"] == mode and retrieval["effective_mode"] == mode)
            check(name + "_bounded_disclosure", retrieval["exhaustive"] is False
                  and retrieval["coverage"] == "bounded_candidates" and retrieval["llm_called"] is False)
            check(name + "_count_and_unique_ids", result["count"] == len(result["recipes"])
                  and len({r["id"] for r in result["recipes"]}) == result["count"])
            check(name + "_candidate_count_consistent", retrieval["returned_candidates"] >= retrieval["eligible_candidates"] >= result["count"])
            if name in {"chinese_semantic_example", "chinese_hybrid_example", "known_keyword_mentions"}:
                check(name + "_has_results", result["count"] > 0)
            if mode != "keyword":
                check(name + "_actual_vector_retriever", retrieval["retriever"] == "neo4j_graphrag.VectorCypherRetriever"
                      and retrieval["index"] == manifest["index"]["name"] and retrieval["semantic"]["available"] is True)
            for recipe in result["recipes"]:
                if recipe["id"] in all_recipes:
                    check(name + "_same_original_recipe_" + recipe["id"], all_recipes[recipe["id"]] == recipe)
                all_recipes[recipe["id"]] = recipe
                names = {normalize_ingredient(m["name"]) for m in recipe["ingredient_mentions"]}
                check(name + "_hard_filters_" + recipe["id"], all(normalize_ingredient(i) in names for i in ingredients)
                      and not any(normalize_ingredient(e) in names for e in excluded))
            searches[name] = result
            report["retrieval_examples"].append({"case": name, "request": payload, "elapsed_ms": elapsed,
                "count": result["count"], "returned": [{"recipe_id": r["id"], "title": r["title"], "step_count": len(r["steps"])}
                                                          for r in result["recipes"]],
                "retrieval": retrieval, "graph_evidence": result["graph_evidence"]})
            print(json.dumps({"case": name, "count": result["count"], "elapsed_ms": elapsed}, ensure_ascii=False), flush=True)

        report["source_quality_observation"] = source_quality_observation(all_recipes.values())
        if report["source_quality_observation"]["flagged_original_titles"]:
            report["limitations"].append(SOURCE_QUALITY_WARNING)

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                      connection_timeout=10, max_transaction_retry_time=10)
        params = {"build_id": manifest["build_id"], "revision": DATASET_REVISION, "owner": OWNER,
                  "text_owner": TEXT_OWNER, "model": MODEL, "model_revision": MODEL_REVISION,
                  "signature": manifest["signature"], "recipe_ids": list(all_recipes)}
        counts = read(NATIVE_COUNTS, params)
        check("independent_native_original_counts", counts == [{"recipes": 5898, "steps": 44802}])
        vectors = read(NATIVE_VECTORS, params)
        check("independent_native_vector_recipe_coverage", vectors == [{"vectors": 5898, "recipes": 5898}])
        index = read("SHOW VECTOR INDEXES YIELD name,state,labelsOrTypes,properties,options WHERE name=$name RETURN *",
                     {"name": manifest["index"]["name"]})
        check("independent_native_index_online_schema", len(index) == 1 and index[0]["state"] == "ONLINE"
              and index[0]["labelsOrTypes"] == ["RecipeGenText"] and index[0]["properties"] == ["embedding"]
              and index[0]["options"]["indexConfig"]["vector.dimensions"] == 384
              and str(index[0]["options"]["indexConfig"]["vector.similarity_function"]).lower() == "cosine")
        native = {row["recipe_id"]: row for row in read(NATIVE_RECIPES, params)}
        check("all_returned_recipes_have_independent_native_records", set(native) == set(all_recipes))
        report["native"] = {"original_counts": counts[0], "vector_counts": vectors[0], "index": index[0], "recipe_readback": []}
        for recipe_id, recipe in all_recipes.items():
            row = native[recipe_id]
            native_sources = {s["source_id"]: s for s in row["sources"]}
            check("native_original_steps_" + recipe_id, row["steps"] == recipe["steps"] and row["title"] == recipe["title"])
            check("native_source_id_set_" + recipe_id, set(native_sources) == {s["source_id"] for s in recipe["sources"]})
            check("native_ingredient_id_set_" + recipe_id, set(row["ingredient_ids"]) == {m["id"] for m in recipe["ingredient_mentions"]})
            for source in recipe["sources"]:
                check("native_source_artifact_" + source["source_id"], all(native_sources[source["source_id"]][key] == source.get(key)
                      for key in ("id", "member", "archive", "sha256", "url", "build_id", "split", "dataset_revision")))
            check("native_steps_resolve_to_own_sources_" + recipe_id, all(s["source_id"] in native_sources for s in row["steps"]))
            source_text = row["title"] + "\n" + "\n".join(s["text"] for s in row["steps"])
            text_hash = hashlib.sha256(source_text.encode()).hexdigest()
            check("native_vector_text_hash_" + recipe_id, len(row["embeddings"]) == 1 and row["embeddings"][0]["text"] == source_text
                  and row["embeddings"][0]["text_sha256"] == text_hash)
            report["native"]["recipe_readback"].append({"recipe_id": recipe_id, "title": row["title"],
                "step_ids": [s["id"] for s in row["steps"]], "source_ids": sorted(native_sources),
                "ingredient_ids": sorted(row["ingredient_ids"]), "steps_sha256": _json_hash(row["steps"]),
                "source_text_sha256": text_hash, "embedding_id": row["embeddings"][0]["id"]})
        semantic_evidence_count = 0
        for name, result in searches.items():
            hits = result["graph_evidence"]
            if result["retrieval"]["effective_mode"] == "semantic":
                check(name + "_every_semantic_hit_has_graph_evidence", {h["recipe_id"] for h in hits} == {r["id"] for r in result["recipes"]})
            for hit in hits:
                row = native[hit["recipe_id"]]
                check(name + "_native_expansion_" + hit["recipe_id"],
                      set(hit["step_ids"]) == {s["id"] for s in row["steps"]}
                      and set(hit["source_ids"]) == {s["id"] for s in row["sources"]}
                      and set(hit["ingredient_ids"]) == set(row["ingredient_ids"])
                      and hit["embedding_id"] == row["embeddings"][0]["id"]
                      and hit["text_sha256"] == row["embeddings"][0]["text_sha256"]
                      and hit["source_recipe_id"] == hit["recipe_id"] and hit["namespace"] == TEXT_OWNER
                      and hit["build_id"] == manifest["build_id"] and hit["dataset_revision"] == DATASET_REVISION
                      and hit["split"] == "test" and hit["embedding_model"] == MODEL and hit["embedding_revision"] == MODEL_REVISION)
                semantic_evidence_count += 1
        check("actual_semantic_graph_evidence_readback", semantic_evidence_count > 0)
        report["native"]["semantic_evidence_records_checked"] = semantic_evidence_count

        recipe_id = searches["chinese_semantic_example"]["recipes"][0]["id"]
        payload = {"question": "按原始顺序整理这道食谱，保留每个步骤和来源", "recipe_ids": [recipe_id],
                   "mode": "grounded", "retrieval_mode": "hybrid", "limit": 1}
        code, generated, _ = request("/api/generate", payload)
        check("grounded_generate_http_200", code == 200 and generated["status"] == "ok")
        check("grounded_full_original_steps", generated["recipes"][0]["steps"] == native[recipe_id]["steps"]
              and all(s["text"] in generated["answer"] for s in native[recipe_id]["steps"]))
        check("grounded_no_llm_call", generated["generation"]["llm_called"] is False and generated["generation"]["effective_mode"] == "grounded")
        for step in native[recipe_id]["steps"]:
            check("grounded_exact_step_source_" + step["id"], any(e["kind"] == "step" and e["recipe_id"] == recipe_id
                  and e["graph_id"] == step["id"] and e["text"] == step["text"] and e["source_id"] == step["source_id"]
                  for e in generated["evidence"]))
        code, pending, _ = request("/api/generate", {**payload, "mode": "api"})
        check("pending_api_http_200", code == 200 and pending["status"] == "ok")
        check("pending_api_original_answer_no_call", pending["generation"]["pending_api"] is True
              and pending["generation"]["api_status"] == "not_configured" and pending["generation"]["llm_called"] is False
              and pending["generation"]["effective_mode"] == "grounded" and pending["generation"]["fallback"] is False
              and pending["answer"] == generated["answer"] and pending["recipes"][0]["steps"] == native[recipe_id]["steps"])
        report["generation"] = {"grounded": generated, "pending_api": pending}
        for name, path, invalid in [
            ("invalid_limit_422", "/api/search", {"query": "番茄", "limit": 0}),
            ("invalid_retrieval_mode_422", "/api/search", {"query": "番茄", "retrieval_mode": "unsupported"}),
            ("invalid_generation_mode_422", "/api/generate", {**payload, "mode": "unsupported"}),
        ]:
            check(name, request(path, invalid)[0] == 422)
        check("media_snapshot_unchanged", progress_path.read_bytes() == media_before)
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
        check("private_credentials_not_in_report", all(not secret or secret not in serialized
              for secret in (settings.neo4j_password, settings.llm_api_key)))
        report["status"] = "verified"
        report["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    except Exception as error:
        report["status"] = "failed"
        # 原生连接错误可能带有 URL/配置；只保存类型和我们自己的检查名。
        report["failure"] = {"type": type(error).__name__, "check": str(error) if isinstance(error, AssertionError) else None}
    finally:
        if driver is not None:
            driver.close()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        def redact(value):
            if isinstance(value, str):
                for secret in (settings.neo4j_password, settings.llm_api_key):
                    if secret:
                        value = value.replace(secret, "[REDACTED]")
                return value
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            return value
        report = redact(report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": sum(report["checks"].values()),
                      "checks_total": len(report["checks"]), "report": str(report_path),
                      "failure": report.get("failure")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
