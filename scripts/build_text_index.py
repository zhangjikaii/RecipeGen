#!/usr/bin/env python3
"""从真实 Neo4j 原文建立可续跑 E5 文本索引；不处理图片视频，不调用 LLM。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from recipegen.catalog import RecipeCatalog, _rows, NAMESPACE, REVISION as DATASET_REVISION
from recipegen.config import Settings
from recipegen.text_embeddings import MODEL, REVISION, DIMENSION, MAX_TOKENS, validate_vector

TEXT_OWNER = "recipegen-text-v1"
OUT = ROOT / "data/graphrag"
PROGRESS = ROOT / "reports/graphrag-index-progress.json"
CORPUS_QUERY = """// recipegen-text:original-corpus
MATCH (r:RecipeGen:Recipe {build_id:$build_id,split:'test',dataset_revision:$dataset_revision})
WHERE r.kg_import_namespace=$owner
CALL (r) {
 MATCH (r)-[h:HAS_STEP]->(s:RecipeGen:Step)
 WHERE h.kg_import_namespace=$owner AND h.build_id=$build_id AND h.split='test'
   AND h.dataset_revision=$dataset_revision AND s.kg_import_namespace=$owner
   AND s.build_id=$build_id AND s.split='test' AND s.dataset_revision=$dataset_revision
 WITH s ORDER BY s.order,s.kg_csv_id
 RETURN collect({id:s.kg_csv_id,order:s.order,text:s.text,source_id:s.source_id}) AS steps
}
CALL (r) {
 MATCH (r)-[h:HAS_SOURCE]->(s:RecipeGen:Source)
 WHERE h.kg_import_namespace=$owner AND h.build_id=$build_id AND h.split='test'
   AND h.dataset_revision=$dataset_revision AND s.kg_import_namespace=$owner
   AND s.build_id=$build_id AND s.split='test' AND s.dataset_revision=$dataset_revision
 RETURN collect(s.source_id) AS source_ids
}
RETURN r.kg_csv_id AS recipe_id,r.title AS title,steps,source_ids
ORDER BY recipe_id
"""
IMPORT_QUERY = """// recipegen-text:add-derived-vectors
UNWIND $rows AS row
MATCH (r:RecipeGen:Recipe {kg_csv_id:row.recipe_id,build_id:$build_id,split:'test',dataset_revision:$dataset_revision})
WHERE r.kg_import_namespace=$owner
MERGE (n:RecipeGenText:TextEmbedding {id:row.embedding_id})
SET n.kg_import_namespace=$text_owner,n.source_recipe_id=row.recipe_id,n.build_id=$build_id,
    n.dataset_revision=$dataset_revision,n.split='test',n.embedding_model=$model,n.embedding_revision=$model_revision,
    n.text_sha256=row.text_sha256,n.text=row.text,n.embedding=row.embedding,
    n.max_tokens=$max_tokens,n.truncated=row.truncated,n.token_count=row.token_count,
    n.index_signature=$signature,n.semantics='derived_text_embedding'
MERGE (n)-[h:TEXT_OF]->(r)
SET h.kg_import_namespace=$text_owner,h.build_id=$build_id,h.dataset_revision=$dataset_revision,h.split='test',
    h.embedding_model=$model,h.embedding_revision=$model_revision
RETURN count(DISTINCT n) AS imported
"""


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="有界本地验证，不激活完整索引")
    parser.add_argument("--import", dest="import_graph", action="store_true", help="显式新增派生节点、TEXT_OF关系和向量索引")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64 or args.limit is not None and args.limit <= 0:
        parser.error("batch-size须为1～64，limit须为正整数")
    if args.limit and args.import_graph:
        parser.error("有界样本不能激活完整图谱索引，请去掉--limit")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ("smoke" if args.limit else "full")
    target.mkdir(exist_ok=True)
    started = time.monotonic()
    catalog = RecipeCatalog(settings=Settings.from_env())
    progress = None
    try:
        status = catalog.status()
        build = status["build"]
        rows = catalog._read(lambda tx: _rows(tx, CORPUS_QUERY, catalog._parameters(build)))
        if len(rows) != status["graph"]["recipes"] or len({r["recipe_id"] for r in rows}) != len(rows):
            raise ValueError("原文记录范围或去重核验失败")
        for row in rows:
            if not row["title"] or any(step["source_id"] not in row["source_ids"] for step in row["steps"]):
                raise ValueError("原步骤来源核验失败")
            row["text"] = row["title"] + "\n" + "\n".join(step["text"] for step in row["steps"])
            row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        corpus_sha = fingerprint([{key: r[key] for key in ("recipe_id", "text_sha256")} for r in rows])
        signature = fingerprint({"build_id": build["build_id"], "corpus_sha256": corpus_sha,
                                 "model": MODEL, "revision": REVISION, "max_tokens": MAX_TOKENS,
                                 "prefix": "passage: ", "pooling": "attention_mask_mean", "normalization": "l2"})
        if args.limit:
            rows = rows[:args.limit]
        corpus_path = target / "corpus.jsonl"
        corpus_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        cache = target / "embeddings.jsonl"
        completed = {}
        if cache.is_file():
            for line in cache.read_text().splitlines():
                value = json.loads(line)
                validate_vector(value["embedding"])
                if value["signature"] != signature or value["model_revision"] != REVISION:
                    raise ValueError("缓存模型/原文签名不一致，请为新版本使用新的输出目录")
                completed[value["recipe_id"]] = value
        expected = {r["recipe_id"]: r for r in rows}
        if any(key not in expected or value["text_sha256"] != expected[key]["text_sha256"] for key, value in completed.items()):
            raise ValueError("缓存食谱或原文校验不一致")
        progress = {"status": "encoding", "stage": "text_embeddings", "pid": os.getpid(),
                    "scope": "test_recipe_text", "expected_recipes": len(rows), "encoded_recipes": len(completed),
                    "model": MODEL, "revision": REVISION, "dimension": DIMENSION, "device": args.device,
                    "completed_full_scope": False, "media_processing_started": False, "llm_called": False}
        def update(**values):
            progress.update(values, updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic()-started, 2))
            write_json(PROGRESS, progress)
        update()
        pending = [row for row in rows if row["recipe_id"] not in completed]
        if pending:
            import torch
            from local_text_embedding import load_encoder, encode
            tokenizer, model = load_encoder()
            if args.device == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS不可用，可用--device cpu续跑")
            model.to(args.device)
            with cache.open("a") as output:
                for offset in range(0, len(pending), args.batch_size):
                    batch = pending[offset:offset + args.batch_size]
                    texts = [r["text"] for r in batch]
                    token_lengths = [len(value) for value in tokenizer(["passage: " + text for text in texts], truncation=False)["input_ids"]]
                    vectors = encode(texts, tokenizer, model, prefix="passage: ", device=args.device)
                    for row, vector, tokens in zip(batch, vectors, token_lengths):
                        value = {"recipe_id": row["recipe_id"], "text": row["text"], "text_sha256": row["text_sha256"],
                                 "embedding_id": "text:" + fingerprint([signature, row["recipe_id"]])[:32],
                                 "signature": signature, "model_revision": REVISION, "embedding": vector,
                                 "truncated": tokens > MAX_TOKENS, "token_count": tokens}
                        output.write(json.dumps(value, ensure_ascii=False) + "\n")
                        completed[row["recipe_id"]] = value
                    output.flush()
                    update(encoded_recipes=len(completed))
                    if offset % (args.batch_size * 10) == 0:
                        print(json.dumps({"encoded": len(completed), "total": len(rows), "seconds": progress["elapsed_seconds"]}), flush=True)
            del model
            if args.device == "mps":
                torch.mps.empty_cache()
        manifest = {"schema_version": 1, "status": "encoded", "namespace": TEXT_OWNER,
                    "dataset": {"id": "RUOXUAN123/RecipeGen", "revision": DATASET_REVISION, "split": "test", "is_demo": False},
                    "build_id": build["build_id"], "signature": signature, "corpus_sha256": corpus_sha,
                    "index": {"name": "recipegen_text_e5_" + signature[:12], "label": "RecipeGenText", "property": "embedding",
                              "dimensions": DIMENSION, "similarity_function": "cosine", "count": len(rows)},
                    "embedding": {"model": MODEL, "revision": REVISION, "dimension": DIMENSION, "max_tokens": MAX_TOKENS,
                                  "pooling": "attention_mask_mean", "normalization": "l2", "query_prefix": "query: ", "passage_prefix": "passage: "},
                    "encoding": {"records": len(completed), "truncated_records": sum(v["truncated"] for v in completed.values()),
                                 "original_steps_preserved_in_graph": True},
                    "source": {"backend": "neo4j", "build_content_sha256": build["content_sha256"]},
                    "completed_full_scope": not args.limit and len(completed) == status["graph"]["recipes"],
                    "semantic_accuracy_verified": False, "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(target / "manifest.json", manifest)
        if args.import_graph:
            update(status="importing", stage="derived_graph_import")
            driver = catalog._connect()
            database = catalog._connection.database
            params = {"owner": NAMESPACE, "text_owner": TEXT_OWNER, "build_id": build["build_id"],
                      "dataset_revision": DATASET_REVISION, "model": MODEL, "model_revision": REVISION,
                      "max_tokens": MAX_TOKENS, "signature": signature}
            with driver.session(database=database) as session:
                session.run("CREATE CONSTRAINT recipegen_text_id IF NOT EXISTS FOR (n:RecipeGenText) REQUIRE n.id IS UNIQUE").consume()
                values = [completed[row["recipe_id"]] for row in rows]
                imported = 0
                for offset in range(0, len(values), 100):
                    batch = values[offset:offset+100]
                    count = session.execute_write(lambda tx: tx.run(IMPORT_QUERY, rows=batch, **params).single()["imported"])
                    if count != len(batch):
                        raise ValueError("派生向量与原食谱连接数量不一致")
                    imported += count
                    update(imported_recipes=imported)
                name = manifest["index"]["name"]  # 固定安全前缀 + SHA，不能由用户文本注入。
                session.run(f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:RecipeGenText) ON (n.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}}}").consume()
                session.run("CALL db.awaitIndexes(120)").consume()
                index = session.run("SHOW VECTOR INDEXES YIELD name,state,labelsOrTypes,properties,options WHERE name=$name RETURN *", name=name).single()
                if not index or index["state"] != "ONLINE" or index["labelsOrTypes"] != ["RecipeGenText"] or index["properties"] != ["embedding"]:
                    raise ValueError("向量索引未ONLINE或字段不符")
                check = session.run("""MATCH (n:RecipeGenText)-[h:TEXT_OF]->(r:RecipeGen:Recipe)
WHERE n.kg_import_namespace=$text_owner AND n.index_signature=$signature
  AND n.build_id=$build_id AND n.dataset_revision=$dataset_revision AND n.split='test'
  AND n.embedding_model=$model AND n.embedding_revision=$model_revision AND size(n.embedding)=384
  AND h.kg_import_namespace=$text_owner AND h.build_id=$build_id AND h.dataset_revision=$dataset_revision AND h.split='test'
  AND r.kg_import_namespace=$owner AND r.build_id=$build_id AND r.dataset_revision=$dataset_revision AND r.split='test'
  AND n.source_recipe_id=r.kg_csv_id
RETURN count(DISTINCT n) AS vectors,count(DISTINCT r) AS recipes""", **params).single()
                if check["vectors"] != len(rows) or check["recipes"] != len(rows):
                    raise ValueError("全量向量和原食谱原生回读不一致")
                manifest["index"]["status"] = index["state"]
                manifest["status"] = "verified"
                manifest["verification"] = {"vectors": check["vectors"], "recipes": check["recipes"], "native_readback": True}
                write_json(OUT / "index-manifest.json", manifest)
                write_json(ROOT / "reports/graphrag-index-verification.json", manifest)
        update(status="verified" if args.import_graph else "encoded", stage="complete",
               completed_full_scope=manifest["completed_full_scope"], truncated_records=manifest["encoding"]["truncated_records"])
        print(json.dumps({"status": progress["status"], "encoded": len(completed), "full_scope": manifest["completed_full_scope"],
                          "index": manifest["index"], "seconds": progress["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        if progress is not None:
            update(status="failed", stage="stopped_after_error", error_type=type(error).__name__,
                   completed_full_scope=False)
        raise
    finally:
        catalog.close()


if __name__ == "__main__":
    main()
