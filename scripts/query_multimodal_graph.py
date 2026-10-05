#!/usr/bin/env python3
"""English query -> local CLIP512 -> read-only visual candidates + source facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipegen.multimodal_query import MultimodalQueryError, apply_evidence_selection, search_session
from query_test_graph import PRIVATE_CONFIG, QueryError, load_connection


def connect_and_search(connection, embedding: list[float], **request) -> dict:
    from neo4j import GraphDatabase, READ_ACCESS
    with GraphDatabase.driver(connection.uri, auth=(connection.user, connection.password),
                              connection_timeout=10, max_transaction_retry_time=10) as driver:
        driver.verify_connectivity()
        with driver.session(database=connection.database, default_access_mode=READ_ACCESS) as session:
            return search_session(session, embedding, **request)


def run_query(query: str, *, connection, models, limit: int = 5,
              candidate_limit: int = 100, llm: bool = False, search=connect_and_search) -> dict:
    """可注入本地模型/只读连接作离线合同测试，不生成缺失事实。"""
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 500:
        raise MultimodalQueryError("query 必须为 1～500 个字符的英文检索描述。")
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1 <= limit <= 20:
        raise MultimodalQueryError("limit 必须为 1～20。")
    if isinstance(candidate_limit,bool) or not isinstance(candidate_limit,int) or not limit <= candidate_limit <= 1000:
        raise MultimodalQueryError("candidate_limit 必须介于 limit 与 1000。")
    query = query.strip()
    space = models.embedding_space
    if space.get("dimension") != 512:
        raise MultimodalQueryError("本地模型必须提供 CLIP512 embedding 空间。")
    vectors = models.text_embeddings([query])
    if not isinstance(vectors,list) or len(vectors) != 1:
        raise MultimodalQueryError("本地 CLIP 未返回唯一实际查询向量。")
    report = search(connection, vectors[0], query=query, limit=limit, candidate_limit=candidate_limit,
                    embedding_model=space["model"], embedding_revision=space["revision"])
    if llm and report["results"]:
        report = apply_evidence_selection(report, models.select_evidence(query,report["results"]))
    report["local_model"] = {"embedding":dict(space), "selector":models.model_info if report["llm_called"] else None,
                             "calls":dict(models.calls)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="英文视觉/动作描述；使用真实本地 CLIP text embedding")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--llm", action="store_true", help="可选本地 VLM 仅从候选中选择已有证据 ID")
    parser.add_argument("--config", type=Path, default=PRIVATE_CONFIG)
    parser.add_argument("--output", "--report", dest="output", type=Path, help="保存实际只读检索报告 JSON")
    args = parser.parse_args(argv)
    models = None
    try:
        if not args.query.strip() or len(args.query.strip()) > 500 or not 1 <= args.limit <= 20 or not args.limit <= args.candidate_limit <= 1000:
            raise MultimodalQueryError("query 必须为 1～500 字符，limit 1～20，candidate-limit 介于 limit 与 1000。")
        connection = load_connection(args.config)
        from recipegen.multimodal_models import LocalMultimodalModels
        models = LocalMultimodalModels(ROOT, load_vision=bool(args.llm))
        report = run_query(args.query, connection=connection, models=models, limit=args.limit,
                           candidate_limit=args.candidate_limit, llm=args.llm)
        code = 0
    except (MultimodalQueryError, QueryError) as error:
        called = bool(models is not None and models.calls.get("vision",0))
        report = {"status":"failed", "read_only":True, "llm_called":called, "error":str(error)}
        code = 1
    except Exception as error:
        # 驱动/模型异常可能含连接或本地路径，保留类型，不输出凭据与原始异常。
        called = bool(models is not None and models.calls.get("vision",0))
        report = {"status":"failed", "read_only":True, "llm_called":called, "error_type":type(error).__name__,
                  "error":"本地多模态只读检索未完成；检查模型、服务、私密配置、活动构建和向量索引。"}
        code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
