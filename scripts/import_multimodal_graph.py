#!/usr/bin/env python3
"""Validate a multimodal extension bundle; --import explicitly writes Neo4j.

The default and --validate-only never load credentials or access the network.
Original CSV/source contracts are reused from the test-graph importer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipegen.multimodal_graph import (
    MultimodalValidationError, MultimodalVerificationError,
    import_bundle, load_base_index, parse_json, validate_bundle,
)
from scripts.import_test_graph_neo4j import ImportValidationError, load_bundle as load_base_graph


def connect_and_import(bundle: Any, config_path: Path | None, batch_size: int) -> dict[str, Any]:
    # Import connection helpers only on the explicitly requested write path.
    from scripts.query_test_graph import PRIVATE_CONFIG, load_connection
    from neo4j import GraphDatabase, WRITE_ACCESS
    connection = load_connection(config_path or PRIVATE_CONFIG)
    with GraphDatabase.driver(connection.uri, auth=(connection.user, connection.password),
                              connection_timeout=10, max_transaction_retry_time=10) as driver:
        driver.verify_connectivity()
        with driver.session(database=connection.database, default_access_mode=WRITE_ACCESS) as session:
            return import_bundle(session, bundle, batch_size=batch_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--nodes-csv", type=Path, default=PROJECT_ROOT / "data/test_graph/neo4j_import/nodes.csv")
    parser.add_argument("--relationships-csv", type=Path, default=PROJECT_ROOT / "data/test_graph/neo4j_import/relationships.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="离线校验（默认）；不读取凭据、不连接数据库")
    mode.add_argument("--import", dest="import_requested", action="store_true", help="显式向现有 verified 活动图谱追加扩展")
    parser.add_argument("--config", type=Path, default=None, help="私密 Neo4j 配置，默认复用 active-neo4j.json 接口")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.batch_size <= 5000:
            raise MultimodalValidationError("batch-size 必须为 1～5000")
        standard_nodes = PROJECT_ROOT / "data/test_graph/neo4j_import/nodes.csv"
        standard_edges = PROJECT_ROOT / "data/test_graph/neo4j_import/relationships.csv"
        if args.nodes_csv.resolve() == standard_nodes.resolve() and args.relationships_csv.resolve() == standard_edges.resolve():
            original = load_base_index(PROJECT_ROOT)
        else:
            # Custom CSVs require the full canonical contract; they are not
            # silently treated as files covered by this project's old report.
            original = load_base_graph(args.nodes_csv, args.relationships_csv)
        bundle = validate_bundle(parse_json(args.bundle.read_text(encoding="utf-8")), original)
        if args.import_requested:
            report = connect_and_import(bundle, args.config, args.batch_size)
        else:
            report = {"status": "validated", "validation_only": True, "connected_to_neo4j": False,
                      "input": bundle.summary(), "semantic_accuracy_verified": False,
                      "note": "仅通过离线 bundle/CSV/来源/时间契约校验；未写库、未实测模型。"}
        code = 0
    except MultimodalVerificationError as error:
        report, code = error.report, 1
    except (MultimodalValidationError, ImportValidationError, OSError) as error:
        report, code = {"status": "unverified", "validation_only": not args.import_requested,
                        "connected_to_neo4j": False, "error": str(error)}, 1
    except Exception as error:
        # Do not serialize driver/config exceptions that might expose a URI/password.
        report, code = {"status": "unverified", "validation_only": not args.import_requested,
                        "error_type": type(error).__name__,
                        "error": "扩展导入或回读未完成；检查私密连接配置、活动构建与来源契约。凭据不会输出。"}, 1
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
