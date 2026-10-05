from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .evaluate import evaluate
from .graph import Neo4jGraphStore, SQLiteGraphStore, load_document
from .models import RecommendRequest
from .pipeline import Recommender


def main() -> int:
    parser = argparse.ArgumentParser(description="RecipeGen 图谱检索与结构化生成系统")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="启动本地网页和 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    demo = sub.add_parser("demo", help="执行一次推荐并输出 JSON")
    demo.add_argument("--question", default="我有番茄和鸡蛋，20分钟内能做什么，不吃辣")
    demo.add_argument("--llm", action="store_true")
    demo.add_argument("--output", type=Path)
    imp = sub.add_parser("import", help="导入规范 JSON 到本地图谱；替换所选本地数据库")
    imp.add_argument("path", type=Path)
    imp.add_argument("--db", type=Path)
    ev = sub.add_parser("evaluate", help="执行手工功能评测集")
    ev.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation.json")
    ev.add_argument("--cases", type=Path)
    ev.add_argument("--llm", action="store_true")
    sub.add_parser("inspect-neo4j", help="只读查看远程 Neo4j labels、关系及属性键")
    seed = sub.add_parser("seed-neo4j", help="显式导入 RG* 专属节点，替换该库已有 RG* 数据")
    seed.add_argument("path", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "serve":
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0
    if args.command == "import":
        result = SQLiteGraphStore(args.db or settings.db_path).import_document(load_document(args.path))
    elif args.command in {"inspect-neo4j", "seed-neo4j"}:
        store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database)
        try:
            result = store.inspect() if args.command == "inspect-neo4j" else store.import_document(load_document(args.path))
        finally:
            store.close()
    else:
        service = Recommender(settings)
        try:
            result = service.recommend(RecommendRequest(question=args.question, use_llm=args.llm)) if args.command == "demo" else evaluate(service, args.cases, args.llm)
        finally:
            if hasattr(service.store, "close"):
                service.store.close()
    output = getattr(args, "output", None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.command == "evaluate":
        print(json.dumps({"total": result["total"], "passed": result["passed"], "summary": result["summary"], "llm_calls": result["successful_llm_calls"], "report": str(output)}, ensure_ascii=False, indent=2))
        return 0 if result["passed"] == result["total"] else 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
