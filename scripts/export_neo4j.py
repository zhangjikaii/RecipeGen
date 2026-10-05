#!/usr/bin/env python3
"""Read-only, explicit-schema Neo4j export into RecipeGen canonical JSON.

No command-line credentials, guessed labels, arbitrary Cypher, or writes are accepted here.
Use --inspect first, then fill docs/neo4j_mapping.example.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


class ExportError(ValueError):
    """A source value cannot be exported without inventing information."""


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[^\W\d]\w*", value, re.UNICODE):
        raise ExportError(f"非法 label/relationship 标识符：{value!r}；请使用字母、数字、下划线或中文。")
    return f"`{value}`"


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{field} 缺少非空字符串，不能自动补齐。")
    return value.strip()


def string_array(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ExportError(f"{field} 必须是字符串数组；字符串分隔格式需先显式清洗。")
    result = [required_string(item, field) for item in value]
    if nonempty and not result:
        raise ExportError(f"{field} 没有记录，不能自动补齐。")
    if field == "steps":
        return result
    return list(dict.fromkeys(result))


def read_rows(session: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    # execute_read sets READ routing. Every query in this script is fixed MATCH/CALL/RETURN.
    return session.execute_read(lambda tx: [record.data() for record in tx.run(query, **params)])


def inspect_schema(session: Any, sample_count: int) -> dict[str, Any]:
    labels = [row["label"] for row in read_rows(session, "CALL db.labels() YIELD label RETURN label ORDER BY label")]
    relationships = [row["relationshipType"] for row in read_rows(
        session, "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
    )]
    samples = {}
    for label in labels:
        rows = read_rows(
            session,
            f"MATCH (n:{identifier(label)}) RETURN elementId(n) AS record_id, keys(n) AS property_keys LIMIT $count",
            count=sample_count,
        )
        samples[label] = rows
    return {"labels": labels, "relationship_types": relationships, "samples": samples,
            "note": "只读结构检查；样本只显示记录标识和属性键。属性类型和关系方向仍需检查。"}


def field_value(session: Any, record: dict[str, Any], spec: Any, field: str) -> Any:
    if not isinstance(spec, dict):
        raise ExportError(f"映射 fields.{field} 必须是对象。")
    if spec.get("unrecorded") is True:
        if field == "minutes":
            return None
        if field in {"seasonings", "tags"}:
            return []
        raise ExportError(f"{field} 是必要字段，不支持 unrecorded。")
    keys = spec.get("property_fallbacks", [spec.get("property")])
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ExportError(f"映射 fields.{field} 必须声明 property 或非空 property_fallbacks。")

    def value_from(properties: dict[str, Any]) -> Any:
        for key in keys:
            value = properties.get(key)
            if value is not None and value != "":
                return value
        raise ExportError(f"原记录 {record['record_id']} 的候选属性 {keys!r} 均缺失（映射到 {field}）。")

    if "relationship" not in spec:
        # Explicit null cooking time means unknown, never a guessed duration.
        if field == "minutes":
            try:
                return value_from(record["properties"])
            except ExportError:
                if any(key in record["properties"] and record["properties"][key] is None for key in keys):
                    return None
                raise
        return value_from(record["properties"])
    relation = identifier(spec["relationship"])
    label = f":{identifier(spec['target_label'])}" if spec.get("target_label") else ""
    direction = spec.get("direction")
    if direction == "out":
        pattern = f"(r)-[e:{relation}]->(n{label})"
    elif direction == "in":
        pattern = f"(r)<-[e:{relation}]-(n{label})"
    else:
        raise ExportError(f"fields.{field}.direction 必须显式填 out 或 in。")
    rows = read_rows(
        session,
        f"MATCH {pattern} WHERE elementId(r) = $record_id "
        "RETURN properties(n) AS properties, properties(e) AS relationship_properties, elementId(n) AS node_id",
        record_id=record["record_id"],
    )
    if field == "steps":
        order_key = spec.get("order_property")
        if not isinstance(order_key, str) or not order_key:
            raise ExportError("steps 关系映射必须声明 order_property，避免猜测烹饪顺序。")
        order_on = spec.get("order_on", "node")
        if order_on not in {"node", "relationship"}:
            raise ExportError("steps.order_on 必须是 node 或 relationship。")
        properties_key = "properties" if order_on == "node" else "relationship_properties"
        orders = [row[properties_key].get(order_key) for row in rows]
        if any(isinstance(order, bool) or not isinstance(order, (int, float)) or not math.isfinite(order) for order in orders):
            raise ExportError(f"原记录 {record['record_id']} 的步骤缺少有效数值顺序 {order_key!r}。")
        if len(set(orders)) != len(orders):
            raise ExportError(f"原记录 {record['record_id']} 的步骤顺序重复。")
        rows.sort(key=lambda row: row[properties_key][order_key])
    else:
        rows.sort(key=lambda row: row["node_id"])
    return [value_from(row["properties"]) for row in rows]


def export_source(record: dict[str, Any], spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ExportError("source 映射必须是对象。")
    if spec.get("internal_record") is True:
        id_property = spec.get("recipe_id_property")
        source_record_id = record["properties"].get(id_property) if id_property else record["record_id"]
        if source_record_id is None or source_record_id == "":
            raise ExportError(f"原记录缺少内部来源标识属性 {id_property!r}。")
        return {"id": f"neo4j:{source_record_id}",
                "title": f"原图谱记录 {source_record_id}", "url": None}
    props = record["properties"]
    source_id = props.get(spec.get("id_property"))
    title = props.get(spec.get("title_property"))
    url_key = spec.get("url_property")
    if url_key and url_key not in props:
        raise ExportError(f"原记录缺少来源 URL 属性 {url_key!r}；若原图谱无 URL，请将 url_property 设为 null。")
    url = props.get(url_key) if url_key else None
    if url is not None and not isinstance(url, str):
        raise ExportError("source.url 必须是字符串或 null。")
    return {"id": required_string(source_id, "source.id"),
            "title": required_string(title, "source.title"), "url": url}


def export_graph(session: Any, mapping: dict[str, Any], limit: int | None) -> dict[str, Any]:
    label = identifier(mapping.get("recipe_label"))
    fields = mapping.get("fields")
    required = {"id", "name", "ingredients", "seasonings", "minutes", "tags", "steps"}
    if not isinstance(fields, dict) or not required.issubset(fields):
        raise ExportError(f"fields 必须包含：{', '.join(sorted(required))}。")
    dataset = mapping.get("dataset")
    if not isinstance(dataset, dict):
        raise ExportError("映射文件必须有 dataset 元数据对象。")
    for key in ("id", "name", "description", "source", "license"):
        required_string(dataset.get(key), f"dataset.{key}")
    if dataset.get("is_demo") is not False:
        raise ExportError("真实图谱导出的 dataset.is_demo 必须显式为 false。")
    query = f"MATCH (r:{label}) RETURN elementId(r) AS record_id, properties(r) AS properties ORDER BY record_id"
    if limit is not None:
        query += " LIMIT $limit"
    records = read_rows(session, query, **({"limit": limit} if limit is not None else {}))
    if not records:
        raise ExportError("指定 label 没有菜谱节点；请先检查 schema 和数据库名称。")
    recipes = []
    seen_ids = set()
    for record in records:
        try:
            values = {name: field_value(session, record, fields[name], name) for name in sorted(required)}
            recipe_id = values["id"]
            if isinstance(recipe_id, int) and not isinstance(recipe_id, bool):
                recipe_id = str(recipe_id)
            recipe_id = required_string(recipe_id, "id")
            if recipe_id in seen_ids:
                raise ExportError(f"重复菜谱 id：{recipe_id}。")
            seen_ids.add(recipe_id)
            minutes = values["minutes"]
            if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0):
                raise ExportError("minutes 必须是正整数分钟或 null；字符串或其他单位需要先显式整理。")
            recipes.append({
                "id": recipe_id,
                "name": required_string(values["name"], "name"),
                "ingredients": string_array(values["ingredients"], "ingredients", nonempty=True),
                "seasonings": string_array(values["seasonings"], "seasonings"),
                "minutes": minutes,
                "tags": string_array(values["tags"], "tags"),
                "steps": string_array(values["steps"], "steps", nonempty=True),
                "source": export_source(record, mapping.get("source")),
            })
        except ExportError as exc:
            raise ExportError(f"原节点 {record['record_id']} 导出失败：{exc}") from exc
    aliases = mapping.get("ingredient_aliases", {})
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip() for k, v in aliases.items()):
        raise ExportError("ingredient_aliases 必须是非空字符串到非空字符串的映射。")
    return {"schema_version": 1, "dataset": dataset, "ingredient_aliases": aliases, "recipes": recipes}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="只读检查 Neo4j 或按显式映射导出 RecipeGen JSON；不执行写入。")
    result.add_argument("--inspect", action="store_true", help="只显示 labels、关系类型和属性键，不导出菜谱")
    result.add_argument("--mapping", type=Path, help="用户核对过的 schema 映射 JSON")
    result.add_argument("--output", type=Path, help="输出 canonical graph.json；已有文件会拒绝覆盖")
    result.add_argument("--limit", type=int, help="仅导出前 N 个节点；省略则导出全部")
    result.add_argument("--sample-count", type=int, default=3, help="inspect 每个 label 的属性键样本数（1~20）")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.inspect and (args.mapping is None or args.output is None):
        parser().error("导出需要 --mapping 和 --output；或使用 --inspect。")
    if args.inspect and (args.mapping is not None or args.output is not None):
        parser().error("--inspect 与导出选项分开运行。")
    if args.limit is not None and args.limit <= 0:
        parser().error("--limit 必须大于 0。")
    if not 1 <= args.sample_count <= 20:
        parser().error("--sample-count 必须为 1~20。")
    if args.output and args.output.exists():
        parser().error("输出文件已存在，请换一个路径，避免覆盖既有导出。")
    try:
        # Validate the top-level mapping and label before a network connection.
        mapping = None
        if args.mapping:
            mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                raise ExportError("映射文件顶层必须为对象。")
            identifier(mapping.get("recipe_label"))
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            raise ExportError("请设置 NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD 环境变量；脚本不读取或打印密码。")
        try:
            from neo4j import GraphDatabase, READ_ACCESS
        except ImportError as exc:
            raise ExportError("缺少 neo4j 驱动，请先安装项目 requirements.txt。") from exc
        with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10.0) as driver:
            with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
                if args.inspect:
                    print(json.dumps(inspect_schema(session, args.sample_count), ensure_ascii=False, indent=2))
                    return 0
                payload = export_graph(session, mapping, args.limit)
        project_dir = str(Path(__file__).resolve().parents[1])
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        from recipegen.models import GraphDocument
        try:
            GraphDocument.model_validate(payload)
        except ValueError as exc:
            raise ExportError(f"导出结果不符合项目数据规范：{exc}") from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create also prevents a race from overwriting an existing file.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"已只读导出 {len(payload['recipes'])} 条菜谱至 {args.output}；尚未验证真实推荐效果。")
        return 0
    except (ExportError, OSError, json.JSONDecodeError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Database/driver exceptions can contain server details; do not expose credentials or queries.
        print(f"Neo4j 操作失败（{type(exc).__name__}）；请检查连接、权限和映射。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
