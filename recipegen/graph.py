from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from .models import Constraints, Dataset, GraphDocument, Recipe


def load_document(path: str | Path) -> GraphDocument:
    return GraphDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))


def node_id(kind: str, value: str) -> str:
    return f"{kind}:{hashlib.sha256(value.encode()).hexdigest()[:20]}"


def build_graph(document: GraphDocument) -> tuple[list[dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(identifier: str, label: str, kind: str, properties: dict) -> None:
        nodes[identifier] = {"id": identifier, "label": label, "type": kind, "properties": properties}

    def add_edge(source: str, target: str, label: str) -> None:
        edges.append({"id": node_id("edge", f"{source}|{label}|{target}"), "source": source, "target": target, "label": label})

    for recipe in document.recipes:
        rid = f"recipe:{recipe.id}"
        add_node(rid, recipe.name, "Recipe", {"recipe_id": recipe.id, "minutes": recipe.minutes})
        for relation, names in [("USES_INGREDIENT", recipe.ingredients), ("USES_SEASONING", recipe.seasonings)]:
            for name in names:
                iid = node_id("ingredient", name)
                add_node(iid, name, "Ingredient", {"name": name})
                add_edge(rid, iid, relation)
        for name in recipe.tags:
            tid = node_id("tag", name)
            add_node(tid, name, "Tag", {"name": name})
            add_edge(rid, tid, "HAS_TAG")
        for index, text in enumerate(recipe.steps, 1):
            sid = f"step:{recipe.id}:{index}"
            add_node(sid, f"步骤 {index}", "Step", {"order": index, "text": text})
            add_edge(rid, sid, "HAS_STEP")
        sid = node_id("source", recipe.source.id)
        add_node(sid, recipe.source.title, "Source", recipe.source.model_dump())
        add_edge(rid, sid, "HAS_SOURCE")
    return list(nodes.values()), edges


def eligible(recipe: Recipe, constraints: Constraints) -> bool:
    all_ingredients = set(recipe.ingredients + recipe.seasonings)
    if all_ingredients.intersection(constraints.excluded_ingredients):
        return False
    if constraints.max_minutes is not None and (recipe.minutes is None or recipe.minutes > constraints.max_minutes):
        return False
    if not set(constraints.required_tags).issubset(recipe.tags):
        return False
    if constraints.recipe_name and constraints.recipe_name not in recipe.name:
        return False
    if not constraints.allow_missing and (constraints.available_ingredients or not constraints.recipe_name):
        if not set(recipe.ingredients).issubset(constraints.available_ingredients):
            return False
    return True


def rank_key(recipe: Recipe, constraints: Constraints) -> tuple:
    missing = len(set(recipe.ingredients) - set(constraints.available_ingredients))
    matched = len(set(recipe.ingredients).intersection(constraints.available_ingredients))
    return (missing, -matched, recipe.minutes if recipe.minutes is not None else 10_000, recipe.id)


class GraphStore(Protocol):
    def document(self) -> GraphDocument: ...
    def retrieve(self, constraints: Constraints) -> list[Recipe]: ...
    def graph(self, recipe_id: str | None = None) -> dict: ...
    def stats(self) -> dict: ...


class SQLiteGraphStore:
    """A local property graph persisted as nodes/edges, traversed for retrieval."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def import_document(self, document: GraphDocument) -> dict:
        # Validate and materialize before opening transaction: no partial import.
        document = GraphDocument.model_validate(document.model_dump())
        nodes, edges = build_graph(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, label TEXT, type TEXT, properties TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS edges (id TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL, label TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS edge_source ON edges(source, label);
                CREATE INDEX IF NOT EXISTS edge_target ON edges(target, label);
            """)
            connection.execute("DELETE FROM nodes")
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM metadata")
            connection.executemany("INSERT INTO nodes VALUES (:id,:label,:type,:properties)", [
                {**node, "properties": json.dumps(node["properties"], ensure_ascii=False)} for node in nodes
            ])
            connection.executemany("INSERT INTO edges VALUES (:id,:source,:target,:label)", edges)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("document", document.model_dump_json()),
                ("content_sha256", hashlib.sha256(document.model_dump_json().encode()).hexdigest()),
            ])
        return self.stats()

    def document(self) -> GraphDocument:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='document'").fetchone()
        if not row:
            raise ValueError("图谱未导入")
        return GraphDocument.model_validate_json(row["value"])

    def retrieve(self, constraints: Constraints) -> list[Recipe]:
        # Traverse persisted Recipe -> Ingredient/Tag edges. Other fields come from
        # the validated source document; no embedding or LLM-generated query.
        document = self.document()
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT r.id AS recipe_node, e.label AS relation, i.label AS name
                FROM nodes r JOIN edges e ON e.source=r.id
                JOIN nodes i ON i.id=e.target
                WHERE r.type='Recipe' AND e.label IN ('USES_INGREDIENT','USES_SEASONING','HAS_TAG')
            """).fetchall()
        relations: dict[str, dict[str, list[str]]] = {}
        for row in rows:
            relations.setdefault(row["recipe_node"], {}).setdefault(row["relation"], []).append(row["name"])
        candidates = []
        for recipe in document.recipes:
            related = relations.get(f"recipe:{recipe.id}", {})
            actual = recipe.model_copy(update={
                "ingredients": related.get("USES_INGREDIENT", []),
                "seasonings": related.get("USES_SEASONING", []),
                "tags": related.get("HAS_TAG", []),
            })
            if actual.ingredients and eligible(actual, constraints):
                candidates.append(actual)
        return sorted(candidates, key=lambda recipe: rank_key(recipe, constraints))

    def graph(self, recipe_id: str | None = None) -> dict:
        with self._connect() as connection:
            if recipe_id is None:
                # Bound full graph preview; canonical data remains complete.
                ids = [r["id"] for r in connection.execute("SELECT id FROM nodes WHERE type='Recipe' ORDER BY id LIMIT 3")]
            else:
                ids = [f"recipe:{recipe_id}"]
            if not ids:
                return {"nodes": [], "edges": []}
            placeholders = ",".join("?" for _ in ids)
            edges = [dict(row) for row in connection.execute(f"SELECT * FROM edges WHERE source IN ({placeholders})", ids)]
            node_ids = list(set(ids + [edge["target"] for edge in edges]))
            placeholders = ",".join("?" for _ in node_ids)
            nodes = [{**dict(row), "properties": json.loads(row["properties"])} for row in connection.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", node_ids)]
        return {"nodes": nodes, "edges": edges, "preview": recipe_id is None}

    def stats(self) -> dict:
        with self._connect() as connection:
            counts = {row["type"]: row["n"] for row in connection.execute("SELECT type,count(*) n FROM nodes GROUP BY type")}
            edges = connection.execute("SELECT count(*) n FROM edges").fetchone()["n"]
        return {"recipes": counts.get("Recipe", 0), "ingredients": counts.get("Ingredient", 0), "nodes": sum(counts.values()), "edges": edges}


class Neo4jGraphStore:
    """Adapter for the documented RG* schema. Existing schemas require mapping/export."""

    def __init__(self, uri: str, user: str, password: str, database: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def _read(self, query: str, parameters: dict | None = None) -> list[dict]:
        from neo4j import READ_ACCESS
        with self.driver.session(database=self.database, default_access_mode=READ_ACCESS) as session:
            return session.execute_read(lambda tx: [record.data() for record in tx.run(query, parameters or {})])

    def document(self) -> GraphDocument:
        metadata = self._read("MATCH (m:RGDataset {key:'current'}) RETURN m.document AS document")
        if not metadata:
            raise ValueError("该 Neo4j 库未导入 RG* schema；请先导出原图谱并映射，或在单独数据库导入示例")
        base = GraphDocument.model_validate_json(metadata[0]["document"])
        # Read native graph relations, not only the metadata snapshot.
        rows = self._read("""
            MATCH (r:RGRecipe)
            CALL (r) { MATCH (r)-[:USES_INGREDIENT]->(i:RGIngredient) RETURN collect(i.name) AS ingredients }
            CALL (r) { OPTIONAL MATCH (r)-[:USES_SEASONING]->(s:RGIngredient) RETURN collect(s.name) AS seasonings }
            CALL (r) { OPTIONAL MATCH (r)-[:HAS_TAG]->(t:RGTag) RETURN collect(t.name) AS tags }
            CALL (r) { MATCH (r)-[:HAS_STEP]->(step:RGStep) WITH step ORDER BY step.order RETURN collect(step.text) AS steps }
            MATCH (r)-[:HAS_SOURCE]->(source:RGSource)
            RETURN r.id AS id,r.name AS name,r.minutes AS minutes,ingredients,seasonings,tags,steps,
              {id:source.id,title:source.title,url:source.url} AS source ORDER BY id
        """)
        return GraphDocument(schema_version=1, dataset=base.dataset, ingredient_aliases=base.ingredient_aliases, recipes=rows)

    def retrieve(self, constraints: Constraints) -> list[Recipe]:
        return sorted([recipe for recipe in self.document().recipes if eligible(recipe, constraints)], key=lambda r: rank_key(r, constraints))

    def graph(self, recipe_id: str | None = None) -> dict:
        document = self.document()
        document = document.model_copy(update={"recipes": [r for r in document.recipes if r.id == recipe_id] if recipe_id else document.recipes[:3]})
        nodes, edges = build_graph(document)
        return {"nodes": nodes, "edges": edges, "preview": recipe_id is None}

    def stats(self) -> dict:
        document = self.document()
        nodes, edges = build_graph(document)
        return {"recipes": len(document.recipes), "ingredients": sum(n["type"] == "Ingredient" for n in nodes), "nodes": len(nodes), "edges": len(edges)}

    def inspect(self) -> dict:
        labels = self._read("CALL db.labels() YIELD label RETURN label")
        relations = self._read("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        properties = self._read("MATCH (n) WITH labels(n) AS labels, keys(n) AS keys LIMIT 30 RETURN labels,keys")
        return {"labels": labels, "relationship_types": relations, "sample_property_keys": properties, "read_only": True}

    def import_document(self, document: GraphDocument) -> dict:
        document = GraphDocument.model_validate(document.model_dump())

        def write(tx):
            # Delete only dedicated RecipeGen nodes, never arbitrary user nodes.
            tx.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) DETACH DELETE n", labels=["RGRecipe", "RGIngredient", "RGTag", "RGStep", "RGSource", "RGDataset"]).consume()
            tx.run("CREATE (:RGDataset {key:'current',document:$document})", document=document.model_dump_json()).consume()
            for recipe in document.recipes:
                tx.run("CREATE (:RGRecipe {id:$id,name:$name,minutes:$minutes})", id=recipe.id, name=recipe.name, minutes=recipe.minutes).consume()
                for relation, names in [("USES_INGREDIENT", recipe.ingredients), ("USES_SEASONING", recipe.seasonings)]:
                    tx.run(f"MATCH (r:RGRecipe {{id:$id}}) UNWIND $names AS name MERGE (i:RGIngredient {{name:name}}) MERGE (r)-[:{relation}]->(i)", id=recipe.id, names=names).consume()
                tx.run("MATCH (r:RGRecipe {id:$id}) UNWIND $tags AS name MERGE (t:RGTag {name:name}) MERGE (r)-[:HAS_TAG]->(t)", id=recipe.id, tags=recipe.tags).consume()
                for index, text in enumerate(recipe.steps, 1):
                    tx.run("MATCH (r:RGRecipe {id:$id}) CREATE (s:RGStep {id:$sid,order:$order,text:$text}) CREATE (r)-[:HAS_STEP]->(s)", id=recipe.id, sid=f"{recipe.id}:{index}", order=index, text=text).consume()
                tx.run("MATCH (r:RGRecipe {id:$id}) MERGE (s:RGSource {id:$source_id}) SET s.title=$title,s.url=$url MERGE (r)-[:HAS_SOURCE]->(s)", id=recipe.id, source_id=recipe.source.id, title=recipe.source.title, url=recipe.source.url).consume()

        with self.driver.session(database=self.database) as session:
            session.execute_write(write)
        return self.stats()
