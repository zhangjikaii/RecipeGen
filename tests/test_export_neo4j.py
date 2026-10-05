"""Offline contracts for the explicit-schema, read-only export script."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re
import unittest

from recipegen.models import GraphDocument

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_neo4j_contract", PROJECT / "scripts" / "export_neo4j.py")
EXPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPORT)


class Record(dict):
    def data(self):
        return dict(self)


class ReadOnlySession:
    """Record queries and return a small original Recipe/Ingredient/Step schema."""

    def __init__(self):
        self.queries = []
        self.recipe = {"recipe_id": "recipe-1", "title": "番茄炒蛋"}
        self.ingredients = [{"normalized_name": "番茄"}, {"name": "鸡蛋"}]
        self.steps = [{"order": 3, "text": "加入水"}, {"order": 1, "text": "加入水"}, {"order": 2, "text": "加热"}]

    def execute_read(self, callback):
        return callback(self)

    def run(self, query, **params):
        self.queries.append(query)
        if re.search(r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE)\b", query, re.IGNORECASE):
            raise AssertionError("Export must never issue a write query")
        if query.startswith("MATCH (r:"):
            return [Record(record_id="4:original:1", properties=self.recipe)]
        if "HAS_INGREDIENT" in query:
            return [Record(properties=item, relationship_properties={}, node_id=str(i)) for i, item in enumerate(self.ingredients)]
        if "HAS_STEP" in query:
            return [Record(properties=item, relationship_properties={}, node_id=str(i)) for i, item in enumerate(self.steps)]
        raise AssertionError(f"Unexpected query: {query}")


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.mapping = json.loads((PROJECT / "docs" / "neo4j_mapping.example.json").read_text(encoding="utf-8"))
        self.session = ReadOnlySession()

    def test_original_schema_unknown_time_and_stable_provenance(self):
        payload = EXPORT.export_graph(self.session, self.mapping, None)
        document = GraphDocument.model_validate(payload)
        recipe = document.recipes[0]
        self.assertIsNone(recipe.minutes)
        self.assertEqual(recipe.ingredients, ["番茄", "鸡蛋"])
        self.assertEqual(recipe.steps, ["加入水", "加热", "加入水"])
        self.assertEqual(recipe.source.id, "neo4j:recipe-1")
        self.assertIsNone(recipe.source.url)
        self.assertFalse(document.dataset.is_demo)
        self.assertGreaterEqual(len(self.session.queries), 3)

    def test_absent_ingredient_name_does_not_become_placeholder(self):
        self.session.ingredients[0] = {}
        with self.assertRaisesRegex(EXPORT.ExportError, "候选属性"):
            EXPORT.export_graph(self.session, self.mapping, None)

    def test_absent_steps_fail_instead_of_inventing_method(self):
        self.session.steps = []
        with self.assertRaisesRegex(EXPORT.ExportError, "steps"):
            EXPORT.export_graph(self.session, self.mapping, None)

    def test_steps_need_unique_explicit_order(self):
        self.session.steps[1]["order"] = 3
        with self.assertRaisesRegex(EXPORT.ExportError, "顺序重复"):
            EXPORT.export_graph(self.session, self.mapping, None)

    def test_missing_order_is_not_guessed(self):
        del self.session.steps[0]["order"]
        with self.assertRaisesRegex(EXPORT.ExportError, "步骤缺少"):
            EXPORT.export_graph(self.session, self.mapping, None)

    def test_missing_source_record_id_is_rejected(self):
        altered = copy.deepcopy(self.mapping)
        altered["source"]["recipe_id_property"] = "not_in_graph"
        with self.assertRaisesRegex(EXPORT.ExportError, "内部来源标识"):
            EXPORT.export_graph(self.session, altered, None)

    def test_unknown_time_requires_an_explicit_mapping(self):
        altered = copy.deepcopy(self.mapping)
        altered["fields"]["minutes"] = {"property": "minutes"}
        with self.assertRaisesRegex(EXPORT.ExportError, "minutes"):
            EXPORT.export_graph(self.session, altered, None)

    def test_declared_property_fallback_preserves_actual_minutes(self):
        altered = copy.deepcopy(self.mapping)
        altered["fields"]["minutes"] = {"property_fallbacks": ["minutes", "total_minutes"]}
        self.session.recipe.update(minutes=None, total_minutes=18)
        payload = EXPORT.export_graph(self.session, altered, None)
        self.assertEqual(payload["recipes"][0]["minutes"], 18)

    def test_identifier_injection_rejected_before_query(self):
        self.mapping["recipe_label"] = "Recipe`) DETACH DELETE r //"
        with self.assertRaises(EXPORT.ExportError):
            EXPORT.export_graph(self.session, self.mapping, None)
        self.assertEqual(self.session.queries, [])

    def test_chinese_identifiers_are_supported(self):
        self.assertEqual(EXPORT.identifier("食谱节点"), "`食谱节点`")


if __name__ == "__main__":
    unittest.main()
