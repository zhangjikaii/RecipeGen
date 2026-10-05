"""Adversarial contracts: do not relax user constraints or fabricate graph facts."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from recipegen.config import Settings
from recipegen.graph import Neo4jGraphStore, SQLiteGraphStore, eligible
from recipegen.models import Constraints, GraphDocument, RecommendRequest, Source
from recipegen.pipeline import Recommender, create_store
from recipegen.query import QueryParser


def fixture() -> GraphDocument:
    return GraphDocument.model_validate({
        "schema_version": 1,
        "dataset": {"id": "test", "name": "离线测试夹具", "is_demo": True},
        "ingredient_aliases": {"西红柿": "番茄"},
        "recipes": [
            {"id": "known", "name": "番茄炒蛋", "ingredients": ["番茄", "鸡蛋"], "seasonings": ["盐"],
             "minutes": 15, "tags": ["不辣"], "steps": ["加入水", "加热", "加入水"],
             "source": {"id": "internal:known", "title": "测试原记录", "url": None}},
            {"id": "unknown", "name": "番茄做法", "ingredients": ["番茄"], "seasonings": [],
             "minutes": None, "tags": [], "steps": ["处理番茄"],
             "source": {"id": "internal:unknown", "title": "未记录总用时的测试记录", "url": None}},
            {"id": "spicy", "name": "辣炒鸡蛋", "ingredients": ["鸡蛋"], "seasonings": ["辣椒"],
             "minutes": 20, "tags": ["辣"], "steps": ["翻炒"],
             "source": {"id": "internal:spicy", "title": "测试辣味记录", "url": None}},
        ],
    })


class ScriptedLLM:
    def __init__(self, extracted: dict, generated: dict):
        self.extracted = extracted
        self.generated = generated
        self.calls = []

    def json(self, system, payload, stage):
        self.calls.append(stage)
        return self.extracted if stage == "extract" else self.generated


class InvariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "graph.sqlite3"
        self.document = fixture()
        self.store = SQLiteGraphStore(self.path)
        self.store.import_document(self.document)
        self.settings = Settings(db_path=self.path)
        self.parser = QueryParser(self.document)

    def test_unknown_minutes_excluded_only_when_time_constrained(self):
        unknown = self.document.recipes[1]
        self.assertTrue(eligible(unknown, Constraints(available_ingredients=["番茄"])))
        self.assertFalse(eligible(unknown, Constraints(available_ingredients=["番茄"], max_minutes=20)))
        self.assertEqual(self.store.retrieve(Constraints(available_ingredients=["番茄"], max_minutes=20)), [])

    def test_unknown_tags_do_not_mean_not_spicy(self):
        self.assertFalse(eligible(self.document.recipes[1], Constraints(available_ingredients=["番茄"], required_tags=["不辣"])))

    def test_exclusion_applies_to_seasonings_even_when_missing_allowed(self):
        constraints = Constraints(available_ingredients=[], excluded_ingredients=["辣椒"], allow_missing=True)
        self.assertNotIn("spicy", [recipe.id for recipe in self.store.retrieve(constraints)])

    def test_repeated_steps_survive_storage_and_response(self):
        result = Recommender(self.settings, store=self.store).recommend(RecommendRequest(question="番茄炒蛋怎么做"))
        self.assertEqual(result["recommendations"][0]["steps"], ["加入水", "加热", "加入水"])
        steps = [row["object"] for row in result["evidence"] if row["predicate"] == "HAS_STEP"]
        self.assertEqual([row["order"] for row in steps], [1, 2, 3])

    def test_nl_excluded_ingredient_is_not_added_to_inventory(self):
        constraints = self.parser.parse("我有番茄和鸡蛋，不吃鸡蛋")
        self.assertEqual(constraints.available_ingredients, ["番茄", "鸡蛋"])
        self.assertEqual(constraints.excluded_ingredients, ["鸡蛋"])
        constraints = self.parser.parse("我有番茄，不吃鸡蛋和辣椒")
        self.assertEqual(constraints.available_ingredients, ["番茄"])
        self.assertEqual(set(constraints.excluded_ingredients), {"鸡蛋", "辣椒"})

    def test_negative_inventory_is_not_owned_inventory(self):
        constraints = self.parser.parse("我只有番茄，没有鸡蛋")
        self.assertEqual(constraints.available_ingredients, ["番茄"])
        self.assertNotIn("known", [recipe.id for recipe in self.store.retrieve(constraints)])

    def test_unpurchased_ingredient_is_not_owned_inventory(self):
        constraints = self.parser.parse("我有番茄，鸡蛋还没买")
        self.assertEqual(constraints.available_ingredients, ["番茄"])
        self.assertNotIn("known", [recipe.id for recipe in self.store.retrieve(constraints)])

    def test_desired_ingredient_is_not_added_to_explicit_inventory(self):
        constraints = self.parser.parse("我有番茄，想用鸡蛋做菜")
        self.assertEqual(constraints.available_ingredients, ["番茄"])
        self.assertNotIn("known", [recipe.id for recipe in self.store.retrieve(constraints)])

    def test_llm_cannot_remove_hard_constraints(self):
        baseline = self.parser.parse("我有番茄和鸡蛋，20分钟内，不吃辣，不放盐")
        model = Constraints(available_ingredients=["番茄", "鸡蛋"], max_minutes=100, allow_missing=True)
        merged = self.parser.merge_llm(baseline, model, "我有番茄和鸡蛋，20分钟内，不吃辣，不放盐")
        self.assertEqual(merged.max_minutes, 20)
        self.assertIn("盐", merged.excluded_ingredients)
        self.assertIn("不辣", merged.required_tags)
        self.assertFalse(merged.allow_missing)

    def test_llm_cannot_use_mentioned_recipe_name_to_bypass_inventory(self):
        question = "我只有番茄，番茄炒蛋也行吗"
        baseline = self.parser.parse(question)
        self.assertIsNone(baseline.recipe_name)
        model = Constraints(available_ingredients=["番茄"], recipe_name="番茄炒蛋")
        merged = self.parser.merge_llm(baseline, model, question)
        self.assertEqual(self.store.retrieve(merged), [], "菜名出现不能使 LLM 绕过缺鸡蛋的库存限制")

    def test_fabricated_or_malformed_model_recommendations_fallback(self):
        question = "我有番茄和鸡蛋，20分钟内，不吃辣"
        extracted = self.parser.parse(question).model_dump()
        invalid_outputs = {
            "foreign_id": {"recommendations": [{"recipe_id": "invented", "reason_codes": ["ingredient_match"]}]},
            "false_reason": {"recommendations": [{"recipe_id": "known", "reason_codes": ["needs_shopping"]}]},
            "unknown_reason": {"recommendations": [{"recipe_id": "known", "reason_codes": ["nutrition_proven"]}]},
            "invented_steps": {"recommendations": [{"recipe_id": "known", "reason_codes": ["ingredient_match"], "steps": ["虚构步骤"]}]},
            "invented_source": {"recommendations": [{"recipe_id": "known", "reason_codes": ["ingredient_match"], "source": {"id": "fake", "url": "https://fake.invalid"}}]},
            "duplicate_id": {"recommendations": [{"recipe_id": "known", "reason_codes": ["ingredient_match"]}] * 2},
            "wrong_shape": {"answer": "未经检索的自由文本"},
        }
        for name, generated in invalid_outputs.items():
            with self.subTest(name=name):
                llm = ScriptedLLM(extracted, generated)
                result = Recommender(self.settings, store=self.store, llm=llm).recommend(RecommendRequest(question=question, use_llm=True))
                self.assertEqual(result["generation"]["mode"], "fallback")
                self.assertIn("error", result["generation"])
                self.assertTrue(any(row["stage"] == "generate" and row["status"] == "fallback" for row in result["trace"]))
                self.assertEqual([item["recipe_id"] for item in result["recommendations"]], ["known"])
                self.assertTrue(result["validation"]["passed"])
                self.assertEqual(result["recommendations"][0]["source"], self.document.recipes[0].source.model_dump())

    def test_bad_extract_falls_back_without_attempting_generation(self):
        llm = ScriptedLLM({"unknown_schema": True}, {"recommendations": []})
        result = Recommender(self.settings, store=self.store, llm=llm).recommend(RecommendRequest(question="我有番茄和鸡蛋", use_llm=True))
        self.assertEqual(llm.calls, ["extract"])
        self.assertEqual(result["generation"]["mode"], "fallback")
        self.assertTrue(result["validation"]["passed"])

    def test_neo4j_failure_does_not_seed_or_switch_to_sqlite(self):
        destination = Path(self.temp.name) / "must_not_exist.sqlite3"
        settings = Settings(graph_backend="neo4j", db_path=destination, neo4j_password="")
        with patch("recipegen.pipeline.Neo4jGraphStore", side_effect=ValueError("Neo4j 未配置")):
            with self.assertRaisesRegex(ValueError, "Neo4j"):
                create_store(settings)
        self.assertFalse(destination.exists())

    def test_neo4j_original_schema_is_not_claimed_as_compatible(self):
        store = object.__new__(Neo4jGraphStore)
        with patch.object(store, "_read", return_value=[]):
            with self.assertRaisesRegex(ValueError, "RG"):
                store.document()

    def test_neo4j_uses_native_recipe_facts_not_only_metadata(self):
        store = object.__new__(Neo4jGraphStore)
        native = [recipe.model_dump() for recipe in self.document.recipes]
        native[0]["minutes"] = None
        native[0]["steps"] = ["原生图谱更新过的步骤"]
        native[0]["source"] = {"id": "native:updated", "title": "实际原生记录", "url": None}
        with patch.object(store, "_read", side_effect=[[{"document": self.document.model_dump_json()}], native]):
            document = store.document()
        self.assertIsNone(document.recipes[0].minutes)
        self.assertEqual(document.recipes[0].steps, ["原生图谱更新过的步骤"])
        self.assertEqual(document.recipes[0].source.id, "native:updated")
        self.assertFalse(eligible(document.recipes[0], Constraints(available_ingredients=["番茄", "鸡蛋"], max_minutes=20)))

    def test_conflicting_sources_and_non_http_urls_rejected(self):
        with self.assertRaises(ValidationError):
            Source(id="x", title="x", url="javascript:alert(1)")
        payload = self.document.model_dump()
        payload["recipes"][1]["source"] = {"id": "internal:known", "title": "不同标题", "url": None}
        with self.assertRaises(ValidationError):
            GraphDocument.model_validate(payload)

    def test_boolean_is_not_a_recorded_cooking_time(self):
        payload = self.document.model_dump()
        payload["recipes"][0]["minutes"] = True
        with self.assertRaises(ValidationError):
            GraphDocument.model_validate(payload)

    def test_malformed_source_urls_are_not_valid_sources(self):
        for url in ["https://", "http://not a url"]:
            with self.subTest(url=url):
                with self.assertRaises(ValidationError):
                    Source(id="x", title="x", url=url)


if __name__ == "__main__":
    unittest.main()
