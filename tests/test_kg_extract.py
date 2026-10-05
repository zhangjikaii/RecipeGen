from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recipegen.kg_extract import DEFAULT_LEXICON, RuleExtractor, extract_step, extract_steps, load_lexicon


def names(mentions, kind):
    return [item["normalized_name"] for item in mentions if item["kind"] == kind]


class ExtractionEvidenceTests(unittest.TestCase):
    def test_mentions_have_exact_spans_and_original_source(self):
        text = "Chop cherry tomatoes and garlic; stir in extra-virgin olive oil in a frying pan. Bake at 180°C for 20-25 minutes."
        source = {"archive": "test.zip", "member": "test/example.json", "recipe_id": "example"}
        result = extract_step(text, 2, source=source)
        self.assertEqual(names(result, "Ingredient"), ["cherry tomato", "garlic", "olive oil"])
        self.assertEqual(names(result, "Action"), ["chop", "stir", "bake"])
        self.assertEqual(names(result, "Tool"), ["frying pan"])
        self.assertEqual([row["name"] for row in result if row["kind"] == "Temperature"], ["180°C"])
        self.assertEqual([row["name"] for row in result if row["kind"] == "Duration"], ["20-25 minutes"])
        for row in result:
            evidence = row["evidence"]
            self.assertEqual(evidence["text"], text)
            self.assertEqual(text[evidence["span_start"]:evidence["span_end"]], row["name"])
            self.assertEqual(row["step_order"], 2)
            self.assertEqual(row["source"], source)
            self.assertEqual(row["method"], "rule")
            self.assertIsNone(row["confidence"])
        result[0]["source"]["member"] = "changed"
        self.assertEqual(source["member"], "test/example.json")

    def test_known_plural_normalization_and_explicit_aliases(self):
        result = extract_step("Slice tomatoes, potatoes, strawberries, mangoes and scallions; add bay leaves and capers.", 1)
        self.assertEqual(names(result, "Ingredient"), ["tomato", "potato", "strawberry", "mango", "green onion", "bay leaf", "caper"])

    def test_multiword_food_prevents_generic_duplicates(self):
        result = extract_step("Add coconut milk, peanut butter, egg whites and tomato paste.", 1)
        self.assertEqual(names(result, "Ingredient"), ["coconut milk", "peanut butter", "egg white", "tomato paste"])
        self.assertNotIn("milk", names(result, "Ingredient"))
        self.assertNotIn("egg", names(result, "Ingredient"))

    def test_food_words_in_tool_names_are_not_ingredients(self):
        result = extract_step("Use the rice cooker, garlic press, pepper mill, egg beater and stock pot.", 1)
        self.assertEqual(names(result, "Ingredient"), [])
        self.assertEqual(names(result, "Tool"), ["rice cooker", "garlic press", "pepper mill", "egg beater", "stockpot"])
        self.assertNotIn("press", names(result, "Action"))

    def test_word_boundaries_prevent_substring_entities(self):
        result = extract_step("The honeycomb, flourless mixture and countertop are beside a blender.", 1)
        self.assertEqual(names(result, "Ingredient"), [])
        self.assertEqual(names(result, "Action"), [])
        self.assertEqual(names(result, "Tool"), ["blender"])

    def test_action_inflections_and_multiverb_priority(self):
        result = extract_step("Stir-fry carrots, then deep-fry potatoes and whisk eggs.", 1)
        self.assertEqual(names(result, "Action"), ["stir fry", "deep fry", "whisk"])
        self.assertEqual(names(result, "Ingredient"), ["carrot", "potato", "egg"])
        self.assertEqual(names(result, "Tool"), [])

    def test_tool_and_ingredient_noun_phrases_do_not_become_actions(self):
        result = extract_step("Add baking powder to a mixing bowl and put it on a roasting pan.", 1)
        self.assertEqual(names(result, "Ingredient"), ["baking powder"])
        self.assertEqual(names(result, "Tool"), ["mixing bowl", "roasting pan"])
        self.assertEqual(names(result, "Action"), ["add"])

    def test_ambiguous_whisk_verb_and_tool_have_local_evidence(self):
        result = extract_step("Whisk eggs with a whisk.", 1)
        self.assertEqual(names(result, "Action"), ["whisk"])
        self.assertEqual(names(result, "Tool"), ["whisk"])
        self.assertEqual(names(result, "Ingredient"), ["egg"])
        self.assertEqual([row["name"] for row in result if row["kind"] == "Action"], ["Whisk"])

    def test_multiple_durations_are_preserved_without_total(self):
        result = extract_steps(["Rest for 5 minutes.", "Bake for 10 to 15 minutes, then cool for half an hour."])
        durations = [row for row in result if row["kind"] == "Duration"]
        self.assertEqual([row["name"] for row in durations], ["5 minutes", "10 to 15 minutes", "half an hour"])
        self.assertEqual([row["step_order"] for row in durations], [1, 2, 2])
        self.assertFalse(any("total" in key.lower() for row in result for key in row))

    def test_temperature_keeps_case_symbols_ranges_and_units(self):
        text = "Heat to 350°F, lower to 180 degrees Celsius, then freeze at -10°C."
        result = extract_step(text, 1)
        values = [row["name"] for row in result if row["kind"] == "Temperature"]
        self.assertEqual(values, ["350°F", "180 degrees Celsius", "-10°C"])
        self.assertEqual(names(result, "Duration"), [])

    def test_unknown_degrees_and_plain_numbers_are_not_temperatures(self):
        result = extract_step("Turn the tray 180 degrees and add 350 grams of flour.", 1)
        self.assertEqual(names(result, "Temperature"), [])
        self.assertEqual(names(result, "Duration"), [])

    def test_original_unicode_indices_and_duplicate_mentions(self):
        text = "🥕 Add carrots, then add carrots."
        result = extract_step(text, 7)
        mentions = [row for row in result if row["kind"] == "Ingredient"]
        self.assertEqual(len(mentions), 2)
        self.assertNotEqual(mentions[0]["evidence"]["span_start"], mentions[1]["evidence"]["span_start"])
        for row in mentions:
            self.assertEqual(text[row["evidence"]["span_start"]:row["evidence"]["span_end"]], "carrots")

    def test_empty_or_unknown_text_does_not_invent_entities(self):
        self.assertEqual(extract_step("", 1), [])
        self.assertEqual(extract_step("Glorp the thingamajig until ready.", 1), [])

    def test_structured_steps_preserve_order_and_step_source(self):
        result = extract_steps([{"order": 4, "text": "Chop tomatoes.", "source": {"member": "step-4"}}], source={"member": "fallback"})
        self.assertTrue(result)
        self.assertTrue(all(row["step_order"] == 4 and row["source"] == {"member": "step-4"} for row in result))

    def test_invalid_step_types_and_orders_fail_explicitly(self):
        for order in [0, -1, True, "1"]:
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    extract_step("Chop tomatoes.", order)
        with self.assertRaises(TypeError):
            extract_steps("Chop tomatoes.")
        with self.assertRaises(TypeError):
            extract_steps([{"caption": "not original text"}])

    def test_lexicon_has_at_least_requested_ingredient_coverage(self):
        self.assertGreaterEqual(len(DEFAULT_LEXICON.ingredients), 150)

    def test_custom_json_terms_are_literal_and_can_extend_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            path.write_text(json.dumps({"ingredients": [{"normalized_name": "soursop", "zh_name": "刺果番荔枝", "aliases": ["guanabana"]}]}), encoding="utf-8")
            lexicon = load_lexicon(path)
            result = extract_step("Mix guanabana and tomatoes.", 1, lexicon=lexicon)
            self.assertEqual(names(result, "Ingredient"), ["soursop", "tomato"])
            self.assertEqual(next(row["zh_name"] for row in result if row["normalized_name"] == "soursop"), "刺果番荔枝")
            path.write_text(json.dumps({"ingredients": [{"normalized_name": "literal", "aliases": ["milk.*"]}]}), encoding="utf-8")
            literal = load_lexicon(path, include_defaults=False)
            self.assertEqual(extract_step("milk powder", 1, lexicon=literal), [])
            self.assertEqual(names(extract_step("milk.*", 1, lexicon=literal), "Ingredient"), ["literal"])

    def test_custom_alias_conflict_is_not_silently_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            path.write_text(json.dumps({"ingredients": [{"normalized_name": "different food", "aliases": ["tomato"]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "别名冲突"):
                RuleExtractor(load_lexicon(path))


if __name__ == "__main__":
    unittest.main()
