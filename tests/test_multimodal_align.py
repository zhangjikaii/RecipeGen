"""Synthetic unit fixtures validate algorithms, never real-model acceptance."""
from __future__ import annotations

import copy
import itertools
import json
import math
import random

import pytest

from recipegen.multimodal_align import AlignmentError, align_multimodal, cosine_similarity


SPACE = {"model":"synthetic-unit-fixture", "revision":"not-real-model-evidence", "dimension":2}


def step(identifier, order, vector):
    return {"id":identifier, "order":order, "text":"Original step " + identifier,
            "embedding":vector}


def media(identifier, vector, *, kind="image", timestamp=None, timeline=None, **extras):
    return {"id":identifier, "kind":kind, "embedding":vector,
            "caption":"", "actions":[], "objects":[],
            "model":{"name":"synthetic-unit-fixture", "real_inference":False},
            "evidence":{"source_id":"synthetic:" + identifier},
            **({"start_seconds":timestamp} if timestamp is not None else {}),
            **({"timeline_id":timeline} if timeline is not None else {}), **extras}


def payload(items=None, steps=None, **extras):
    return {"schema_version":1, "embedding_space":dict(SPACE),
            "steps":steps if steps is not None else [step("s1",1,[1,0]),step("s2",2,[0,1])],
            "media":items if items is not None else [media("m1",[1,0])], "entities":[], **extras}


def by_id(report):
    return {item["media_id"]:item for item in report["alignments"]}


@pytest.mark.parametrize("left,right,expected",[
    ([2,0],[3,0],1), ([2,0],[0,9],0), ([1,0],[-2,0],-1),
    ([3,4],[4,3],0.96), ([1e308,1e308],[1e308,1e308],1),
    ([1e-308,1e-308],[1e-308,1e-308],1),
])
def test_cosine_handles_scale_and_known_geometries(left,right,expected):
    assert cosine_similarity(left,right) == pytest.approx(expected)


@pytest.mark.parametrize("left,right",[
    ([0,0],[1,0]), ([1],[1,0]), ([True,0],[1,0]),
    ([math.inf,0],[1,0]), ([math.nan,0],[1,0]), ([],[1,0]),
])
def test_invalid_vectors_never_produce_fake_similarity(left,right):
    with pytest.raises(AlignmentError): cosine_similarity(left,right)


def test_real_shape_512_contract_without_running_a_model():
    value = payload([media("m1",[1]+[0]*511)], [step("s1",1,[1]+[0]*511)])
    value["embedding_space"].pop("dimension")
    report = align_multimodal(value)
    assert report["embedding_space"]["dimension"] == 512
    assert report["alignments"][0]["score"] == pytest.approx(1)
    assert report["model_inference_performed"] is False


def test_explicit_step_order_and_real_timestamps_override_file_order():
    items = [media("late-first-filename",[0,1],kind="clip",timeline="video",timestamp=10,end_seconds=12),
             media("early-last-filename",[1,0],kind="frame",timeline="video",timestamp=1)]
    steps = [step("s2",2,[0,1]),step("s1",1,[1,0])]
    value = payload(items,steps)
    original = copy.deepcopy(value)
    rows = by_id(align_multimodal(value))
    assert rows["early-last-filename"]["step_id"] == "s1"
    assert rows["late-first-filename"]["step_id"] == "s2"
    assert all(row["alignment_mode"] == "temporal_monotonic" for row in rows.values())
    assert value == original
    value["media"].reverse()
    assert by_id(align_multimodal(value)) == rows


def test_monotonic_optimizer_chooses_global_fit_instead_of_greedy_best():
    # Early media prefers s2, but choosing s1 permits the strong later s1 match.
    rows = by_id(align_multimodal(payload([
        media("early",[0.6,0.8],kind="clip",timeline="v",timestamp=0),
        media("late",[1,0],kind="clip",timeline="v",timestamp=5),
    ],config={"similarity_threshold":0.5})))
    assert rows["early"]["alternatives"][0]["step_id"] == "s2"
    assert rows["early"]["step_id"] == rows["late"]["step_id"] == "s1"


def test_conflicting_order_allows_unmatched_media_without_forcing_an_edge():
    report = align_multimodal(payload([
        media("early",[0,1],kind="frame",timeline="v",timestamp=0),
        media("late",[1,0],kind="frame",timeline="v",timestamp=2),
    ],config={"similarity_threshold":0.8}))
    assert report["summary"]["matched_candidates"] == 1
    unmatched = next(row for row in report["alignments"] if row["status"] == "unmatched")
    assert unmatched["step_id"] is None and unmatched["score"] is None
    assert unmatched["reason"] == "temporal_conflict"
    assert unmatched["alternatives"][0]["passes_threshold"] is True


def test_equal_timestamps_are_unordered_and_do_not_use_identifiers_as_time():
    rows = by_id(align_multimodal(payload([
        media("a",[0,1],kind="frame",timeline="v",timestamp=1),
        media("b",[1,0],kind="frame",timeline="v",timestamp=1),
        media("c",[0,1],kind="frame",timeline="v",timestamp=2),
    ],config={"similarity_threshold":0.8})))
    assert rows["a"]["step_id"] == "s2"
    assert rows["b"]["step_id"] == "s1"
    assert rows["c"]["step_id"] == "s2"


def test_separate_timelines_images_and_missing_timestamps_are_independent():
    rows = by_id(align_multimodal(payload([
        media("v-a",[0,1],kind="clip",timeline="a",timestamp=0),
        media("v-b",[1,0],kind="clip",timeline="b",timestamp=8),
        media("image",[1,0],kind="image",timeline="a",timestamp=20),
        media("no-time",[1,0],kind="frame",timeline="a"),
        media("no-timeline",[0,1],kind="clip",timestamp=1),
    ])))
    assert all(row["status"] == "candidate" for row in rows.values())
    assert rows["v-a"]["step_id"] == "s2" and rows["v-b"]["step_id"] == "s1"
    assert all(rows[key]["alignment_mode"] == "independent" for key in ("image","no-time","no-timeline"))


def test_threshold_no_steps_and_empty_media_are_explicit():
    report = align_multimodal(payload([media("opposite",[-1,0])], [step("s1",1,[1,0])]))
    assert report["status"] == "no_semantic_match"
    assert report["alignments"][0]["reason"] == "below_threshold"
    report = align_multimodal(payload(steps=[]))
    assert report["alignments"][0]["reason"] == "no_steps"
    assert report["alignments"][0]["alternatives"] == []
    report = align_multimodal(payload([]))
    assert report["summary"]["media"] == 0 and report["alignments"] == []


def test_threshold_inclusive_boundary_can_match_without_invented_confidence():
    report = align_multimodal(payload([media("m",[0,1],kind="clip",timeline="v",timestamp=0)],
                                      [step("s1",1,[1,0])], config={"similarity_threshold":0}))
    row = report["alignments"][0]
    assert row["status"] == "candidate" and row["score"] == 0
    assert "confidence" not in row


def test_dictionary_aliases_preserve_objects_and_avoid_latin_substring_matches():
    objects = [{"id":"box1","name":"TOMATO","bbox":[1,2,3,4],"raw_score":0.62}, "knife"]
    relations = [{"subject":"TOMATO", "predicate":" ON-top_OF ", "object":"Board", "raw_score":0.54}]
    item = media("m",[1,0], caption="番茄炒蛋；shampoo on bread.", actions=["SLICE"],
                 objects=objects, raw_output={"relations":relations,"original_response":"verbatim"})
    dictionary = [{"id":"i:tomato","type":"Ingredient","name":"番茄","aliases":["tomato"]},
                  {"id":"i:ham","type":"Ingredient","name":"ham","aliases":[]},
                  {"id":"t:knife","type":"Tool","name":"刀","aliases":["knife"]},
                  {"id":"t:board","type":"Tool","name":"砧板","aliases":["board"]},
                  {"id":"a:slice","type":"Action","name":"切","aliases":["slice"]}]
    value = payload([item],entities=dictionary)
    original = copy.deepcopy(value)
    report = align_multimodal(value)
    matched = {row["entity_id"] for row in report["semantic_entities"]}
    assert matched == {"i:tomato","t:knife","a:slice"}
    tomato = next(row for row in report["semantic_entities"] if row["entity_id"] == "i:tomato")
    assert any(match.get("object_id") == "box1" for match in tomato["matches"])
    assert tomato["score_kind"] == "dictionary_exact_alias" and tomato["status"] == "candidate"
    assert report["evidence"][0]["raw_objects"] == objects
    triple = report["visual_triples"][0]
    assert (triple["subject"],triple["predicate"],triple["object"]) == ("tomato","on top of","board")
    assert triple["subject_entity_candidates"] == ["i:tomato"]
    assert triple["object_entity_candidates"] == ["t:board"]
    assert triple["raw_relation"] == relations[0]
    assert triple["model"] == item["model"]
    assert value == original
    report["evidence"][0]["raw_objects"][0]["bbox"][0] = 999
    assert value == original
    json.dumps(report,allow_nan=False)


def test_captions_and_entity_hits_never_fabricate_visual_relations():
    report = align_multimodal(payload([media("m",[1,0],caption="Knife cuts tomato.")],
        entities=[{"id":"t:knife","type":"Tool","name":"knife"}]))
    assert report["semantic_entities"] and report["visual_triples"] == []


def test_ambiguous_aliases_remain_multiple_existing_entity_candidates():
    report = align_multimodal(payload([media("m",[1,0],objects=["pan"])],entities=[
        {"id":"tool:1","type":"Tool","name":"pan"},
        {"id":"tool:2","type":"Tool","name":"frying pan","aliases":["pan"]},
    ]))
    assert {row["entity_id"] for row in report["semantic_entities"]} == {"tool:1","tool:2"}


def test_malformed_raw_relation_is_preserved_as_unusable_not_repaired():
    raw = [{"subject":"tomato","predicate":"on","object":None},"unstructured relation"]
    report = align_multimodal(payload([media("m",[1,0],raw_output={"relations":raw})]))
    assert all(row["status"] == "unusable" for row in report["visual_triples"])
    assert [row["raw_relation"] for row in report["visual_triples"]] == raw
    assert report["summary"]["visual_triple_candidates"] == 0


@pytest.mark.parametrize("change",[
    "media_dimension", "step_dimension", "media_space", "step_space", "duplicate_media", "duplicate_step", "duplicate_order",
    "negative_time", "backward_interval", "end_without_start", "unknown_kind", "missing_model", "missing_source", "bad_objects",
    "nan_evidence", "threshold", "dictionary_threshold", "top_k_bool", "duplicate_entity", "entity_type", "bad_alias", "zero_step",
])
def test_invalid_artifact_contract_fails_before_emitting_candidates(change):
    value = payload()
    if change == "media_dimension": value["media"][0]["embedding"] = [1]
    if change == "step_dimension": value["steps"][0]["embedding"] = [1]
    if change in {"media_space","step_space"}:
        target = value["media" if change == "media_space" else "steps"][0]
        target["embedding_space"] = {**SPACE,"model":"different-model"}
    if change == "duplicate_media": value["media"].append(copy.deepcopy(value["media"][0]))
    if change == "duplicate_step": value["steps"][1]["id"] = "s1"
    if change == "duplicate_order": value["steps"][1]["order"] = 1
    if change == "negative_time": value["media"][0]["start_seconds"] = -1
    if change == "backward_interval": value["media"][0].update(start_seconds=5,end_seconds=1)
    if change == "end_without_start": value["media"][0]["end_seconds"] = 1
    if change == "unknown_kind": value["media"][0]["kind"] = "text"
    if change == "missing_model": value["media"][0]["model"] = {}
    if change == "missing_source": value["media"][0]["evidence"] = {}
    if change == "bad_objects": value["media"][0]["objects"] = [{"bbox":[1,2,3,4]}]
    if change == "nan_evidence": value["media"][0]["evidence"]["score"] = math.nan
    if change == "threshold": value["config"] = {"similarity_threshold":2}
    if change == "dictionary_threshold": value["config"] = {"entity_min_score":-1}
    if change == "top_k_bool": value["config"] = {"top_k":True}
    if change in {"duplicate_entity","entity_type","bad_alias"}:
        value["entities"] = [{"id":"i:1","type":"Ingredient","name":"tomato"}]
        if change == "duplicate_entity": value["entities"].append(copy.deepcopy(value["entities"][0]))
        if change == "entity_type": value["entities"][0]["type"] = "Recipe"
        if change == "bad_alias": value["entities"][0]["aliases"] = ["!!!"]
    if change == "zero_step": value["steps"][0]["embedding"] = [0,0]
    with pytest.raises(AlignmentError): align_multimodal(value)


def test_declared_same_space_is_accepted_without_claiming_real_model_execution():
    value = payload()
    value["media"][0]["embedding_space"] = dict(SPACE)
    value["steps"][0]["embedding_space"] = dict(SPACE)
    report = align_multimodal(value)
    assert report["status"] == "ok"
    assert report["model_inference_performed"] is False


def test_temporal_dynamic_program_matches_exhaustive_small_problem_optima():
    # A separate enumeration validates partial monotonic optimization, including
    # skip states. These vectors are deliberately synthetic algorithm fixtures.
    rng = random.Random(47)
    for _ in range(12):
        angles = [rng.random()*math.pi for _ in range(7)]
        vectors = [[math.cos(angle),math.sin(angle)] for angle in angles]
        steps = [step("s"+str(i),i+1,vector) for i,vector in enumerate(vectors[:3])]
        items = [media("m"+str(i),vector,kind="clip",timeline="v",timestamp=i*2) for i,vector in enumerate(vectors[3:])]
        threshold = 0.45
        value = payload(items,steps,config={"similarity_threshold":threshold})
        report = align_multimodal(value)
        actual = (math.fsum(row["score"]-threshold for row in report["alignments"] if row["score"] is not None),
                  report["summary"]["matched_candidates"])
        possibilities = []
        for assignment in itertools.product([-1,0,1,2],repeat=4):
            chosen = [index for index in assignment if index >= 0]
            if chosen != sorted(chosen): continue
            scores = [cosine_similarity(items[i]["embedding"],steps[index]["embedding"])
                      for i,index in enumerate(assignment) if index >= 0]
            if any(score < threshold for score in scores): continue
            possibilities.append((math.fsum(score-threshold for score in scores),len(scores)))
        expected = max(possibilities)
        assert actual[0] == pytest.approx(expected[0],abs=1e-12)
        assert actual[1] == expected[1]
