"""Synthetic offline contracts; no real models, database, or accuracy claims."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

from recipegen import multimodal_query as query

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recipegen_mm_query_cli_contract", ROOT/"scripts/query_multimodal_graph.py")
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
assert SPEC.loader is not None
SPEC.loader.exec_module(CLI)
MODEL, REVISION = "openai/clip-vit-base-patch32", "a"*40
VECTOR = [1.0]+[0.0]*511


def build():
    return {"build_id":"synthetic-test-build", "dataset_revision":"b"*40,
            "split":"test", "active":True, "status":"verified", "content_sha256":"c"*64}


def candidate(identifier="mm:observation:synthetic1", *, video=False, orphan=False):
    value = {"observation_id":identifier, "caption":"A bowl on a table.",
             "source_media_id":"video:fixture" if video else "image:fixture", "source_id":"source:fixture",
             "media_kind":"video" if video else "image", "parent_kind":"VideoClip" if video else "Image",
             "start_sec":5.0 if video else None, "end_sec":10.0 if video else None,
             "timeline_origin_sec":0.25 if video else None,
             "time_reference":"relative_to_first_decoded_frame" if video else None,
             "step_id":None if orphan else "step:fixture", "step_text":None if orphan else "Add the tomato.",
             "step_order":None if orphan else 2, "step_source_id":None if orphan else "source:steps",
             "recipe_id":None if orphan else "recipe:fixture", "recipe_title":None if orphan else "Make soup",
             "raw_json":json.dumps({"caption":"A bowl on a table.", "objects":[], "visual_relations":[]}),
             "score":0.79, "run_id":"synthetic-model-run", "build_id":build()["build_id"],
             "dataset_revision":build()["dataset_revision"], "split":"test", "namespace":query.NAMESPACE,
             "embedding_model":MODEL, "embedding_revision":REVISION,
             "association_run_id":"synthetic-model-run",
             "alignment_run_id":None if orphan else "synthetic-model-run",
             "source_archive":"test-video.zip" if video else "test.zip", "source_member":"test/fixture/0.jpg"}
    value["evidence_json"] = json.dumps({key:value[key] for key in
        ("source_media_id","source_id","run_id","build_id","split","dataset_revision")})
    return value


class Record(dict):
    def data(self): return dict(self)


class ReadOnlySession:
    def __init__(self, rows=None, builds=None):
        self.rows = [candidate()] if rows is None else rows
        self.builds = [build()] if builds is None else builds
        self.queries = []
        self.reads = 0

    def execute_read(self, callback):
        self.reads += 1
        return callback(self)

    def execute_write(self, callback):
        raise AssertionError("no writes allowed")

    def run(self, cypher, parameters):
        assert not re.search(r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|LOAD)\b",cypher,re.I)
        self.queries.append((cypher,copy.deepcopy(parameters)))
        if cypher == query.ACTIVE_BUILD_QUERY: return [Record(row) for row in self.builds]
        if cypher == query.SEARCH_QUERY: return [Record(row) for row in self.rows]
        raise AssertionError("only fixed read queries")


def search(session, **kwargs):
    return query.search_session(session,VECTOR,embedding_model=MODEL,embedding_revision=REVISION,**kwargs)


def test_fixed_read_contract_and_original_facts():
    rows = [candidate(),candidate("mm:observation:synthetic2",video=True)]
    session = ReadOnlySession(rows)
    report = search(session,query="chopping tomato",limit=5,candidate_limit=100)
    assert session.reads == 1 and len(session.queries) == 2
    assert report["read_only"] is True and report["llm_called"] is False
    assert report["embedding_space"] == {"model":MODEL,"revision":REVISION,"dimension":512}
    assert report["results"][1]["start_sec"] == 5.0
    assert report["results"][1]["step_text"] == rows[1]["step_text"]
    assert report["results"][0]["evidence_json"] == rows[0]["evidence_json"]
    assert report["results"][0]["verified"] is False
    cypher, params = session.queries[1]
    assert "db.index.vector.queryNodes('recipegen_visual_embedding', $candidate_limit, $embedding)" in cypher
    assert "observation.kg_csv_id AS observation_id" in cypher
    assert "h.run_id = observation.run_id" in cypher and "alignment.run_id = observation.run_id" in cypher
    assert "observation.embedding_revision = $embedding_revision" in cypher
    assert params["embedding"] == VECTOR and params["archives"] == ["test.zip","test-video.zip"]


@pytest.mark.parametrize("vector",[[1.0]*511,[0.0]*512,[True]+[0.0]*511,[float("nan")]+[0.0]*511])
def test_invalid_vector_never_accesses_database(vector):
    session = ReadOnlySession()
    with pytest.raises(query.MultimodalQueryError):
        query.search_session(session,vector,embedding_model=MODEL,embedding_revision=REVISION)
    assert session.reads == 0


@pytest.mark.parametrize("builds",[[],[build(),build()],[{**build(),"split":"train"}]])
def test_active_verified_unique_test_build_required(builds):
    session = ReadOnlySession(builds=builds)
    with pytest.raises(query.MultimodalQueryError): search(session)
    assert len(session.queries) == 1


@pytest.mark.parametrize("field,value",[("split","train"),("embedding_revision","wrong"),
     ("association_run_id","other-run"),("alignment_run_id","other-run"),
     ("source_archive","train.zip"),("step_text",None),("score",float("inf"))])
def test_returned_source_scope_and_run_contract_rejected(field,value):
    row = candidate()
    row[field] = value
    with pytest.raises(query.MultimodalQueryError): search(ReadOnlySession([row]))


def test_empty_and_orphan_matches_do_not_invent_steps_or_recipes():
    report = search(ReadOnlySession([]))
    assert report["status"] == "no_match" and report["count"] == 0 and report["llm_called"] is False
    orphan = search(ReadOnlySession([candidate(orphan=True)]))["results"][0]
    assert orphan["recipe_title"] is None and orphan["step_text"] is None
    with pytest.raises(query.MultimodalQueryError,match="无候选"):
        query.apply_evidence_selection(report,{"selected_observation_ids":[],"reason":""})


def test_model_selection_preserves_facts_and_can_decline_all():
    original = search(ReadOnlySession([candidate(),candidate("mm:observation:synthetic2")]))
    selection = {"selected_observation_ids":["mm:observation:synthetic2"], "reason":"Closest image",
                 "caption":"invented replacement", "raw_text":"synthetic output", "model":{"name":"fixture"}}
    selected = query.apply_evidence_selection(original,selection)
    assert selected["results"] == [original["results"][1]]
    assert selected["candidates"] == original["results"] and selected["llm_called"] is True
    assert selected["results"][0]["caption"] == "A bowl on a table."
    assert "caption" not in selected["selection"]
    assert original["llm_called"] is False
    declined = query.apply_evidence_selection(original,{"selected_observation_ids":[],"reason":"Insufficient evidence"})
    assert declined["status"] == "no_match" and declined["results"] == [] and declined["llm_called"] is True


@pytest.mark.parametrize("ids",[["invented"],["mm:observation:synthetic1"]*2,[None]])
def test_unknown_or_duplicate_model_selected_ids_fail(ids):
    report = search(ReadOnlySession())
    with pytest.raises(query.MultimodalQueryError):
        query.apply_evidence_selection(report,{"selected_observation_ids":ids,"reason":"fixture"})


class FakeModels:
    embedding_space = {"model":MODEL,"revision":REVISION,"dimension":512}
    model_info = {"name":"synthetic-local-selector","revision":"d"*40}
    def __init__(self): self.calls = {"embedding_text":0,"vision":0}
    def text_embeddings(self,texts):
        assert texts == ["tomato bowl"]
        self.calls["embedding_text"] += 1
        return [VECTOR]
    def select_evidence(self,text,candidates):
        self.calls["vision"] += 1
        return {"selected_observation_ids":[candidates[0]["observation_id"]],"reason":"synthetic","model":self.model_info}


@pytest.mark.parametrize("has_match,expected_calls",[(False,0),(True,1)])
def test_cli_orchestration_uses_actual_vector_and_skips_llm_when_empty(has_match,expected_calls):
    models = FakeModels()
    def fake_search(connection,embedding,**kwargs):
        assert connection == "not-real-credentials" and embedding == VECTOR
        assert kwargs["embedding_revision"] == REVISION
        return search(ReadOnlySession([candidate()] if has_match else []),query=kwargs["query"])
    report = CLI.run_query(" tomato bowl ",connection="not-real-credentials",models=models,llm=True,search=fake_search)
    assert models.calls == {"embedding_text":1,"vision":expected_calls}
    assert report["llm_called"] is bool(expected_calls)
    assert report["local_model"]["selector"] == (models.model_info if has_match else None)


def test_query_text_never_enters_cypher():
    payload = "tomato') DETACH DELETE n //"
    session = ReadOnlySession([])
    assert search(session,query=payload)["query"] == payload
    assert all(payload not in cypher for cypher,_ in session.queries)
