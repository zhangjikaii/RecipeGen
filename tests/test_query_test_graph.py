"""Read-only, bounded, source-preserving native graph search contracts."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recipegen_native_search_contract", PROJECT / "scripts/query_test_graph.py")
QUERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUERY
assert SPEC.loader is not None
SPEC.loader.exec_module(QUERY)


def build_record():
    return {"build_id":"test-contract", "split":"test", "status":"verified", "active":True,
            "dataset_revision":"a"*40, "content_sha256":"b"*64, "nodes":100, "relationships":130}


def native_record(identifier="recipe:1"):
    sources = [{"id":"source:"+role, "source_id":"source:"+role, "role":role,
                "source_type":"huggingface_zip_text", "dataset":"test-owner/RecipeGen", "split":"test",
                "dataset_revision":"a"*40, "build_id":"test-contract", "archive":"test.zip",
                "member":role+".txt", "artifact_path":"test-owner/RecipeGen/test.zip/"+role+".txt",
                "url":"https://huggingface.co/datasets/test-owner/RecipeGen/resolve/"+"a"*40+"/test.zip",
                "sha256":"c"*64} for role in ["title", "steps"]]
    return {"recipe":{"id":identifier, "recipe_id":identifier, "title":"Cook tomato soup",
                      "origin_archive":"test.zip", "steps_count":3, "ingredient_status":"rule_extracted_incomplete",
                      "total_cooking_time_status":"unknown", "media_state":"metadata_only",
                      "text_complete":True, "quality_flags":["rule_ingredients_are_incomplete"]},
            "title_match":True, "ingredient_match":True,
            "steps":[{"id":f"{identifier}:step:{i}", "order":i, "text":text, "source_id":"source:steps",
                       "order_method":"nonempty_original_lines"} for i,text in enumerate(["Wash tomatoes.","Add water.","Heat the pan."],1)],
            "sources":sources,
            "ingredient_mentions":[{"id":"ingredient:tomato", "normalized_name":"tomato", "zh_name":"番茄",
                                    "extraction_method":"dictionary_rule", "semantics":"text_mention_candidate",
                                    "verified":False, "complete_ingredient_list":False, "evidence_json":"[]"}],
            "media":{"images":7, "videos":1, "downloaded_images":0, "downloaded_videos":0,
                     "recognition_not_run":8, "storage_modes":["remote_archive_member"], "association_semantics":"directory_membership_only"}}


class Record(dict):
    def data(self): return dict(self)


class ReadOnlySession:
    def __init__(self, records=None, builds=None):
        self.records = [native_record()] if records is None else records
        self.builds = [build_record()] if builds is None else builds
        self.queries, self.read_transactions = [], 0

    def execute_read(self, callback):
        self.read_transactions += 1
        return callback(self)

    def execute_write(self, callback):
        raise AssertionError("query CLI must never use write routing")

    def run(self, query, parameters):
        assert not re.search(r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|LOAD)\b",query,re.I)
        self.queries.append((query,copy.deepcopy(parameters)))
        if query == QUERY.ACTIVE_BUILD_QUERY:
            return [Record(**row) for row in self.builds]
        if query == QUERY.SEARCH_QUERY:
            return [Record(**row) for row in self.records]
        raise AssertionError("only fixed native search queries are permitted")


def test_read_only_native_search_preserves_steps_sources_and_counts():
    rows = [native_record("recipe:"+str(i)) for i in range(3)]
    session = ReadOnlySession(rows)
    report = QUERY.search_session(session," TOMATO ",3)
    assert session.read_transactions == 1
    assert len(session.queries) == 2  # Constant query count, independent of results.
    assert report["query"] == "TOMATO" and report["count"] == 3
    assert report["read_only"] is True and report["llm_called"] is False
    assert report["build"]["split"] == "test"
    first = report["results"][0]
    assert first["steps"] == rows[0]["steps"]
    assert first["sources"] == rows[0]["sources"]
    assert first["media"]["images"] == 7 and first["media"]["videos"] == 1
    assert first["media"]["recognition_not_run"] == 8
    assert first["inventory_or_exclusion_verified"] is False
    assert any("不完整" in text for text in report["limitations"])
    params = session.queries[1][1]
    assert params["normalized_query"] == "tomato" and params["limit"] == 3
    assert params["build_id"] == "test-contract"


def test_fixed_query_aggregates_independent_branches_without_cartesian_counts():
    query = QUERY.SEARCH_QUERY
    assert query.count("CALL (r) {") == 4
    assert "WITH DISTINCT s ORDER BY s.order" in query
    assert "WITH DISTINCT m" in query
    assert "collect(DISTINCT m.storage)" in query
    assert "toLower(coalesce(i.normalized_name, '')) = $normalized_query" in query
    assert "coalesce(i.zh_name, '') = $query" in query
    assert "CONTAINS $normalized_query" in query
    assert "LIMIT $limit" in query


def test_query_injection_stays_in_parameter_data():
    session = ReadOnlySession(records=[])
    payload = "tomato') DETACH DELETE n //"
    report = QUERY.search_session(session,payload)
    assert report["status"] == "no_match"
    assert all(payload not in query for query,_ in session.queries)
    assert session.queries[1][1]["query"] == payload


@pytest.mark.parametrize("query,limit",[("",3),("   ",3),("x"*501,3),("tomato",0),("tomato",21),("tomato",True)])
def test_invalid_query_fails_before_database_read(query,limit):
    session = ReadOnlySession()
    with pytest.raises(QUERY.QueryError): QUERY.search_session(session,query,limit)
    assert session.read_transactions == 0


@pytest.mark.parametrize("builds",[[],[build_record(),build_record()]])
def test_missing_or_ambiguous_active_build_never_queries_recipes(builds):
    session = ReadOnlySession(builds=builds)
    with pytest.raises(QUERY.QueryError,match="唯一"): QUERY.search_session(session,"tomato")
    assert len(session.queries) == 1


@pytest.mark.parametrize("field,value",[("split","train"),("status","importing"),("active",False)])
def test_invalid_active_build_contract_is_rejected(field,value):
    build = build_record()
    build[field] = value
    with pytest.raises(QUERY.QueryError,match="范围约束"):
        QUERY.search_session(ReadOnlySession(builds=[build]),"tomato")


@pytest.mark.parametrize("change,match",[("unordered","顺序"),("duplicate_order","顺序"),("missing_text","原文"),
                                        ("wrong_step_source","Source"),("wrong_source_split","范围"),
                                        ("missing_url","url"),("missing_sources","真实文本"),
                                        ("wrong_count","数量"),("negative_media","非负")])
def test_incomplete_native_data_is_not_invented(change,match):
    row = native_record()
    if change == "unordered": row["steps"].reverse()
    if change == "duplicate_order": row["steps"][1]["order"] = 1
    if change == "missing_text": row["steps"][0]["text"] = None
    if change == "wrong_step_source": row["steps"][0]["source_id"] = "source:title"
    if change == "wrong_source_split": row["sources"][0]["split"] = "train"
    if change == "missing_url": row["sources"][0]["url"] = None
    if change == "missing_sources": row["sources"] = []
    if change == "wrong_count": row["recipe"]["steps_count"] = 4
    if change == "negative_media": row["media"]["images"] = -1
    with pytest.raises(QUERY.QueryError,match=match):
        QUERY.search_session(ReadOnlySession(records=[row]),"tomato")


def config_file(tmp_path):
    path = tmp_path / "local-neo4j.json"
    values = {"uri":"bolt://127.0.0.1:8767", "user":"neo4j", "password":"private-test-secret", "database":"neo4j"}
    path.write_text(json.dumps(values),encoding="utf-8")
    path.chmod(0o600)
    return path


def test_private_config_and_environment_precedence_keep_password_out_of_repr(tmp_path):
    path = config_file(tmp_path)
    local = QUERY.load_connection(path,{})
    assert local.uri == "bolt://127.0.0.1:8767"
    assert local.password == "private-test-secret"
    assert "private-test-secret" not in repr(local)
    override = QUERY.load_connection(path,{"NEO4J_PASSWORD":"env-secret","NEO4J_DATABASE":"chosen"})
    assert override.password == "env-secret" and override.database == "chosen"


def test_complete_environment_does_not_read_local_config(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("NOT JSON")
    connection = QUERY.load_connection(path,{"NEO4J_URI":"bolt://127.0.0.1:9876", "NEO4J_USER":"reader", "NEO4J_PASSWORD":"env-only"})
    assert connection.user == "reader"


@pytest.mark.skipif(os.name != "posix",reason="POSIX permission contract")
def test_nonprivate_credential_file_is_rejected(tmp_path):
    path = config_file(tmp_path)
    path.chmod(0o644)
    with pytest.raises(QUERY.QueryError,match="600"): QUERY.load_connection(path,{})


def test_cli_error_redacts_driver_error_credentials(tmp_path,monkeypatch,capsys):
    path = config_file(tmp_path)
    monkeypatch.setattr(QUERY,"load_connection",lambda _:QUERY.Connection("bolt://127.0.0.1:8767","neo4j","must-not-print"))
    def failed(*args): raise RuntimeError("auth failed password=must-not-print")
    monkeypatch.setattr(QUERY,"connect_and_search",failed)
    output = tmp_path / "report.json"
    assert QUERY.main(["--query","tomato","--config",str(path),"--output",str(output)]) == 1
    assert "must-not-print" not in capsys.readouterr().out
    assert "must-not-print" not in output.read_text()


def test_cli_success_saves_graph_only_report(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(QUERY,"load_connection",lambda _:QUERY.Connection("bolt://127.0.0.1:8767","neo4j","hidden"))
    monkeypatch.setattr(QUERY,"connect_and_search",lambda connection,query,limit:QUERY.search_session(ReadOnlySession(),query,limit))
    output = tmp_path / "result.json"
    assert QUERY.main(["--query","番茄","--limit","3","--output",str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["llm_called"] is False
    assert result["results"][0]["steps"][0]["text"] == "Wash tomatoes."
    assert "hidden" not in capsys.readouterr().out
