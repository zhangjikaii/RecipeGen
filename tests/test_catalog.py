"""Public synthetic native contracts; no database, models, downloads or credentials."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest

from recipegen import catalog as c
from recipegen.config import Settings

RID = "recipegen:test:test:01234567890123456789"
BUILD = {"build_id":"synthetic-catalog-build","dataset_revision":c.REVISION,"split":"test",
         "active":True,"status":"verified","content_sha256":"a"*64}
COUNTS = {"recipes":2,"steps":4,"ingredient_entities":2,"images":3,"videos":1,
          "image_observations":3,"distinct_recognized_images":2}


def source(identifier,role,member,kind="text"):
    return {"id":identifier,"source_id":identifier,"role":role,"dataset":c.DATASET_ID,
            "source_type":"huggingface_zip_"+kind,"build_id":BUILD["build_id"],
            "dataset_revision":c.REVISION,"split":"test","archive":"test.zip","member":member,
            "artifact_path":c.DATASET_ID+"/test.zip/"+member,
            "url":f"https://huggingface.co/datasets/{c.DATASET_ID}/resolve/{c.REVISION}/test.zip",
            "sha256":"b"*64 if kind=="text" else None}


def native_row():
    sources = [source("source:title","title","recipe/goal.txt"),source("source:steps","steps","recipe/steps.txt")]
    inventory = [{"id":kind+":fixture","kind":kind,"source_id":"source:"+kind,
                  "archive":"test.zip","member":"recipe/media."+suffix,"display_path":"recipe/media."+suffix,
                  "size_bytes":100,"crc32":"a"*8,"storage":"remote_archive_member",
                  "source":source("source:"+kind,"media","recipe/media."+suffix,"media")}
                 for kind,suffix in [("image","jpg"),("video","mp4")]]
    observation = {"observation_id":"mm:observation:fixture","caption":"Tomatoes in a bowl.",
                   "source_media_id":"image:fixture","source_id":"source:image","run_id":"synthetic-run",
                   "finished_at":None,"step_id":"step:1","step_order":1,"score":0.6,
                   "build_id":BUILD["build_id"],"dataset_revision":c.REVISION,"split":"test",
                   "raw_json":json.dumps({"caption":"Tomatoes in a bowl."})}
    observation["evidence_json"] = json.dumps({key:observation[key] for key in
        ("source_media_id","source_id","run_id","build_id","dataset_revision","split")})
    evidence = [{"step_order":1,"source":{"id":"source:steps"},"text":"Wash tomatoes."}]
    return {"recipe":{"id":RID,"recipe_id":RID,"title":"Make tomato soup","origin_archive":"test.zip",
                      "steps_count":2,"text_complete":True,"quality_flags":[],"is_demo":False,
                      "build_id":BUILD["build_id"],"dataset_revision":c.REVISION,"split":"test"},
            "steps":[{"id":"step:1","order":1,"text":"Wash tomatoes.","source_id":"source:steps"},
                     {"id":"step:2","order":2,"text":"Boil water.","source_id":"source:steps"}],
            "sources":sources,"ingredient_mentions":[{"id":"ingredient:tomato","normalized_name":"tomato",
                "zh_name":"番茄","extraction_method":"dictionary_rule","semantics":"text_mention_candidate",
                "verified":False,"complete_ingredient_list":False,"evidence_json":json.dumps(evidence)}],
            "media":{"images":1,"videos":1,"downloaded_images":0,"downloaded_videos":0,
                     "recognition_not_run":2,"storage_modes":["remote_archive_member"],"inventory":inventory},
            "visual_evidence":[observation]}


class Record(dict):
    def data(self): return dict(self)


class Driver:
    def __init__(self,row=None,builds=None):
        self.row = native_row() if row is None else row
        self.builds = [BUILD] if builds is None else builds
        self.queries = []
        self.reads = 0
        self.closed = False
        self.failure = None
        self.graph_bad_endpoint = False

    def session(self,**kwargs):
        assert kwargs["default_access_mode"] == "READ"
        return self
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def execute_read(self,callback):
        self.reads += 1
        if self.failure: raise self.failure
        return callback(self)
    def execute_write(self,*args): raise AssertionError("catalog cannot write")
    def close(self): self.closed=True
    def run(self,cypher,parameters):
        assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b",cypher,re.I)
        self.queries.append((cypher,copy.deepcopy(parameters)))
        if cypher==c.ACTIVE_BUILD_QUERY:return [Record(v) for v in self.builds]
        assert parameters["build_id"] == BUILD["build_id"]
        assert parameters["dataset_revision"] == c.REVISION
        if cypher==c.STATUS_QUERY:return [Record(COUNTS)]
        if cypher in (c.DETAIL_QUERY,c.SEARCH_QUERY):return [] if self.row is False else [Record(copy.deepcopy(self.row))]
        if cypher==c.GRAPH_NODES_QUERY:
            return [Record({"id":identifier,"label":"VisualObservation" if identifier.startswith("mm:") else
                {"recipegen":"Recipe","ingredient":"Ingredient","source":"Source","step":"Step",
                 "image":"Image","video":"Video"}[identifier.split(":")[0]],"name":identifier})
                    for identifier in parameters["node_ids"]]
        if cypher==c.GRAPH_EDGES_QUERY:
            return [Record({"source":RID,"target":"other:leak" if self.graph_bad_endpoint else "step:1",
                            "type":"HAS_STEP","candidate":False})]
        raise AssertionError("fixed queries only")


@pytest.fixture
def backend(tmp_path):
    driver = Driver()
    return c.RecipeCatalog(driver=driver,project_root=tmp_path),driver


def test_status_is_real_native_count_scope_and_never_demo(backend):
    catalog,driver = backend
    value = catalog.status()
    assert value["graph"] == COUNTS and value["backend"] == "neo4j"
    assert value["dataset"]["is_demo"] is False and value["dataset"]["revision"] == c.REVISION
    assert value["capabilities"]["inventory_guarantee"] is False
    assert driver.reads == 1 and len(driver.queries)==2


def test_detail_preserves_original_text_sources_and_visual_candidate(backend):
    catalog,driver = backend
    value = catalog.recipe(RID)
    assert value["steps"] == native_row()["steps"]
    assert value["sources"] == native_row()["sources"]
    assert value["ingredient_mentions"][0]["name"] == "tomato"
    assert value["ingredient_mentions"][0]["source_id"] == "source:steps"
    assert value["media"]["recognized_images"] == 1 and value["media"]["videos"] == 1
    assert value["semantics"]["ingredients_complete"] is False and value["semantics"]["duration_known"] is False
    assert value["visual_evidence"][0]["candidate"] is True
    assert value["visual_evidence"][0]["latest_unverified"] is True
    assert "WITH DISTINCT r,image" in c.DETAIL_BODY
    assert "finished_at IS NULL ASC" in c.DETAIL_BODY
    assert "LIMIT 1" in c.DETAIL_BODY
    assert driver.reads==1


@pytest.mark.parametrize("question,positive,excluded",[
    ("番茄不要鸡蛋",["tomato"],["egg"]),
    ("西红柿和马铃薯，不加洋葱",["tomato","potato"],["onion"]),
    ("What can I cook with tomatoes and rice without eggs?",["tomato","rice"],["egg"]),
    ("不要鸡蛋和洋葱，但可以用番茄",["tomato"],["egg","onion"]),
    ("tomato soup without onion",["tomato"],["onion"]),
    ("tomato without eggs and mystery-ingredient",["tomato"],["egg","mystery ingredient"]),
])
def test_natural_filters_remove_negated_mentions(question,positive,excluded):
    parsed = c.parse_query_filters(question)
    assert parsed["positive"] == positive and parsed["excluded"] == excluded
    assert parsed["semantics"] == "dictionary_text_mentions_only"


def test_search_natural_query_and_structured_filters_stay_parameters(backend):
    catalog,driver = backend
    value = catalog.search("用番茄做什么，不吃鸡蛋",ingredients=["西红柿"],excluded_ingredients=["洋葱"],limit=6)
    query,parameters = driver.queries[-1]
    assert query==c.SEARCH_QUERY
    assert parameters["terms"] == [{"raw":"番茄","names":["番茄","tomato"]}] or parameters["terms"] == [{"raw":"tomato","names":["tomato"]}]
    assert parameters["ingredients"][0]["names"]==["西红柿","tomato"]
    assert [term["names"][-1] for term in parameters["excluded"]]==["onion","egg"]
    assert value["applied_filters"]["excluded_ingredients"]==["onion","egg"]
    assert value["applied_filters"]["allergy_safe"] is False
    assert "all(term IN $ingredients" in query and "none(term IN $excluded" in query


def test_english_title_keywords_remain_title_search_and_empty_browse(backend):
    catalog,driver = backend
    catalog.search("tomato soup")
    assert [t["names"] for t in driver.queries[-1][1]["terms"]] == [["tomato"],["soup"]]
    catalog.search("")
    assert driver.queries[-1][1]["terms"] == []


def test_full_title_priority_is_parameterized_before_limit_and_keeps_filters(backend):
    catalog,driver = backend
    title = "Steamed Salted Pork with Winter Melon"
    catalog.search(f"  {title.upper()} without eggs",ingredients=["pork"],excluded_ingredients=["onion"],limit=6)
    query,parameters = driver.queries[-1]
    assert query == c.SEARCH_QUERY and title.lower() not in query
    assert parameters["title_query"] == title.lower()
    assert parameters["terms"] == c._terms(title.upper())
    assert parameters["ingredients"] == [c._term("pork")]
    assert [term["names"][-1] for term in parameters["excluded"]] == ["onion","egg"]
    filters,ranking = query.split("WITH r,toLower(trim(coalesce(r.title,''))) AS normalized_title",1)
    assert all(clause in filters for clause in ("all(term IN $terms","all(term IN $ingredients","none(term IN $excluded"))
    assert "$title_query" not in filters  # Ranking cannot widen matching scope.
    assert ranking.count("$title_query <> ''") == 2
    assert ranking.index("normalized_title = $title_query THEN 0") < ranking.index("normalized_title CONTAINS $title_query THEN 1")
    assert ranking.index("ELSE 2 END") < ranking.index("toLower(coalesce(r.title,'')),r.kg_csv_id") < ranking.index("LIMIT $limit")
    catalog.search("")
    assert driver.queries[-1][1]["title_query"] == ""


def test_direct_id_uses_detail_with_same_exclusion_rule(backend):
    catalog,driver = backend
    value=catalog.search(RID,excluded_ingredients=["番茄"])
    assert value["count"]==0 and driver.queries[-1][0]==c.DETAIL_QUERY


def test_missing_recipe_and_empty_search_have_different_contracts(tmp_path):
    driver=Driver(row=False);catalog=c.RecipeCatalog(driver=driver,project_root=tmp_path)
    assert catalog.search("tomato")["count"]==0
    with pytest.raises(c.RecipeNotFound):catalog.recipe(RID)


@pytest.mark.parametrize("builds",[[],[BUILD,BUILD],[{**BUILD,"dataset_revision":"c"*40}],
                                  [{**BUILD,"split":"train"}],[{**BUILD,"active":False}]])
def test_fixed_revision_and_unique_active_verified_build_required(tmp_path,builds):
    driver=Driver(builds=builds);catalog=c.RecipeCatalog(driver=driver,project_root=tmp_path)
    with pytest.raises(c.CatalogError):catalog.status()
    assert len(driver.queries)==1


@pytest.mark.parametrize("defect",["steps_reversed","missing_source","demo","media_source_mismatch",
                                   "ingredient_complete","ingredient_source","duplicate_observation",
                                   "foreign_step","visual_source_json"])
def test_incomplete_or_foreign_native_facts_fail_closed(tmp_path,defect):
    row=native_row()
    if defect=="steps_reversed":row["steps"].reverse()
    elif defect=="missing_source":row["sources"]=[]
    elif defect=="demo":row["recipe"]["is_demo"]=True
    elif defect=="media_source_mismatch":row["media"]["inventory"][0]["source"]["archive"]="train.zip"
    elif defect=="ingredient_complete":row["ingredient_mentions"][0]["complete_ingredient_list"]=True
    elif defect=="ingredient_source":row["ingredient_mentions"][0]["evidence_json"]='[{"source":{"id":"other"}}]'
    elif defect=="duplicate_observation":row["visual_evidence"]*=2
    elif defect=="foreign_step":row["visual_evidence"][0]["step_id"]="step:other"
    else:row["visual_evidence"][0]["evidence_json"]='{}'
    catalog=c.RecipeCatalog(driver=Driver(row=row),project_root=tmp_path)
    with pytest.raises(c.CatalogError):catalog.recipe(RID)


def test_graph_queries_actual_nodes_and_edges_in_one_read_transaction(backend):
    catalog,driver = backend
    value=catalog.graph(RID)
    assert value["edges"]==[{"source":RID,"target":"step:1","type":"HAS_STEP","candidate":False}]
    assert "source:image" in {node["id"] for node in value["nodes"]}
    assert "mm:observation:fixture" in {node["id"] for node in value["nodes"]}
    assert driver.reads==1 and len(driver.queries)==4
    assert driver.queries[-1][1]["visual_selection"]==[{"media_id":"image:fixture","run_id":"synthetic-run"}]
    assert "selected.media_id = h.source_media_id" in c.GRAPH_EDGES_QUERY
    driver.graph_bad_endpoint=True
    with pytest.raises(c.CatalogError):catalog.graph(RID)


def test_latest_priority_uses_verified_finished_at_and_cache_not_mtime(tmp_path):
    path=tmp_path/"data/multimodal/runs/public/state.json";path.parent.mkdir(parents=True)
    state={"signature":{"dataset_revision":c.REVISION},"records":{
        "r1":{"status":"verified","run_id":"older","finished_at":"2026-10-01T00:00:00+00:00"},
        "r2":{"status":"verified","run_id":"newer","finished_at":"2026-10-02T00:00:00+00:00"},
        "r3":{"status":"failed","run_id":"failed","finished_at":"2026-10-03T00:00:00+00:00"}}}
    path.write_text(json.dumps(state))
    catalog=c.RecipeCatalog(driver=Driver(),project_root=tmp_path)
    priorities={p["run_id"]:p["finished_at"] for p in catalog._run_priorities()}
    assert priorities["newer"]>priorities["older"] and "failed" not in priorities
    path.unlink()
    assert catalog._run_priorities()==[{"run_id":key,"finished_at":value} for key,value in sorted(priorities.items())]


def test_driver_failure_is_redacted_without_demo_fallback(backend):
    catalog,driver=backend
    driver.failure=RuntimeError("secret-test-password and bolt://private.invalid")
    with pytest.raises(c.CatalogError) as captured:catalog.status()
    assert "secret-test-password" not in str(captured.value) and "private.invalid" not in str(captured.value)
    assert "示例" in str(captured.value)


@pytest.mark.parametrize("query,ingredients,excluded,limit",[(None,[],[],6),("x",[{}],[],6),
    ("x",[],"egg",6),("x",[],[],True),("x",[],[],31)])
def test_bad_request_rejected_before_database_read(backend,query,ingredients,excluded,limit):
    catalog,driver=backend
    with pytest.raises(c.CatalogRequestError):catalog.search(query,ingredients,excluded,limit)
    assert driver.reads==0


@pytest.mark.parametrize("query,ingredients,excluded", [
    ("x" * 501, [], []),
    (" ".join("unknown" + str(i) for i in range(13)), [], []),
    ("x", ["item" + str(i) for i in range(21)], []),
    ("x", [], ["x" * 81]),
])
def test_request_limits_use_request_error_and_do_not_read_database(backend, query, ingredients, excluded):
    catalog, driver = backend
    with pytest.raises(c.CatalogRequestError):
        catalog.search(query, ingredients, excluded)
    assert driver.reads == 0


def test_request_error_is_distinct_from_unavailable_or_invalid_graph():
    assert issubclass(c.CatalogRequestError, ValueError)
    assert not issubclass(c.CatalogRequestError, c.CatalogError)


def test_injection_text_is_parameter_data_only(backend):
    catalog,driver=backend
    payload="soup') DETACH DELETE n //"
    catalog.search(payload)
    assert all(payload not in cypher for cypher,_ in driver.queries)


def test_settings_constructor_is_lazy_and_owned_driver_closes(monkeypatch,tmp_path):
    import neo4j
    driver=Driver();seen={}
    def factory(uri,**kwargs):
        seen.update(uri=uri,kwargs=kwargs)
        return driver
    monkeypatch.setattr(neo4j.GraphDatabase,"driver",factory)
    settings=Settings(neo4j_uri="bolt://fixture.invalid:7687",neo4j_user="reader",neo4j_password="synthetic-secret")
    catalog=c.RecipeCatalog(settings=settings,project_root=tmp_path)
    assert not seen
    assert catalog.status()["backend"]=="neo4j"
    assert seen["kwargs"]["auth"]==("reader","synthetic-secret")
    assert "synthetic-secret" not in repr(catalog)
    catalog.close()
    assert driver.closed is True


def test_normalization_known_aliases_is_not_a_broad_ingredient_guess():
    assert c.normalize_ingredient("西红柿")=="tomato"
    assert c.normalize_ingredient(" Eggs ")=="egg"
    assert c.normalize_ingredient("青瓜")=="cucumber"
    assert c.normalize_ingredient("蛋")=="蛋"
