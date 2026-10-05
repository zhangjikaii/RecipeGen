"""Synthetic artifact fixtures test assembly, never genuine model acceptance."""
from __future__ import annotations

from collections import Counter
import copy
import json
from types import SimpleNamespace

import pytest

from recipegen.multimodal_align import align_multimodal
from recipegen.multimodal_bundle import MultimodalBundleError, build_multimodal_bundle
from recipegen.multimodal_graph import validate_bundle

REVISION = "a"*40
SCOPE = {"build_id":"synthetic-build", "dataset_revision":REVISION, "split":"test"}
SPACE = {"model":"synthetic-clip-fixture", "revision":"c"*40, "dimension":512}
MODEL = {"name":"synthetic-vlm-fixture", "revision":"b"*40}
MODELS = {**MODEL, "pipeline_run_id":"unit-test-controller", "embedding_space":SPACE}
ENTITIES = [{"id":"ingredient:tomato","type":"Ingredient","name":"tomato","aliases":["番茄"]},
            {"id":"ingredient:ham","type":"Ingredient","name":"ham","aliases":[]},
            {"id":"tool:knife","type":"Tool","name":"knife","aliases":[]},
            {"id":"action:slice","type":"Action","name":"slice","aliases":[]}]


def vector(axis):
    return [float(index == axis) for index in range(512)]


def artifacts():
    image = {"media_id":"image:1","source_id":"source:image","path":"synthetic/image.jpg","sha256":"1"*64,
             "archive":"test.zip","member":"images/1.jpg","revision":REVISION,"split":"test"}
    video = {"media_id":"video:1","source_id":"source:video","path":"synthetic/video.mp4","sha256":"2"*64,
             "archive":"test-video.zip","member":"videos/1.mp4","revision":REVISION,"split":"test",
             "timeline_origin_sec":100,"time_base":"1/1000","duration_sec":10,"clips":[
                {"clip_id":"clip:1","start_sec":0,"end_sec":5,"media_id":"video:1","frames":[
                    {"frame_id":"frame:1","path":"synthetic/f1.jpg","sha256":"3"*64,"timestamp_sec":100,"offset_sec":0,"pts":100000,"time_base":"1/1000"},
                    {"frame_id":"frame:2","path":"synthetic/f2.jpg","sha256":"4"*64,"timestamp_sec":104,"offset_sec":4,"pts":104000,"time_base":"1/1000"}]},
                {"clip_id":"clip:2","start_sec":5,"end_sec":10,"media_id":"video:1","frames":[
                    {"frame_id":"frame:3","path":"synthetic/f3.jpg","sha256":"5"*64,"timestamp_sec":106,"offset_sec":6,"pts":106000,"time_base":"1/1000"}]}]}
    prepared = {"split":"test","revision":REVISION,"recipe_id":None,"images":[image],"videos":[video],
                "steps":[{"id":"step:1","order":1,"text":"Wash tomatoes.","source_id":"source:text"},
                         {"id":"step:2","order":2,"text":"Slice tomatoes.","source_id":"source:text"}]}
    observations = []
    for identifier,kind,source,axis in [("image:1","image",image,0),("clip:1","clip",video,1)]:
        objects = [{"name":"tomato","id":"tomato-box","bbox":[0.1,0.2,0.4,0.6]},
                   {"name":"bowl","id":"bowl-box","bbox":None}, {"name":"knife","id":"knife-box","bbox":None}]
        caption = "A tomato and knife, ham in the caption only."
        raw = {"caption":caption,"objects":copy.deepcopy(objects),"actions":["slice"],
               "relations":[{"subject":"tomato","predicate":"on","object":"bowl","raw_score":0.45}]}
        observations.append({"id":identifier,"kind":kind,"source_media_id":source["media_id"],"source_id":source["source_id"],
                             "caption":caption,"objects":objects,"actions":["slice"],"raw_output":raw,
                             "embedding":vector(axis),"embedding_space":dict(SPACE),"model":dict(MODEL),
                             "evidence":{"source_media_id":source["media_id"],"source_id":source["source_id"],
                                         "media_sha256":source["sha256"],"input_sha256":"6"*64,"synthetic_fixture":True}})
    metadata = {**SCOPE,"entities":copy.deepcopy(ENTITIES)}
    alignment = align(prepared,observations,metadata)
    return [prepared,observations,alignment,metadata]


def align(prepared,observations,metadata):
    media = [{**row,"kind":row["kind"],"timeline_id":row["source_media_id"],
              **({"start_seconds":0,"end_seconds":5} if row["kind"] == "clip" else {})} for row in observations]
    steps = [{**row,"embedding":vector(index)} for index,row in enumerate(prepared["steps"])]
    return align_multimodal({"embedding_space":SPACE,"steps":steps,"media":media,"entities":metadata["entities"]})


def build(*args,run_id="synthetic-run",models=None):
    return build_multimodal_bundle(*(args or artifacts()),run_id,models or MODELS)


def base_graph(prepared,metadata):
    nodes,edges = [],[]
    for label,entries in (("Image",prepared["images"]),("Video",prepared["videos"])):
        for row in entries:
            source = {**SCOPE,"source_type":"huggingface_zip_media","source_id":row["source_id"],
                      "archive":row["archive"],"member":row["member"],
                      "artifact_path":"synthetic/RecipeGen/"+row["archive"]+"/"+row["member"],
                      "url":"https://huggingface.co/datasets/synthetic/RecipeGen/resolve/"+REVISION+"/"+row["archive"]}
            nodes.extend([{"id":row["media_id"],"label":label,"properties":{**SCOPE,"source_id":row["source_id"],"archive":row["archive"],"member":row["member"]}},
                          {"id":row["source_id"],"label":"Source","properties":source}])
            edges.append({"id":"source-edge:"+row["media_id"],"start_id":row["media_id"],"end_id":row["source_id"],"type":"HAS_SOURCE","properties":SCOPE})
    nodes.extend({"id":row["id"],"label":"Step","properties":{**row,**SCOPE}} for row in prepared["steps"])
    nodes.extend({"id":row["id"],"label":row["type"],"properties":{**row,**SCOPE}} for row in metadata["entities"])
    return SimpleNamespace(build_id=SCOPE["build_id"],dataset_revision=REVISION,content_sha256="d"*64,nodes=nodes,relationships=edges)


def label_rows(bundle,label):
    return [row for row in bundle["nodes"] if row["label"] == label]


def test_bundle_passes_actual_offline_graph_contract_with_bound_source_evidence():
    prepared,observations,alignment,metadata = artifacts()
    original = copy.deepcopy((prepared,observations,alignment,metadata))
    bundle = build(prepared,observations,alignment,metadata)
    verified = validate_bundle(bundle,base_graph(prepared,metadata))
    assert verified.summary()["semantic_accuracy_verified"] is False
    counts = Counter(row["label"] for row in bundle["nodes"])
    assert counts["VisualObservation"] == 2 and counts["VisualObject"] == 6
    assert counts["VideoClip"] == 2 and counts["Frame"] == 3 and counts["SemanticRun"] == 1
    assert "Recipe" not in counts
    assert len([row for row in bundle["relationships"] if row["type"] == "HAS_OBSERVATION"]) == 2
    run = label_rows(bundle,"SemanticRun")[0]["properties"]
    assert run["pipeline_run_id"] == "unit-test-controller" and run["observation_count"] == 2
    assert (prepared,observations,alignment,metadata) == original
    for row in bundle["nodes"]:
        if row["properties"].get("reference"):
            assert row["properties"] == {"reference":True}
        elif row["label"] != "SemanticRun":
            props = row["properties"]
            assert props["verified"] is False and props["confidence"] is None
            evidence = json.loads(props["evidence_json"])
            assert evidence["source_media_id"] == props["source_media_id"]
            assert evidence["source_id"] == props["source_id"]
    for row in bundle["relationships"]:
        assert row["properties"]["confidence"] is None and row["properties"]["verified"] is False
    json.dumps(bundle,allow_nan=False)


def test_observation_embeddings_record_clip_space_and_preserve_raw_model_output():
    prepared,observations,alignment,metadata = artifacts()
    bundle = build(prepared,observations,alignment,metadata)
    for node in label_rows(bundle,"VisualObservation"):
        p = node["properties"]
        obs = next(row for row in observations if row["id"] == p["observation_id"])
        assert p["embedding"] == obs["embedding"]
        assert p["embedding_model"] == SPACE["model"] != MODEL["name"]
        assert p["embedding_revision"] == SPACE["revision"]
        assert p["model"] == MODEL["name"]
        assert json.loads(p["raw_json"]) == obs["raw_output"]
        assert json.loads(p["model_json"]) == MODEL


def test_frame_offset_clock_preserves_original_pts_and_strict_before():
    bundle = build()
    frames = {row["properties"]["frame_id"]:row for row in label_rows(bundle,"Frame")}
    assert frames["frame:1"]["properties"]["timestamp_seconds"] == 0
    assert frames["frame:1"]["properties"]["source_timestamp_seconds"] == 100
    assert frames["frame:1"]["properties"]["pts"] == 100000
    by_node = {row["id"]:row for row in bundle["nodes"]}
    temporal = [row for row in bundle["relationships"] if row["type"] == "BEFORE"]
    assert len(temporal) == 3
    for edge in temporal:
        start,end = by_node[edge["start_id"]],by_node[edge["end_id"]]
        if start["label"] == "Frame":
            assert start["properties"]["timestamp_seconds"] < end["properties"]["timestamp_seconds"]
        else:
            assert start["properties"]["end_seconds"] <= end["properties"]["start_seconds"]


def test_missing_offset_can_be_derived_only_from_the_declared_original_clock():
    args = artifacts()
    for clip in args[0]["videos"][0]["clips"]:
        for frame in clip["frames"]: frame.pop("offset_sec")
    bundle = build(*args)
    validate_bundle(bundle,base_graph(args[0],args[3]))
    assert {row["properties"]["timestamp_seconds"] for row in label_rows(bundle,"Frame")} == {0,4,6}


def test_caption_only_entities_do_not_become_objects_or_depiction_edges():
    bundle = build()
    assert not any(row["id"] == "ingredient:ham" for row in bundle["nodes"])
    assert not any(row["end_id"] == "ingredient:ham" for row in bundle["relationships"])
    evidence = [json.loads(row["properties"]["alignment_json"]) for row in label_rows(bundle,"VisualObservation")]
    assert any(candidate["entity_id"] == "ingredient:ham" for row in evidence for candidate in row["semantic_entities"])
    assert all(row["properties"]["name"] != "ham" for row in label_rows(bundle,"VisualObject"))


def test_actions_require_model_action_list_and_main_step_only_has_cosine_edges():
    args = artifacts()
    args[1][0]["actions"] = []
    args[1][0]["caption"] += " slice"
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    actions = [row for row in bundle["relationships"] if row["type"] == "SHOWS_ACTION"]
    assert len(actions) == 1 and actions[0]["start_id"] != "image:1"
    step_links = [row for row in bundle["relationships"] if row["type"] == "ALIGNED_WITH" and row["end_id"].startswith("step:")]
    assert len(step_links) == 2
    assert all(row["properties"]["similarity_score"] == pytest.approx(1) for row in step_links)
    assert not any(row["start_id"] == "image:1" and row["end_id"] == "step:2" for row in step_links)


@pytest.mark.parametrize("predicate,expected",[("on","ON"),("inside","IN"),("in","IN"),("beside","BESIDE"),
                                               ("holding","HOLDING"),("using","USING"),("mixed_with","MIXED_WITH")])
def test_model_spatial_predicate_whitelist_maps_to_safe_relationships(predicate,expected):
    args = artifacts()
    args[1][0]["raw_output"]["relations"][0]["predicate"] = predicate
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    assert any(row["type"] == expected for row in bundle["relationships"])
    validate_bundle(bundle,base_graph(args[0],args[3]))


@pytest.mark.parametrize("change,reason",[("unknown","unknown_predicate"),("absent","missing_or_ambiguous_object_endpoint"),
                                         ("duplicate","missing_or_ambiguous_object_endpoint"),("malformed","malformed_relation")])
def test_unresolved_relations_are_preserved_with_rejections_without_invented_edges(change,reason):
    args = artifacts()
    relation = args[1][0]["raw_output"]["relations"][0]
    if change == "unknown": relation["predicate"] = "DELETE n"
    if change == "absent": relation["subject"] = "unseen person"
    if change == "duplicate": args[1][0]["objects"].append({"name":"tomato","bbox":None})
    if change == "malformed": relation["object"] = None
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    image_obs = next(row for row in label_rows(bundle,"VisualObservation") if row["properties"]["source_media_id"] == "image:1")
    assert json.loads(image_obs["properties"]["raw_json"])["relations"][0] == relation
    assert any(row["reason"] == reason for row in json.loads(image_obs["properties"]["rejections_json"]))
    assert not any(row["type"] in {"ON","IN","BESIDE","HOLDING","USING","MIXED_WITH"} and row["properties"]["source_media_id"] == "image:1" for row in bundle["relationships"])


def test_explicit_object_ids_resolve_duplicate_names_without_cartesian_edges():
    args = artifacts()
    args[1][0]["objects"].append({"id":"other-tomato","name":"tomato","bbox":None})
    args[1][0]["raw_output"]["relations"][0]["subject_id"] = "tomato-box"
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    assert len([row for row in bundle["relationships"] if row["type"] == "ON" and row["properties"]["source_media_id"] == "image:1"]) == 1


def test_rejected_bbox_retains_original_annotation_and_is_not_written_as_geometry():
    args = artifacts()
    obj = args[1][0]["objects"][0]
    obj.update(bbox=[-99,0,9999,4],bbox_rejected="out_of_coordinate_range")
    args[1][0]["raw_output"]["objects"][0] = copy.deepcopy(obj)
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    node = next(row for row in label_rows(bundle,"VisualObject") if row["properties"]["source_media_id"] == "image:1" and row["properties"]["object_index"] == 0)
    assert node["properties"]["predicted_bbox"] is None and node["properties"]["bbox_rejected"] is True
    assert json.loads(node["properties"]["raw_json"])["bbox_rejected"] == "out_of_coordinate_range"
    assert json.loads(node["properties"]["evidence_json"])["raw_object"] == obj


def test_orphan_media_with_empty_steps_keeps_observations_without_step_edges():
    args = artifacts()
    args[0]["steps"] = []
    args[2] = align(args[0],args[1],args[3])
    bundle = build(*args)
    assert len(label_rows(bundle,"VisualObservation")) == 2
    assert not label_rows(bundle,"Step") and not label_rows(bundle,"Recipe")
    assert all(json.loads(row["properties"]["alignment_json"])["main"]["status"] == "unmatched" for row in label_rows(bundle,"VisualObservation"))
    validate_bundle(bundle,base_graph(args[0],args[3]))


def test_failed_or_missing_observations_still_keep_real_prepared_video_structure():
    prepared,_,_,metadata = artifacts()
    bundle = build(prepared,[],{},metadata)
    assert not label_rows(bundle,"VisualObservation") and not label_rows(bundle,"VisualObject")
    assert len(label_rows(bundle,"Frame")) == 3 and len(label_rows(bundle,"VideoClip")) == 2
    assert label_rows(bundle,"SemanticRun")[0]["properties"]["observation_count"] == 0
    validate_bundle(bundle,base_graph(prepared,metadata))


def test_same_input_same_run_is_deterministic_and_another_run_has_new_extension_identity():
    args = artifacts()
    first,second = build(*args),build(*args)
    assert first == second
    another = build(*args,run_id="another-run")
    a = {row["id"] for row in first["nodes"] if not row["properties"].get("reference")}
    b = {row["id"] for row in another["nodes"] if not row["properties"].get("reference")}
    assert a.isdisjoint(b)


@pytest.mark.parametrize("change",["source_id","source_media_id","evidence_source","media_hash","zero_vector","short_vector","bool_vector",
                                  "embedding_model","revision","unknown_clip","duplicate_observation","unknown_step","unknown_entity",
                                  "frame_outside","backward_clip","mixed_split","unknown_alignment"])
def test_inconsistent_artifacts_fail_before_returning_a_graph(change):
    args = list(artifacts())
    p,o,a,m = args
    if change == "source_id": o[0]["source_id"] = "wrong"
    if change == "source_media_id": o[0]["source_media_id"] = "unknown"
    if change == "evidence_source": o[0]["evidence"]["source_id"] = "wrong"
    if change == "media_hash": o[0]["evidence"]["media_sha256"] = "0"*64
    if change == "zero_vector": o[0]["embedding"] = [0.0]*512
    if change == "short_vector": o[0]["embedding"] = [1.0]
    if change == "bool_vector": o[0]["embedding"][0] = True
    if change == "embedding_model": o[0]["embedding_space"]["model"] = "another-space"
    if change == "revision": p["revision"] = "f"*40
    if change == "unknown_clip": o[1]["id"] = "absent-clip"
    if change == "duplicate_observation": o.append(copy.deepcopy(o[0]))
    if change == "unknown_step": a["alignments"][0]["step_id"] = "absent-step"
    if change == "unknown_entity": a["semantic_entities"][0]["entity_id"] = "absent-entity"
    if change == "frame_outside": p["videos"][0]["clips"][0]["frames"][0]["offset_sec"] = 9
    if change == "backward_clip": p["videos"][0]["clips"][0]["end_sec"] = 0
    if change == "mixed_split": p["images"][0]["split"] = "train"
    if change == "unknown_alignment": a["alignments"][0]["media_id"] = "absent-observation"
    with pytest.raises(MultimodalBundleError): build(*args)
