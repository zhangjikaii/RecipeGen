"""Offline model-output parsing; no model inference or downloads are performed."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from recipegen.multimodal_models import (
    LocalMultimodalModels, MissingCaptionError, ModelOutputError,
    parse_evidence_selection, parse_short_caption_output, parse_visual_output, sha_file,
)


def visual(**changes):
    return {"caption":"Ingredients sitting in a bowl.", "objects":[], "actions":[], "relations":[], **changes}


def test_observed_string_object_format_is_normalized_without_inventing_boxes():
    # Same output shape reported by the real smoke run. This fixture only tests
    # parsing; its noun labels are not a new real-model/grounding measurement.
    raw = json.dumps(visual(objects=["glass bowl","ginger","dried red fruit","white spoon"]))
    original = raw
    parsed = parse_visual_output(raw)
    assert parsed["objects"] == [{"name":name,"bbox":None,"bbox_status":"not_returned"}
                                  for name in ["glass bowl","ginger","dried red fruit","white spoon"]]
    assert raw == original
    assert parsed["actions"] == []


def test_valid_mixed_object_formats_keep_model_extra_fields():
    source = visual(objects=["ginger",{"name":"bowl","bbox":[1,2,900,999],"unknown_evidence":{"marker":"verbatim"}}],
                    unknown_field={"finite_score":0.25})
    original = copy.deepcopy(source)
    parsed = parse_visual_output(json.dumps(source))
    assert parsed["objects"][1]["bbox"] == [1,2,900,999]
    assert parsed["objects"][1]["unknown_evidence"] == {"marker":"verbatim"}
    assert parsed["unknown_field"] == source["unknown_field"]
    assert source == original


@pytest.mark.parametrize("bbox",[[1,2,3],[2,1,1,4],[1,4,5,2],[-1,0,500,900],[0,0,1001,999],[True,0,500,900],"not a box"])
def test_invalid_finite_box_is_retained_and_rejected_without_dropping_objects(bbox):
    parsed = parse_visual_output(json.dumps(visual(objects=[{"name":"bowl","bbox":bbox}])))
    obj = parsed["objects"][0]
    assert obj["bbox"] is None and obj["bbox_rejected"] is True
    assert obj["bbox_raw"] == bbox and obj["bbox_status"] == "rejected"
    assert obj["name"] == "bowl"


@pytest.mark.parametrize("object_row",[{"name":"ginger"},{"name":"ginger","bbox":None}])
def test_missing_or_null_box_remains_an_ungrounded_object_candidate(object_row):
    obj = parse_visual_output(json.dumps(visual(objects=[object_row])))["objects"][0]
    assert obj.get("bbox") is None and obj["bbox_status"] == "not_returned"
    assert "bbox_rejected" not in obj


@pytest.mark.parametrize("wrapper",["{}","  {}\n","```json\n{}\n```","```\n{}\n```"])
def test_one_full_json_object_or_one_full_code_fence_is_accepted(wrapper):
    raw = json.dumps(visual(caption="A literal {brace} in model text."))
    assert parse_visual_output(wrapper.format(raw))["caption"] == "A literal {brace} in model text."


@pytest.mark.parametrize("raw",[
    'prefix {"caption":"x","objects":[],"actions":[],"relations":[]}',
    '{"caption":"x","objects":[],"actions":[],"relations":[]} suffix',
    '{"caption":"x","objects":[],"actions":[],"relations":[]} {}',
    '```json\n{"caption":"x","objects":[],"actions":[],"relations":[]}\n``` trailing',
    '[]','null','42','"caption"','{"caption":"x"',
    '{"caption":"x","caption":"y","objects":[],"actions":[],"relations":[]}',
    '{"caption":"x","objects":[],"actions":[],"relations":[],"metadata":{"x":1,"x":2}}',
])
def test_ambiguous_trailing_duplicate_or_nonobject_json_is_rejected(raw):
    with pytest.raises(ModelOutputError): parse_visual_output(raw)


@pytest.mark.parametrize("constant",["NaN","Infinity","-Infinity","1e999","-1e999"])
@pytest.mark.parametrize("location",["root","unknown_nested","box"])
def test_nonfinite_numbers_are_rejected_even_in_unknown_fields(constant,location):
    value = visual()
    if location == "root": value["unknown_score"] = "NONFINITE"
    if location == "unknown_nested": value["unknown"] = [{"deep":{"score":"NONFINITE"}}]
    if location == "box": value["objects"] = [{"name":"ginger","bbox":[0,0,"NONFINITE",999]}]
    raw = json.dumps(value).replace('"NONFINITE"',constant)
    with pytest.raises(ModelOutputError): parse_visual_output(raw)


@pytest.mark.parametrize("change",[
    {"caption":" "},{"caption":None},{"objects":None},{"objects":[1]}, {"objects":[" "]},
    {"objects":[{}]},{"objects":[{"name":False}]},{"actions":[False]},{"actions":[" "]},
    {"relations":[{}]},{"relations":[{"subject":"ginger","predicate":"on","object":None}]},
    {"objects":["ginger"]*33},{"actions":["slice"]*33},{"relations":[{"subject":"a","predicate":"on","object":"b"}]*33},
])
def test_required_interface_shapes_are_enforced(change):
    with pytest.raises(ModelOutputError): parse_visual_output(json.dumps(visual(**change)))


def test_unknown_visual_predicate_is_preserved_for_later_graph_rejection():
    relation = {"subject":"ginger","predicate":"unknown_predicate","object":"bowl","raw_extra":"retained"}
    parsed = parse_visual_output(json.dumps(visual(objects=["ginger","bowl"],relations=[relation])))
    assert parsed["relations"] == [relation]


@pytest.mark.parametrize("selected",[[],["obs:1"],["obs:1","obs:2"]])
def test_evidence_selector_accepts_only_supplied_unique_ids(selected):
    raw = json.dumps({"selected_observation_ids":selected,"reason":"Model candidates may be relevant."})
    assert parse_evidence_selection(raw,{"obs:1","obs:2"})["selected_observation_ids"] == selected


@pytest.mark.parametrize("value",[
    {"selected_observation_ids":["unknown"],"reason":"x"},
    {"selected_observation_ids":["obs:1","obs:1"],"reason":"x"},
    {"selected_observation_ids":[True],"reason":"x"},
    {"selected_observation_ids":None,"reason":"x"},
    {"selected_observation_ids":[],"reason":" "},
    {"selected_observation_ids":[],"reason":False},
])
def test_selector_invalid_identifiers_or_missing_explanation_are_not_accepted(value):
    with pytest.raises(ModelOutputError): parse_evidence_selection(json.dumps(value),{"obs:1"})


@pytest.mark.parametrize("raw",[
    '{"selected_observation_ids":[],"reason":"x","extra":NaN}',
    '{"selected_observation_ids":[],"reason":"x","extra":{"score":1e999}}',
    '{"selected_observation_ids":[],"selected_observation_ids":["obs:1"],"reason":"x"}',
    'prefix {"selected_observation_ids":[],"reason":"x"}',
    '[]',
])
def test_selector_shares_strict_json_boundary(raw):
    with pytest.raises(ModelOutputError): parse_evidence_selection(raw,{"obs:1"})


@pytest.mark.parametrize("raw",[None,42,{},b'{}'])
def test_nontext_model_outputs_raise_the_interface_error(raw):
    with pytest.raises(ModelOutputError): parse_visual_output(raw)


@pytest.mark.parametrize("caption", [None, "", " \n"])
def test_only_missing_caption_has_the_special_valid_structure_boundary(caption):
    raw = json.dumps(visual(caption=caption,objects=["ginger"]))
    with pytest.raises(MissingCaptionError) as error:
        parse_visual_output(raw)
    assert error.value.output["objects"] == [{"name":"ginger","bbox":None,"bbox_status":"not_returned"}]


def test_absent_caption_has_the_same_valid_structure_boundary():
    payload = visual()
    del payload["caption"]
    with pytest.raises(MissingCaptionError): parse_visual_output(json.dumps(payload))


@pytest.mark.parametrize("changes", [
    {"objects":None}, {"objects":[1]}, {"actions":[None]}, {"relations":[{}]}, {"caption":42},
])
def test_invalid_structure_or_wrong_caption_type_is_not_the_completion_boundary(changes):
    with pytest.raises(ModelOutputError) as error:
        parse_visual_output(json.dumps(visual(**{"caption":None,**changes})))
    assert not isinstance(error.value,MissingCaptionError)


@pytest.mark.parametrize("raw", [None, "", " \n\t", 42])
def test_independent_caption_must_be_actual_nonempty_model_text(raw):
    with pytest.raises(ModelOutputError): parse_short_caption_output(raw)


def test_short_caption_keeps_the_actual_words_without_object_based_assembly():
    assert parse_short_caption_output("  A spoon rests beside sliced ginger.\n") == "A spoon rests beside sliced ginger."


@pytest.fixture
def observer(tmp_path,monkeypatch):
    """Exercise real observe control flow with offline generation/decoder stubs.

    There is no model inference. The image stub writes deterministic bytes only
    to check that structured and caption calls receive the exact same inputs.
    """
    state = SimpleNamespace(responses=[],calls=[],prompts=[],cache_clears=0)
    model = LocalMultimodalModels.__new__(LocalMultimodalModels)
    model.root = tmp_path
    model.vlm = SimpleNamespace(config={"offline_fixture":True})
    model.processor = object()
    model.image_side,model.max_tokens = 448,512
    model.vision_info = {"repo":"offline-vlm","revision":"b"*40}
    model.embedding_info = {"repo":"offline-clip","revision":"c"*40}
    model.calls = {"vision":0,"embedding_image":0,"embedding_text":0}
    state.model = model

    class Image:
        def __init__(self,path): self.path = Path(path)
        def __enter__(self): return self
        def __exit__(self,*_): return None
        def convert(self,_): return self
        def thumbnail(self,_): return None
        def save(self,target,**_): Path(target).write_bytes(b"offline-decoder-stub:"+self.path.read_bytes())
    monkeypatch.setitem(sys.modules,"PIL",SimpleNamespace(Image=SimpleNamespace(open=Image)))

    def generate(vlm,processor,prompt,**kwargs):
        state.calls.append({"prompt":prompt,**copy.deepcopy(kwargs)})
        response = state.responses.pop(0)
        if isinstance(response,Exception): raise response
        return SimpleNamespace(text=response)
    def apply_template(processor,config,instruction,**kwargs):
        state.prompts.append({"instruction":instruction,**kwargs})
        return instruction
    def clear_cache(): state.cache_clears += 1
    core = SimpleNamespace(clear_cache=clear_cache)
    monkeypatch.setitem(sys.modules,"mlx_vlm",SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules,"mlx_vlm.prompt_utils",SimpleNamespace(apply_chat_template=apply_template))
    monkeypatch.setitem(sys.modules,"mlx",SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules,"mlx.core",core)
    state.paths = []
    for index in range(3):
        path = tmp_path / f"offline-frame-{index}.png"
        path.write_bytes(f"synthetic input {index}".encode())
        state.paths.append(path)
    return state


@pytest.mark.parametrize("timestamps", [None,[1.25,1.50,1.75]])
def test_same_single_image_or_video_frames_generate_a_separate_evidenced_caption(observer,timestamps):
    original = json.dumps({"objects":["ginger","bowl"],"actions":[],"relations":[],"extra":"retained"})
    caption_raw = "  Sliced ginger rests in a bowl.\n"
    observer.responses = [original,caption_raw]
    paths = observer.paths[:1] if timestamps is None else observer.paths
    record = observer.model.observe(paths,timestamps=timestamps)
    assert record["raw_text"] == original and record["attempts"] == 1
    assert record["output"]["caption"] == caption_raw.strip()
    assert record["output"]["extra"] == "retained"
    completion = record["output"]["caption_completion"]
    assert completion["raw_text"] == caption_raw and completion["structured_raw_text"] == original
    assert completion["status"] == "generated" and completion["timestamps"] == timestamps
    assert completion["input_sha256"] == record["input_sha256"]
    assert completion["input_paths"] == record["input_paths"]
    assert completion["input_sha256"] == [sha_file(observer.model.root / p) for p in completion["input_paths"]]
    assert completion["model"]["name"] == "offline-vlm" and completion["model"]["revision"] == "b"*40
    assert completion["model"]["max_tokens"] == 96
    assert completion["generation"] == {"max_tokens":96,"temperature":0.0,"repetition_penalty":1.1,"repetition_context_size":128}
    assert completion["inference_seconds"] >= 0
    assert observer.calls[0]["image"] == observer.calls[1]["image"]
    assert [call["max_tokens"] for call in observer.calls] == [512,96]
    assert observer.calls[1]["repetition_penalty"] == 1.1
    assert all(prompt["num_images"] == len(paths) for prompt in observer.prompts)
    # The caption prompt is independent of object nouns in structured output.
    assert "ginger" not in observer.calls[1]["prompt"] and "bowl" not in observer.calls[1]["prompt"]
    assert "still image" in observer.calls[1]["prompt"] if timestamps is None else str(timestamps) in observer.calls[1]["prompt"]
    assert observer.model.calls["vision"] == 2 and observer.cache_clears == 1


def test_complete_structured_caption_does_not_trigger_an_extra_call(observer):
    original = json.dumps(visual())
    observer.responses = [original]
    record = observer.model.observe(observer.paths[:1])
    assert record["raw_text"] == original and "caption_completion" not in record["output"]
    assert observer.model.calls["vision"] == 1


@pytest.mark.parametrize("failure", ["", " \n", RuntimeError("offline synthetic generation error")])
@pytest.mark.parametrize("timestamps", [None,[1.25,1.50,1.75]])
def test_empty_or_failed_caption_is_preserved_then_retried_with_new_structure(observer,failure,timestamps):
    missing = json.dumps({"objects":["ginger"],"actions":[],"relations":[]})
    complete = json.dumps(visual(caption="A real second-call fixture description."))
    observer.responses = [missing,failure,complete]
    paths = observer.paths[:1] if timestamps is None else observer.paths
    record = observer.model.observe(paths,timestamps=timestamps)
    assert record["raw_text"] == complete and record["attempts"] == 2
    first = record["prior_invalid_outputs"][0]
    assert first["raw_text"] == missing
    completion = first["caption_completion"]
    assert completion["status"] == "failed" and completion["timestamps"] == timestamps
    assert completion["raw_text"] == (None if isinstance(failure,Exception) else failure)
    assert completion["structured_raw_text"] == missing
    assert completion["input_sha256"] == record["input_sha256"]
    assert "error_type" in completion and "error" in completion
    assert "caption_completion" not in record["output"]
    assert observer.model.calls["vision"] == 3


def test_two_failed_caption_calls_never_return_a_successful_observation(observer):
    missing = json.dumps({"objects":[],"actions":[],"relations":[]})
    observer.responses = [missing,"",missing,RuntimeError("offline synthetic caption failure")]
    with pytest.raises(ModelOutputError) as error:
        observer.model.observe(observer.paths[:1])
    evidence = json.loads(str(error.value))
    assert len(evidence["outputs"]) == 2
    assert all(item["caption_completion"]["status"] == "failed" for item in evidence["outputs"])
    assert all(item["raw_text"] == missing for item in evidence["outputs"])
    assert observer.model.calls["vision"] == 4


@pytest.mark.parametrize("invalid", [
    '{"objects":[],"actions":[],"relations":[],"unknown":NaN}',
    '{"objects":[],"actions":[],"relations":[],"unknown":{"score":1e999}}',
    '{"objects":[],"actions":[],"relations":[],"objects":[]}',
    '{"objects":[],"actions":[false],"relations":[]}',
    '{"objects":[],"actions":[],"relations":[{}]}',
    '{"objects":[],"actions":[],"relations":[]',
    '{"objects":[],"actions":[],"relations":[],"caption":42}',
])
def test_invalid_structured_json_never_uses_caption_completion(observer,invalid):
    observer.responses = [invalid,invalid]
    with pytest.raises(ModelOutputError): observer.model.observe(observer.paths[:1])
    assert observer.model.calls["vision"] == 2
    assert [call["max_tokens"] for call in observer.calls] == [512,512]
    assert not any("sentence only" in call["prompt"] for call in observer.calls)
