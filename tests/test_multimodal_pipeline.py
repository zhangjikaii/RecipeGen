"""Real controller flow with isolated synthetic model/media/DB dependencies.

These tests never load models, read project-private config, download media or
connect to Neo4j. Algorithm/graph contracts have their own independent tests.
"""
from __future__ import annotations

import copy
import gc
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multimodal_controller_contract",PROJECT / "scripts/run_multimodal_pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def test_checkpoint_compression_preserves_exact_bytes_and_all_raw_fields(tmp_path):
    original = {}
    for name in ("prepared.json","observations.json","alignment.json","bundle.json"):
        value = {"name":name,"raw_text":"{ 原始模型文本 }\n","embedding":[0.125]*512,"extra":{"preserved":True}}
        PIPELINE.write_json(tmp_path / name,value)
        original[name] = (tmp_path / name).read_bytes()
    PIPELINE.compress_checkpoints(tmp_path)
    for name,raw in original.items():
        assert not (tmp_path / name).exists()
        assert gzip.decompress((tmp_path / (name+".gz")).read_bytes()) == raw
        assert PIPELINE.read_checkpoint(tmp_path / name) == json.loads(raw)
    PIPELINE.compress_checkpoints(tmp_path)  # Already compressed is harmless.


def test_new_plain_checkpoint_takes_precedence_over_old_compressed_checkpoint(tmp_path):
    path = tmp_path / "observations.json"
    PIPELINE.write_json(path,[{"id":"old"}])
    PIPELINE.compress_checkpoints(tmp_path)
    PIPELINE.write_json(path,[{"id":"new"}])
    assert PIPELINE.read_checkpoint(path) == [{"id":"new"}]
    PIPELINE.compress_checkpoints(tmp_path)
    assert PIPELINE.read_checkpoint(path) == [{"id":"new"}]


def test_failed_compressed_replacement_keeps_the_original_checkpoint(tmp_path,monkeypatch):
    path = tmp_path / "bundle.json"
    PIPELINE.write_json(path,{"unaltered":"evidence"})
    original = path.read_bytes()
    real_replace = Path.replace
    def fail_part_replace(source,target):
        if str(source).endswith(".gz.part"):
            raise OSError("synthetic disk write failure")
        return real_replace(source,target)
    monkeypatch.setattr(Path,"replace",fail_part_replace)
    with pytest.raises(OSError): PIPELINE.compress_checkpoints(tmp_path)
    assert path.read_bytes() == original
    assert PIPELINE.read_checkpoint(path) == {"unaltered":"evidence"}


def test_corrupted_gzip_is_an_error_instead_of_empty_evidence(tmp_path):
    path = tmp_path / "observations.json"
    path.with_suffix(".json.gz").write_bytes(b"broken gzip")
    with pytest.raises((gzip.BadGzipFile,OSError,EOFError)):
        PIPELINE.read_checkpoint(path)


@pytest.fixture
def harness(tmp_path,monkeypatch):
    root = tmp_path / "project"
    (root / ".runtime").mkdir(parents=True)
    (root / ".runtime/multimodal-models.json").write_text(json.dumps({
        "vision":{"repo":"synthetic-vlm","revision":"b"*40},
        "embedding":{"repo":"synthetic-clip","revision":"c"*40}}))
    h = SimpleNamespace(root=root,prepared_calls=0,prepared_inputs=[],cleanup_calls=[],alive={},models=[],vision_errors=[],
                        import_results=[],import_payloads=[],build_calls=0,validated_payloads=[],
                        cleanup_error=None,graph_uri="bolt://synthetic.invalid:7687",free_bytes=10*1024**3,
                        steps=[],text_vector_limit=None,
                        records=[{"id":"recipe:fixture","media":[{"kind":"image","media_id":"image:fixture"}]}])
    monkeypatch.setattr(PIPELINE,"ROOT",root)
    monkeypatch.setattr(PIPELINE,"__file__",str(root / "scripts/run_multimodal_pipeline.py"))
    monkeypatch.setattr(PIPELINE,"sha_file",lambda path:"synthetic-code-signature")
    monkeypatch.setattr(PIPELINE.shutil,"disk_usage",lambda path:SimpleNamespace(free=h.free_bytes))
    monkeypatch.setattr(PIPELINE,"read_config",lambda:SimpleNamespace(uri=h.graph_uri,database="neo4j",user="fixture",password="synthetic-only"))
    base = SimpleNamespace(build_id="fixture-build",dataset_revision="a"*40,content_sha256="d"*64,
                           nodes_by_id={"recipe:fixture":{"id":"recipe:fixture","label":"Recipe","properties":{}},
                                        "image:fixture":{"id":"image:fixture","label":"Image","properties":{}}})
    monkeypatch.setattr(PIPELINE,"load_base_index",lambda _:base)
    monkeypatch.setattr(PIPELINE,"select_records",lambda *_,**__:iter(copy.deepcopy(h.records)))
    # The pure scope checker is exercised against real ID contracts in its own
    # tests. This boundary stub isolates failure/resume bookkeeping from graph IO.
    monkeypatch.setattr(PIPELINE,"verify_full_scope",lambda records,orphans,base:{
        "verified":True,"base_content_sha256":base.content_sha256,
        "counts":{"Recipe":1,"Image":1,"Video":0},"orphan_media":0,
        "id_set_sha256":{"Recipe":"synthetic-recipe-set","Image":"synthetic-image-set","Video":"synthetic-video-set"}})

    class Preparer:
        def __init__(self,*_,**__): pass
        def iter_orphan_media(self): return []
        def prepare_record(self,record):
            h.prepared_calls += 1
            h.prepared_inputs.append(copy.deepcopy(record))
            prepared = {"recipe_id":record["id"],"steps":copy.deepcopy(h.steps),"status":"complete","errors":[],"storage_state":"local",
                        "images":[{"media_id":"image:fixture","source_id":"source:fixture","path":"fixture.jpg",
                                   "archive":"test.zip","member":"fixture.jpg","revision":"a"*40,"sha256":"1"*64,"zip_crc_verified":True}],
                        "videos":[],"counts":{"images":1,"videos":0,"clips":0,"frames":0}}
            h.alive[record["id"]] = copy.deepcopy(prepared)
            return prepared
        def cleanup_record(self,identifier):
            h.cleanup_calls.append(identifier)
            if h.cleanup_error:
                raise h.cleanup_error
            # Reproduce the real preparer's missing-workspace fallback: metadata
            # is gone once cleanup completed, so repeated cleanup must not erase
            # the controller's preserved complete preparation checkpoint.
            result = h.alive.pop(identifier,{"recipe_id":identifier,"images":[],"videos":[],"status":"cleanup_without_checkpoint"})
            result["storage_state"] = "cleaned"
            return result
        def close(self): pass
    monkeypatch.setattr(PIPELINE,"MediaPreparer",Preparer)

    class Models:
        def __init__(self,*_):
            self.calls = {"vision":0,"embedding_image":0,"embedding_text":0}
            self.embedding_space = {"model":"synthetic-clip","revision":"c"*40,"dimension":512}
            self.model_info = {"name":"synthetic-vlm","revision":"b"*40}
            self.text_inputs = []
            h.models.append(self)
        def observe(self,paths,**_):
            self.calls["vision"] += 1
            if h.vision_errors:
                raise h.vision_errors.pop(0)
            return {"output":{"caption":"synthetic controller fixture","objects":[],"actions":[],"relations":[]},
                    "raw_text":"synthetic test evidence, no inference", "model":self.model_info,
                    "input_sha256":["2"*64],"input_paths":["fixture.vlm.jpg"],"inference_seconds":0.01,
                    "attempts":1,"prior_invalid_outputs":[]}
        def image_embeddings(self,paths):
            self.calls["embedding_image"] += len(paths)
            return [[1.0]+[0.0]*511 for _ in paths]
        def text_embeddings(self,texts):
            self.text_inputs.append(texts)
            self.calls["embedding_text"] += len(texts)
            vectors = [[1.0]+[0.0]*511 for _ in texts]
            return vectors if h.text_vector_limit is None else vectors[:h.text_vector_limit]
    monkeypatch.setattr(PIPELINE,"LocalMultimodalModels",Models)

    def build(prepared,observations,alignment,metadata,run_id,model_info):
        h.build_calls += 1
        return {"run_id":run_id,"immutable_fixture_payload":copy.deepcopy(observations),"counts":prepared["counts"]}
    def validate(bundle,base):
        h.validated_payloads.append(copy.deepcopy(bundle))
        return SimpleNamespace(nodes=[{"fixture":True}],relationships=[],summary=lambda:{"synthetic_fixture":True},payload=bundle)
    def import_bundle(session,bundle):
        h.import_payloads.append(copy.deepcopy(bundle.payload))
        value = h.import_results.pop(0) if h.import_results else {"status":"verified","synthetic_fixture":True}
        if isinstance(value,Exception): raise value
        return value
    monkeypatch.setattr(PIPELINE,"build_multimodal_bundle",build)
    monkeypatch.setattr(PIPELINE,"validate_bundle",validate)
    monkeypatch.setattr(PIPELINE,"import_bundle",import_bundle)
    monkeypatch.setattr(PIPELINE,"ensure_vector_index",lambda session:None)
    session = SimpleNamespace(close=lambda:None)
    driver = SimpleNamespace(verify_connectivity=lambda:None,session=lambda **_:session,close=lambda:None)
    monkeypatch.setitem(sys.modules,"neo4j",SimpleNamespace(GraphDatabase=SimpleNamespace(driver=lambda *_,**__:driver)))

    def state():
        return json.loads(next((root / "data/multimodal/runs").glob("*/state.json")).read_text())
    h.state = state
    h.progress = lambda:json.loads((root / "reports/multimodal-progress.json").read_text())
    h.folder = lambda:next((root / "data/multimodal/runs").glob("*/")) / PIPELINE.digest("recipe:fixture")[:24]
    return h


def test_image_only_excludes_video_before_media_preparation_and_keeps_text(harness):
    harness.records[0]["media"].append({"kind":"video", "media_id":"video:forbidden"})
    harness.records[0]["title"] = "original title"
    harness.steps = [{"id":"step:fixture", "text":"original recipe step", "order":1}]
    assert PIPELINE.main(["--modality","images","--import-neo4j","--max-attempts","1"]) == 0
    assert harness.prepared_inputs[0]["media"] == [{"kind":"image","media_id":"image:fixture"}]
    assert harness.prepared_inputs[0]["title"] == "original title"
    assert harness.models[0].text_inputs == [["original recipe step"]]
    p = harness.progress()
    assert p["scope"] == "full_test_images" and p["processing_modalities"] == ["image"]
    assert p["expected_images"] == 1 and p["expected_videos"] == 0
    assert p["processed_counts"]["videos"] == p["processed_counts"]["clips"] == p["processed_counts"]["frames"] == 0
    assert p["completed_image_scope"] is True and p["completed_full_scope"] is False
    assert harness.state()["signature"]["processing_modalities"] == ["image"]


def test_image_only_rejects_preparer_video_without_any_vision_inference(harness,monkeypatch):
    original = PIPELINE.MediaPreparer.prepare_record
    def invalid(preparer, record):
        row = original(preparer, record)
        row["videos"] = [{"media_id":"video:unexpected", "clips":[]}]
        return row
    monkeypatch.setattr(PIPELINE.MediaPreparer,"prepare_record",invalid)
    assert PIPELINE.main(["--modality","images","--import-neo4j","--max-attempts","1"]) == 1
    assert harness.models[0].calls["vision"] == 0 and not harness.import_payloads
    assert harness.progress()["completed_image_scope"] is False


def test_image_only_excludes_orphan_video_from_work_queue(harness,monkeypatch):
    monkeypatch.setattr(PIPELINE.MediaPreparer,"iter_orphan_media",lambda _: [{"kind":"video","media_id":"video:orphan"}])
    assert PIPELINE.main(["--modality","images","--import-neo4j","--max-attempts","1"]) == 0
    assert harness.progress()["expected_orphan_media"] == 0
    assert harness.progress()["expected_videos"] == 0
    assert harness.progress()["total_work_items"] == 1


def test_image_only_offline_processing_does_not_claim_native_completion(harness):
    assert PIPELINE.main(["--modality","images","--max-attempts","1"]) == 0
    assert harness.progress()["completed_image_scope"] is False
    assert harness.progress()["completed_full_scope"] is False


def test_no_import_processing_is_never_reported_as_full_native_verification(harness):
    assert PIPELINE.main(["--max-attempts","1"]) == 0
    record = harness.state()["records"]["recipe:fixture"]
    assert record["status"] == "processed"
    report = harness.progress()
    assert report["completed_work_items"] == 1 and report["failed_work_items"] == 0
    assert report["completed_full_scope"] is False and not report["neo4j_import_requested"]
    assert harness.models[0].text_inputs == [[]]  # Empty original steps are supported.


def test_unverified_import_is_counted_failed_instead_of_verified(harness):
    harness.import_results = [{"status":"unverified"}]
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 1
    assert harness.state()["records"]["recipe:fixture"]["status"] == "failed"
    report = harness.progress()
    assert report["failed_work_items"] == 1 and report["completed_work_items"] == 0
    assert report["completed_full_scope"] is False


def test_inference_failure_does_not_create_success_or_a_bundle(harness):
    harness.vision_errors = [ValueError("synthetic invalid model output")]
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 1
    assert harness.progress()["failed_work_items"] == 1
    assert harness.progress()["completed_work_items"] == 0 and harness.build_calls == 0
    assert harness.cleanup_calls == ["recipe:fixture"]


def test_cleanup_failure_persists_failed_state_and_blocks_completion(harness):
    harness.cleanup_error = OSError("synthetic cleanup failure")
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 1
    record = harness.state()["records"]["recipe:fixture"]
    assert record["status"] == "failed" and "清理失败" in record["error"]
    assert harness.progress()["failed_work_items"] == 1
    assert harness.progress()["completed_full_scope"] is False


def test_compression_failure_keeps_raw_evidence_and_records_failure(harness,monkeypatch):
    def fail(_): raise OSError("synthetic ENOSPC")
    monkeypatch.setattr(PIPELINE,"compress_checkpoints",fail)
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 1
    assert harness.state()["records"]["recipe:fixture"]["status"] == "failed"
    assert (harness.folder() / "observations.json").is_file()
    assert (harness.folder() / "bundle.json").is_file()
    assert harness.progress()["completed_full_scope"] is False


def test_import_failure_then_resume_reuses_immutable_bundle_without_inference(harness):
    harness.import_results = [RuntimeError("synthetic transient native import failure"),{"status":"verified"}]
    options = ["--import-neo4j","--max-attempts","1"]
    assert PIPELINE.main(options) == 1
    gc.collect()  # A CLI restart releases the previous process's lock descriptor.
    assert PIPELINE.main(options) == 0
    assert harness.prepared_calls == 1 and harness.build_calls == 1
    assert harness.import_payloads[0] == harness.import_payloads[1]
    assert harness.models[1].calls == {"vision":0,"embedding_image":0,"embedding_text":0}
    assert harness.state()["records"]["recipe:fixture"]["status"] == "verified"
    assert harness.progress()["completed_full_scope"] is True


def test_cached_bundle_retry_does_not_erase_preserved_prepared_counts(harness):
    harness.import_results = [RuntimeError("synthetic transient failure"),{"status":"verified"}]
    options = ["--import-neo4j","--max-attempts","1"]
    assert PIPELINE.main(options) == 1
    gc.collect()
    assert PIPELINE.main(options) == 0
    prepared = PIPELINE.read_checkpoint(harness.folder() / "prepared.json")
    assert prepared["counts"] == {"images":1,"videos":0,"clips":0,"frames":0}
    assert len(prepared["images"]) == 1


def test_keep_media_signature_cannot_be_reused_as_a_cleaning_run(harness):
    assert PIPELINE.main(["--limit","1","--keep-media","--run-id","fixed-signature","--max-attempts","1"]) == 0
    with pytest.raises(ValueError,match="Same pipeline_id"):
        PIPELINE.main(["--limit","1","--run-id","fixed-signature","--max-attempts","1"])


def test_different_graph_target_requires_a_different_pipeline_signature(harness):
    assert PIPELINE.main(["--import-neo4j","--run-id","fixed-signature","--max-attempts","1"]) == 0
    harness.graph_uri = "bolt://another-synthetic.invalid:7687"
    with pytest.raises(ValueError,match="Same pipeline_id"):
        PIPELINE.main(["--import-neo4j","--run-id","fixed-signature","--max-attempts","1"])


def test_low_disk_stops_without_counting_unprocessed_items_as_success(harness):
    harness.free_bytes = 1024
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 10
    report = harness.progress()
    assert report["status"] == "stopped_low_disk"
    assert report["completed_work_items"] == 0 and report["completed_full_scope"] is False
    assert harness.prepared_calls == 0


def test_full_keep_media_is_rejected_before_model_setup(harness):
    with pytest.raises(SystemExit) as error:
        PIPELINE.main(["--keep-media"])
    assert error.value.code == 2 and harness.models == []


def test_full_scope_rejection_prevents_model_setup_and_completion_report(harness,monkeypatch):
    def reject(*_): raise ValueError("synthetic missing original Video ID")
    monkeypatch.setattr(PIPELINE,"verify_full_scope",reject)
    with pytest.raises(ValueError,match="missing original Video"):
        PIPELINE.main(["--import-neo4j","--max-attempts","1"])
    assert harness.models == []
    assert not (harness.root / "reports/multimodal-progress.json").exists()


def test_short_text_vector_result_is_failed_instead_of_truncating_original_steps(harness):
    harness.steps = [{"id":"step:fixture","order":1,"text":"原始步骤","source_id":"source:fixture"}]
    harness.text_vector_limit = 0
    assert PIPELINE.main(["--import-neo4j","--max-attempts","1"]) == 1
    state = harness.state()["records"]["recipe:fixture"]
    assert state["status"] == "failed" and "文本向量数量不一致" in state["error"]
    assert harness.build_calls == 0 and harness.import_payloads == []


def test_failed_workspaces_are_cleaned_before_low_disk_resume_check(harness):
    harness.vision_errors = [ValueError("synthetic invalid output")]
    options = ["--import-neo4j","--max-attempts","1"]
    assert PIPELINE.main(options) == 1
    gc.collect()
    harness.free_bytes = 1024
    assert PIPELINE.main(options) == 10
    assert harness.cleanup_calls == ["recipe:fixture","recipe:fixture"]
    assert harness.prepared_calls == 1
    assert harness.progress()["completed_full_scope"] is False
