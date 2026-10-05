"""Pure assembly of prepared media and supplied model artifacts into a graph.

No files, models, APIs or databases are accessed. Existing graph entities are
references; generated visual/semantic content remains explicitly unverified.
Caption mentions cannot create objects, actions or spatial relationships.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping

SCHEMA_VERSION = "recipegen-mm-v1"
PREDICATES = {"on":"ON", "on top of":"ON", "inside":"IN", "in":"IN", "beside":"BESIDE",
              "holding":"HOLDING", "using":"USING", "mixed with":"MIXED_WITH"}


class MultimodalBundleError(ValueError):
    """Supplied artifacts are inconsistent or cannot retain their provenance."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise MultimodalBundleError("Artifacts must be finite JSON-compatible values") from error


def _id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(c) < 32 for c in value):
        raise MultimodalBundleError(f"{where} must be a nonempty identifier without whitespace/control characters")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultimodalBundleError(f"{where} must be nonempty text")
    return value


def _number(value: Any, where: str, *, nonnegative: bool = True) -> float:
    if type(value) not in (int, float):
        raise MultimodalBundleError(f"{where} must be finite numeric data")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise MultimodalBundleError(f"{where} must be finite numeric data") from error
    if not math.isfinite(result) or nonnegative and result < 0:
        raise MultimodalBundleError(f"{where} must be finite {'nonnegative ' if nonnegative else ''}numeric data")
    return result


def _normalize(value: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in unicodedata.normalize("NFKC", value).casefold()).split())


def _rows(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise MultimodalBundleError(f"{where} must be a list of objects")
    return value


def build_multimodal_bundle(prepared: Mapping[str, Any], observations: list[dict[str, Any]],
                            alignment: Mapping[str, Any], base_metadata: Mapping[str, Any],
                            run_id: str, model_info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict ``recipegen-mm-v1`` node/relationship JSON contract.

    Prepared clip seconds are relative to the first decoded video frame. Frame
    ``timestamp_seconds`` uses offset_sec, retaining original PTS time separately.
    Only the selected step alignment becomes an edge; ranked alternatives and
    caption-only dictionary candidates remain in observation evidence JSON.
    """
    for name, value in (("prepared",prepared), ("alignment",alignment), ("base_metadata",base_metadata), ("model_info",model_info)):
        if not isinstance(value, Mapping):
            raise MultimodalBundleError(f"{name} must be an object")
    _json([prepared, observations, alignment, base_metadata, model_info])
    prepared, observations, alignment, metadata, models = copy.deepcopy((dict(prepared), observations, dict(alignment), dict(base_metadata), dict(model_info)))
    run_id = _id(run_id, "run_id")
    build_id = _id(metadata.get("build_id"), "base_metadata.build_id")
    revision = _id(metadata.get("dataset_revision"), "base_metadata.dataset_revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise MultimodalBundleError("dataset_revision must be a pinned 40-character commit")
    if prepared.get("split", "test") != "test" or prepared.get("revision", revision) != revision:
        raise MultimodalBundleError("prepared media must belong to the same test revision")
    scope = {"build_id":build_id, "dataset_revision":revision, "split":"test", "run_id":run_id}
    space = alignment.get("embedding_space", models.get("embedding_space", {}))
    if observations and (not isinstance(space, dict) or space.get("dimension", 512) != 512):
        raise MultimodalBundleError("Observation vectors must use the declared 512-dimensional CLIP space")
    embedding_model = _id(space.get("model"), "embedding_space.model") if observations else None
    embedding_revision = _id(space.get("revision"), "embedding_space.revision") if observations else None
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def identifier(kind: str, *parts: Any) -> str:
        digest = hashlib.sha256(_json([run_id, *parts]).encode()).hexdigest()
        return f"mm:{kind}:{digest}"

    def node(nid: str, label: str, properties: dict[str, Any]) -> None:
        value = {"id":nid, "label":label, "properties":properties}
        if nid in nodes and nodes[nid] != value:
            raise MultimodalBundleError(f"Conflicting node artifacts for {nid}")
        nodes[nid] = value

    def reference(nid: str, label: str) -> None:
        node(_id(nid, "reference.id"), label, {"reference":True})

    def properties(source: dict[str, Any], *, model: str, semantics: str,
                   detail: dict[str, Any] | None = None, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence = {key:value for key,value in source.items() if key not in {"clips", "frames"}}
        if observation is not None:
            provided = observation.get("evidence")
            if not isinstance(provided, dict) or not provided:
                raise MultimodalBundleError("Observation evidence must preserve actual input provenance")
            for key, expected in (("source_media_id",source["media_id"]), ("source_id",source["source_id"])):
                if key in provided and provided[key] != expected:
                    raise MultimodalBundleError(f"Observation evidence contradicts {key}")
            if "media_sha256" in provided and provided["media_sha256"] != source["sha256"]:
                raise MultimodalBundleError("Observation media hash differs from prepared media")
            evidence.update(provided)
            evidence.update(observation_id=observation["id"], model_metadata=observation["model"])
            if "embedding_space" in observation:
                evidence["embedding_space"] = observation["embedding_space"]
        evidence.update(scope, source_media_id=source["media_id"], source_id=source["source_id"])
        if detail:
            evidence.update(detail)
        return {**scope, "source_media_id":source["media_id"], "source_id":source["source_id"],
                "evidence_json":_json(evidence), "verified":False, "semantics":semantics,
                "model":model, "confidence":None}

    def edge(start: str, end: str, kind: str, props: dict[str, Any], *discriminator: Any) -> None:
        if start == end:
            raise MultimodalBundleError("Candidate relationships must not be self-loops")
        eid = identifier("edge", start, kind, end, *discriminator)
        value = {"id":eid, "start_id":start, "end_id":end, "type":kind, "properties":props}
        if eid in edges and edges[eid] != value:
            raise MultimodalBundleError("Conflicting relationship evidence")
        edges[eid] = value

    images, videos, clips = {}, {}, {}
    for label, entries, target in (("Image",prepared.get("images", []),images), ("Video",prepared.get("videos", []),videos)):
        for source in _rows(entries, label):
            mid = _id(source.get("media_id"), f"{label}.media_id")
            _id(source.get("source_id"), f"{label}.source_id")
            if source.get("split", "test") != "test" or source.get("revision", revision) != revision:
                raise MultimodalBundleError("Mixed media splits or dataset revisions")
            if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
                raise MultimodalBundleError("Prepared media must have a verified-input SHA256 value")
            if mid in images or mid in videos:
                raise MultimodalBundleError("Prepared media IDs must be unique")
            target[mid] = source
            reference(mid, label)
    structure_model = _id(models.get("structure_model", "media-preparation"), "structure_model")
    for video_id, video in videos.items():
        frame_nodes, clip_nodes = {}, []
        origin = _number(video.get("timeline_origin_sec", 0), "timeline_origin_sec", nonnegative=False)
        for clip in _rows(video.get("clips", []), "video.clips"):
            cid = _id(clip.get("clip_id"), "clip.clip_id")
            if cid in clips:
                raise MultimodalBundleError("Clip IDs must be globally unique")
            if clip.get("media_id", video_id) != video_id:
                raise MultimodalBundleError("Clip belongs to a different source video")
            start = _number(clip.get("start_sec"), "clip.start_sec")
            end = _number(clip.get("end_sec"), "clip.end_sec")
            if end <= start:
                raise MultimodalBundleError("Clip end must be later than its start")
            nid = identifier("clip", cid)
            clip_detail = {"clip":{key:value for key,value in clip.items() if key != "frames"}}
            props = properties(video, model=structure_model, semantics="media_structure", detail=clip_detail)
            node(nid, "VideoClip", {**props, "clip_id":cid, "start_seconds":start, "end_seconds":end,
                                    "timeline_origin_seconds":origin, "time_reference":"relative_to_first_decoded_frame"})
            clips[cid] = {"node_id":nid, "source":video, "prepared":clip}
            clip_nodes.append((start,end,nid,clip))
            edge(video_id,nid,"HAS_CLIP",props)
            for frame in _rows(clip.get("frames", []), "clip.frames"):
                fid = _id(frame.get("frame_id"), "frame.frame_id")
                raw_time = _number(frame.get("timestamp_sec"), "frame.timestamp_sec", nonnegative=False)
                moment = _number(frame.get("offset_sec", raw_time-origin), "frame.offset_sec")
                if not start <= moment <= end:
                    raise MultimodalBundleError("Frame offset is outside its real clip interval")
                if not re.fullmatch(r"[0-9a-f]{64}", str(frame.get("sha256", ""))):
                    raise MultimodalBundleError("Prepared frames must retain their image SHA256")
                fnid = identifier("frame", video_id, fid)
                fp = properties(video, model=structure_model, semantics="media_structure", detail={"frame":frame})
                optional = {key:frame[key] for key in ("pts","time_base","width","height","path","sha256") if key in frame}
                node(fnid,"Frame",{**fp, **optional, "frame_id":fid, "timestamp_seconds":moment,
                                    "source_timestamp_seconds":raw_time, "timeline_origin_seconds":origin,
                                    "time_reference":"relative_to_first_decoded_frame"})
                frame_nodes[fnid] = (moment,frame)
                edge(nid,fnid,"HAS_FRAME",properties(video,model=structure_model,semantics="media_structure",detail={**clip_detail,"frame":frame}))
        ordered = sorted(clip_nodes)
        for left,right in zip(ordered,ordered[1:]):
            if left[1] <= right[0]:
                edge(left[2],right[2],"BEFORE",properties(video,model=structure_model,semantics="media_structure",
                    detail={"before_clip_id":left[3]["clip_id"],"after_clip_id":right[3]["clip_id"],"before_end_sec":left[1],"after_start_sec":right[0]}))
        groups: dict[float, list[tuple[str, dict[str, Any]]]] = {}
        for fid,(moment,frame) in frame_nodes.items():
            groups.setdefault(moment,[]).append((fid,frame))
        times = sorted(groups)
        for early,late in zip(times,times[1:]):
            for first,second in ((a,b) for a in groups[early] for b in groups[late]):
                edge(first[0],second[0],"BEFORE",properties(video,model=structure_model,semantics="media_structure",
                    detail={"before_frame":first[1],"after_frame":second[1],"before_offset_sec":early,"after_offset_sec":late}))
    step_ids = set()
    for row in _rows(prepared.get("steps", []), "prepared.steps"):
        sid = _id(row.get("id"), "step.id")
        if sid in step_ids:
            raise MultimodalBundleError("Prepared step IDs must be unique")
        step_ids.add(sid)
    dictionary = {}
    for entity in _rows(metadata.get("entities", []), "base_metadata.entities"):
        eid = _id(entity.get("id"), "entity.id")
        if eid in dictionary or entity.get("type") not in {"Ingredient","Tool","Action"}:
            raise MultimodalBundleError("Base entity IDs/types must be unique existing graph entities")
        dictionary[eid] = entity
    selected = {}
    for row in _rows(alignment.get("alignments", []), "alignment.alignments"):
        mid = _id(row.get("media_id"), "alignment.media_id")
        if mid in selected:
            raise MultimodalBundleError("There must be at most one main alignment per observation")
        selected[mid] = row
    candidates = _rows(alignment.get("semantic_entities", []), "alignment.semantic_entities")
    supplied_triples = _rows(alignment.get("visual_triples", []), "alignment.visual_triples")
    observation_ids = set()
    rejected_count = 0
    for observation in _rows(observations, "observations"):
        oid = _id(observation.get("id"), "observation.id")
        if oid in observation_ids:
            raise MultimodalBundleError("Observation IDs must be unique")
        observation_ids.add(oid)
        kind = observation.get("kind")
        source_mid = _id(observation.get("source_media_id"), "observation.source_media_id")
        if kind == "image" and source_mid in images:
            source, parent = images[source_mid], source_mid
        elif kind == "clip" and oid in clips:
            source, parent = clips[oid]["source"], clips[oid]["node_id"]
        else:
            raise MultimodalBundleError("Observation must refer to a prepared image or clip")
        if source_mid != source["media_id"] or observation.get("source_id") != source["source_id"]:
            raise MultimodalBundleError("Observation source IDs contradict prepared media")
        model = observation.get("model")
        if not isinstance(model, dict):
            raise MultimodalBundleError("Observation model metadata must be preserved")
        model_name = _id(model.get("name"), "observation.model.name")
        _id(model.get("revision"), "observation.model.revision")
        observation_space = observation.get("embedding_space", space)
        if not isinstance(observation_space,dict) or observation_space.get("dimension",512) != 512:
            raise MultimodalBundleError("Observation embedding_space must describe 512-dimensional CLIP output")
        embedding_model = _id(observation_space.get("model"),"observation.embedding_space.model")
        embedding_revision = _id(observation_space.get("revision"),"observation.embedding_space.revision")
        if any(observation_space.get(key) != space.get(key) for key in ("model","revision")):
            raise MultimodalBundleError("Observation embedding space differs from the alignment space")
        vector = observation.get("embedding")
        if not isinstance(vector,list) or len(vector) != 512:
            raise MultimodalBundleError("Observation must have a 512-dimensional CLIP embedding")
        vector = [_number(value,"observation.embedding",nonnegative=False) for value in vector]
        if not any(vector):
            raise MultimodalBundleError("Observation embedding cannot be zero")
        raw = observation.get("raw_output")
        if not isinstance(raw,dict) or not isinstance(raw.get("relations",[]),list):
            raise MultimodalBundleError("Observation raw_output must preserve a JSON object and relation list")
        caption = observation.get("caption", "")
        if not isinstance(caption,str):
            raise MultimodalBundleError("Observation caption must remain text")
        objects = _rows(observation.get("objects", []), "observation.objects")
        actions = observation.get("actions", [])
        if not isinstance(actions,list) or any(not isinstance(action,str) or not action.strip() for action in actions):
            raise MultimodalBundleError("Observation actions must be the original model action strings")
        onid = identifier("observation",oid)
        raw_alignment = {"main":selected.get(oid), "semantic_entities":[row for row in candidates if row.get("media_id") == oid],
                         "visual_triples":[row for row in supplied_triples if row.get("media_id") == oid]}
        op = properties(source,model=model_name,semantics="model_candidate",observation=observation)
        node(onid,"VisualObservation",{**op,"observation_id":oid,"caption":caption,"embedding":vector,
              "embedding_model":embedding_model,"embedding_revision":embedding_revision,"model_json":_json(model),
              "raw_json":_json(raw),"alignment_json":_json(raw_alignment),"rejections_json":"[]"})
        edge(parent,onid,"HAS_OBSERVATION",properties(source,model=model_name,semantics="media_structure",observation=observation))
        rejections = []
        visual_objects = []
        for index,obj in enumerate(objects):
            name = _text(obj.get("name"), "object.name")
            vnid = identifier("object",oid,index)
            bbox = obj.get("bbox")
            rejected_bbox = bool(obj.get("bbox_rejected", False))
            if bbox is not None and not rejected_bbox:
                if not isinstance(bbox,list) or len(bbox) != 4 or any(type(value) not in (int,float) or not math.isfinite(value) for value in bbox):
                    rejected_bbox = True
                else:
                    bbox = [float(value) for value in bbox]
            if rejected_bbox:
                bbox = None
            vp = properties(source,model=model_name,semantics="model_candidate",observation=observation,detail={"raw_object":obj,"object_index":index})
            node(vnid,"VisualObject",{**vp,"name":name,"observation_id":oid,"object_index":index,
                                      "predicted_bbox":bbox,"bbox_rejected":rejected_bbox,"raw_json":_json(obj)})
            visual_objects.append((vnid,obj,index))
            edge(parent,vnid,"DEPICTS" if kind == "image" else "CONTAINS_OBJECT",vp,index)
        main = selected.get(oid)
        if main and main.get("status") == "candidate":
            sid = main.get("step_id")
            if sid not in step_ids:
                raise MultimodalBundleError("Selected step is absent from prepared original steps")
            score = _number(main.get("score"),"step alignment score",nonnegative=False)
            if not -1 <= score <= 1:
                raise MultimodalBundleError("Step cosine score must be within [-1,1]")
            reference(sid,"Step")
            ap = properties(source,model=model_name,semantics="model_candidate",observation=observation,detail={"selected_alignment":main})
            edge(parent,sid,"ALIGNED_WITH",{**ap,"similarity_score":score,"alignment_method":"clip_cosine_temporal_candidate",
                                         "embedding_model":embedding_model,"embedding_revision":embedding_revision})
        for candidate in raw_alignment["semantic_entities"]:
            if candidate.get("status") != "candidate":
                continue
            eid = candidate.get("entity_id")
            entity = dictionary.get(eid)
            if entity is None or candidate.get("entity_type") != entity["type"]:
                raise MultimodalBundleError("Semantic entity is not an existing base dictionary candidate")
            matches = _rows(candidate.get("matches", []), "candidate.matches")
            if entity["type"] == "Action":
                supported = [match for match in matches if match.get("field") == "actions" and match.get("raw_text") in actions]
                if supported:
                    reference(eid,"Action")
                    edge(parent,eid,"SHOWS_ACTION",properties(source,model=model_name,semantics="model_candidate",observation=observation,
                         detail={"dictionary_candidate":candidate,"model_actions":supported}),eid)
                continue
            supporting_objects = []
            for vnid,obj,index in visual_objects:
                supported = [match for match in matches if match.get("field") == "objects" and
                             _normalize(str(match.get("raw_text", ""))) == _normalize(obj["name"]) and
                             (match.get("object_id") is None or match["object_id"] == obj.get("id"))]
                if supported:
                    supporting_objects.append(vnid)
                    reference(eid,entity["type"])
                    ep = properties(source,model=model_name,semantics="model_candidate",observation=observation,
                         detail={"dictionary_candidate":candidate,"raw_object":obj,"object_index":index,"object_matches":supported})
                    ep.update(alignment_method="dictionary_exact_alias",lexical_score=1.0)
                    edge(vnid,eid,"ALIGNED_WITH",ep,eid,index)
                    if kind == "image":
                        edge(parent,eid,"DEPICTS_INGREDIENT" if entity["type"] == "Ingredient" else "DEPICTS_TOOL",ep,eid,index)
                    elif entity["type"] == "Ingredient":
                        edge(parent,eid,"CONTAINS_INGREDIENT",ep,eid,index)
            if not supporting_objects:
                rejections.append({"kind":"dictionary_candidate","entity_id":eid,"reason":"no_model_object_evidence"})
        relations = list(raw.get("relations", []))
        # Supplied normalized triples must retain the model's raw relation. Their
        # endpoints are resolved against this observation's actual object list.
        for triple in raw_alignment["visual_triples"]:
            if "raw_relation" in triple and _json(triple["raw_relation"]) not in {_json(row) for row in relations}:
                relations.append(triple["raw_relation"])
        for index,relation in enumerate(relations):
            if not isinstance(relation,dict) or not all(isinstance(relation.get(key),str) and relation[key].strip() for key in ("subject","predicate","object")):
                rejections.append({"kind":"visual_relation","index":index,"reason":"malformed_relation","raw_relation":relation})
                continue
            predicate = PREDICATES.get(_normalize(relation["predicate"]))
            if predicate is None:
                rejections.append({"kind":"visual_relation","index":index,"reason":"unknown_predicate","raw_relation":relation})
                continue
            def resolve(key: str) -> list[str]:
                explicit = relation.get(key + "_id")
                return [vnid for vnid,obj,_ in visual_objects if (obj.get("id") == explicit if explicit is not None else
                         _normalize(obj["name"]) == _normalize(relation[key]) or obj.get("id") == relation[key])]
            subjects,targets = resolve("subject"),resolve("object")
            if len(subjects) != 1 or len(targets) != 1 or subjects[0] == targets[0]:
                rejections.append({"kind":"visual_relation","index":index,"reason":"missing_or_ambiguous_object_endpoint","raw_relation":relation})
                continue
            edge(subjects[0],targets[0],predicate,properties(source,model=model_name,semantics="model_candidate",observation=observation,
                 detail={"raw_relation":relation,"relation_index":index}),index)
        nodes[onid]["properties"]["rejections_json"] = _json(rejections)
        rejected_count += len(rejections)
    if set(selected) - observation_ids or any(row.get("media_id") not in observation_ids for row in candidates + supplied_triples):
        raise MultimodalBundleError("Alignment evidence references an unknown observation")
    run_props = {**scope,"verified":False,"semantics":"run_metadata","semantic_accuracy_verified":False,
                 "model_info_json":_json(models),"observation_count":len(observations),"rejected_candidates":rejected_count,
                 "artifact_origin":"caller_supplied_model_artifacts"}
    parent_run = prepared.get("pipeline_run_id",models.get("pipeline_run_id"))
    if parent_run is not None:
        run_props["pipeline_run_id"] = _id(parent_run,"pipeline_run_id")
    node(identifier("run",run_id),"SemanticRun",run_props)
    return {"schema_version":SCHEMA_VERSION,**scope,"nodes":sorted(nodes.values(),key=lambda row:row["id"]),
            "relationships":sorted(edges.values(),key=lambda row:row["id"])}
