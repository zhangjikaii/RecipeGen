"""Pure V5 candidate alignment over externally produced visual/CLIP evidence.

``align_multimodal(payload)`` accepts JSON-compatible ``embedding_space``,
``steps``, ``media``, ``entities`` and optional ``config``. This module performs
no model inference, file access, network access or database writes. A cosine or
dictionary score is an alignment score, never a calibrated confidence.

Frame/clip records with both a timeline_id and start_seconds use a nondecreasing
step sequence. Equal timestamps form an unordered group; input/file order is
never a temporal signal. Images and records without usable timeline metadata
are aligned independently. Unmatched records remain unmatched.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


class AlignmentError(ValueError):
    """The supplied model artifacts cannot satisfy the alignment contract."""


@dataclass(frozen=True)
class _Step:
    identifier: str
    order: int
    text: str
    unit_vector: tuple[float, ...]


@dataclass(frozen=True)
class _Media:
    identifier: str
    kind: str
    timeline: str | None
    start: float | None
    end: float | None
    unit_vector: tuple[float, ...]
    raw: dict[str, Any]


def _json_copy(value: Any, where: str) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise AlignmentError(f"{where} must be finite JSON-compatible evidence") from error
    return copy.deepcopy(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlignmentError(f"{where} must be a nonempty string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentError(f"{where} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise AlignmentError(f"{where} must be a finite number") from error
    if not math.isfinite(number):
        raise AlignmentError(f"{where} must be a finite number")
    return number


def _unit_vector(vector: Any, dimension: int | None, where: str) -> tuple[float, ...]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise AlignmentError(f"{where} must be a nonempty embedding vector")
    if dimension is not None and len(vector) != dimension:
        raise AlignmentError(f"{where} dimension differs from embedding_space")
    values = [_number(item, where) for item in vector]
    scale = max(abs(item) for item in values)
    if scale == 0:
        raise AlignmentError(f"{where} is a zero vector")
    # Scaling before normalization avoids overflowing squares of finite values.
    scaled = [item / scale for item in values]
    norm = math.sqrt(math.fsum(item * item for item in scaled))
    return tuple(item / norm for item in scaled)


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(first, second))))


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Cosine of two finite, nonzero vectors with equal dimensions."""
    left = _unit_vector(first, None, "first embedding")
    right = _unit_vector(second, len(left), "second embedding")
    return _dot(left, right)


def _normalize_term(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def _named_entries(value: Any, where: str) -> list[tuple[str, Any]]:
    if not isinstance(value, list):
        raise AlignmentError(f"{where} must be a list of strings or named objects")
    result = []
    for index, entry in enumerate(value):
        name = entry.get("name") if isinstance(entry, dict) else entry
        result.append((_string(name, f"{where}[{index}].name"), entry))
    return result


def _space(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AlignmentError(f"{where} must identify the shared CLIP space")
    result = dict(value)
    _string(result.get("model"), f"{where}.model")
    _string(result.get("revision"), f"{where}.revision")
    dimension = result.setdefault("dimension", 512)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise AlignmentError(f"{where}.dimension must be a positive integer")
    return result


def _check_item_space(item: dict[str, Any], shared: dict[str, Any], where: str) -> None:
    if "embedding_space" in item:
        declared = _space(item["embedding_space"], f"{where}.embedding_space")
        if any(declared[key] != shared[key] for key in ("model", "revision", "dimension")):
            raise AlignmentError(f"{where} uses a different embedding_space")


def _read_steps(values: Any, space: dict[str, Any]) -> list[_Step]:
    if not isinstance(values, list):
        raise AlignmentError("steps must be a list")
    steps, identifiers, orders = [], set(), set()
    for index, item in enumerate(values):
        where = f"steps[{index}]"
        if not isinstance(item, dict):
            raise AlignmentError(f"{where} must be an object")
        identifier = _string(item.get("id"), f"{where}.id")
        order = item.get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise AlignmentError(f"{where}.order must be a positive integer")
        if identifier in identifiers or order in orders:
            raise AlignmentError("step IDs and order values must be unique")
        identifiers.add(identifier)
        orders.add(order)
        _check_item_space(item, space, where)
        steps.append(_Step(identifier, order, _string(item.get("text"), f"{where}.text"),
                           _unit_vector(item.get("embedding"), space["dimension"], where)))
    return sorted(steps, key=lambda step: step.order)


def _read_media(values: Any, space: dict[str, Any]) -> list[_Media]:
    if not isinstance(values, list):
        raise AlignmentError("media must be a list")
    result, identifiers = [], set()
    for index, item in enumerate(values):
        where = f"media[{index}]"
        if not isinstance(item, dict):
            raise AlignmentError(f"{where} must be an object")
        identifier = _string(item.get("id"), f"{where}.id")
        if identifier in identifiers:
            raise AlignmentError("media IDs must be unique")
        identifiers.add(identifier)
        kind = item.get("kind")
        if kind not in {"image", "frame", "clip"}:
            raise AlignmentError(f"{where}.kind must be image, frame or clip")
        _check_item_space(item, space, where)
        for key in ("model", "evidence"):
            if not isinstance(item.get(key), dict) or not item[key]:
                raise AlignmentError(f"{where}.{key} must preserve model/source metadata")
        if not isinstance(item.get("caption", ""), str):
            raise AlignmentError(f"{where}.caption must be a string")
        _named_entries(item.get("actions", []), f"{where}.actions")
        _named_entries(item.get("objects", []), f"{where}.objects")
        raw_output = item.get("raw_output", {})
        if not isinstance(raw_output, dict) or not isinstance(raw_output.get("relations", []), list):
            raise AlignmentError(f"{where}.raw_output.relations must be a list")
        timeline = item.get("timeline_id")
        if timeline is not None:
            timeline = _string(timeline, f"{where}.timeline_id")
        start = None if item.get("start_seconds") is None else _number(item["start_seconds"], f"{where}.start_seconds")
        end = None if item.get("end_seconds") is None else _number(item["end_seconds"], f"{where}.end_seconds")
        if start is not None and start < 0 or end is not None and end < 0:
            raise AlignmentError(f"{where} timestamps must be nonnegative")
        if end is not None and (start is None or end < start):
            raise AlignmentError(f"{where} end_seconds must follow start_seconds")
        result.append(_Media(identifier, kind, timeline, start, end,
                             _unit_vector(item.get("embedding"), space["dimension"], where), item))
    return result


def _dictionary(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise AlignmentError("entities must be a list of existing graph entities")
    result, identifiers = [], set()
    for index, entity in enumerate(values):
        where = f"entities[{index}]"
        if not isinstance(entity, dict):
            raise AlignmentError(f"{where} must be an object")
        identifier = _string(entity.get("id"), f"{where}.id")
        if identifier in identifiers:
            raise AlignmentError("entity IDs must be unique")
        identifiers.add(identifier)
        if entity.get("type") not in {"Ingredient", "Tool", "Action"}:
            raise AlignmentError(f"{where}.type must be Ingredient, Tool or Action")
        name = _string(entity.get("name"), f"{where}.name")
        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list):
            raise AlignmentError(f"{where}.aliases must be a list")
        terms = [_string(alias, f"{where}.aliases") for alias in [name, *aliases]]
        normalized = {_normalize_term(term): term for term in reversed(terms)}
        if "" in normalized:
            raise AlignmentError(f"{where} has an alias with no lexical content")
        result.append({"id":identifier, "type":entity["type"], "name":name, "terms":normalized})
    return result


def _contains_alias(text: str, alias: str) -> bool:
    # CJK compounds have no separating spaces; Latin aliases require boundaries
    # so that an existing entity "ham" is not inferred from "shampoo".
    if re.search(r"[\u3400-\u9fff\uf900-\ufaff]", alias):
        return alias in text
    return re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) is not None


def _entity_candidates(media: _Media, entities: list[dict[str, Any]], minimum: float) -> list[dict[str, Any]]:
    fields = [("caption", media.raw.get("caption", ""), None)]
    for field in ("actions", "objects"):
        for name, entry in _named_entries(media.raw.get(field, []), field):
            fields.append((field, name, entry.get("id") if isinstance(entry, dict) else None))
    result = []
    for entity in entities:
        matches = []
        for field, raw_text, object_id in fields:
            normalized_text = _normalize_term(raw_text)
            for term, raw_alias in entity["terms"].items():
                if _contains_alias(normalized_text, term):
                    match = {"field":field, "raw_text":raw_text, "alias":raw_alias, "normalized_alias":term}
                    if object_id is not None:
                        match["object_id"] = object_id
                    matches.append(match)
        if matches and minimum <= 1:
            result.append({"media_id":media.identifier, "entity_id":entity["id"], "entity_type":entity["type"],
                           "entity_name":entity["name"], "status":"candidate", "score":1.0,
                           "score_kind":"dictionary_exact_alias", "matches":matches,
                           "evidence_id":"media:" + media.identifier})
    return result


def _endpoint_candidates(term: str | None, entities: list[dict[str, Any]]) -> list[str]:
    return sorted(entity["id"] for entity in entities if term in entity["terms"])


def _visual_triples(media: _Media, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, relation in enumerate(media.raw.get("raw_output", {}).get("relations", [])):
        raw = relation if isinstance(relation, dict) else {}
        normalized = {key:_normalize_term(raw[key]) if isinstance(raw.get(key), str) else None
                      for key in ("subject", "predicate", "object")}
        missing = [key for key, value in normalized.items() if not value]
        result.append({"id":f"{media.identifier}:relation:{index}", "media_id":media.identifier,
                       "status":"unusable" if missing else "candidate", **normalized,
                       "missing_fields":missing, "raw_relation":copy.deepcopy(relation),
                       "subject_entity_candidates":_endpoint_candidates(normalized["subject"], entities),
                       "object_entity_candidates":_endpoint_candidates(normalized["object"], entities),
                       "model":copy.deepcopy(media.raw["model"]), "evidence_id":"media:" + media.identifier})
    return result


def _better(first: tuple[float, int, dict[str, int]], second: tuple[float, int, dict[str, int]] | None) -> bool:
    if second is None:
        return True
    left, right = first[:2], second[:2]
    if left != right:
        return left > right
    return tuple(sorted(first[2].items())) < tuple(sorted(second[2].items()))


def _monotonic_choices(media: list[_Media], scores: Mapping[str, list[float]], threshold: float,
                       step_count: int) -> dict[str, int]:
    # State is the maximum step reached. Equal-time media have no internal order.
    bins: dict[float, list[_Media]] = {}
    for item in media:
        assert item.start is not None
        bins.setdefault(item.start, []).append(item)
    states: dict[int, tuple[float, int, dict[str, int]]] = {-1:(0.0, 0, {})}
    for timestamp in sorted(bins):
        group = sorted(bins[timestamp], key=lambda item: item.identifier)
        next_states: dict[int, tuple[float, int, dict[str, int]]] = {}
        for previous, path in states.items():
            if _better(path, next_states.get(previous)):
                next_states[previous] = path
            best: dict[str, int] = {}
            for upper in range(max(previous, 0), step_count):
                for item in group:
                    score = scores[item.identifier][upper]
                    current = best.get(item.identifier)
                    if score >= threshold and (current is None or score > scores[item.identifier][current]):
                        best[item.identifier] = upper
                if not best:
                    continue
                last = max(previous, max(best.values()))
                gain = math.fsum([path[0], *(max(0.0, scores[key][step] - threshold) for key, step in best.items())])
                candidate = (gain, path[1] + len(best), {**path[2], **best})
                if _better(candidate, next_states.get(last)):
                    next_states[last] = candidate
        states = next_states
    chosen = None
    for path in states.values():
        if _better(path, chosen):
            chosen = path
    return {} if chosen is None else chosen[2]


def align_multimodal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Align genuine artifacts supplied by the caller; provenance is preserved.

    Required shared space: ``{model, revision, dimension}`` (dimension defaults
    to 512). Each media record has ``id``, ``kind``, ``embedding``, nonempty
    ``model`` and ``evidence`` metadata. Steps have ``id, order, text, embedding``.
    Entities are existing ``{id,type,name,aliases?}`` dictionary records.

    Defaults are exploratory thresholds, not empirically calibrated guarantees.
    Temporal objective maximizes the sum of cosine-minus-threshold margins, then
    the matched-record count. Repeated matches to the same step are allowed.
    """
    if not isinstance(payload, Mapping):
        raise AlignmentError("payload must be a JSON object")
    data = _json_copy(dict(payload), "payload")
    if data.get("schema_version", 1) != 1:
        raise AlignmentError("unsupported schema_version")
    space = _space(data.get("embedding_space"), "embedding_space")
    steps = _read_steps(data.get("steps"), space)
    media = _read_media(data.get("media"), space)
    entities = _dictionary(data.get("entities", []))
    config = data.get("config", {})
    if not isinstance(config, dict):
        raise AlignmentError("config must be an object")
    threshold = _number(config.get("similarity_threshold", 0.25), "similarity_threshold")
    minimum = _number(config.get("entity_min_score", 1.0), "entity_min_score")
    top_k = config.get("top_k", 3)
    if not -1 <= threshold <= 1 or not 0 <= minimum <= 1:
        raise AlignmentError("thresholds must be within cosine [-1,1] and dictionary [0,1] ranges")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise AlignmentError("top_k must be a positive integer")
    scores = {item.identifier:[_dot(item.unit_vector, step.unit_vector) for step in steps] for item in media}
    temporal: dict[str, list[_Media]] = {}
    choices, modes = {}, {}
    for item in media:
        if item.kind in {"frame", "clip"} and item.timeline is not None and item.start is not None:
            temporal.setdefault(item.timeline, []).append(item)
            modes[item.identifier] = "temporal_monotonic"
        else:
            modes[item.identifier] = "independent"
            eligible = [index for index, score in enumerate(scores[item.identifier]) if score >= threshold]
            if eligible:
                choices[item.identifier] = max(eligible, key=lambda index: (scores[item.identifier][index], -steps[index].order))
    for group in temporal.values():
        choices.update(_monotonic_choices(group, scores, threshold, len(steps)))
    alignments, semantic_entities, visual_triples, evidence = [], [], [], []
    for item in media:
        ranked = sorted(range(len(steps)), key=lambda index: (-scores[item.identifier][index], steps[index].order))
        alternatives = [{"step_id":steps[index].identifier, "step_order":steps[index].order,
                         "score":scores[item.identifier][index], "passes_threshold":scores[item.identifier][index] >= threshold}
                        for index in ranked[:top_k]]
        selected = choices.get(item.identifier)
        qualified = any(score >= threshold for score in scores[item.identifier])
        reason = ("selected_by_cosine_and_temporal_constraint" if modes[item.identifier] == "temporal_monotonic" else "passed_cosine_threshold") if selected is not None else (
            "no_steps" if not steps else "temporal_conflict" if qualified else "below_threshold")
        alignments.append({"media_id":item.identifier, "step_id":None if selected is None else steps[selected].identifier,
                           "step_order":None if selected is None else steps[selected].order,
                           "status":"unmatched" if selected is None else "candidate",
                           "score":None if selected is None else scores[item.identifier][selected],
                           "score_kind":"clip_cosine_similarity", "threshold":threshold,
                           "alignment_mode":modes[item.identifier], "timeline_id":item.timeline,
                           "start_seconds":item.start, "end_seconds":item.end, "reason":reason,
                           "alternatives":alternatives, "evidence_id":"media:" + item.identifier})
        semantic_entities.extend(_entity_candidates(item, entities, minimum))
        visual_triples.extend(_visual_triples(item, entities))
        evidence.append({"id":"media:" + item.identifier, "media_id":item.identifier, "kind":item.kind,
                         "model":copy.deepcopy(item.raw["model"]), "source":copy.deepcopy(item.raw["evidence"]),
                         "raw_caption":item.raw.get("caption", ""), "raw_actions":copy.deepcopy(item.raw.get("actions", [])),
                         "raw_objects":copy.deepcopy(item.raw.get("objects", [])), "raw_output":copy.deepcopy(item.raw.get("raw_output", {})),
                         "embedding_space":copy.deepcopy(space)})
    matched = sum(item["status"] == "candidate" for item in alignments)
    return {"schema_version":1, "status":"ok" if matched else "no_semantic_match", "embedding_space":space,
            "config":{"similarity_threshold":threshold, "entity_min_score":minimum, "top_k":top_k},
            "alignments":alignments, "semantic_entities":semantic_entities, "visual_triples":visual_triples,
            "evidence":evidence, "steps":[{"id":step.identifier, "order":step.order, "text":step.text} for step in steps],
            "summary":{"media":len(media), "matched_candidates":matched, "unmatched":len(media)-matched,
                       "semantic_entity_candidates":len(semantic_entities), "visual_triple_candidates":sum(row["status"] == "candidate" for row in visual_triples)},
            "model_inference_performed":False,
            "limitations":["Scores are candidate alignment scores, not calibrated confidence or semantic ground truth.",
                           "Shared model/revision metadata is checked; numeric vectors alone cannot prove model execution or provenance.",
                           "Temporal constraints use start_seconds within each timeline; overlapping intervals are ordered by their starts and equal starts are unordered.",
                           "Images and media without both timeline_id and timestamps use independent similarity; file order has no semantic meaning.",
                           "Entity matches use exact dictionary names/aliases and are mention candidates; they do not verify object identity, ingredients or inventory.",
                           "Visual triples come only from supplied raw model relations; captions never create visual relations."]}
