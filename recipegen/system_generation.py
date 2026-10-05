"""Evidence-bound RecipeGen recipe assembly and optional local selection.

The language model selects supplied records; it cannot introduce steps,
quantities, timing, or safety claims. Original graph text remains untrusted data.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

PROTOCOL = "recipegen-local-generation-v1"
MODEL_REPO = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
MODEL_REVISION = "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
MAX_CONTEXT_CHARS = 16_000
MAX_CANDIDATES = 12
MAX_OUTPUT_CHARS = 65_536
REASON_KINDS = {"graph_record": "graph_record", "ingredient_mentions": "ingredient_mentions", "original_steps": "step"}
SAFE_ERRORS = {
    "busy": "本地模型正在处理另一条请求，已使用图谱原步骤生成。",
    "timeout": "本地模型请求超时，已使用图谱原步骤生成。",
    "unavailable": "本地模型或运行环境不可用，已使用图谱原步骤生成。",
    "model_error": "本地模型调用失败，已使用图谱原步骤生成。",
    "invalid_output": "本地模型选择未通过来源与结构校验，已使用图谱原步骤生成。",
    "invalid_runner_output": "本地模型进程返回格式无效，已使用图谱原步骤生成。",
}
LIMITATIONS = [
    "步骤直接保留当前图谱中的原文；图谱食材提及并非完整配方，未补造用量或总用时。",
    "排除条件只检查已记录的食材提及和步骤文字；数据不完整，不能据此确认过敏或饮食安全。",
    "来源与结构校验不代表原数据或模型语义准确率已验证。",
]


class GenerationValidationError(ValueError):
    pass


def _strict_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_OUTPUT_CHARS:
        raise GenerationValidationError("invalid_output")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise GenerationValidationError("invalid_output")
            value[key] = item
        return value

    def constant(_):
        raise GenerationValidationError("invalid_output")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise GenerationValidationError("invalid_output") from error
    if not isinstance(value, dict):
        raise GenerationValidationError("invalid_output")
    return value


def decode_runner_stdout(stdout: str) -> dict[str, Any]:
    """Accept one protocol envelope; library log lines cannot become selections."""
    if not isinstance(stdout, str) or len(stdout) > 1_000_000:
        raise GenerationValidationError("invalid_runner_output")
    candidates = []
    for line in stdout.splitlines():
        try:
            value = _strict_object(line)
        except GenerationValidationError:
            continue
        if value.get("protocol") == PROTOCOL:
            candidates.append(value)
    if len(candidates) != 1:
        raise GenerationValidationError("invalid_runner_output")
    value = candidates[0]
    if value.get("status") not in {"ok", "unavailable", "model_error"}:
        raise GenerationValidationError("invalid_runner_output")
    if type(value.get("llm_called")) is not bool:
        raise GenerationValidationError("invalid_runner_output")
    if value["status"] == "ok" and (not isinstance(value.get("raw_text"), str) or not value["llm_called"]):
        raise GenerationValidationError("invalid_runner_output")
    return value


def run_local_subprocess(project_root: Path, payload: dict[str, Any], *, timeout: float = 150) -> dict[str, Any]:
    """Run one short-lived, offline, text-only MLX worker under a process lock."""
    root = Path(project_root).resolve()
    executable = root / ".venv-mm/bin/python"
    script = root / "scripts/local_recipe_generation.py"
    if not executable.is_file() or not script.is_file():
        return {"status": "unavailable", "llm_called": False}
    runtime = root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lock_path = runtime / "recipe-generation.lock"
    with lock_path.open("a+") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "llm_called": False}
        env = dict(os.environ)
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        try:
            completed = subprocess.run(
                [str(executable), str(script)], input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                capture_output=True, text=True, encoding="utf-8", cwd=root, env=env, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # The run() context kills and waits for its worker; partial output is
            # not a valid selection, and does not prove generation was reached.
            return {"status": "timeout", "llm_called": False, "inference_started_unknown": True}
        except (OSError, ValueError, TypeError):
            return {"status": "unavailable", "llm_called": False}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            result = decode_runner_stdout(completed.stdout)
        except GenerationValidationError:
            return {"status": "invalid_runner_output", "llm_called": False, "inference_started_unknown": True}
        if completed.returncode != 0 and result["status"] == "ok":
            return {"status": "model_error", "llm_called": result["llm_called"]}
        return result


def _terms(values: Any, label: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)) or len(values) > 32:
        raise ValueError(f"{label} 必须为最多 32 项的列表")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 80 for item in values):
        raise ValueError(f"{label} 必须为非空短字符串")
    return list(dict.fromkeys(item.strip() for item in values))


def _safe_model(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    keys = ("name", "repo", "revision", "backend", "max_tokens", "temperature", "repetition_penalty", "repetition_context_size", "num_images", "mlx_cache_limit_bytes")
    result = {key: value[key] for key in keys if key in value and type(value[key]) in (str, int, float, bool)}
    return result or None


def _record_contains_exclusion(record, terms):
    names = {str(mention.get(key, "")).strip().casefold()
             for mention in record.get("ingredient_mentions", [])
             for key in ("normalized_name", "canonical_name", "name", "zh_name")}
    text = "\n".join([record["title"], *(step["text"] for step in record["steps"])]).casefold()
    for term in terms:
        normalized = term.casefold()
        if normalized in names:
            return True
        # English tokens need word boundaries (egg != eggplant). Chinese text
        # has no equivalent space-delimited word boundary.
        pattern = r"(?<!\w)" + re.escape(normalized) + r"(?!\w)" if re.fullmatch(r"[a-z0-9 _-]+", normalized) else re.escape(normalized)
        if re.search(pattern, text):
            return True
    return False


def _candidate_records(recipes: list[dict[str, Any]], excluded: list[str]):
    eligible, unavailable, seen = [], [], set()
    for original in recipes:
        if not isinstance(original, dict) or not isinstance(original.get("id"), str) or not original["id"]:
            unavailable.append({"recipe_id": None, "reason": "invalid_recipe_record"})
            continue
        identifier = original["id"]
        if identifier in seen:
            unavailable.append({"recipe_id": identifier, "reason": "duplicate_recipe_id"})
            continue
        seen.add(identifier)
        title = original.get("title")
        steps = original.get("steps")
        sources = original.get("sources")
        if not isinstance(title, str) or not title.strip() or not isinstance(steps, list) or not isinstance(sources, list):
            unavailable.append({"recipe_id": identifier, "reason": "invalid_recipe_record"})
            continue
        if not steps:
            unavailable.append({"recipe_id": identifier, "reason": "no_original_steps"})
            continue
        source_map = {source["source_id"]: source for source in sources if isinstance(source, dict) and isinstance(source.get("source_id"), str) and source["source_id"]}
        step_ids, orders = set(), set()
        valid = True
        for step in steps:
            if (not isinstance(step, dict) or not isinstance(step.get("id"), str) or not step["id"]
                    or not isinstance(step.get("text"), str) or not step["text"].strip()
                    or type(step.get("order")) is not int or step["order"] <= 0
                    or not isinstance(step.get("source_id"), str) or step["source_id"] not in source_map
                    or step["id"] in step_ids or step["order"] in orders):
                valid = False
                break
            step_ids.add(step["id"])
            orders.add(step["order"])
        if not valid:
            unavailable.append({"recipe_id": identifier, "reason": "invalid_or_unresolved_steps"})
            continue
        mentions = original.get("ingredient_mentions", [])
        if not isinstance(mentions, list) or any(not isinstance(m, dict) or not isinstance(m.get("id"), str)
                or not isinstance(m.get("name"), str) or not m["name"].strip() for m in mentions):
            unavailable.append({"recipe_id": identifier, "reason": "invalid_ingredient_mentions"})
            continue
        if _record_contains_exclusion(original, excluded):
            unavailable.append({"recipe_id": identifier, "reason": "recorded_excluded_ingredient"})
            continue
        record = copy.deepcopy(original)
        record["steps"] = sorted(record["steps"], key=lambda step: step["order"])
        record["ingredient_mentions"] = copy.deepcopy(mentions)
        eligible.append(record)
    return eligible, unavailable


def _evidence(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []

    def add(kind, recipe, graph_id, text, source_id=None):
        item = {"id": f"E{len(evidence) + 1}", "kind": kind, "recipe_id": recipe["id"], "graph_id": graph_id, "text": text, "source_id": source_id}
        source = next((s for s in recipe["sources"] if isinstance(s, dict) and s.get("source_id") == source_id), None) if source_id else None
        if source and isinstance(source.get("url"), str) and source["url"]:
            item["source_url"] = source["url"]
        evidence.append(item)

    for recipe in recipes:
        # A Recipe's title is a graph-record claim, not a proof of ingredients.
        known_sources = {source["source_id"] for source in recipe["sources"]
                         if isinstance(source, dict) and isinstance(source.get("source_id"), str)}
        title_sources = {source["source_id"] for source in recipe["sources"]
                         if isinstance(source, dict) and source.get("role") == "title"
                         and source.get("source_id") in known_sources}
        title_source = next(iter(title_sources)) if len(title_sources) == 1 else None
        add("graph_record", recipe, recipe["id"], recipe["title"], title_source)
        for ingredient in recipe["ingredient_mentions"]:
            # Catalog associations now retain verified extraction Source IDs.
            # Legacy records without that association remain unresolved.
            source_id = ingredient.get("source_id")
            if not isinstance(source_id, str) or source_id not in known_sources:
                source_id = None
            add("ingredient_mentions", recipe, ingredient["id"], ingredient.get("zh_name") or ingredient["name"], source_id)
            ids = ingredient.get("source_ids")
            if isinstance(ids, list) and ids and all(isinstance(sid, str) and sid in known_sources for sid in ids):
                evidence[-1]["source_ids"] = list(dict.fromkeys(ids))
                evidence[-1]["source_urls"] = [source["url"] for source in recipe["sources"]
                                               if isinstance(source, dict) and source.get("source_id") in ids
                                               and isinstance(source.get("url"), str) and source["url"]]
        for step in recipe["steps"]:
            add("step", recipe, step["id"], step["text"], step["source_id"])
    return evidence


def _selection_payload(question, recipes, evidence, ingredients, excluded, limit):
    by_recipe = {recipe["id"]: [] for recipe in recipes}
    for item in evidence:
        by_recipe[item["recipe_id"]].append(item)
    candidates, aliases, supplied = [], {}, {}
    for recipe in recipes[:MAX_CANDIDATES]:
        alias = f"R{len(candidates) + 1}"
        rows = by_recipe[recipe["id"]]
        counts = {kind: 0 for kind in REASON_KINDS.values()}
        excerpts = []
        for item in rows:
            cap = 4 if item["kind"] == "step" else (8 if item["kind"] == "ingredient_mentions" else 1)
            if counts[item["kind"]] >= cap:
                continue
            counts[item["kind"]] += 1
            excerpt = {"id": item["id"], "kind": item["kind"], "text": item["text"][:240], "is_excerpt": len(item["text"]) > 240}
            excerpts.append(excerpt)
        candidate = {"recipe": alias, "title": recipe["title"][:160], "evidence": excerpts,
                     "context_is_partial": len(excerpts) < len(rows) or any(e["is_excerpt"] for e in excerpts),
                     "original_step_count": len(recipe["steps"])}
        trial = {"question": question, "ingredients": ingredients, "excluded_ingredients": excluded,
                 "limit": limit, "candidates": candidates + [candidate]}
        if len(json.dumps(trial, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
            break
        candidates.append(candidate)
        aliases[alias] = recipe["id"]
        supplied.update({item["id"]: item for item in rows if item["id"] in {e["id"] for e in excerpts}})
    payload = {"protocol": PROTOCOL, "question": question, "ingredients": ingredients,
               "excluded_ingredients": excluded, "limit": min(limit, len(candidates)), "candidates": candidates}
    return payload, aliases, supplied


def validate_local_selection(raw: str, aliases: dict[str, str], evidence: dict[str, dict[str, Any]], limit: int):
    value = _strict_object(raw)
    if set(value) != {"selections"} or not isinstance(value["selections"], list) or not 1 <= len(value["selections"]) <= limit:
        raise GenerationValidationError("invalid_output")
    selections, seen = [], set()
    for selection in value["selections"]:
        if not isinstance(selection, dict) or set(selection) != {"recipe", "reason_codes", "evidence_ids"}:
            raise GenerationValidationError("invalid_output")
        alias = selection.get("recipe")
        reasons, ids = selection.get("reason_codes"), selection.get("evidence_ids")
        if not isinstance(alias, str) or alias not in aliases or alias in seen:
            raise GenerationValidationError("invalid_output")
        if (not isinstance(reasons, list) or not reasons or len(reasons) > len(REASON_KINDS)
                or any(not isinstance(reason, str) or reason not in REASON_KINDS for reason in reasons)
                or len(reasons) != len(set(reasons)) or not isinstance(ids, list) or not ids
                or len(ids) > 16 or any(not isinstance(identifier, str) or identifier not in evidence for identifier in ids)
                or len(ids) != len(set(ids))):
            raise GenerationValidationError("invalid_output")
        if any(evidence[identifier]["recipe_id"] != aliases[alias] for identifier in ids):
            raise GenerationValidationError("invalid_output")
        if any(not any(evidence[identifier]["kind"] == REASON_KINDS[reason] for identifier in ids) for reason in reasons):
            raise GenerationValidationError("invalid_output")
        seen.add(alias)
        selections.append({"recipe_id": aliases[alias], "reason_codes": list(reasons), "evidence_ids": list(ids)})
    return selections


def _answer(recipes, evidence):
    lines = ["以下食谱依据知识图谱中的原始步骤组织。食材为已记录提及，非完整清单；未补充用量或总用时。"]
    for index, recipe in enumerate(recipes, 1):
        items = [item for item in evidence if item["recipe_id"] == recipe["id"]]
        title_evidence = next(item for item in items if item["kind"] == "graph_record")
        lines.extend(["", f"{index}. {recipe['title']} [{title_evidence['id']}]"])
        ingredient_evidence = [item for item in items if item["kind"] == "ingredient_mentions"]
        if ingredient_evidence:
            lines.append("图谱提及食材：" + "、".join(f"{item['text']} [{item['id']}]" for item in ingredient_evidence))
        else:
            lines.append("图谱未记录食材提及，不能据此补出完整配方。")
        lines.append("原始步骤：")
        for step in recipe["steps"]:
            item = next(item for item in items if item["kind"] == "step" and item["graph_id"] == step["id"])
            lines.append(f"{step['order']}. {step['text']} [{item['id']}]")
    lines.extend(["", "排除条件只能检查已有记录；不能据此确认过敏或饮食安全。"])
    return "\n".join(lines)


class RecipeGenerator:
    def __init__(self, project_root: Path | str | None = None, *, local_runner: Callable[[dict[str, Any]], Any] | None = None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.local_runner = local_runner or (lambda payload: run_local_subprocess(self.root, payload))

    def status(self) -> dict[str, Any]:
        available, model = False, None
        try:
            info = json.loads((self.root / ".runtime/multimodal-models.json").read_text())["vision"]
            path = Path(info["path"]).resolve()
            allowed = (self.root / ".runtime/model-cache").resolve()
            available = (info["repo"] == MODEL_REPO and info["revision"] == MODEL_REVISION
                         and path.is_relative_to(allowed) and (path / "model.safetensors").is_file()
                         and (path / "config.json").is_file() and (self.root / ".venv-mm/bin/python").is_file()
                         and (self.root / "scripts/local_recipe_generation.py").is_file())
            model = {"name": info["repo"], "revision": info["revision"], "backend": "mlx", "num_images": 0}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return {"default_mode": "grounded", "grounded_available": True, "local_available": bool(available),
                "local_model_available": bool(available),
                "model": model, "local_timeout_seconds": 150, "local_max_concurrency": 1,
                "semantic_accuracy_verified": False, "limitations": list(LIMITATIONS)}

    def generate(self, question: str, recipes: list[dict[str, Any]], *, mode: str = "grounded",
                 ingredients=None, excluded_ingredients=None, limit: int = 3) -> dict[str, Any]:
        if not isinstance(question, str) or len(question) > 1_000:
            raise ValueError("question 必须为最多 1000 字符的字符串")
        if mode not in {"grounded", "local"}:
            raise ValueError("mode 必须为 grounded 或 local")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("limit 必须为 1–5")
        if not isinstance(recipes, list) or len(recipes) > 100:
            raise ValueError("recipes 必须为最多 100 条菜谱详情")
        ingredients = _terms(ingredients, "ingredients")
        excluded = _terms(excluded_ingredients, "excluded_ingredients")
        trace, started = [], time.perf_counter()
        eligible, unavailable = _candidate_records(recipes, excluded)
        all_evidence = _evidence(eligible)
        trace.append({"stage": "graph_evidence", "status": "ok" if eligible else "no_match", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
        generation = {"mode": mode, "effective_mode": "grounded", "model": None, "model_info": None, "llm_called": False,
                      "fallback": False, "errors": [], "raw_text": None, "selection": []}
        checks = [{"name": "original_steps_have_source", "passed": True}, {"name": "evidence_matches_supplied_graph_records", "passed": True}]
        if not eligible:
            answer = "没有可据此生成食谱的图谱候选。"
            if any(item["reason"] == "no_original_steps" for item in unavailable):
                answer += "部分菜谱没有原始步骤，不能补造做法。"
            return {"status": "no_match", "answer": answer, "recipes": [], "evidence": [], "generation": generation,
                    "validation": {"passed": True, "checks": checks, "semantic_accuracy_verified": False},
                    "trace": trace, "unavailable_recipes": unavailable, "limitations": list(LIMITATIONS)}
        selected = eligible[:limit]
        if mode == "local":
            payload, aliases, supplied = _selection_payload(question, eligible, all_evidence, ingredients, excluded, limit)
            generation["context"] = {"candidate_count": len(aliases), "eligible_candidate_count": len(eligible),
                                     "evidence_count": len(supplied), "bounded_context": True,
                                     "original_steps_truncated_in_answer": False,
                                     "partial_context_candidates": sum(c["context_is_partial"] for c in payload["candidates"])}
            tick = time.perf_counter()
            error_code = None
            try:
                result = self.local_runner(payload)
                if isinstance(result, str):
                    result = {"status": "ok", "raw_text": result, "llm_called": True}
                if not isinstance(result, dict):
                    raise GenerationValidationError("invalid_runner_output")
                generation["llm_called"] = result.get("llm_called") is True
                generation["model_info"] = _safe_model(result.get("model"))
                if generation["model_info"]:
                    generation["model"] = generation["model_info"].get("name") or generation["model_info"].get("repo")
                if result.get("inference_started_unknown") is True:
                    generation["inference_started_unknown"] = True
                raw = result.get("raw_text")
                if isinstance(raw, str):
                    generation["raw_text"] = raw[:MAX_OUTPUT_CHARS]
                    generation["raw_text_truncated"] = len(raw) > MAX_OUTPUT_CHARS
                if result.get("status") != "ok":
                    error_code = result.get("status") if result.get("status") in SAFE_ERRORS else "model_error"
                elif not generation["llm_called"]:
                    error_code = "invalid_runner_output"
                else:
                    selections = validate_local_selection(raw, aliases, supplied, limit)
                    records = {recipe["id"]: recipe for recipe in eligible}
                    selected = [records[item["recipe_id"]] for item in selections]
                    generation["selection"] = selections
                    generation["effective_mode"] = "local_selection_grounded_answer"
            except GenerationValidationError as error:
                error_code = str(error) if str(error) in SAFE_ERRORS else "invalid_output"
            except Exception:
                error_code = "model_error"
            if error_code:
                generation["fallback"] = True
                generation["errors"] = [{"code": error_code, "message": SAFE_ERRORS[error_code]}]
            checks.append({"name": "local_selection_valid", "passed": error_code is None})
            trace.append({"stage": "local_selection", "status": "fallback" if error_code else "ok", "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3)})
        selected_ids = {recipe["id"] for recipe in selected}
        evidence = [item for item in all_evidence if item["recipe_id"] in selected_ids]
        tick = time.perf_counter()
        answer = _answer(selected, evidence)
        checks.extend([{"name": "answer_steps_are_original_and_complete", "passed": True},
                       {"name": "answer_has_resolvable_step_citations", "passed": True},
                       {"name": "no_generated_quantities_or_total_duration", "passed": True}])
        trace.append({"stage": "grounded_assembly", "status": "ok", "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3)})
        return {"status": "ok", "answer": answer, "recipes": selected, "evidence": evidence,
                "generation": generation,
                # A recovered answer passes assembly checks; rejected model
                # selection remains visible as its own failed check.
                "validation": {"passed": all(check["passed"] for check in checks if check["name"] != "local_selection_valid"),
                               "checks": checks, "semantic_accuracy_verified": False},
                "trace": trace, "unavailable_recipes": unavailable, "limitations": list(LIMITATIONS)}
