from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import prompts
from .config import Settings
from .graph import GraphStore, Neo4jGraphStore, SQLiteGraphStore, eligible, load_document
from .llm import LLMClient, LLMError
from .models import Constraints, GeneratedRecommendations, Recipe, RecommendRequest
from .query import QueryParser


def create_store(settings: Settings) -> GraphStore:
    if settings.graph_backend == "neo4j":
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database)
    store = SQLiteGraphStore(settings.db_path)
    if not settings.db_path.exists():
        store.import_document(load_document(settings.example_path))
    return store


def recipe_evidence(recipe: Recipe) -> list[dict]:
    rows = []

    def add(predicate: str, value, suffix: str):
        rows.append({"id": f"{recipe.id}:{suffix}", "recipe_id": recipe.id, "subject": recipe.name, "predicate": predicate, "object": value, "source": recipe.source.model_dump()})

    for index, name in enumerate(recipe.ingredients):
        add("USES_INGREDIENT", name, f"ingredient:{index}")
    for index, name in enumerate(recipe.seasonings):
        add("USES_SEASONING", name, f"seasoning:{index}")
    add("minutes", recipe.minutes if recipe.minutes is not None else "未记录", "minutes")
    for index, tag in enumerate(recipe.tags):
        add("HAS_TAG", tag, f"tag:{index}")
    for index, step in enumerate(recipe.steps, 1):
        add("HAS_STEP", {"order": index, "text": step}, f"step:{index}")
    return rows


def valid_reasons(recipe: Recipe, constraints: Constraints) -> list[str]:
    reasons = []
    available = set(constraints.available_ingredients)
    if set(recipe.ingredients).intersection(available):
        reasons.append("ingredient_match")
    if constraints.max_minutes is not None and recipe.minutes is not None and recipe.minutes <= constraints.max_minutes:
        reasons.append("within_time")
    if constraints.required_tags and set(constraints.required_tags).issubset(recipe.tags):
        reasons.append("tag_match")
    if constraints.allow_missing and set(recipe.ingredients) - available:
        reasons.append("needs_shopping")
    if constraints.recipe_name and constraints.recipe_name in recipe.name:
        reasons.append("named_recipe")
    return reasons


def render_recommendation(recipe: Recipe, constraints: Constraints, codes: list[str]) -> dict:
    missing = sorted(set(recipe.ingredients) - set(constraints.available_ingredients)) if constraints.available_ingredients or constraints.allow_missing else []
    reasons = []
    if "ingredient_match" in codes:
        matched = [name for name in recipe.ingredients if name in constraints.available_ingredients]
        reasons.append("使用现有主料：" + "、".join(matched))
    if "within_time" in codes:
        reasons.append(f"图谱记录用时 {recipe.minutes} 分钟，满足时间限制")
    if "tag_match" in codes:
        reasons.append("符合标签：" + "、".join(constraints.required_tags))
    if "needs_shopping" in codes:
        reasons.append("需补充主料：" + "、".join(missing))
    if "named_recipe" in codes:
        reasons.append("找到指定菜谱的图谱记录")
    evidence = recipe_evidence(recipe)
    return {
        "recipe_id": recipe.id, "name": recipe.name, "ingredients": recipe.ingredients,
        "seasonings": recipe.seasonings, "minutes": recipe.minutes, "tags": recipe.tags,
        "steps": recipe.steps, "missing_ingredients": missing,
        "reason": "；".join(reasons), "reason_codes": codes,
        "evidence_ids": [row["id"] for row in evidence], "source": recipe.source.model_dump(),
    }


class Recommender:
    def __init__(self, settings: Settings, store: GraphStore | None = None, llm: LLMClient | None = None):
        self.settings = settings
        self.store = store or create_store(settings)
        self.llm = llm or LLMClient(settings)

    def status(self) -> dict:
        document = self.store.document()
        return {
            "backend": self.settings.graph_backend,
            "dataset": document.dataset.model_dump(), "graph": self.store.stats(),
            "llm": {"provider": self.settings.llm_provider, "configured": self.settings.llm_configured, "model": self.settings.llm_model or None},
        }

    def recommend(self, request: RecommendRequest) -> dict:
        started = time.perf_counter()
        document = self.store.document()
        parser = QueryParser(document)
        trace = []
        errors = []

        def log(stage: str, status: str, detail: str, start: float) -> None:
            trace.append({"stage": stage, "status": status, "detail": detail, "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)})

        t = time.perf_counter()
        constraints = parser.parse(request.question)
        log("parse", "ok", "使用可审计的中文规则基线提取条件", t)
        if request.use_llm:
            t = time.perf_counter()
            try:
                raw = self.llm.json(prompts.EXTRACT, {
                    "question": request.question, "ingredient_aliases": document.ingredient_aliases,
                    "available_tags": sorted(parser.tags), "known_ingredients": sorted(set(parser.aliases.values())),
                }, "extract")
                constraints = parser.merge_llm(constraints, Constraints.model_validate(raw), request.question)
                log("llm_extract", "ok", "模型条件通过 schema 和用户提及内容校验", t)
            except (LLMError, ValueError) as error:
                detail = str(error) if isinstance(error, LLMError) else "模型提取条件未通过校验，保留规则解析条件"
                errors.append(detail)
                log("llm_extract", "fallback", detail, t)
        constraints = parser.apply_overrides(constraints, request.constraints)
        generation = {"mode": "fallback" if errors else "template", "provider": self.settings.llm_provider if request.use_llm else "disabled", "model": self.settings.llm_model or None, "llm_requested": request.use_llm}
        result = {
            "request_id": str(uuid.uuid4()), "question": request.question,
            "constraints": constraints.model_dump(), "recommendations": [], "evidence": [],
            "generation": generation, "dataset": document.dataset.model_dump(), "trace": trace,
            "validation": {"passed": True, "checks": []}, "candidate_count": 0,
        }
        if not constraints.available_ingredients and not constraints.allow_missing and not constraints.recipe_name:
            result.update(status="needs_clarification", answer="请补充现有主料；也可以允许缺料，我会列出需要补充的食材。")
            log("retrieve", "skipped", "缺少库存信息，未擅自放宽条件", time.perf_counter())
            return self._finish(result, started, errors)

        t = time.perf_counter()
        candidates = self.store.retrieve(constraints)
        result["candidate_count"] = len(candidates)
        log("retrieve", "ok", f"通过图关系检索得到 {len(candidates)} 个符合硬约束的候选", t)
        if not candidates:
            result.update(status="no_match", answer="图谱中没有满足全部条件的菜谱。可以调整现有主料或时间条件，或明确允许缺料。未记录总用时的菜谱无法通过时间上限筛选。")
            return self._finish(result, started, errors)

        context = candidates[:max(self.settings.context_candidates, constraints.top_k)]
        choices = [{"recipe_id": recipe.id, "reason_codes": valid_reasons(recipe, constraints)} for recipe in context[:constraints.top_k]]
        if request.use_llm and not errors:
            t = time.perf_counter()
            try:
                raw = self.llm.json(prompts.GENERATE, {
                    "question": request.question, "constraints": constraints.model_dump(),
                    "candidates": [recipe.model_dump() for recipe in context],
                    "evidence": [row for recipe in context for row in recipe_evidence(recipe)],
                }, "generate")
                generated = GeneratedRecommendations.model_validate(raw)
                by_id = {recipe.id: recipe for recipe in context}
                chosen_ids = [choice.recipe_id for choice in generated.recommendations]
                if len(chosen_ids) > constraints.top_k or len(set(chosen_ids)) != len(chosen_ids):
                    raise ValueError("候选数量或唯一性错误")
                for choice in generated.recommendations:
                    if choice.recipe_id not in by_id:
                        raise ValueError("模型引用了未检索到的菜谱")
                    if not set(choice.reason_codes).issubset(valid_reasons(by_id[choice.recipe_id], constraints)):
                        raise ValueError("模型推荐理由不受图谱事实支持")
                choices = [choice.model_dump() for choice in generated.recommendations]
                generation["mode"] = "llm"
                log("generate", "ok", "模型结构化推荐的候选 ID 和理由均通过图谱校验", t)
            except (LLMError, ValueError) as error:
                detail = str(error) if isinstance(error, LLMError) else "模型推荐未通过候选 ID、数量或事实校验，改用图谱模板"
                errors.append(detail)
                generation["mode"] = "fallback"
                log("generate", "fallback", detail, t)
        elif not request.use_llm:
            log("generate", "ok", "使用图谱事实模板，未调用大模型", time.perf_counter())

        selected = {recipe.id: recipe for recipe in context}
        recommendations = [render_recommendation(selected[choice["recipe_id"]], constraints, choice["reason_codes"]) for choice in choices]
        evidence = [row for choice in choices for row in recipe_evidence(selected[choice["recipe_id"]])]
        t = time.perf_counter()
        checks = [
            {"name": "constraint_satisfaction", "passed": all(eligible(selected[r["recipe_id"]], constraints) for r in recommendations), "detail": "食材排除、时间、标签、库存条件均重新核验"},
            {"name": "candidate_membership", "passed": all(r["recipe_id"] in selected for r in recommendations), "detail": "所有推荐 ID 来自本次检索结果"},
            {"name": "evidence_references", "passed": all(set(r["evidence_ids"]).issubset({e["id"] for e in evidence}) for r in recommendations), "detail": "引用对应图谱中的原始事实和来源"},
            {"name": "canonical_details", "passed": all(r["steps"] == selected[r["recipe_id"]].steps and r["ingredients"] == selected[r["recipe_id"]].ingredients and r["minutes"] == selected[r["recipe_id"]].minutes for r in recommendations), "detail": "步骤、食材、用时读取原记录，未由模型编造"},
        ]
        passed = all(check["passed"] for check in checks)
        result["validation"] = {"passed": passed, "checks": checks}
        if not passed:
            raise ValueError("推荐结果未通过独立校验")
        log("validate", "ok", "引用与硬约束校验通过", t)
        result.update(status="ok", recommendations=recommendations, evidence=evidence)
        names = "、".join(r["name"] for r in recommendations)
        mode_label = "模型结构化推荐" if generation["mode"] == "llm" else "图谱事实模板"
        result["answer"] = f"根据图谱记录推荐：{names}。以下食材、用时和步骤均来自所列来源。本次输出方式：{mode_label}。"
        return self._finish(result, started, errors)

    def _finish(self, result: dict, started: float, errors: list[str]) -> dict:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if errors:
            result["generation"]["error"] = "；".join(dict.fromkeys(errors))
        if self.settings.trace_path:
            path = Path(self.settings.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {key: result[key] for key in ["request_id", "status", "constraints", "generation", "trace", "elapsed_ms"]}
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result
