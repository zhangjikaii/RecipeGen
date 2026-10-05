from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Source(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    url: str | None = None

    @field_validator("url")
    @classmethod
    def public_url(cls, value: str | None) -> str | None:
        if value:
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(c.isspace() for c in value) or parsed.username or parsed.password:
                raise ValueError("来源 URL 必须为有效且不含凭据的 http(s)，或使用 null 表示内部来源")
        return value


class Dataset(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    is_demo: bool
    description: str = ""
    source: str = ""
    license: str = "unspecified"


class Recipe(StrictModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.:-]+$")
    name: str = Field(min_length=1, max_length=200)
    ingredients: list[str] = Field(min_length=1, max_length=100)
    seasonings: list[str] = Field(default_factory=list, max_length=100)
    minutes: int | None = Field(default=None, gt=0, le=1440, strict=True)
    tags: list[str] = Field(default_factory=list, max_length=50)
    steps: list[str] = Field(min_length=1, max_length=100)
    source: Source
    images: list[str] = Field(default_factory=list, max_length=100)
    videos: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("ingredients", "seasonings", "tags", "steps")
    @classmethod
    def clean_values(cls, values: list[str], info: ValidationInfo) -> list[str]:
        values = [v.strip() for v in values]
        if any(not v or len(v) > 2000 for v in values):
            raise ValueError("列表元素不能为空或超过 2000 字符")
        return values if info.field_name == "steps" else list(dict.fromkeys(values))


class GraphDocument(StrictModel):
    schema_version: Literal[1] = 1
    dataset: Dataset
    ingredient_aliases: dict[str, str] = Field(default_factory=dict)
    recipes: list[Recipe] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def unique_ids(self) -> "GraphDocument":
        ids = [r.id for r in self.recipes]
        if len(set(ids)) != len(ids):
            raise ValueError("菜谱 ID 必须唯一")
        sources: dict[str, Source] = {}
        for recipe in self.recipes:
            if recipe.source.id in sources and sources[recipe.source.id] != recipe.source:
                raise ValueError("同一个来源 ID 对应不同来源信息")
            sources[recipe.source.id] = recipe.source
        known = {x for r in self.recipes for x in r.ingredients + r.seasonings}
        for alias, target in self.ingredient_aliases.items():
            if not alias.strip() or target not in known:
                raise ValueError(f"非法别名映射: {alias} -> {target}")
        return self


class Constraints(StrictModel):
    available_ingredients: list[str] = Field(default_factory=list, max_length=100)
    excluded_ingredients: list[str] = Field(default_factory=list, max_length=100)
    max_minutes: int | None = Field(default=None, gt=0, le=1440, strict=True)
    required_tags: list[str] = Field(default_factory=list, max_length=50)
    allow_missing: bool = False
    top_k: int = Field(default=3, ge=1, le=10, strict=True)
    recipe_name: str | None = Field(default=None, max_length=200)

    @field_validator("available_ingredients", "excluded_ingredients", "required_tags")
    @classmethod
    def clean_names(cls, values: list[str]) -> list[str]:
        values = list(dict.fromkeys(v.strip() for v in values))
        if any(not v or len(v) > 100 for v in values):
            raise ValueError("约束名称不能为空或超过 100 字符")
        return values


class ConstraintOverrides(StrictModel):
    available_ingredients: list[str] | None = None
    excluded_ingredients: list[str] | None = None
    max_minutes: int | None = Field(default=None, gt=0, le=1440, strict=True)
    required_tags: list[str] | None = None
    allow_missing: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=10, strict=True)
    recipe_name: str | None = None


class RecommendRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    constraints: ConstraintOverrides | None = None
    use_llm: bool = False


ReasonCode = Literal["ingredient_match", "within_time", "tag_match", "needs_shopping", "named_recipe"]


class GeneratedChoice(StrictModel):
    recipe_id: str
    reason_codes: list[ReasonCode] = Field(min_length=1, max_length=5)


class GeneratedRecommendations(StrictModel):
    recommendations: list[GeneratedChoice] = Field(min_length=1, max_length=10)
