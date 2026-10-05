"""Public requests for the real test-graph application, separate from demo data."""
from __future__ import annotations

from typing import Literal
import re

from pydantic import Field, field_validator

from .models import StrictModel


class SearchRequest(StrictModel):
    query: str = Field(default="", max_length=500)
    ingredients: list[str] = Field(default_factory=list, max_length=20)
    excluded_ingredients: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=6, ge=1, le=20, strict=True)
    retrieval_mode: Literal["keyword", "semantic", "hybrid"] | None = None

    @field_validator("ingredients", "excluded_ingredients")
    @classmethod
    def ingredient_names(cls, values: list[str]) -> list[str]:
        values = [value.strip() for value in values]
        if any(not value or len(value) > 80 or any(ord(c) < 32 for c in value) for value in values):
            raise ValueError("食材名称必须为 1～80 个可打印字符")
        return list(dict.fromkeys(values))


class GenerateRequest(SearchRequest):
    question: str = Field(min_length=1, max_length=500)
    recipe_ids: list[str] = Field(default_factory=list, max_length=3)
    mode: Literal["grounded", "local", "api"] = "grounded"
    limit: int = Field(default=3, ge=1, le=3, strict=True)

    @field_validator("recipe_ids")
    @classmethod
    def identities(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[\w.:-]{1,120}", value) for value in values) or len(set(values)) != len(values):
            raise ValueError("菜谱 ID 必须合法且不能重复")
        return values
