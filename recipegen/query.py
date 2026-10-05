from __future__ import annotations

import re

from .models import ConstraintOverrides, Constraints, GraphDocument

CHINESE_DIGITS = dict(zip("零一二三四五六七八九", range(10)))
CHINESE_DIGITS["两"] = 2


def parse_number(text: str) -> int:
    if text.isdigit():
        return int(text)
    if "百" in text:
        head, tail = text.split("百", 1)
        return CHINESE_DIGITS.get(head, 1) * 100 + (parse_number(tail.lstrip("零")) if tail.lstrip("零") else 0)
    if "十" in text:
        head, tail = text.split("十", 1)
        return CHINESE_DIGITS.get(head, 1) * 10 + CHINESE_DIGITS.get(tail, 0)
    return CHINESE_DIGITS.get(text, 0)


class QueryParser:
    """Transparent Chinese rule baseline; never presented as an LLM."""

    def __init__(self, document: GraphDocument):
        self.document = document
        self.aliases = {name: name for r in document.recipes for name in r.ingredients + r.seasonings}
        self.aliases.update(document.ingredient_aliases)
        self.words = sorted(self.aliases, key=len, reverse=True)
        self.tags = {tag for r in document.recipes for tag in r.tags}

    def normalize(self, name: str) -> str:
        return self.aliases.get(name.strip(), name.strip())

    def names_in(self, text: str) -> list[str]:
        matches = []
        occupied: set[int] = set()
        for word in self.words:
            for match in re.finditer(re.escape(word), text, re.IGNORECASE):
                positions = set(range(match.start(), match.end()))
                if not positions.intersection(occupied):
                    matches.append((match.start(), self.aliases[word]))
                    occupied.update(positions)
        return list(dict.fromkeys(name for _, name in sorted(matches)))

    def parse(self, question: str) -> Constraints:
        excluded: list[str] = []
        positive = re.sub(r"(?:我)?没有(?:现成)?(?:主料|食材|库存)", " ", question)
        pattern = r"(?:不能吃|不能用|不能放|不能加|不吃|不放|不加|不要|不用|排除|去掉|忌口)[^，,。；;！？!?]*(?=[，,。；;！？!?]|$)"
        for match in re.finditer(pattern, question):
            excluded.extend(self.names_in(match.group()))
            positive = positive.replace(match.group(), " ")

        unavailable = []
        for match in re.finditer(r"(?:没有|还没买|没买|缺了|缺少)([^，,。；;！？!?]+)", positive):
            unavailable.extend(self.names_in(match.group()))
        for word in self.words:
            if re.search(re.escape(word) + r"(?:还)?(?:没买|没有|未买|没了)", positive):
                unavailable.append(self.aliases[word])

        # Explicit ownership clauses override ingredient mentions in wishes,
        # dish names or questions. '我有番茄，想用鸡蛋' owns only tomato.
        inventories = list(re.finditer(
            r"(?:我(?:现在)?有|家里有|现有|手头有|只有|库存(?:是|有)|食材(?:是|有)|冰箱里有|还有|(?:^|[，,。；;])有)([^，,。；;！？!?]+)",
            positive,
        ))
        if inventories:
            available = []
            for inventory in inventories:
                raw = re.split(r"(?:但是|不过|可是|但|想|希望|打算|能|可以|做什么|做哪些|做菜|推荐|请|不能|没有|还没|[0-9零一二两三四五六七八九十百]+\s*(?:分钟|小时))", inventory.group(1))[0]
                for part in re.split(r"[、和与及,，\s]+", raw):
                    part = part.strip()
                    known = self.names_in(part)
                    if known:
                        available.extend(known)
                    elif part and len(part) <= 30:
                        available.append(part)
        else:
            available = self.names_in(positive)
        available = [name for name in available if name not in unavailable]

        limits = []
        for match in re.finditer(r"([0-9零一二两三四五六七八九十百]+)\s*(分钟|分|min(?:utes?)?|小时|hour(?:s?)?)", question, re.IGNORECASE):
            value = parse_number(match.group(1))
            if match.group(2).lower() in {"小时", "hour", "hours"}:
                value *= 60
            if 0 < value <= 1440:
                limits.append(value)
        if "半小时" in question:
            limits.append(30)
        required_tags = []
        if any(word in question for word in ["不辣", "不吃辣", "不要辣", "不能吃辣", "不放辣", "不加辣", "清淡"]):
            required_tags.append("不辣")
        elif any(word in positive for word in ["辣一点", "要辣", "吃辣", "辣的", "辣菜"]):
            required_tags.append("辣")
        if any(word in question for word in ["素食", "吃素", "素菜"]):
            required_tags.append("素食")
        for tag in sorted(self.tags - {"辣", "不辣", "素食"}):
            if tag in positive:
                required_tags.append(tag)
        allow_missing = any(word in question for word in ["允许缺料", "允许缺少食材", "可以缺料", "可缺料", "可以买", "可以买", "可以补买", "采购清单", "不限制食材"])
        recipe_name = next((r.name for r in self.document.recipes if r.name in question and any(word in question for word in ["怎么做", "做法", "步骤", "查询"])), None)
        # In direct dish lookup, ingredients mentioned only inside the dish name
        # do not establish a user inventory.
        if recipe_name and not re.search(r"我有|家里有|现有|手头有|只有|食材有|冰箱里有", positive):
            available = self.names_in(positive.replace(recipe_name, ""))
        return Constraints(
            available_ingredients=list(dict.fromkeys(available)),
            excluded_ingredients=list(dict.fromkeys(excluded)),
            max_minutes=min(limits) if limits else None,
            required_tags=list(dict.fromkeys(required_tags)),
            allow_missing=allow_missing,
            recipe_name=recipe_name,
        )

    def apply_overrides(self, base: Constraints, overrides: ConstraintOverrides | None) -> Constraints:
        values = base.model_dump()
        if overrides is not None:
            values.update(overrides.model_dump(exclude_unset=True))
        for field in ["available_ingredients", "excluded_ingredients"]:
            values[field] = list(dict.fromkeys(self.normalize(name) for name in (values[field] or [])))
        values["required_tags"] = values["required_tags"] or []
        return Constraints.model_validate(values)

    def merge_llm(self, baseline: Constraints, model: Constraints, question: str) -> Constraints:
        # A model cannot introduce ingredients the user did not mention.
        for name in model.available_ingredients + model.excluded_ingredients:
            names = [word for word, canonical in self.aliases.items() if canonical == self.normalize(name)]
            if not any(word.lower() in question.lower() for word in names + [name]):
                raise ValueError("模型提取了用户未提及的食材")
        if model.recipe_name and model.recipe_name not in question:
            raise ValueError("模型提取了用户未指定的菜谱")
        values = model.model_dump()
        # High-confidence baseline restrictions remain binding. LLM may refine,
        # but may not remove exclusions or enlarge a recognized time limit.
        values["excluded_ingredients"] = list(dict.fromkeys(baseline.excluded_ingredients + model.excluded_ingredients))
        values["required_tags"] = list(dict.fromkeys(baseline.required_tags + model.required_tags))
        if baseline.max_minutes is not None:
            values["max_minutes"] = min(baseline.max_minutes, model.max_minutes) if model.max_minutes is not None else baseline.max_minutes
        if baseline.available_ingredients:
            values["available_ingredients"] = baseline.available_ingredients
        values["allow_missing"] = baseline.allow_missing  # opt-in only, never inferred by model
        values["top_k"] = baseline.top_k
        values["recipe_name"] = baseline.recipe_name or model.recipe_name
        return self.apply_overrides(Constraints.model_validate(values), None)
