# 测试集图谱的离线文本抽取

## 范围与来源

本轮目标是从用户指定的 **`test.zip` 与 `test-video.zip`** 重建测试集图谱；不把训练集或验证集混入此范围。归档读取、成员映射和图谱写入由数据构建模块负责，`recipegen/kg_extract.py` 只接收**真实原始英文步骤文本**，不读取图像或视频生成描述。

实现参考 Obsidian《RecipeGen 多模态知识图谱构建教程：从文本、图像、视频到 Neo4j》中的词表匹配、食材规范化、动作/工具抽取和步骤时间/温度模式。教程还描述了依存分析、LLM、视觉对象、视频动作和跨模态对齐等方法；这些描述不代表本模块已实现相应功能。

## 当前实现

| 信息 | 方法 | 保留的证据 |
|---|---|---|
| Ingredient | 335 个可检查的食材概念，含明确英语别名、已知词复数和中文显示名 | 原始步骤、精确命中片段、字符区间、步骤号 |
| Action | 烹饪动作词表及明确的词形变化 | 原词面及其上下文；不是依存句法确定的动宾关系 |
| Tool | 厨具词表；多词优先，部分动词/工具歧义使用局部规则 | 原词面及其上下文 |
| Duration | 数字或范围＋时间单位，以及 `half an hour`、`overnight` 等明确表达 | **原始命中字符串**，不换算成总用时 |
| Temperature | 带 C/F/Celsius/Fahrenheit 的温度表达 | **原始命中字符串**，不猜测缺失单位 |

单个概念可以在不同位置重复出现；每次出现保留独立跨度。食材和工具采用整词/整短语匹配，不把任意名词都当作食材，也不把 `blend` 从 `blender` 内部截出。较长词面优先，例如 `coconut milk` 不重复生成 `coconut` 和 `milk`；`rice cooker`、`garlic press`、`stock pot` 的工具跨度阻止其中的食材词被误计。

食材规范化只根据可检查的词典进行，例如 `tomatoes → tomato`、`scallions → green onion`。未知词不通过任意去尾缀方式强行归并。`cherry tomato` 等子类保留为独立概念；中文显示名用于阅读，不是机器翻译后的独立来源事实。

## 接口与输出

```python
from recipegen.kg_extract import extract_steps

mentions = extract_steps(
    ["Chop tomatoes. Bake at 180°C for 20-25 minutes."],
    source={"archive": "test.zip", "member": "真实原成员路径", "recipe_id": "真实ID"},
)
```

代表性输出：

```json
{
  "kind": "Ingredient",
  "name": "tomatoes",
  "normalized_name": "tomato",
  "zh_name": "番茄",
  "step_order": 1,
  "evidence": {
    "text": "Chop tomatoes. Bake at 180°C for 20-25 minutes.",
    "span_start": 5,
    "span_end": 13
  },
  "method": "rule",
  "confidence": null,
  "source": {
    "archive": "test.zip",
    "member": "真实原成员路径",
    "recipe_id": "真实ID"
  }
}
```

`evidence.text` 是完整原步骤，`name` 是精确原始命中片段。跨度使用 Python 字符索引：

```python
text[span_start:span_end] == name
```

它不是 UTF-8 字节偏移。`confidence=null` 表示未标定抽取准确率，不能赋一个高分冒充模型置信度。输入的原始来源元数据原样附加，调用方需确保成员身份来自真正读取的归档，不填写猜测路径。

`Duration` 和 `Temperature` 的 `name` 用于 Step 属性，保持原始大小写、符号、单位及范围；`normalized_name` 只做小写和空白整理。例如 `20-25 minutes` 仍是范围原文，`overnight` 不会变成某个小时数。**不把多个步骤时间相加，不写入推算的 Recipe 总时间。**

长任务可创建 `RuleExtractor()` 复用匹配器；默认便捷函数也缓存编译后的词典。可传入字符串步骤数组，或带 `text`、`order`、可选 `source` 的步骤对象。缺少原始文本、无效步骤号或歧义词典会显式报错。

## 外部词典

可针对实际测试集发现的词汇补充一个 JSON 文件：

```json
{
  "ingredients": [
    {"normalized_name": "soursop", "zh_name": "刺果番荔枝", "aliases": ["guanabana"]}
  ],
  "actions": [],
  "tools": []
}
```

```python
from recipegen.kg_extract import RuleExtractor, load_lexicon

extractor = RuleExtractor(load_lexicon("your_lexicon.json"))
mentions = extractor.extract_step("Mix guanabana and tomatoes.", 1)
```

默认扩充内置词典；`include_defaults=False` 可使用纯自定义词典。词条按字面匹配，别名不当正则执行；同一个词面映射到两个不同概念会拒绝，避免静默选择。修订词典后应保存词典文件、版本或哈希，并重新生成对应抽取结果；不同词典产生的图谱规模不能混报。

## 必须保留的边界

1. **步骤中抽到食材，不等于拿到了完整配料表。** 原数据若没有独立配料表，只能标注 `inferred_from_steps` 等来源说明及“配料完整性未确认”。漏掉的食材、数量、调料不能补假值。不能用这些结果宣称“已有库存一定足够”。
2. 规则结果是**文本提及**，尚未经过语法、否定和真实用量核验。`do not add salt` 仍可能包含 `add` 与 `salt` 的词面提及；它不能被解释为确定执行了加盐。以 `USES_INGREDIENT` 或 `USES_TOOL` 存储时，应附规则与原文本证据并说明这是候选关联。
3. 词表覆盖有限，复数规则只覆盖已知概念；专有食材、拼写变体及非英文文本可能漏检。中文显示名不能消除英文概念本身的歧义。
4. 规则词面命中不证明抽取准确率。要评价质量，应人工标注独立步骤样本，分别统计 Ingredient/Action/Tool/时间/温度的准确与漏检，保留错误案例。
5. 图像和视频在这一阶段可保存路径或成员关联；**没有自动 caption、物体识别、视频动作识别或语义对齐结果**。文件名关联和实体共同出现不能包装成视觉模型识别。
6. 完整遍历指定测试集与抽取字段完整/准确是不同指标。运行记录应分别报告实际读取菜谱数、媒体成员数、抽取提及数、来源可追溯检查和未识别/缺失情况。

## 已执行的本模块验证

```bash
python -m pytest tests/test_kg_extract.py -q
```

离线单元验证 **18 项通过，4 个子测试通过**，覆盖跨度还原、复数与别名、多词优先、工具中的食材词、词内误匹配、动作词形、名词短语歧义、多时间原文、温度原文、重复提及、未知词不编造、来源复制、外部词典和别名冲突。

这组测试证明指定规则契约，**不是测试集抽取准确率**。真实 `test.zip` / `test-video.zip` 的读取数量与构图结果，应引用数据构建任务实际生成的清单和报告，而不是用单元测试数代替。
