# RecipeGen 文本 GraphRAG 设计

**源码发布说明：**下文的 5,898 条索引与检查结果是 2026-10-04 开发机器的历史记录。仓库不附原始图谱、向量或模型缓存；新机器先按 [测试集图谱构建](测试集图谱构建.md)导入，再按 [README](../README.md)准备 E5 并重建索引。由本机生成的 manifest 和 `/api/system/status` 才能说明自己的实际状态。公开结果见 [发布验收摘要](../reports/发布验收.md)，节点与关系见 [图谱 Schema](图谱Schema.md)。

本轮扩展食谱检索与有证据的回答：**文本召回 → 固定图扩展 → 原文证据 → 回答与引用校验**。图片、视频处理保持停止；页面只读取已有的图像观察候选。LLM API 尚待用户提供配置，前端不采集密钥。

## 数据与索引

数据来自 `RUOXUAN123/RecipeGen` 固定 revision `2506260d8cb193ecdb18ac31fc725ffec43f6602` 的 `test.zip`、`test-video.zip`。现有图谱保留 5,898 条食谱记录和 44,802 条原始步骤，食谱未去重。文本索引以这些图谱记录为来源；实际索引名称、记录数、模型 revision、维度及是否就绪，以后端状态和构建报告为准，不能把原始步骤数直接当作已成功写入的向量数。

索引绑定 `build_id`、数据集 revision、split、原始 Recipe ID 与对应 Source 身份。当前代码选用本地 `intfloat/multilingual-e5-small` 文本编码器，固定模型 revision 为 `614241f622f53c4eeff9890bdc4f31cfecc418b3`，输出 384 维向量；导入、查询与模型身份必须一致。索引失效、缺失或与当前构建不一致时，应披露不可用或关键词回退，不能伪称语义召回成功。

**当前每条食谱构建一个向量**，输入为标题和全部有序步骤，加上 `passage: ` 前缀；问题使用 `query: ` 前缀。编码器最多接收 512 个 token，超长食谱在编码时截断，并保存实际 token 数与 `truncated` 标记。attention-mask 平均池化后做 L2 归一化，Neo4j 索引用余弦相似度。完整原文仍保存在图谱，召回 Recipe ID 后展开的步骤不会按 512 token 截断。本轮尚未实现步骤分块向量或图像／视频向量检索。[模型说明](https://huggingface.co/intfloat/multilingual-e5-small)

`scripts/build_text_index.py` 从已验证 Neo4j 读取完整文本，将语料与向量检查点保存到 `data/graphrag/full/`，导入派生的 `RecipeGenText:TextEmbedding` 节点，通过 `TEXT_OF` 连接原始食谱。派生节点保留模型 revision、全文 SHA256、构建与数据集身份，不改写原始 Recipe／Step／Source。完成编码、导入、索引 ONLINE 状态及 5,898 条 Recipe 的原生回读后，才写出 `data/graphrag/index-manifest.json` 的 verified 状态。小样本检查不激活全量索引。运行进度见 `reports/graphrag-index-progress.json`。

## 一次请求如何执行

1. 接收问题、希望使用和排除的食材提及，以及 `retrieval_mode`。
2. 通过关键词、语义或混合方式召回有限候选，再映射到真实 Recipe ID。
3. 通过固定、参数化图查询展开 `Recipe → Step / Ingredient / Source`，绑定当前构建与测试范围，读取原始有序步骤、食材提及证据和来源。
4. 根据已记录提及进行过滤，保留候选范围与过滤数量；食材提及可能不完整，不能据此证明库存足够或饮食安全。
5. 为真实图谱条目分配证据 ID，组织回答，核对引用与来源身份。缺少原始步骤的记录不能补写做法。

当前 `recipegen/graphrag_retrieval.py` 调用官方 `neo4j_graphrag.retrievers.VectorCypherRetriever`，向量候选随后通过固定查询展开图上下文。混合路径由项目把关键词与语义排名按 RRF（`K=60`）融合，再由 `RecipeCatalog` 读取和核验完整原始记录。实际路径由每次响应的 retriever、effective_mode 与图谱证据披露。图扩展不让模型自由编写 Cypher。Neo4j 官方区分了向量／全文混合检索、开发者指定查询的图扩展，以及让 LLM 生成查询的 Text2Cypher 路径。[官方 RAG 指南](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html#hybrid-cypher-retrievers)

## 三种检索模式

| 请求模式 | 意图 | 页面必须披露 |
| --- | --- | --- |
| `keyword` | 明确关键词与图谱中已记录的食材提及 | 实际模式、状态、候选及过滤数量 |
| `semantic` | 按文本语义相近程度召回候选 | 索引状态；失败或不可用时的实际行为 |
| `hybrid`（默认） | 结合关键词与语义候选，按 RRF 融合排名 | 实际执行路径、回退原因、候选范围 |

本轮 `returned_candidates` 是有限召回范围内的数量，**不是全库匹配数量**。`coverage=bounded_candidates` 和 `no_match_scope` 用于说明覆盖范围；候选为空或不足不证明整个知识库不存在相关食谱。相似度、融合或排名分数用于排序，不能展示成语义准确率或校准置信度。

## 前后端约定

`POST /api/search` 传入：

```json
{
  "query": "番茄和鸡蛋有哪些做法",
  "ingredients": ["番茄", "鸡蛋"],
  "excluded_ingredients": [],
  "limit": 6,
  "retrieval_mode": "hybrid"
}
```

响应 `retrieval` 的 `mode` 是请求值，`effective_mode` 才是实际模式。页面读取：

- `status`：`ok`、`fallback` 或 `unavailable`；`fallback_reason` 保留实际回退原因。
- `returned_candidates`、`eligible_candidates`、`filtered_candidates`、`candidate_limit`：有限候选数量与上限。
- `method`、`index`、`coverage`、`insufficient_candidates`、`no_match_scope`：实际方法与证据范围。
- 顶级 `trace`（若服务提供）：`stage`、`status`、`elapsed_ms`。缺失字段明确显示未提供，不填成功记录。

`GET /api/system/status` 的 `retrieval` 提供默认模式、关键词／语义可用性、索引状态与文本编码模型。状态显示不替代单次请求的执行证据。

`POST /api/generate` 同样传入 `retrieval_mode`。页面已选 Recipe ID 时生成以选中证据为依据；无选中 ID 的服务端路径可按该模式检索。不能把“已选 ID 的生成”描述为又完成了一次语义召回。

## 生成与 API 配置边界

| 模式 | 行为与披露 |
| --- | --- |
| `grounded` | 程序按真实原始步骤组织回答，提供证据引用，披露未调用模型。 |
| `local` | 保留已有本地模型选择／理由路径，程序保留原步骤；披露实际调用、输出拒绝和回退。 |
| `api` | 后端 API 配置完成后可对已有全部步骤做中文改写，固定 Recipe／Step／证据／Source 引用，并列保留原文；按实际 `effective_mode`、`llm_called`、错误与回退展示结果。 |

`generation.api_configured=false` 时页面显示“API 待配置”并禁用 API 开关。前端没有 API key 输入框、存储或转发逻辑，用户后续通过后端 `.env` 配置。API 与 local 互斥，两者均关闭时采用 grounded。当前 API 答案模块 `recipegen/graphrag_answer.py` 在未配置时保留原文组织结果和 `pending_api`／`api_status=not_configured` 与实际 grounded 模式，保持 `llm_called=false`。成功改写的实际模式为 `graphrag_api_grounded_paraphrase`；字段变化以服务响应为准。

生成回答的事实边界取决于原始证据与后端校验，不能靠界面徽标证明。中文改写的语义可靠性需要单独验收；系统不支持自由创作新配方。结构和来源校验通过不等同于逐条事实准确。官方 GraphRAG 实现也区分检索、提示组织与模型生成三个阶段，项目自己的原文绑定和引用核验需另有实现与证据。[官方 GraphRAG 实现](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/generation/graphrag.html)

## 验收与当前状态

本文件记录实现设计与接口约定。新增语义索引、混合检索与 API 分支的实际验收，分别保存索引构建报告、检索／图扩展报告和浏览器结果；旧版 HTTP／浏览器报告不能替代本轮新功能验证。

2026-10-04 的全量文本索引构建已完成，`data/graphrag/index-manifest.json` 为 verified。原生回读确认 **5,898 个向量与 5,898 条食谱**，索引 `recipegen_text_e5_674e17876d7e` 为 ONLINE，384 维、cosine 相似度，`RecipeGenText.embedding` 为实际索引属性。**95 条食谱**在向量编码时超过 512 token，被标为 truncated；原始完整步骤仍在图谱中。构建检查见 `reports/graphrag-index-verification.json`，进度报告的 `completed_full_scope=true` 仅指本轮文本索引，不代表全量图片／视频识别或语义准确率已通过。

原始标题／步骤中可能含翻译提示、拒绝语句或其他数据质量问题。本轮保留原始来源内容，不把完成向量索引、图谱身份一致或引用校验通过描述为总体食谱质量提升；相关清洗与语义质量评估需要另设样本和指标。

至少检查：固定构建身份及向量数回读、自然语言语义查询、混合候选来源、索引不可用时回退、候选过滤、原始步骤与引用绑定、API 未配置时零外部请求，以及移动端与按钮状态。API 成功调用与中文步骤改写的验收留待用户提供后端配置之后。图片、视频全量识别与语义准确率不在本轮完成范围内。

`scripts/verify_graphrag_browser.cjs --fallback-only` 已在真实本机服务及 Neo4j 上通过 **27 项**浏览器检查：语义索引尚未激活时，hybrid 请求实际回退 keyword，界面显示中文回退说明；5,898 条食谱和 44,802 条步骤与 API 状态一致；API 开关禁用、没有生成请求、没有外部浏览器请求或页面脚本错误。报告为 `reports/graphrag-browser-fallback.json`，不使用 mock 替代实际数据。该报告仅确认回退界面和关键词路径。

索引激活后，`scripts/verify_graphrag_browser.cjs` 已通过 **69 项**真实浏览器检查，报告为 `reports/graphrag-browser-smoke.json`。覆盖三种检索请求与实际模式转发、官方 VectorCypherRetriever 和图扩展 ID 对齐、混合 RRF、原始步骤／Source ID／链接／图节点、已选 Recipe ID 的 grounded 回答与证据引用、API 待配置开关禁用。390px 移动端文档宽度为 390px，没有横向溢出；页面脚本、console 和请求均无错误。该脚本没有发送 API／local 模型生成请求，没有外部浏览器请求，也未恢复图片／视频处理。截图位于 `reports/graphrag-browser-screenshots/`。

后端及来源原生核验另见 `reports/graphrag-system-smoke.json`：**182 项检查**全部通过，包含三种检索路径、食材条件、图扩展与完整源文本 SHA256 回读、grounded 原文回答、API 未配置时 pending 状态和媒体任务停止状态。该报告的 `semantic_accuracy_verified` 与 `quality_or_recall_improvement_verified` 均为 false；这些功能检查不能当作独立准确率评估。真实 LLM API 中文改写的端到端验收仍待用户最后提供配置。
