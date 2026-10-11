# RecipeGen LLM API 接入

当前厨神交流使用 **DeepSeek API 路由、重排与证据问答**。本地 E5 是检索编码器，不承担问答，也不通过 API 调用。需要检索时的链路为：三路候选（BM25、图谱先筛 E5、图谱条件）→ 合并核验与 RRF → 图谱证据回读 → 完整有界上下文 → 最多 Top20 候选 API 重排 → 前 `limit` 个食谱完整步骤问答 → 引用与数字校验；普通交流由路由调用直接回答。

## 后端配置

新环境先执行 `cp .env.example .env`，填写自己的 Neo4j 与 DeepSeek 配置，再执行 `chmod 600 .env`。密钥只保存在服务端，不进入前端、Git 或验收报告。已有配置应增量修改，避免覆盖 Neo4j 连接。

当前适配器使用兼容 Chat Completions 的接口：

```dotenv
RECIPEGEN_LLM_PROVIDER=chat_completions
RECIPEGEN_LLM_BASE_URL=https://api.deepseek.com
RECIPEGEN_LLM_MODEL=deepseek-flash
RECIPEGEN_LLM_API_KEY=在本机填写
RECIPEGEN_LLM_TIMEOUT=60
RECIPEGEN_RETRIEVAL_MODE=hybrid
```

`BASE_URL` 是接口前缀，不包含 `/chat/completions`。API 采用固定 JSON 对象、`temperature=0` 和输出上限。当前网页适配器不使用 Ollama 或本地问答模型。修改配置后重启 Web 服务；`/api/system/status` 只披露配置可用性、接口类型和模型名称，不返回密钥。

普通 `/api/search` 和 `grounded` 回答不调用 LLM API。统一首页选择厨神交流后进入 `/?view=chat`（旧 `/compose` 兼容），使用 `/api/chat`，只有提示词输入，没有食材或模型开关。DeepSeek 依据最近对话返回固定动作：直接回答，或改写问题、提取食材条件后执行检索。直接回答成功时一次 API 调用，检索问答成功时路由、重排、问答共三次。聊天合同见 [厨神交流](厨神交流.md)。兼容的 `/api/generate` 中 `GenerateRequest.mode` 仍默认 `grounded`，旧 API 客户端需要明确传 `mode=api`。

## 问答请求与返回

```json
{
  "question": "请找一道用番茄和鸡蛋做的菜，不要牛奶，并依据原始步骤说明做法。",
  "ingredients": ["番茄", "鸡蛋"],
  "excluded_ingredients": ["牛奶"],
  "retrieval_mode": "hybrid",
  "mode": "api",
  "limit": 1
}
```

请求路径为 `POST /api/generate`。未传 `recipe_ids` 时执行三路检索、条件核验和 RRF，再由图谱回读候选原文，构造最多 20 个完整候选的上下文。传入最多 3 个 `recipe_ids` 时仅以这些显式选择的记录作为证据，仍检查排除条件，不重复声称执行 E5 检索。

重排要求返回输入候选 ID 的完整排列；问答使用重排后的前 `limit` 个记录的全部原步骤（`limit=1～3`）。完整上下文超过预算时减少候选数量，不静默截断原文。每个回答段落必须引用提供的 E 证据 ID，数字须来自该段所引步骤；Step 与 Source 身份保持可核对。

重点检查：

- `rerank.status`、`ordered_recipe_ids`、`api_call_attempted`：重排实际是否完成并通过排列校验。
- `qa.status`、`paragraphs`、`generation.api_call_attempted`：问题回答、证据不足或失败的独立状态。
- 顶级 `generation.effective_mode=graphrag_api_evidence_qa`、`api_status`、`llm_called`、`api_calls_attempted`、`fallback`、`errors`。
- `evidence`：原步骤文本、Recipe／Step／Source ID、来源 URL；`validation` 与 `trace`：实际校验及阶段记录。

重排失败保留检索排序并继续问答。问答失败返回原始步骤并明确回退；证据不足返回 `insufficient_evidence` 和“不足以回答”说明，不伪装成已回答。无有效步骤候选时不发 API 请求。未配置时 `pending_api=true`、`llm_called=false`，保留原文组织结果。被拒绝或包含凭据的模型输出不回传客户端。

`validation.passed` 只说明程序的结构、引用与数字检查通过，**不能证明每段事实语义全部正确**。原文与引用仍需核对；食材提及不等于完整配料或饮食安全。

## 模块与验证边界

当前网页使用 `recipegen/web_rag_answer.py` 的 `WebRAGAnswerer`，复用 `recipegen/experimental_qa.py` 的 API 客户端及校验函数。旧 `GraphRAGAnswerer` 保留完整步骤中文转述合同，实际模式为 `graphrag_api_grounded_paraphrase`，不是当前默认 API 问答模块。

当前三路链在开发机器完成一次真实聊天集成验证：DeepSeek 路由、重排、问答三次调用成功，候选排列及引用合同通过，没有回退。公开范围与验证口径见 [公开验收摘要](../reports/发布验收.md)；私密配置和原始输入输出不随源码发布。

32 条原字段自动标签 test 查询的检索 Precision@10 为 78.44%、Recall@10 为 41.65%，对应 DeepSeek 重排前的生产检索。该基准不能证明融合优于单路，也不评估模型重排后的语义效果。**真实 API 集成与结构、引用、数字检查不等于人工问答准确率**。

独立步骤分块 CLI 保留为实验，不替换网页的食谱级 E5 向量。新克隆需要重建自己的图谱、模型、索引和连接后再运行；旧中文转述和本地模型分支不属于当前默认网页链路。
