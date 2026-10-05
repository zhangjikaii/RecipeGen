# RecipeGen LLM API 接入

本地 GraphRAG 检索使用 E5 文本模型和 Neo4j，不需要 LLM API。当前原文回答可运行；后续 API 用于把检索到的原始步骤改写为中文，保留步骤与来源引用，不自由补写配方。

用户只需准备 **接口地址、模型名称、API key**。新克隆先复制 `.env.example` 为 `.env`；密钥填写到本机 `.env`，保留自己的 Neo4j 配置，不写入前端、Git 或验收报告。仓库不包含任何已配置的接口密钥。

支持兼容 Chat Completions 的接口：

```dotenv
RECIPEGEN_LLM_PROVIDER=chat_completions
RECIPEGEN_LLM_BASE_URL=https://your-provider.example/v1
RECIPEGEN_LLM_MODEL=your-model-name
RECIPEGEN_LLM_API_KEY=在本机填写
RECIPEGEN_LLM_TIMEOUT=60
```

`BASE_URL` 填接口前缀，不包含 `/chat/completions`。当前适配器使用固定结构化 JSON 请求，实际供应商的模型兼容性、中文改写和引用输出须在配置后联调。填写完成后重启 Web 服务；`/api/system/status` 只披露是否配置、模型名称和接口类型，不返回密钥。页面 API 开关就绪后，选择该模式才发起生成请求，普通检索和 grounded 回答不会调用 API。

本地 Ollama 可用 `RECIPEGEN_LLM_PROVIDER=ollama`，地址例如 `http://127.0.0.1:11434`（不包含 `/api/chat`），填入已经下载的模型名称。该选项不自动下载模型。

```json
{
  "question": "帮我用中文整理番茄相关食谱的做法",
  "ingredients": ["番茄"],
  "retrieval_mode": "hybrid",
  "mode": "api",
  "limit": 1
}
```

请求路径为 `POST /api/generate`。返回 `generation.llm_called`、`effective_mode`、`pending_api`、`api_status`、`fallback` 和校验结果。未配置时标记待接入，返回原始步骤；超时、未知 ID、步骤缺失、引用归属错误或未经证据支持的数字会回退，并披露原因。被拒绝的模型响应不回传客户端。

程序检查结构、步骤完整性、引用和数字字面量，不能证明中文改写的事实语义全部正确。API key 提供后，还需要真实请求、失败回退和人工抽查的独立验收。当前模拟接口测试不等于真实供应商联调。
