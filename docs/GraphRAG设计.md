# RecipeGen 三路检索与图谱问答

网页默认使用 `hybrid`：**BM25 关键词候选＋图谱先筛后的 E5 语义候选＋Cypher 图谱条件候选 → 合并与条件核验 → BM25/E5 双排名 RRF → 一阶图谱证据回读 → 有界上下文 → DeepSeek API 重排 → 问答与引用校验**。食谱探索返回检索食谱，不调用 LLM；厨神交流由 DeepSeek 先判断是否需要检索，普通交流直接回答，需要食谱证据时执行完整链路。

## 数据与模型

固定来源为 `RUOXUAN123/RecipeGen` 的 `test.zip`、`test-video.zip`，revision `2506260d8cb193ecdb18ac31fc725ffec43f6602`。复用 **5,898 条分来源食谱记录、44,802 行原步骤**的已验证 Neo4j 构建，不重建图谱、不改变 Recipe／Step／Source。不同归档身份保留，不声称食谱已经去重。图片与视频处理保持停止。

检索编码器为本地 `intfloat/multilingual-e5-small`，revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`；不是 Embedding API。食谱输入为 `passage: `＋标题＋全部有序步骤，问题输入为 `query: `＋问题；512-token 编码上限、attention-mask 平均池化、L2 归一化、384 维。95 条历史食谱向量有编码截断标记，图谱及问答仍使用完整原步骤。

Neo4j 的 `RecipeGenText:TextEmbedding.embedding` 经 `TEXT_OF` 连接 Recipe；本地检查点为 `data/graphrag/full/corpus.jsonl`、`embeddings.jsonl`、`manifest.json`，激活清单为 `data/graphrag/index-manifest.json`。语料的 `corpus_sha256` 使用原索引构建器对 `{recipe_id,text_sha256}` 有序数组的 canonical JSON fingerprint；文件字节哈希与此身份哈希分别记录。

最终问答使用已有 **DeepSeek API**。E5 仅用于检索，LLM 不提供自由 Cypher、不改图谱，也不能增加未召回的食谱 ID。

## 请求执行流程

1. **解析输入。** 探索传所需／排除食材；聊天传问题与最近对话，由 DeepSeek 返回经过 JSON 校验的 `respond`／`retrieve`。检索分支输出独立问题与正负食材数组。聊天设置 `infer_query_filters=false`，避免把可选食材的字面提及当作必需条件。
2. **关键词候选 A。** 基于原标题与全部步骤建立本地 Okapi BM25，`k1=1.5`、`b=0.75`；英文词与中文单字分词，NFKC、小写。有正向食材时，词面输入使用生产别名的英文展开；无正向食材时使用问题词面。排除食材不加入正向别名加分。取全库正分 Top50。语料按已激活 manifest 与每条原文哈希校验后缓存。
3. **图谱候选 C。** 固定参数化只读 Cypher 查询同一构建的 `HAS_INGREDIENT`：单项别名 OR、多项必需食材 AND，排除食材做集合差。返回全部合格 Recipe 与对应 Text 向量；不在此阶段限制 TopK。构建、revision、split、向量模型、signature、recipe/text 身份和覆盖数量必须一致。
4. **语义候选 B。** 本地 E5 编码完整问题；只在 C 的 live Recipe 向量矩阵进行 float32 精确余弦评分，按分数与稳定 ID 取 Top50。没有全库向量 Top50 后再过滤，没有 ANN。hybrid 分数为原 L2 余弦（约 −1～1），不等于语义准确率。
5. **合并与融合。** A、B、无序 C 并集后核验硬条件，非法候选不能通过词面高分进入结果。C 全集参与，因此当前最终池受图条件约束；BM25 可以改变排序，并不扩大合法池。对完整合法池重算 BM25 正分排名和 E5 排名，`RRF=1/(60+rank_BM25)+1/(60+rank_E5)`；零 BM25 没有词面 rank，图谱没有第三个 rank 或人为奖励。
6. **图谱证据回读。** RRF 排序后才选取内部最多 50 条候选，固定一阶查询回读其 Ingredient、全部 ordered Step 与 Source；完整详情再次核对原文 SHA256、步骤 ID、来源归属及构建身份。不是通用多跳扩展，不引入新食谱。搜索最终返回请求的 1～20 条。
7. **有界上下文与 API 重排。** 问答最多提供前 20 条候选的完整原步骤，预算为 60,000 个 JSON 字符。超过预算时减少候选前缀并披露遗漏数，不静默截断步骤。DeepSeek 必须返回输入候选 ID 的完整排列。
8. **API 问答与校验。** 从重排结果选前 `limit` 条（1～3），给完整原步骤分配 E 证据 ID，构建问答上下文。DeepSeek 用中文段落回答，每段引用给定步骤；验证 JSON、引用 ID/归属及数字字面量，并保留 Step、Source 与原文供核对。重排失败保留检索排序；问答失败明确回退原步骤，证据不足返回 `insufficient_evidence`。

```text
问题 → DeepSeek 路由／条件解析（探索入口直接提供条件）
  ├─ BM25 关键词 → A
  └─ Cypher 图谱条件 → 全部 C
       └─ C 内 E5 排序 → B
A ∪ B ∪ C → 硬条件核验 → BM25/E5 双排名 RRF
  → Top 候选一阶图证据 → 完整有界上下文
  → DeepSeek 重排 → 选定食谱完整上下文
  → DeepSeek 问答 → 引用、身份与数字校验
```

## 模式与阶段信息

| 模式 | 行为 |
|---|---|
| `hybrid`（默认） | 三路候选、双排名 RRF、融合后图证据回读；不可用时回退必须明确披露。 |
| `semantic` | 保留原图谱先筛全部候选、Neo4j 精确余弦排序与证据回读。 |
| `keyword` | 保留目录的词典／标题匹配，不调用 E5 或 LLM。 |

生产 hybrid 的 `method=graph_first_bm25_e5_rrf`、`rrf_k=60`。`seed_ids` 区分 BM25／dense／graph；`candidate_pool`、`candidate_counts` 与 `candidate_rankings` 保留候选和双排名。`graph_candidate_count=scored_recipe_count` 证明完整 C 被评分；不代表所有候选都会返回。`approximate_search=false`、`index_used_for_ann=false`。

`trace` 串联 `bm25_retrieval`、`graph_condition_retrieval`、`graph_prefiltered_e5`、`candidate_condition_check`、`rrf_fusion`、`graph_evidence_expansion`、`bounded_api_context`、`api_rerank`、`api_qa_context`、`api_question_answering` 与 `api_answer_validation`。上下文记录预算单位、实际字符数、遗漏候选及哈希；输出 token 上限与输入字符预算不同。

食材关系是原文规则提及，不是完整配料表。图谱条件排除不证明实际不含某食材。引用和数字校验属于工程检查，不自动证明回答的全部事实语义正确。

## 验证与复现

新生产检索使用 `scripts/verify_serial_hybrid_retrieval.py --prepare` 冻结协议和代码/数据哈希，再运行 `--run`。复用原 40 条条件回归查询，8 dev 单列、32 test；原 E5 查询向量与原字面自动 qrels 不变。实际连接 Neo4j，使用生产 hybrid，不调用 LLM、不调参，指标在 DeepSeek 重排前。P@10 分母固定为 10，R@10 分母为原 qrels 全部相关记录；按查询宏平均。查询结果曾被查看，不作为未知查询泛化或人工语义准确率。

真实聊天链的 DeepSeek 路由、重排、问答分别核对，成功需要三次实际调用、有效候选排列与可回读引用，不能只看 HTTP 200。验收报告和历史结果按版本保存，不叠加用例数量或把旧阶段结果当作新验证。

真实 Neo4j 消融核查确认 RRF 改变顺序和 Top10 集合，但现有条件标签下的 Precision@10、Recall@10 与 E5 相同。完整图条件池决定合法范围，26/32 条 test 查询的全部候选都被标签判为相关；现有基准不能证明融合收益，需要做法、偏好相关性及难负例才能评价排序质量。

公开执行范围与原始报告哈希见 [公开验收摘要](../reports/发布验收.md)。原始冻结查询、标签、向量和运行报告不随源码发布，上述回归脚本依赖这些本地实验产物；新环境需重建数据并建立独立评测，不能将克隆源码描述为复现了原分数。

步骤分块 `scripts/query_experimental_rag.py` 仍是独立 CLI 实验；网页继续使用原食谱级向量。历史中文转述与本地模型分支不属于默认网页问答。
