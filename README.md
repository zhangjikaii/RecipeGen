# RecipeGen：基于知识图谱的食谱检索与生成系统

系统采用 **Neo4j GraphRAG：文本召回 → 图谱扩展 → 带来源引用的回答**。真实测试图谱有 **5,898 条 RecipeGen 食谱记录、44,802 行原始步骤**，已建立 5,898 条本地 E5 文本向量及 ONLINE 索引；支持关键词、语义与 RRF 混合检索，食材提及筛选、详情、来源和子图查看。数据库不可用时明确报错，不切换人工示例。

默认 `grounded` 模式由程序按图谱原始步骤组织回答，不调用大模型。可选 `local` 模式让本地模型选择已有菜谱 ID；`api` 分支已预留结构化中文改写与引用校验，未配置时返回原始步骤并标记 API 待接入。**LLM API 尚未提供，当前不声称真实 API 改写已通过。**

GraphRAG 方法与边界见 [GraphRAG 设计](docs/GraphRAG设计.md)，节点、关系与来源约束见 [图谱 Schema](docs/图谱Schema.md)，后续配置见 [LLM 接入说明](docs/LLM接入说明.md)。原接口说明见 [食谱检索与生成系统](docs/食谱检索与生成系统.md)，简历描述见 [简历项目表述](docs/简历项目表述.md)。

2026-10-04 的开发机器验收记录：**787 项代码测试、182 项真实接口／Neo4j 回读检查、69 项真实网页检查通过**，另有 27 项实际回退界面检查；API 仍待配置。[发布验收摘要](reports/发布验收.md)说明这些历史结果与本次源码发布的范围。完整原始报告、运行日志和本机数据未随仓库发布，相关结果不代表克隆后的机器已经部署成功。

## 仓库包含什么

**这是源码仓库，克隆后不会自带 Neo4j 数据库、真实测试语料、媒体、模型权重、向量检查点或已激活的语义索引。** 仓库保留应用、构建脚本、测试、文档及 20 条人工演示数据。上述规模来自开发机器上的历史构建；新机器必须完成下方重建步骤，再以自己生成的报告和 `/api/system/status` 核对范围。

原始数据从 [RecipeGen 数据集](https://huggingface.co/datasets/RUOXUAN123/RecipeGen)获取，数据集与预训练模型的授权以各来源为准。本次源码发布没有审核或重新授予第三方数据及模型的使用权。

## 已验证的数据与范围

| 内容 | 实际范围 |
|---|---|
| 来源 | `RUOXUAN123/RecipeGen`，只使用 `test.zip`、`test-video.zip` |
| 固定 revision | `2506260d8cb193ecdb18ac31fc725ffec43f6602` |
| 基础图谱 | 5,898 条分来源菜谱记录、44,802 行步骤、296 个规则食材实体 |
| 基础规模 | 159,040 个节点、425,687 条关系 |
| 派生文本索引 | 5,898 个 `RecipeGenText:TextEmbedding` 节点及 `TEXT_OF` 关系；384 维、余弦检索 |
| 媒体清单 | 46,553 张图片、1,512 个视频的真实归档成员定位信息 |
| 原生系统验收 | 状态、番茄/鸡蛋检索、否定食材排除、详情、子图、grounded 生成及步骤引用校验通过 |
| 本地模型功能检查 | 实际调用 MLX Qwen3-VL-2B 4bit，已知 ID 选择、原步骤组装与来源校验通过，未回退 |

记录保留跨归档来源身份，未声称去重为 5,898 道独立菜。2026-10-04 的原生系统检查来自实际 Neo4j 读取；生成为 `grounded`，`llm_called=false`。历史结果见 [发布验收摘要](reports/发布验收.md)，数据方法见 [测试集图谱构建](docs/测试集图谱构建.md)及 [历史 Neo4j 部署记录](docs/用户Neo4j部署.md)。

同日的本地模型检查记录 `llm_called=true`、`fallback=false`、`effective_mode=local_selection_grounded_answer`，结构与来源检查通过。该次只向模型提供文本上下文（`num_images=0`），选择一条已有菜谱；不证明自由生成新配方或语义效果。仓库不包含该本地模型。

**图片与视频批处理均保持停止。** 系统可读取已入库的图像描述候选，不重新下载或识别媒体。验收快照有 3,831 个图像观察节点，覆盖 2,220 张不同图片；同图可留有多次观察。详情按原图像 ID 选择已有观察，排序时间来自公开成功检查点；缺少可核验时间时标注 `latest_unverified`。这些数量不是视觉准确率，也不表示全量媒体已处理。

## 本地启动

在项目目录中使用 Python 3.11 或更新版本。首次准备应用环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 首次重建真实系统

1. 在项目目录复制 `.env.example` 为 `.env`，填写自己 Neo4j 实例的 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`。凭据只保存在本机；建议为本项目准备独立数据库。
2. 按 [测试集图谱构建](docs/测试集图谱构建.md)的步骤 1～4，读取固定 revision 的两个测试归档、生成图谱、离线校验并导入 Neo4j。现有 Neo4j 可使用文档中的导入脚本；macOS ARM64 的独立运行时安装属于可选路径。
3. 需要语义或混合检索时，按下文准备 `.venv-mm` 文本运行时，下载固定 E5 权重，并运行完整 `build_text_index.py --import`。完成原生回读后，脚本在本机生成 `data/graphrag/index-manifest.json`。
4. 启动服务并检查 `/api/system/status`。只建立基础图谱时可以使用 `keyword`；语义索引未就绪时 `hybrid` 明确回退，`semantic` 返回不可用。LLM API 可保持 disabled，原文回答不依赖它。

```bash
cp .env.example .env
# 编辑 .env 后，按测试集图谱构建文档重建并导入数据。
```

首次读取语料和下载权重需要网络，时间与磁盘需求取决于来源及运行环境。默认测试语料脚本读取文本与媒体成员目录，不下载全部图片、视频负载。启动页面不替代数据重建。

先启动已有的真实 Neo4j 数据库，再启动服务：

```bash
source .venv/bin/activate
python -m recipegen serve --host 127.0.0.1 --port 8765
```

打开 [本地界面](http://127.0.0.1:8765/)或 [API 文档](http://127.0.0.1:8765/docs)。`Ctrl+C` 停止服务。启动 Web 服务不会启动媒体批处理。

也可用项目管理脚本启动、查看或停止网页服务：

```bash
python scripts/manage_recipegen_system.py start
python scripts/manage_recipegen_system.py status
python scripts/manage_recipegen_system.py stop
```

默认端口为 `8765`。脚本停止前核对本项目进程的启动命令和工作目录；`status` 的健康检查仅表示网页服务可用，真实图谱状态以 `/api/system/status` 为准。

连接使用 Settings / `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`；历史本机部署也支持私密 `.runtime/active-neo4j.json` 或 `.runtime/local-neo4j.json`，这些文件不在仓库中。私密 JSON 须为权限 `600`，不要提交凭据。目录后端只执行参数化读取，要求唯一 `active+verified` 的 `recipegen-test-v1` 测试构建与固定 revision。

新系统不需要先导入 `data/example_graph.json`；该文件属于旧演示，不能用来修复真实图谱连接错误。

## 系统 API

| 方法与路径 | 用途 |
|---|---|
| `GET /api/system/status` | 真实数据身份、数量、模型可用状态与批处理状态 |
| `POST /api/search` | `keyword` / `semantic` / `hybrid` 召回及食材提及筛选，返回原步骤、来源、检索执行信息 |
| `GET /api/recipes/{id}` | 一条原菜谱详情 |
| `GET /api/system/graph?recipe_id={id}` | 该菜谱的真实节点与关系子图 |
| `POST /api/generate` | 从检索候选或显式菜谱 ID 组织带引用的回答 |

```bash
curl http://127.0.0.1:8765/api/system/status
curl -X POST http://127.0.0.1:8765/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"番茄 鸡蛋","ingredients":["番茄","鸡蛋"],"excluded_ingredients":[],"retrieval_mode":"hybrid","limit":6}'
```

`ingredients` 每项都须匹配已有规则提及，表示“记录提到这些食材”，不表示库存足够。“番茄不要鸡蛋”“不加洋葱”“without eggs”等否定片段合并到排除条件，显示在 `applied_filters` 中。中文别名映射为规范英文食材；英文短菜名保留标题关键词。

`query` 最多 500 字符，生成 `question` 为 1～500 字符；食材与排除列表各最多 20 项，每项 1～80 个可打印字符。搜索 `limit` 为 1～20，生成 `limit` 为 1～3；不符合请求合同返回 `422`。

`retrieval_mode` 可选三种模式，省略时采用后端 `RECIPEGEN_RETRIEVAL_MODE`（默认 hybrid）。混合检索结合最多 20 条关键词候选与 50 条向量候选，按 RRF（K=60）排序；原始图事实和来源需回读校验。响应 `retrieval.effective_mode`、`fallback_reason`、候选数量和 `graph_evidence` 表示实际执行路径。语义不可用时 hybrid 可明确回退 keyword，单独 semantic 请求返回不可用错误。有限候选为空不表示全库无匹配。

默认生成：

```bash
curl -X POST http://127.0.0.1:8765/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"用番茄和鸡蛋能做什么？","ingredients":["番茄","鸡蛋"],"mode":"grounded","limit":2}'
```

也可提交检索结果中的 `recipe_ids`，最多 3 条；显式选菜仍检查问题及参数中的排除条件。响应包含 `answer`、`recipes`、`evidence`、`generation`、`validation` 和阶段 `trace`。无证据或没有原步骤时说明无法生成，不补造做法。

`"mode":"local"` 请求本地模型选择，不恢复媒体批处理。模型不可用、超时或返回未知 ID 等情况，会记录错误与 `fallback`，使用已有原步骤组织回答。应核对 `llm_called`、`effective_mode` 和单独的选择校验，不能仅凭最终回答可用认定模型选择通过。

图像描述候选用于详情及子图展示；当前生成模型输入只包含标题、食材提及和步骤摘录，不纳入图像观察。

`"mode":"api"` 通过 GraphRAG 上下文提供全部原步骤及证据 ID，供后续 API 做中文改写。未配置时保持 `pending_api=true`、`llm_called=false`，页面禁用 API 开关。填写后端 `.env` 并重启服务才启用，具体见 [LLM 接入说明](docs/LLM接入说明.md)。

## 方法与限制

```text
问题 / 结构化食材条件
        ↓
中英文别名与否定片段处理
        ↓
关键词召回 / 本地 E5 文本向量召回
        ↓
RRF 混合排序 + 固定 Neo4j 一阶关系扩展
        ↓
原 Recipe / ordered Step / Source / 规则食材提及
        ↓
程序组织 / 可选本地 ID 选择 / 待接入 API 中文改写
        ↓
原步骤组织 + 来源引用 + 结构与证据校验
```

食材来自词典规则抽取，**不是完整配料表**；没有完整结构化用量或总烹饪时长。原步骤已有数字仍保留，不把步骤时间相加或视频时长换算成总用时。提及排除不构成忌口、过敏或饮食安全保证。

视觉描述、对象和对齐关系均保留为模型候选。当前未做语义抽取准确率、推荐准确率或直接 LLM 对照评测。系统采用显式检索与验证工作流，未实现模型自主规划。

本轮每条食谱只构建一个文本向量，标题和有序步骤以 `passage: ` 编码，问题以 `query: ` 编码。E5 最多接收 512 token，**95 条原文超长并截断编码**；全文哈希、token 数和截断标记保留，图谱展开仍提供全部原步骤。原数据存在菜名包含翻译说明或缺失提示的情况；保留原始来源，不把功能检查当作检索质量评测。

文本检查点由本机运行生成在 `data/graphrag/full/`，全量范围和原生回读通过后才激活 `data/graphrag/index-manifest.json`。这些产物不随源码发布；已有自己检查点时可以续跑：

```bash
.venv-mm/bin/python scripts/build_text_index.py --device cpu --batch-size 8 --import
```

该命令只处理 Neo4j 原始文本与派生索引，不下载或识别图片、视频。权重由准备脚本下载到忽略的 `.runtime/text-model-cache`，固定模型版本和运行方法见 GraphRAG 设计。

新机器可单独准备文本运行时，已有 `.venv-mm` 时复用该目录：

```bash
python3.11 -m venv .venv-mm
.venv-mm/bin/python -m pip install -r requirements-text.txt
.venv-mm/bin/python scripts/prepare_text_model.py
.venv-mm/bin/python scripts/build_text_index.py --device cpu --limit 8
.venv-mm/bin/python scripts/build_text_index.py --device cpu --batch-size 8 --import
```

需要先有已验证的原始 Neo4j 测试图谱。小样本命令仅检查编码，不导入或激活完整索引；全量命令可续跑。应用查询使用 CPU，最多一个编码进程，同问题在服务内缓存 32 条，不在磁盘保存用户问题。下载脚本只获取固定 E5 模型文件。

## 旧 20 条人工演示的兼容入口

| 真实系统 | 旧人工演示 |
|---|---|
| `/api/system/status`、`/api/search`、`/api/recipes/{id}`、`/api/system/graph`、`/api/generate` | `/api/status`、`/api/graph`、`/api/recommend` |
| 真实测试图谱，Neo4j `RecipeGen` 命名空间 | 20 条手工菜谱，SQLite 或 `RG*` 演示图谱 |
| 食材提及筛选，完整配方与时长未知 | 手工完整字段可演示库存、时间、辣味、缺料过滤 |

旧 CLI 仍用于兼容演示：

```bash
python -m recipegen import data/example_graph.json --db var/recipegen.sqlite3
python -m recipegen demo --question '番茄和鸡蛋，20分钟内，不吃辣'
python -m recipegen evaluate --output reports/evaluation.json
```

早期 50 条手工功能回归只证明指定演示用例通过，不是新系统真实数据或模型准确率。上述 `evaluate` 命令在本机重新生成报告，历史范围见 [发布验收摘要](reports/发布验收.md)。

## 代码与证据

| 文件 | 职责 |
|---|---|
| `recipegen/catalog.py` | 只读目录、别名/否定处理、详情与子图 |
| `recipegen/system_generation.py` | 原记录证据组织、可选本地选择、引用与输出校验 |
| `recipegen/graphrag_retrieval.py` | 官方 VectorCypherRetriever 图扩展、RRF、固定范围与全文哈希核验 |
| `recipegen/graphrag_answer.py` | 待接入 API 的结构化改写、步骤/引用/数字校验与回退 |
| `recipegen/text_embeddings.py`、`scripts/local_text_embedding.py` | 本地离线 E5 问题编码、模型身份校验及服务内查询缓存 |
| `scripts/build_text_index.py` | 全量文本检查点、派生节点导入、索引与原生回读 |
| `recipegen/system_api_models.py`、`recipegen/app.py` | 请求校验、新系统 API 与兼容接口 |
| `scripts/local_recipe_generation.py` | 本地模型选择进程接口 |
| `scripts/build_test_corpus.py`、`scripts/build_test_graph.py` | 测试文本/归档成员读取与来源图谱构建 |
| `reports/发布验收.md` | 公开源码范围与历史验收摘要 |
| `scripts/manage_recipegen_system.py` | 本项目网页服务的 start/status/stop |

离线合同测试使用 `python -m pytest -q`；与实际数据库、本地模型运行、语义效果评测分别记录。真实接口／浏览器验收还需要自己重建的数据库与文本模型，浏览器检查需安装 Playwright 和对应浏览器，不能将历史浏览器报告当作当前机器结果。历史媒体方法见 [多模态语义处理](docs/多模态语义处理.md)，其中处理命令不表示批处理当前运行。
