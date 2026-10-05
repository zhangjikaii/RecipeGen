# 接入现有 Neo4j

## 已知情况

用户说明原图谱在其他电脑上。本次重写参考了 Obsidian 技术文档《多模态知识图谱构建方法》，没有读到原项目代码或数据库，也没有对真实 Neo4j 执行查询。

文档给出的主要关系是：

```text
(Recipe {recipe_id, title})-[:HAS_INGREDIENT]->(Ingredient {normalized_name / name})
(Recipe)-[:HAS_STEP]->(Step {text, order, duration 等})
Recipe 还可关联 Image / Video
```

文档提到数据来源 `RUOXUAN123/RecipeGen`。这里不把文档中的构想、数据量或流程描述当成用户完成的实测结果。当前重写先支持文字图谱检索；图片、视频及多模态对齐没有迁移或实现。

**步骤时长不是菜谱总时长。** 原文档未给出确定的 `Recipe.minutes`。未知用时保留为 `null`；要求“20 分钟内”的查询将排除未知记录，不生成猜测的时长。

## 路线选择

| 路线 | 做法 | 适用情况 |
|---|---|---|
| 只读导出到本地 | 检查原 schema → 映射导出 JSON → 导入本地 SQLite | 先完成本地演示，原库不方便一直联网 |
| 适配原 schema 直接查询 | 核验标签与属性 → 编写相应固定查询 → 对真实库只读验证 | 希望保留原 Neo4j 为运行后端；需要后续针对实际库适配 |
| 本地演示 Neo4j | 启动空库 → 显式 `seed-neo4j` 写入人工示例 | 只测试项目规范 schema 的 Neo4j 后端 |

目前导出脚本给出了原文档 schema 的参考映射。项目运行中的 Neo4j 后端使用专门的 `RG*` 标签，**并不是已经适配了原库 `Recipe` 标签**。

## 1. 配置连接

在保存原图谱的电脑上复制或下载这个项目并安装依赖；可使用项目 `.env` 或环境变量：

```bash
export NEO4J_URI='bolt://127.0.0.1:7687'
export NEO4J_USER='your-user'
export NEO4J_DATABASE='neo4j'
read -s NEO4J_PASSWORD
export NEO4J_PASSWORD
```

地址填写真实可访问的服务；远程加密连接按服务器要求使用 `neo4j+s://` 或 `bolt+s://`。不要把密码发到聊天、提交到 Git 或写进命令参数。独立导出脚本只读取环境变量，不自动读取 `.env`。

驱动的 READ 路由及 `execute_read` 用于读事务；脚本仅含固定读取语句、不接受任意 Cypher。使用数据库授予的只读账号还能在服务端限制权限。参考 [Neo4j 官方 Python 事务说明](https://neo4j.com/docs/python-manual/current/transactions/)。

## 2. 只读检查结构

```bash
python scripts/export_neo4j.py --inspect
```

输出 labels、关系类型以及每类节点的少量属性键，不输出凭据，不修改图谱。之后还需要核对关系方向、字段类型、步骤顺序和缺失值。

至少核对下面几项：

- `Recipe.recipe_id` 是否唯一，`title` 是否有值。
- `HAS_INGREDIENT` 是否为 Recipe → Ingredient，食材名是 `normalized_name` 还是 `name`。
- `HAS_STEP` 是否指向 Step，步骤是否具有文本及唯一数值顺序。
- 是否真实记录菜谱总用时，以及单位是不是分钟。
- 食材记录是否完整，是否区分调味料，标签是否具有明确的含义与覆盖率。
- 来源是否能追溯回原菜谱或原始页面。

## 3. 核对映射后导出

将 `docs/neo4j_mapping.example.json` 复制为你自己的映射文件，核对后运行：

```bash
python scripts/export_neo4j.py --mapping your_mapping.json --output exports/real_graph.json --limit 20
```

先导出少量记录手工核验，再去掉 `--limit` 导出全部。已有输出文件会被拒绝覆盖。此导出是本地文件创建，原数据库不发生写入。

映射项说明：

| 映射 | 用途 |
|---|---|
| `recipe_label` | 原菜谱标签；示例使用文档中的 `Recipe` |
| `fields.*.property` | 从 Recipe 属性读取字段 |
| `fields.*.relationship`、`direction`、`target_label` | 明确指定关联节点及方向，读取列表字段 |
| `property_fallbacks` | 按显式顺序读取真实属性；示例先 `normalized_name`，再 `name` |
| `steps.order_property`、`order_on` | 规定节点或关系上的步骤顺序，不猜测顺序 |
| `minutes.unrecorded` | 原图谱无总用时时设为 `true`，输出 `null` |
| `seasonings.unrecorded`、`tags.unrecorded` | 没有单独记录这些字段时输出空列表，表示没有该类图谱证据 |
| `source.internal_record`、`recipe_id_property` | 使用原 `recipe_id` 作为内部来源标识，不编造公开链接 |
| `dataset` | 填写真正的数据来源、版本、构建方式与授权情况 |

所有 `HAS_INGREDIENT` 食材仍保留在 `ingredients`，不会因为没有单独调味料关系而删掉盐或辣椒。空标签表示未记录，不能据此声称“确定不辣”“确定素食”。需要这类约束时应补充真实标注或采用保守检索。

若原图谱已有来源字段，`source` 可改成：

```json
{
  "id_property": "source_id",
  "title_property": "source_title",
  "url_property": "source_url"
}
```

属性名必须来自真实记录；无 URL 时将 `url_property` 设为 `null`，保留内部来源。不要把“技术文档提到某数据集”自动变成每条菜谱的原网页出处。

导出的规范格式如下，数值和内容仅作字段示意：

```json
{
  "schema_version": 1,
  "dataset": {
    "id": "your-data-id",
    "name": "真实图谱名称",
    "is_demo": false,
    "description": "实际构建方式与版本",
    "source": "真实来源",
    "license": "实际授权情况"
  },
  "ingredient_aliases": {"西红柿": "番茄"},
  "recipes": [
    {
      "id": "原 recipe_id",
      "name": "原菜谱标题",
      "ingredients": ["番茄", "鸡蛋"],
      "seasonings": [],
      "minutes": null,
      "tags": [],
      "steps": ["原步骤一", "原步骤二"],
      "source": {"id": "neo4j:原 recipe_id", "title": "原图谱记录 原 recipe_id", "url": null}
    }
  ]
}
```

缺少 ID、标题、食材或非空有序步骤会报错；重复 ID、无效时长和无效数组也会拒绝。请补齐真实资料或调整明确的映射，不要写假值让校验通过。

## 4. 导入并验证

```bash
python -m recipegen import exports/real_graph.json --db var/recipegen.sqlite3
python -m recipegen demo --question '用番茄和鸡蛋做菜'
python -m recipegen demo --question '番茄和鸡蛋，20分钟内'
```

第二个问题应排除 `minutes=null` 的菜谱。检查原菜谱内容是否保持、证据来源是否对应，避免把人工示例残留当成真实数据结果。之后针对真实图谱建立独立测试用例，内置示例评测不自动成为真实数据评测。

## 项目 Neo4j 演示 schema

```text
RGRecipe {id,name,minutes}
  ├─USES_INGREDIENT→ RGIngredient {name}
  ├─USES_SEASONING → RGIngredient {name}
  ├─HAS_TAG         → RGTag {name}
  ├─HAS_STEP        → RGStep {id,order,text}
  └─HAS_SOURCE      → RGSource {id,title,url}
```

只有显式运行 `seed-neo4j` 才会写入这些记录。Compose 使用一个独立本地库；参见 README。不能用这个写入演示的成功替代原图谱读取与适配验证。
