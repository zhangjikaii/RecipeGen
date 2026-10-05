# RecipeGen 用户 Neo4j 部署记录

部署日期：2026-10-02。本文是开发机器的**历史基础图谱部署记录**，当时已实际连接、导入、回读校验并执行检索，只使用官方测试集的 `test.zip` 和 `test-video.zip`。连接配置、Neo4j 存储和原始运行报告不随源码发布；下面的本机地址不指向在线公共服务。新机器需先按 [测试集图谱构建](测试集图谱构建.md)重建并导入，配置自己的 `.env`，当前系统设计见 [GraphRAG 设计](GraphRAG设计.md)。

| 项目 | 已验证结果 |
|---|---|
| 连接地址 | `neo4j://127.0.0.1:7687` |
| 数据库 / 用户 | `neo4j` / `neo4j` |
| 服务版本 | Neo4j Enterprise 2026.09.0，用户已有的 Neo4j Desktop 实例 |
| 导入前 | 0 个节点，0 条关系 |
| Recipe 记录 | 5,898 条，未跨归档去重 |
| 图谱节点 / 关系 | 159,040 / 425,687 |
| 导入核验 | 7 项全部通过，构建已激活 |
| 实际检索 | `tomato`、`番茄`各返回 3 条；步骤顺序和来源关联通过检查 |
| Browser | <http://127.0.0.1:7474/browser/>，实际 HTTP 200 |

构建 ID：`test-a7e899308295eca8ca01`。图谱公共标签已统一为 `RecipeGen`，构建管理标签已统一为 `RecipeGenBuild`；保留业务标签 `Recipe`、`Step`、`Ingredient` 等，以及 `split=test` 和构建身份。**用户实例的标签迁移已实际完成并核验**：159,040 个图谱节点与 1 个管理节点完成更名，旧标签节点为 0，新标签唯一约束生效；425,687 条关系保留，全库共 159,041 个节点，7 项原生回读检查全部通过。

## 查看图谱

打开 [Neo4j Browser](http://127.0.0.1:7474/browser/)，使用用户提供的连接信息登录，选择数据库 `neo4j`。执行以下查询查看一条含 tomato 标题的食谱及步骤、食材候选、来源：

```cypher
MATCH (r:RecipeGen:Recipe {
  build_id: 'test-a7e899308295eca8ca01', split: 'test'
})
WHERE toLower(coalesce(r.title, '')) CONTAINS 'tomato'
WITH r ORDER BY r.kg_csv_id LIMIT 1
MATCH (r)-[e:HAS_STEP|HAS_INGREDIENT|HAS_SOURCE]->
  (n:RecipeGen {build_id: 'test-a7e899308295eca8ca01', split: 'test'})
WHERE e.build_id = 'test-a7e899308295eca8ca01' AND e.split = 'test'
RETURN r, e, n LIMIT 60;
```

核对食谱记录数：

```cypher
MATCH (r:RecipeGen:Recipe {
  build_id: 'test-a7e899308295eca8ca01', split: 'test'
})
RETURN count(r) AS recipes;
```

## 项目查询

在 `<PROJECT_ROOT>` 目录运行：

```bash
.venv/bin/python scripts/query_test_graph.py --query 番茄 --limit 3
.venv/bin/python scripts/query_test_graph.py --query tomato --limit 3
```

历史机器默认读取 `.runtime/active-neo4j.json`，当时已配置到指定实例；没有该文件时才回退到项目独立实例配置。也支持 `--config .runtime/user-neo4j.json`。显式设置的 `NEO4J_*` 环境变量优先于私密文件。新克隆不包含这些连接文件，需配置自己的 Neo4j 连接。

历史本机 `.env` 也保存了该连接，供导入脚本等使用。连接文件和 `.env` 的权限均为 `600`，被 Git 和源码发布排除。重新导入同一构建会使用版本化 ID 和 MERGE，保留其他构建；完整命令为：

```bash
.venv/bin/python scripts/import_test_graph_neo4j.py \
  --report reports/user-neo4j-deployment.json
```

## 数据位置与证据

- 图谱构建文件：`<PROJECT_ROOT>/data/test_graph/`。
- Neo4j 存储位置由自己的实例管理，例如 `<NEO4J_HOME>/data`；仓库不附数据库存储。
- 公开历史结果见 [发布验收摘要](../reports/发布验收.md)；详细服务目录、预检、导入和查询原始报告不随源码发布。
- 历史 Browser 示例返回 15 条图谱关联，食谱计数为 5,898，全库含管理节点共 159,041 个节点。
- 历史标签迁移状态为 `verified`，新标签中文与英文检索各返回 3 条，步骤顺序与来源完整。

更名只在用户的 `7687` 实例执行；早期项目独立实例 `8767` 保留历史状态。此前 JSON 报告保留当时的标签，作为历史证据。已有旧标签的实例不能直接用新标签重复导入，应先进行有范围限定的标签迁移与回读核验。

本记录对应的早期部署只包含测试集基础图谱及只读检索，当时图片和视频主要是归档元数据，网页还是人工样例。后续已加入真实图谱网页、文本 GraphRAG 与部分视觉观察候选；当前接口见 [食谱检索与生成系统](食谱检索与生成系统.md)和 [GraphRAG 设计](GraphRAG设计.md)。食材、动作与工具仍为规则提及候选，不构成完整配方、库存或饮食安全验证。历史阶段结果不能代替新机器的实际导入与回读。
