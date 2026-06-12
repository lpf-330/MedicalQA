# 数据库规范

> **与 ECC 关系**：补充。ECC 全局 rules 覆盖通用安全和错误处理，本规则补充项目特有的 Neo4j/Milvus 数据库结构和查询规范。数据库详细信息以最新的数据库设计文档为准。

## 数据库概览

两类数据库配合使用：

1. **Neo4j 图数据库** — DiseaseKG 医疗知识图谱
2. **Milvus 向量数据库** — MedicalEntityVector 医学实体向量库

**访问原则**：所有数据库访问必须通过资源池 + 适配层，严禁直接创建连接。

## Neo4j 图数据库

### 节点类型（8种）

| 节点类型 | 数量 | 属性 |
|----------|------|------|
| Disease | 8,809 | name, desc, prevent, cause, easy_get, cure_lasttime, cured_prob |
| Drug | 3,828 | name |
| Food | 4,870 | name |
| Check | 3,353 | name |
| Department | 54 | name |
| Producer | 17,201 | name |
| Symptom | 5,998 | name |
| Cure | 544 | name |

### 关系类型（11种）

| 关系类型 | 起始→终止 | 含义 |
|----------|-----------|------|
| recommand_eat | Disease→Food | 推荐食谱 |
| no_eat | Disease→Food | 忌吃 |
| do_eat | Disease→Food | 宜吃 |
| belongs_to | Disease/Department→Department | 属于 |
| common_drug | Disease→Drug | 常用药品 |
| drugs_of | Producer→Drug | 生产药品 |
| recommand_drug | Disease→Drug | 好评药品 |
| need_check | Disease→Check | 诊断检查 |
| has_symptom | Disease→Symptom | 症状 |
| acompany_with | Disease→Disease | 并发症 |
| cure_way | Disease→Cure | 治疗方法 |

### Cypher 查询规范

1. 所有查询必须使用参数化查询，防止注入
2. 通过适配层 `Neo4jAdapter` 接口执行查询
3. 使用 `elementId()` 而非已废弃的 `id()` 函数查询节点
4. `neo4j_node_id` 在向量数据库中存储为 elementId 字符串（格式如 `"4:UUID:序号"`），直接传递给 Cypher `WHERE elementId(n) = $node_id`

## Milvus 向量数据库

### 集合设计（3个）

| 集合名称 | 向量维度 | 索引类型 | 相似度 | 实体数量 | 用途 |
|----------|----------|----------|--------|----------|------|
| medical_entity | 1024 | IVF_FLAT | COSINE | 44,657 | 所有医学实体向量 |
| entity_attributes | 1024 | IVF_FLAT | COSINE | 52,720 | 实体属性向量 |
| entity_relations | 1024 | IVF_FLAT | COSINE | 312,226 | 实体关系向量 |

索引参数：nlist=1024, nprobe=16

### medical_entity 集合字段

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | Int64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| entity_name | VarChar(512) | 实体名称 |
| entity_type | VarChar(64) | Disease/Drug/Symptom/Food/Check/Department/Producer/Cure |
| neo4j_node_id | VarChar(128) | 对应 Neo4j 节点 elementId |

### entity_attributes 集合字段

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | Int64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| entity_name | VarChar(512) | 实体名称 |
| entity_type | VarChar(64) | 实体类型（如 Disease） |
| attribute_name | VarChar(64) | 属性名（desc, cause, prevent 等） |
| attribute_value | VarChar(10000) | 属性值内容 |
| neo4j_node_id | VarChar(128) | 对应 Neo4j 节点 elementId |

### entity_relations 集合字段

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | Int64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| relation_type | VarChar(64) | 关系类型 |
| source_entity_name | VarChar(512) | 源实体名称 |
| source_entity_type | VarChar(64) | 源实体类型 |
| target_entity_name | VarChar(512) | 目标实体名称 |
| target_entity_type | VarChar(64) | 目标实体类型 |
| neo4j_relation_id | VarChar(128) | 对应 Neo4j 关系 elementId |

## 混合检索策略

### 三集合并行检索

1. **medical_entity** — 实体名称检索，权重 **0.40**
2. **entity_attributes** — 实体属性检索，权重 **0.30**
3. **entity_relations** — 实体关系检索，权重 **0.30**

### 融合算法流程

1. 并行 Top-K 检索（K=20）
2. 基于 Neo4j ID 去重
3. 分数归一化到 [0,1]
4. 加权融合
5. 降序排列
6. 阈值 0.6 过滤

### neo4j_node_id 锚定查询

向量检索返回 `neo4j_node_id` → 根据 `entity_type` 判断节点类型 → Cypher 语句通过 `elementId()` 查询节点。

`neo4j_node_id` 为 elementId 字符串，直接传递给 Cypher 查询，无需类型转换。

### 向量数据文本格式

- 属性文本：`"{实体类型}：{实体名称}，{属性名}：{属性值}"`
- 关系文本：`"{源实体类型}：{源实体名称}，{关系类型}，{目标实体类型}：{目标实体名称}"`

### 检索性能参考

| 检索类型 | 平均延迟 | P95延迟 | 召回率 |
|----------|----------|---------|--------|
| 实体检索 | 64.66ms | 172.49ms | 70.00% |
| 属性检索 | 76.15ms | 161.86ms | 55.00% |
| 关系检索 | 128.46ms | 231.59ms | 60.00% |
| 混合检索 | 184.56ms | 217.46ms | 67.50% |

## 数据库交互强制规范

1. **禁止直接创建连接** — 必须通过 GlobalResourceManager + 适配层
2. **禁止硬编码连接信息** — 数据库 URI、用户名、密码、token 等运行期配置只能写入未入库的 `config/application.yaml`
3. **禁止跨层读取配置** — 数据库相关 Tool、资源封装和适配层不得自行读取配置文件或环境变量，只能使用 ConfigManager 合并后的配置对象
4. **参数化查询** — 防止注入攻击
5. **资源释放** — 使用上下文管理器自动释放
6. **错误处理** — 数据库操作必须全面处理异常
7. **连接复用** — 通过资源池复用连接，避免频繁创建/销毁
8. **日志脱敏** — 日志、`to_dict()`、`__repr__()` 不得输出密码、token 等真实值；连接地址和账号类字段按是否存在输出即可
