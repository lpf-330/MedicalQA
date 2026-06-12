# MedicalKnowledgeVector 医疗知识向量部署项目

## 项目简介

MedicalKnowledgeVector是一个医疗知识向量数据库部署项目，用于将医疗知识图谱中的实体、属性、关系数据转换为向量表示，并存储到Zilliz Cloud向量数据库中，以支持高效的语义检索和相似度查询。

该项目实现了从Neo4j图数据库提取医疗实体数据、使用本地向量模型（BAAI/bge-large-zh-v1.5）生成向量、创建向量集合、批量导入向量数据的完整流程，并提供了部署验证和质量评估功能。

**核心特性**：
- 本地向量模型部署（BAAI/bge-large-zh-v1.5，1024维）
- 三集合架构（实体、属性、关系向量分离）
- Neo4j定制化优化（实体名称预处理、批量生成优化）
- 向量数据库质量评估
- 混合检索效果评估
- GPU加速支持

**数据规模**：
- 实体向量：44,655条
- 属性向量：52,720条
- 关系向量：312,159条
- 总向量数：409,534条

## 项目结构

```
MedicalKnowledgeVector/
├── config.py                          # 统一配置文件
├── config_template.py                 # 配置模板文件
├── logger.py                          # 日志记录模块
├── deploy_comprehensive.py            # 综合部署脚本
├── redeploy_medical_entity.py         # 实体集合快速重部署脚本
├── extract_entities.py                # 实体数据提取模块
├── extract_entity_attributes.py       # 属性数据提取模块
├── extract_relations.py               # 关系数据提取模块
├── generate_vectors.py                # 实体向量生成模块（火山引擎API）
├── generate_attribute_vectors.py      # 属性向量生成模块（本地GPU）
├── generate_relation_vectors.py       # 关系向量生成模块（本地GPU）
├── create_collection.py               # 实体集合创建模块
├── create_entity_attributes_collection.py  # 属性集合创建模块
├── create_entity_relations_collection.py   # 关系集合创建模块
├── validate_vector_database.py        # 数据验证模块
├── test_retrieval.py                  # 检索测试模块
├── model_manager.py                   # 模型管理模块
├── hybrid_retrieval_service.py        # 混合检索服务模块
├── README.md                          # 项目说明文档
├── data/                              # 数据目录
│   ├── entities.json                  # 实体数据
│   ├── disease_attributes.json        # 属性数据
│   ├── relations.json                 # 关系数据
│   └── DATA_INTEGRITY.md              # 数据完整性说明文档
├── test/                              # 测试目录
│   ├── reports/                       # 测试报告
│   ├── test_cases/                    # 测试用例
│   └── results/                       # 测试结果
└── logs/                              # 日志目录
```

## Neo4j ID 说明

### id() 已废弃，使用 elementId()

Neo4j 5.x 中 `id()` 函数已废弃，返回的整数 ID 在数据库重建后会改变（非持久化）。本项目已迁移至 `elementId()`。

**elementId 格式**（Neo4j Aura 云端）：
- 节点：`4:{UUID}:{序号}`，如 `4:5ea717a9-2355-4798-ba5d-885a858af3d9:0`
- 关系：`5:{UUID}:{序号}`，如 `5:5ea717a9-2355-4798-ba5d-885a858af3d9:1164187100745039873`

**Milvus 字段映射**：

| Milvus 字段 | 存储内容 | VARCHAR 长度 |
|-------------|---------|--------------|
| `neo4j_node_id` | Neo4j 节点 elementId | 128 |
| `neo4j_relation_id` | Neo4j 关系 elementId | 128 |

应用代码通过 `WHERE elementId(n) = $node_id` 查询 Neo4j，`node_id` 为字符串类型。

## 集合字段定义

### medical_entity 集合

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | INT64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| entity_name | VarChar(512) | 实体名称 |
| entity_type | VarChar(64) | Disease/Drug/Symptom/Food/Check/Department/Producer/Cure |
| neo4j_node_id | VarChar(128) | 对应 Neo4j 节点 elementId |

### entity_attributes 集合

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | INT64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| entity_name | VarChar(255) | 实体名称 |
| entity_type | VarChar(50) | 实体类型（如 Disease） |
| attribute_name | VarChar(50) | 属性名（desc, cause, prevent 等） |
| attribute_value | VarChar(10000) | 属性值内容 |
| neo4j_node_id | VarChar(128) | 对应 Neo4j 节点 elementId |

### entity_relations 集合

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| id | INT64 | 主键自增 |
| vector | FloatVector(1024) | BAAI/bge-large-zh-v1.5 生成 |
| relation_type | VarChar(50) | 关系类型 |
| source_entity_name | VarChar(255) | 源实体名称 |
| source_entity_type | VarChar(50) | 源实体类型 |
| target_entity_name | VarChar(255) | 目标实体名称 |
| target_entity_type | VarChar(50) | 目标实体类型 |
| relation_description | VarChar(500) | 关系描述 |
| neo4j_relation_id | VarChar(128) | 对应 Neo4j 关系 elementId |

## 代码文件说明

### 核心配置文件

| 文件 | 功能 | 说明 |
|------|------|------|
| config.py | 统一配置管理 | 包含所有数据库连接、模型、向量、部署等配置项 |
| config_template.py | 配置模板 | 不含敏感信息的配置模板，用于快速部署 |

### 数据提取模块

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| extract_entities.py | 从Neo4j提取医疗实体 | Neo4j数据库 | data/entities.json |
| extract_entity_attributes.py | 从Neo4j提取疾病属性 | Neo4j数据库 | data/disease_attributes.json |
| extract_relations.py | 从Neo4j提取实体关系 | Neo4j数据库 | data/relations.json |

### 向量生成模块

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| generate_vectors.py | 生成实体名称向量（火山引擎API） | entities.json | 向量数据库 |
| generate_attribute_vectors.py | 生成属性向量（本地GPU） | disease_attributes.json | 向量数据库 |
| generate_relation_vectors.py | 生成关系向量（本地GPU） | relations.json | 向量数据库 |

### 集合创建模块

| 文件 | 功能 | 输出 |
|------|------|------|
| create_collection.py | 创建实体向量集合 | medical_entity集合 |
| create_entity_attributes_collection.py | 创建属性向量集合 | entity_attributes集合 |
| create_entity_relations_collection.py | 创建关系向量集合 | entity_relations集合 |

### 部署与验证模块

| 文件 | 功能 | 说明 |
|------|------|------|
| deploy_comprehensive.py | 综合部署脚本 | 自动执行完整部署流程（支持 `--auto` 自动继续） |
| redeploy_medical_entity.py | 实体集合快速重部署 | 提取+创建+生成+导入一条龙 |
| validate_vector_database.py | 数据验证 | 验证向量数据库完整性和正确性 |
| test_retrieval.py | 检索测试 | 测试向量检索功能和性能 |

### 辅助模块

| 文件 | 功能 | 说明 |
|------|------|------|
| logger.py | 日志记录 | 提供统一的日志记录功能 |
| model_manager.py | 模型管理 | 管理本地向量模型的加载和推理 |
| hybrid_retrieval_service.py | 混合检索服务 | 提供三集合融合检索服务 |

## 环境配置步骤

### 1. 使用项目 Conda 环境

```bash
conda activate MedicalQA
```

### 2. 确认依赖包

```bash
pip install pymilvus>=2.3.0
pip install neo4j>=5.0.0
pip install sentence-transformers>=2.2.0
pip install torch>=2.0.0
pip install numpy>=1.24.0
```

### 3. 配置数据库连接

编辑 `config.py`，填入 Neo4j 和 Zilliz Cloud 的连接信息。

### 4. 确认向量模型

确保向量模型已下载到 `config.py` 中 `LOCAL_MODEL_CONFIG['cache_dir']` 指定的目录。

## 部署执行步骤

### 方式一：使用综合部署脚本（推荐）

```bash
conda activate MedicalQA
cd build/MedicalKnowledgeVector/
python deploy_comprehensive.py --auto
```

### 方式二：分步执行

```bash
# 1. 提取数据
python extract_entities.py
python extract_entity_attributes.py
python extract_relations.py

# 2. 创建集合
python create_collection.py
python create_entity_attributes_collection.py
python create_entity_relations_collection.py

# 3. 生成并导入向量
python generate_attribute_vectors.py
python generate_relation_vectors.py

# 4. 验证部署
python validate_vector_database.py
```

### 方式三：快速重部署 medical_entity

```bash
python redeploy_medical_entity.py
```

该脚本会自动完成：连接数据库 → 提取实体 → 创建集合 → 生成向量 → 导入数据 → 验证。

## 验证与测试

### 数据完整性验证

```bash
python validate_vector_database.py
```

### 检索功能测试

```bash
python test_retrieval.py
```

## 项目改名说明

本项目原名为 `MedicalEntityVector`，已更名为 `MedicalKnowledgeVector`，因为项目不仅包含实体向量，还包含属性向量和关系向量，新名称更准确反映其内容。

## 更新记录

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2026-06-10 | v2.0 | 项目更名为 MedicalKnowledgeVector；Neo4j id() 迁移至 elementId()；VARCHAR 字段扩容至128；更新README |
| 2026-04-11 | v1.1 | 完成善后工作整理，更新README |
| 2026-04-09 | v1.0 | 初始版本 |

---

**最后更新时间**：2026-06-10
