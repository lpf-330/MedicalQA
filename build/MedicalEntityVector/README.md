# MedicalEntityVector 向量数据库部署项目

## 项目简介

MedicalEntityVector是一个医疗实体向量数据库部署项目，用于将医疗知识图谱中的实体数据转换为向量表示，并存储到Zilliz Cloud向量数据库中，以支持高效的语义检索和相似度查询。

该项目实现了从Neo4j图数据库提取医疗实体数据、使用本地向量模型（BAAI/bge-large-zh-v1.5）生成向量、创建向量集合、批量导入向量数据的完整流程，并提供了部署验证和质量评估功能。

**核心特性**：
- 本地向量模型部署（BAAI/bge-large-zh-v1.5，1024维）
- 三集合架构（实体、属性、关系向量分离）
- Neo4j定制化优化（实体名称预处理、批量生成优化）
- 向量数据库质量评估
- 混合检索效果评估
- GPU加速支持

**数据规模**：
- 实体向量：44,657条
- 属性向量：52,720条
- 关系向量：312,226条
- 总向量数：409,603条
- 数据完整性：99.997%

## 项目结构

```
MedicalEntityVector/
├── config.py                          # 统一配置文件
├── config_template.py                 # 配置模板文件
├── logger.py                          # 日志记录模块
├── deploy_comprehensive.py            # 综合部署脚本
├── extract_entities.py                # 实体数据提取模块
├── extract_entity_attributes.py       # 属性数据提取模块
├── extract_relations.py               # 关系数据提取模块
├── generate_vectors.py                # 实体向量生成模块
├── generate_attribute_vectors.py      # 属性向量生成模块
├── generate_relation_vectors.py       # 关系向量生成模块
├── create_collection.py               # 实体集合创建模块
├── create_entity_attributes_collection.py  # 属性集合创建模块
├── create_entity_relations_collection.py   # 关系集合创建模块
├── validate_vector_database.py        # 数据验证模块
├── test_retrieval.py                  # 检索测试模块
├── model_manager.py                   # 模型管理模块
├── hybrid_retrieval_service.py        # 混合检索服务模块
├── README.md                          # 项目说明文档
├── data/                              # 数据目录
│   ├── entities.json                  # 实体数据（44,657条）
│   ├── disease_attributes.json        # 属性数据（52,720条）
│   ├── relations.json                 # 关系数据（312,226条）
│   ├── vectors.json                   # 实体向量数据
│   └── DATA_INTEGRITY.md              # 数据完整性说明文档
├── test/                              # 测试目录
│   ├── reports/                       # 测试报告
│   ├── test_cases/                    # 测试用例
│   └── results/                       # 测试结果
└── logs/                              # 日志目录
```

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
| generate_vectors.py | 生成实体名称向量 | entities.json | 向量数据库 |
| generate_attribute_vectors.py | 生成属性向量 | disease_attributes.json | 向量数据库 |
| generate_relation_vectors.py | 生成关系向量 | relations.json | 向量数据库 |

### 集合创建模块

| 文件 | 功能 | 输出 |
|------|------|------|
| create_collection.py | 创建实体向量集合 | medical_entity集合 |
| create_entity_attributes_collection.py | 创建属性向量集合 | entity_attributes集合 |
| create_entity_relations_collection.py | 创建关系向量集合 | entity_relations集合 |

### 部署与验证模块

| 文件 | 功能 | 说明 |
|------|------|------|
| deploy_comprehensive.py | 综合部署脚本 | 自动执行完整部署流程 |
| validate_vector_database.py | 数据验证 | 验证向量数据库完整性和正确性 |
| test_retrieval.py | 检索测试 | 测试向量检索功能和性能 |

### 辅助模块

| 文件 | 功能 | 说明 |
|------|------|------|
| logger.py | 日志记录 | 提供统一的日志记录功能 |
| model_manager.py | 模型管理 | 管理本地向量模型的加载和推理 |
| hybrid_retrieval_service.py | 混合检索服务 | 提供三集合融合检索服务 |

## 环境配置步骤

### 1. 创建Conda环境

```bash
# 创建conda环境
conda create -n medical_entity_vector python=3.9 -y

# 激活环境
conda activate medical_entity_vector
```

### 2. 安装依赖包

```bash
# 激活环境
conda activate medical_entity_vector

# 安装依赖
pip install pymilvus>=2.3.0
pip install neo4j>=5.0.0
pip install sentence-transformers>=2.2.0
pip install torch>=2.0.0
pip install numpy>=1.24.0
```

或使用requirements.txt批量安装：

```bash
pip install -r requirements.txt
```

### 3. 配置数据库连接

复制配置模板并填入实际连接信息：

```bash
cp config_template.py config.py
# 编辑config.py，填入Neo4j和Zilliz Cloud的连接信息
```

### 4. 下载向量模型

确保向量模型已下载到指定目录：

```bash
# 模型路径配置在config.py的LOCAL_MODEL_CONFIG['cache_dir']
# 默认路径：/home/project/MedicalQA/base_models/
```

## 部署执行步骤

### 方式一：使用综合部署脚本（推荐）

```bash
conda activate medical_entity_vector
python deploy_comprehensive.py
```

部署脚本会自动执行以下步骤：
1. 环境检查
2. 配置验证
3. 数据提取（实体、属性、关系）
4. 集合创建（三个集合）
5. 向量生成与导入
6. 部署验证

### 方式二：分步执行

#### 步骤1：提取数据

```bash
# 提取实体数据
python extract_entities.py

# 提取属性数据
python extract_entity_attributes.py

# 提取关系数据
python extract_relations.py
```

#### 步骤2：创建集合

```bash
# 创建实体集合
python create_collection.py

# 创建属性集合
python create_entity_attributes_collection.py

# 创建关系集合
python create_entity_relations_collection.py
```

#### 步骤3：生成并导入向量

```bash
# 生成并导入实体向量
python generate_vectors.py

# 生成并导入属性向量
python generate_attribute_vectors.py

# 生成并导入关系向量
python generate_relation_vectors.py
```

#### 步骤4：验证部署

```bash
# 验证数据完整性
python validate_vector_database.py

# 测试检索功能
python test_retrieval.py
```

## 验证与测试

### 1. 数据完整性验证

```bash
python validate_vector_database.py
```

验证内容：
- 实体向量数量验证（预期：44,657条）
- 属性向量数量验证（预期：52,720条）
- 关系向量数量验证（预期：312,226条）
- 数据一致性验证

### 2. 检索功能测试

```bash
python test_retrieval.py
```

测试内容：
- 实体检索测试
- 属性检索测试
- 关系检索测试
- 混合检索测试
- 性能评估（延迟、召回率）

### 3. 测试报告

测试报告保存在 `test/reports/` 目录：
- `validation_report.md` - 数据验证报告
- `performance_report.md` - 性能评估报告
- `retrieval_performance_report.json` - 详细测试数据

## 数据完整性说明

数据完整性详细说明请查看：[data/DATA_INTEGRITY.md](data/DATA_INTEGRITY.md)

**完整性摘要**：
- 总体完整性：99.997%
- 实体数据：100%匹配
- 属性数据：99.98%匹配（差异来自Neo4j重复疾病名称去重）
- 关系数据：100%匹配

## 配置说明

### Neo4j配置

```python
NEO4J_CONFIG = {
    "uri": "neo4j+s://your-instance.databases.neo4j.io",
    "user": "your_username",
    "password": "your_password"
}
```

### Zilliz Cloud配置

```python
ZILLIZ_CONFIG = {
    "user": "your_username",
    "password": "your_password",
    "uri": "https://your-instance.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    "token": "your_api_token"
}
```

### 向量模型配置

```python
LOCAL_MODEL_CONFIG = {
    "model_name": "BAAI/bge-large-zh-v1.5",
    "device": "cuda",  # 或 "cpu"
    "dimension": 1024,
    "batch_size": 512
}
```

## 常见问题

### 问题1：连接Zilliz Cloud失败

**解决方法**：
1. 检查config.py中的ZILLIZ_CONFIG配置是否正确
2. 确认Zilliz Cloud集群状态是否正常
3. 检查网络连接是否正常

### 问题2：向量生成速度慢

**解决方法**：
1. 确保使用GPU加速（config.py中device设置为"cuda"）
2. 调整batch_size参数（根据GPU内存调整）
3. 检查模型是否正确加载

### 问题3：数据完整性验证失败

**解决方法**：
1. 检查data/DATA_INTEGRITY.md了解差异原因
2. 重新运行数据提取脚本
3. 检查Neo4j数据库连接是否正常

## 技术栈

- **Python 3.9**：编程语言
- **Neo4j 5.0+**：图数据库（存储医疗实体）
- **Zilliz Cloud**：向量数据库服务（存储向量数据）
- **BAAI/bge-large-zh-v1.5**：向量生成模型（1024维）
- **pymilvus 2.3.0+**：Milvus Python SDK
- **sentence-transformers 2.2.0+**：向量模型库
- **torch 2.0.0+**：深度学习框架

## 更新记录

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2026-04-11 | v1.1 | 完成善后工作整理，更新README |
| 2026-04-09 | v1.0 | 初始版本 |

---

**最后更新时间**：2026-04-11
