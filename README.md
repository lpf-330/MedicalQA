# 老人健康监测系统-健康咨询子项目

## 项目简介

本项目是"老人健康监测系统"的核心子项目，专注于为老年人提供专业的健康咨询和报告生成服务。系统依托Neo4j医疗知识图谱数据库和Milvus向量数据库，结合大语言模型技术，实现智能化的健康管理和疾病预防服务。

### 核心功能

- **健康咨询**：基于医疗知识图谱的智能问答服务，提供疾病、用药、饮食等专业咨询
- **健康报告生成**：基于监测数据生成个性化的综合健康评估报告，包含8大评估维度

### 技术特点

- 采用**7层架构设计**，职责清晰、易于维护
- 支持**SSE流式响应**，提升用户体验
- 基于**医学知识图谱**，确保建议的科学性和权威性
- 支持**向量语义检索**，提高检索准确性
- 采用**状态机驱动**的业务流程，逻辑清晰可控

## 系统架构

### 架构设计

项目采用7层架构设计，遵循单向依赖原则：

```
接入层 → 服务层 → 编排层 → MCP代理层 → Tool工具层 → 资源管理层 → 适配层
```

### 核心层级说明

| 层级 | 职责 | 关键组件 |
|------|------|---------|
| **接入层** | HTTP协议处理、请求参数校验、SSE流式响应 | ConsultController, ReportController |
| **服务层** | 业务逻辑封装、上下文构建、资源组合 | ConsultService, ReportService |
| **编排层** | 状态机驱动、策略执行、流程编排 | Agent, Chain, StateMachine |
| **MCP代理层** | 工具调用接口、协议转换、生命周期管理 | MCPProxyFactory, MCPTool |
| **Tool工具层** | 业务能力实现、外部系统操作 | Neo4jMedicalTool, VectorRetrievalTool |
| **资源管理层** | 资源池化、生命周期管理、配置管理 | GlobalResourceManager, ResourcePool |
| **适配层** | 外部依赖适配、接口封装 | Neo4jAdapter, MilvusAdapter, VLLMAdapter |

详细架构设计请参考：[项目架构设计v2.1](doc/项目设计文档/项目架构设计/项目架构设计v2.1.md)

## 项目结构

```
MedicalQA/
├── base_models/              # 模型源码文件夹
├── logs/                     # 日志文件夹
├── doc/                      # 项目设计文档文件夹
│   └── 项目设计文档/
│       ├── 项目架构设计/
│       ├── 项目需求设计/
│       ├── 项目详细设计/
│       └── 数据库设计/
├── build/                    # 项目部署文件夹
├── src/                      # 源码文件夹
│   ├── config/              # 配置管理
│   │   ├── resources/       # 资源配置
│   │   └── business/        # 业务配置
│   ├── schemas/             # 数据类定义
│   ├── utils/               # 工具类
│   ├── controller/          # 接入层
│   ├── service/             # 服务层
│   ├── orchestration/       # 编排层
│   │   ├── agent/           # Agent策略
│   │   ├── chain/           # Chain策略
│   │   ├── state_machine/   # 状态机
│   │   ├── tool_call_handler/           # Tool调用处理器
│   │   └── model_business_service/      # 模型业务服务
│   ├── mcp/                 # MCP代理层
│   │   ├── factory/         # 工厂类
│   │   └── proxy/           # 代理实现
│   ├── tools/               # Tool工具层
│   ├── resource_manager/    # 资源管理层
│   └── adapters/            # 适配层
└── README.md
```

## 核心功能模块

### 1. 健康咨询模块

**功能描述**：提供基于医疗知识图谱的智能问答服务

**核心特性**：
- 疾病相关知识查询（症状、病因、预防、治疗）
- 用药指导和建议
- 健康生活方式建议
- 检查项目解读
- 饮食营养建议

**技术实现**：
- 采用**顺序检索模式**：向量检索锚定实体 → 图查询结构化推理增强
- 基于**状态机**的7状态流程控制
- 支持**多轮对话**，上下文由上游管理
- **SSE流式返回**，提升用户体验

详细设计请参考：[项目业务详细设计v3](doc/项目设计文档/项目详细设计/项目业务详细设计v3.md)

### 2. 健康报告生成模块

**功能描述**：基于监测数据生成个性化的综合健康评估报告

**核心特性**：
- **8大评估维度**：疾病风险评估、用药建议、治疗方案、饮食建议、检查建议、并发症预警、预防措施、易感人群分析
- **6项监测指标**：心率、血糖、灌注指数、血氧、睡眠、血压
- **4个时间维度**：latest（当日最新）、daily_stats（最近30天）、weekly_stats（最近12周）、monthly_stats（最近6个月）
- **个性化建议**：基于用户档案和监测数据定制
- **Markdown格式**报告，结构清晰易读

**技术实现**：
- 采用**双路并发模式**：8维度评估任务 ∥ 顺序检索任务
- 基于**状态机**的10状态流程控制
- 支持**空值降级**，自适应调整报告内容
- **SSE流式返回**，实时展示生成进度

详细设计请参考：[项目需求设计v1.1](doc/项目设计文档/项目需求设计/项目需求设计v1.1.md)

## 数据库设计

### Neo4j 图数据库

**用途**：存储结构化医疗知识图谱

**数据规模**：
- 节点：44,657个（8种类型：Disease、Drug、Symptom、Food、Check、Department、Producer、Cure）
- 关系：312,226条（11种类型）

**核心节点类型**：
- **Disease**：疾病节点（8,809个）
- **Drug**：药物节点（3,828个）
- **Symptom**：症状节点（5,998个）
- **Food**：食物节点（4,870个）
- **Check**：检查节点（3,353个）

**核心关系类型**：
- **has_symptom**：疾病-症状关系（54,717条）
- **common_drug**：疾病-常用药物关系（14,651条）
- **recommand_drug**：疾病-推荐药物关系（59,495条）
- **do_eat/no_eat/recommand_eat**：疾病-食物关系
- **need_check**：疾病-检查关系（39,421条）
- **acompany_with**：疾病-并发症关系（12,025条）

### Milvus 向量数据库

**用途**：存储医学实体向量，支持语义检索

**数据规模**：
- 向量总数：409,603条
- 向量维度：1024维
- 数据完整性：99.997%

**集合设计**：
- **medical_entity**：实体名称向量（44,657条）
- **entity_attributes**：实体属性向量（52,720条）
- **entity_relations**：实体关系向量（312,226条）

**检索策略**：
- 采用**三集合并行检索**策略
- **加权融合**算法综合三个集合的检索结果
- 支持**混合检索**：向量语义检索 + 图结构化推理

详细设计请参考：[数据库说明文档v1.2](doc/项目设计文档/数据库设计/数据库说明文档v1.2.md)

## 模型配置

### 显存分配方案（总预算21G）

| 模型类型 | 模型选型 | 显存占用 | 用途 |
|---------|---------|---------|------|
| 大语言模型 | Qwen3-4B-Instruct-2507 | 7G | 报告生成和对话理解 |
| 向量编码模型 | BAAI/bge-large-zh-v1.5 | 2G | 语义编码和知识匹配 |
| 意图分类模型 | FreedomIntelligence/Apollo-0.5B | 1G | 医疗意图分类 |
| 健康风险因子分类模型 | iic/nlp_raner_named-entity-recognition_chinese-base-cmeee | 与健康风险预测模型合计9G | 医学实体识别 |
| 健康风险预测模型 | MedGemma-1.5-4B-IT | 与健康风险因子分类模型合计9G | 疾病风险评估 |
| 系统预留/冗余 | / | 2G | 运行时内存、缓冲 |

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.0+（GPU推理）
- 内存：32GB+ 推荐
- 显存：21GB+

### 安装依赖

```bash
# 创建conda环境
conda create -n MedicalQA python=3.9 -y
conda activate MedicalQA

# 安装依赖
pip install -r requirements.txt
```

### 配置数据库

1. **配置Neo4j连接**

编辑 `src/config/resources/neo4j_config.py`：

```python
class Neo4jResourceConfig(BaseResourceConfig):
    uri: str = "neo4j+s://your-neo4j-uri"
    user: str = "your-username"
    password: str = "your-password"
    database: str = "neo4j"
```

2. **配置Milvus连接**

编辑 `src/config/resources/milvus_config.py`：

```python
class MilvusResourceConfig(BaseResourceConfig):
    uri: str = "https://your-milvus-uri"
    user: str = "your-username"
    password: str = "your-password"
    token: str = "your-token"
```

3. **配置模型路径**

编辑 `src/config/resources/vllm_config.py`：

```python
class VLLMResourceConfig(BaseResourceConfig):
    model_path: str = "/path/to/Qwen3-4B-Instruct-2507"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
```

### 启动服务

```bash
# 激活环境
conda activate MedicalQA

# 启动服务
python src/main.py
```

服务将在 `http://0.0.0.0:8001` 启动。

### API接口

#### 1. 健康咨询接口

**请求地址**：`POST /api/v1/consult`

**请求示例**：

```json
{
  "request_id": "req-123456",
  "user_id": "user-001",
  "body": {
    "task_id": "task-001",
    "chat_history": [
      {"role": "user", "content": "我最近总是头痛"}
    ],
    "question": "我最近总是头痛，应该怎么办？"
  }
}
```

**响应格式**：SSE流式返回

```
event: message
data: {"content": "您好！关于头痛的问题..."}

event: message
data: {"content": "我为您整理了以下几点建议..."}

event: end
data: {"type": "end", "task_id": "task-001", "sources": ["neo4j_node_id_1"]}
```

#### 2. 健康报告生成接口

**请求地址**：`POST /api/v1/report`

**请求示例**：

```json
{
  "request_id": "req-123456",
  "user_id": "user-001",
  "body": {
    "task_id": "task-001",
    "monitoring_data": {
      "heart_rate": {
        "latest": [{"value": 72, "unit": "bpm", "time": "2024-01-01 08:00:00"}],
        "daily_stats": [{"date": "2024-01-01", "avg": 70}]
      },
      "blood_pressure": {
        "latest": [{"systolic": 120, "diastolic": 80, "unit": "mmHg"}]
      }
    },
    "user_profile": {
      "user_id": 1,
      "gender": "male",
      "birth_date": "1955-03-15",
      "height": 170.0,
      "weight": 75.0,
      "past_medical_history": "冠心病史5年"
    }
  }
}
```

**响应格式**：SSE流式返回

```
event: message
data: {"content": "# 健康评估报告\n\n## 一、健康综合评分\n..."}

event: end
data: {"type": "end", "health_score": 78, "risk_level": "轻度风险"}
```

详细API文档请参考：[API文档](doc/项目设计文档/项目需求设计/API文档.md)

## 性能指标

### 响应时间

| 接口 | 目标时长 | 首字节时间 |
|------|---------|-----------|
| 健康咨询 | 5-15秒 | ≤30秒 |
| 健康报告 | 3-5分钟 | ≤30秒 |

### 并发能力

- 支持≥100 QPS并发
- 系统可用性≥99.9%

### 检索性能

- 单集合检索延迟：<150ms
- 混合检索延迟：<200ms
- 召回率：>67%

## 开发规范

### 架构原则

- **单向依赖**：上层依赖下层，下层不依赖上层
- **职责清晰**：每层只负责自己的核心职责
- **接口隔离**：层与层之间通过接口交互
- **依赖倒置**：高层模块不依赖低层模块，两者都依赖抽象

### 命名规范

- **Controller类**：{业务}Controller（如ConsultController）
- **Service类**：{业务}Service（如ConsultService）
- **Agent策略类**：{业务}Strategy（如ConsultStrategy）
- **Chain策略类**：{功能}Chain（如KnowledgeRetrievalChain）
- **Tool类**：{功能}Tool（如Neo4jMedicalTool）

详细规范请参考：[项目架构原则与使用规范v1](doc/项目设计文档/项目架构设计/项目架构原则与使用规范v1.md)

## 项目文档

### 设计文档

- [项目架构设计v2.1](doc/项目设计文档/项目架构设计/项目架构设计v2.1.md)
- [项目架构原则与使用规范v1](doc/项目设计文档/项目架构设计/项目架构原则与使用规范v1.md)
- [项目需求设计v1.1](doc/项目设计文档/项目需求设计/项目需求设计v1.1.md)
- [项目业务详细设计v3](doc/项目设计文档/项目详细设计/项目业务详细设计v3.md)
- [数据库说明文档v1.2](doc/项目设计文档/数据库设计/数据库说明文档v1.2.md)

### API文档

- [API文档](doc/项目设计文档/项目需求设计/API文档.md)

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-04-01 | 初始版本，完成基础架构搭建 |
| v1.1 | 2026-04-10 | 完成健康咨询和健康报告生成功能 |
| v1.2 | 2026-04-15 | 优化检索策略，调整并发模式 |
| v1.3 | 2026-04-19 | 完善文档，补充API文档 |

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过项目Issue反馈。
