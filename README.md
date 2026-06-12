# MedicalQA

> 老人健康监测系统 — 健康咨询子项目

基于医疗知识图谱与向量语义检索的智能健康咨询与健康报告生成系统。

---

## 核心能力

- **健康咨询** — 意图分类 → 实体识别 → 知识检索(向量+图) → 充分性评估 → 补充检索 → 流式回答生成
- **综合健康报告** — 监测数据解析 → 知识检索 → 8维度并发分析 → 风险量化评估 → 报告流式生成

## 技术栈

| 类别 | 技术 |
|------|------|
| Web框架 | FastAPI + Uvicorn |
| 图数据库 | Neo4j（8种节点、11种关系、44,657个节点、312,226条关系） |
| 向量数据库 | Milvus/Zilliz（3个集合、1024维、COSINE相似度） |
| 推理引擎 | SGLang（Qwen3-4B-AWQ + MedPsy-4B-AWQ 双模型并发） |
| Embedding | BAAI/bge-large-zh-v1.5（1024维，本地CUDA） |
| 意图分类 | Apollo-0.5B（本地CUDA） |
| 实体识别 | NER-CMEEE（本地CUDA） |

## 架构

七层单向依赖，严禁反向依赖和跨层调用：

```
Controller → Service → Orchestration → MCP → Tools → ResourceManager → Adapters
```

辅助层：`config/` `schemas/` `errors/` `utils/`

### 各层一览

| 层 | 目录 | 组件 |
|----|------|------|
| 接入层 | `src/controller/` | ConsultController, ReportController — RESTful API、SSE流式响应 |
| 服务层 | `src/service/` | ConsultService, ReportService — 构建上下文、组合编排策略 |
| 编排层 | `src/orchestration/` | 4个Agent策略 + 7个Chain策略 + StateMachine + ModelBusinessService |
| MCP代理层 | `src/mcp/` | MCPProxyFactory + 4个Proxy实现 — 统一工具调用接口 |
| Tool工具层 | `src/tools/` | Neo4jMedicalTool, VectorRetrievalTool, IntentClassificationTool, NERModelTool |
| 资源管理层 | `src/resource_manager/` | GlobalResourceManager + 7组四封装类（Resource/Config/Factory/Client） |
| 适配层 | `src/adapters/` | Neo4jAdapter, MilvusAdapter, SGLangAdapter, LangChainAdapter, TransformersAdapter |

### 编排层详情

**Agent策略**（基于FSM状态机驱动）：

| 策略 | 状态数 | 用途 |
|------|--------|------|
| ConsultStrategy | 7 | 健康咨询全流程 |
| ReportStrategy | 10 | 健康报告全流程 |
| KnowledgeRetrievalStrategy | — | 知识检索子策略 |
| ComprehensiveHealthAnalysisStrategy | — | 综合健康分析（检索规划+知识选择+充分性评估） |

**Chain策略**（固定流程）：

| Chain | 用途 |
|-------|------|
| KnowledgeRetrievalChain | 咨询知识检索 |
| ReportKnowledgeRetrievalChain | 报告知识检索 |
| MultiAnalysisChain | 8维度并发分析 |
| DataPrepareChain | 监测数据解析 |
| AnswerGenerationChain | 咨询回答生成 |
| ReportGenerationChain | 报告生成 |
| HealthAssessmentChain | MedPsy风险量化评估 |

**ModelBusinessService**（直连SGLang，不经MCP代理）：

| 服务 | 模型 | 用途 |
|------|------|------|
| ConsultModelService | Qwen3-4B-AWQ | 咨询回答生成 |
| ReportModelService | Qwen3-4B-AWQ | 报告生成 |
| HealthAssessmentModelService | MedPsy-4B-AWQ | 疾病风险量化评估 |

## 业务流程

### 健康咨询（7状态FSM）

```
INITIAL → QUERY_PARSE → KNOWLEDGE_RETRIEVAL → KNOWLEDGE_INTEGRATION → ANSWER_GENERATION → STREAMING → FINISHED
```

1. **QUERY_PARSE** — 意图分类（Apollo-0.5B）+ 实体识别（NER-CMEEE），失败时规则引擎降级
2. **KNOWLEDGE_RETRIEVAL** — 三集合并行向量检索 → 加权融合 → Neo4j锚定查询获取完整知识
3. **KNOWLEDGE_INTEGRATION** — 充分性评估，不足时触发补充检索
4. **ANSWER_GENERATION** — Qwen3流式生成，SSE推送
5. **STREAMING** — 持续推送至生成完毕

### 健康报告（10状态FSM）

```
INITIAL → DATA_PREPARE → DATA_PARSE → COMPREHENSIVE_HEALTH_ANALYSIS → REPORT_GENERATION → STREAMING → ASSEMBLY → FINISHED
```

1. **DATA_PREPARE** — 监测数据解析（6项指标 × 4个时间维度）
2. **COMPREHENSIVE_HEALTH_ANALYSIS** — 双路并发：知识检索链 + 8维度分析链，MedPsy量化评估
3. **REPORT_GENERATION** — Qwen3流式生成个性化报告

**8大评估维度**：疾病风险、用药建议、治疗方案、饮食建议、检查建议、并发症预警、预防措施、易感人群

**6项监测指标**：心率、血糖、灌注指数、血氧、睡眠、血压（latest / daily_stats / weekly_stats / monthly_stats）

## 混合检索策略

```
用户问题 → Embedding编码
         ↓
  ┌──────────────────────────────────────────┐
  │         三集合并行Top-K检索(K=20)          │
  │  medical_entity(0.40) │ entity_attributes(0.30) │ entity_relations(0.30) │
  └──────────────────────────────────────────┘
         ↓
  Neo4j ID去重 → 分数归一化[0,1] → 加权融合 → 降序排列 → 阈值0.6过滤
         ↓
  neo4j_node_id锚定 → Cypher elementId()查询 → 完整医疗知识
```

## 数据库

### Neo4j 知识图谱（DiseaseKG）

| 节点类型 | 数量 | 关系类型 | 含义 |
|----------|------|----------|------|
| Disease | 8,809 | has_symptom | 疾病→症状 |
| Drug | 3,828 | common_drug / recommand_drug | 疾病→药品 |
| Symptom | 5,998 | acompany_with | 疾病→并发症 |
| Food | 4,870 | do_eat / no_eat / recommand_eat | 疾病→食物 |
| Check | 3,353 | need_check | 疾病→检查 |
| Department | 54 | belongs_to | 归属科室 |
| Producer | 17,201 | drugs_of | 厂家→药品 |
| Cure | 544 | cure_way | 疾病→治疗方式 |

### Milvus 向量数据库（MedicalKnowledgeVector）

| 集合 | 记录数 | 用途 |
|------|--------|------|
| medical_entity | 44,657 | 实体名称向量 |
| entity_attributes | 52,720 | 实体属性向量（desc/cause/prevent等） |
| entity_relations | 312,226 | 实体关系向量 |

索引：IVF_FLAT（nlist=1024, nprobe=16），向量维度1024，COSINE相似度。

## 目录结构

```
MedicalQA/
├── config/
│   └── application.yaml              # 统一运行期配置（唯一来源，不入库）
├── src/
│   ├── main.py                       # FastAPI入口，lifespan管理启动/关闭
│   ├── controller/                   # 接入层
│   ├── service/                      # 服务层
│   ├── orchestration/                # 编排层
│   │   ├── agent/                    # 4个Agent策略
│   │   ├── chain/                    # 7个Chain策略
│   │   ├── model_business_service/   # 3个模型服务（直连SGLang）
│   │   ├── tool_call_handler/        # 4个Handler（Neo4j/向量/意图/NER）
│   │   └── state_machine/            # FSM引擎
│   ├── mcp/                          # MCP代理层
│   │   ├── factory/                  # MCPProxyFactory
│   │   └── proxy/impl/              # 4个Proxy实现
│   ├── tools/                        # 4个Tool
│   ├── resource_manager/             # GlobalResourceManager + 7组四封装
│   ├── adapters/                     # 5个适配（Neo4j/Milvus/SGLang/LangChain/Transformers）
│   ├── config/                       # 配置层（resources/ + business/ + ConfigManager）
│   ├── schemas/                      # 数据类
│   ├── errors/                       # 错误码与异常
│   └── utils/                        # 工具
├── build/                            # 构建脚本
│   ├── diseaseKG/                    # 知识图谱构建与部署
│   ├── MedicalKnowledgeVector/       # 向量数据构建与部署
│   ├── medpsy-awq-quantization/      # MedPsy模型AWQ量化
│   └── sgl-kernel/                   # SGLang内核构建
├── base_models/                      # 模型权重
├── logs/                             # 运行日志
└── doc/                              # 设计文档
    └── 项目设计文档/
        ├── 项目架构设计/             # 架构设计v4 + 架构原则v4 + SGLang适配v2 + 依赖适配v1
        ├── 项目需求设计/             # 需求设计v1.1 + API文档
        ├── 项目详细设计/             # 业务详细设计v8
        └── 数据库设计/               # 数据库说明v1.2
```

## 资源管理

所有外部依赖通过 `GlobalResourceManager` 单例统一管理，禁止直接创建连接。

**7种资源**（每组四封装类：Resource/Config/Factory/Client）：

| 资源 | 池容量 | 预检 | 说明 |
|------|--------|------|------|
| neo4j_connection | max=10, idle=2 | 关 | 图数据库连接 |
| milvus_connection | max=10, idle=2 | 关 | 向量数据库连接 |
| reasoning_model | max=4, idle=1 | 关 | Qwen3推理模型（SGLang :30000） |
| health_assessment_model | max=3, idle=1 | 关 | MedPsy评估模型（SGLang :30001） |
| vector_model | max=1, idle=1 | 开 | BGE向量编码模型 |
| intent_model | max=1, idle=1 | 开 | Apollo意图分类模型 |
| ner_model | max=1, idle=1 | 开 | NER实体识别模型 |

**使用模式**：

```python
# 上下文管理器（推荐）
with GlobalResourceManager.INSTANCE.acquire("neo4j_connection") as handle:
    client = handle.client
    result = client.execute_query(query, params)
# 自动释放

# 手动管理
handle = GlobalResourceManager.INSTANCE.acquire("neo4j_connection")
try:
    result = handle.client.execute_query(query, params)
finally:
    GlobalResourceManager.INSTANCE.release(handle)
```

## 配置

所有运行期配置统一写入 `config/application.yaml`，由 `ConfigManager` 加载合并校验。

仓库只提交 `config/application.example.yaml`，真实配置不入库。

配置段包括：`server`（服务端参数）、`resources`（7种资源连接配置）、`resource_pools`（7个资源池参数）、`business.consult_service_config`（咨询业务参数）、`business.report_service_config`（报告业务参数）。

## API

### 健康咨询

```
POST /api/v1/consult
```

```json
{
  "request_id": "req-001",
  "user_id": "user-001",
  "body": {
    "task_id": "task-001",
    "question": "最近总是头痛怎么办",
    "chat_history": [{"role": "user", "content": "我有高血压"}]
  }
}
```

SSE流式返回。

### 健康报告

```
POST /api/v1/report
```

```json
{
  "request_id": "req-002",
  "user_id": "user-001",
  "body": {
    "task_id": "task-002",
    "monitoring_data": {
      "heart_rate": {"latest": [{"value": 72, "unit": "bpm", "time": "2024-01-01T08:00:00"}]},
      "blood_pressure": {"latest": [{"systolic": 120, "diastolic": 80, "unit": "mmHg"}]}
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

SSE流式返回。

### 系统接口

```
GET  /          # 服务状态
GET  /health    # 健康检查（含资源池统计）
```

## 启动

### 环境要求

- Python 3.9+ / CUDA 11.0+ / 内存 32GB+ / GPU显存 21GB+

### 启动步骤

```bash
conda activate MedicalQA
nvidia-smi                    # 检查显存
python src/main.py           # SGLang模型自动启动
```

服务地址：`http://0.0.0.0:8001`

启动流程：ConfigManager加载配置 → GlobalResourceManager初始化 → 资源池创建 → SGLang子进程启动 → 业务组件初始化 → 模型预热 → Uvicorn启动

### 关闭

Ctrl+C 触发 lifespan shutdown：释放所有模型服务 → 释放Handler → GlobalResourceManager.shutdown() → SGLang子进程终止

## 构建工具

| 目录 | 用途 |
|------|------|
| `build/diseaseKG/` | 知识图谱数据爬取、构建、Neo4j部署 |
| `build/MedicalKnowledgeVector/` | 向量数据提取、编码、Milvus部署、检索验证 |
| `build/medpsy-awq-quantization/` | MedPsy-4B AWQ量化：校准数据构建 → 量化 → 评估对比 |
| `build/sgl-kernel/` | SGLang自定义CUDA内核构建与基准测试 |

## 设计文档

| 文档 | 路径 |
|------|------|
| 架构设计 v4 | `doc/项目设计文档/项目架构设计/项目架构设计v4.md` |
| 架构原则与使用规范 v4 | `doc/项目设计文档/项目架构设计/项目架构原则与使用规范v4.md` |
| 业务详细设计 v8 | `doc/项目设计文档/项目详细设计/项目业务详细设计v8.md` |
| 数据库说明 v1.2 | `doc/项目设计文档/数据库设计/数据库说明文档v1.2.md` |
| 需求设计 v1.1 | `doc/项目设计文档/项目需求设计/项目需求设计v1.1.md` |
| API文档 | `doc/项目设计文档/项目需求设计/API文档.md` |
| SGLang适配说明 v2 | `doc/项目设计文档/项目架构设计/SGLang推理框架适配说明v2.md` |
| 依赖适配接口 v1 | `doc/项目设计文档/项目架构设计/依赖适配接口文档v1.md` |

## 版本

| 版本 | 内容 |
|------|------|
| v1.0.0 | 七层架构搭建，咨询与报告基础功能 |
| v1.1.0 | 综合健康分析重构、双LLM引擎、SGLang适配、充分性评估与补充检索、混合检索优化、资源池化与配置统一 |
