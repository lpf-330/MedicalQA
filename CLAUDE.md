# MedicalQA 项目指令

## 项目概述

老人健康监测系统 — 健康咨询子项目。FastAPI + Python。

核心能力：Neo4j 知识图谱 + Milvus 向量检索 + SGLang 大模型推理 → 智能健康咨询和健康报告生成。

技术栈：FastAPI, Neo4j, Milvus/Zilliz, SGLang, LangChain, BAAI/bge-large-zh-v1.5 (1024维)

## Conda 环境

**环境名**：MedicalQA，**路径**：`/home/ai_env/miniforge3/envs/MedicalQA`

所有运行操作必须使用此环境：`conda activate MedicalQA`

## 显存检查

启动前必须检查：`nvidia-smi`。SGLang 模型加载需要大量显存。

## 关键文档

| 文档类型 | 文件夹路径 |
|----------|-----------|
| 项目架构设计 | `doc/项目设计文档/项目架构设计/` |
| 架构原则与使用规范 | `doc/项目设计文档/项目架构设计/` |
| 数据库设计 | `doc/项目设计文档/数据库设计/` |
| 开发强制规范 | `.trae/rules/development-mandatory-requirements-specification.md` |

## 架构一句话

七层单向依赖：Controller → Service → Orchestration → MCPProxy → Tools → ResourceManager → Adapters

辅助层：config/ schemas/ errors/ utils/

**关键**：MCP代理层只代理真正的Tool，模型调用由编排层 ModelBusinessService 直接完成。

## 项目目录

```
src/
├── controller/          # 接入层
├── service/             # 服务层
├── orchestration/       # 编排层 (agent/ chain/ tool_call_handler/ model_business_service/ state_machine/)
├── mcp/                 # MCP代理层 (factory/ proxy/)
├── tools/               # Tool工具层
├── resource_manager/    # 资源管理层 (每资源四封装类)
├── adapters/            # 适配层 (每依赖 Adapter接口 + AdapterImpl)
├── config/              # 配置层 (resources/ business/)
├── schemas/             # 数据类层
├── errors/              # 错误码与异常层
└── utils/               # 工具层
```

## 开发原则与 Rules 映射

| 原则 | 说明 | Rule 文件 |
|------|------|-----------|
| 七层单向依赖 | 严禁反向依赖和跨层调用 | `architecture.md` |
| 命名规范 | 类名/方法名/变量名/包名/文件名 | `naming.md` |
| 资源池化访问 | 禁止直接创建连接，必须通过 GlobalResourceManager | `resource-management.md` |
| 数据库规范 | Neo4j/Milvus 查询和访问规范 | `database.md` |
| 业务开发流程 | 架构依托、codegraph实时更新、自上而下开发 | `development-workflow.md` |
| 测试规范（用户监督） | 先架构测试再业务测试、阶段汇报+用户审核 | `testing-supervised.md` |
| 测试修复规范（无监督） | 除特殊情况外AI全自主测试修复迭代 | `testing-fix-autonomous.md` |
| 测试后修复 | 测试记录驱动、核对设计依据、最小修复和回归验证 | `post-test-fix.md` |

## 任务类型与 Agents 映射

| 任务类型 | 项目 Agent | 何时调用 |
|----------|-----------|----------|
| 开发新业务功能 | `feature-developer` | 新功能需求、新业务流程 |
| 开发新Tool/资源/适配 | `feature-developer` | 新Tool、新资源封装、新适配器 |
| 测试验收与架构审查 | `test-acceptance` | 测试验收、架构合规验证、白盒/黑盒测试 |
| 编写数据库查询 | `database-query-helper` | Cypher/向量检索代码 |
| 排查日志问题 | `test-acceptance` | 日志分析、问题排查 |
| 启动系统 | — | 直接按 `system-startup` skill 执行 |

## 开发场景与 Skills 映射

| 开发场景 | Skill | 说明 |
|----------|-------|------|
| 开发新业务功能 | `new-feature` | 完整开发流程 |
| 封装新资源 | `new-resource` | 四封装类模式 |
| 适配外部依赖 | `new-adapter` | Adapter接口+实现 |
| 开发新Tool | `new-tool` | Tool+MCP代理+Handler |
| 添加配置 | `new-config` | 资源配置+业务配置 |
| 测试验收 | `test-acceptance` | 架构测试→业务测试 |
| 分析日志 | `log-analysis` | 严格审视+根因分析 |
| 启动系统 | `system-startup` | 显存检查→环境激活→启动 |

## codegraph 工具

| 命令 | 用途 |
|------|------|
| `codegraph sync` | 更新索引（**每个类/接口增删改后必须执行**） |
| `codegraph context <文件>` | 查看上下文关系 |
| `codegraph callers <函数>` | 谁调用了它 |
| `codegraph callees <函数>` | 它调用了谁 |
| `codegraph impact <文件>` | 变更影响范围 |
| `codegraph search <关键词>` | 搜索符号 |
| `codegraph status` | 索引状态 |
