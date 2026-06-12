# 命名规范

> **与 ECC 关系**：补充。ECC 全局 rules 中的 `common/coding-style.md` 定义了通用命名约定（camelCase、PascalCase、UPPER_SNAKE_CASE 等），本规则补充项目特有的类名/方法名/变量命名规范，以项目架构原则与使用规范文档为准。

## 类命名规范

### 业务层类命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Controller类 | {业务}Controller | ConsultController |
| Service类 | {业务}Service | ConsultService |
| Agent策略类 | {业务}Strategy | ConsultStrategy |
| Chain策略类 | {功能}Chain | ConsultWithKnowledgeChain |
| ToolCallHandler类 | {Tool}Handler | Neo4jMedicalHandler |
| ModelBusinessService类 | {业务}ModelService | ConsultModelService |

### MCP代理层类命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| MCP代理类 | {Tool}Proxy | Neo4jMedicalProxy |

### Tool层类命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Tool类 | {功能}Tool | Neo4jMedicalTool |

### 资源管理层类命名（四封装类模式）

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Resource封装类 | {资源}Resource | Qwen3ModelResource, MedPsyModelResource |
| Config封装类 | {资源}Config | Qwen3ModelConfig, MedPsyModelConfig |
| Factory封装类 | {资源}Factory | SGLangModelFactory, MedPsyModelFactory |
| Client封装类 | {资源}Client | SGLangModelClient, MedPsyModelClient |

### 适配层类命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Adapter接口 | {依赖}Adapter | Neo4jAdapter |
| Adapter实现类 | {依赖}AdapterImpl | Neo4jAdapterImpl |

### 配置层类命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| 资源配置类 | {资源}Config（继承BaseResourceConfig） | Qwen3Config, MedPsyConfig |
| 业务配置类 | {业务}Config（继承BaseConfig） | ConsultServiceConfig |
| 资源池配置类 | PoolConfig（通用） | PoolConfig |

## 方法命名规范

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| 公共方法 | 动词+名词 | process_consult, call_tool |
| 私有方法 | _动词+名词 | _init_tool, _validate_request |
| 初始化方法 | _init_{资源} | _init_tool, _init_Model |
| 释放方法 | release_{资源} | release_source, release_tool |
| 获取方法 | get_{属性} | get_client, get_resource_type |
| 构建方法 | _build_{对象} | _build_response, _build_context |
| 转换方法 | _convert_{对象} | _convert_result |
| 校验方法 | _validate_{对象} | _validate_request |

### 特殊方法命名

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Tool资源初始化 | _init_resource | — |
| Tool资源释放 | release_source | — |
| Tool资源销毁 | destroy_source | — |
| MCP工具初始化 | _init_tool | — |
| MCP工具释放 | release_tool | — |

## 变量命名规范

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| 实例变量 | _名词 | _tool, _model, _client |
| 临时变量 | 名词 | result, context, data |
| 常量 | UPPER_SNAKE_CASE | MAX_SIZE, MIN_IDLE |
| 类型变量 | 大写字母 | T, I, O |
| 枚举值 | UPPER_SNAKE_CASE | STANDARD, FAKE |

## 包命名规范

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Agent策略包 | {策略名}（小写下划线） | consult/ |
| Chain策略包 | {链名}（小写下划线） | consult_with_knowledge/ |
| Tool包 | {tool名}（小写下划线） | neo4j_medical/ |
| 资源封装包 | {资源名}（小写下划线） | qwen3_model/, medpsy_model/ |
| 适配包 | {依赖名}（小写下划线） | neo4j/, sglang/ |
| 配置文件 | {资源类型}_config.py | qwen3_config.py, medpsy_config.py |

## 文件命名规范

| 类型 | 命名规范 | 示例 |
|------|----------|------|
| Agent策略文件 | {strategy}_strategy.py | consult_strategy.py |
| Agent上下文文件 | {strategy}_context.py | consult_context.py |
| Agent结果文件 | {strategy}_result.py | consult_result.py |
| Chain策略文件 | {chain}_chain.py | consult_with_knowledge_chain.py |
| Chain上下文文件 | {chain}_context.py | consult_with_knowledge_context.py |
| Chain结果文件 | {chain}_result.py | consult_with_knowledge_result.py |
| Chain资源文件 | {chain}_resource.py | consult_with_knowledge_resource.py |
