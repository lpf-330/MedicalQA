# 架构规范

> **与 ECC 关系**：补充。ECC 全局 rules 覆盖通用编码规范（不可变性、KISS/DRY/YAGNI、错误处理等），本规则只补充项目特有的七层架构约束。

## 七层架构职责定义

| 层级 | 包路径 | 职责 |
|------|--------|------|
| 接入层 | `src/controller/` | RESTful API接口、SSE流式响应、请求参数校验、异常统一处理、协议转换 |
| 服务层 | `src/service/` | 接收Controller参数、构建上下文、初步业务判断、组合编排层策略与资源 |
| 编排层 | `src/orchestration/` | 基于FSM的业务逻辑控制、封装业务策略、驱动工具调用和内容生成、管理执行状态 |
| MCP代理层 | `src/mcp/` | 统一工具调用接口、支持标准MCP协议和高效直连、工具实例生命周期管理 |
| Tool工具层 | `src/tools/` | 具体业务能力实现、调用资源管理与适配层接口访问外部资源 |
| 资源管理层 | `src/resource_manager/` | 资源包装、统一生命周期管理和池化复用、连接创建/复用/销毁/监控 |
| 适配层 | `src/adapters/` | 对重要底层框架、重要依赖方法的适配封装 |

辅助层：
- 配置层 `src/config/` — 资源配置管理、业务配置管理
- 数据类层 `src/schemas/` — 项目数据类管理
- 错误码与异常层 `src/errors/` — 统一错误码与异常定义
- 工具层 `src/utils/` — 实用程序

## 单向依赖原则

```
接入层 → 服务层 → 编排层 → MCP代理层 → Tool工具层 → 资源管理层 → 适配层
```

**强制约束**：

1. 只允许从左到右的依赖，严禁反向依赖
2. 严禁跨层直接调用（如接入层直接调用Tool工具层）
3. 相邻层之间通过接口交互，不依赖具体实现
4. 辅助层（config/schemas/errors/utils）可被所有层引用，但不依赖任何业务层

## 层间交互规范

### 接入层 → 服务层

- Controller 调用 Service 的公共方法
- Controller 只做参数校验和协议转换，不包含业务逻辑

### 服务层 → 编排层

- Service 组合 Agent/Chain 策略
- Service 管理 Agent 容器（Agent[I,O]）的创建和资源绑定
- Service 不直接操作 Tool 或 MCP 代理

### 编排层内部

- Agent 策略可使用 Chain 策略（Agent 基于 FSM，Chain 是固定流程）
- ToolCallHandler 管理工具调用，含自动重新初始化机制
- ModelBusinessService 直接完成模型调用（**不经过MCP代理层**）
- StateMachine 管理状态转换

### 编排层 → MCP代理层

- 编排层通过 MCPProxyFactory 获取工具代理
- MCPProxyFactory 缓存工具实例
- MCP代理层只代理真正的Tool，不代理模型调用

### MCP代理层 → Tool工具层

- MCPStandardProxy（真代理）通过标准MCP协议调用Tool
- MCPFakeProxy（伪代理）直连调用Tool

### Tool工具层 → 资源管理层

- Tool 必须通过 GlobalResourceManager 获取资源
- 禁止直接创建连接
- 资源获取/释放遵循 `_init_resource()` / `release_source()` / `destroy_source()` 三方法

### 资源管理层 → 适配层

- 资源管理层通过适配层访问外部依赖
- ResourceFactory 创建资源时使用适配层接口

## 适配层使用规范

1. 使用外部依赖前，**必须先检查** `src/adapters/` 是否已有适配接口
2. 如有适配接口，使用适配层提供的内部接口
3. 如无适配且该依赖为重要依赖，**必须先适配再使用**
4. 如无适配且非重要依赖，可直接使用该外部依赖
5. 适配层适配时，**必须按照最新架构设计文档实现**
6. 适配完成后，**必须立即更新**依赖适配接口文档
7. 每个适配包必须有一个对外统一暴露接口和一个实现类

## 不可修改架构设计

在开发业务功能时，**不能修改项目架构的任何设计**。只能在项目架构的各级分层中实现业务功能的逻辑。

如需修改架构，必须汇报开发者。这包括但不限于：
- 修改层级间的依赖关系
- 修改核心接口定义
- 修改资源管理机制
- 修改MCP代理模式

## 架构修改规范

当开发者授权修改架构时，必须遵循以下约束：

1. **必须按架构设计文档实现** — 修改内容必须与项目架构设计文档一致
2. **修改后必须更新设计文档** — 同步更新最新的项目架构设计文档和架构原则与使用规范文档
3. **修改后必须执行架构测试** — 验证修改未破坏现有架构合规性（详见 `testing-supervised.md`）
4. **必须评估影响范围** — 使用 `codegraph impact` 评估对现有业务功能的影响
5. **每个类/接口修改后必须立即 `codegraph sync`** — 确保后续开发基于最新索引

## 配置层规范

### 统一配置文件

- 所有运行期配置值统一写入唯一文件 `config/application.yaml`
- `ConfigManager` 是运行期配置文件加载、合并和校验的唯一入口
- 业务层、资源管理层、Tool层、MCP代理层和适配层不得自行读取配置文件或环境变量
- 仓库只提交 `config/application.example.yaml` 示例文件，真实 `config/application.yaml` 不入库

### 资源配置

- 路径：`src/config/resources/`
- 命名：`{资源类型}_config.py`
- 每个资源配置文件只包含资源配置类（继承 BaseResourceConfig，含 config_id）、字段定义、校验逻辑和资源池配置结构
- 禁止将真实运行期配置值写入受版本控制的 Python 配置结构模块

### 业务配置

- 路径：`src/config/business/`
- 命名：`{业务名称}_config.py`
- 每个业务配置文件只包含 business_id、resource_configs 列表、业务参数字段定义和校验逻辑
- 业务运行期参数同样由统一配置文件的 `business` 段覆盖

### 模型调用参数规范

- **禁止在业务代码中硬编码模型调用参数**（如 `enable_thinking`、`repetition_penalty`、`temperature` 等）
- 模型调用参数必须在业务配置类中定义字段，由 `application.yaml` 覆盖默认值
- ModelBusinessService 调用适配层时，参数从配置类读取，不得使用字面量常量
- 换模型或调参时只需修改 `application.yaml`，无需改代码

### 配置加载流程

加载统一配置文件 → 扫描业务配置结构 → 合并业务配置 → 资源配置去重 → 加载资源配置结构 → 合并资源与资源池配置 → 验证配置 → 创建资源池

### 资源共享机制

多个业务引用同一资源配置时，只创建一个 ResourcePool 实例。业务关闭不影响共享 ResourcePool。

## 编排层包结构规范

| 子包 | 路径 | 文件命名 |
|------|------|---------|
| Agent策略 | `orchestration/agent/{策略名}/` | `{strategy}_strategy.py`, `{strategy}_context.py`, `{strategy}_result.py` |
| Chain策略 | `orchestration/chain/{策略名}/` | `{chain}_chain.py`, `{chain}_context.py`, `{chain}_result.py`, `{chain}_resource.py` |
| ToolCallHandler | `orchestration/tool_call_handler/Impl/` | `{tool_name}_handler.py` |
| ModelBusinessService | `orchestration/model_business_service/Impl/` | `{business}_model_service.py` |
| MCP代理 | `mcp/proxy/Impl/` | `{tool_name}_proxy.py` |
| Tool | `tools/{tool名}/` | `{tool_name}.py` |
| 资源封装 | `resource_manager/{资源名}/` | `{resource}_resource.py`, `{resource}_config.py`, `{resource}_factory.py`, `{resource}_client.py` |