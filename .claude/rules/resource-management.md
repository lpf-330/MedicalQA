# 资源管理规范

> **与 ECC 关系**：补充。ECC 全局 rules 覆盖通用错误处理和不可变性模式，本规则补充项目特有的 GlobalResourceManager 使用规范和资源池化机制。

## GlobalResourceManager 使用方式

GlobalResourceManager 是全局静态类，单例模式（`INSTANCE`），管理所有资源。

### 核心方法

| 方法 | 说明 |
|------|------|
| `initialize()` | 统一初始化，自动完成配置加载、验证、工厂注册、资源池创建 |
| `initialize_from_config_manager()` | 从 ConfigManager 初始化 |
| `acquire(resourceType)` | 申请资源，返回 ResourceHandle |
| `release(handle)` | 释放资源 |
| `shutdown()` | 优雅关闭 |

### 推荐使用模式：上下文管理器

```python
with GlobalResourceManager.INSTANCE.acquire("resource_type") as handle:
    client = handle.client
    result = client.some_method()
# 自动释放，无需手动调用 release
```

### 手动管理模式

```python
handle = GlobalResourceManager.INSTANCE.acquire("resource_type")
try:
    client = handle.client
    result = client.some_method()
finally:
    GlobalResourceManager.INSTANCE.release(handle)
```

**优先使用上下文管理器**，避免资源泄漏。

## 四封装类模式

每个资源封装必须包含四个封装类：

| 封装类 | 接口/基类 | 职责 |
|--------|-----------|------|
| Resource封装类 | 实现 Resource 接口 | 资源生命周期管理（get_type, activate, deactivate, destroy） |
| Config封装类 | 实现 ResourceConfig 接口 | 资源配置（resource_type, resource_name, config_protocol） |
| Factory封装类 | 实现 ResourceFactory 接口 | 资源创建和销毁（create, destroy） |
| Client封装类 | 实现 ResourceClient 接口 | 资源客户端访问（get_resource_type, get_raw_resource） |

### 文件组织

每个资源封装在独立子包中：

```
resource_manager/{资源名}/
├── {resource}_resource.py   # Resource封装类
├── {resource}_config.py     # Config封装类
├── {resource}_factory.py    # Factory封装类
└── {resource}_client.py     # Client封装类
```

## 资源生命周期

| 状态 | 触发方法 | 说明 |
|------|----------|------|
| 空闲(IDLE) | 创建后初始状态 | 资源已创建未被使用 |
| 活跃(ACTIVE) | activate() | 资源正在被使用 |
| 已销毁(DESTROYED) | destroy() | 资源彻底释放 |

**关键语义**：
- `deactivate()` 仅标记状态，**不断开连接**
- `destroy()` 才**断开连接**

## 资源池配置规范

### PoolConfig 核心参数

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_size | int | — | 资源池最大资源数 |
| min_idle | int | — | 最小空闲资源数 |
| idle_timeout | int | — | 空闲超时时间(毫秒) |
| max_wait_time | int | — | 最大等待时间(毫秒) |

### PoolConfig 扩展参数

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| allow_dynamic_creation | bool | True | 是否允许动态创建资源 |
| max_pending_requests | int | 100 | 最大等待请求数 |
| creation_timeout | int | 60000 | 资源创建超时时间(毫秒) |
| pre_create_check_enabled | bool | True | 是否启用创建前资源检查 |
| min_memory_mb | int | 512 | 创建资源所需最小内存(MB) |
| min_vram_mb | int | 0 | 创建资源所需最小显存(MB) |

### 各资源类型的预检配置建议

| 资源类型 | pre_create_check_enabled | min_memory_mb | min_vram_mb |
|----------|--------------------------|---------------|-------------|
| 数据库连接 | False | 256 | 0 |
| 向量模型 | True | 1024 | 2048 |
| LLM模型 | True | 4096 | 8192 |
| Embedding模型 | True | 512 | 1024 |

## 资源申请队列规范

### acquire 流程

1. 检查空闲池 → 有空闲资源则直接分配
2. 检查动态创建 → 允许且未达上限则创建新资源
3. 创建前检查 → 检查系统资源是否足够
4. 进入等待队列 → FIFO 排队等待资源释放
5. Event 通知 → 资源释放后通知队首请求

### release 流程

1. 归还空闲池 → 资源状态标记为 IDLE
2. 检查等待队列 → 如有等待请求则分配给队首
3. 排队算法 → FIFO（`collections.deque`，右端 append 左端 popleft）

### 线程安全

`_lock` 保护 `_idle_resources`、`_active_resources`、`_pending_requests`。

## 创建保护三层机制

1. **创建前资源预检** — 检查内存/显存是否满足最低要求
2. **创建过程异常隔离** — 创建失败不影响资源池状态
3. **创建后资源验证** — 验证资源是否可用

## Tool 资源协作规范

1. **禁止直接创建连接** — Tool 不直接创建数据库/模型连接
2. `_init_resource()` — 从 GlobalResourceManager 获取资源句柄
3. `release_source()` — 归还资源池（不断开连接）
4. `destroy_source()` — 彻底销毁资源（断开连接）

## 系统启动流程

```
1. ConfigManager加载config/application.yaml → 2. 初始化GlobalResourceManager → 3. 创建初始资源实例 → 4. 初始化业务组件 → 5. 启动服务
```

核心原则：配置优先、资源预创建、按需激活、优雅关闭。

**配置边界**：GlobalResourceManager 只接收 ConfigManager 合并后的配置对象，不直接读取配置文件或环境变量。资源封装类只能消费合并后的 ResourceConfig 和 PoolConfig。

## ResourceHandle

支持上下文管理器（`__enter__`/`__exit__`），适配 with 语法。

属性：
- `resource_id` — 资源唯一标识
- `resource_type` — 资源类型
- `client` — 资源客户端（泛型 T）
- `manager_ref` — 管理器引用
- `is_released` — 是否已释放
