# MedicalEntityVector 日志系统

## 功能特性

### 1. 多处理器日志记录

- **控制台日志处理器**: 实时显示部署进度和状态（INFO及以上级别）
- **文件日志处理器**: 记录详细操作日志到 `logs/deploy.log`（DEBUG及以上级别）
- **错误日志处理器**: 将ERROR及以上级别的日志记录到 `logs/error.log`

### 2. 日志级别

支持标准Python日志级别：
- `DEBUG`: 调试信息
- `INFO`: 普通信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

### 3. 日志格式

```
[时间戳] [级别] 消息内容
```

示例：
```
[2026-04-09 23:07:50] [INFO] 开始部署 MedicalEntityVector 向量数据库
[2026-04-09 23:07:50] [ERROR] [部署失败] 加载模型 - 模型文件不存在
```

### 4. 日志轮转

- 单个日志文件最大 10MB
- 保留最近 5 个备份文件
- 自动创建日志目录

## 使用方法

### 方式一：使用Logger类

```python
from logger import get_logger

logger = get_logger()

logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 方式二：使用便捷函数

```python
from logger import info, error, log_progress, log_deployment_step

info("这是一条信息")
error("这是一条错误")

log_progress(50, 100, "处理数据中")
log_deployment_step("初始化数据库", "开始")
```

### 方式三：部署专用函数

```python
from logger import (
    log_deployment_step,
    log_deployment_success,
    log_deployment_failure
)

log_deployment_step("初始化向量数据库", "开始")
log_deployment_success("初始化向量数据库")
log_deployment_failure("加载模型", "模型文件不存在")
```

## 文件结构

```
MedicalEntityVector/
├── logger.py              # 日志模块主文件
├── logger_example.py      # 使用示例
├── README_LOGGER.md       # 本文档
└── logs/
    ├── deploy.log         # 完整部署日志
    └── error.log          # 错误日志
```

## 最佳实践

### 1. 部署流程日志

```python
logger = get_logger()

logger.info("=" * 60)
logger.info("开始部署 MedicalEntityVector 向量数据库")
logger.info("=" * 60)

steps = ["检查环境", "加载数据", "创建索引"]
for i, step in enumerate(steps, 1):
    log_deployment_step(step, "开始")
    try:
        log_deployment_step(step, "完成")
        log_progress(i, len(steps), step)
    except Exception as e:
        log_deployment_failure(step, str(e))
        raise
```

### 2. 错误处理

```python
try:
    result = some_operation()
except Exception as e:
    logger.exception("操作失败")
    logger.error(f"错误详情: {str(e)}")
```

### 3. 进度跟踪

```python
total_items = 1000
for i, item in enumerate(items, 1):
    process_item(item)
    if i % 100 == 0:
        log_progress(i, total_items, f"处理 {item.name}")
```

## 日志文件说明

### deploy.log
- 记录所有级别的日志（DEBUG及以上）
- 包含详细的部署过程和调试信息
- 用于问题排查和审计

### error.log
- 仅记录ERROR和CRITICAL级别的日志
- 包含完整的异常堆栈信息
- 用于快速定位严重问题

## 注意事项

1. 日志文件会自动创建，无需手动创建logs目录
2. 日志文件达到10MB时会自动轮转
3. 控制台只显示INFO及以上级别的日志
4. 建议在部署脚本开始时初始化logger
5. 异常处理时使用 `logger.exception()` 可自动记录堆栈信息
