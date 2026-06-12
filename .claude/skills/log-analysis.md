---
name: log-analysis
description: 日志分析技能。当需要分析日志、排查运行时问题、检查系统输出、分析错误信息时触发。典型触发词：分析日志、排查问题、检查日志、warning、error、日志输出。
---

# 日志分析技能

> **与 ECC 关系**：本技能使用 codegraph 追踪调用链，使用 sequential-thinking 分步分析根因。

## 执行步骤

1. **严格审视态度**：
   - 本该输出但未输出日志 → 视为该功能执行失败
   - 不放过任何 warning/error → 必须记录
   - 所有发现必须记录，不可遗漏
2. **定位问题** — 读取日志文件，标记所有 warning/error 及缺失日志
3. **追踪调用链** — `codegraph callers` / `codegraph callees` 追踪相关函数
4. **分析根因** — 使用 sequential-thinking 分步推理
5. **评估影响范围** — `codegraph impact` 评估变更影响
6. **记录问题** — 只记录不修复（测试验收规范）

## 关键约束速查

- 未输出日志视为执行失败 → `development-workflow.md`
- 不放过任何 warning/error → `development-workflow.md`
- 测试验收只记录不修复 → `testing-supervised.md`
- 数据库访问异常检查适配层 → `database.md`
