---
name: new-feature
description: 新业务功能开发技能。当用户要求开发新功能、添加新业务、实现新流程、创建新API接口时触发。典型触发词：开发、添加、实现、新建业务功能。
---

# 新业务功能开发技能

> **与 ECC 关系**：本技能在适当步骤中调用 ECC skills（/fastapi-patterns、/python-patterns、/tdd-workflow、/error-handling、/healthcare-phi-compliance）和 agents（python-reviewer、fastapi-reviewer、tdd-guide）。

## 执行步骤

1. **阅读架构文档** — 最新的项目架构设计文档和架构原则与使用规范文档
2. **使用 codegraph 分析现有代码** — `codegraph context`、`codegraph callers`、`codegraph callees`
3. **确定涉及层级** — 标记哪些层需要新代码、哪些可复用
4. **检查适配层** — `codegraph search adapter`，外部依赖是否已有适配
5. **逐层开发** — Controller → Service → Agent/Chain → ToolCallHandler → MCP代理 → Tool → 资源封装 → 适配层
6. **每个类完成后** — 立即 `codegraph sync` 更新索引
7. **架构测试** — 验证七层依赖、适配层使用、资源管理合规（详见 `testing-supervised.md`）
8. **业务测试** — 验证功能逻辑、API 接口、数据流（详见 `testing-supervised.md`）
9. **代码审查** — 调用 ECC python-reviewer / fastapi-reviewer

## 关键约束速查

- 禁止跨层调用、反向依赖 → `architecture.md`
- 命名必须遵循规范 → `naming.md`
- 资源必须通过 GlobalResourceManager 获取 → `resource-management.md`
- 数据库访问必须通过资源池+适配层 → `database.md`
- 禁止修改架构设计 → `development-workflow.md`
- 每个类完成后必须立即 `codegraph sync` → `development-workflow.md`
- 先架构测试再业务测试 → `testing-supervised.md`
- 异常处理遵循项目规范 → `/error-handling`
- 健康信息合规检查 → `/healthcare-phi-compliance`
