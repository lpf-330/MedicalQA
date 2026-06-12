---
name: new-adapter
description: 新适配器开发技能。当需要适配新的外部依赖、创建Adapter接口和实现类时触发。典型触发词：适配、适配器、Adapter、封装外部依赖。
---

# 新适配器开发技能

> **与 ECC 关系**：本技能调用 ECC /mcp-server-patterns skill 参考 MCP 适配模式，调用 ECC python-reviewer agent 审查。

## 执行步骤

1. **检查适配层** — `codegraph search adapter`，确认 `src/adapters/` 下是否已有该依赖适配
2. **判断是否重要依赖** — 核心业务/资源管理/需替换实现 → 重要依赖必须适配
3. **创建适配包** — `src/adapters/{依赖名}/__init__.py` + `{dep}_adapter.py` + `{dep}_adapter_impl.py`
4. **创建 Adapter 接口** — 使用 abc.ABC + @abstractmethod，完成后 `codegraph sync`
5. **创建 AdapterImpl 实现类** — 继承接口，封装外部依赖细节和异常转换，完成后 `codegraph sync`
6. **更新适配接口文档** — 适配完成后**必须立即更新**
7. **验证无绕过** — `codegraph search {外部依赖包名}` 确认无直接调用
8. **审查** — 调用 ECC python-reviewer

## 关键约束速查

- 适配必须按最新架构设计文档实现 → `architecture.md`
- 每个适配包必须有一个对外统一暴露接口和一个实现类 → `architecture.md`
- 适配后必须立即更新接口文档 → `development-workflow.md`
- 实现类必须封装外部依赖异常 → `architecture.md`
- 命名：{依赖}Adapter（接口）/ {依赖}AdapterImpl（实现）→ `naming.md`
- 每个类完成后必须立即 `codegraph sync` → `development-workflow.md`
