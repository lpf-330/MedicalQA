---
name: new-tool
description: 新Tool开发技能。当需要添加新的Tool工具、创建Tool实现类、开发MCP代理Tool时触发。典型触发词：添加Tool、新建工具、开发Tool、MCP代理工具。
---

# 新Tool开发技能

> **与 ECC 关系**：本技能调用 ECC /mcp-server-patterns skill 参考 MCP 代理模式，调用 ECC python-reviewer agent 审查。

## 执行步骤

1. **Tool资源分析** — 确认该 Tool 需要哪些资源（数据库连接、模型等）
2. **创建 Tool 子包** — `src/tools/{tool名}/{tool_name}.py`
3. **实现 Tool 接口** — `_init_resource()` / `release_source()` / `destroy_source()`，完成后 `codegraph sync`
4. **资源获取** — 必须通过 GlobalResourceManager，禁止直接创建连接
5. **如需 MCP 代理**：
   - 创建 MCPProxy → `src/mcp/proxy/Impl/{tool_name}_proxy.py`
   - 参考 ECC /mcp-server-patterns skill
   - 选择 STANDARD（标准MCP协议）或 FAKE（高效直连）
   - 完成后 `codegraph sync`
6. **如需 ToolCallHandler**：
   - 创建 Handler → `src/orchestration/tool_call_handler/Impl/{tool_name}_handler.py`
   - 完成后 `codegraph sync`
7. **验证** — `codegraph context` 确认依赖关系
8. **审查** — 调用 ECC python-reviewer

## 关键约束速查

- Tool 禁止直接创建数据库/模型连接 → `resource-management.md`
- _init_resource 从 GlobalResourceManager 获取 → `resource-management.md`
- release_source 归还资源池（不断开），destroy_source 彻底销毁（断开）→ `resource-management.md`
- MCP代理只代理真正Tool，不代理模型调用 → `architecture.md`
- 命名：{功能}Tool / {Tool}Proxy / {Tool}Handler → `naming.md`
- ToolCallHandler 含自动重新初始化机制 → `architecture.md`
- 每个类完成后必须立即 `codegraph sync` → `development-workflow.md`
