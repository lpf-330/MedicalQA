---
name: new-resource
description: 新资源封装技能。当需要添加新资源类型、封装新数据库连接、封装新模型、创建资源池时触发。典型触发词：添加资源、封装连接、资源池、四封装类。
---

# 新资源封装技能

> **与 ECC 关系**：本技能调用 ECC python-reviewer agent 审查封装类代码。

## 执行步骤

1. **确认四封装类模式** — Resource / Config / Factory / Client，缺一不可
2. **创建 Config** — `src/resource_manager/{资源名}/{resource}_config.py`，继承 ResourceConfig，完成后 `codegraph sync`
3. **创建 Client** — `src/resource_manager/{资源名}/{resource}_client.py`，继承 ResourceClient，完成后 `codegraph sync`
4. **创建 Factory** — `src/resource_manager/{资源名}/{resource}_factory.py`，继承 ResourceFactory，**必须通过适配层创建连接**，完成后 `codegraph sync`
5. **创建 Resource** — `src/resource_manager/{资源名}/{resource}_resource.py`，继承 Resource，完成后 `codegraph sync`
6. **添加资源配置** — `src/config/resources/{资源类型}_config.py`，完成后 `codegraph sync`
7. **注册到 GlobalResourceManager** — 在业务配置中引用，initialize() 自动注册
8. **验证** — `codegraph context` 确认依赖关系正确
9. **审查** — 调用 ECC python-reviewer

## 关键约束速查

- deactivate() 仅标记状态不断开连接，destroy() 才断开 → `resource-management.md`
- Factory 必须使用适配层接口创建连接 → `architecture.md`
- PoolConfig 预检：数据库(256,0)、向量(1024,2048)、LLM(4096,8192) → `resource-management.md`
- 多业务引用同一资源配置只创建一个 ResourcePool → `architecture.md`
- 命名：{资源}Resource/{资源}Config/{资源}Factory/{资源}Client → `naming.md`
- 每个类完成后必须立即 `codegraph sync` → `development-workflow.md`
