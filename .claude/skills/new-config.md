---
name: new-config
description: 配置开发技能。当需要添加资源配置、创建业务配置、修改配置文件、添加资源池配置时触发。典型触发词：配置、config、资源配置、业务配置、PoolConfig。
---

# 配置开发技能

> **与 ECC 关系**：本技能调用 ECC python-reviewer agent 审查配置代码。

## 执行步骤

1. **确定配置类型**
   - 资源配置（数据库连接、模型参数、池化参数）→ `src/config/resources/`
   - 业务配置（业务参数、资源配置引用）→ `src/config/business/`

2. **创建资源配置** — `src/config/resources/{资源类型}_config.py`
   - 资源配置类（继承 BaseResourceConfig，含 config_id）
   - 资源池配置（PoolConfig 实例）
   - 每个类完成后立即 `codegraph sync`

3. **创建业务配置** — `src/config/business/{业务名称}_config.py`
   - business_id
   - resource_configs 列表（引用资源配置）
   - 业务参数
   - 每个类完成后立即 `codegraph sync`

4. **在业务配置中引用资源配置** — 确保引用正确，资源共享机制生效

5. **验证配置加载**
   - 检查 GlobalResourceManager.initialize() 能正确发现配置
   - 确认资源共享：多业务引用同一资源配置时只创建一个 ResourcePool

## 关键约束速查

- 资源配置命名：{资源类型}_config.py → `naming.md`
- 业务配置命名：{业务名称}_config.py → `naming.md`
- 资源配置继承 BaseResourceConfig → `architecture.md`
- 配置加载流程：扫描业务→解析→去重→加载→创建池 → `architecture.md`
- 多业务引用同一资源配置只创建一个 ResourcePool → `architecture.md`
- 每个类完成后必须立即 `codegraph sync` → `development-workflow.md`
