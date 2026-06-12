---
name: system-startup
description: 系统启动技能。当需要启动项目系统、运行服务、启动MedicalQA时触发。典型触发词：启动系统、启动服务、运行项目、启动MedicalQA。
---

# 系统启动技能

> **与 ECC 关系**：本技能按照项目特有的启动流程规范执行。

## 执行步骤

1. **检查显存** — `nvidia-smi`，确认可用显存满足 SGLang 需求
2. **激活环境** — `conda activate MedicalQA`
3. **启动系统** — `conda run -n MedicalQA python src/main.py`
4. **验证资源池** — 检查启动日志中 Neo4j/Milvus/SGLang/向量模型/LangChain 初始化是否成功
5. **确认代码图** — `codegraph status`，如有变更执行 `codegraph sync`

## 启动流程

```
加载配置 → 初始化GlobalResourceManager → 创建初始资源 → 初始化业务组件 → 启动服务
```

## 启动后验证清单

- [ ] 显存检查通过
- [ ] 配置加载成功
- [ ] 所有资源池创建成功
- [ ] 业务组件初始化成功
- [ ] API 服务启动成功
- [ ] 无 ERROR 日志

## 关键约束速查

- 启动前必须检查显存 → `development-workflow.md`
- 必须使用 MedicalQA conda 环境 → `development-workflow.md`
- 日志分析：未输出日志视为失败 → `testing-supervised.md`
- 启动后用 `codegraph status` 确认索引是否最新 → `development-workflow.md`
