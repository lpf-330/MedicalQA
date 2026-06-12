---
name: test-acceptance
description: 测试验收技能。当需要执行测试验收、架构测试、业务测试、代码审查、白盒审查、黑盒测试、生产模拟时触发。典型触发词：测试、验收、审查、白盒、黑盒、生产模拟、架构测试。
---

# 测试验收技能

> **与 ECC 关系**：本技能在架构/业务测试后调用 ECC agents（python-reviewer、fastapi-reviewer、security-reviewer）补充通用审查。

## 执行模式

本技能支持两种执行模式，由触发时的上下文决定：

| 模式 | 规则依据 | 说明 |
|------|---------|------|
| 用户监督 | `testing-supervised.md` | 阶段性完成并汇报，用户审核后才进入下一阶段 |
| 无监督 | `testing-fix-autonomous.md` | 除特殊情况外AI全自主，用户只关注最终结果 |

默认使用用户监督模式。当用户指示"全自动测试修复"或通过 `/loop` 启动时，使用无监督模式。

## 执行顺序

**必须先架构测试再业务测试。** 架构违规会导致业务测试的基础不成立。

## 阶段一：架构测试（先执行）

### 白盒审查

| 检查项 | 方法 | 工具 |
|--------|------|------|
| 七层依赖方向 | 检查是否存在反向依赖或跨层调用 | `codegraph callers`、`codegraph callees` |
| 适配层使用 | 检查外部依赖是否通过适配层访问 | `codegraph search` |
| 资源管理合规 | 检查资源是否通过 GlobalResourceManager 获取 | 代码审查 |
| 命名规范 | 检查类名/方法名/变量名/包名/文件名 | 代码审查 |
| 层间交互 | 检查相邻层是否通过接口交互 | `codegraph context` |

架构测试通过后，才能进入业务测试。

## 阶段二：业务测试（后执行）

### 白盒测试

- 代码逻辑正确性
- 边界条件和异常场景
- 资源获取和释放

### 黑盒测试

- API 接口行为（请求参数校验、响应格式、状态码）
- 数据流完整性（从接入层到适配层的完整流转）
- 异常场景的返回格式

**黑盒测试命令示例**：

```bash
# 健康咨询接口（SSE流式响应）
curl -X POST http://localhost:8000/api/v1/consult \
  -H "Content-Type: application/json" \
  -d '{
    "chat_history": [],
    "question": "老年人高血压应该注意什么？",
    "session_id": "test-session-001",
    "user_profile": {"age": 65, "gender": "male"}
  }'

# 健康报告生成接口（SSE流式响应）
curl -X POST http://localhost:8000/api/v1/report \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {"user_id": 1, "gender": "male", "birth_date": "1955-03-15"},
    "session_id": "test-session-002"
  }'

# 健康检查接口
curl http://localhost:8000/health
```

### 生产模拟

- 使用 MedicalQA conda 环境启动系统
- 检查资源池状态（Neo4j/Milvus/SGLang）
- 检查日志输出（严格审视：未输出日志视为失败）

**生产模拟命令示例**：

```bash
# 1. 检查显存
nvidia-smi

# 2. 激活环境并启动
conda activate MedicalQA
cd /home/project/MedicalQA && python -m src.main

# 3. 启动后检查资源池日志（应包含以下关键日志）
# - [GlobalResourceManager] 初始化完成
# - [ResourcePool] neo4j_pool 创建成功
# - [ResourcePool] milvus_pool 创建成功
# - [ResourcePool] sglang_pool 创建成功
# - [Qwen3ModelResource] 模型加载完成

# 4. 执行黑盒测试命令验证接口可用性
```

## 测试验收记录

**测试阶段只记录问题，不修复问题。**

记录格式：
- 问题编号
- 严重级别（CRITICAL/HIGH/MEDIUM/LOW）
- 问题描述
- 复现步骤
- 影响范围

### 测试结果持久化

**必须使用 memory 工具持久化测试结果和关键过程信息**，防止上下文压缩导致信息丢失，确保修复阶段能精确回溯。

必须持久化的内容：
- 测试问题清单（编号、严重级别、描述、影响范围）
- 关键测试发现（架构违规、设计缺陷等）
- 修复状态跟踪（已修复/未修复/待裁决）

不应持久化的内容：
- 过程性细节（中间调试步骤、临时分析）
- 可从代码或日志重新获取的信息

**memory 更新与清理**：
- 修复完成后，更新对应问题的状态为"已修复"，补充修复方式
- 整轮测试通过后，清理该轮测试的 memory，只保留未解决问题和待裁决项
- 问题已全部解决时，删除对应的测试结果 memory

## 测试后修复

测试记录完成后，按 `post-test-fix.md` 执行修复：

| 根因类型 | 处理方式 |
|----------|---------|
| 实现缺陷 | 自动修复，无需用户审批 |
| 测试缺陷 | 自动修正，无需用户审批 |
| 设计缺陷 | **暂停，汇报用户**，等待裁决 |
| 环境问题 | **暂停，汇报用户**，等待确认 |
| 数据问题 | 自动处理；涉及生产数据时暂停汇报 |

修复后自动执行回归验证和代码审查（ECC python-reviewer / fastapi-reviewer）。

### 用户监督模式下的阶段汇报

用户监督模式下，每个测试-修复阶段完成后必须汇报用户，包含：
1. 本阶段执行内容摘要
2. 发现的问题清单
3. 修复内容和验证结果
4. 需要用户裁决的设计问题
5. 下一阶段计划
6. 等待用户审核批准

### 无监督模式下的特殊情况

无监督模式下，AI 全自主执行，仅在以下特殊情况暂停汇报用户：
1. 遇到设计缺陷、设计冲突、设计缺失
2. 遇到环境问题
3. 同一问题修复 3 次仍失败
4. 用户主动叫停

## 关键约束速查

- 先架构测试再业务测试 → `testing-supervised.md` / `testing-fix-autonomous.md`
- 测试阶段只记录不修复 → `testing-supervised.md` / `testing-fix-autonomous.md`
- 修复阶段全自动执行，设计缺陷和环境问题暂停 → `post-test-fix.md`
- 用户监督：阶段汇报+用户审核 → `testing-supervised.md`
- 无监督：除特殊情况外AI全自主 → `testing-fix-autonomous.md`
- 未输出日志视为执行失败 → `testing-supervised.md` / `testing-fix-autonomous.md`
- 不放过任何 warning/error → `development-workflow.md`
- 同一问题修复 3 次仍失败则暂停汇报 → `post-test-fix.md`
- 测试覆盖率 80% → `testing-supervised.md` / `testing-fix-autonomous.md`
