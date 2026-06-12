# SGLang 推理框架适配说明 v3

> 本文档记录 SGLang 推理框架在 RTX 2080 Ti (sm_75) 上的适配过程、运行配置及测试结果。
>
> 所有修改均因 sm_75 架构限制（不支持 bfloat16、CuTe/CUTLASS、FP8 等特性）而起。源码修改、补丁和编译步骤见附录。
>
> **v2 更新内容**：MedPsy 升级为 MedPsy-4B（基于 Qwen3-4B-Thinking-2507，FP16 权重~8GB），替代 MedPsy-4B（~4.1GB）。MedPsy-4B 为 Thinking 模型，需 `--reasoning-parser qwen3` 分离思考与内容；系统提示优化为正向引导减少思考（"3秒内、不超过50字完成思考"）；max_tokens 从 1024 提升至 2048（思考占额外 token）。Qwen3-4B-AWQ 并发从2降为1（--max-running-requests=1，--max-total-tokens=8192），释放约2.3GB KV Cache 显存给 MedPsy-4B。Qwen3-4B-AWQ 不加 `--reasoning-parser`（实测该参数导致 content=null 的 bug），改用 system prompt 末尾 `/no_think` 关闭 thinking。模型调用参数（enable_thinking、repetition_penalty 等）统一走配置链路，禁止硬编码。
>


## 1. 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 2080 Ti |
| 显存 | 22528 MiB (22 GB) |
| 计算能力 | 7.5 (sm_75, Turing 架构) |
| bfloat16 支持 | 否 |
| Tensor Core | FP16 only (无 BF16/FP8 Tensor Core) |
| CUDA 版本 | 13.0 (cu130) |
| PyTorch 版本 | 2.11.0+cu130 |
| Python 版本 | 3.11 |
| Conda 环境 | SGLangTest (`/home/ai_env/miniforge3/envs/SGLangTest`) |
| SGLang 版本 | 0.5.12.post1 |
| sgl-kernel 版本 | 0.4.3 (源码编译 sm_75 版) |
| flashinfer 版本 | 0.6.11.post1 |
| 源码构建目录 | `/tmp/sgl-kernel-build/sgl-kernel/` |

## 2. sm_75 架构限制

RTX 2080 Ti 属于 Turing 架构 (sm_75)，与 sm_80+ (Ampere 及以上) 存在以下关键差异：

| 特性 | sm_75 | sm_80+ |
|------|-------|--------|
| bfloat16 Tensor Core | 不支持 | 支持 |
| FP8 Tensor Core | 不支持 | sm_90+ 支持 |
| CuTe/CUTLASS DSL | 不支持 | sm_80+ 支持 |
| FlashAttention 稀疏注意力 | 不支持 | sm_80+ 支持 |
| Programmatic Dependent Launch (PDL) | 不支持 | sm_90+ 支持 |
| NVFP4 | 不支持 | sm_100+ 支持 |

## 3. 不可用功能

以下功能在 sm_75 上不可用或受限：

| 功能 | 状态 | 替代方案 |
|------|------|---------|
| flashinfer attention (prefill/decode) | 不可用 | 使用 `--attention-backend triton` |
| flashinfer norm (rmsnorm 等) | 不可用 | 使用 rmsnorm_sm75.cu PyTorch 回退 |
| FlashAttention 稀疏注意力 | 不可用 | Qwen3-4B 不使用此功能，无影响 |
| bfloat16 数据类型 | 不可用 | 使用 `--dtype float16` |
| FP8 量化推理 | 不可用 | 使用 FP16 量化或无量化 |
| NVFP4 量化 | 不可用 | 不适用 |
| CUDA Graph PDL | 不可用 | `--disable-cuda-graph` 或普通 CUDA graph |
| FA3 (FlashAttention 3) | 不可用 | 使用 triton attention |

## 4. 运行配置

### 4.1 必须配置项

| 参数 | 推荐值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--model-path` | 模型路径 | — | 模型文件目录 |
| `--dtype` | `float16` | — | sm_75 不支持 bfloat16，必须使用 float16 |
| `--attention-backend` | `triton` | `torch_native` | flashinfer attention 在 sm_75 不可用 |

### 4.2 重要配置项

| 参数 | 推荐值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--context-length` | `2048` | `1024` / `4096` / `8192` | 单请求最大上下文长度，**不影响 KV Cache 大小**。2048 适合短对话，4096 适合长文生成 |
| `--mem-fraction-static` | `0.85` | `0.70` / `0.90` | KV Cache 显存占比。0.85=基准，0.70=多模型共存，0.90=最大吞吐（风险高）|
| `--max-total-tokens` | (自动) | `8192` / `16384` | **所有运行中请求的 token 总数上限**。比 `--mem-fraction-static` 更精准：设 8192 时 KV Cache 仅 1.12 GB，释放约 10.9 GB。适合多模型共存 |
| `--max-running-requests` | (自动) | `1` / `2` / `4` | 最大同时运行请求数。显存紧张时限制并发防止 OOM |
| `--sampling-backend` | `pytorch` | `flashinfer` | 两者在 sm_75 均可用。pytorch 更稳定，flashinfer 理论更快 |
| `--schedule-policy` | `lpm` | `fcfs` | 调度策略。`lpm`=前缀匹配优先（配合 RadixCache 多轮对话高效），`fcfs`=先到先得 |
| `--disable-cuda-graph` | (不添加) | (添加) | 不添加=启用 CUDA graph（decode 更快，+0.79GB 显存）。首次启动多约 80 秒捕获。添加=禁用（启动快，decode 稍慢） |
| `--enable-metrics` | (添加) | (不添加) | 启用 Prometheus 指标端点，零显存开销，生产环境建议开启 |

### 4.3 可选配置项

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--host` | `127.0.0.1` | `0.0.0.0` | 监听地址，`0.0.0.0` 允许远程访问 |
| `--port` | `30000` | 任意可用端口 | 服务端口 |
| `--chunked-prefill-size` | `2048` | `512` / `4096` | 分块预填充大小。设为 512 时可能出现 piecewise CUDA graph 不稳定，建议加 `--disable-piecewise-cuda-graph` |
| `--disable-piecewise-cuda-graph` | (不添加) | (添加) | 禁用分片 CUDA graph。`--chunked-prefill-size 512` 时建议添加 |
| `--max-prefill-tokens` | `16384` | `2048` / `4096` | 单次 prefill batch 最大 token 数。显存紧张时可调小，防止长 prompt 显存尖峰 |
| `--schedule-conservativeness` | `1.0` | `0.3`~`2.0` | 调度保守度。<1.0 更保守（减少显存尖峰），>1.0 更激进（更大 batch） |
| `--disable-overlap-schedule` | (不添加) | (添加) | 禁用 prefill/decode 重叠调度，略省显存但降低吞吐 |
| `--disable-radix-cache` | (不添加) | (添加) | 禁用前缀缓存。**不建议禁用**，RadixCache 不占额外显存且提升多轮对话性能 |
| `--api-key` | 无 | 字符串 | API 密钥，生产部署建议设置 |
| `--served-model-name` | 模型路径 | 自定义名 | API 返回的模型名，可覆写实际路径 |
| `--reasoning-parser` | 无 | `qwen3` | Qwen3 思考模式解析器。启用后思考内容输出到 `reasoning_content` 字段 |
| `--allow-auto-truncate` | (不添加) | (添加) | 自动截断超长输入，防止超出 context-length 导致错误 |
| `--strip-thinking-cache` | (不添加) | (添加) | 清理思考缓存，减少显存占用 |
| `--radix-eviction-policy` | `lru` | `lfu` | RadixCache 驱逐策略。`lfu`=最不常用优先驱逐 |
| `--enable-custom-logit-processor` | (不添加) | (添加) | 启用自定义 logit 处理器接口 |
| `--max-queued-requests` | 无限 | `10` / `100` | 等待队列最大请求数，超出返回 429 |
| `--show-time-cost` | (不添加) | (添加) | 在响应中显示耗时信息 |
| `--enable-request-time-stats-logging` | (不添加) | (添加) | 启用请求时间统计日志 |
| `--grammar-backend` | `xgrammar` | `outlines` | 约束解码语法后端。xgrammar 在 sm_75 可用 |
| `--tool-call-parser` | 无 | `qwen3_coder` | 工具调用解析器。注意有效值是 `qwen3_coder` 而非 `qwen3` |
| `--stream-interval` | `1` | `2` / `5` | 流式输出间隔（秒），值越大批量越大 |
| `--incremental-streaming-output` | (不添加) | (添加) | 增量流式输出模式 |
| `--enable-cache-report` | (不添加) | (添加) | 在响应中报告缓存命中情况 |
| `--cpu-offload-gb` | `0` | `2` | CPU 卸载 KV cache 大小 (GB)。启动子进程做卸载，需设置 `LD_LIBRARY_PATH` 让子进程找到 `libcudart.so.13` |
| `--enable-weights-cpu-backup` | (不添加) | (添加) | 将模型权重备份到 CPU 内存，GPU 权重卸载后可从 CPU 重新加载，节省 GPU 显存 |
| `--enable-memory-saver` | (不添加) | (添加) | 内存节省模式，推理完成后释放中间显存，减少 GPU 常驻显存占用 |

### 4.4 不可用配置项

| 参数 | 不可用值 | 原因 | 替代方案 |
|------|---------|------|---------|
| `--attention-backend` | `flashinfer` | flashinfer attention 使用 CuTe/CUTLASS (sm_80+)，sm_75 上 `invalid argument` | `--attention-backend triton` |
| `--dtype` | `bfloat16` | 2080 Ti 无 BF16 Tensor Core | `--dtype float16` |
| FP8 量化 | 所有 FP8 选项 | FP8 需要 sm_90+ | float16 无量化 |
| FA3 attention | — | FlashAttention 3 需要 sm_90+ | triton attention |

### 4.5 未测试配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-hierarchical-cache` | (不添加) | 层级缓存，将 GPU 不活跃 KV cache 卸载到 CPU，需要大量 CPU 内存 |
| `--hicache-ratio` | `1.0` | CPU 端缓存池大小 = GPU KV cache × ratio。ratio 1.0 需 10.67 GB CPU 内存，ratio 2.0 需 25.45 GB |

**未测试原因**：当前系统 CPU 内存不足（hicache-ratio 1.0 需 10.67 GB，系统仅 0.76 GB 空闲）。**非 sm_75 限制**，内存充裕（32GB+）的系统应可使用。

### 4.6 项目核心问题优化建议

针对项目三个核心问题，给出 SGLang 配置优化方案。

#### 4.6.1 显存紧张 + 缓存命中率优化

项目场景具有高前缀命中率特征：多轮健康咨询共享固定 system prompt、Agent 每步在同一上下文追加、报告生成步骤间共享报告模板。缓存命中不仅加速推理，更**减少重复 prefill 的显存尖峰**——在显存极度紧张时，每次重复 prefill 都可能触发 OOM。

**推荐配置组合**：

| 参数 | 推荐值 | 优化原理 |
|------|--------|----------|
| `--schedule-policy` | `lpm` | 前缀匹配优先调度。主动优先调度与运行中请求共享前缀的新请求，提高 RadixCache 命中率。命中时跳过重复 prefill，省计算且省显存尖峰 |
| `--max-total-tokens` | `16384` | 精准控制所有运行请求的 token 总数上限。KV Cache 按 token 数分配，设上限可防止并发请求累积导致 OOM。16384 对应约 2.24 GB KV Cache（vs 默认 9.94 GB），释放约 7.7 GB |
| `--enable-memory-saver` | (添加) | 推理完成后释放中间张量（激活值、注意力矩阵等），减少 GPU 常驻显存占用。这些中间结果在推理结束后不再需要，但默认不释放 |
| `--enable-cache-report` | (添加) | 在响应中报告缓存命中情况，用于监控和调优缓存命中率 |
| `--max-prefill-tokens` | `4096` | 限制单次 prefill batch 最大 token 数，防止长 prompt 导致显存尖峰。健康咨询场景单次 prompt 通常不超过 4096 token |

**缓存命中率预期**：健康咨询场景下 system prompt（角色设定+工具描述）约 2000-4000 token，占上下文大部分。RadixCache + LPM 下，同一会话的后续请求缓存命中率预计 **60-80%**（跳过 system prompt 重复 prefill），对应节省约 60-80% 的 prefill 计算和显存尖峰。

#### 4.6.2 Agent 多模型并发 + 显存节约优化

Agent 运行时需要 LLM + embedding + NER + intent 多模型共存，4 个模型共享 22GB 显存。LLM 权重 ~7.7GB 是大户，但 Agent 等待工具结果（Neo4j 查询、Milvus 检索）时 LLM 空闲，权重仍占显存。

**推荐配置组合**：

| 参数 | 推荐值 | 优化原理 |
|------|--------|----------|
| `--enable-weights-cpu-backup` | (添加) | 将模型权重备份到 CPU 内存。LLM 空闲时可将 GPU 权重卸载（~7.7GB），腾出显存给其他模型使用；需要推理时从 CPU 加载回来。代价：加载回 GPU 约 5-10 秒 |
| `--enable-memory-saver` | (添加) | 推理后释放中间显存。配合 weights-cpu-backup，推理完成后先释放中间张量再卸载权重，最大化释放 GPU 显存 |
| `--cpu-offload-gb` | `2` | 将 2GB 不活跃 KV cache 卸载到 CPU，进一步减少 GPU 常驻显存。需设置 `LD_LIBRARY_PATH` |
| `--max-total-tokens` | `8192` | 严格限制 KV Cache 总量（仅 1.12 GB），为其他模型留足显存 |
| `--max-running-requests` | `2` | 限制最大并发请求数，防止 Agent 多步并发导致 KV Cache 超限 |
| `--mem-fraction-static` | `0.70` | 降低 KV Cache 占比基准，配合 max-total-tokens 双重保险 |

**显存分配预期（Agent 场景）**：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| LLM 权重 | ~7.7 GB | 可通过 weights-cpu-backup 空闲时卸载 |
| KV Cache | ~1.12 GB | max-total-tokens=8192 |
| CUDA Graph | ~0.79 GB | 可通过 --disable-cuda-graph 省掉 |
| LLM 小计 | ~9.6 GB | 权重卸载后仅剩 KV Cache + 少量运行时 |
| Embedding 模型 | ~1.3 GB | BAAI/bge-large-zh-v1.5 |
| NER 模型 | ~1.0 GB | |
| Intent 模型 | ~1.0 GB | |
| 其他模型小计 | ~3.3 GB | |
| **总计** | **~12.9 GB** | 剩余约 9 GB 安全余量 |

#### 4.6.3 健康报告长上下文 + 长输出优化

报告生成是多步骤链式流程，每步累积上下文，最终可达 8K-16K token。长上下文的 KV cache 占用大，且多个步骤间共享大量前缀（报告模板、system prompt）。

**推荐配置组合**：

| 参数 | 推荐值 | 优化原理 |
|------|--------|----------|
| `--context-length` | `16384` | 允许报告生成的长上下文累积。**不影响 KV Cache 大小**，仅限制单请求最大长度 |
| `--max-total-tokens` | `32768` | 总 token 预算。32K token 对应约 4.5 GB KV Cache，确保报告生成不会 OOM，同时允许 2 个并发请求 |
| `--schedule-policy` | `lpm` | 报告生成步骤间共享 system prompt + 报告模板，LPM 调度让后续步骤跳过共享前缀的 prefill |
| `--max-prefill-tokens` | `4096` | 限制单次 prefill 大小，防止长 prompt 导致显存尖峰 |
| `--schedule-conservativeness` | `0.5` | 更保守的调度策略，减少显存尖峰风险。报告生成对延迟不敏感，宁可排队也不 OOM |

**关键机制**：`--max-total-tokens` 作为安全阀——当所有运行请求的 token 总数达到上限时，新请求排队等待而不是挤占显存导致 OOM。这对长上下文场景特别重要：一个 16K token 的报告生成请求可能占用大量 KV Cache，如果同时来了另一个请求，max-total-tokens 确保总量不超限。

### 4.7 启动脚本示例

**单模型场景（最大吞吐）**：

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/Qwen3-4B-Instruct-2507/base_model \
  --context-length 2048 \
  --mem-fraction-static 0.85 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**多模型共存场景（与 embedding 模型共享 GPU）**：

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/Qwen3-4B-Instruct-2507/base_model \
  --context-length 4096 \
  --mem-fraction-static 0.70 \
  --max-total-tokens 8192 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --max-running-requests 2 \
  --host 0.0.0.0 \
  --port 30000
```

**Agent 场景（LLM + embedding + NER + intent 多模型共存，显存节约优先）**：

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/home/ai_env/miniforge3/envs/SGLangTest/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/Qwen3-4B-Instruct-2507/base_model \
  --context-length 8192 \
  --mem-fraction-static 0.70 \
  --max-total-tokens 8192 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --max-running-requests 2 \
  --enable-weights-cpu-backup \
  --enable-memory-saver \
  --cpu-offload-gb 2 \
  --enable-cache-report \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**报告生成场景（长上下文 + 缓存优化，防 OOM）**：

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/Qwen3-4B-Instruct-2507/base_model \
  --context-length 16384 \
  --mem-fraction-static 0.70 \
  --max-total-tokens 32768 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --max-prefill-tokens 4096 \
  --schedule-conservativeness 0.5 \
  --enable-memory-saver \
  --enable-cache-report \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**调试场景（快速启动）**：

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/Qwen3-4B-Instruct-2507/base_model \
  --context-length 2048 \
  --mem-fraction-static 0.85 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --host 0.0.0.0 \
  --port 30000
```

**生产场景（双实例 — Qwen3-4B-AWQ + MedPsy-4B）**：

```bash
#!/bin/bash
# 实例 1：主推理模型 Qwen3-4B-AWQ（不加 --reasoning-parser，改用 /no_think 关闭 thinking）
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/Qwen3-4B-AWQ \
  --context-length 16384 \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --mem-fraction-static 0.78 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000 &

# 实例 2：健康评估模型 MedPsy-4B（基于 Qwen3-4B-Thinking-2507，必须加 --reasoning-parser qwen3）
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/MedPsy-4B \
  --context-length 32768 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.50 \
  --disable-cuda-graph \
  --dtype float16 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 30001 &

wait
```

## 5. 注意事项

1. **pip install 后必须重新补丁**：`load_utils.py` 和 `elementwise.py` 会被 pip 覆盖，每次安装后需手动重新补丁（见附录 B）
2. **不可使用 flashinfer attention 后端**：sm_75 上 flashinfer attention 内核运行时崩溃（`invalid argument`），必须使用 `--attention-backend triton`
3. **不可使用 bfloat16**：2080 Ti 没有 BF16 Tensor Core，SGLang 会自动检测并转为 float16
4. **`--context-length` 只限制单请求最大长度，不控制 KV Cache 大小**：KV Cache 大小由 `--mem-fraction-static` 决定。将 context-length 从 2048 增大到 4096 或 8192 不会增加显存占用
5. **`--mem-fraction-static` 是显存分配的核心控制项**：0.85 是平衡值；0.70 可释放约 3GB 给其他模型；0.90 仅剩约 1GB 空闲显存，有 OOM 风险
6. **`--max-total-tokens` 可精准控制显存**：设为 8192 时 KV Cache 仅 1.12 GB（释放约 10.9 GB），比 `--mem-fraction-static` 更精确，适合多模型共存场景
7. **`--chunked-prefill-size 512` 存在不稳定性**：小 chunk 值搭配 piecewise CUDA graph 可能出现 `CUDA error: illegal instruction`（间歇性），建议保持默认 2048 或加 `--disable-piecewise-cuda-graph` 规避
8. **RadixCache 不占额外显存**：前缀缓存的 Radix Tree 在 CPU 端维护，GPU 端零开销。`--schedule-policy lpm` 配合 RadixCache，多轮对话可跳过重复 prefill
9. **CUDA graph 首次捕获耗时约 80 秒**：启动后首次请求会延迟较久，后续请求不受影响。调试时可加 `--disable-cuda-graph` 跳过
10. **rmsnorm_sm75.cu 性能**：PyTorch 原生实现比 flashinfer CuTe 实现慢（多几次内核启动和类型转换），但对 Qwen3-4B 这样的中小模型影响有限
11. **版本兼容性**：sglang 0.5.12.post1 官方要求 sgl-kernel==0.4.2.post2，我们编译的是 0.4.3。pip 会有版本冲突警告，但不影响运行
12. **flashinfer sampling/renorm 在 sm_75 可用**：与 flashinfer attention 不同，sampling 和 renorm 是纯 CUDA 实现，不依赖 CuTe/CUTLASS
13. **`--reasoning-parser qwen3` 行为与 bug**：MedPsy-4B 必须加此参数，将思考内容解析到 `reasoning_content`，`content` 保留干净输出。**Qwen3-4B-AWQ 禁止加此参数**——实测该参数对 AWQ 量化版本有 bug，导致 `content=null`（无论 `enable_thinking` 为 true 或 false）。Qwen3-4B-AWQ 改用 system prompt 末尾 `/no_think` 关闭 thinking，`_extract_content()` 需剥离空 thinking 标签前缀
14. **`--cpu-offload-gb` 需设置 LD_LIBRARY_PATH**：CPU 卸载会启动子进程，子进程需找到 `libcudart.so.13`，启动前需设置 `export LD_LIBRARY_PATH=.../nvidia/cu13/lib:$LD_LIBRARY_PATH`。此要求仅与 `--cpu-offload-gb` 有关，与 `--enable-memory-saver` 和 `--enable-weights-cpu-backup` 无关
15. **`--enable-hierarchical-cache` 在本系统不可用**：hicache-ratio 1.0 需 10.67 GB CPU 内存，系统不足。非 sm_75 限制，内存充裕系统可用
16. **`--tool-call-parser` 有效值注意**：Qwen3 模型的工具调用解析器有效值是 `qwen3_coder`（不是 `qwen3`）

## 6. AWQ 4-bit 量化部署

### 6.1 选型理由

项目原计划使用 Qwen3-4B FP16 推理，模型权重约 7.7 GB。在 22 GB 显存上与 MedPsy-4B + 3 个小模型共存时，显存极度紧张（5 模型共存需 ~17 GB 仅权重）。

采用 Qwen3-4B-AWQ（官方 4-bit 量化）后权重仅 2.57 GB，节省约 5.1 GB 显存，使 5 模型共存成为可能。

**AWQ 量化规格**：

| 项目 | 值 |
|------|-----|
| 模型来源 | Qwen/Qwen3-4B-AWQ (HuggingFace 官方) |
| 量化位数 | 4 bit |
| 分组大小 | 128 |
| 量化版本 | gemm |
| 零点 | true |
| 模型路径 | `/home/project/MedicalQA/base_models/Qwen3-4B-AWQ` |
| 权重显存 | 2.57 GB |

### 6.2 sm_75 上的 AWQ 性能特征

sm_75 不支持 Marlin AWQ 内核（需要 sm_80+），SGLang 自动回退到 Triton 实现。性能影响：

| 指标 | FP16 | AWQ (Triton) | 差异 |
|------|------|-------------|------|
| 咨询推理 (512 token) | 10.5s | 23.6s | AWQ 慢 2.2x |
| 报告生成 (~1500 token) | 39.2s | 78.5s | AWQ 慢 2.0x |
| 显存占用 | 10.8 GB | 7.7 GB | AWQ 省 3.1 GB |
| decode 吞吐量 | ~21.5 tok/s | ~21.2 tok/s | 差异极小 |

**关键发现**：AWQ 在 sm_75 上的性能瓶颈主要在 prefill 阶段（Triton 反量化开销），decode 阶段吞吐量与 FP16 接近。对于长输出场景（如报告生成），decode 占比大，AWQ 速度劣势缩小。

### 6.3 AWQ 启动配置

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/Qwen3-4B-AWQ \
  --context-length 16384 \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --mem-fraction-static 0.78 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

### 6.4 AWQ 显存分解

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| 模型权重 | 2.57 GB | AWQ 4-bit 量化权重 |
| KV Cache | 4.52 GB | --max-total-tokens=16384，2并发 |
| CUDA Graph | 1.24 GB | 含 piecewise CUDA graph |
| 其他运行时 | ~1.6 GB | 上下文、临时张量等 |
| **总计** | **~9.96 GB** | 剩余约 12.2 GB |

## 7. MedPsy-4B 部署

### 7.1 替代 MedGemma 的原因

项目原计划使用 MedGemma-1.5-4B-IT 作为健康评估模型。经过多轮测试，MedGemma 在 SGLang 上不可用：

| 尝试方案 | 结果 |
|----------|------|
| MedGemma SGLang 原生加载 | 输出全零 token（Gemma3ForConditionalGeneration 架构，禁用视觉编码器后生成崩溃） |
| MedGemma 文本模式提取 | 架构为多模态设计，text-only 模式下 decoder 无法正常工作 |
| MedGemma transformers 推理 | 可正常推理，但需 ~9.3 GB 显存（FP16），多模型共存不可行 |

**结论**：MedGemma 的 Gemma3ForConditionalGeneration 架构与 SGLang 不兼容，且显存占用过高。

### 7.2 MedPsy-4B 特性

MedPsy-4B 基于 Qwen3-4B-Thinking-2507，是 Thinking 模型，SGLang 原生支持：

| 项目 | 值 |
|------|-----|
| 模型架构 | Qwen3ForCausalLM |
| 基座模型 | Qwen3-4B-Thinking-2507 |
| 参数量 | 4B |
| 模型路径 | `/home/project/MedicalQA/base_models/MedPsy-4B` |
| 加载精度 | FP16 |
| 权重显存 | ~8.0 GB |
| 总显存（含 KV Cache） | ~8.5 GB |
| 能力 | 医学心理健康评估、健康风险分析 |
| Thinking 模式 | 内置思考，不可关闭，需 `--reasoning-parser qwen3` 分离 |

### 7.3 Thinking 模型适配要点

MedPsy-4B 基于 Qwen3-4B-Thinking-2507，思考能力已融入模型权重，**无法通过 `enable_thinking=False` 关闭思考**。适配策略：

1. **`--reasoning-parser qwen3`**（MedPsy-4B必须，Qwen3-4B-AWQ禁止）：MedPsy-4B 启动时必须添加此参数，将思考内容解析到 `reasoning_content` 字段，`content` 字段保留干净的 JSON 输出。**Qwen3-4B-AWQ 不得添加此参数**——实测该参数对 AWQ 量化版本有 bug，导致 `content=null`（无论 `enable_thinking` 为 true 或 false）
2. **Qwen3-4B-AWQ 关闭 thinking 的方式**：不加 `--reasoning-parser`，在 system prompt 末尾添加 `/no_think` 关闭 thinking。此模式下 content 可能包含空 thinking 标签前缀（`<think>\n\n</think>\n\n`），`_extract_content()` 需自动剥离
3. **`enable_thinking=False`**（MedPsy-4B仍需传递）：虽然不能关闭思考，但该参数控制 chat_template 是否预填充思考块，减少无效 token 生成。Qwen3-4B-AWQ **不传此参数**（加了 `--reasoning-parser` 时导致 content=null 的 bug）
4. **`_extract_content()` 适配**：reasoning-parser 分离后，`content` 是干净的 JSON，`reasoning_content` 是思考过程。代码只需使用 `content`，丢弃 `reasoning_content`。同时需剥离 `/no_think` 模式下 content 中的空 thinking 标签前缀
5. **系统提示正向引导减少思考**：实验验证，"3秒内、不超过50字完成思考"的组合约束可将思考量降低 15%，同时保证 JSON 结构率 100%
6. **max_tokens 提升至 2048**：思考内容消耗额外 token，1024 不够导致风险因子场景完全失败

### 7.4 Prompt 优化实验结论

经多轮 A/B 实验对比，最优系统提示为：

```
你是一位全科医生，擅长精炼评估。请在3秒内、不超过50字完成思考，然后直接输出JSON。
```

| 方案 | 结构率 | 平均思考量 | 说明 |
|------|--------|-----------|------|
| **组合:3秒且50字内** | **100%** | **1250c** | **最优，实验推荐** |
| 基准:思考50字内 | 100% | 1464c | 次优 |
| 对照组:生产配置 | 67% | 1986c | 风险因子场景失败 |
| 纯时间:3秒内思考 | 50% | 2266c | 模型不理解时间约束，逆反 |

关键发现：
- 纯时间约束（"3秒内"）模型不理解，反而激发更长思考
- 纯字数约束（"50字内"）有效但不够强
- **时间+字数双约束组合效果最佳**——"3秒"增加紧迫感，"50字"提供具体锚点
- 过于具体的约束（"30字内""1句话"）触发逆反效应

### 7.5 MedPsy-4B 启动配置

```bash
#!/bin/bash
/home/ai_env/miniforge3/envs/MedicalQA/medqa_env/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/MedPsy-4B \
  --context-length 4096 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.50 \
  --disable-cuda-graph \
  --dtype float16 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 30001
```

**关键参数说明**：
- `--reasoning-parser qwen3`（必须）：解析 Qwen3 Thinking 模型的 `<think>` 标签，分离思考与内容
- `--disable-cuda-graph`：4B 模型 CUDA Graph 显存开销大，禁用以节省约1.2GB
- `--mem-fraction-static 0.50`：降低 KV Cache 预分配，4B 模型权重占用大
- `--max-total-tokens 4096`：单次健康评估输入+输出控制在 4096 token 内
- `--port 30001`：第二个 SGLang 实例端口，与 Qwen3-4B-AWQ(:30000) 隔离

## 8. 双 SGLang 实例架构

### 8.1 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    GPU (RTX 2080 Ti, 22GB)               │
│                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │  SGLang 实例 1        │  │  SGLang 实例 2          │  │
│  │  Qwen3-4B-AWQ        │  │  MedPsy-4B              │  │
│  │  Port: 30000         │  │  Port: 30001            │  │
│  │  主推理模型           │  │  健康评估模型            │  │
│  │  ~7.7 GB             │  │  ~8.5 GB                │  │
│  └──────────────────────┘  └─────────────────────────┘  │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐ │
│  │ bge-zh   │ │ernie-h-zh│ │ nlp_raner                │ │
│  │ ~1.4 GB  │ │ ~1.0 GB  │ │ ~0.4 GB                  │ │
│  │ (trans-  │ │ (trans-  │ │ (transformers pipeline)  │ │
│  │  formers)│ │  formers)│ │                          │ │
│  └──────────┘ └──────────┘ └──────────────────────────┘ │
│                                                         │
│  总计: ~16.5 GB  │  剩余: ~5.7 GB (25.4%)              │
└─────────────────────────────────────────────────────────┘
```

### 8.2 实例配置对比

| 配置项 | Qwen3-4B-AWQ (:30000) | MedPsy-4B (:30001) |
|--------|----------------------|---------------------|
| 模型路径 | `base_models/Qwen3-4B-AWQ` | `base_models/MedPsy-4B` |
| 精度 | FP16 (AWQ 4-bit 权重) | FP16 |
| context-length | 16384 | 4096 |
| max-total-tokens | 8192 | 4096 |
| max-running-requests | 1 | 1 |
| mem-fraction-static | 0.78 | 0.50 |
| reasoning-parser | —（不加，有bug） | qwen3（必须） |
| disable-cuda-graph | — | 是 |
| 用途 | 咨询对话 + 报告生成 | 健康评估子指标评分 |
| thinking控制 | system prompt加`/no_think` | `--reasoning-parser qwen3` + `enable_thinking=false` |
| 显存占用 | ~7.7 GB | ~8.5 GB |

### 8.3 故障隔离

双实例架构提供进程级故障隔离：

1. **独立进程**：每个 SGLang 实例是独立 Python 进程，一个实例崩溃不影响另一个
2. **独立 KV Cache**：各自管理自己的 KV Cache，不会互相干扰
3. **独立端口**：:30000 和 :30001 互不影响
4. **独立调度**：各自独立的请求调度和排队

### 8.4 3 小模型加载方式

bge-large-zh、ernie-health-zh、nlp_raner 为非生成式模型，使用 transformers/sentence-transformers 直接加载，不经过 SGLang：

| 模型 | 加载方式 | 原因 |
|------|---------|------|
| bge-large-zh-v1.5 | sentence-transformers | 向量编码模型，非自回归生成 |
| ernie-health-zh | transformers AutoModel (Embedding相似度) | 医疗预训练模型,CLS embedding余弦相似度分类 |
| nlp_raner | transformers pipeline (ner) | NER 模型，单次前向推理 |

这三个模型不需要 SGLang 的 KV Cache、流式输出、前缀缓存等特性，直接使用 transformers 更轻量且无需额外适配。

## 9. 全模型部署测试结果

### 9.1 阶段 0.5：各模型兼容性验证

| 模型 | 加载方式 | 推理结果 | 显存 | 备注 |
|------|---------|---------|------|------|
| Qwen3-4B | SGLang Runtime | PASS | ~10.8 GB | FP16 基准 |
| bge-large-zh | sentence-transformers | PASS | ~1.4 GB | dim=1024 向量正常 |
| ernie-health-zh | transformers AutoModel | PASS | ~0.4 GB | Embedding相似度分类正常 |
| nlp_raner | transformers pipeline | PASS | ~0.4 GB | NER 实体提取正常 |
| MedGemma-1.5-4B-IT | SGLang Runtime | FAIL | — | 输出全零 token，架构不兼容 |
| MedPsy-4B | SGLang Runtime | PASS | ~4.1 GB | Qwen3ForCausalLM 一等公民 |

### 9.2 阶段 1：Qwen3-4B-AWQ 显存扫描

| 测试 | 参数 | VRAM 占用 | 可用显存 |
|------|------|----------|---------|
| 1a | max_total_tokens=8192, mem_frac=0.70 | 11.6 GB | 10.7 GB |
| 1b | max_total_tokens=16384, mem_frac=0.70 | 12.8 GB | 9.4 GB |
| 1c | 无 max_total_tokens, mem_frac=0.70 | 17.3 GB | 4.9 GB |

### 9.3 阶段 2：5 模型共存测试

| 模型 | 加载方式 | 显存占用 |
|------|---------|---------|
| Qwen3-4B-AWQ (SGLang :30000) | SGLang Runtime | — |
| MedPsy-4B (SGLang :30001) | SGLang Runtime | — |
| bge-large-zh | sentence-transformers | ~1.4 GB |
| ernie-health-zh | transformers AutoModel | ~0.4 GB |
| nlp_raner | transformers pipeline | ~0.4 GB |
| **5 模型总计** | — | **15,373 MB (~15.4 GB)** |
| **剩余** | — | **6,858 MB (~6.9 GB, 30.4%)** |

> **注**：上表为 Stage 2 实测数据（单并发，--max-total-tokens=8192）。升级到2并发（--max-total-tokens=16384）后，Qwen3-4B-AWQ显存从~7.7GB增加到~9.96GB，5模型总计约~17.7GB，剩余~4.8GB。

**结论**：5 模型共存通过，2并发下剩余~4.8GB显存，仍满足安全冗余要求。

各模型独立推理结果：

| 模型 | 推理内容 | 耗时 | 状态 |
|------|---------|------|------|
| Qwen3-4B-AWQ | 健康咨询 465 字 | 11.22s | OK |
| MedPsy-4B | 健康评估 514 字 | 5.00s | OK |
| bge-large-zh | 向量编码 1024 维 | 0.27s | OK |
| Apollo-0.5B | 意图分类 LABEL_1 (0.8953) | 0.24s | OK |
| nlp_raner | NER 提取 13 实体 | 0.01s | OK |

### 9.4 阶段 3：并行推理测试

两轮并行推理，5 个模型同时请求：

**第 1 轮**（总耗时 32.71s）：

| 模型 | 推理内容 | 耗时 |
|------|---------|------|
| nlp_raner | NER 26 实体 | 0.52s |
| bge-large-zh | 向量 1024 维 | 0.76s |
| Apollo-0.5B | 分类 LABEL_1 (0.8933) | 0.80s |
| MedPsy-4B | 健康评估 816 字 / 512 token | 15.50s |
| Qwen3-4B-AWQ | 咨询 915 字 / 512 token | 32.71s |

**第 2 轮**（总耗时 74.57s）：

| 模型 | 推理内容 | 耗时 |
|------|---------|------|
| nlp_raner | NER 26 实体 | 0.07s |
| bge-large-zh | 向量 1024 维 | 0.12s |
| Apollo-0.5B | 分类 LABEL_1 (0.8932) | 0.16s |
| MedPsy-4B | 健康评估 768 字 / 512 token | 15.49s |
| Qwen3-4B-AWQ | 报告 2306 字 / 1432 completion token | 74.57s |

**结论**：5 模型并行推理 PASS，小模型（NER/向量/分类）几乎不受并行影响，LLM 并行时总耗时由最长的 Qwen3 推理决定。

### 9.5 阶段 4：SGLang vs vLLM 对比

| 配置 | 咨询 512 token | 报告 ~1500 token | 显存占用 |
|------|---------------|-----------------|---------|
| vLLM FP16 (Qwen3-4B) | 9.7s | 26.4s | 13.6 GB |
| SGLang FP16 (Qwen3-4B) | 10.5s | 39.2s | 10.8 GB |
| vLLM AWQ (Qwen3-4B-AWQ) | 29.5s | 90.7s | 18.4 GB |
| SGLang AWQ (Qwen3-4B-AWQ) | 23.6s | 78.5s | 7.7 GB |

**关键发现**：

1. **SGLang AWQ 是最优方案**：虽然推理速度最慢，但显存仅 7.7 GB，比 vLLM AWQ 节省 10.7 GB，使得 5 模型共存成为可能
2. **vLLM AWQ 显存异常偏高**：18.4 GB 显存占用不合理（AWQ 权重仅 2.57 GB），vLLM 的 KV Cache 预分配机制导致
3. **SGLang FP16 vs vLLM FP16**：SGLang 咨询慢 8%，报告慢 49%。报告慢的原因是 SGLang 的 decode 调度策略不同
4. **SGLang AWQ vs vLLM AWQ**：SGLang AWQ 比 vLLM AWQ 快 20%，可能因 SGLang 的 Triton 回退实现优于 vLLM 的实现

## 10. 部署结论与推荐配置

### 10.1 推荐部署方案

| 组件 | 模型 | 加载方式 | 端口 | 显存 |
|------|------|---------|------|------|
| 主推理 | Qwen3-4B-AWQ | SGLang Runtime | 30000 | ~7.7 GB |
| 健康评估 | MedPsy-4B | SGLang Runtime | 30001 | ~8.5 GB |
| 向量编码 | bge-large-zh-v1.5 | sentence-transformers | — | ~1.4 GB |
| 意图分类 | ernie-health-zh | transformers AutoModel | — | ~0.4 GB |
| NER | nlp_raner | transformers pipeline | — | ~0.4 GB |
| **总计** | — | — | — | **~19.5 GB** |
| **剩余** | — | — | — | **~2.7 GB (12.0%)** |

### 10.2 推荐启动参数

**Qwen3-4B-AWQ (:30000)**：

```bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/Qwen3-4B-AWQ \
  --context-length 16384 \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --mem-fraction-static 0.78 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lpm \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**MedPsy-4B (:30001)**：

```bash
/home/ai_env/miniforge3/envs/SGLangTest/bin/python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/MedPsy-4B \
  --context-length 32768 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.50 \
  --disable-cuda-graph \
  --dtype float16 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 30001
```

### 10.3 性能预期

| 场景 | 模型 | 预期耗时 | 说明 |
|------|------|---------|------|
| 健康咨询 (512 token 输出) | Qwen3-4B-AWQ | ~24s | AWQ 在 sm_75 上比 FP16 慢 2-3x |
| 健康报告 (~1500 token 输出) | Qwen3-4B-AWQ | ~78s | decode 阶段吞吐量 ~21 tok/s |
| 健康评估 (512 token 输出) | MedPsy-4B | ~15s | FP16，Thinking 模型，/no_think + 正向引导减少思考 |
| 向量编码 | bge-large-zh | ~0.3s | 单次编码 |
| 意图分类 | ernie-health-zh | ~0.01s | Embedding相似度分类 |
| NER 提取 | nlp_raner | ~0.01s | 单次提取 |

### 10.4 AWQ 在 sm_75 上的权衡

| 方面 | FP16 | AWQ 4-bit | 结论 |
|------|------|-----------|------|
| 推理速度 | 基准 | 慢 2-3x | AWQ 慢，但可接受 |
| 显存占用 | 10.8 GB | 7.7 GB | AWQ 省 3.1 GB |
| 多模型共存 | 不可行 (需 ~17 GB 仅权重) | 可行 (总计 ~15.4 GB) | AWQ 是唯一可行方案 |
| 输出质量 | 基准 | 轻微下降 | 4-bit 量化损失极小 |

**结论**：在 RTX 2080 Ti (22 GB) 上，AWQ 的显存节省是 5 模型共存的必要条件。推理速度下降 2-3x 是 sm_75 硬件限制的代价，可通过 SGLang 的 RadixCache 和 LPM 调度部分弥补。

## 11. 服务进程自动启动

### 11.1 设计背景

之前 SGLang 作为外部 HTTP 服务运行，项目只连接不管理其生命周期——运维人员需要手动启动 SGLang 进程后才能启动项目服务。现在将 SGLang 服务进程的生命周期管理纳入资源管理层，使项目能够根据配置自动启动和管理 SGLang 子进程。

### 11.2 auto_start 开关

资源配置中的 `auto_start` 字段控制是否由项目自动启动 SGLang 进程：

| auto_start 值 | 行为 | 适用场景 |
|---------------|------|---------|
| `True` | 资源管理层在 `activate()` 阶段自动启动 SGLang 子进程 | 单机部署，项目全权管理 SGLang 生命周期 |
| `False` | 仅连接外部已运行的 SGLang HTTP 服务（向后兼容） | SGLang 由外部进程管理器（systemd 等）维护 |

`auto_start=False` 为默认行为，与此前完全一致，不做任何进程管理操作。

### 11.3 启动流程（auto_start=True）

当 `auto_start=True` 时，资源封装类的 `activate()` 阶段执行以下流程：

1. **检查端口占用** — 检测 `launch_host:launch_port` 是否已有服务监听
2. **启动子进程** — 若端口未被占用，通过 `subprocess.Popen` 启动 SGLang 服务进程
3. **轮询等待就绪** — 以 `health_check_interval` 为间隔轮询 `/v1/models` 端点，直到返回正常响应或超过 `startup_timeout`
4. **连接 HTTP 服务** — 就绪后通过适配层正常连接 SGLang HTTP API

若端口已被占用（外部 SGLang 已运行），跳过步骤 2-3，直接连接现有服务。

### 11.4 关闭流程

当 `auto_start=True` 且 SGLang 进程由本项目启动（`_launched_by_us=True`）时，资源封装类的 `destroy()` 阶段执行以下流程：

1. **发送 SIGTERM** — 向 SGLang 子进程发送优雅终止信号
2. **等待退出** — 在 `shutdown_timeout` 时间内等待子进程正常退出
3. **超时 SIGKILL** — 若超过 `shutdown_timeout` 子进程仍未退出，发送 SIGKILL 强制终止
4. **断开连接** — 断开 HTTP 连接，清理资源

若 `_launched_by_us=False`（连接的是外部 SGLang 服务），`destroy()` 仅断开 HTTP 连接，不终止外部进程。

### 11.5 异常退出保护

ProcessManager 通过以下机制保证项目异常退出时 SGLang 子进程被清理：

| 机制 | 说明 |
|------|------|
| `atexit` 注册 | Python 进程正常退出时自动调用清理函数，终止所有由本项目启动的 SGLang 子进程 |
| `signal handler` | 捕获 SIGTERM/SIGINT 等信号，在信号处理函数中终止 SGLang 子进程后再退出 |

双重保障确保即使项目因未捕获异常或外部信号而退出，SGLang 子进程也不会成为孤儿进程。

### 11.6 配置参数

以下参数在 SGLang 资源配置类（如 `SGLangModelConfig`）中定义，由 `application.yaml` 覆盖默认值：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_start` | bool | `False` | 是否由项目自动启动 SGLang 进程 |
| `model_path` | str | — | 模型文件路径，作为 `--model-path` 参数传给 SGLang |
| `launch_host` | str | `127.0.0.1` | SGLang 进程监听地址 |
| `launch_port` | int | `30000` | SGLang 进程监听端口 |
| `launch_args` | list[str] | `[]` | 传给 `sglang.launch_server` 的额外启动参数列表（如 `["--dtype", "float16", "--attention-backend", "triton"]`） |
| `startup_timeout` | int | `300` | 启动超时时间（秒），轮询 `/v1/models` 等待就绪的最长时间 |
| `health_check_interval` | int | `5` | 健康检查间隔（秒），轮询 `/v1/models` 的间隔 |
| `shutdown_timeout` | int | `30` | 关闭超时时间（秒），SIGTERM 后等待子进程退出的最长时间 |

**配置示例**（`application.yaml` 片段）：

```yaml
resources:
  sglang_qwen3:
    auto_start: true
    model_path: /home/project/MedicalQA/base_models/Qwen3-4B-AWQ
    launch_host: "0.0.0.0"
    launch_port: 30000
    launch_args:
      - "--context-length"
      - "16384"
      - "--max-total-tokens"
      - "8192"
      - "--dtype"
      - "float16"
      - "--attention-backend"
      - "triton"
    startup_timeout: 300
    health_check_interval: 5
    shutdown_timeout: 30

  sglang_medpsy:
    auto_start: true
    model_path: /home/project/MedicalQA/base_models/MedPsy-4B
    launch_host: "0.0.0.0"
    launch_port: 30001
    launch_args:
      - "--context-length"
      - "32768"
      - "--max-total-tokens"
      - "4096"
      - "--mem-fraction-static"
      - "0.50"
      - "--disable-cuda-graph"
      - "--dtype"
      - "float16"
      - "--attention-backend"
      - "triton"
      - "--reasoning-parser"
      - "qwen3"
    startup_timeout: 300
    health_check_interval: 5
    shutdown_timeout: 30
```

### 11.7 适配层接口不变

SGLangAdapter / SGLangAdapterImpl 不做任何修改。进程生命周期管理完全在资源管理层内部完成，对适配层透明——适配层始终通过 HTTP 连接 SGLang 服务，不感知服务进程是自动启动还是外部启动的。

### A.1 CMakeLists.txt 修改

**文件路径**: `/tmp/sgl-kernel-build/sgl-kernel/CMakeLists.txt`

#### A.1.1 基础 gencode 修改

将默认 gencode 从 sm_90 改为 sm_75：

```cmake
# 原始: "-gencode=arch=compute_90,code=sm_90"
# 修改为:
"-gencode=arch=compute_75,code=sm_75"
```

#### A.1.2 精度选项禁用

```cmake
# [SM75 PATCH] BF16 disabled - 2080Ti (sm_75) has no BF16 tensor core support
option(SGL_KERNEL_ENABLE_BF16  "Enable BF16"  OFF)
# [SM75 PATCH] FP8 disabled - FP8 requires sm_90+, not available on sm_75
option(SGL_KERNEL_ENABLE_FP8   "Enable FP8"   OFF)
```

#### A.1.3 ENABLE_BELOW_SM90 子架构

仅保留 sm_75，移除 sm_80/87/89：

```cmake
if (ENABLE_BELOW_SM90)
    # [SM75 PATCH] Only sm_75
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_75,code=sm_75"
    )
endif()
```

#### A.1.4 禁用 FA3 和 sm_90a gencode

```cmake
# [SM75 PATCH] Disable FA3 and sm_90a - not needed for sm_75
# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
#     set(SGL_KERNEL_ENABLE_FA3 ON)
#     list(APPEND SGL_KERNEL_CUDA_FLAGS "-gencode=arch=compute_90a,code=sm_90a")
# endif()
```

#### A.1.5 禁用 NVFP4

```cmake
# [SM75 PATCH] NVFP4 disabled - FP4 is Blackwell (sm_100+), not available on sm_75
# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_FP4)
#     list(APPEND SGL_KERNEL_CUDA_FLAGS "-DENABLE_NVFP4=1")
# endif()
```

#### A.1.6 源文件替换

禁用 flashinfer norm.cu（CuTe/CUTLASS），替换为 PyTorch 原生回退实现：

```cmake
# [SM75 PATCH] Disabled - uses flashinfer norm.cuh (CuTe/CUTLASS, sm_80+ only)
# "csrc/elementwise/fused_add_rms_norm_kernel.cu"
# [SM75 PATCH] Pure PyTorch fallback rmsnorm
"csrc/elementwise/rmsnorm_sm75.cu"
```

#### A.1.7 flashinfer 源文件调整

```cmake
# [SM75 PATCH] Disabled flashinfer norm (uses CuTe/CUTLASS, sm_80+ only)
# "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
# [SM75 PATCH] Re-enabled renorm.cu - pure CUDA kernel, compatible with sm_75
"${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
```

#### A.1.8 禁用 FlashAttention 稀疏注意力源文件

```cmake
# [SM75 PATCH] Disabled flash-attention - use triton attention backend instead
# "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_sparse_hdim128_bf16_causal_sm80.cu"
# ... (4个 sparse .cu 文件 + flash_sparse_api.cpp 全部注释)
```

#### A.1.9 构建目标改为单 sm75 目标

将 sm90+sm100 双目标改为单 sm75 目标，安装到 `sgl_kernel/sm75` 目录：

```cmake
# =========================== Common SM75 Build ============================= #
Python_add_library(common_ops_sm75_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_compile_options(common_ops_sm75_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS} -use_fast_math>
)
target_include_directories(common_ops_sm75_build PRIVATE ${INCLUDES})
set_target_properties(common_ops_sm75_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm75"
)
target_link_libraries(common_ops_sm75_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt mscclpp_static)
target_compile_definitions(common_ops_sm75_build PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
)
install(TARGETS common_ops_sm75_build LIBRARY DESTINATION sgl_kernel/sm75)
```

### A.2 新增文件: rmsnorm_sm75.cu

**文件路径**: `/tmp/sgl-kernel-build/sgl-kernel/csrc/elementwise/rmsnorm_sm75.cu`

替代 `fused_add_rms_norm_kernel.cu`（使用 flashinfer norm.cuh，依赖 CuTe/CUTLASS sm_80+）。

使用 PyTorch 原生操作实现 rmsnorm，兼容 sm_75：

```cpp
// [SM75 PATCH] Pure CUDA rmsnorm implementations for sm_75 compatibility
// Replaces flashinfer norm.cu (which requires CuTe/CUTLASS, sm_80+ only)
// Uses PyTorch native operations as fallback - no custom CUDA kernels needed

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>    // 注意：必须用 torch/all.h，不能用 torch/extension.h
#include <cmath>
#include "utils.h"

// rmsnorm: output = input * weight / sqrt(mean(input^2) + eps)
void rmsnorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    auto input_sq = input.to(torch::kFloat32).pow(2);
    auto variance = input_sq.mean(/*dim=*/1, /*keepdim=*/true);
    auto input_normed = input.to(torch::kFloat32) / torch::sqrt(variance + eps);
    auto result = (input_normed * weight.to(torch::kFloat32)).to(input.scalar_type());
    output.copy_(result);
}

// sgl_fused_add_rmsnorm: residual += input; input = residual * weight / sqrt(mean(residual^2) + eps)
void sgl_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    residual.add_(input);
    auto residual_f = residual.to(torch::kFloat32);
    auto variance = residual_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = residual_f / torch::sqrt(variance + eps);
    input.copy_((normed * weight.to(torch::kFloat32)).to(input.scalar_type()));
}

// gemma_rmsnorm: output = input * (1 + weight) / sqrt(mean(input^2) + eps)
void gemma_rmsnorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    auto input_f = input.to(torch::kFloat32);
    auto variance = input_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = input_f / torch::sqrt(variance + eps);
    auto result = (normed * (1.0 + weight.to(torch::kFloat32))).to(input.scalar_type());
    output.copy_(result);
}

// gemma_fused_add_rmsnorm: residual += input; input = residual * (1 + weight) / sqrt(mean(residual^2) + eps)
void gemma_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    CHECK_DIM(2, input);
    residual.add_(input);
    auto residual_f = residual.to(torch::kFloat32);
    auto variance = residual_f.pow(2).mean(/*dim=*/1, /*keepdim=*/true);
    auto normed = residual_f / torch::sqrt(variance + eps);
    input.copy_((normed * (1.0 + weight.to(torch::kFloat32))).to(input.scalar_type()));
}
```

**关键实现注意事项**：

1. **头文件必须用 `<torch/all.h>` 而非 `<torch/extension.h>`**：`.cu` 文件由 nvcc 编译，`<torch/extension.h>` 会引入 pybind11 头文件，导致 nvcc + pybind11 产生 100+ 编译错误（`Py_buffer undefined` 等）。`<torch/all.h>` 只引入 ATen 和 torch 核心头，不含 pybind11。
2. **函数签名必须用 `at::Tensor&` 引用参数**：头文件 `sgl_kernel_ops.h` 中的声明使用引用参数 `at::Tensor&`，实现必须匹配，否则 C++ name mangling 不一致导致链接失败。值参数 `at::Tensor` 和引用参数 `at::Tensor&` 的 mangled name 不同：
   - 值参数: `_Z7rmsnormN2at6TensorES0_S0_db`
   - 引用参数: `_Z7rmsnormRN2at6TensorES1_S1_db`

### A.3 sgl_kernel_ops.h 修改

**文件路径**: `/tmp/sgl-kernel-build/sgl-kernel/include/sgl_kernel_ops.h`

修改 `sgl_fused_add_rmsnorm` 的声明，从值参数改为引用参数：

```cpp
// 原始声明：
void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl);

// 修改为：
void sgl_fused_add_rmsnorm(
    at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps, bool enable_pdl);
```

**原因**：`common_extension.cc` 中 `m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm)` 使用头文件声明解析函数指针，声明与实现不一致会导致未定义符号。

### A.4 common_extension.cc 修改

**文件路径**: `/tmp/sgl-kernel-build/sgl-kernel/csrc/common_extension.cc`

注释掉 flash sparse attention 的 `m.impl` 注册（保留 `m.def` 模式定义）：

```cpp
  // [SM75 PATCH] Disabled: sparse flash attention requires sm_80+
  // m.impl("fwd_sparse", torch::kCUDA, &flash::mha_fwd_sparse);

  // [SM75 PATCH] Disabled: sparse flash attention requires sm_80+
  // m.impl("varlen_fwd_sparse", torch::kCUDA, &flash::mha_varlen_fwd_sparse);
```

**说明**：`m.def` 保留用于定义 op schema，如果运行时调用会抛出明确的错误信息。`m.impl` 必须注释掉，否则链接时找不到 `flash::mha_fwd_sparse` 符号。Qwen3-4B 不使用 sparse attention，不影响功能。

## 附录 B: Python 层补丁

以下两个文件在每次 `pip install` 后会被覆盖，**必须重新补丁**。

### B.1 load_utils.py — GPU 架构检测

**文件路径**: `/home/ai_env/miniforge3/envs/SGLangTest/lib/python3.11/site-packages/sgl_kernel/load_utils.py`

在 `_load_architecture_specific_ops()` 函数中添加 sm_75 映射：

```python
# 原始代码（第 60-63 行）：
if compute_capability == 90:
    ops_subdir = "sm90"
    variant_name = "SM90 (Hopper/H100 with fast math optimization)"
elif compute_capability is not None:

# 修改为（插入 sm_75 分支）：
if compute_capability == 90:
    ops_subdir = "sm90"
    variant_name = "SM90 (Hopper/H100 with fast math optimization)"
elif compute_capability == 75:
    ops_subdir = "sm75"
    variant_name = "SM75 (Turing/RTX 2080 Ti with source-compiled sm_75 kernels)"
elif compute_capability is not None:
```

**作用**：让 `sgl_kernel` 加载 `sgl_kernel/sm75/common_ops.abi3.so` 而非不存在的 `sm90/sm100` 目录。

### B.2 elementwise.py — 禁用 flashinfer norm

**文件路径**: `/home/ai_env/miniforge3/envs/SGLangTest/lib/python3.11/site-packages/sgl_kernel/elementwise.py`

在模块顶部添加 sm_75 检测，禁用 flashinfer norm 路径：

```python
# 原始代码（第 6-11 行）：
try:
    import flashinfer.norm as _flashinfer_norm
    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False

# 修改为：
try:
    import flashinfer.norm as _flashinfer_norm
    _has_flashinfer = True
    # [SM75 PATCH] flashinfer norm uses CuTe/CUTLASS (sm_80+ only), disable on sm_75
    if torch.cuda.is_available():
        _cc = torch.cuda.get_device_properties(torch.cuda.current_device()).major * 10 + \
              torch.cuda.get_device_properties(torch.cuda.current_device()).minor
        if _cc < 80:
            _has_flashinfer = False
            _flashinfer_norm = None
except ImportError:
    _has_flashinfer = False
```

**作用**：`elementwise.py` 中的 `rmsnorm`/`fused_add_rmsnorm`/`gemma_rmsnorm`/`gemma_fused_add_rmsnorm` 四个函数默认优先使用 flashinfer 的 CuTe 实现（sm_80+），在 sm_75 上会崩溃。此补丁让它们回退到 `torch.ops.sgl_kernel.*` 内部实现（即 rmsnorm_sm75.cu 的 PyTorch 原生回退）。

## 附录 C: 编译步骤

```bash
# 1. 进入源码目录
cd /tmp/sgl-kernel-build/sgl-kernel

# 2. 编译安装（限制并行度避免 OOM）
CMAKE_BUILD_PARALLEL_LEVEL=2 \
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
/home/ai_env/miniforge3/envs/SGLangTest/bin/pip install . --no-build-isolation

# 3. 编译完成后，必须重新补丁 Python 文件
#    （pip install 会覆盖 load_utils.py 和 elementwise.py）
#    补丁内容见附录 B
```

**编译注意事项**：
- `CMAKE_BUILD_PARALLEL_LEVEL=2`：限制并行编译线程数，7+ gencode 目标 × 多 .cu 文件会超出 16GB 内存
- `CMAKE_POLICY_VERSION_MINIMUM=3.5`：cmake 4.x 与 dlpack 的 CMakeLists.txt 不兼容，需此兼容标志
- 编译时间约 3-5 分钟（单 sm_75 gencode，并行度 2）
- 编译日志保存在 `/tmp/sgl-kernel-build-logs/build_sm75_v9.log`

## 附录 D: 测试结果

### D.1 基础功能验证

| 测试项 | 结果 | 备注 |
|--------|------|------|
| sgl_kernel 导入 | 通过 | `import sgl_kernel` 成功 |
| rmsnorm 函数 | 通过 | PyTorch 原生回退，float16 张量 |
| fused_add_rmsnorm 函数 | 通过 | PyTorch 原生回退 |
| gemma_rmsnorm 函数 | 通过 | PyTorch 原生回退 |
| gemma_fused_add_rmsnorm 函数 | 通过 | PyTorch 原生回退 |
| top_p_renorm_prob | 通过 | flashinfer renorm.cu (纯 CUDA) |
| top_k_renorm_prob | 通过 | flashinfer renorm.cu (纯 CUDA) |
| SGLang 服务启动 (Qwen3-4B) | 通过 | triton attention + pytorch sampling |
| 推理请求 (中文) | 通过 | 响应质量正常 |

### D.2 资源占用（Qwen3-4B, context_length=2048）

| 指标 | 值 |
|------|-----|
| 模型权重 | 7.71 GB |
| KV Cache (K) | 4.97 GB |
| KV Cache (V) | 4.97 GB |
| CUDA Graph | 0.79 GB |
| 可用显存 | 2.14 GB |
| 总显存使用 | ~20.0 GB / 22.0 GB |
| 模型加载时间 | 15.61 s |
| CUDA Graph 编译时间 | 80.80 s |
| 服务启动总时间 | ~100 s |

### D.3 基础配置测试

测试模型：Qwen3-4B-Instruct-2507 (float16)，基础参数：`--context-length 2048 --mem-fraction-static 0.85 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph`

| 编号 | 测试名 | 变更参数 | KV Cache (tokens) | KV Cache 大小 | 可用显存 | 推理结果 | 备注 |
|------|--------|----------|-------------------|--------------|---------|---------|------|
| B1 | baseline | 无 (基准) | 72,374 | K=4.97G V=4.97G | 2.14 GB | 通过 | 基准配置，GPU 使用 20394 MiB |
| B2 | ctx1024 | `--context-length 1024` | 72,374 | K=4.97G V=4.97G | 2.13 GB | 通过 | **context-length 不影响 KV Cache 大小**，仅限制单请求最大长度 |
| B3 | mem70 | `--mem-fraction-static 0.70` | 49,673 | K=3.41G V=3.41G | 5.24 GB | 通过 | KV Cache 缩小约 31%，释放约 3GB 显存，适合多模型共存 |
| B4 | mem90 | `--mem-fraction-static 0.90` | 79,941 | K=5.49G V=5.49G | 1.00 GB | 通过 | KV Cache 增大约 10%，仅剩 686 MiB 空闲，**风险高** |
| B5 | flashinfer_sampling | `--sampling-backend flashinfer` | 72,374 | K=4.97G V=4.97G | 2.12 GB | 通过 | flashinfer sampling (纯 CUDA) 在 sm_75 可用 |
| B6 | torch_native_attn | `--attention-backend torch_native` | 72,374 | K=4.97G V=4.97G | 2.29 GB | 通过 | PyTorch 原生注意力，可用显存略多 (无 CUDA graph) |

**关键发现**：

1. **`--context-length` 仅限制单请求最大上下文长度，不影响 KV Cache 分配大小**。KV Cache 大小由 `--mem-fraction-static` 决定。B1 (ctx=2048) 和 B2 (ctx=1024) 的 KV Cache 完全相同（72,374 tokens），只是 `context_len` 日志值不同。
2. **`--mem-fraction-static` 是控制显存分配的核心参数**：0.70 → 49,673 tokens / 5.24 GB 可用；0.85 → 72,374 tokens / 2.14 GB 可用；0.90 → 79,941 tokens / 1.00 GB 可用。
3. **flashinfer sampling 在 sm_75 可用**：flashinfer 的 sampling 和 renorm 是纯 CUDA 实现（不依赖 CuTe/CUTLASS），与 flashinfer attention 不同。
4. **torch_native attention 可用但效率低于 triton**：PyTorch 原生 SDP 注意力兼容 sm_75，但无 KV cache 优化，适合调试。

### D.4 高级配置测试

| 编号 | 测试名 | 变更参数 | KV Cache (tokens) | KV Cache 大小 | 可用显存 | 推理结果 | 备注 |
|------|--------|----------|-------------------|--------------|---------|---------|------|
| A1 | cuda_graph_enabled | 去掉 `--disable-cuda-graph` | 72,374 | K=4.97G V=4.97G | 2.07 GB | 通过 | CUDA graph 占用约 0.79 GB，decode 速度提升 |
| A2 | chunked_prefill_512 | `--chunked-prefill-size 512` | 72,374 | K=4.97G V=4.97G | ~2.3 GB | 通过 | 小 chunk 可能出现 piecewise CUDA graph 不稳定崩溃，可加 `--disable-piecewise-cuda-graph` 规避 |
| A3 | ctx4096 | `--context-length 4096` | 72,374 | K=4.97G V=4.97G | 2.03 GB | 通过 | context-length=4096，KV Cache 大小不变 |
| A4 | ctx4096_mem70 | `--context-length 4096 --mem-fraction-static 0.70` | 49,673 | K=3.41G V=3.41G | 5.21 GB | 通过 | 长上下文 + 低显存占用，5.21 GB 可供其他模型 |
| A5 | chunked_512_no_piecewise | `--chunked-prefill-size 512 --disable-piecewise-cuda-graph` | 72,374 | K=4.97G V=4.97G | ~2 GB | 通过 | 加 `--disable-piecewise-cuda-graph` 后正常 |
| A6 | ctx8192_mem85 | `--context-length 8192` | 72,374 | K=4.97G V=4.97G | ~2 GB | 通过 | context-length=8192，KV Cache 大小不变 |

**关键发现**：

1. **CUDA graph 启用可行**：去掉 `--disable-cuda-graph` 后服务正常启动，CUDA graph 捕获耗时约 80 秒，但后续 decode 速度更快。显存多占约 0.79 GB。
2. **`--chunked-prefill-size 512` 可用但存在不稳定性**：小 chunk 值搭配 piecewise CUDA graph 可能出现 `CUDA error: illegal instruction`（间歇性，非必现）。建议保持默认 2048，或使用 512 时加 `--disable-piecewise-cuda-graph` 规避。
3. **context-length 可安全设为 4096 或 8192**：不增加 KV Cache 显存占用，仅影响单请求最大上下文长度限制。在 `--mem-fraction-static 0.70` 下设为 4096 仍有 5.21 GB 可用显存。
4. **多模型共存场景**：`--mem-fraction-static 0.70` + `--context-length 4096` 可释放约 5 GB 显存给 embedding 模型或其他服务。

### D.5 补充配置测试

| 编号 | 测试名 | 变更参数 | KV Cache (tokens) | KV Cache 大小 | 可用显存 | 推理结果 | 备注 |
|------|--------|----------|-------------------|--------------|---------|---------|------|
| S1 | radix_cache_enabled | 默认（启用） | 72,374 | K=4.97G V=4.97G | 2.11 GB | 通过 | RadixAttention 默认启用，不额外占显存 |
| S2 | radix_cache_disabled | `--disable-radix-cache` | 72,374 | K=4.97G V=4.97G | 2.08 GB | 通过 | 禁用后 KV Cache 大小不变，Radix Tree 在 CPU 端 |
| S3 | schedule_lpm | `--schedule-policy lpm` | 72,374 | K=4.97G V=4.97G | 2.11 GB | 通过 | 前缀匹配优先调度，配合 RadixCache 多轮对话更高效 |
| S4 | max_running_2 | `--max-running-requests 2` | 72,374 | K=4.97G V=4.97G | 2.10 GB | 通过 | 限制最大并发请求数，KV Cache 不变 |
| S5 | max_total_tokens_8192 | `--max-total-tokens 8192` | 8,192 | K=0.56G V=0.56G | 10.91 GB | 通过 | **KV Cache 缩小到 1.12 GB**，释放约 10.9 GB 显存 |
| S6 | enable_metrics | `--enable-metrics` | 72,374 | K=4.97G V=4.97G | 2.11 GB | 通过 | Prometheus 指标端点可用，零显存开销 |
| S7 | schedule_conservative_05 | `--schedule-conservativeness 0.5` | 72,374 | K=4.97G V=4.97G | 2.11 GB | 通过 | 更保守的调度策略，减少显存尖峰风险 |
| S8 | max_prefill_2048 | `--max-prefill-tokens 2048` | 72,374 | K=4.97G V=4.97G | 2.09 GB | 通过 | 限制单次 prefill token 数，防止长 prompt 显存尖峰 |
| S9 | disable_overlap_schedule | `--disable-overlap-schedule` | 72,374 | K=4.97G V=4.97G | 2.09 GB | 通过 | 禁用 overlap 调度，略省显存 |

**关键发现**：

1. **`--max-total-tokens` 是控制显存的精准利器**：设为 8192 时 KV Cache 仅 1.12 GB（vs 默认 9.94 GB），释放约 10.9 GB 给其他模型。比 `--mem-fraction-static` 更精确，直接控制总 token 数上限。
2. **RadixCache 不占额外显存**：启用/禁用 KV Cache 大小完全相同。Radix Tree 在 CPU 端维护，GPU 端零开销。
3. **`--schedule-policy lpm`**：健康咨询场景推荐使用，多轮对话时如果 system prompt 相同，前缀命中率高，可跳过重复 prefill。
4. **`--enable-metrics`**：零开销启用 Prometheus 指标，生产环境建议开启。

### D.6 最大覆盖配置测试

通过最大覆盖→失败隔离策略，对补充测试未覆盖的配置项进行分组测试。

| 编号 | 测试名 | 变更参数 | 推理结果 | 备注 |
|------|--------|----------|---------|------|
| M1 | stream_api | `--stream-interval 2 --incremental-streaming-output --served-model-name Qwen3-4B --enable-cache-report` | 通过 | 流式输出间隔和增量模式均正常；served-model-name 覆写 API 返回模型名；缓存报告可用 |
| M2 | memory_opt | `--cpu-offload-gb 2 --enable-weights-cpu-backup --enable-memory-saver` | 通过 | CPU 卸载 2GB + 权重 CPU 备份 + 内存节省模式均可用。注意：`--cpu-offload-gb` 需设置 LD_LIBRARY_PATH |
| M3 | reasoning_parser | `--reasoning-parser qwen3` | 通过 | Qwen3 思考模式输出到 `reasoning_content` 字段，`content` 为空（正常行为） |
| M4 | quality_params | `--allow-auto-truncate --strip-thinking-cache --radix-eviction-policy lfu --enable-custom-logit-processor` | 通过 | 自动截断超长输入；清理思考缓存；LFU 驱逐策略；自定义 logit 处理器 |
| M5 | schedule_monitor | `--max-queued-requests 10 --show-time-cost --enable-request-time-stats-logging --grammar-backend xgrammar --tool-call-parser qwen3_coder` | 通过 | 请求排队限制；耗时显示；请求时间统计；xgrammar 语法后端；Qwen3 工具调用解析 |

**关键发现**：

1. **`--reasoning-parser qwen3` 行为说明**：启用后 Qwen3 模型的思考内容输出到 `reasoning_content` 字段，`content` 字段为空。这是 Qwen3 thinking mode 的正常行为，不是 bug。客户端需要检查 `reasoning_content` 字段获取思考过程。
2. **`--cpu-offload-gb` 需要设置 LD_LIBRARY_PATH**：CPU 卸载会启动子进程，需要确保 `libcudart.so.13` 可被找到：`export LD_LIBRARY_PATH=/home/ai_env/miniforge3/envs/SGLangTest/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`。此要求仅与 `--cpu-offload-gb` 有关，与 `--enable-memory-saver` 和 `--enable-weights-cpu-backup` 无关。
3. **`--tool-call-parser` 的有效值是 `qwen3_coder`**：不是 `qwen3`。有效值包括 `auto`, `deepseekv3`, `qwen`, `qwen25`, `qwen3_coder` 等。
4. **`--grammar-backend xgrammar` 可用**：xgrammar 作为约束解码后端在 sm_75 上正常工作。
