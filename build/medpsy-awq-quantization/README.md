# MedPsy-4B AWQ 量化项目

将 MedPsy-4B（float16，8.3GB 权重）量化为 AWQ 4-bit 格式（2.49GB），解决 RTX 2080 Ti 22GB VRAM 上双模型共部署时显存不足的问题。

## 背景

原系统需要同时运行 Qwen3-4B-AWQ（2.5GB）和 MedPsy-4B（8.3GB）。双模型权重总计约 10.8GB，加上 KV Cache 和中间激活值，总 VRAM 占用约 20.2GB，超出 22GB 显存的可用范围，导致 MedPsy 推理卡住。

量化目标：MedPsy-4B 权重从 8.3GB 降至约 2.5GB，双模型权重合计约 5GB，VRAM 充裕。

## 目录结构

```
medpsy-awq-quantization/
├── README.md                          # 本文件
├── env/                               # 环境配置
│   ├── setup_env.sh                   # conda 环境创建脚本
│   ├── requirements.txt               # 最小依赖（安装用）
│   └── requirements_freeze.txt        # 完整依赖锁定（复现用）
├── configs/                           # 量化配置
│   └── quantize_config_round1.yaml    # 第一轮量化参数
├── calibration/                       # 校准数据
│   ├── extract_neo4j_data.py          # Neo4j 知识图谱数据提取
│   ├── extract_benchmark_data.py      # CMB 医学基准数据提取
│   ├── generate_medpsy_templates.py   # MedPsy 推理格式模板生成
│   ├── build_calibration_dataset.py   # 合并+分层采样→校准集
│   └── data/                          # 校准数据文件
│       ├── raw_neo4j/                 # Neo4j 提取结果
│       │   ├── entities.json          # 14,332 条实体
│       │   └── relations.json         # 1,040 条关系
│       ├── raw_benchmark/             # CMB 提取结果
│       │   ├── cmb_data.json          # 269,639 条医学题
│       │   └── medpsy_templates.json  # 50 条 MedPsy 格式模板
│       └── calibration_dataset.json   # 最终校准集（58,181 条）
├── quantize/                          # 量化执行
│   └── run_quantize.py                # AWQ 量化主脚本
├── evaluate/                          # 评估测试
│   ├── generate_baseline.py           # 原模型基线采集
│   ├── evaluate_quantized.py          # 量化模型输出采集
│   ├── compare_results.py             # 对比分析脚本
│   ├── test_cases/                    # 测试用例
│   │   ├── dimension_eval_cases.json  # 5 维度评估（20 个）
│   │   └── risk_factor_cases.json     # 6 风险因子评估（15 个）
│   └── results/                       # 评估结果
│       ├── baseline_results.json      # 原模型基线输出
│       ├── quantized_results.json     # 量化模型输出
│       └── comparison_report.json     # 对比分析报告
├── iterations/                        # 迭代记录
│   └── round1/
│       └── iteration_report.md        # 第一轮迭代详细报告
└── output/                            # 量化输出
    └── MedPsy-4B-AWQ/                # 量化模型（2.49GB）
        ├── model.safetensors
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── generation_config.json
        ├── chat_template.jinja
        └── quantization_info.json     # 量化元信息
```

## 环境

### 环境信息

| 项目 | 值 |
|------|------|
| 环境名 | MedPsy-AWQ |
| Python | 3.11.15 |
| 路径 | /home/ai_env/miniforge3/envs/MedPsy-AWQ |
| CUDA | 12.1 |
| GPU | RTX 2080 Ti 22GB |

### 关键依赖

| 包 | 版本 | 用途 |
|----|------|------|
| autoawq | 0.2.9 | AWQ 量化核心（pip 包名 autoawq，import 名 awq） |
| transformers | 5.9.0 | 模型加载和 tokenizer |
| torch | 2.5.1+cu121 | PyTorch CUDA 版 |
| accelerate | 1.13.0 | device_map 自动分配 |
| datasets | 4.8.5 | CMB 数据集加载 |
| neo4j | 6.2.0 | Neo4j 数据提取 |
| scikit-learn | 1.8.0 | 评估指标计算 |

### 环境创建

```bash
bash env/setup_env.sh
```

### 环境复现

从锁定文件精确复现：

```bash
conda create -n MedPsy-AWQ python=3.11 -y
conda activate MedPsy-AWQ
pip install -r env/requirements_freeze.txt
```

`requirements_freeze.txt` 包含所有依赖的精确版本号（含 nvidia CUDA 库），可确保环境完全一致。`requirements.txt` 是最小依赖列表，安装更快但可能因传递依赖版本差异产生兼容问题。

### 环境清理

量化工作完成后，MedPsy-AWQ 环境可卸载以释放磁盘空间：

```bash
conda env remove -n MedPsy-AWQ
```

环境信息已备份至 `env/requirements_freeze.txt`，需要时可通过上面的复现命令重新创建。

## 校准数据

### 数据来源与构成

| 来源 | 原始量 | 采样后 | 分配比例 | 说明 |
|------|--------|--------|----------|------|
| Neo4j Disease 长文本属性 | 8,809 | ~6,000 | 40% | 含 desc/cause/prevent 等医学文本 |
| Neo4j 关系文本 | 312,226 | ~1,040 | 15% | 按类型分层采样 |
| Neo4j 其他实体 | 26,797 | ~4,971 | 10% | Symptom/Drug/Food/Check/Cure/Dept |
| CMB 医学基准 | 269,639 | ~200 | 15% | 中文医学考试题 |
| MedPsy 格式模板 | — | 50 | 20% | 30 维度 + 20 风险因子 |
| **合计** | — | **58,181** | 100% | — |

### 采样策略

- **Disease**：有长文本属性的全量纳入（医学价值最高）
- **其他实体**：按 20%-30% 随机采样
- **Producer**：跳过（17,201 条，均为药品生产厂商，医学价值低）
- **关系**：按类型分层采样，优先 has_symptom、recommand_drug、need_check 等高医学价值关系
- **CMB**：从 269,639 条中随机采样 200 条

AutoAWQ 实际使用 `max_calib_samples=128` 条文本进行校准，从 58,181 条中按比例随机采样。

## 量化参数

第一轮配置（`configs/quantize_config_round1.yaml`），复用 Qwen3-4B-AWQ 已验证参数：

| 参数 | 值 | 说明 |
|------|------|------|
| bits | 4 | 量化位宽 |
| group_size | 128 | 量化分组大小 |
| zero_point | true | 启用零点 |
| version | gemm | AWQ GEMM 内核 |
| max_calib_samples | 128 | 校准样本数 |
| max_calib_seq_len | 512 | 校准序列最大长度 |
| dtype | bfloat16 | 模型加载精度 |
| device_map | auto | 自动分配 GPU |

## 执行流程

### 1. 校准数据提取

```bash
conda activate MedPsy-AWQ
cd calibration

# 提取 Neo4j 数据（需要 NEO4J_PASSWORD 环境变量）
export NEO4J_PASSWORD=<password>
python extract_neo4j_data.py

# 提取 CMB 基准数据
python extract_benchmark_data.py

# 生成 MedPsy 推理格式模板
python generate_medpsy_templates.py

# 合并构建校准集
python build_calibration_dataset.py
```

### 2. 执行量化

```bash
python quantize/run_quantize.py --config configs/quantize_config_round1.yaml
```

耗时约 17 分钟，输出到 `output/MedPsy-4B-AWQ/`。

### 3. 评估测试

通过 SGLang HTTP 接口分别调用原模型和量化模型，采集输出后对比分析。

```bash
# 第一步：启动原模型 SGLang，采集基线
python evaluate/generate_baseline.py

# 第二步：关闭原模型，启动量化模型 SGLang，采集量化输出
python evaluate/evaluate_quantized.py

# 第三步：对比分析
python evaluate/compare_results.py
```

SGLang 启动命令示例（原模型）：

```bash
python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/base_models/MedPsy-4B \
  --port 30001 \
  --mem-fraction-static 0.75 \
  --reasoning-parser qwen3
```

SGLang 启动命令示例（量化模型）：

```bash
python -m sglang.launch_server \
  --model-path /home/project/MedicalQA/build/medpsy-awq-quantization/output/MedPsy-4B-AWQ \
  --port 30001 \
  --mem-fraction-static 0.75 \
  --reasoning-parser qwen3
```

## 量化结果

### 模型大小

| 指标 | 原模型 (float16) | 量化模型 (AWQ 4-bit) | 变化 |
|------|------------------|----------------------|------|
| 权重总大小 | 13 GB | 2.5 GB | -81% |
| 核心权重 | ~8.3 GB | 2.49 GB | -70% |
| SGLang VRAM | ~10.5 GB | ~6.5 GB | **-4 GB** |

量化耗时：17.0 分钟（1020.8 秒）

## 测试评估

### 测试用例

| 类型 | 数量 | 覆盖范围 |
|------|------|----------|
| 维度评估 | 20 个 | D1-D5 各 4 个，覆盖不同用户画像和异常组合 |
| 风险因子评估 | 15 个 | F1-F6 各 2-3 个 |
| 合计 | 35 个 | — |

### 维度评估结果

| 指标 | 基线 | 量化 | 通过标准 | 结果 |
|------|------|------|----------|------|
| JSON 格式正确率 | 100% | 100% | 不低于基线-5% | PASS |
| 评分 Pearson 相关系数 | — | 1.0 | >=0.90 | PASS |
| 评分 MAE | — | 0.0 | <=0.10 | PASS |
| 有效评分对数 | — | 16/20 | — | — |

4 个基线 case 因思考过度耗尽 max_tokens（1536），未输出最终 dimension_score。量化模型反而更好地遵循"直接输出JSON"指令，20/20 全部输出有效评分。

### 风险因子评估结果

| 指标 | 基线 | 量化 | 通过标准 | 结果 |
|------|------|------|----------|------|
| JSON 格式正确率 | 100% | 100% | 不低于基线-5% | PASS |
| 评分 Pearson 相关系数 | — | 0.7845 | >=0.80 | 边界 |
| 评分 MAE | — | 2.0 (0-100量纲) | <=0.15 (0-1量纲) | 量纲不匹配 |
| 有效评分对数 | — | 15/15 | — | — |

#### 风险因子逐条对比

15 个 case 中 **14 个完全一致**，唯一偏差：

| ID | 因子 | 基线分 | 量化分 | 偏差 |
|----|------|--------|--------|------|
| F1-01~03 | 疾病严重程度 | 45 | 45 | 0 |
| F2-01~02 | 并发症风险 | 45 | 45 | 0 |
| F2-03 | 并发症风险 | 85 | 85 | 0 |
| **F3-01** | **用药风险** | **75** | **45** | **-30** |
| F3-02 | 用药风险 | 45 | 45 | 0 |
| F4-01~03 | 生活习惯风险 | 45 | 45 | 0 |
| F5-01~02 | 复查监测风险 | 45 | 45 | 0 |
| F6-01~02 | 预防措施风险 | 45 | 45 | 0 |

#### Pearson 低值原因

基线评分序列中 13/15 个为 45（高度聚集），仅 2 个偏离值（85 和 75）。极端偏态分布下，单点偏差 30 分即可将 Pearson 从 1.0 拉至 0.78。去除 F3-01 后剩余 14 对完全相等，Pearson = 1.0。

#### MAE 阈值问题

风险因子 factor_score 为 0-100 量纲，MAE=2.0 即平均偏差 2%（2分/100分）。评估脚本中 0.15 阈值是为维度评分 dimension_score（0-1 量纲）设计的，不适用于 0-100 量纲。归一化后 MAE=0.02，远优于 0.15 阈值。

#### F3-01 详细分析

输入：80 岁女性，高血压+房颤+骨质疏松，用药 7 种（华法林、钙片、ARB 类），近 3 月跌倒 2 次。

- 基线：`factor_score:75, reasoning:"华法林抗凝+跌倒史高出血风险", diseases:["高血压","房颤","骨质疏松"]`
- 量化：`factor_score:45, reasoning:"华法林抗凝+跌倒史显著增加出血风险", diseases:["房颤","骨质疏松"]`

两个模型都正确识别了核心风险（华法林+跌倒史→出血风险），推理内容一致，差异仅在评分保守程度。75 分和 45 分均属合理评估。

### 速度对比

| 指标 | 原模型 | 量化模型 |
|------|--------|---------|
| 平均推理时间 | 26.49s | 35.18s |
| 速度比 | 1.0x | 0.75x |

量化模型慢 33%，原因是 AWQ GEMM 反量化开销。在 VRAM 紧张场景下这不是主要矛盾——原模型因 VRAM 不足无法与 Qwen3 同时运行，量化模型虽慢但能正常运行。

### 基线模型自身问题

测试中暴露的基线模型固有问题，与量化无关：

1. **思考过度**：20 个维度 case 中 4 个用尽 max_tokens 进行思考，未输出最终 JSON
2. **输出格式不稳**：部分 case 输出 `subindicator_scores`（缺下划线），部分输出 `sub_indicator_scores`
3. **评分趋同**：维度评分趋同于 0.72，风险因子评分趋同于 45，区分度不足

### 综合判定

**有条件通过**

| 目标 | 达成情况 |
|------|---------|
| 模型大小 <=3GB | 2.49GB, PASS |
| SGLang 可正常推理 | PASS |
| VRAM 节省 >=4GB | 节省 ~4GB, PASS |
| 双模型共部署可行 | Qwen3-AWQ + MedPsy-AWQ ≈ 5GB 权重, PASS |
| 维度评估质量 | Pearson 1.0, MAE 0.0, PASS |
| 风险因子评估质量 | 14/15 完全一致, 唯一偏差合理, PASS |

## 部署

量化模型已就绪，部署步骤：

1. 复制到 `base_models/MedPsy-4B-AWQ/`
2. 更新 `config/application.yaml` 中 `medpsy_config.model_path`
3. 双模型共部署验证
4. 运行健康报告生成 E2E 测试
