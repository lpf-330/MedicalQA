# MedicalQA Docker 镜像

全环境备份镜像，包含自编译推理框架（SGLang + sglang-kernel）、自量化模型（AWQ）及完整依赖。

## 目录结构

```
docker/
├── Dockerfile              # 4 层构建定义
├── docker-build.sh         # 构建脚本
├── docker-export.sh        # 导出脚本
├── docker-push-acr.sh      # 阿里云ACR推送脚本
├── wheels/
│   └── sglang_kernel.tar   # 自编译 sglang-kernel 0.4.3（从 site-packages 打包）
└── images/                  # 本地离线镜像（不入Git）
```

## 镜像信息

| 项 | 值 |
|----|-----|
| 镜像名 | `medicalqa:v1.2.0` |
| 基础镜像 | `nvidia/cuda:12.9.1-devel-ubuntu22.04` |
| Python | 3.11 (Miniforge3) |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.12.post1 |
| sglang-kernel | 0.4.3 (自编译，源码修改版) |
| transformers | 5.6.0 |
| 镜像大小 | ~46GB (压缩后 ~16GB) |

## 镜像详细内容

### 系统层 (Layer 1, ~3GB)

- 基础：nvidia/cuda:12.9.1-devel-ubuntu22.04
- apt 包：git, wget, curl, ca-certificates, libffi-dev, libssl-dev, libibverbs-dev, librdmacm-dev, build-essential, cmake, libnuma1

### 依赖层 (Layer 2, ~9GB)

- Miniforge3 (conda)
- Python 3.11 conda 环境
- sglang-kernel 0.4.3 自编译包（直接解压到 site-packages）
- sglang 0.5.12.post1 (--no-deps 安装，避免与自编译 kernel 冲突)
- 全部 pip 依赖（见 requirements.txt）

### 模型层 (Layer 3, ~10GB)

| 模型 | 路径 | 大小 | 用途 |
|------|------|------|------|
| Qwen3-4B-AWQ | `/app/base_models/Qwen3-4B-AWQ` | ~2.5GB | 健康咨询推理 (SGLang) |
| MedPsy-4B-AWQ | `/app/base_models/MedPsy-4B-AWQ` | ~2.5GB | 健康评估推理 (SGLang) |
| bge-large-zh-v1.5 | `/app/base_models/models--BAAI--bge-large-zh-v1.5` | ~2.5GB | 向量编码 (sentence-transformers) |
| ernie-health-zh | `/app/base_models/ernie-health-zh` | ~0.4GB | 意图分类 (transformers, Embedding余弦相似度) |
| nlp_raner | `/app/base_models/nlp_raner_named-entity-recognition_chinese-base-cmeee` | ~0.4GB | 命名实体识别 (transformers, CRF+Viterbi) |

### 代码层 (Layer 4, ~5MB)

- `/app/src/` — 项目源码
- `/app/config/` — 配置文件（含 application.example.yaml，**不含** application.yaml 真实配置）

## Dockerfile 分层

| 层 | 内容 | 大小 | 说明 |
|----|------|------|------|
| Layer 1 | 系统依赖 (apt) | ~3GB | 改动最少 |
| Layer 2 | Miniforge + pip 依赖 + sglang-kernel | ~9GB | 最耗时 |
| Layer 3 | 模型文件 (COPY) | ~10GB | AWQ 量化版 + bge + ernie-health-zh + ner |
| Layer 4 | 项目代码 + 配置 (COPY) | ~5MB | 最常改，放最上层 |

## 阿里云 ACR 镜像仓库

镜像已推送至阿里云容器镜像服务：

```
crpi-mgzpewsi19pa9wsm.cn-hangzhou.personal.cr.aliyuncs.com/medical_qa_docker/medicalqa:v1.2.0
```

### 拉取

```bash
docker pull crpi-mgzpewsi19pa9wsm.cn-hangzhou.personal.cr.aliyuncs.com/medical_qa_docker/medicalqa:v1.2.0
```

### 推送新版本

```bash
export ACR_PASSWORD="<密码>"
./docker/docker-push-acr.sh v1.2.0
```

## 快速部署

### 方式一：从阿里云 ACR 拉取（推荐）

```bash
docker pull crpi-mgzpewsi19pa9wsm.cn-hangzhou.personal.cr.aliyuncs.com/medical_qa_docker/medicalqa:v1.2.0
docker tag crpi-mgzpewsi19pa9wsm.cn-hangzhou.personal.cr.aliyuncs.com/medical_qa_docker/medicalqa:v1.2.0 medicalqa:v1.2.0
docker run --gpus all -p 8001:8001 medicalqa:v1.2.0
```

### 方式二：从 tar.gz 导入（离线部署）

```bash
gunzip -c docker/images/medicalqa-v1.2.0.tar.gz | docker load
docker run --gpus all -p 8001:8001 medicalqa:v1.2.0
```

### 方式三：从源码构建

```bash
./docker/docker-build.sh v1.2.0
```

加 `--no-cache` 强制全量重建：

```bash
./docker/docker-build.sh v1.2.0 --no-cache
```

## 运行与验证

```bash
# 启动容器
docker run --gpus all -p 8001:8001 medicalqa:v1.2.0

# 健康检查
curl http://localhost:8001/health

# 快速验证 GPU + 依赖
docker run --rm medicalqa:v1.2.0 conda run -n MedicalQA python -c \
  "import torch; print(torch.cuda.get_device_name(0)); import sgl_kernel; print('sgl_kernel: OK')"
```

## 导出与推送

```bash
# 导出为tar.gz
./docker/docker-export.sh v1.2.0
# 输出: docker/images/medicalqa-v1.2.0.tar.gz

# 推送到阿里云ACR
export ACR_PASSWORD="<密码>"
./docker/docker-push-acr.sh v1.2.0
```

## sglang-kernel 说明

sglang-kernel 0.4.3 是基于源码修改后的自编译版本，pip 上无对应发行包。构建时通过 `docker/wheels/sglang_kernel.tar` 直接解压到容器 site-packages，避免容器内重新编译。

如需更新 sglang-kernel，在宿主机重新打包：

```bash
SITE="/home/ai_env/miniforge3/envs/MedicalQA/lib/python3.11/site-packages"
tar -cf docker/wheels/sglang_kernel.tar -C "$SITE" sgl_kernel sglang_kernel-0.4.3.dist-info
```

## 注意事项

- 需要宿主机安装 NVIDIA GPU 驱动（>= 595.x）和 NVIDIA Container Toolkit
- 启动前需在 `/app/config/` 下创建 `application.yaml`，填入真实的 Neo4j/Milvus 连接信息
- 默认 runtime 设为 `nvidia`，所有容器自动可用 GPU
- 镜像包含全部5个模型，拉取后无需额外下载即可运行
