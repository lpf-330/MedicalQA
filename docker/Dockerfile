FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniforge3
ENV CONDA_ENV=MedicalQA

# ============================================================================
# Layer 1: 系统依赖
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        libffi-dev libssl-dev \
        libibverbs-dev librdmacm-dev \
        build-essential cmake \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Layer 2: Miniforge + conda 环境 + pip 依赖
# ============================================================================
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN conda create -n ${CONDA_ENV} python=3.11 -y \
    && conda clean -afy

# sglang-kernel: 自编译包（从宿主机 site-packages 打包为 tar，直接解压到 site-packages）
COPY docker/wheels/sglang_kernel.tar /tmp/sglang_kernel.tar

# sglang: --no-deps 安装（其依赖 sglang-kernel==0.4.2.post2 与自编译 0.4.3 冲突）
# 其余: 从过滤后的 requirements 安装
COPY requirements.txt /tmp/requirements.txt

RUN SITE_PKGS="${CONDA_DIR}/envs/${CONDA_ENV}/lib/python3.11/site-packages" \
    && tar -xf /tmp/sglang_kernel.tar -C "${SITE_PKGS}" \
    && grep -v "^sglang-kernel==" /tmp/requirements.txt \
       | grep -v "^sglang==" > /tmp/requirements.filtered.txt \
    && conda run -n ${CONDA_ENV} pip install --no-cache-dir --no-deps sglang==0.5.12.post1 \
    && conda run -n ${CONDA_ENV} pip install --no-cache-dir -r /tmp/requirements.filtered.txt \
    && rm /tmp/sglang_kernel.tar /tmp/requirements.txt /tmp/requirements.filtered.txt \
    && conda clean -afy

# ============================================================================
# Layer 3: 模型文件
# ============================================================================
COPY base_models/Qwen3-4B-AWQ /app/base_models/Qwen3-4B-AWQ
COPY base_models/MedPsy-4B-AWQ /app/base_models/MedPsy-4B-AWQ
COPY base_models/models--BAAI--bge-large-zh-v1.5 /app/base_models/models--BAAI--bge-large-zh-v1.5
COPY base_models/ernie-health-zh /app/base_models/ernie-health-zh
COPY base_models/nlp_raner_named-entity-recognition_chinese-base-cmeee /app/base_models/nlp_raner_named-entity-recognition_chinese-base-cmeee

# ============================================================================
# Layer 4: 项目代码 + 配置
# ============================================================================
COPY src/ /app/src/
COPY config/ /app/config/

WORKDIR /app

EXPOSE 8001

CMD ["conda", "run", "-n", "MedicalQA", "python", "src/main.py"]
