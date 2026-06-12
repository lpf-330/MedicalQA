#!/bin/bash
set -e

ENV_NAME="MedPsy-AWQ"
PYTHON_VERSION="3.11"
CONDA_PATH="/home/ai_env/miniforge3"

echo "=== 创建 MedPsy-AWQ conda 环境 ==="

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
    echo "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 激活环境
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== 安装依赖 ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "=== 验证关键包 ==="
python -c "import autoawq; print(f'autoawq: {autoawq.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== 环境创建完成 ==="
echo "使用方式: conda activate ${ENV_NAME}"
