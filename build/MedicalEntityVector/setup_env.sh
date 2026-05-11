# AI辅助生成：GLM-5, 2026-04-18
#!/bin/bash

ENV_NAME="medical_entity_vector"
PYTHON_VERSION="3.9"
PROJECT_DIR="/home/project/MedicalQA/build/MedicalEntityVector"
ENV_FILE="${PROJECT_DIR}/environment.txt"

echo "========================================="
echo "开始创建conda环境: ${ENV_NAME}"
echo "Python版本: ${PYTHON_VERSION}"
echo "========================================="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，是否删除并重新创建？(y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "取消环境创建"
        exit 0
    fi
fi

echo ""
echo "步骤1: 创建conda环境..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

if [ $? -ne 0 ]; then
    echo "错误: conda环境创建失败"
    exit 1
fi

echo ""
echo "步骤2: 激活环境并安装依赖包..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if [ ! -f "${ENV_FILE}" ]; then
    echo "错误: 找不到依赖文件 ${ENV_FILE}"
    exit 1
fi

echo "从 ${ENV_FILE} 读取依赖列表..."
while IFS= read -r package || [ -n "$package" ]; do
    if [ -n "$package" ]; then
        echo "安装: ${package}"
        pip install "${package}"
        if [ $? -ne 0 ]; then
            echo "警告: 安装 ${package} 失败，继续..."
        fi
    fi
done < "${ENV_FILE}"

echo ""
echo "步骤3: 验证环境安装..."
echo "Python版本:"
python --version

echo ""
echo "已安装的包:"
pip list

echo ""
echo "验证关键依赖包..."
python -c "import pymilvus; print(f'pymilvus版本: {pymilvus.__version__}')" && \
python -c "import neo4j; print(f'neo4j版本: {neo4j.__version__}')" && \
python -c "import requests; print(f'requests版本: {requests.__version__}')" && \
python -c "import numpy; print(f'numpy版本: {numpy.__version__}')"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 环境创建成功！"
    echo "环境名称: ${ENV_NAME}"
    echo "Python版本: ${PYTHON_VERSION}"
    echo "========================================="
    echo ""
    echo "使用以下命令激活环境:"
    echo "  conda activate ${ENV_NAME}"
else
    echo ""
    echo "========================================="
    echo "✗ 环境验证失败，请检查安装日志"
    echo "========================================="
    exit 1
fi
