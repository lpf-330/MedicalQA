#!/bin/bash
# MedicalQA Docker 镜像构建脚本
# 用法: ./docker/docker-build.sh [版本号] [--no-cache]
set -e

VERSION="${1:-v1.2.0}"
NO_CACHE=""
if [[ "$2" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
fi

IMAGE_NAME="medicalqa:${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================"
echo "构建 Docker 镜像: ${IMAGE_NAME}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "============================================"

# 从项目根目录构建（.dockerignore 在根目录生效）
docker build ${NO_CACHE} -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${PROJECT_ROOT}"

echo ""
echo "构建完成: ${IMAGE_NAME}"
echo "运行命令: docker run --gpus all -p 8001:8001 ${IMAGE_NAME}"
echo "导出命令: ./docker/docker-export.sh ${VERSION}"
