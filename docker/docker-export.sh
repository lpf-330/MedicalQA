#!/bin/bash
# MedicalQA Docker 镜像导出脚本
# 用法: ./docker/docker-export.sh [版本号]
set -e

VERSION="${1:-v1.2.0}"
IMAGE_NAME="medicalqa:${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/images"
OUTPUT_FILE="${OUTPUT_DIR}/medicalqa-${VERSION}.tar.gz"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "导出 Docker 镜像: ${IMAGE_NAME}"
echo "输出文件: ${OUTPUT_FILE}"
echo "============================================"

docker save "${IMAGE_NAME}" | gzip > "${OUTPUT_FILE}"

echo ""
echo "导出完成: ${OUTPUT_FILE}"
echo "文件大小: $(du -h "${OUTPUT_FILE}" | cut -f1)"
echo ""
echo "导入命令: gunzip -c ${OUTPUT_FILE} | docker load"
