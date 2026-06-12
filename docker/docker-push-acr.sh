#!/bin/bash
# 阿里云ACR推送脚本
set -e

VERSION="${1:-v1.2.0}"
ACR_REGISTRY="crpi-mgzpewsi19pa9wsm.cn-hangzhou.personal.cr.aliyuncs.com"
ACR_NAMESPACE="medical_qa_docker"
ACR_REPO="medicalqa"
LOCAL_IMAGE="medicalqa:${VERSION}"
ACR_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/${ACR_REPO}:${VERSION}"

echo "============================================"
echo "推送 Docker 镜像到阿里云 ACR"
echo "本地镜像: ${LOCAL_IMAGE}"
echo "远程镜像: ${ACR_IMAGE}"
echo "============================================"

# 登录（密码通过环境变量传入，避免泄露到命令行历史）
if [ -n "${ACR_PASSWORD}" ]; then
    printf '%s\n' "${ACR_PASSWORD}" | docker login --username=nick3755471836 --password-stdin "${ACR_REGISTRY}"
else
    echo "请设置 ACR_PASSWORD 环境变量"
    exit 1
fi

# 打标签
docker tag "${LOCAL_IMAGE}" "${ACR_IMAGE}"
echo "标签已创建: ${ACR_IMAGE}"

# 推送
docker push "${ACR_IMAGE}"
echo ""
echo "推送完成: ${ACR_IMAGE}"
echo "拉取命令: docker pull ${ACR_IMAGE}"
