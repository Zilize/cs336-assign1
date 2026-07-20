#!/usr/bin/env bash
# 构建 / 推送镜像，二者分离：
#   构建需要外网（pip 拉包），在「不连 VPN」时做；
#   推送需要访问内网 ACR（VPC 域名），在「连 VPN」时做。
#
# 用法:
#   ./docker.sh build [TAG]   # 不连 VPN 时构建
#   ./docker.sh push  [TAG]   # 连 VPN 时推送
#   ./docker.sh login         # 连 VPN 时登录 ACR（首次或凭证过期时）
#
# TAG 不传则默认用 git 短 commit（拿不到就用 latest）。

set -euo pipefail

REGISTRY="robocraft-acr-beijing-registry-vpc.cn-beijing.cr.aliyuncs.com"
NAMESPACE="research"
REPO="llm"
DOCKERFILE="Dockerfile.train"
PLATFORM="linux/amd64"

default_tag() {
  git rev-parse --short HEAD 2>/dev/null || echo "latest"
}

CMD="${1:-}"
TAG="${2:-$(default_tag)}"
IMAGE="${REGISTRY}/${NAMESPACE}/${REPO}:${TAG}"

case "$CMD" in
  build)
    echo ">>> 构建镜像: ${IMAGE}"
    docker build --platform "${PLATFORM}" -f "${DOCKERFILE}" -t "${IMAGE}" .
    echo ">>> 构建完成。连上 VPN 后执行:  ./docker.sh push ${TAG}"
    ;;

  push)
    echo ">>> 推送镜像: ${IMAGE}"
    docker push "${IMAGE}"
    echo ">>> 推送完成。"
    ;;

  login)
    echo ">>> 登录 ACR: ${REGISTRY}"
    docker login "${REGISTRY}"
    ;;

  *)
    echo "用法: $0 {build|push|login} [TAG]"
    exit 1
    ;;
esac
