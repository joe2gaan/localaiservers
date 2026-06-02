#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_REPO="${DOCKERHUB_REPO:-joe2gaan/localaiservers}"
DEPLOY_IMAGE="${DEPLOY_IMAGE:-qwen36-gfx906-c1-topk8-fastpath-reproducible}"
IMAGE_VARIANT="${IMAGE_VARIANT:-qwen36-gfx906-c1-topk8-runtime}"
ARCHIVE_PATH="${ARCHIVE_PATH:-}"
PUBLISH_PUSH="${PUBLISH_PUSH:-0}"
PUBLISH_LATEST="${PUBLISH_LATEST:-1}"
PUBLISH_LOAD="${PUBLISH_LOAD:-1}"

RUN_DIR="$(pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./publish-dockerhub.sh [archive.docker.tar]

Defaults:
  DOCKERHUB_REPO=joe2gaan/localaiservers
  IMAGE_VARIANT=qwen36-gfx906-c1-topk8-runtime
  PUBLISH_PUSH=0

Examples:
  ./publish-dockerhub.sh
  PUBLISH_PUSH=1 ./publish-dockerhub.sh
  DOCKERHUB_REPO=joe2gaan/localaiservers PUBLISH_PUSH=1 ./publish-dockerhub.sh ./.repro-docker-archives/qwen36-gfx906-c1-topk8-fastpath-reproducible.docker.tar

If this build used an isolated private Docker daemon, source the generated env first:
  source ./.deploy.docker-host.env
  PUBLISH_PUSH=1 ./publish-dockerhub.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  ARCHIVE_PATH="$1"
fi

if [[ -z "${ARCHIVE_PATH}" ]]; then
  default_archive="${RUN_DIR}/.repro-docker-archives/${DEPLOY_IMAGE//\//_}.docker.tar"
  if [[ -f "${default_archive}" ]]; then
    ARCHIVE_PATH="${default_archive}"
  else
    ARCHIVE_PATH="$(find "${RUN_DIR}/.repro-docker-archives" -maxdepth 1 -type f -name '*.docker.tar' 2>/dev/null | sort | tail -n 1 || true)"
  fi
fi

if [[ -z "${ARCHIVE_PATH}" || ! -f "${ARCHIVE_PATH}" ]]; then
  echo "error: Docker archive not found. Pass archive path or build first." >&2
  usage >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker CLI not found" >&2
  exit 1
fi

deploy_sha="unknown"
if [[ -f "${RUN_DIR}/deploy.sh" ]]; then
  deploy_sha="$(sha256sum "${RUN_DIR}/deploy.sh" | awk '{print $1}')"
fi
deploy_tag="${deploy_sha:0:8}"
archive_sha="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
archive_tag="${archive_sha:0:12}"

source_ref="${DEPLOY_IMAGE}:latest"
version_tag="${DOCKERHUB_REPO}:${IMAGE_VARIANT}-${deploy_tag}"
archive_hash_tag="${DOCKERHUB_REPO}:${IMAGE_VARIANT}-archive-${archive_tag}"
latest_tag="${DOCKERHUB_REPO}:${IMAGE_VARIANT}-latest"

echo "Docker Hub repo: ${DOCKERHUB_REPO}"
echo "Archive: ${ARCHIVE_PATH}"
echo "Archive sha256: ${archive_sha}"
echo "deploy.sh sha256: ${deploy_sha}"
echo "Local source image: ${source_ref}"
echo "Tags:"
echo "  ${version_tag}"
echo "  ${archive_hash_tag}"
if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "  ${latest_tag}"
fi

if [[ "${PUBLISH_LOAD}" == "1" ]]; then
  echo "Loading Docker archive..."
  docker load -i "${ARCHIVE_PATH}"
fi

if ! docker image inspect "${source_ref}" >/dev/null 2>&1; then
  if docker image inspect "${DEPLOY_IMAGE}" >/dev/null 2>&1; then
    source_ref="${DEPLOY_IMAGE}"
  else
    echo "error: loaded image ${DEPLOY_IMAGE}:latest not found after docker load" >&2
    exit 1
  fi
fi

docker tag "${source_ref}" "${version_tag}"
docker tag "${source_ref}" "${archive_hash_tag}"
if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  docker tag "${source_ref}" "${latest_tag}"
fi

echo "Tagged image successfully."

if [[ "${PUBLISH_PUSH}" != "1" ]]; then
  cat <<EOF

Dry run complete. To push to Docker Hub:
  docker login
  PUBLISH_PUSH=1 ./publish-dockerhub.sh "${ARCHIVE_PATH}"

Pull command after publish:
  docker pull ${version_tag}
EOF
  exit 0
fi

echo "Pushing ${version_tag}"
docker push "${version_tag}"
echo "Pushing ${archive_hash_tag}"
docker push "${archive_hash_tag}"
if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "Pushing ${latest_tag}"
  docker push "${latest_tag}"
fi

cat <<EOF

Published:
  docker pull ${version_tag}
  docker pull ${archive_hash_tag}
EOF
