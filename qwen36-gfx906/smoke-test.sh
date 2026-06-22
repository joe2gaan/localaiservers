#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-}"
if [[ -z "${PORT}" ]]; then
  for env_file in \
    "${SCRIPT_DIR}/.deploy.runtime.env" \
    "${SCRIPT_DIR}/.deploy.defaults.env" \
    "${SCRIPT_DIR}/deploy.env"; do
    if [[ -f "${env_file}" ]]; then
      PORT="$(awk -F= '$1 == "PORT" {print substr($0, index($0, "=") + 1); exit}' "${env_file}")"
      [[ -n "${PORT}" ]] && break
    fi
  done
fi
PORT="${PORT:-8001}"

export ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}/v1}"

echo "waiting for health endpoint: ${ENDPOINT}/health"
for _ in {1..60}; do
  if curl -sS "${ENDPOINT}/health" >/dev/null; then
    break
  fi
  sleep 5
done

curl -sS "${ENDPOINT}/models" >/dev/null
echo "smoke test passed: ${ENDPOINT}/models"
