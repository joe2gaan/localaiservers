#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deploy.env"

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
