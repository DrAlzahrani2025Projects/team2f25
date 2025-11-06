#!/usr/bin/env bash
set -euo pipefail

# If you started with DETACH=true, we can stop the named container.
NAME="${NAME:-csusb-internship-agent}"
IMAGE="${IMAGE:-internship-chatbot}"
REMOVE_IMAGE="${REMOVE_IMAGE:-false}"

echo "==> Stopping/removing container (if running): ${NAME}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

# Also stop any container currently publishing port 5002 (safety net)
CID_ON_PORT=$(docker ps --filter "publish=5002" --format "{{.ID}}" || true)
if [[ -n "${CID_ON_PORT}" ]]; then
  echo "==> Stopping container exposing port 5002: ${CID_ON_PORT}"
  docker stop "${CID_ON_PORT}" >/dev/null 2>&1 || true
fi

if [[ "${REMOVE_IMAGE}" == "true" ]]; then
  echo "==> Removing image: ${IMAGE}"
  docker rmi "${IMAGE}" >/dev/null 2>&1 || true
fi

echo "✓ Cleanup done."
