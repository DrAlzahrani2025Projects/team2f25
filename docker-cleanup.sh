#!/usr/bin/env bash
# Stop & remove the team2f25 container (idempotent)
# Usage: ./docker-cleanup.sh [-i|--image]   # also remove the image
set -euo pipefail

IMAGE="team2f25:latest"
CONTAINER="team2f25"

echo "🛑 Stopping container (if running)…"
docker stop "${CONTAINER}" >/dev/null 2>&1 || true

echo "🗑️  Removing container (if exists)…"
docker rm "${CONTAINER}" >/dev/null 2>&1 || true

if [[ "${1:-}" == "-i" || "${1:-}" == "--image" ]]; then
  echo "🧽 Removing image ${IMAGE} (if exists)…"
  docker rmi "${IMAGE}" >/dev/null 2>&1 || true
fi

echo "✅ Cleanup complete."
