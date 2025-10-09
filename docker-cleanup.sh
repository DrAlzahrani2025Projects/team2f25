#!/usr/bin/env bash
# ------------------------------------------------------------
# Team2f25 - Cleanup Script
# Stops and removes container, image, and optionally the volume
# ------------------------------------------------------------
set -euo pipefail

APP_NAME="${APP_NAME:-team2f25-app}"
IMAGE_TAG="${IMAGE_TAG:-team2f25-app:latest}"
VOL_NAME="${VOL_NAME:-team2f25-ollama}"
PURGE_MODELS="${PURGE_MODELS:-0}"  # set to 1 to remove Ollama models too

echo "🧱 [1/3] Stopping & removing container: ${APP_NAME}"
docker rm -f "${APP_NAME}" 2>/dev/null || true

echo "🗑️ [2/3] Removing image: ${IMAGE_TAG}"
docker rmi "${IMAGE_TAG}" 2>/dev/null || true

if [[ "${PURGE_MODELS}" == "1" ]]; then
  echo "💥 [3/3] Removing Ollama volume: ${VOL_NAME}"
  docker volume rm "${VOL_NAME}" 2>/dev/null || true
else
  echo "📦 [3/3] Keeping Ollama model volume (${VOL_NAME})."
  echo "      (Run with PURGE_MODELS=1 to delete it.)"
fi

echo "✅ Cleanup complete."
