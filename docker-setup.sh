#!/usr/bin/env bash
# ------------------------------------------------------------
# Team2f25 - Docker Setup Script
# Builds and starts the Streamlit + Ollama container
# ------------------------------------------------------------
set -euo pipefail

# --- Config (you can change ports if needed) ---
APP_NAME="${APP_NAME:-team2f25-app}"
IMAGE_TAG="${IMAGE_TAG:-team2f25-app:latest}"
UI_PORT="${UI_PORT:-5002}"          # Streamlit web UI
OLLAMA_PORT="${OLLAMA_PORT:-11434}" # Ollama API port
BASE_PATH="${BASE_PATH:-team2f25}"  # subpath for Streamlit
MODEL_NAME="${MODEL_NAME:-qwen2:0.5b}"  # LLM model
VOL_NAME="${VOL_NAME:-team2f25-ollama}" # Docker volume for Ollama models

echo "🚀 [1/5] Building Docker image: ${IMAGE_TAG}"
docker build -t "${IMAGE_TAG}" .

echo "🧹 [2/5] Removing old container (if exists): ${APP_NAME}"
docker rm -f "${APP_NAME}" 2>/dev/null || true

echo "💾 [3/5] Creating volume for Ollama models: ${VOL_NAME}"
docker volume create "${VOL_NAME}" >/dev/null

echo "▶️ [4/5] Starting new container..."
docker run -d \
  --name "${APP_NAME}" \
  -p "${UI_PORT}:${UI_PORT}" \
  -p "${OLLAMA_PORT}:${OLLAMA_PORT}" \
  -e UI_PORT="${UI_PORT}" \
  -e BASE_PATH="${BASE_PATH}" \
  -e MODEL_NAME="${MODEL_NAME}" \
  -e OLLAMA_HOST="http://127.0.0.1:${OLLAMA_PORT}" \
  -v "${VOL_NAME}:/root/.ollama" \
  "${IMAGE_TAG}"

echo "📋 [5/5] Running containers:"
docker ps --filter "name=${APP_NAME}"

echo
echo "✅ Setup complete!"
echo "🌐 Visit: http://localhost:${UI_PORT}/${BASE_PATH}"
echo "   (If /team2f25 doesn’t work, try http://localhost:${UI_PORT}/)"
echo "💡 Ollama models are cached in Docker volume '${VOL_NAME}'."
