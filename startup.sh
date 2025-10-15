#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-internship-chatbot}"
PORT="${PORT:-5002}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
BASE_PATH="${STREAMLIT_SERVER_BASE_URL_PATH:-team2f25}"
MODEL_NAME="${MODEL_NAME:-qwen2.5:0.5b}"
DETACH="${DETACH:-false}"
NAME="${NAME:-csusb-internship-agent}"

if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh startup.sh cleanup.sh 2>/dev/null || true
fi
chmod +x entrypoint.sh cleanup.sh || true

docker build -t "${IMAGE}" .

RUN_ARGS=(-p "${PORT}:${PORT}" -p "${OLLAMA_PORT}:${OLLAMA_PORT}"
          -e "STREAMLIT_SERVER_PORT=${PORT}"
          -e "STREAMLIT_SERVER_BASE_URL_PATH=${BASE_PATH}"
          -e "MODEL_NAME=${MODEL_NAME}")

if [[ "${DETACH}" == "true" ]]; then
  docker run -d --name "${NAME}" "${RUN_ARGS[@]}" "${IMAGE}"
  echo "Running at http://localhost:${PORT}/${BASE_PATH}"
else
  docker run --rm "${RUN_ARGS[@]}" "${IMAGE}"
fi
