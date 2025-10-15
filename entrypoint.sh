#!/usr/bin/env bash
set -euo pipefail

: "${STREAMLIT_SERVER_PORT:=5002}"
: "${STREAMLIT_SERVER_BASE_URL_PATH:=team2f25}"
: "${MODEL_NAME:=qwen2.5:0.5b}"
: "${OLLAMA_HOST:=http://127.0.0.1:11434}"

if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh || true
fi

# Start Ollama in the background if available (inside container)
if command -v ollama >/dev/null 2>&1; then
  (ollama serve >/tmp/ollama.log 2>&1 &) || true
fi

exec streamlit run app.py \
  --server.port "${STREAMLIT_SERVER_PORT}" \
  --server.baseUrlPath "${STREAMLIT_SERVER_BASE_URL_PATH}" \
  --browser.gatherUsageStats false
