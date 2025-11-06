#!/usr/bin/env bash
set -euo pipefail

# Set defaults for all environment variables
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-5002}"
export STREAMLIT_SERVER_BASE_URL_PATH="${STREAMLIT_SERVER_BASE_URL_PATH:-team2f25}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5:1.5b}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"

if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh || true
fi

echo "=== CSUSB Internship Finder - Startup ==="
echo "OLLAMA_HOST: $OLLAMA_HOST"
echo "MODEL_NAME: $MODEL_NAME"
echo "BACKEND_PORT: $BACKEND_PORT"
echo "STREAMLIT_PORT: $STREAMLIT_SERVER_PORT"
echo ""

# ============================================================================
# 1. START OLLAMA
# ============================================================================
echo "[1/4] Checking Ollama..."
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama found. Starting service..."
  (ollama serve >/tmp/ollama.log 2>&1 &) || true
  sleep 3
  
  echo "Waiting for Ollama to be ready..."
  MAX_RETRIES=30
  RETRY=0
  while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready!"
      break
    fi
    echo "  Retry $((RETRY+1))/$MAX_RETRIES..."
    sleep 1
    RETRY=$((RETRY+1))
  done
  
  if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama failed to start"
    cat /tmp/ollama.log
    exit 1
  fi
  
  # Pull model if needed
  if ! curl -s "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"${MODEL_NAME}\""; then
    echo "Pulling model: $MODEL_NAME"
    ollama pull "${MODEL_NAME}" || {
      echo "ERROR: Failed to pull model"
      exit 1
    }
  else
    echo "Model $MODEL_NAME already available"
  fi
else
  echo "WARNING: Ollama not found. Backend will fail to connect."
fi

# ============================================================================
# 4. START STREAMLIT
# ============================================================================
echo ""
echo "[4/4] Starting Streamlit..."
echo "Access the app at: http://localhost:${STREAMLIT_SERVER_PORT}/${STREAMLIT_SERVER_BASE_URL_PATH}"
echo ""

exec streamlit run app.py \
  --server.port "$STREAMLIT_SERVER_PORT" \
  --server.baseUrlPath "$STREAMLIT_SERVER_BASE_URL_PATH" \
  --browser.gatherUsageStats false