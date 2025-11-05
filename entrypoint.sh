#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Config (with sensible defaults)
# ==========================
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-5002}"
export STREAMLIT_SERVER_BASE_URL_PATH="${STREAMLIT_SERVER_BASE_URL_PATH:-team2f25}"

# Backends: chat (main.py) on 8000, navigator (backend_navigator.py) on 8001
export BACKEND_CHAT_PORT="${BACKEND_CHAT_PORT:-8000}"
export BACKEND_NAV_PORT="${BACKEND_NAV_PORT:-8001}"

# Where Streamlit should call for navigation
export BACKEND_URL="${BACKEND_URL:-http://localhost:${BACKEND_NAV_PORT}}"

# LLM settings (OpenAI)
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

# ==========================
# Helpers
# ==========================
wait_for_http() {
  # wait_for_http <url> <timeout_seconds>
  local url="$1"
  local timeout="${2:-30}"
  local t=0
  while [ "$t" -lt "$timeout" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    t=$((t+1))
  done
  return 1
}

graceful_shutdown() {
  echo "Shutting down..."
  # Kill backends if running
  if [ -n "${CHAT_PID:-}" ] && kill -0 "$CHAT_PID" 2>/dev/null; then
    kill "$CHAT_PID" 2>/dev/null || true
  fi
  if [ -n "${NAV_PID:-}" ] && kill -0 "$NAV_PID" 2>/dev/null; then
    kill "$NAV_PID" 2>/dev/null || true
  fi
}
trap graceful_shutdown EXIT

# ==========================
# Banner
# ==========================
echo "================= Startup ================="
echo " Streamlit:       http://localhost:${STREAMLIT_SERVER_PORT}/${STREAMLIT_SERVER_BASE_URL_PATH}"
echo " Chat API:        http://localhost:${BACKEND_CHAT_PORT} (main.py)"
echo " Navigator API:   http://localhost:${BACKEND_NAV_PORT} (backend_navigator.py)"
echo " BACKEND_URL:     ${BACKEND_URL}"
echo " LLM_PROVIDER:    ${LLM_PROVIDER}"
echo " LLM_MODEL:       ${LLM_MODEL}"
echo "===================================================="
echo

# ==========================
# Start chat API (main.py) on 8000
# ==========================
echo "[1/3] Starting chat API (main.py) on :${BACKEND_CHAT_PORT} ..."
( python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port "${BACKEND_CHAT_PORT}" \
    --log-level info \
  > /tmp/main.log 2>&1 ) &
CHAT_PID=$!
echo "  → main.py PID=${CHAT_PID}"

# ==========================
# Start navigator API (backend_navigator.py) on 8001
# ==========================
echo "[2/3] Starting navigator API (backend_navigator.py) on :${BACKEND_NAV_PORT} ..."
( python -m uvicorn backend_navigator:app \
    --host 0.0.0.0 \
    --port "${BACKEND_NAV_PORT}" \
    --log-level info \
  > /tmp/navigator.log 2>&1 ) &
NAV_PID=$!
echo "  → navigator PID=${NAV_PID}"

# ==========================
# Health checks (non-fatal if chat fails; required for navigator)
# ==========================
echo
echo "Waiting for backends to become healthy..."

CHAT_HEALTH_URL="http://localhost:${BACKEND_CHAT_PORT}/healthz"
NAV_HEALTH_URL="http://localhost:${BACKEND_NAV_PORT}/health"

CHAT_OK=false
if wait_for_http "$CHAT_HEALTH_URL" 30; then
  CHAT_OK=true
  echo "  ✓ Chat API healthy: ${CHAT_HEALTH_URL}"
else
  echo "  ✗ Chat API failed health check: ${CHAT_HEALTH_URL}"
  if [ -f /tmp/main.log ]; then
    echo "  --- Last lines of /tmp/main.log ---"
    tail -n 60 /tmp/main.log || true
    echo "  ----------------------------------"
  fi
  # Do NOT exit; continue with navigator + Streamlit
fi

if ! wait_for_http "$NAV_HEALTH_URL" 40; then
  echo "  ✗ Navigator API failed health check: ${NAV_HEALTH_URL}"
  if [ -f /tmp/navigator.log ]; then
    echo "  --- Last lines of /tmp/navigator.log ---"
    tail -n 80 /tmp/navigator.log || true
    echo "  ----------------------------------------"
  fi
  echo "Navigator is required. Exiting."
  exit 1
fi
echo "  ✓ Navigator healthy: ${NAV_HEALTH_URL}"

echo
echo "[3/3] Starting Streamlit on :${STREAMLIT_SERVER_PORT}"
echo "Open: http://localhost:${STREAMLIT_SERVER_PORT}/${STREAMLIT_SERVER_BASE_URL_PATH}"
echo

# ==========================
# Run Streamlit (PID 1 replacement)
# ==========================
exec streamlit run app.py \
  --server.port "${STREAMLIT_SERVER_PORT}" \
  --server.baseUrlPath "${STREAMLIT_SERVER_BASE_URL_PATH}" \
  --browser.gatherUsageStats false
