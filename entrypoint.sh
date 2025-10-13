#!/usr/bin/env bash
set -euo pipefail

: "${STREAMLIT_SERVER_ADDRESS:=0.0.0.0}"
: "${STREAMLIT_SERVER_PORT:=5002}"
: "${STREAMLIT_SERVER_BASE_URL_PATH:=team2f25}"

echo "[entrypoint] Streamlit address: ${STREAMLIT_SERVER_ADDRESS}"
echo "[entrypoint] Streamlit port:    ${STREAMLIT_SERVER_PORT}"
echo "[entrypoint] Base URL path:     ${STREAMLIT_SERVER_BASE_URL_PATH}"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[entrypoint] OPENAI_API_KEY detected (AI agent enabled)"
else
  echo "[entrypoint] No OPENAI_API_KEY (using heuristic parser)"
fi

exec streamlit run app.py \
  --server.address "${STREAMLIT_SERVER_ADDRESS}" \
  --server.port "${STREAMLIT_SERVER_PORT}" \
  --server.baseUrlPath "${STREAMLIT_SERVER_BASE_URL_PATH}"
