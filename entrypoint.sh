#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-qwen2:0.5b}"
UI_PORT="${UI_PORT:-5002}"

echo "[entrypoint] starting Ollama..."
ollama serve &

# Start Streamlit immediately so the page loads fast
echo "[entrypoint] starting Streamlit on :${UI_PORT}"
streamlit run /app/app.py --server.address=0.0.0.0 --server.port "${UI_PORT}" &
ST_PID=$!

# In the background: wait for Ollama, then ensure model is present
(
  echo -n "[entrypoint] waiting for Ollama"
  for i in {1..120}; do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then echo " ✓"; break; fi
    echo -n "."; sleep 1
  done

  echo "[entrypoint] checking model '${MODEL_NAME}'..."
  if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"${MODEL_NAME}\""; then
    echo "[entrypoint] pulling ${MODEL_NAME} (first run only)"
    curl -sf http://127.0.0.1:11434/api/pull -d "{\"name\":\"${MODEL_NAME}\"}" || true
  else
    echo "[entrypoint] model already present; skipping pull."
  fi
) &

# Keep container alive with Streamlit in foreground
wait "${ST_PID}"
