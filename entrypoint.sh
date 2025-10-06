#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting Ollama..."
/usr/local/bin/ollama serve &

# Wait for Ollama
echo -n "[entrypoint] waiting for Ollama"
for i in {1..120}; do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    echo " ✓"; break
  fi
  echo -n "."; sleep 1
done

MODEL_NAME="${MODEL_NAME:-qwen2:0.5b}"

# First-run-only download (uses persisted /root/.ollama)
echo "[entrypoint] checking model '${MODEL_NAME}'..."
if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"${MODEL_NAME}\""; then
  echo "[entrypoint] pulling ${MODEL_NAME} (first run only)"
  curl -sf http://127.0.0.1:11434/api/pull -d "{\"name\":\"${MODEL_NAME}\"}" || true
else
  echo "[entrypoint] model already present; skipping pull."
fi

# Launch Streamlit on :5002
export STREAMLIT_SERVER_PORT="${UI_PORT:-5002}"
exec streamlit run /app/app.py --server.address=0.0.0.0 --server.port "${STREAMLIT_SERVER_PORT}"
