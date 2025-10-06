#!/usr/bin/env bash
set -e

# Start Ollama server
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama on :11434..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 0.5
done

# Kick off model pull in the background (non-blocking)
if [ -n "${LLM_MODEL}" ]; then
  echo "Pulling model in background: ${LLM_MODEL}"
  ( ollama pull "${LLM_MODEL}" || true ) &
fi

# Start Streamlit right away
exec streamlit run /app/app.py \
  --server.port=5002 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false

