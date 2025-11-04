#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Config (override via env) =====
IMAGE="${IMAGE:-team2f25-streamlit}"
NAME="${NAME:-team2f25}"               # app container name
PORT="${PORT:-5002}"                   # Streamlit port exposed to host
BASE_PATH="${BASE_PATH:-team2f25}"     # streamlit base path (no leading slash)
LLM_MODEL="${LLM_MODEL:-gpt-5.1}"      # OpenAI model to use

# Normalize BASE_PATH for clean URL printing (strip any leading slash)
BASE_PATH="${BASE_PATH#/}"
# Auto-load local secrets if present
if [ -f .env ]; then set -a; . ./.env; set +a; fi


# ===== Safety checks =====
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is not set in your environment."
  echo "   Example (don’t paste your key here):"
  echo "   OPENAI_API_KEY=sk-... ./startup.sh"
  exit 1
fi

# ===== Cleanup =====
./cleanup.sh --hard > /dev/null 2>&1 || true

# Remove any existing app container with the same name
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Removing existing container named ${NAME}..."
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

# Free the app port if occupied by another container
if ids="$(docker ps -q --filter "publish=${PORT}")"; then
  [[ -n "${ids}" ]] && docker rm -f ${ids} >/dev/null 2>&1 || true
fi

# Also free host processes holding the port (Linux/macOS)
if command -v lsof >/dev/null 2>&1; then
  pids=$(lsof -t -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
  [[ -n "${pids:-}" ]] && kill -9 $pids >/dev/null 2>&1 || true
fi

# Normalize line endings and ensure scripts are executable
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh startup.sh cleanup.sh 2>/dev/null || true
fi
chmod +x entrypoint.sh cleanup.sh 2>/dev/null || true

# ===== Build app image =====
echo "Building Docker image..."
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" .

# ===== Run app container (detached) =====
echo "Starting Docker container..."
docker run -d \
  --name "${NAME}" \
  -p "${PORT}:${PORT}" \
  -e "STREAMLIT_SERVER_PORT=${PORT}" \
  -e "STREAMLIT_SERVER_BASE_URL_PATH=${BASE_PATH}" \
  -e "LLM_PROVIDER=openai" \
  -e "LLM_MODEL=${LLM_MODEL}" \
  -e OPENAI_API_KEY \
  "${IMAGE}"

echo "Application is running on http://localhost:${PORT}/${BASE_PATH}"
echo "LLM provider: openai"
