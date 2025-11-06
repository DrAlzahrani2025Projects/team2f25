#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Config (override via env) =====
IMAGE="${IMAGE:-team2f25-streamlit}"
NAME="${NAME:-team2f25}"                 # app container name
PORT="${PORT:-5002}"                     # app/Streamlit port exposed to host
BASE_PATH="${BASE_PATH:-team2f25}"       # streamlit base path
MODEL_NAME="${MODEL_NAME:-qwen2.5:1.5b}" # ollama model to pull

# Ollama sidecar settings
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:latest}"
OLLAMA_NAME="${OLLAMA_NAME:-team2f25-ollama}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"      # Ollama REST port
OLLAMA_VOLUME="${OLLAMA_VOLUME:-ollama}" # named volume for model cache

# Network so app can talk to Ollama by name
NET="${NET:-team2f25-net}"

# Normalize BASE_PATH for clean URL printing (strip any leading slash)
BASE_PATH="${BASE_PATH#/}"

# ===== Cleanup like Team 1 =====
./cleanup.sh --hard > /dev/null 2>&1 || true

# Remove any existing app container with the same name
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Removing existing container named ${NAME}..."
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

# Remove any existing Ollama container with the same name (optional refresh)
# Comment these two lines if you prefer to KEEP the sidecar running across restarts.
if docker ps -a --format '{{.Names}}' | grep -qx "${OLLAMA_NAME}"; then
  echo "Removing existing Ollama container ${OLLAMA_NAME}..."
  docker rm -f "${OLLAMA_NAME}" >/dev/null 2>&1 || true
fi

# Free ports if some other containers are holding them
OCC1="$(docker ps -q --filter "publish=${PORT}")"
OCC2="$(docker ps -q --filter "publish=${OLLAMA_PORT}")"
OCCUPYING_IDS="$(printf "%s\n%s\n" "$OCC1" "$OCC2" | sort -u | tr '\n' ' ' | xargs -r echo)"
if [ -n "${OCCUPYING_IDS// }" ]; then
  echo "Freeing ports ${PORT}/${OLLAMA_PORT} from containers: ${OCCUPYING_IDS}"
  docker rm -f ${OCCUPYING_IDS} >/dev/null 2>&1 || true
fi

# Also free host processes holding the ports (Linux/macOS)
if command -v lsof >/dev/null 2>&1; then
  for p in "$PORT" "$OLLAMA_PORT"; do
    pids=$(lsof -t -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null || true)
    [ -n "$pids" ] && kill -9 $pids >/dev/null 2>&1 || true
  done
fi


# Normalize line endings and ensure scripts are executable
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh startup.sh cleanup.sh 2>/dev/null || true
fi
chmod +x entrypoint.sh cleanup.sh 2>/dev/null || true

# ===== Network for app <-> Ollama =====
if ! docker network ls --format '{{.Name}}' | grep -qx "${NET}"; then
  docker network create "${NET}" >/dev/null
fi

# ===== Start/prepare Ollama sidecar =====
echo "Starting Ollama sidecar..."
docker run -d \
  --name "${OLLAMA_NAME}" \
  --network "${NET}" \
  -p "${OLLAMA_PORT}:${OLLAMA_PORT}" \
  -e OLLAMA_HOST=0.0.0.0 \
  -v "${OLLAMA_VOLUME}":/root/.ollama \
  "${OLLAMA_IMAGE}" >/dev/null

# Wait for Ollama to be ready
echo -n "Waiting for Ollama (${OLLAMA_NAME}) to be ready"
for i in {1..30}; do
  if curl -fsS "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    echo " ... ready!"
    break
  fi
  echo -n "."
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo
    echo "⚠️  Ollama did not become ready in time. Continuing anyway."
  fi
done

# Pull the requested model into Ollama (cached in the named volume)
echo "Pulling Ollama model: ${MODEL_NAME}"
if ! docker exec "${OLLAMA_NAME}" ollama pull "${MODEL_NAME}"; then
  echo "⚠️  Failed to pull model ${MODEL_NAME}. The app may try to pull it on demand."
fi

# ===== Build app image (BuildKit like Team 1) =====
echo "Building Docker image..."
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" .

# ===== Run app detached (Team 1 style), wired to Ollama sidecar =====
# We set OLLAMA_HOST to reach the sidecar by its container name on the user-defined network.
echo "Starting Docker container..."
docker run -d \
  --name "${NAME}" \
  --network "${NET}" \
  -p "${PORT}:${PORT}" \
  -e "STREAMLIT_SERVER_PORT=${PORT}" \
  -e "STREAMLIT_SERVER_BASE_URL_PATH=${BASE_PATH}" \
  -e "MODEL_NAME=${MODEL_NAME}" \
  -e "OLLAMA_HOST=http://${OLLAMA_NAME}:${OLLAMA_PORT}" \
  "${IMAGE}"

echo "Application is running on http://localhost:${PORT}/${BASE_PATH}"
