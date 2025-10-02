#!/usr/bin/env bash
# Single-container setup for team2f25 Streamlit app
# Usage: ./docker-setup.sh [--no-cache]
set -euo pipefail

IMAGE="team2f25:latest"
CONTAINER="team2f25"
PORT="5002"

echo "🔧 Preparing to build image: ${IMAGE}"
if [[ "${1:-}" == "--no-cache" ]]; then
  docker build --no-cache -t "${IMAGE}" .
else
  docker build -t "${IMAGE}" .
fi

# Remove any old container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; then
  echo "🧹 Removing existing container ${CONTAINER}…"
  docker rm -f "${CONTAINER}" >/dev/null
fi

echo "🚀 Starting container: ${CONTAINER} (port ${PORT})"
docker run -d --name "${CONTAINER}" -p "${PORT}:${PORT}" "${IMAGE}" >/dev/null

echo "⏳ Waiting for app to boot (few seconds)…"
# quick, optional check (best-effort, won’t fail the script)
if command -v curl >/dev/null 2>&1; then
  for i in {1..20}; do
    if curl -fsS "http://localhost:${PORT}/team2f25" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

echo
echo "✅ App should be live locally at:  http://localhost:${PORT}/team2f25"
echo "✅ App should be live at:  https://sec.cse.csusb.edu/team2f25"
echo
echo "👉 Logs (tail): docker logs -f ${CONTAINER}"
echo "👉 Stop:        docker stop ${CONTAINER}"
echo "👉 Remove:      docker rm   ${CONTAINER}"
