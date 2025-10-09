#!/usr/bin/env bash
set -euo pipefail

PORT=5002
NAME=team2f25
IMAGE=team2f25-streamlit:latest

echo "==> Preflight: normalize line endings for shell scripts (host side)"
if command -v sed >/dev/null 2>&1; then
  find . -type f -name "*.sh" -exec sed -i 's/\r$//' {} \;
fi
chmod +x entrypoint.sh cleanup.sh || true

echo "==> Stop any container publishing port ${PORT}"
CID_ON_PORT=$(docker ps --filter "publish=${PORT}" --format "{{.ID}}" || true)
if [[ -n "${CID_ON_PORT}" ]]; then
  docker stop ${CID_ON_PORT}
fi

echo "==> Remove stale container named ${NAME} (if any)"
docker rm -f "${NAME}" 2>/dev/null || true

echo "==> Building image ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Starting container ${NAME} on port ${PORT}"
# Rely on ENTRYPOINT in the Dockerfile to start the app
docker run -d --name "${NAME}" -p ${PORT}:${PORT} "${IMAGE}"

# Optional: basic health printout
echo "==> Container started. Recent logs:"
docker logs --since=10s "${NAME}" || true

echo
echo "Open your app at: http://localhost:${PORT}/team2f25"
echo "Follow logs with: docker logs -f ${NAME}"

# If the container exits immediately, print why and fail the script
sleep 2
if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo
  echo "!! Container is not running. Last 100 log lines:"
  docker logs --tail 100 "${NAME}" || true
  exit 1
fi
