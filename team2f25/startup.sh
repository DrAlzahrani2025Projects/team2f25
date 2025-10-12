#!/usr/bin/env bash
set -euo pipefail

IMAGE="internship-finder:team2f25"
NAME="team2f25"
HOST_PORT=5002
DATA_DIR="$(pwd)/data"

echo "[1/3] Building image: ${IMAGE}"
docker build -t "${IMAGE}" .

echo "[2/3] Stopping previous container (if any)"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
for id in $(docker ps -q --filter "publish=${HOST_PORT}"); do docker rm -f "$id"; done

echo "[3/3] Running container on :${HOST_PORT}"
docker run --rm -p ${HOST_PORT}:5002 \
  -v "${DATA_DIR}:/app/data" \
  --name "${NAME}" \
  "${IMAGE}"

# Note: open http://localhost:${HOST_PORT}/team2f25 (or override base path via env)
