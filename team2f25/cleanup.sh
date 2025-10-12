#!/usr/bin/env bash
set -euo pipefail

IMAGE="internship-finder:team2f25"
NAME="team2f25"

echo "Stopping & removing container (if running)…"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "Remove dangling containers using port 5002 (if any)…"
for id in $(docker ps -q --filter "publish=5002"); do docker rm -f "$id"; done

read -p "Also remove image ${IMAGE}? [y/N] " yn
case "$yn" in
  [Yy]*) docker rmi "${IMAGE}" || true ;;
  *) echo "Keeping image." ;;
esac

echo "Done."
