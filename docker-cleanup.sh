#!/usr/bin/env bash
set -euo pipefail

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo "ERROR: Docker Compose not found. Install Docker Desktop or the compose plugin." >&2
    exit 1
  fi
}

COMPOSE="$(detect_compose)"

echo "🛑 Stopping containers and cleaning up..."
$COMPOSE down -v
docker system prune -f
echo "✅ Done."
