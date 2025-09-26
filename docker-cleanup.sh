#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Team2f25 Project - Docker Cleanup Script
# Stops containers, removes volumes, and prunes unused Docker resources
# ------------------------------------------------------------------------------

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo "❌ ERROR: Docker Compose not found. Please install Docker Desktop or the compose plugin." >&2
    exit 1
  fi
}

COMPOSE="$(detect_compose)"

echo "🛑 Stopping containers and removing volumes..."
$COMPOSE down -v || true

echo "🧹 Pruning unused Docker resources (images, networks, cache)..."
docker system prune -af --volumes

echo
echo "✅ Cleanup complete."
echo "ℹ️  All containers, networks, and dangling images have been removed."
echo "⚠️  If you had other Docker containers running, they may have been stopped."
