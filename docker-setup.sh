#!/usr/bin/env bash
set -euo pipefail

# Detect Compose command (v2 > v1). Sets $COMPOSE to either "docker compose" or "docker-compose".
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

echo "🛑 Stopping any existing containers..."
$COMPOSE down -v || true

echo "🔨 Building images..."
$COMPOSE build

echo "🚀 Starting containers..."
$COMPOSE up -d

echo
echo "✅ Services are running!"
echo "Flask:   http://localhost:2502/team2f25/"
echo "Jupyter: http://localhost:2502/team2f25/jupyter/"
