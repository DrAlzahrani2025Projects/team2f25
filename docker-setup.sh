#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Team2f25 Project - Docker Setup Script
# Builds and runs Apache (reverse proxy), Streamlit, and Jupyter services
# ------------------------------------------------------------------------------

# Detect Docker Compose (v2 plugin or legacy binary)
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

echo "🛑 Stopping any existing containers..."
$COMPOSE down -v || true

echo "🔨 Building images..."
$COMPOSE build --parallel

echo "🚀 Starting services..."
$COMPOSE up -d

echo
echo "✅ Setup complete. Services are running:"
echo "   • Streamlit: http://localhost:2502/team2f25/"
echo "   • Jupyter:   http://localhost:2502/team2f25/jupyter/"
echo
echo "📜 Logs: run '$COMPOSE logs -f proxy' to see Apache activity"
echo "📜 Logs: run '$COMPOSE logs -f team2-app' to see Streamlit logs"
echo "📜 Logs: run '$COMPOSE logs -f team2-jupyter' to see Jupyter logs"
