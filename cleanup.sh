#!/usr/bin/env sh
set -e

echo "🧹 Cleaning up old containers and images for team2f25..."

# Stop and remove any container named team2f25 (if it exists)
if docker ps -a --format '{{.Names}}' | grep -q '^team2f25$'; then
  echo "Stopping and removing container 'team2f25'..."
  docker rm -f team2f25 >/dev/null 2>&1 || true
fi

# Stop any container using port 5002
if docker ps --filter "publish=5002" --format '{{.ID}}' | grep -q .; then
  echo "Stopping containers using port 5002..."
  docker stop $(docker ps -q --filter "publish=5002") >/dev/null 2>&1 || true
fi

# Optionally remove the old image (uncomment if you want a full rebuild each time)
# echo "Removing old Docker image 'team2f25-streamlit:latest'..."
# docker rmi -f team2f25-streamlit:latest >/dev/null 2>&1 || true

# Clean dangling images/volumes (optional but nice)
docker image prune -f >/dev/null 2>&1 || true
docker container prune -f >/dev/null 2>&1 || true

echo "✅ Cleanup complete!"
