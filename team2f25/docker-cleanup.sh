#!/usr/bin/env bash
set -euo pipefail
docker compose down -v
docker system prune -f
echo "--------------------------------------------------"
echo "Team2F25 stack stopped and cleaned up!"
echo "--------------------------------------------------"