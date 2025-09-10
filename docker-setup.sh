#!/usr/bin/env bash
set -euo pipefail
docker compose build
docker compose up -d
echo "--------------------------------------------------"
echo "Team2F25 stack is running!"
echo "Flask App:   http://localhost/team2/app"
echo "Jupyter Lab: http://localhost/team2/jupyter"
echo "--------------------------------------------------"