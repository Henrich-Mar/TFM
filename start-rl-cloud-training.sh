#!/usr/bin/env bash
set -euo pipefail

# Cloud launcher for Vast.ai / Ubuntu hosts.
# It generates a compose file sized to host RAM/CPU and starts training.

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.rl_cloud.generated.yml}"
BASE_COMPOSE="${BASE_COMPOSE:-docker-compose.rl_hard.yml}"
PUBLIC_HOST="${PUBLIC_HOST:-localhost}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or current user has no access."
  exit 1
fi

mkdir -p rl-models rl-logs

python3 scripts/generate_rl_cloud_compose.py \
  --base-compose "$BASE_COMPOSE" \
  --output "$COMPOSE_FILE" \
  --public-host "$PUBLIC_HOST"

docker compose -f "$COMPOSE_FILE" up -d --build

DASHBOARD_HOST="${DASHBOARD_ACCESS_HOST:-127.0.0.1}"
echo
echo "Training stack started with: $COMPOSE_FILE"
echo "Dashboard: http://${DASHBOARD_HOST}:5000/dashboard"
echo "Stats:     http://${DASHBOARD_HOST}:5000/stats"
echo
echo "Useful commands:"
echo "  docker compose -f $COMPOSE_FILE ps"
echo "  docker compose -f $COMPOSE_FILE logs -f rl-coordinator"
