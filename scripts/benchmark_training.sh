#!/usr/bin/env bash
set -euo pipefail

# Run the RL training benchmark inside the coordinator container.
#
# Usage (from repo root on the cloud VM):
#   bash scripts/benchmark_training.sh [OPTIONS]
#
# Options are forwarded to benchmark_training.py, e.g.:
#   bash scripts/benchmark_training.sh --cpu-only
#   bash scripts/benchmark_training.sh --ppo-rollout-steps 16384

# Pick whichever compose file exists
if [ -f "docker-compose.rl_cloud.generated.yml" ]; then
    COMPOSE_FILE="docker-compose.rl_cloud.generated.yml"
elif [ -f "docker-compose.rl_hard.yml" ]; then
    COMPOSE_FILE="docker-compose.rl_hard.yml"
else
    echo "ERROR: No compose file found. Run from the TFM repo root."
    exit 1
fi

echo "Using compose file: $COMPOSE_FILE"

# The benchmark script lives in rl-environment/ which is mounted as /app
# inside the coordinator container.
BENCH_CMD="python benchmark_training.py"

# If the compose stack is already running, exec into the coordinator.
COORD_SERVICE=""
if docker compose -f "$COMPOSE_FILE" ps --status=running 2>/dev/null | grep -q "rl-coordinator"; then
    COORD_SERVICE="rl-coordinator"
elif docker compose -f "$COMPOSE_FILE" ps --status=running 2>/dev/null | grep -q "rl-coordinator-1"; then
    COORD_SERVICE="rl-coordinator-1"
fi

if [ -n "$COORD_SERVICE" ]; then
    echo "Running benchmark inside existing $COORD_SERVICE container..."
    docker compose -f "$COMPOSE_FILE" exec "$COORD_SERVICE" \
        $BENCH_CMD "$@"
else
    # Use 'docker compose run' — builds if needed, sets up GPU, volumes,
    # env vars all from the compose file automatically.
    echo "No running coordinator found. Starting one-shot benchmark via docker compose run..."
    docker compose -f "$COMPOSE_FILE" run --rm --build --no-deps \
        rl-coordinator \
        $BENCH_CMD "$@"
fi
