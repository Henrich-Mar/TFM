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
        python scripts/benchmark_training.py "$@"
else
    # Use 'docker compose run' — builds if needed, sets up GPU, volumes,
    # env vars all from the compose file automatically.
    echo "No running coordinator found. Starting one-shot benchmark via docker compose run..."
    docker compose -f "$COMPOSE_FILE" run --rm --build \
        -e AGENT_HIDDEN_SIZE="${AGENT_HIDDEN_SIZE:-1024}" \
        -e AGENT_NUM_LAYERS="${AGENT_NUM_LAYERS:-8}" \
        -e AGENT_CARD_TOKEN_DIM="${AGENT_CARD_TOKEN_DIM:-20}" \
        -e AGENT_TABLEAU_TOKEN_COUNT="${AGENT_TABLEAU_TOKEN_COUNT:-10}" \
        -e AGENT_HAND_TOKEN_COUNT="${AGENT_HAND_TOKEN_COUNT:-4}" \
        -e AGENT_OPPONENT_TOKEN_COUNT="${AGENT_OPPONENT_TOKEN_COUNT:-6}" \
        -e AGENT_TRANSFORMER_EMBED_DIM="${AGENT_TRANSFORMER_EMBED_DIM:-256}" \
        -e AGENT_TRANSFORMER_HEADS="${AGENT_TRANSFORMER_HEADS:-16}" \
        -e AGENT_TRANSFORMER_LAYERS="${AGENT_TRANSFORMER_LAYERS:-4}" \
        -e POPULATION_SIZE="${POPULATION_SIZE:-16}" \
        -e GAMES_PER_EVAL="${GAMES_PER_EVAL:-6}" \
        rl-coordinator \
        python scripts/benchmark_training.py "$@"
fi
