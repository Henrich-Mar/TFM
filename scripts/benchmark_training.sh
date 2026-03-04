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

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.rl_cloud.generated.yml}"

# If the compose stack is already running, exec into the coordinator.
if docker compose -f "$COMPOSE_FILE" ps --status=running 2>/dev/null | grep -q "rl-coordinator"; then
    echo "Running benchmark inside existing rl-coordinator container..."
    docker compose -f "$COMPOSE_FILE" exec rl-coordinator \
        python scripts/benchmark_training.py "$@"
else
    # Otherwise, run a one-shot container with GPU access.
    echo "No running coordinator found. Starting a one-shot benchmark container..."
    echo "(Building image if needed...)"
    docker compose -f "$COMPOSE_FILE" build rl-coordinator 2>/dev/null || \
    docker compose -f docker-compose.rl_hard.yml build rl-coordinator 2>/dev/null || true

    # Determine the image name from the compose file.
    IMAGE=$(docker compose -f "$COMPOSE_FILE" images rl-coordinator --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | head -1)
    if [ -z "$IMAGE" ] || [ "$IMAGE" = ":" ]; then
        # Fallback: try the hard compose
        IMAGE=$(docker compose -f docker-compose.rl_hard.yml images rl-coordinator --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | head -1)
    fi
    if [ -z "$IMAGE" ] || [ "$IMAGE" = ":" ]; then
        echo "ERROR: Could not determine coordinator image. Build the stack first:"
        echo "  docker compose -f $COMPOSE_FILE build rl-coordinator"
        exit 1
    fi

    docker run --rm -it \
        --gpus all \
        -v "$(pwd)/rl-environment:/app" \
        -v "$(pwd)/scripts:/app/scripts" \
        -v "$(pwd)/card_metadata.json:/app/card_metadata.json:ro" \
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
        "$IMAGE" \
        python scripts/benchmark_training.py "$@"
fi
