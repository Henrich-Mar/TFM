#!/usr/bin/env bash
set -euo pipefail

# Cloud launcher for Vast.ai / Ubuntu hosts.
# It generates a compose file sized to host RAM/CPU and starts training.

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.rl_cloud.generated.yml}"
BASE_COMPOSE="${BASE_COMPOSE:-docker-compose.rl_hard.yml}"
PUBLIC_HOST="${PUBLIC_HOST:-localhost}"
NUM_COORDS="${RL_NUM_COORDINATORS:-1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or current user has no access."
  exit 1
fi

mkdir -p rl-models rl-logs rl-models-global
if [ "$NUM_COORDS" -gt 1 ]; then
  for i in $(seq 1 "$NUM_COORDS"); do mkdir -p "rl-models-coord-${i}"; done
fi

python3 scripts/generate_rl_cloud_compose.py \
  --base-compose "$BASE_COMPOSE" \
  --output "$COMPOSE_FILE" \
  --public-host "$PUBLIC_HOST" \
  --num-coordinators "${RL_NUM_COORDINATORS:-1}" \
  --coordinators-share-gpu "${RL_COORDINATORS_SHARE_GPU:-0}" \
  --gpu-index "${RL_GPU_INDEX:-0}" \
  --population-size "${RL_POPULATION_SIZE:-0}" \
  --games-per-eval "${RL_GAMES_PER_EVAL:-0}" \
  --games-per-server "${RL_GAMES_PER_SERVER:-0}" \
  --max-servers "${RL_MAX_SERVERS:-128}" \
  --min-servers "${RL_MIN_SERVERS:-4}" \
  --server-mem-mb "${RL_SERVER_MEM_MB:-1400}" \
  --node-heap-mb "${RL_NODE_HEAP_MB:-0}" \
  --training-profile "${RL_TRAINING_PROFILE:-balanced}" \
  --ppo-rollout-steps "${RL_PPO_ROLLOUT_STEPS:-0}" \
  --ppo-buffer-max-steps "${RL_PPO_BUFFER_MAX_STEPS:-0}" \
  --agent-poll-interval-sec "${RL_AGENT_POLL_INTERVAL_SEC:--1}" \
  --agent-failure-pause-sec "${RL_AGENT_FAILURE_PAUSE_SEC:--1}" \
  --tm-game-timeout-sec "${RL_TM_GAME_TIMEOUT_SEC:-0}" \
  --tm-http-force-close-connections "${RL_TM_HTTP_FORCE_CLOSE_CONNECTIONS:-0}" \
  --tm-recycle-session-on-disconnect "${RL_TM_RECYCLE_SESSION_ON_DISCONNECT:-0}" \
  --initial-cards-jitter-ms "${RL_INITIAL_CARDS_JITTER_MS:--1}"


docker compose -f "$COMPOSE_FILE" up -d --build

DASHBOARD_HOST="${DASHBOARD_ACCESS_HOST:-127.0.0.1}"
echo
echo "Training stack started with: $COMPOSE_FILE"
if [ "$NUM_COORDS" -gt 1 ]; then
  for i in $(seq 1 "$NUM_COORDS"); do
    port=$((4999 + i))
    echo "Coordinator $i - Dashboard: http://${DASHBOARD_HOST}:${port}/dashboard  Stats: http://${DASHBOARD_HOST}:${port}/stats"
  done
else
  echo "Dashboard: http://${DASHBOARD_HOST}:5000/dashboard"
  echo "Stats:     http://${DASHBOARD_HOST}:5000/stats"
fi
echo
echo "Useful commands:"
echo "  docker compose -f $COMPOSE_FILE ps"
if [ "$NUM_COORDS" -gt 1 ]; then
  echo "  docker compose -f $COMPOSE_FILE logs -f rl-coordinator-1"
  echo "  docker compose -f $COMPOSE_FILE logs -f rl-champion-orchestrator"
else
  echo "  docker compose -f $COMPOSE_FILE logs -f rl-coordinator"
fi
