# Vast.ai Cloud Setup (Ubuntu)

This setup keeps your RL code in this repo (`Vast-cloud` branch), keeps the game code in a local `terraforming-mars/` folder, and starts auto-sized training.

## 1) Prepare VM

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3 python3-venv

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
```

If your VM has NVIDIA GPU:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2) Clone repos

```bash
mkdir -p ~/tfm-cloud && cd ~/tfm-cloud
git clone -b Vast-cloud https://github.com/Henrich-Mar/TFM.git TFM
cd TFM
git clone https://github.com/terraforming-mars/terraforming-mars.git
```

Expected layout:

```text
TFM/
  Dockerfile.rl
  docker-compose.rl_hard.yml
  start-rl-cloud-training.sh
  scripts/generate_rl_cloud_compose.py
  terraforming-mars/
```

## 3) Start auto-sized training

```bash
cd ~/tfm-cloud/TFM
chmod +x start-rl-cloud-training.sh
./start-rl-cloud-training.sh
```

The launcher creates `docker-compose.rl_cloud.generated.yml` with dynamic:
- number of `tfm-server-*` services
- `GLOBAL_GAME_CONCURRENCY`
- `TOURNAMENT_CONCURRENCY`
- `MAX_ACTIVE_GAMES_PER_SERVER`
- HTTP connector limits

## 4) Benchmark before training (optional)

Run the hardware benchmark to discover optimal parameters for your machine:

```bash
cd ~/tfm-cloud/TFM
# If the stack is running:
bash benchmark_training.sh

# Or one-shot without a running stack:
bash benchmark_training.sh --ppo-rollout-steps 16384
```

The benchmark tests:
- **GPU inference** — forward-pass latency at batch sizes 1–64 (CPU vs GPU)
- **PPO training** — gradient update throughput at minibatch sizes 512–8192
- **VRAM usage** — peak memory per minibatch size

Output includes a recommended `export` block — paste it before running `start-rl-cloud-training.sh`.

**Flags:**
| Flag | Effect |
|------|--------|
| `--cpu-only` | Skip GPU benchmarks (useful for debugging) |
| `--skip-inference` | Only run PPO benchmark |
| `--skip-ppo` | Only run inference benchmark |
| `--ppo-rollout-steps N` | Synthetic rollout size (default 8192) |
| `--ppo-minibatch-sizes 1024,4096` | Custom minibatch sizes to test |

## 5) Tune speed (optional)

### Balanced profile (default)

```bash
export PUBLIC_HOST=$(curl -s ifconfig.me)
./start-rl-cloud-training.sh
```

### Saturate profile (recommended for your 40 vCPU / 387 GB node)

This profile aims for much more training volume per generation.
With the current hard base (`GAMES_PER_EVAL=6`), saturate mode targets `~10x` (`~60`).

```bash
# Use localhost for SSH port forwarding; use $(curl -s ifconfig.me) for direct VM access
export RL_MIN_SERVERS=12
export RL_HTTP_CONNECTOR_LIMIT=9216
export RL_HTTP_CONNECTOR_LIMIT_PER_HOST=288
# Capacity controls for 40 vCPU, 387 GB RAM
export RL_MAX_SERVERS=6
export PUBLIC_HOST=localhost
export RL_TRAINING_PROFILE=saturate
export RL_CPU_SERVER_RATIO=0.9 
export RL_SERVER_MEM_MB=1400
export RL_NODE_HEAP_MB=1050
export RL_GAMES_PER_SERVER=4
export RL_COORDINATORS_SHARE_GPU=1
export RL_AGENT_POLL_INTERVAL_SEC=0.03
export RL_AGENT_FAILURE_PAUSE_SEC=0.05
export RL_GLOBAL_GAME_CAP_PER_COORD=20
export RL_TOURNAMENT_CAP_PER_COORD=20
export RL_INITIAL_CARDS_JITTER_MS=1200
export RL_TM_RECYCLE_SESSION_ON_DISCONNECT=0

# Optional hard overrides (leave unset for auto from profile)
export RL_GAMES_PER_EVAL=30
# export RL_PPO_ROLLOUT_STEPS=131072
export RL_TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL=7
export RL_TM_GAME_TIMEOUT_SEC=1200   # if games cancelled unexpectedly
chmod +x start-rl-cloud-training.sh
./start-rl-cloud-training.sh
```
### Low-latency optimized (1 game/server, 40 vCPU / 387 GB)

```bash
export PUBLIC_HOST=localhost

export RL_TRAINING_PROFILE=saturate
export RL_NUM_COORDINATORS=3
export RL_COORDINATOR_BASE_PORT=5100
export RL_COORDINATORS_SHARE_GPU=1

# Force total servers = 30  => 8 per coordinator
export RL_MIN_SERVERS=24
export RL_MAX_SERVERS=24

# 2 concurrent games per server
export RL_GAMES_PER_SERVER=2

# 5 servers * 2 games = 10 concurrent games per coordinator
export RL_GLOBAL_GAME_CAP_PER_COORD=16
export RL_TOURNAMENT_CAP_PER_COORD=16

export RL_CPU_SERVER_RATIO=0.90
export RL_SERVER_MEM_MB=900
export RL_NODE_HEAP_MB=512
export RL_GAMES_PER_EVAL=2
export RL_INFRA_OVERHEAD_MB=6144
export RL_TM_GAME_TIMEOUT_SEC=2400
export RL_INITIAL_CARDS_JITTER_MS=1200
export RL_TM_RECYCLE_SESSION_ON_DISCONNECT=0
export RL_POPULATION_SIZE=16

export PPO_MINIBATCH_SIZE=1024
export AGENT_INFERENCE_BATCH_SIZE=64
export AGENT_INFERENCE_THREADS=8

# Force PPO off the shared GPU; keep inference on GPU.
export PPO_DEVICE=cpu

# Important for your current PPO-zero issue on shared GPU:
export PPO_PARALLEL_AGENTS=1
export PPO_EXECUTOR_WORKERS=1
export RL_PPO_ROLLOUT_STEPS=16384

export TM_TRANSPORT_TIMING_DEBUG=0
export TM_HTTP_PREWARM_CONNECTIONS=4
export RL_TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL=7

chmod +x start-rl-cloud-training.sh
./start-rl-cloud-training.sh

```

### Bootstrap from a saved champion

If you want a new machine to start from an existing champion checkpoint instead of a random population:

```bash
export BOOTSTRAP_CHECKPOINT_PATH=/app/rl-models-global/champion/current/champion.pth
export BOOTSTRAP_POPULATION_MODE=mutated_copies   # clone | mutated_copies | seed_and_fresh
export BOOTSTRAP_MUTATION_RATE=0.08
export TRAINING_START_GENERATION=0
```

Typical uses:
- Continue numbering from a chosen point: `export TRAINING_START_GENERATION=100`
- Start from the imported champion as a fresh run: `export TRAINING_START_GENERATION=0`


You can verify the generated training volume in `docker-compose.rl_cloud.generated.yml`:
- `GAMES_PER_EVAL`
- `POPULATION_SIZE`
- `GLOBAL_GAME_CONCURRENCY`
- `PPO_ROLLOUT_STEPS`

### GPU-accelerated inference (automatic)

When a CUDA GPU is available inside the coordinator container (requires NVIDIA Container Toolkit), agent inference runs on GPU automatically (`AGENT_INFERENCE_DEVICE=auto`). All 16 agent networks are kept on GPU in FP16 autocast mode, and concurrent inference requests from the same agent are batched into a single forward pass. This typically cuts per-decision latency from ~5-20 ms (CPU) to <1 ms.

| Variable | Default | Effect |
|----------|---------|--------|
| `AGENT_INFERENCE_DEVICE` | `auto` | `auto` = CUDA if available, else CPU. Set `cpu` to force CPU inference. |
| `AGENT_INFERENCE_BATCH` | `auto` | `auto` = batching enabled on CUDA. Set `0` to disable batching. |
| `AGENT_INFERENCE_BATCH_SIZE` | `32` | Max batch size per flush (higher = more throughput, slightly more latency). |
| `AGENT_INFERENCE_BATCH_DEADLINE_MS` | `3.0` | Max wait before flushing a partial batch (ms). |
| `AGENT_INFERENCE_THREADS` | `0` (auto) | Thread pool size. Auto picks 4 for CUDA, `min(32, cpu_count)` for CPU. |

With the saturate profile on a single-GPU host, the 16 agents consume ~2 GB VRAM for inference (FP16), leaving ~9 GB for PPO training batches.

### Coordinator asyncio optimizations (automatic)

Coordinator and champion orchestrator use **uvloop** when available (Linux): 2–4x faster event loop for HTTP and game I/O. On Windows, the default asyncio loop is used.

**PPO offload**: PPO training runs in a dedicated `ThreadPoolExecutor`, so the event loop stays responsive for HTTP, polling, and game handling. Agents are optimized in parallel (limited by `PPO_PARALLEL_AGENTS`).

**orjson**: JSON parsing/serialization uses `orjson` when available for faster HTTP handling.

Optional tuning:

| Variable | Default | Effect |
|----------|---------|--------|
| `PPO_EXECUTOR_WORKERS` | `4` | Thread pool size for PPO. Increase if PPO becomes a bottleneck (1–16). |
| `PPO_PARALLEL_AGENTS` | `4` | Max agents optimized in parallel. Higher = faster PPO phase; watch GPU memory (1–16). |

### If games exceed ~1200s (long game duration)

Per-move latency can be acceptable while full game duration is still too high when draft-heavy options are enabled.
Use the fast training preset (no draft, no Moon/Venus):

```bash
export RL_GAME_OPTIONS_FILE=/app/game_options.fast_training.json
./start-rl-cloud-training.sh
```

The cloud compose generator now forwards this into coordinator env as `GAME_OPTIONS_FILE`.

## 6) Monitor

```bash
docker compose -f docker-compose.rl_cloud.generated.yml ps
docker compose -f docker-compose.rl_cloud.generated.yml logs -f rl-coordinator    # or rl-coordinator-1, rl-coordinator-2, etc.
```

Dashboard (use 127.0.0.1 when accessing via SSH port forwarding, e.g. Vast.ai):
- `http://127.0.0.1:5000/dashboard`
- `http://127.0.0.1:5000/stats`
- In multi-coordinator mode, `/dashboard` now opens a global aggregate view by default.
- `http://127.0.0.1:5100/dashboard` -> global dashboard
- `http://127.0.0.1:5100/dashboard/local` -> coordinator 1 local detail
- `http://127.0.0.1:5101/dashboard/local` -> coordinator 2 local detail

For port forwarding: forward ports 5000 and 8081â€“8098 (or 8081â€“8098 for 18 game servers), then use the 127.0.0.1 URLs. Set `PUBLIC_HOST=localhost` when running so game links in the dashboard work.

## 7) Troubleshooting / Games Dropped

If the coordinator cannot keep up with many game servers and games are dropped (slot timeouts, create failures), try these env-only tweaks before or in addition to the saturate profile:

| Variable | Suggested | Effect |
|----------|-----------|--------|
| `RL_TRAINING_PROFILE=saturate` | saturate | Higher concurrency, lower poll/failure delays |
| `RL_AGENT_POLL_INTERVAL_SEC=0.03` | 0.03 | Faster polling (more CPU); safe with GPU inference |
| `RL_AGENT_FAILURE_PAUSE_SEC=0.03` | 0.03 | Shorter pause on failure |
| `RL_TM_SERVER_SLOT_WAIT_TIMEOUT_SEC=90` | 90 | Less slot timeout drops |
| `RL_TM_CREATE_GAME_RETRY_ATTEMPTS=6` | 6 | More retries before failing |
| `RL_HTTP_CONNECTOR_LIMIT=3072` | 3072 | More concurrent connections |
| `RL_HTTP_CONNECTOR_LIMIT_PER_HOST=96` | 96 | More connections per host |
| `RL_AGENT_INFERENCE_THREADS=32` | 32 | Thread pool size for inference (0 = auto) |
| `RL_TOURNAMENT_CONCURRENCY=72` | match GLOBAL_GAME_CONCURRENCY | Explicit concurrency override |
| `RL_GLOBAL_GAME_CAP_PER_COORD=20` | 20 | Hard cap for per-coordinator `GLOBAL_GAME_CONCURRENCY` (default: 20 saturate / 24 balanced) |
| `RL_TOURNAMENT_CAP_PER_COORD=20` | 20 | Hard cap for per-coordinator `TOURNAMENT_CONCURRENCY` (default: global cap) |
| `RL_TM_HTTP_REQUEST_TIMEOUT_SEC=120` | 120 | HTTP request timeout (reduces get_player_state TimeoutError) |
| `RL_TM_GET_STATE_RETRY_ATTEMPTS=5` | 5 | More retries for get_player_state |
| `RL_TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL=7` | 7 | More retries for `initialCards` burst disconnects |
| `RL_INITIAL_CARDS_JITTER_MS=1200` | 1200 | Spreads startup `initialCards` posts to avoid thundering herd |
| `RL_TM_RECYCLE_SESSION_ON_DISCONNECT=0` | 0 | Avoid recycle churn during transient disconnect bursts |
| `RL_NUM_COORDINATORS=2` or `3` | 2â€“3 | Multi-coordinator: each gets a subset of servers |
| `RL_COORDINATORS_SHARE_GPU=1` | 1 | When using 2+ coordinators: share one GPU (for pipeline bottleneck, single-GPU hosts) |
| `RL_TM_GAME_TIMEOUT_SEC=600` | 600 | Game timeout in seconds (default 420); increase if games are slow or "cancelled unexpectedly" |
| `RL_TM_HTTP_FORCE_CLOSE_CONNECTIONS=0` | 0 (cloud default) | HTTP keep-alive for better async I/O; 1 = close after each request |


### Global champion orchestrator (multi-coordinator)

When `RL_NUM_COORDINATORS>1`, the generated compose now includes `rl-champion-orchestrator`.
It evaluates top checkpoints across coordinators and keeps one global champion under `./rl-models-global`.

Default orchestrator contract:

```bash
ORCH_COORD_SOURCES=coord-1=/app/coord-models/coord-1,coord-2=/app/coord-models/coord-2
ORCH_OUTPUT_ROOT=/app/rl-models-global
ORCH_TRIGGER_EVERY_N_GENS=1
ORCH_TOP_K_PER_COORD=2
ORCH_GAMES_PER_CANDIDATE=6
ORCH_GLOBAL_GAME_CONCURRENCY=2
ORCH_TOURNAMENT_CONCURRENCY=2
ORCH_MIN_GAMES_FOR_PROMOTION=24
ORCH_MIN_COMPLETION_RATE=0.90
ORCH_WIN_RATE_MARGIN=0.03
```

Coordinator role defaults in multi-coordinator mode:
- **All coordinators** (trainer): `PPO_ENABLE=1`, `SAVE_TOP_K=2`, `SAVE_EVERY_N_GENERATIONS=1`, `MAX_SAVED_GENERATIONS=30`, `TRAINING_POOL_EXTRA_CHECKPOINTS=/app/rl-models-global/champion/current/champion.pth`

Artifacts:
- Current champion checkpoint: `rl-models-global/champion/current/champion.pth`
- Current champion manifest: `rl-models-global/champion/current/champion_manifest.json`
- Round history: `rl-models-global/champion/history/<round_id>/`
- Orchestrator state: `rl-models-global/orchestrator_state.json`

Monitor:

```bash
docker compose -f docker-compose.rl_cloud.generated.yml logs -f rl-champion-orchestrator
```

Troubleshooting:
- If champion updates never happen, verify `GAME_SERVERS` reachability from orchestrator and check `ORCH_COORD_SOURCES` mounts.
- If promotion is too strict/slow, lower `ORCH_MIN_GAMES_FOR_PROMOTION` or `ORCH_WIN_RATE_MARGIN`.
- If orchestrator competes too much with training throughput, lower `ORCH_GLOBAL_GAME_CONCURRENCY` (default `2`).
- If host port `5000` is occupied, set `RL_COORDINATOR_BASE_PORT` (e.g. `5100`) before `./start-rl-cloud-training.sh`.

