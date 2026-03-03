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

## 4) Tune speed (optional)

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
export PUBLIC_HOST=localhost
# Capacity controls for 40 vCPU, 387 GB RAM
export RL_TRAINING_PROFILE=saturate
export RL_MIN_SERVERS=12
export RL_MAX_SERVERS=18
export RL_CPU_SERVER_RATIO=1.0
export RL_SERVER_MEM_MB=1400
export RL_NODE_HEAP_MB=1050
export RL_GAMES_PER_SERVER=4
export RL_HTTP_CONNECTOR_LIMIT_PER_HOST=96
export RL_HTTP_CONNECTOR_LIMIT=3072
export RL_AGENT_POLL_INTERVAL_SEC=0.08
export RL_AGENT_FAILURE_PAUSE_SEC=0.05
export RL_INITIAL_CARDS_JITTER_MS=250

# Optional hard overrides (leave unset for auto from profile)
export RL_GAMES_PER_EVAL=30
# export RL_PPO_ROLLOUT_STEPS=131072

./start-rl-cloud-training.sh
```

You can verify the generated training volume in `docker-compose.rl_cloud.generated.yml`:
- `GAMES_PER_EVAL`
- `POPULATION_SIZE`
- `GLOBAL_GAME_CONCURRENCY`
- `PPO_ROLLOUT_STEPS`

## 5) Monitor

```bash
docker compose -f docker-compose.rl_cloud.generated.yml ps
docker compose -f docker-compose.rl_cloud.generated.yml logs -f rl-coordinator    # or rl-coordinator-1, rl-coordinator-2, etc.
```

Dashboard (use 127.0.0.1 when accessing via SSH port forwarding, e.g. Vast.ai):
- `http://127.0.0.1:5000/dashboard`
- `http://127.0.0.1:5000/stats`

For port forwarding: forward ports 5000 and 8081–8098 (or 8081–8098 for 18 game servers), then use the 127.0.0.1 URLs. Set `PUBLIC_HOST=localhost` when running so game links in the dashboard work.

## 6) Troubleshooting / Games Dropped

If the coordinator cannot keep up with many game servers and games are dropped (slot timeouts, create failures), try these env-only tweaks before or in addition to the saturate profile:

| Variable | Suggested | Effect |
|----------|-----------|--------|
| `RL_TRAINING_PROFILE=saturate` | saturate | Higher concurrency, lower poll/failure delays |
| `RL_AGENT_POLL_INTERVAL_SEC=0.05` | 0.05 | Faster polling (more CPU) |
| `RL_AGENT_FAILURE_PAUSE_SEC=0.03` | 0.03 | Shorter pause on failure |
| `RL_TM_SERVER_SLOT_WAIT_TIMEOUT_SEC=90` | 90 | Less slot timeout drops |
| `RL_TM_CREATE_GAME_RETRY_ATTEMPTS=6` | 6 | More retries before failing |
| `RL_HTTP_CONNECTOR_LIMIT=3072` | 3072 | More concurrent connections |
| `RL_HTTP_CONNECTOR_LIMIT_PER_HOST=96` | 96 | More connections per host |
| `RL_AGENT_INFERENCE_THREADS=32` | 32 | Thread pool size for inference (0 = auto) |
| `RL_TOURNAMENT_CONCURRENCY=72` | match GLOBAL_GAME_CONCURRENCY | Explicit concurrency override |
| `RL_TM_HTTP_REQUEST_TIMEOUT_SEC=120` | 120 | HTTP request timeout (reduces get_player_state TimeoutError) |
| `RL_TM_GET_STATE_RETRY_ATTEMPTS=5` | 5 | More retries for get_player_state |
| `RL_NUM_COORDINATORS=2` or `3` | 2–3 | Multi-coordinator: each gets a subset of servers |
| `RL_COORDINATORS_SHARE_GPU=1` | 1 | When using 2+ coordinators: share one GPU (for pipeline bottleneck, single-GPU hosts) |

### Multi-coordinator (if single coordinator still times out)

With 18 servers and persistent `get_player_state` timeouts, run 2–3 coordinators in parallel, each with a subset of servers.

**Option A – Dedicated GPUs** (default): Each coordinator gets its own GPU. Requires N GPUs for N coordinators.

**Option B – Shared GPU**: When the bottleneck is the pipeline (game servers, HTTP), not GPU compute, coordinators can share one GPU:

```bash
export RL_NUM_COORDINATORS=2   # or 3
export RL_COORDINATORS_SHARE_GPU=1
export RL_TRAINING_PROFILE=saturate
# ... other saturate vars ...
./start-rl-cloud-training.sh
```

**Note:** Multiple coordinators on one GPU share VRAM. Ensure the GPU has enough (e.g. 2 coordinators typically need 8GB+ total). Use `RL_GPU_INDEX=0` (default) or another index if you have multiple GPUs and want coordinators to share a specific one.

Dashboards: `http://127.0.0.1:5000`, `http://127.0.0.1:5001`, etc.

Check coordinator logs for:
- `No game server capacity available` (slot timeout)
- `Failed to create game` (create_game retries exhausted)
- High `payment_reject_count` (server overload)
