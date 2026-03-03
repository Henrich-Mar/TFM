# Vast.ai Cloud Setup (Ubuntu)

This setup keeps your RL code in this repo (`Vast-cloud` branch), keeps the game code in a local `terraforming-mars/` folder, and starts auto-sized training.

## 1) Prepare VM

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3 python3-venv

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
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
git clone -b Vast-cloud <YOUR_TFM_RL_REPO_URL> TFM
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

Override sizing with env vars before launch:

```bash
export RL_MAX_SERVERS=64
export RL_SERVER_MEM_MB=1600
export RL_GAMES_PER_SERVER=5
export RL_NODE_HEAP_MB=1200
export RL_CPU_SERVER_RATIO=2.5
export PUBLIC_HOST=$(curl -s ifconfig.me)
./start-rl-cloud-training.sh
```

## 5) Monitor

```bash
docker compose -f docker-compose.rl_cloud.generated.yml ps
docker compose -f docker-compose.rl_cloud.generated.yml logs -f rl-coordinator
```

Dashboard:
- `http://<vm-ip>:5000/dashboard`
- `http://<vm-ip>:5000/stats`
