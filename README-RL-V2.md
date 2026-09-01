# Terraforming Mars RL v2

V2 is a clean experiment: fresh model weights, dataset, rollouts, metrics and
champions. It never discovers or loads checkpoints from the legacy training
directories.

## 1. Build and initialize

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml build
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator
```

The second command only validates and creates `/app/v2`; it cannot start the
legacy evolution loop.

## 2. Collect teacher games

Stage 0 uses beginner corporations and the base Tharsis game:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_collect_teacher --stage 0 --games 100 --seed-start 10000
```

Stage 1 enables Corporate Era and corporation selection:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -e GAME_OPTIONS_FILE=/app/game_options.v2_stage1.json rl-coordinator python -m training.v2_collect_teacher --stage 1 --games 100 --seed-start 20000
```

Continue collection until the dataset contains at least 100,000 decisions and
teacher fallback stays below 5%.

## 3. Add human corrections

Run teacher collection with the annotation API in the same process (use enough
games to leave time for review):

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm --service-ports rl-coordinator python -m training.v2_collect_teacher --stage 0 --games 1000 --seed-start 30000 --serve-api
```

Open `http://localhost:5000/decision-explainer`, arm a capture, then select one
or more acceptable legal actions or mark the position skipped. Low-confidence
teacher decisions and large teacher/student disagreements should be reviewed
first; both values are shown in the captured decision details.
Import the annotations:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_import_annotations --snapshots /app/debug_snapshots --annotations /app/debug_snapshots/annotations --dataset /app/v2/teacher-dataset
```

At least 100 non-skipped human labels are required by the pretraining gate.

## 4. Pretrain fresh weights

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_pretrain --dataset /app/v2/teacher-dataset --output /app/v2/pretrain
```

PPO remains blocked unless `pretrain_report.json` records at least 85% top-1,
97% top-3 and 80% top-3 on the human annotations.

## 5. Run the single-learner curriculum

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -e V2_ALLOW_RESUME=1 rl-coordinator python -m training.v2_self_play --bc-checkpoint /app/v2/pretrain/bc_best.pth --root /app/v2
```

The learner starts at Stage 0, benchmarks every 25,000 decisions, promotes only
against fixed baselines, then advances to Stage 1. Only the main learner writes
PPO rollouts; teacher, random and champion opponents are frozen.

## Tests

```powershell
venv\Scripts\python.exe -m pytest -q -p no:cacheprovider rl-environment/tests/test_v2_*.py
```
