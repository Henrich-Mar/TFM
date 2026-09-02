# Terraforming Mars RL v2

V2 is a clean experiment: fresh model weights, dataset, rollouts, metrics and
champions. It never discovers or loads checkpoints from the legacy training
directories.

## 1. Build and initialize

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml build
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm --no-deps rl-coordinator
```

The second command only validates and creates `/app/v2`; it cannot start the
legacy evolution loop.

The v2 override enables request-controlled seeds only on the RL game servers.
Normal Terraforming Mars deployments keep random create-game seeds. It also
disables legacy startup autosubmit so startup choices enter the teacher/BC
dataset and later PPO rollouts.

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
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_pretrain --dataset /app/v2/teacher-dataset --output /app/v2/pretrain --batch-size 16
```

PPO remains blocked unless `pretrain_report.json` records at least 85% top-1,
97% top-3 and 80% top-3 on the human annotations.
Pretraining is deterministic by default (`--seed 20260901`) and refuses to
overwrite a non-empty output directory. For an intentional clean BC restart,
set `V2_ALLOW_PRETRAIN_OVERWRITE=1`; this is not a checkpoint resume.

## 5. Run the single-learner curriculum

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -e V2_ALLOW_RESUME=1 rl-coordinator python -m training.v2_self_play --bc-checkpoint /app/v2/pretrain/bc_best.pth --root /app/v2
```

-- CPU 
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -e V2_ALLOW_RESUME=1 -e PPO_DEVICE=cpu rl-coordinator python -m training.v2_self_play --bc-checkpoint /app/v2/pretrain/bc_best.pth --root /app/v2

The learner starts at Stage 0, benchmarks every 25,000 decisions, promotes only
against fixed baselines, then advances to Stage 1. Only the main learner writes
PPO rollouts; teacher, random and champion opponents are frozen.
Self-play uses `SELFPLAY_CONCURRENCY` simultaneous games per batch (default `2`
in `docker-compose.rl_v2.yml`). PPO runs only after the entire batch completes,
so every rollout in that batch came from the same policy. Start at `2`; after a
stable run, raise it to at most the number of available game-server slots.
The self-play worker does not serve the FastAPI dashboard; monitor its terminal
output or `rl-v2/metrics/selfplay_progress.json`. The dashboard URL is for the
coordinator/API process used during annotation and legacy coordinator runs.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider rl-environment/tests -k v2
npm.cmd --prefix terraforming-mars run test:server -- --grep ApiCreateGame
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml config --quiet
```

These tests cover full-episode PPO/GAE handling, strict policy-version
filtering, atomic teacher shards, benchmark-seed and game split leakage,
teacher legality/action families, clean-runtime isolation, BC checkpoint
creation, and fixed-seed server initialization. The 100-game smoke test and
Stage 0/1 gates are empirical training gates and therefore run after data/model
generation rather than as repository unit tests.
