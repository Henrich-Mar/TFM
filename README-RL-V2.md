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

Open `http://localhost:5000/decision-explainer`. In the regular correction
workflow, the collector continues playing while you review its snapshots:

1. Select a snapshot and click **Arm capture**.
2. When its decision appears, select every legal action you would accept (or
   choose **Skip** if none should become training data).
3. Save the annotation, then arm the next snapshot.

Low-confidence teacher decisions and large teacher/student disagreements should
be reviewed first; both values are shown in the captured decision details.

For a deliberate, full-game review of one player, use guided annotation. It
automatically creates a snapshot **before every decision** for the selected
seat and holds the game until that snapshot is labelled. Select every action
you consider acceptable and save the annotation; do not press "Arm Next
Snapshot" in this mode. Run one game at a time, then start the next seed when
the process finishes:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm --service-ports rl-coordinator python -m training.v2_collect_teacher --stage 0 --games 1 --seed-start 10000 --serve-api --annotate-seat 0
```

Seat 0 is `teacher-v1-seat-0`. Use `--annotate-seat 1`, `2`, or `3` to review a
different player. Guided annotation waits indefinitely by default; add
`--annotation-timeout-sec 600` only if you want an automatic timeout. These
are manual labels: the game will not advance until you save or skip the current
decision.

Import the annotations:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_import_annotations --snapshots /app/debug_snapshots --annotations /app/debug_snapshots/annotations --dataset /app/v2/teacher-dataset
```

At least 100 non-skipped human labels are required by the pretraining gate.

## 4. Passively record your own games

The guided Decision Explainer is useful for corrections, but it is not needed
to learn directly from a game you play yourself. The passive listener receives
each submitted UI input together with the exact private player view that was
available before that choice. It never waits for the collector and never
changes a submitted move.

Start the game servers with the listener endpoint configured. On Docker Desktop
the game-server containers can reach the listener on the host through
`host.docker.internal`. This recreates those four game-server containers, so
finish any active games first:

```powershell
$env:TFM_HUMAN_LISTENER_URL = "http://host.docker.internal:5000/human-listener/decision"
$env:TFM_HUMAN_LISTENER_TOKEN = "replace-with-a-long-local-secret"
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml up -d --build tfm-server-1 tfm-server-2 tfm-server-3 tfm-server-4
```

Then run the listener. Use a stable name for yourself (for example `You`) in
each game; `--player-name` makes the listener ignore the bot seats even though
the game server forwards every move to the local endpoint.

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm --service-ports `
  -e TFM_HUMAN_LISTENER_TOKEN=$env:TFM_HUMAN_LISTENER_TOKEN `
  -e MAX_ACTIVE_GAMES_PER_SERVER=3 rl-coordinator `
  python -m training.v2_listen_human --player-name You --token $env:TFM_HUMAN_LISTENER_TOKEN `
  --games 10 --human-name You --beginner
```

When the listener starts with `--games 10`, it creates ten games in which your
seat is named `You` and the other three seats are teacher bots. It prints a
private `player_url` for each game in the terminal, writes the same list to
`rl-v2/human-game-links.json` on the host, and displays them in the **Passive
Human Games** panel at `http://localhost:5000/decision-explainer`.

For each game, open its **player URL** (not the public game URL), choose
corporation/cards, and play normally. There is nothing to arm or label in
Decision Explainer: each successful move by `You` is automatically recorded
with the exact board/player state visible immediately before that move. The
listener ignores the three bot seats. After a game ends, it writes one complete
human episode to `/app/v2/teacher-dataset`; raw events (including currently
unsupported input types) are retained in `/app/v2/human-listener-events`.
Final rank and VP are stored as value targets. Keep the listener terminal open
while playing; stop it with `Ctrl+C` only after you have finished the batch.
<!-- Superseded wording retained only to avoid changing an existing malformed encoding.
Completed games also record final rank and VP as value targets. Stop the
listener with `Ctrl+C` after your 10–20 games. The command prints one private
`player_url` per game and writes them to `rl-v2/human-game-links.json` on the host.
-->

## 5. Pretrain fresh weights

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm rl-coordinator python -m training.v2_pretrain --dataset /app/v2/teacher-dataset --output /app/v2/pretrain --batch-size 32 
```

PPO remains blocked unless `pretrain_report.json` records at least 85% top-1,
97% top-3 and 80% top-3 on the human annotations.
Pretraining is deterministic by default (`--seed 20260901`) and refuses to
overwrite a non-empty output directory. For an intentional clean BC restart,
set `V2_ALLOW_PRETRAIN_OVERWRITE=1`; this is not a checkpoint resume.

## 6. Run the single-learner curriculum

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
Use these commands in the future:
# Gracefully stop and save current progress
docker kill --signal=SIGINT tfm-rl-v2-selfplay
# Resume in the background
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -d --name tfm-rl-v2-selfplay -e V2_ALLOW_RESUME=1 rl-coordinator python -m training.v2_self_play --bc-checkpoint /app/v2/pretrain/bc_best.pth --root /app/v2
#recycling :
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml run --rm -d --name tfm-rl-v2-selfplay -e V2_ALLOW_RESUME=1 -e V2_RECYCLE_IDLE_SERVERS=1 rl-coordinator python -m training.v2_self_play --bc-checkpoint /app/v2/pretrain/bc_best.pth --root /app/v2

# Follow the output
docker logs -f --tail 100 tfm-rl-v2-selfplay
The active run successfully loaded: