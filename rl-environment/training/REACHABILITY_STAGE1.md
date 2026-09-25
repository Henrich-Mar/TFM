# Tharsis stage-1 reachability experiment

Scope: four players, Tharsis, Corporate Era, no expansions or draft. The source
of truth is `game_options.v2_stage1.json`. Run from the repository root with the
V4 compose stack and mounted game servers running.

## 1. Test the teacher before collecting

```bash
docker compose -f docker-compose.rl_v4.yml run --rm rl-coordinator \
  python -m training.v4_teacher_ab --output /app/v4/benchmarks/teacher_ab_940000.json
```

This runs exactly 20 games, seeds 940000–940019, with the reachability teacher
rotating seats against three original teachers. The report contains every rank,
completion count, rejection count and a 95% interval. `decision=collect` needs a
mean rank below 2.35, `decision=retune` means above 2.65, and `run_40_more`
means the result is too close to 2.5 to call. In the last case run another 40
games starting at 940020 and combine the 60 finished ranks before deciding.
Do not collect if games fail or server actions are rejected.

## 2. Collect and fine-tune only after the teacher wins

Use a fresh directory and seeds starting at 960000:

```bash
docker compose -f docker-compose.rl_v4.yml run --rm rl-coordinator \
  python -m training.v4_collect_teacher --stage 1 --seed-start 960000 \
  --games 500 --dataset /app/v4/teacher-dataset-reachability
```

Check the dataset audit for at least 100,000 teacher decisions from completed
games. Increase `--games` with a new, non-overlapping seed range if needed.
Copy only the human shards from `teacher-dataset-v5` into the matching train and
test splits; retain the eight training games and two held-out test games. Never
copy previous teacher shards. Verify the audit before training.

```bash
docker compose -f docker-compose.rl_v4.yml run --rm rl-coordinator \
  python -m training.v4_pretrain \
  --dataset /app/v4/teacher-dataset-reachability \
  --output /app/v4/pretrain-reachability \
  --init-checkpoint /app/v4/pretrain-v2/bc_best.pth \
  --learning-rate 1e-5 --keep-action-tail
```

The existing `rl-v4/eval_strength.py` and runaway cap were not present on the
GitHub branch at the time of this change. Bring the local changes into the
branch before evaluating the student. Evaluate stage 1 against the new teacher
using seeds 950000–950019; attach the checkpoint hash, completed count, mean
rank, runaway count, teacher identity, zero server rejections and stage to the
strength report. Set `V4_STRENGTH_REPORT` to that JSON path when launching PPO.
The PPO gate requires at least 20 finished games, mean rank at most 2.55,
at most one runaway, matching checkpoint hashes, and a clean bound smoke run.

Only claim a human win after at least 40 finished, seat-rotated games against
people, with the 95% interval on mean rank entirely below 2.5.
