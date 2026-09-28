# AlphaGo Zero-style four-seat self-play

This profile trains one PPO policy from random initialization in four-player
Terraforming Mars games. All live seats share the same network, optimizer, and
rollout buffer. It is inspired by AlphaGo Zero's self-play loop, but it uses PPO
without Monte Carlo tree search.

The TypeScript search-simulation foundation for the next MCTS phase is
documented in [aplhago-mcts.md](aplhago-mcts.md). It ships disabled and does
not change the current PPO training loop.

The profile now uses inexpensive screening benchmarks during training and runs
the full promotion suite only for promising candidates. This keeps routine
work near an 80/10/10 split between self-play, teacher screening, and champion
screening instead of spending most server time on evaluation.

## Default training budget

The important defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `PPO_ROLLOUT_STEPS` | 12,288 | Decisions collected before one PPO update |
| Benchmark interval | 100,000 decisions | Cadence for the light screen |
| Screen seeds | 8 | 32 games per baseline after four seat rotations |
| History snapshot interval | 8 PPO updates | Cadence for adding a frozen past policy |
| Retained historical policies | 8 | Opponent-diversity window |
| Historical game probability | 25% | One of four seats is historical in selected games |

In the initial run, one complete self-play game produced approximately 406
decisions. At that rate, 100,000 decisions gives approximately 246 self-play
games between screens. Each screen uses 32 teacher and 32 champion games:

```text
self-play : teacher screen : champion screen
     246  :       32       :        32
     7.7  :        1       :         1
```

The exact ratio changes with game length. A useful formula is:

```text
self-play games = benchmark interval / recent decisions per self-play game
screen games per baseline = number of screen seeds * 4 seats
```

## Evaluation and promotion flow

At every 100,000-decision interval, the runner saves a candidate and performs
the following steps:

1. Play 32 games against the heuristic teacher using eight dedicated screen
   seeds and all four candidate seat positions.
2. Play 32 games against the current champion on the same screen seeds.
3. Continue only if at least 90% of each screen completed, neither screen had a
   policy rejection, teacher first-place rate is at least 25%, and champion
   pairwise score is at least 0.50.
4. Run the full 120-game teacher benchmark on 30 separate promotion seeds.
5. If and only if the full teacher gate passes, run the full 120-game champion
   benchmark.
6. Promote only if both full gates pass.

The full teacher gate requires a 95% Wilson lower bound on first-place rate
above 25%, at least 99% game completion, and zero policy rejections. The full
champion gate requires pairwise score of at least 0.50, at least 99% completion,
and zero policy rejections.

Screen seeds are stored in
`rl-environment/benchmark_screen_seeds.v1.json`. Promotion seeds remain in
`rl-environment/benchmark_seeds.v1.json`. The sets are intentionally disjoint,
so repeatedly screening candidates does not select directly on the final
promotion games. Neither set is used for self-play training.

Screen reports have `_screen.json` filenames. Full promotion reports keep the
original filenames, for example:

```text
benchmark_candidate_000200000_stage1_teacher_screen.json
benchmark_candidate_000200000_stage1_champion_screen.json
benchmark_candidate_000200000_stage1_teacher.json
benchmark_candidate_000200000_stage1_champion.json
```

If a screen fails, the last two files are not produced. If the full teacher
gate fails, the full champion file is not produced.

## Historical opponents versus the champion

The champion and the historical pool now have separate responsibilities:

- `champion.pth` changes only after both strict promotion gates pass.
- Every eight successful PPO updates, the live learner is also archived as a
  non-champion historical snapshot.
- The last eight historical checkpoints are eligible as frozen opponents.
- In 25% of self-play games, one of four live seats is replaced by one frozen
  historical seat.

This means approximately 6.25% of training seats are historical after the
first snapshot, while 93.75% remain live-policy seats. Training therefore gains
opponent diversity even when a from-scratch learner has not beaten the teacher
yet.

## Optional MCTS search for benchmarks (disabled by default)

The Python search consumer ([aplhago-mcts.md](aplhago-mcts.md)) can let a
frozen benchmark candidate search its strategic top-level prompts with
determinized rollouts through the game server's `/api/rl/search` service.
Search is evaluation-only: searched actions never enter the PPO buffer, and
self-play training does not use it.

Search requires the servers to run the search service with determinization
enabled, because opponent continuations need fair hidden-information samples:

```powershell
$env:ALPHAGO_RL_SEARCH_ENABLED="1"
$env:ALPHAGO_RL_SEARCH_DETERMINIZATION_ENABLED="1"
$env:ALPHAGO_SEARCH_ENABLED="1"      # coordinator-side consumer
$env:ALPHAGO_SEARCH_MODE="lookahead" # or "puct"
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate
```

Then run a searched candidate with the ordinary benchmark entry point, for
example `python -m training.v2_benchmark --checkpoint ... --baseline teacher ...`.
The report gains a `search` block:

```json
{
  "search": {
    "enabled": true,
    "mode": "lookahead",
    "searched_decisions": 1810,
    "search_fallbacks": 12,
    "mean_replay_batch_sec": 0.21,
    "applied_inputs_per_sec": 132.5
  }
}
```

`applied_inputs_per_sec` is the measured feed for the 100 inputs/s hardware
gate in [aplhago-mcts.md](aplhago-mcts.md). Tuning controls:
`ALPHAGO_SEARCH_TOP_K`, `ALPHAGO_SEARCH_DETERMINIZATIONS`,
`ALPHAGO_SEARCH_SIMULATIONS`, `ALPHAGO_SEARCH_DEPTH`,
`ALPHAGO_SEARCH_PUCT_C`, `ALPHAGO_SEARCH_SELECTION`, `ALPHAGO_SEARCH_SEED`.
The client keeps the default 8 candidates × 8 determinizations inside the
server's 64-branch batch limit. If the search service is off, full, or a root
is not reconstructible, the candidate silently falls back to plain policy
sampling — completion and rejection rates stay the promotion criteria.

For a strict A/B of search versus sampling, run the same checkpoint and seeds
once with `ALPHAGO_SEARCH_ENABLED=0` and once with it set; weights, seats, and
seeds are identical, so any difference comes from search alone.

To see one decision at a time, `training.search_inspect` plays a single
fixed-seed game and prints/saves each searched tree (`P` prior, `N` visits,
`V` mean simulated value, `*` chosen):

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml run --rm --no-deps `
  -e GAME_SERVERS=tfm-server-1:8080 rl-coordinator python -m training.search_inspect `
  --checkpoint /app/alphago/checkpoints/latest_learner.pth --game-seed 920003 --mode puct
```

Traces land in `rl-alphago/metrics/mcts_trace_seed<seed>_seat<n>.txt` plus a
JSON with the full visit distribution per decision. Keep
`ALPHAGO_SEARCH_ENABLED=0` for normal training runs: search is evaluation and
inspection tooling, and searched actions never train the PPO policy.

## Start training

Build and start the hard-rules base plus the AlphaGo overlay:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up --build
```

No bootstrap checkpoint is required. Outputs are isolated under
`rl-alphago/`. The initial random policy is saved as
`rl-alphago/checkpoints/champion.pth`.

For a detached run:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up --build -d
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml logs -f rl-coordinator
```

## Apply the new cadence to an existing run

The Python source is bind-mounted, but the benchmark interval is part of the
container command. Recreate the coordinator to apply the new default. The
learner resumes from `latest_learner.pth` and `selfplay_state.json`.

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml stop rl-coordinator
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate rl-coordinator
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml logs -f rl-coordinator
```

If the old coordinator is interrupted during a benchmark, completed training
is retained, but that partially completed benchmark is not added to the state
report history.

## Continue beyond one million decisions

The decision target is cumulative:

```powershell
$env:ALPHAGO_MAX_DECISIONS=2000000
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate rl-coordinator
```

## Tune the budget

PowerShell environment variables can override the defaults before recreating
the coordinator:

```powershell
$env:ALPHAGO_BENCHMARK_INTERVAL=100000
$env:ALPHAGO_PPO_ROLLOUT_STEPS=12288
$env:ALPHAGO_HISTORY_SNAPSHOT_INTERVAL_UPDATES=8
$env:ALPHAGO_SCREEN_MIN_COMPLETION_RATIO=0.90
$env:ALPHAGO_SCREEN_TEACHER_MIN_FIRST_PLACE_RATE=0.25
$env:ALPHAGO_SCREEN_CHAMPION_MIN_PAIRWISE_SCORE=0.50

docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate rl-coordinator
```

Recommended ranges:

| Control | Recommended range | Trade-off |
| --- | ---: | --- |
| Benchmark interval | 100k-200k decisions | Larger values favor self-play throughput |
| PPO rollout | 12,288-16,384 decisions | Larger values improve batch diversity but update less often |
| History snapshot cadence | 8-16 updates | Smaller values adapt the opponent pool faster |
| Teacher screen threshold | 0.25-0.30 | Higher values run fewer full gates |

Keep `PPO_ROLLOUT_STEPS=12288` unless PPO metrics show that the batch is too
small or optimization is unstable. Benchmark frequency and rollout size solve
different problems: increasing rollout size is not a substitute for reducing
evaluation overhead.

For the old full 120+120 benchmark behavior, a reasonable interval is at least
250k decisions. Approximately 393k decisions, or 32 default rollouts, produces
about an 8:1:1 self-play/teacher/champion ratio. The progressive screen normally
makes such a long interval unnecessary.

## Monitor the run

Current progress is written to:

```text
rl-alphago/metrics/selfplay_progress.json
```

Important fields are:

- `decisions`: cumulative live-policy decisions.
- `games`: self-play games; benchmark games are not included.
- `rollout_buffer`: currently queued PPO decisions.
- `next_benchmark_decision`: next light-screen threshold.
- `status`: current self-play lifecycle status.

Durable resume and evaluation history are in:

```text
rl-alphago/metrics/selfplay_state.json
```

Each new report contains `screen_teacher`, `screen_regression`, optional full
`teacher` and `regression` reports, `screen_passed`, and `promoted`. A `null`
full report means that a preceding gate deliberately short-circuited it.

Check running services and recent logs with:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml ps
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml logs --tail 200 rl-coordinator
```

Stop the entire stack with:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml down
```
