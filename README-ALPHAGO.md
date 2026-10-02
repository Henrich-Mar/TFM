# AlphaGo Zero-style four-seat self-play

This profile trains one PPO policy from random initialization in four-player
Terraforming Mars games. All live seats share the same network, optimizer, and
rollout buffer. It is inspired by AlphaGo Zero's self-play loop, but it uses PPO
without Monte Carlo tree search.

The TypeScript search-simulation foundation for the next MCTS phase is
documented in [aplhago-mcts.md](aplhago-mcts.md). It ships disabled and does
not change the current PPO training loop.

The profile now uses screening benchmarks during training and runs the full
promotion suite only for promising candidates. Screens are frequent enough to
catch regressions without paying for the full suite at every checkpoint.

## Default training budget

The important defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `PPO_ROLLOUT_STEPS` | 12,288 | Decisions collected before one PPO update |
| PPO learning rate | 0.00005 | Conservative post-champion update size |
| PPO epochs / minibatch | 2 / 128 | Limits repeated fitting of one rollout; 512 exhausts the 6 GB GPU |
| Benchmark interval | 50,000 decisions | Cadence for the light screen |
| Screen seeds | 8 | 32 games per baseline after four seat rotations |
| History snapshot interval | 8 PPO updates | Cadence for adding a frozen past policy |
| Live self-play games | 40% | Four seats use the current policy |
| Champion games | 20% | One live seat plays three frozen champion seats |
| History games | 20% | One live seat plays three distinct recent snapshots |
| Teacher games | 20% | One live seat plays three heuristic-teacher seats |

In the initial run, one complete self-play game produced approximately 406
decisions. At that rate, 50,000 decisions gives approximately 123 self-play
games between screens. Each screen uses 32 teacher and 32 champion games:

```text
self-play : teacher screen : champion screen
     123  :       32       :        32
     3.8  :        1       :         1
```

The exact ratio changes with game length. A useful formula is:

```text
self-play games = benchmark interval / recent decisions per self-play game
screen games per baseline = number of screen seeds * 4 seats
```

## Evaluation and promotion flow

At every 50,000-decision interval, the runner saves a candidate and performs
the following steps:

1. Play 32 games against the heuristic teacher using eight dedicated screen
   seeds and all four candidate seat positions. The candidate samples actions
   the same way as in self-play; the teacher remains a deterministic heuristic.
2. Play 32 games against the current champion on the same screen seeds. Both
   candidate and champion sample from their exact PPO behavior policies, making
   this a symmetric comparison of the distributions PPO actually optimized.
3. Continue only if at least 90% of each screen completed, neither screen had a
   policy rejection, teacher first-place rate is at least 25%, and champion
   pairwise score is at least 0.50.
4. Run the full 120-game teacher benchmark on 30 separate promotion seeds.
5. If and only if the full teacher gate passes and the candidate is not worse
   against the teacher than the current champion was, run the full 120-game
   champion benchmark.
6. Promote only if both full gates pass.

The full teacher gate requires a 95% Wilson lower bound on first-place rate
above `ALPHAGO_TEACHER_MIN_WILSON_LOWER` (default 0.15), at least 99% game
completion, and zero policy rejections. It also
requires the candidate's teacher pairwise score to be at least the champion's
teacher pairwise score minus `PROMOTION_TEACHER_PAIRWISE_MARGIN` (default 0).
Both reports use the same promotion seeds and seat rotations, so the comparison
is paired. The result is stored as `teacher_relative` in each state report, and
the champion's teacher report is stored as `champion_teacher_report` in
`selfplay_state.json`. A saved baseline from a different action-selection mode
is discarded and rebuilt before this comparison. Without this check, a
candidate that only learned to exploit the champion could be promoted while
regressing against the teacher;
the 1,700,750 promotion did exactly that (teacher pairwise 0.681 versus 0.708).
The full champion gate requires pairwise score of at least 0.50, at least 99%
completion, and zero policy rejections.

After every benchmark the log prints a rolling mean of the last three screens.
A single 32-game screen has a 95% interval of roughly ±0.15, so use the rolling
line, not one screen, to judge trends.

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

## Trusted opponents versus diagnostic history

The champion, teacher, and historical snapshots have separate responsibilities:

- `champion.pth` changes only after both strict promotion gates pass.
- Every eight successful PPO updates, the live learner is archived as a
  snapshot. Snapshots are never used for promotion decisions, but the eight
  most recent snapshots and past champions are training sparring partners.
- Forty percent of games remain four-seat live self-play. Twenty percent place
  one live learner against three champion seats, twenty percent against three
  distinct recent snapshots, and twenty percent against three heuristic
  teachers. History games fall back to champion games until three snapshots
  exist. Set `SELFPLAY_HISTORY_FRACTION=0` to disable history opponents.
- Frozen seats never write PPO rollouts. Seat placement, matchup choice, and
  snapshot choice are deterministic from the game seed.

A single frozen champion opponent invites the learner to exploit that one
policy instead of getting stronger in general; the history pool spreads that
pressure across several opponents. Each snapshot loads one frozen network
shared by one seat per concurrent game, and the pool refreshes when a snapshot
is archived. Promotion refreshes the trusted champion pool; periodic learner
snapshots cannot silently replace it.

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
$env:ALPHAGO_SEARCH_ROOT_PROMPTS="or,card,space"
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
`ALPHAGO_SEARCH_PUCT_C`, `ALPHAGO_SEARCH_SELECTION`, `ALPHAGO_SEARCH_SEED`,
`ALPHAGO_SEARCH_ADAPTIVE_SIMULATIONS`,
`ALPHAGO_SEARCH_SIMULATIONS_TWO_ACTIONS`,
`ALPHAGO_SEARCH_SIMULATIONS_FOUR_ACTIONS`, `ALPHAGO_SEARCH_LEAF_BATCH`, and
`ALPHAGO_SEARCH_EARLY_STOP`. Self-play collection is additionally controlled
by `ALPHAGO_SEARCH_SELFPLAY_FRACTION`, `ALPHAGO_SEARCH_REPLAY_DIR`,
`ALPHAGO_SEARCH_REPLAY_MAX_SHARDS`, and
`ALPHAGO_SEARCH_REPLAY_MAX_INVALID_RATE`.
`ALPHAGO_SEARCH_ROOT_PROMPTS` defaults to `or,card,space`: top-level action
menus, Research card purchasing, and Action-phase tile placement. Card and
space continuations are rebuilt by replaying accepted inputs from the preceding
stable action root; drafting and initial-card selection remain unsupported.
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

### What the seed 920003 trace says

The trace in `rl-alphago/metrics/mcts_trace_seed920003_seat0.json` is useful as
a performance diagnostic, but it is not a strength benchmark: it covers one
seat in one game, the game did not complete, and the final ranks/VP are only
failure placeholders. Its aggregate results are:

| Metric | Observed | Interpretation |
| --- | ---: | --- |
| Recorded decisions | 37 | 34 searched, one ineligible, two timed out |
| Mean search time | 19.1075 s | Approximately 11.8 minutes were spent in search |
| Mean replay-batch time | 0.23 s | Individual server batches are not unusually slow |
| Selected simulations | 1,088 | Every successful root requested all 32 simulations |
| Invalid simulations | 177 (16.27%) | Above the MCTS plan's 10% health gate |
| Killed tree edges | 10 | Invalid paths are concentrated, not uniformly random |
| Reported applied inputs/s | 344.61 | Above the 100/s server gate, but includes replayed path prefixes |
| Search changed the policy top choice | 21/34 (61.8%) | Search is influential; reducing candidate coverage is risky |

The invalid simulations are strongly concentrated in small roots. The two
one-action roots consumed 64 simulations and rejected 51 of them (79.7%). The
six two-action roots consumed another 192 simulations and rejected 72 (37.5%).
Together, roots with at most two legal actions caused 123 of all 177 invalid
simulations (69.5%). By prompt type, `or` roots rejected 21.3%, `space` roots
10.2%, and Research `card` roots only 3.1%.

The main latency problem is replay amplification. `search/rollout.py` grows a
branch one input at a time, while every replay request sends its complete path.
`SearchSimulationService.ts` then restores the immutable root, determinizes it,
and reapplies the full path. A path of length `L` therefore applies roughly
`1 + 2 + ... + L` inputs and requires serial Python/HTTP/server rounds. The
current PUCT loop commonly runs one batch for first-visit coverage and another
for the remaining budget, each with as many as 40 replay rounds. Raising the
45-second decision timeout would hide this cost rather than fix it.

The following speed work is now implemented:

1. Per-decision telemetry reports replay rounds, replayed inputs, new inputs,
   path-length distribution, Python inference time, HTTP/server time, timeout
   reason, invalid-reason counts, exact-evaluation cache hits, and payload size.
   Both `applied_inputs_per_sec` and `new_transitions_per_sec` are retained; the
   former counts logical paths while the latter counts newly executed inputs.
2. Search bypasses roots with one legal action and uses an adaptive simulation
   budget for small roots (for example 8 simulations for two actions, 16 for
   three or four, and 32 above that). Do not
   reduce `top_k` first: this trace changed the policy's preferred move in 61.8%
   of searched roots, and actions outside `top_k` already receive no visits.
3. The v1 replay request remains backward compatible, but the server now keeps
   each branch's determinized game in the session. Repeated requests reuse the
   accepted prefix and execute only appended inputs. A divergent prefix falls
   back to an immutable-root rebuild. Session close/TTL releases all cached
   games, and simulation still has zero database/cache side effects.
4. One HTTP client and keep-alive connection is reused for all searched
   decisions in a game. Exact evaluations are deduplicated with a bounded cache
   keyed by observation, player, turn count, and recurrent state. Cross-game
   GPU batching remains a future optimization; within-decision leaf evaluation
   is already batched.
5. `leaf_batch` is now a real PUCT control, with early stopping when the visit
   leader cannot be overtaken. Smaller leaf batches provide fresher Q feedback
   but add serial rounds; larger batches favor throughput. Report the trade-off
   rather than assuming one setting is best.
6. Evaluation defaults to zero root Dirichlet noise and argmax selection.
   Search self-play has separate noise and temperature controls, and PUCT now
   supports visit-temperature selection instead of always taking the maximum.

Before search is allowed to generate training games, require completed
fixed-seed traces, no decision timeouts, less than 10% invalid simulations
(target less than 5%), and a measured end-to-end speedup. Also run paired
search-on/search-off games using the same checkpoint, seats, and seeds: changing
the action frequently is not evidence that the change is stronger.

## Upgrade path: search-generated self-play

Turning search on inside the existing PPO runner is not sufficient. PPO is
configured as strict on-policy training, while an MCTS action is sampled from
the search visit distribution rather than from the stored policy distribution.
Mixing those actions directly into the PPO buffer gives the loss the wrong
behavior-policy probability. Keep `exclude_from_rollout=true` until one of the
following explicit training paths exists.

The implemented first step is search distillation. A configurable fraction of
self-play games can run all four live seats with search and store supervised
records under `rl-alphago/search-replay/`. Those games are excluded in their
entirety from PPO, including decisions where search falls back to the policy.
Each committed record contains:

- the searching player's information-set observation and recurrent input;
- all legal action descriptors and a normalized visit target
  `pi(a) = N(a) / sum_b N(b)` (not only the chosen action);
- the root value/Q diagnostics, determinization count, invalid rate, and search
  configuration for auditability; and
- the final terminal reward `z` from that player's perspective once the game
  completes. Incomplete games must not produce value targets.

Only completed episodes with no timeout/error fallback and an aggregate invalid
rate at or below `ALPHAGO_SEARCH_REPLAY_MAX_INVALID_RATE` are committed. Shards
are atomic, schema-versioned, bounded by a configurable window, and include the
policy version, seed-derived search configuration, seat, and terminal result.

Start a conservative 5% collection pilot with:

```powershell
$env:ALPHAGO_RL_SEARCH_ENABLED="1"
$env:ALPHAGO_RL_SEARCH_DETERMINIZATION_ENABLED="1"
$env:ALPHAGO_SEARCH_MODE="puct"
$env:ALPHAGO_SEARCH_SELFPLAY_FRACTION="0.05"
$env:ALPHAGO_SEARCH_REPLAY_MAX_INVALID_RATE="0.10"
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate
```

Search games use `ALPHAGO_SEARCH_SELFPLAY_SELECTION=temperature`, root noise
weight `0.25`, and visit temperature `1.0` through generation four by default;
later generations switch to argmax. Ordinary PPO games and the default
`ALPHAGO_SEARCH_SELFPLAY_FRACTION=0` behavior are unchanged.

Distill completed shards into a separate candidate rather than mutating the
live PPO learner or its on-policy buffer:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml run --rm --no-deps `
  rl-coordinator python -m training.search_distill `
  --checkpoint /app/alphago/checkpoints/latest_learner.pth `
  --replay-dir /app/alphago/search-replay `
  --output /app/alphago/checkpoints/search_distilled_candidate.pth `
  --epochs 1 --batch-size 64 --learning-rate 1e-5
```

The trainer minimizes policy cross-entropy against `pi` plus a weighted Huber
value loss against `z`, writes a report beside the candidate, and increments
its policy version. Benchmark this candidate against both its source checkpoint
and the champion before promotion. It is never loaded automatically into the
running PPO learner.

The full AlphaGo-Zero-style path is a separate trainer, not a PPO option:

```text
current network -> information-set MCTS actors -> (observation, pi, z) replay
        ^                                                   |
        |                                                   v
   promoted model <- paired gates <- candidate <- policy + value training
```

For that trainer:

1. Let every live seat use the same frozen actor checkpoint for the duration of
   a game. Sample from visit counts early in the game and switch to argmax later.
2. Train the policy head from the full legal-action visit distribution and the
   value head from terminal outcomes, retaining the existing auxiliary losses
   only when their labels remain valid.
3. Use a bounded replay window spanning several recent policy versions. Keep
   policy version, search configuration, game seed, seat, and schema version in
   every shard so stale or incompatible data can be quarantined.
4. Freeze a candidate for evaluation and use the existing teacher/champion
   screens and full promotion gates. Never let evaluation games enter replay.
5. Preserve the historical opponent pool to reduce four-player cycling, but
   identify which seats used which frozen checkpoint in every training game.

Terraforming Mars is an imperfect-information game, so these actors are closer
to information-set MCTS than to the original perfect-information AlphaGo Zero
loop. Every simulation must determinize from only the searching player's root
information, the network must receive only that player's normal observation,
and visit targets must aggregate multiple determinizations. Do not cache or
reuse a node across states unless its key includes the complete information set
available to that player. Candidate truncation also needs attention before full
self-play: a permanently excluded low-prior legal move can never gain visits or
become a positive policy target.

The current safe default remains `ALPHAGO_SEARCH_SELFPLAY_FRACTION=0`. Increasing
the fraction is an explicit data-collection choice and should follow completed
fixed-seed traces plus paired search-on/search-off strength measurements.

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

## Recover a regressed learner

Recovery is an explicit one-shot operation. It backs up the current learner and
state, quarantines queued on-policy rollouts, preserves cumulative counters, and
assigns a fresh policy version:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml run --rm --no-deps `
  rl-coordinator python -m training.v2_recover `
  --root /app/alphago `
  --checkpoint /app/alphago/checkpoints/candidate_000800867.pth
```

During subsequent training, a champion screen below pairwise score `0.35`, or
two consecutive screens below `0.50`, triggers a rollback check. The 32-game
screen uses only eight distinct deals, so the runner first re-tests the
candidate on the full 120-game, 30-deal champion benchmark (report label
`rollback_confirm`, about 20 minutes). The learner rolls back only if that
full pairwise score is below `PPO_ROLLBACK_CONFIRM_PAIRWISE` (default `0.45`)
or the run is incomplete or has rejections; otherwise the failure counter
resets and training continues. Set `PPO_ROLLBACK_CONFIRM_FULL=0` to restore
immediate rollback. The confirmation reuses the promotion seeds, so it adds
mild selection pressure on them; it only decides rollbacks, never promotions. Failed-policy rollout shards are quarantined rather than deleted.
Per-update PPO diagnostics are appended to
`rl-alphago/metrics/ppo_updates.jsonl`.

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
$env:ALPHAGO_BENCHMARK_INTERVAL=50000
$env:ALPHAGO_PPO_ROLLOUT_STEPS=12288
$env:ALPHAGO_HISTORY_SNAPSHOT_INTERVAL_UPDATES=8
$env:ALPHAGO_SCREEN_MIN_COMPLETION_RATIO=0.90
$env:ALPHAGO_SCREEN_TEACHER_MIN_FIRST_PLACE_RATE=0.25
$env:ALPHAGO_SCREEN_CHAMPION_MIN_PAIRWISE_SCORE=0.50
$env:ALPHAGO_TEACHER_MIN_WILSON_LOWER=0.15
$env:ALPHAGO_SELFPLAY_LIVE_FRACTION=0.40
$env:ALPHAGO_SELFPLAY_CHAMPION_FRACTION=0.20
$env:ALPHAGO_SELFPLAY_HISTORY_FRACTION=0.20
$env:ALPHAGO_SELFPLAY_TEACHER_FRACTION=0.20
$env:ALPHAGO_PROMOTION_TEACHER_PAIRWISE_MARGIN=0.0
$env:ALPHAGO_PPO_EPOCHS=2

docker compose -f docker-compose.rl_hard.yml -f docker-compose.alphago.yml up -d --force-recreate rl-coordinator
```

Recommended ranges:

| Control | Recommended range | Trade-off |
| --- | ---: | --- |
| Benchmark interval | 100k-200k decisions | Larger values favor self-play throughput |
| PPO rollout | 12,288-16,384 decisions | Larger values improve batch diversity but update less often |
| History snapshot cadence | 8-16 updates | Smaller values retain denser diagnostic history |
| Champion-game fraction | 0.15-0.30 | Larger values resist forgetting but encourage exploiting one opponent |
| History-game fraction | 0.15-0.30 | Larger values diversify opponents; snapshots may be weaker than the champion |
| PPO epochs | 2-4 | Measured approx KL is 0.001-0.002 against a 0.01 target; change only as an isolated A/B |
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
