# TFM RL V3: warm-started observability curriculum

V3 continues from the stable V2 policy at `candidate_000500494`; it does not
restart the network from random weights. It resets the optimizer, policy
version, rollouts, and training statistics, then introduces the new observation
features over the first 25,000 V3 decisions.

This is the safer branch because the missing behavior is primarily an
observation/representation problem. Starting from scratch would discard the
legal-action, payment, card-play, and board-placement competence already learned
by V2 without making the new signals easier to learn.

## What changed

- Opponent VP is reconstructed from public information. The game still uses
  `showOtherPlayersVP: false`; V3 never reads the hidden opponent total.
- Each opponent token now includes its own public tableau tag profile, card
  resource stacks, board ownership, city/greenery score, estimated VP, and an
  uncertainty value for unresolved dynamic cards.
- Awards and milestones have deterministic identity features. In V2 the
  transformer could not reliably distinguish, for example, Landlord from
  Banker because there are no positional embeddings and the tokens shared one
  generic type.
- Fund-award actions now carry the selected award identity, current standings,
  projected rank points, funding cost, affordability, timing, and early-funding
  risk.
- All additions fit the existing 64-wide token architecture. At
  `V3_FEATURE_SCALE=0`, encoded inputs are exactly V2-compatible.

The public score estimator currently includes TR, claimed milestones, owned
greenery, city adjacency, numeric printed card VP, resource-card VP, and
projected points for funded awards. It reports non-resource dynamic scoring as
uncertainty instead of pretending it is exact.

## 1. Validate the configuration

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v3.yml config
pytest -q rl-environment/tests/test_v3_observability.py
```

V3 writes only to `./rl-v3`. V2 is mounted read-only at `/app/v2`, uses its own
Postgres volume, and remains untouched.

## 2. Create the warm-start checkpoint once

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v3.yml run --rm rl-coordinator `
  python -m training.v3_warm_start `
  --source /app/v2/checkpoints/candidate_000500494.pth `
  --output /app/v3/bootstrap/warm_start.pth
```

This creates `rl-v3/bootstrap/warm_start.pth`, a provenance report, and a V3
manifest. The output checkpoint is marked `tfm-rl-v3`; ordinary V3 runs reject
V2 and legacy checkpoints.

## 3. Run a short adaptation smoke test

Start the game servers and database:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v3.yml up -d `
  tfm-server-1 tfm-server-2 tfm-server-3 tfm-server-4 redis postgres autoheal
```

Then collect 5,000 V3 decisions:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v3.yml run --rm `
  -e V3_ALLOW_RESUME=1 rl-coordinator `
  python -m training.v3_self_play `
  --bootstrap-checkpoint /app/v3/bootstrap/warm_start.pth `
  --root /app/v3 --stage 1 --max-decisions 5000 --benchmark-interval 5000
```

Inspect `rl-v3/metrics/selfplay_progress.json` and the benchmark report before a
long run. In particular, compare award funding frequency, projected award
points at funding time, opponent-estimate error at game end, illegal-action
rate, and win/VP regression against the frozen bootstrap champion.

## 4. Continue the curriculum

If the smoke benchmark has no major regression, continue the same V3 run rather
than making another warm start:

```powershell
docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v3.yml run --rm `
  -e V3_ALLOW_RESUME=1 rl-coordinator `
  python -m training.v3_self_play `
  --bootstrap-checkpoint /app/v3/bootstrap/warm_start.pth `
  --root /app/v3 --stage 1 --max-decisions 150000 --benchmark-interval 25000
```

The feature scale is `V3 decisions / V3_FEATURE_RAMP_DECISIONS`, capped at 1.
The default ramp is 25,000 decisions. Benchmarks always use the full V3 feature
set, which makes them an early warning for feature-induced regressions.

Do not advance difficulty merely because the decision counter is high. Advance
after fixed-seed gates show that the candidate retains bootstrap performance and
uses the new information: fewer clearly bad early awards, more late awards when
ranked first or second, and smaller opponent-score estimation error. If these do
not improve after roughly 50k-100k V3 decisions, inspect the new feature and
action traces before considering a scratch run.
