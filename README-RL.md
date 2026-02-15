# Terraforming Mars RL Environment (RL-First)

This environment now uses an RL-first training loop:
- PPO is the primary optimizer.
- Tournaments are used to collect rollout data and evaluate fitness.
- Evolution is reduced to periodic immigrant injection for diversity.
- League pools are tracked each generation for self-play management.

## Architecture

1. `tournament_manager.py` runs 4-player games and gathers outcomes.
2. `models/agent.py` collects legal-masked rollout transitions during play.
3. `coordinator.py` runs PPO optimization over buffered transitions after rollout collection.
4. `training/fitness.py` computes selection fitness and promotion gating.
5. `agent_evolution.py` keeps the trained population and injects immigrants periodically.
6. `training/league.py` updates `main`, `historical`, and `exploiter` pools and influences tournament seating order.

## Clean File Tree

```text
rl-environment/
  coordinator.py                 # Main orchestration loop + API startup
  tournament_manager.py          # Tournament and game execution
  agent_evolution.py             # RL-first evolution (immigrant injection)
  game_interface.py              # TM server/game API integration
  scoring.py                     # Selection and reward helpers
  logging_setup.py               # Centralized logging

  api/
    server.py                    # Monitoring and control API
    templates/dashboard.html     # Dashboard UI

  metrics/
    tracker.py                   # Generation/tournament metrics persistence

  models/
    agent.py                     # Agent behavior + rollout buffer
    ppo.py                       # PPO rollout step + optimizer implementation
    action_decoder.py            # Legal action decoding/encoding
    state_encoder.py             # API state -> model features

  training/
    fitness.py                   # Fitness calculation + behavior gating
    ppo_cycle.py                 # Population-level PPO optimization cycle
    league.py                    # League pool tracking
```

## RL-First Training Loop

For each generation:

1. Run tournaments and collect game outcomes.
2. Agents queue rollout transitions:
   - `state`
   - `action`
   - `logp_old`
   - `value_old`
   - `reward`
   - `done`
   - `legal_actions`
3. Coordinator runs PPO updates from rollout buffers.
4. Compute raw selection fitness from tournament outcomes.
5. Apply promotion gates and penalties.
6. Update league pools.
7. Save checkpoints and evolve using immigrant injection only.

## PPO Environment Variables

```bash
# Enable RL-first PPO cycle
PPO_ENABLE=1
PPO_ROLLOUT_STEPS=8192
PPO_BUFFER_MAX_STEPS=65536

# PPO core
PPO_EPOCHS=4
PPO_MINIBATCH_SIZE=1024
PPO_CLIP_EPS=0.2
PPO_VALUE_CLIP_EPS=0.2
PPO_GAMMA=0.99
PPO_GAE_LAMBDA=0.95
PPO_ENTROPY_COEF=0.01
PPO_VALUE_COEF=0.5
PPO_MAX_GRAD_NORM=1.0
PPO_TARGET_KL=0.02
PPO_LEARNING_RATE=0.0003
PPO_LR_ADAPT_UP=1.03
PPO_LR_ADAPT_DOWN=0.85
PPO_LR_MIN=0.00005
PPO_LR_MAX=0.0008

# Trajectory validation
STATE_SCHEMA_VERSION=v1
```

## League Environment Variables

```bash
LEAGUE_ENABLE=1
LEAGUE_HISTORICAL_RATIO=0.4
LEAGUE_EXPLOITER_RATIO=0.2
LEAGUE_SNAPSHOT_INTERVAL=5
```

## RL-First Evolution Variables

```bash
RL_FIRST_ENABLE=1
EVOLUTION_IMMIGRANT_RATIO=0.10
EVOLUTION_IMMIGRANT_INTERVAL=3
EVOLUTION_IMMIGRANT_MUTATION_RATE=0.30
```

## Behavior Gates

```bash
PROMOTION_GATE_ENABLED=1
GATE_MIN_CARD_PLAYS_PER_GAME=0.60
GATE_MAX_STANDARD_PROJECT_RATIO=0.58
GATE_MIN_STEEL_SPENT_PER_GAME=0.25
GATE_MIN_TITANIUM_SPENT_PER_GAME=0.12
GATE_MAX_PAYMENT_REJECT_COUNT=0
GATE_PENALTY_POINTS=8.0
GATE_GLOBAL_PAYMENT_PENALTY_POINTS=6.0
```

Steel/titanium gate semantics:
- `GATE_MIN_STEEL_SPENT_PER_GAME` and `GATE_MIN_TITANIUM_SPENT_PER_GAME` are lazy-usage floors.
- An agent fails resource gating only when both steel and titanium are below floor in the same generation window.
- Spending above floor does not add extra gate benefit.

## Core Runtime Variables

```bash
GAME_SERVERS=tm-server-1:8080,tm-server-2:8080,tm-server-3:8080
POPULATION_SIZE=32
TOURNAMENT_SIZE=8
GENERATIONS=1000
GAMES_PER_EVAL=1
POSTGRES_URL=postgresql://postgres:password@postgres:5432/rl_metrics
REDIS_URL=redis://redis:6379
```

## Metrics Added for RL-First

The generation payload now includes PPO and league metrics, including:
- `ppo/policy_loss`
- `ppo/value_loss`
- `ppo/entropy`
- `ppo/approx_kl`
- `ppo/clip_fraction`
- `ppo/explained_variance`
- `ppo/grad_norm`
- `ppo/learning_rate`
- `ppo/early_stop_kl_ratio`
- `rollout/steps_collected`
- `rollout/schema_filtered`
- `league/matchmaking_ordering_applied`
- `league/main_pool_size`
- `league/historical_pool_size`
- `league/exploiter_pool_size`

## Quick Start

```bash
docker compose -f docker-compose.rl.yml up --build
```

## Test Best Agent

Play manually against the best saved checkpoint across all generations:

```bash
python rl-environment/play_vs_generation.py --best --bots 3 --servers localhost:8081,localhost:8082,localhost:8083
```

Or via API:

```bash
curl -X POST http://localhost:5000/play/human-vs-best -H "Content-Type: application/json" -d "{\"human_name\":\"You\",\"bot_count\":3}"
```

## Standalone Live Bot

Attach the best saved checkpoint to an existing live player slot.

Heroku example:

```bash
python rl-environment/standalone_bot.py --player-url "https://terraforming-mars.herokuapp.com/player?id=<PLAYER_ID>" --min-action-delay-ms 1000
```

Or pass base URL + player ID separately:

```bash
python rl-environment/standalone_bot.py --base-url "https://terraforming-mars.herokuapp.com" --player-id "<PLAYER_ID>" --min-action-delay-ms 1000
```

Notes:
- `--min-action-delay-ms` is clamped to at least `1000` to avoid overloading live servers.
- If `--checkpoint` is not provided, the script auto-loads the highest-fitness saved checkpoint from `rl-models`.

Tkinter launcher (same options via GUI):

```bash
python rl-environment/standalone_bot_tk.py
```

Monitor:
- `http://localhost:5000/dashboard`
- `http://localhost:5000/stats`

## Notes

- `PPO_RECURRENT` and `PPO_SEQUENCE_LENGTH` are not active yet in this code path.
- Existing API endpoints remain unchanged.
- Legacy actor-critic update path still exists as fallback when `PPO_ENABLE=0`.


# next implementations
What to implement next (priority order):

Stabilize PPO (highest priority)
Add adaptive LR/pressure control from KL.
Decouple PPO LR from random agent init LR (use fixed env-driven PPO LR).
Tighten KL early-stop threshold and report explicit early_stop_ratio.
Make gate actually selective
Raise thresholds to realistic values from your observed regime (not near-zero floors), or use percentile-based dynamic gates.
Goal: gate pass rate not always 1.0.
Add fixed benchmark evaluation
Every N gens, evaluate vs frozen snapshots/baseline and store benchmark metrics.
Prevents self-play-only drift from looking like progress.
Save more than 1 agent per generation
