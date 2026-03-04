# Rust RL Bottleneck Refactor - Implementation Log

## Scope
- Rust extension is mandatory for runtime (`rust_tfm_rl` hard requirement).
- Build/install in both RL Python images:
  - `rl-environment/Dockerfile.training`
  - `rl-environment/Dockerfile`
- Node server image (`Dockerfile.rl`) unchanged.

## Implemented Changes

### 1) Rust module API (`rust_tfm_rl`)
- `encode_state(json_payload: str, turn_action_count: int, state_size: int) -> np.ndarray[float32]`
- `estimate_affordability(json_player: str, json_card: str) -> float`
- `can_afford_cards(json_player: str, json_cards: str) -> list[bool]`
- `enumerate_card_selection_combos(json_payload: str, limit: int) -> list[list[int]]`
- `rank_startup_plans(json_payload: str, max_plans: int) -> str`
- `backend_info() -> dict`

### 2) Python integration
- Added `models/rust_backend.py` for centralized import/contract checks.
- `models/agent.py`
  - validates Rust backend at startup via `require_backend_info()`
  - logs backend metadata (module/api/crate version)
- `models/state_encoder.py`
  - strict Rust-backed `encode()` (no zero-vector fallback)
  - updated `encode_state` call to new signature with `state_size`
  - affordability hotspot uses Rust functions (`estimate_affordability`, batched `can_afford_cards`)
- `models/action_decoder.py`
  - `_can_afford_card` routed through Rust batch kernel
  - card-selection mask enumeration routed through Rust combo kernel
  - startup-plan candidate ranking/dedup routed through Rust ranking kernel
  - Python still owns orchestration/payload assembly (`build_response_for_input`, `get_available_actions`)

### 3) Build pipeline
- `rl-environment/Dockerfile.training`
  - ensures `maturin` is installed before build
  - validates extension import after wheel install
- `rl-environment/Dockerfile`
  - installs Rust toolchain (`rustup`)
  - builds/install wheel using `maturin`
  - validates extension import after install

### 4) Verification assets
- Added benchmark script:
  - `rl-environment/tests/benchmark_rust_hotspots.py`
- Added contract/parity tests:
  - `rl-environment/tests/test_rust_backend_contract.py`
  - `rl-environment/tests/test_rust_hotspot_parity.py`

## Benchmark Baseline + Compare Template

Run:

```bash
python rl-environment/tests/benchmark_rust_hotspots.py
```

Baseline captured on 2026-03-04 (host Python run; Rust module not yet installed locally):

| Hotspot | Python Baseline (ms/call) | Rust (ms/call) | Speedup |
|---|---:|---:|---:|
| Affordability batch | 0.176 | _pending_ | _pending_ |
| Card-selection combos | 0.562 | _pending_ | _pending_ |

Update after Rust wheel build:

| Hotspot | Python Baseline (ms/call) | Rust (ms/call) | Speedup |
|---|---:|---:|---:|
| Affordability batch | 0.176 | _pending_ | _pending_ |
| Card-selection combos | 0.562 | _pending_ | _pending_ |

Acceptance target:
- parity tests pass
- `>= 2x` speedup on targeted hotspot kernels

## End-to-End Validation Checklist

1. Build images:
   - `docker compose -f docker-compose.rl_hard.yml build rl-coordinator`
2. Run parity tests in coordinator image:
   - `pytest rl-environment/tests/test_rust_backend_contract.py rl-environment/tests/test_rust_hotspot_parity.py`
3. Run microbench:
   - `python rl-environment/tests/benchmark_rust_hotspots.py`
4. Run training benchmark:
   - `python rl-environment/benchmark_training.py`
5. Confirm no regression in:
   - invalid action reject rate
   - action decoder legality behavior
   - state vector size/token tail integrity

## Pipeline Bottleneck Mitigation (Cloud Compose)

### New Runtime Controls
- `RL_GLOBAL_GAME_CAP_PER_COORD`
  - hard cap for `GLOBAL_GAME_CONCURRENCY` generated per coordinator
  - default: `20` (`saturate`), `24` (`balanced`)
- `RL_TOURNAMENT_CAP_PER_COORD`
  - hard cap for `TOURNAMENT_CONCURRENCY` generated per coordinator
  - default: same as global cap
- `AGENT_IDLE_POLL_INTERVAL_SEC`
  - polling interval when `waitingFor` is empty
  - default: `0.12`
- `AGENT_ACTIVE_POLL_INTERVAL_SEC`
  - polling interval when `waitingFor` is present
  - default: fallback to `AGENT_POLL_INTERVAL_SEC`
- `AGENT_TIMING_LOG_EVERY_N_DECISIONS`
  - decision interval for timing summary logs
  - default: `200`

### Coordinator Cap Logic
- `raw_subset_global = subset_count * games_per_server`
- `subset_global = min(raw_subset_global, global_cap_per_coord)`
- `subset_tournament = min(subset_global, tournament_cap_per_coord)`
- `MAX_ACTIVE_GAMES_PER_SERVER = min(games_per_server, ceil(subset_global / subset_count))`

Default target for 54 servers / 6 coordinators:
- per coordinator `GLOBAL_GAME_CONCURRENCY=20`
- per coordinator `TOURNAMENT_CONCURRENCY=20`
- per coordinator `MAX_ACTIVE_GAMES_PER_SERVER=3`

### Instrumentation Added
- Per-agent timing accumulators for:
  - state encoding
  - action-availability construction
  - network forward/sampling
  - action decode
  - `send_player_input`
  - `get_player_state`
- Cluster-level transport timing stats in `GameServerCluster` (`send_player_input`, `get_player_state`).

### Acceptance Metrics (Canary)
- Timeout rate `< 1%` of games.
- No sustained coordinator CPU saturation spikes comparable to prior `~900–1400%`.
- No material increase in action rejection rate.
- Generation completion remains above configured gate thresholds.
