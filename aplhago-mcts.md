# AlphaGo MCTS Phase 1: TypeScript Search Simulator

## Summary

Build and validate the TypeScript simulation foundation before implementing Python MCTS.

The primary reconstruction mechanism will be the current `SerializedGame`, not the initial game seed alone. The snapshot already contains ordered decks, player state, `seed`, and the mutable RNG `currentSeed`. Rebuilding from the initial seed requires replaying every accepted input in exact order, so seed-plus-action replay will serve as an independent reliability audit.

Version 1 is limited to the current Stage‑1 ruleset: four-player Tharsis, Corporate Era, fast mode, no drafting, Prelude, or expansions.

## Implementation status

The Phase 1 simulator and HTTP contract are implemented in the TypeScript
server. Search consumers remain disabled by default. The implementation uses
the serialized root plus `currentSeed`, rebuilds every branch independently,
and never stores serialized child nodes.

- Set `RL_SEARCH_ENABLED=1` only for a token-protected search consumer.
- Set `RL_SEARCH_SHADOW_VALIDATE=1` first to validate accepted live inputs
  without exposing the replay endpoints.
- Set `RL_SEARCH_TRACE_VALIDATE=1` to independently recreate newly created
  Stage-1 games from their original seed and replay the accepted-input journal.
- Set `RL_SEARCH_DETERMINIZATION_ENABLED=1` only after the 10,000-input exact
  shadow gate passes.
- Send `RL_CONTROL_TOKEN` as `x-rl-control-token` on every search request.
- Exact mode is an audit facility. Benchmark and gameplay search must use
  determinized mode.

Prometheus exposes session, batch, branch-status, rejected-input, divergence,
expiration, latency, and active-session metrics with the `rl_search_` prefix.
The 10,000-input zero-divergence soak and the hardware throughput gate are
rollout gates; they must be recorded here before enabling a Python consumer.
The Python consumer now exists (`rl-environment/search/`) and is also disabled
by default; enabling it for benchmarks requires both sides:
`ALPHAGO_SEARCH_ENABLED=1` on the coordinator and
`ALPHAGO_RL_SEARCH_ENABLED=1` plus
`ALPHAGO_RL_SEARCH_DETERMINIZATION_ENABLED=1` on the game servers.

| Reliability gate | Current result |
| --- | --- |
| Focused deterministic replay tests | Automated in the TypeScript test suite |
| Original-seed journal replay | 3/3-game live smoke passed with 1,235 matched inputs and zero divergence; 100-game/20-completion campaign pending |
| Exact live shadow inputs | 715 eligible live inputs matched with zero divergence in the smoke run; target is 10,000 |
| 64-branch throughput | Not yet measured on RL server hardware; target is 100 inputs/s |
| Retained sessions after TTL | Automated cleanup test |

## Simulator and API

- Add a simulation-safe deserialization mode to the game engine:
  - Restore `seed` and `currentSeed` exactly.
  - Preserve player IDs and all ordered card zones.
  - Disable database saves, cache registration, completion marking, timers, human listeners, and production metrics.
  - Never mutate the source snapshot or live game.
- Only start search at reconstructible action roots:
  - Serialize the live game.
  - Deserialize it in simulation mode.
  - Compare normalized private observations and gameplay-state digests.
  - Reject roots whose current prompt cannot be reconstructed, including mid-payment, placement, research, or deferred-action prompts.
- Replay every branch from the immutable root snapshot. Do not serialize child nodes because prompt callbacks are not serializable; branch nodes store action paths.
- Process actions directly through `Player.process()`. Automatic callbacks advance normally, while opponent and continuation choices are supplied by Python as later path steps.
- Add an in-process `SearchSimulationService` with a token-protected HTTP adapter:
  - `POST /api/rl/search/start`
  - `POST /api/rl/search/replay`
  - `POST /api/rl/search/close`
- Reuse `RL_CONTROL_TOKEN` through `x-rl-control-token`; keep the service disabled unless `RL_SEARCH_ENABLED=1`.
- Keep at most 32 in-memory sessions with a 60-second idle TTL. Limit replay batches to 64 branches and 32 accepted inputs per branch.

Public contracts:

```text
SearchStartRequest
  playerId

SearchStartResponse
  schemaVersion
  sessionId
  rootDigest
  rootPlayerId
  observation

SearchReplayRequest
  sessionId
  branches[]
    branchId
    mode: exact | determinized
    determinizationSeed?
    steps[]
      playerId
      input: InputResponse

SearchBranchResult
  branchId
  status: next_prompt | terminal | boundary | rejected
  appliedSteps
  stateDigest
  nextPrompt?: { playerId, observation }
  terminal?: { players: [{ playerId, rank, vp }] }
  error?: { code, stepIndex, message }
```

## Python Search Consumer (Phase 2, shipped disabled)

`rl-environment/search/` implements the first consumer of the contract above:

- `simulator_client.py` — `SearchClient` for `POST /api/rl/search/start|replay|close`
  with `x-rl-control-token`. It validates `tfm.rl-search.v1`, applies the same
  inbound `tfm_schema` observation normalization and outbound payment adaptation
  as the live game client, and maps structured errors (`session_capacity`,
  `unsupported_root`, `determinization_disabled`, …) to policy fallback.
- `evaluator.py` — batched `no_grad` priors/value/recurrent forwards over
  arbitrary `PlayerViewModel` states. Branch-local recurrent maps are cloned
  from live memory; live agent memory is never mutated by search.
- `rollout.py` — advances branch batches (≤64 per request, ≤32 steps per
  branch) with one batched forward per round for all pending prompts. Opponents
  and forced prompts are policy-sampled freshly per determinization; the root
  player's strategic prompts pause the branch for the tree handler, which
  descends one own macro-action, burns a `value_horizon` policy-sampled own
  turn, or stops. A branch stops at a leaf (valued by the reused strategic-prompt
  evaluation), at `terminal` (scored with `calculate_v2_terminal_reward`), or is
  dropped on `boundary`/`rejected`.
- `lookahead.py` — Phase A flat search: top-K macro-actions × D
  determinizations rolled out to the root's next decision, best mean value
  wins (default `top_k=8`, `determinizations=8`).
- `node.py` / `mcts.py` — Phase B PUCT (`Q + c·P·√N/(1+n)`) over root-turn
  macro-actions up to `max_root_turns_depth` (default 2). Nodes store only the
  searching player's own macro-actions (never recorded opponent continuations,
  which are illegal under a fresh determinization); each simulation determinizes
  and samples continuations freshly, depth-2 descent re-matches child edges to
  the current leaf by canonical action payload, falling back to a unique label.
  Rollout failures cost an edge a consecutive failure without touching `Q`;
  only `edge_kill_failures` consecutive failures kill it. Simulations run as
  batched rounds (fair coverage until every live edge has a value, then batched
  PUCT spending, ≤64 branches per replay request) with Dirichlet root noise and
  FPU fallback for unvisited edges.
- `search_agent.py` — `SearchPolicy.decide` gates on strategic top-level
  prompts, runs one search per decision under a timeout, and returns a
  searched action with visit/prior metadata. Any failure degrades to plain
  policy sampling.

`RLAgent._make_move` consults an optional `search_policy`; searched actions are
tagged `action_source="mcts-search"` with `sampled_from_policy=False`, so they
never enter the PPO rollout buffer. Training self-play does not wire search
in; `training/v2_benchmark.py` attaches it to the frozen candidate when
`ALPHAGO_SEARCH_ENABLED=1`, and benchmark reports gain a `search` block with
mode, searched/fallback counts, mean replay batch latency, and applied-inputs
throughput (which feeds the 100 inputs/s hardware gate).

Environment controls: `ALPHAGO_SEARCH_ENABLED`, `ALPHAGO_SEARCH_MODE`
(`lookahead|puct`), `ALPHAGO_SEARCH_TOP_K`, `ALPHAGO_SEARCH_DETERMINIZATIONS`,
`ALPHAGO_SEARCH_SIMULATIONS`, `ALPHAGO_SEARCH_DEPTH`, `ALPHAGO_SEARCH_PUCT_C`,
`ALPHAGO_SEARCH_SELECTION`, `ALPHAGO_SEARCH_SEED`,
`ALPHAGO_SEARCH_ROOT_NOISE_ALPHA` / `ALPHAGO_SEARCH_ROOT_NOISE_WEIGHT` (root
Dirichlet noise, default 0.05/0.25), `ALPHAGO_SEARCH_EDGE_KILL_FAILURES`
(consecutive rollout failures before an edge is pruned, default 3),
`ALPHAGO_SEARCH_VALUE_HORIZON` (extra policy-sampled own turns after the tree
is exhausted, default 0), and `ALPHAGO_SEARCH_ROOT_PROMPTS` (comma list
widening searchable roots beyond `or`, e.g. `or,card`; rollout leaves always
stay `or`). The client clamps `top_k × determinizations` to the 64-branch
batch limit.

Benchmark and trace output now reports per-decision and aggregated rollout
health: `simulations_selected`, `valid_rollouts`, `invalid_rollouts`,
`invalid_rate`, and `killed_edges`, plus `unavailable_by_prompt` attributions
(e.g. `root_not_round_trippable:card`) so fallback causes are visible per
prompt type. The acceptance signal for search quality is a per-decision
invalid-rollout rate under 10 percent and a meaningful Q spread across root
edges; validate with a paired same-checkpoint A/B benchmark
(`ALPHAGO_SEARCH_ENABLED=0/1`, ≥100 seeds) before trusting VP deltas from any
single game.

## Determinism and Hidden Information

- Define a stable SHA-256 gameplay digest containing RNG state, decks, hands, players, board, phase, active player, generation, global parameters, milestones, awards, and passed/done players.
- Exclude non-gameplay fields such as game name, timestamps, logs, spectator ID, save counters, and runtime IDs.
- Implement two explicit modes:
  - `exact`: preserve the real hidden state for determinism tests only.
  - `determinized`: create a fair information set for later MCTS.
- Add determinization only after exact replay passes:
  - Preserve the root player’s hand and all public state.
  - Pool the project draw pile with opponents’ hidden hands.
  - Shuffle with a dedicated `SeededRandom` derived from `rootDigest + determinizationSeed`.
  - Redeal identical opponent hand counts and preserve draw-pile size.
  - Keep the discard pile fixed in version 1 and stop at a boundary before any discard reshuffle.
  - Fork future simulation RNG from the determinization seed so the real future RNG stream is not leaked.
  - Reject roots containing unsupported hidden startup/draft zones.
- Never return serialized hidden state, deck order, RNG cursor, or opponent hands through the HTTP API.
- Require every determinization to produce the same normalized public/private root observation for the searching player.

## Validation and Acceptance

- Add focused unit tests for:
  - RNG restoration from `seed + currentSeed`.
  - Exact serialize/deserialize digest equality.
  - Same root, path, and seed producing identical states.
  - Sibling branches and the live game remaining independent.
  - Simulation mode causing zero database/cache side effects.
  - Authentication, limits, TTL cleanup, and structured failures.
- Add Stage‑1 trace tests:
  - Capture accepted `InputResponse` values in server order.
  - Recreate games from the original seed and replay the complete journal.
  - Compare gameplay digests at every action root and final ranks/VP.
  - Run at least 100 seeded games, including at least 20 completed games.
- Add shadow validation on the live RL stack:
  - At eligible action roots, clone the current snapshot and compare its observation/digest.
  - Replay the next accepted live input on an exact clone and compare the resulting state.
  - Accumulate at least 10,000 accepted inputs with zero divergence before enabling search consumers.
- Test fair determinization invariants:
  - Same determinization seed is reproducible.
  - Different seeds produce hidden-state diversity.
  - Card multisets and zone counts remain valid.
  - The root player’s observation remains unchanged.
  - Branches stop before unsupported phases or discard reshuffles.
- Add load tests for batches of 64 branches up to eight inputs deep. Acceptance is at least 100 applied branch inputs per second on the existing RL server hardware, no live-state mutations, and no retained-session memory growth after TTL cleanup.
- Expose counters for session starts, replay batches, branches by status, divergence, latency, active sessions, expirations, and rejected inputs.

## Rollout and Follow-up

- Ship disabled by default.
- Enable exact shadow validation first.
- Enable determinization only after the zero-divergence gate.
- Do not use exact mode for benchmark or human gameplay.
- Document the simulator contract and reliability results in a new MCTS plan document linked from `README-ALPHAGO.md`.
- Python tree selection and neural leaf batching now ship in `rl-environment/search/` (disabled by default, evaluation-only). Visit-count training targets, replay-buffer distillation, and any PPO replacement remain deferred to Phase 3; searched actions are excluded from the on-policy buffer. The HTTP contract created here remains the consumption boundary.

## Assumptions

- Recommended defaults were selected because no preference response was provided: simulator-first milestone, exact-then-fair validation, and in-process core plus HTTP adapter.
- Search begins only at stable Stage‑1 action roots.
- Snapshot plus `currentSeed` is authoritative; initial seed plus complete action journal is a verification path, not the production cloning mechanism.
- No checkpoint, rollout, or benchmark schema changes are required in this phase.
