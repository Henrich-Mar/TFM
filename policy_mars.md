# Planner-Oriented Clean-Sheet Policy for Terraforming Mars

## Summary
Replace the current `flat state vector + card-only attention + fixed 1000-logit head` design with a planner-oriented policy that reasons over typed game objects and scored legal actions.

Primary goal:
- Improve endgame planning, especially plant/heat carry-over, milestone timing, award races, and multi-step city/greenery or Moon board setups.

Core design choice:
- Use a unified typed-token world encoder plus an action-conditioned policy head.
- Retrain from scratch; do not preserve checkpoint compatibility.

## Key Changes
### 1. Replace the flat encoder with a token bundle
Change `StateEncoder.encode(...)` from returning one flat `np.ndarray` to returning a structured bundle with masks:
- `world_tokens`: typed non-action entities
- `hand_tokens`: owned/candidate card tokens
- `action_tokens`: encoded legal actions for the current `waitingFor`
- `token_types` / `attention_masks`
- small `global_scalars` side channel only for a few stable numeric values if needed

Token groups to include:
- `self_state` token: resources, production, TR, MC tempo, current VP mix
- `tempo` token: generation, action-slot index, pass order pressure, endgame pressure
- `global` tokens: temperature, oxygen, oceans, Venus, Moon rates, funded-award count, milestone slots remaining
- `opponent` tokens: each opponent’s economy, VP profile, board pressure, race pressure
- `milestone` tokens: exists, owned, my progress, opponent-best progress, turns-to-claim estimate, claim-now flag, deny-risk
- `award` tokens: exists, funded, my standing, opponent standing, projected points, fund-now EV, contestability
- `board opportunity` tokens: top-K strategic spaces, not every hex; include city, greenery, ocean, special, mine, habitat, road opportunities with self-value, deny-value, follow-up combo value
- `tableau` tokens: own engine cards with tags, effects, VP/resource behavior
- `hand` tokens: candidate cards with affordability, readiness, reachability, synergy, endgame usefulness

Do not keep awards/milestones as anonymous flat slices.

### 2. Replace card-only attention with a unified world encoder
Model architecture:
- Stage A: typed embedding + positional/group embedding for all `world_tokens`
- Stage B: multi-layer transformer over `world_tokens`
- Stage C: cross-attention from `hand_tokens` into encoded world context
- Stage D: cross-attention from `action_tokens` into encoded world context
- Stage E: produce one scalar logit per legal action token, plus a shared state value

Important constraint:
- Do not attend over the full raw board. Use top-K opportunity tokens to keep compute bounded and training stable.

### 3. Replace the fixed 1000-action head with action-conditioned scoring
Public interface change:
- Network forward should take the token bundle and return logits only for currently legal actions.
- `ActionDecoder` must expose a typed legal-action list with per-action features, not just indices.

Each action token should include:
- action family: play card, standard project, claim milestone, fund award, tile placement, convert plants, convert heat, pass, etc.
- immediate resource delta
- board delta summary if applicable
- milestone/award delta hints if inferable
- carry-over impact flags
- combo tags such as `spend_threshold_resource`, `creates_city_anchor`, `creates_greenery_followup`, `raises_claimability`, `locks_award_lead`

This is the main mechanism for explicit “this action improves Gardener” or “saving plants is better than a bad greenery now”.

### 4. Add explicit planning heads and training targets
Replace the current aux set with planner-oriented targets:
- `milestone_claim_now[70]`
- `milestone_turns_to_claim_bucket[70]`
- `award_fund_now_ev[all_awards]`
- `award_rank_class[all_awards]`
- `carry_save_plants_value`
- `carry_save_heat_value`
- `next_turn_combo_value`
- `next_generation_combo_value`
- `board_opportunity_value[top_k]`
- `deny_risk[top_k or per-race token]`

Training targets should be built from live state plus short-horizon transition labels:
- whether saving plants/heat improved next-turn or next-generation EV
- whether an action increased milestone claimability or award standing
- whether a board action created or destroyed high-value follow-ups
- whether an opportunity was lost to an opponent within a short window

Keep PPO, but train policy/value against this richer action-conditioned representation.

### 5. Rework reward shaping around plan quality, not only immediate spend efficiency
Keep terminal reward and light tactical shaping, but add explicit short-horizon planning components:
- positive reward for preserving threshold resources when projected next-turn/next-generation value is higher than immediate spend
- penalty for bad threshold burns, such as spending 8 plants into a low-value greenery while a stronger combo is imminent
- reward for actions that materially increase `claim-now` probability on milestones
- reward for board actions that improve a funded award or create a near-forced follow-up
- opponent-risk shaping for cases where delaying likely loses a milestone/award/window

Reduce generic “spend resources efficiently” pressure when it conflicts with modeled carry-over value.

## Interface Changes
- `StateEncoder.encode(...)` returns a structured token bundle instead of a flat vector.
- `TerraformingMarsNetwork.forward(...)` accepts token bundles plus legal action tokens.
- `ActionDecoder.get_available_actions(...)` must return typed legal action payloads and an index map.
- Debug snapshots and PPO rollout storage must persist planner targets and action-token metadata.
- Existing checkpoints are intentionally incompatible.

## Test Plan
Add focused tests for:
- token bundle shapes, masks, and deterministic ordering of milestone/award tokens
- legal-action token encoding for `claim milestone`, `fund award`, `convert plants`, `convert heat`, `space`, and `project card`
- policy masking: only legal actions receive logits
- endgame fixture: 7 plants with positive plant production prefers saving over weak greenery
- endgame fixture: city now plus greenery next round beats isolated greenery now
- milestone fixture: claim-now action outranks non-claim actions when denial risk is high
- award fixture: positive-EV late funding is preferred; negative-EV early funding is suppressed
- board fixture: Moon road/habitat/mine opportunities surface as high-value opportunity tokens when relevant
- regression test that no-card states still work because non-card world tokens carry the context

## Assumptions
- Retraining from scratch is acceptable.
- Fixed 1000-way policy logits are removed in favor of legal-action scoring.
- Full-board attention is out of scope; top-K strategic opportunity tokens are the board abstraction.
- PPO remains the training algorithm for v1 of this redesign.
- The existing `RequirementPlanner` logic is reused and expanded as an input feature source, not discarded.
