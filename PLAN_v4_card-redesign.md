# V4 Card-Aware Policy Redesign

## Summary

The isolated `tfm-rl-v4` card-aware experiment is implemented using the existing 512-wide, 3-layer transformer. V2/V3 artifacts remain preserved and V4 warm-starts compatible weights into a fresh optimizer.

V4 is a superset of the existing V3 observability improvements. It keeps 64-wide planner tokens while adding categorical card identity, structured metadata, and explicit action-to-card references.

## Immutable Baseline and Recovery Status

- The completed 12-epoch run in `rl-v4/pretrain` is an immutable diagnostic baseline. Its selected epoch-11 checkpoint achieved held-out teacher top-1 `90.9%` and top-3 `99.1%`; those aggregate results are valid and the checkpoint/report must not be overwritten.
- PPO remains blocked. The baseline has invalid milestone family labels, `select_space` top-1 `80.5%`, `fund_award` top-1 `58%`, no held-out human result, and no checkpoint-bound fixed-seed smoke result.
- Baseline family results were `play_card` `89.3%`, `card_subset` `92.7%`, `select_option` `99.9%`, and payment `97.0%`. Milestone's reported `80.4%` does not measure a coherent claim-milestone target because sampled actions, rather than teacher-argmax targets, defined the old cohort.
- The read-only epoch-11 diagnostic does not qualify the alternate placement gate: target-family tile top-3 is `96.3%` (below `97%`), although `96.5%` of top-1 misses are within score gap `0.18`. The recovery therefore retains the `87%` placement top-1 gate and adds prompt-intent features. Target-family award top-1 is `53.8%` over 143 rows, confirming that award encoding must be repaired before retraining.
- Do not resume PPO or repeat the old 12-epoch recipe. Diagnose epoch 11 read-only, collect a new `teacher_sample.v5` dataset, and run one fresh gate-aware pretrain into `rl-v4/pretrain-v2`.

## Schema and Interface Changes

- Generate `card_catalog.v1.json` from `card_metadata.json`:
  - ID `0` is unknown/padding.
  - Existing 989 cards receive stable append-only IDs `1..989`.
  - Capacity is fixed at 2,048 cards so future additions do not change model shapes.
  - Store a deterministic catalog SHA-256 used by datasets and checkpoints.
- Give every card a fixed 128-value metadata vector:
  - 16 tag flags in the existing canonical tag order.
  - 8 card type/category flags.
  - 8 cost, VP, resource, starting-cash, card-cost, and TR scalars.
  - 16 requirement fields.
  - 12 resource-type and resource-behavior fields.
  - 28 immediate play-effect fields.
  - 28 repeatable action-effect fields.
  - 12 capability, availability, and fallback flags.
- Extend metadata generation to export structured `behavior`, action behavior, production/stock changes, global-parameter changes, drawing, placement, attacks, discounts, and resource movement. Use deterministic description parsing only when a structured behavior is unavailable.
- Extend planner bundles with:
  - `planner_schema_version = "planner.card_aware.v1"`
  - `card_catalog_sha256`
  - `world_card_ids: int64[world]`
  - `hand_card_ids: int64[hand]`
  - `action_card_mask: bool[action, hand]`
- For `play_card`, exactly one hand/candidate position is selected by `action_card_mask`.
- For `card_subset`, the mask represents the exact selected set, including the all-false “buy nothing” action.
- Prompt cards are inserted before ordinary hand extras. If a legal action references a card omitted by the 24-card limit, encoding fails explicitly instead of silently producing an empty reference.
- Use `teacher_sample.v5`, `v4-card-aware.v1` rollout state, and `tfm-rl-v4` checkpoints. V5 persists teacher-argmax `target_action_position` and `target_family` separately from the sampled `chosen_action_position`. Old bundles, rollouts, datasets, and checkpoints are rejected by normal V4 loading.
- Checkpoints store the catalog hash and refuse inference or training against a different catalog.

## Card and Action Encoder

- Keep the 512 hidden size, 3 transformer layers, 4 heads, and 64-wide base tokens.
- Add:
  - A learned `Embedding(2048, 512, padding_idx=0)` for card identity.
  - A learned `Linear(128, 512)` projection for static card metadata.
  - A card-set projection and zero-padded card-count embedding.
- Add card identity and static metadata embeddings to:
  - Tableau/world card tokens before world self-attention.
  - Hand/candidate tokens before hand-to-world attention.
- Make hand encoding residual:
  - `contextual_hand = LayerNorm(projected_hand + attended_world)`.
  - Continue using its masked mean in the global state summary.
- Build each action’s referenced-card representation with batched masked pooling:
  - `pool = sum(contextual_hand * action_card_mask) / sqrt(max(card_count, 1))`.
  - Combine the pool, count embedding, and existing projected action token into the action query.
- Let the resulting action query attend over the concatenation of encoded world and contextual hand tokens.
- Keep the existing three-way policy fuser shape—action query, attended context, and state summary—so its H512 weights can be warm-started.
- Preserve the first 49 existing action-token features. Use the remaining 14 family-specific slots for:
  - `play_card`: payment composition, payable/discounted cost, and metal eligibility.
  - `card_subset`: selected count, prompt bounds, purchase fee, total fee, remaining cash, aggregate cost/VP, tag diversity, and empty/all-selected flags.
  - Existing award, milestone, placement, and startup tails remain family-specific.
- Do not use card order or action position as a policy feature. Subset pooling must remain permutation-invariant.

## Legal Actions, Teacher Data, and Training

- Canonicalize executable payloads before exposing legal actions:
  - Serialize normalized payloads with sorted keys.
  - Keep the first catalog entry for exact duplicates.
  - Record aliases for diagnostics.
  - Reject remaining duplicate payloads during dataset validation.
- For imported legacy/human records, merge probability mass from equivalent actions before normalization.
- Replace the draft teacher’s catalog-position heuristic:
  - Resolve every selected name to its card.
  - Reuse the existing shared card-quality calculation.
  - For paid purchases, use `threshold = min(0.90, 0.60 + 0.05 × cardCost)`.
  - Score a subset as `sum(cardQuality - threshold)` minus a `0.02 × max(0, 14 - remainingMC)` reserve penalty.
  - For free selections, use a zero threshold.
  - Remove all action-position bonuses.
- Preserve the existing play-card teacher logic, now operating on canonical actions.
- Score each resolved milestone in the full legal-action vector using `2.9 + 0.45*phase + 0.55*deny_risk + 0.15*claim_surplus - 0.05*effective_cost`. Inputs include exact identity, own/opponent progress, claimability, denial urgency, generation, cost, and affordability. Reject unresolved identities; never emit the old flat milestone score.
- Encode milestone identity, own/opponent progress, score gap, claim-now, denial risk, cost, affordability, and phase in the action tail. Placement uses the final two action-tail slots for city and greenery prompt intent; ocean is represented by both flags being false.
- Normalize award identity and funded state across `color`, `playerColor`, names, and `funded_by` in the teacher, world tokens, and action tokens.
- Re-encode the existing raw human-listener events through the V4 decoder and encoder. Reserve two complete source games deterministically for test, keep the remaining games in training at natural frequency and unit sample weight, and evaluate the human gate only on the held-out games.
- Collect at least 100,000 fresh V5 teacher decisions in a new dataset, continuing until validation and test each contain at least 100 target-family examples for placement, awards, and milestones. Existing shards cannot be relabeled because they lack the raw state needed to recompute repaired teacher targets.
- Add an isolated `docker-compose.rl_v4.yml` using `./rl-v4`, a separate database volume, V2 mounted read-only for warm-starting, and the full card-aware/V3 feature set.
- Partial warm-start:
  - Require the source checkpoint to be H512/3-layer/4-head/token-64.
  - Reuse compatible world, hand, transformer, recurrent, value, auxiliary, action-attention, fuser, and logit-head weights.
  - Initialize card embeddings, metadata projection, card-set pooling, and count embeddings separately.
  - Reset optimizer, policy version, and training statistics.
  - Emit a provenance report containing source hash and reused/initialized parameter lists.
- Diagnose epoch 11 read-only for executed-family and target-family top-1/top-2/top-3, rank, reciprocal rank, candidate count, logit margin, teacher-score gap, identity confusion, placement value deltas, and award encodings. Placement may switch from the `87%` top-1 gate to a `97%` top-3 gate only if tile top-3 is at least `97%` and at least `80%` of top-1 misses are within teacher score gap `0.18`.
- Start the recovery run from `/app/v4/bootstrap/warm_start.pth` with a fresh AdamW optimizer, learning rate `1e-4`, batch size `128`, a 10-epoch cap, and patience two. Save every epoch immutably and rank candidates lexicographically by gates passed, worst normalized gate margin, aggregate top-1, then top-3.
- A higher aggregate checkpoint cannot displace a checkpoint that clears every family gate. After the first qualifying checkpoint, run at most one confirmation epoch and retain the qualifying checkpoint with the stronger worst-family margin. Evaluate teacher and human held-out tests once on the selected candidate; failure does not automatically launch another run.

## Tests and Promotion Gates

- Metadata/catalog tests:
  - Deterministic generation and hash.
  - All current 989 metadata entries mapped exactly once.
  - Structured behavior and description fallback normalization.
  - Unknown ID behavior and 2,048-card capacity enforcement.
- Encoding tests:
  - Different cards receive different IDs even when cost/tags match.
  - All 16 subsets of a four-card draft have distinct masks.
  - Reordering prompt cards and masks leaves pooled subset representations unchanged.
  - Every play/subset reference resolves to a candidate token.
  - “Buy nothing” remains distinguishable from non-card actions.
- Action tests:
  - Duplicate Lunar Beam-style payloads collapse to one legal action.
  - Canonicalized actions still round-trip through the live server.
  - Teacher draft scores depend on selected card quality, never catalog position.
- Network tests:
  - Padding and batched shapes remain correct.
  - Gradients reach card embeddings, static projections, and set pooling.
  - Changing only card identity can change the corresponding logit.
  - Non-selected card padding cannot affect a candidate score.
- Warm-start tests verify exact reusable-key loading, fresh card-module initialization, optimizer reset, and catalog-hash enforcement.
- PPO remains blocked until the held-out test split satisfies:
  - Zero duplicate executable actions.
  - Zero unresolved known-card references.
  - `play_card` top-1 ≥ 85%.
  - `card_subset` top-1 ≥ 90%.
  - Overall teacher top-1 ≥ 85% and top-3 ≥ 97%.
  - `select_option` top-1 ≥ 95%, `select_payment` top-1 ≥ 90%, `claim_milestone` top-1 ≥ 90%, and `fund_award` top-1 ≥ 80%.
  - Placement meets either `select_space` top-1 ≥ 87%, or top-3 ≥ 97% only when the saved epoch-11 diagnostic qualified the alternate mode.
  - Validation and test each contain at least 100 target-family placement, award, and milestone examples.
  - Human top-3 ≥ 80% on exactly the two held-out source games.
  - The selected checkpoint completes the deterministic fixed-seed candidate-game trace with PPO disabled and zero server-rejected actions.
- The promotion report records target-family metrics/counts, placement diagnostic mode, human game provenance, per-epoch gate status, selected epoch and SHA-256, and explicit failure reasons. Smoke evidence includes the same checkpoint SHA-256 and seed/configuration. `assert_ppo_unlocked` fails closed if the report, checkpoint, gates, or smoke binding disagree.

## Assumptions

- `card_metadata.json` remains the authoritative source.
- Current V2/V3 datasets and checkpoints remain untouched.
- The original V4 epoch-11 checkpoint and report remain untouched and are used only for diagnostics.
- Existing uncommitted user changes are preserved and integrated rather than overwritten.
- Draft prompts continue to have at most four cards, but the mask-based interface supports any prompt fitting the configured 24-card candidate limit.
