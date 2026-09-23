# V4 Card-Aware Policy Redesign

## Summary

Create an isolated `tfm-rl-v4` experiment using the existing 512-wide, 3-layer transformer. Preserve V2/V3 artifacts, partially warm-start from `/app/v2/pretrain-h512/bc_best.pth`, and retrain on newly encoded card-aware teacher data.

V4 will be a superset of the existing V3 observability improvements. It will keep 64-wide planner tokens while adding categorical card identity, structured metadata, and explicit action-to-card references.

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
- Introduce `teacher_sample.v4`, `v4-card-aware.v1` rollout state, and `tfm-rl-v4` checkpoints. Old bundles, rollouts, and checkpoints are rejected by normal V4 loading.
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
- Re-encode the existing raw human-listener events through the V4 decoder and encoder. Match choices by canonical response payload and copy terminal outcomes from the existing completed human episodes; mark value targets invalid when no outcome exists.
- Collect at least 100,000 fresh V4 teacher decisions. Existing V2 planner bundles are not converted because they lack tableau and hand card identities.
- Add an isolated `docker-compose.rl_v4.yml` using `./rl-v4`, a separate database volume, V2 mounted read-only for warm-starting, and the full card-aware/V3 feature set.
- Partial warm-start:
  - Require the source checkpoint to be H512/3-layer/4-head/token-64.
  - Reuse compatible world, hand, transformer, recurrent, value, auxiliary, action-attention, fuser, and logit-head weights.
  - Initialize card embeddings, metadata projection, card-set pooling, and count embeddings separately.
  - Reset optimizer, policy version, and training statistics.
  - Emit a provenance report containing source hash and reused/initialized parameter lists.
- Train with AdamW, learning rate `1e-4`, batch size `128`, up to 12 epochs, and early stopping after three validation epochs without improvement.

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
  - `select_option` ≥ 95%, `select_space` ≥ 87%, milestones ≥ 98%, and payment ≥ 90%.
  - Human top-3 ≥ 80%.
  - No server-rejected action in the fixed-seed smoke run.

## Assumptions

- `card_metadata.json` remains the authoritative source.
- Current V2/V3 datasets and checkpoints remain untouched.
- Existing uncommitted user changes are preserved and integrated rather than overwritten.
- Draft prompts continue to have at most four cards, but the mask-based interface supports any prompt fitting the configured 24-card candidate limit.
