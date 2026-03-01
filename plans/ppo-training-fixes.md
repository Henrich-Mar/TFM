# PPO Training Fixes Plan

## Summary

Four correctness bugs in the PPO training pipeline for the Terraforming Mars RL agent. Each silently degrades training quality. All changes are in two files:

- [`agent.py`](rl-environment/models/agent.py)
- [`ppo.py`](rl-environment/models/ppo.py)

---

## Fix 1: Episode Steps Truncation — Keep Last N, Not First N

### Problem

In [`_make_move()`](rl-environment/models/agent.py:1495), steps are only appended when `len(episode_steps) < self.config.max_episode_steps`. Once the cap is hit, **later decisions are silently dropped**. But in [`_queue_episode_rollout()`](rl-environment/models/agent.py:2520), the last recorded step is treated as terminal and receives the terminal reward. If the real game went longer than `max_episode_steps`, the "terminal step" is actually mid-game — breaking credit assignment.

### Current Code

```python
# agent.py:1194 — episode_steps is a plain list
episode_steps: List[Dict[str, Any]] = []

# agent.py:1495 — guard stops appending after cap
if len(episode_steps) < self.config.max_episode_steps:
    ...
    episode_steps.append({...})
```

### Fix

1. **Change `episode_steps` from `List` to `deque(maxlen=max_episode_steps)`** in [`play_game()`](rl-environment/models/agent.py:1194). This automatically evicts the oldest steps when the cap is exceeded, keeping the most recent N steps.

2. **Remove the `if len(episode_steps) < ...` guard** in [`_make_move()`](rl-environment/models/agent.py:1495). Always append; the deque handles truncation.

3. **Update type hints** on [`_make_move()`](rl-environment/models/agent.py:1416), [`_queue_episode_rollout()`](rl-environment/models/agent.py:2515), and [`_train_from_episode()`](rl-environment/models/agent.py:2709) to accept `deque` instead of `List`.

### Files Changed

| File | Lines | Change |
|------|-------|--------|
| `agent.py` | ~1194 | `episode_steps = deque(maxlen=self.config.max_episode_steps)` |
| `agent.py` | ~1495 | Remove the `if len(episode_steps) < ...` guard; always append |
| `agent.py` | ~1416, ~2515, ~2709 | Update type hints from `List[Dict]` to accept `deque` |

---

## Fix 2: PPO Temperature Mismatch

### Problem

During action selection in [`_get_action_from_network()`](rl-environment/models/agent.py:2234), logits are divided by a per-move `effective_temperature` (which decays over training). The resulting `logp_old` is stored in the rollout step.

During PPO optimization in [`optimize_ppo_policy()`](rl-environment/models/ppo.py:364), logits are divided by a **single** `policy_temperature` — the current temperature at training time, not the temperature used when the step was collected.

This means the PPO importance-sampling ratio `π_θ(a|s) / π_θ_old(a|s)` is computed with inconsistent temperature scaling, violating PPO's core assumption.

### Current Code

```python
# agent.py:2234 — per-move temperature at collection time
policy_temperature = self._effective_policy_temperature()
policy_logits = policy_logits / max(policy_temperature, 1e-3)

# agent.py:2364 — stored but never used in PPO
action_meta["policy_temperature"] = float(policy_temperature)

# ppo.py:364 — uses current temperature, not per-step temperature
logits = logits / max(float(policy_temperature), 1e-3)
```

### Fix — Simplest Approach: Always Use Temperature=1.0 When PPO Is Enabled

When `ppo_enable` is True, the `_effective_policy_temperature()` should return `1.0`. This is already partially done for epsilon (see [`_effective_policy_epsilon()`](rl-environment/models/agent.py:968) which returns 0.0 when `strict_on_policy_sampling` is on), but temperature is not similarly handled.

1. **In [`_effective_policy_temperature()`](rl-environment/models/agent.py:972)**: When `self.strict_on_policy_sampling and self.ppo_enable`, return `1.0` instead of the decayed value.

2. **In [`optimize_ppo_policy()`](rl-environment/models/ppo.py:364)**: The `policy_temperature` parameter will now always be `1.0` when PPO is active, making the division a no-op. This ensures consistency between collection and training.

### Files Changed

| File | Lines | Change |
|------|-------|--------|
| `agent.py` | ~972-978 | Return `1.0` when `strict_on_policy_sampling and ppo_enable` |

---

## Fix 3: Heuristic Probability Reweighting Breaks PPO On-Policy Assumption

### Problem

In [`_sample_action()`](rl-environment/models/agent.py:2423-2433), multiplicative weights are applied to masked probabilities:
- `prefer_project_cards` multiplies play-card actions by `project_card_priority_weight` (default 1.6)
- OR-menu title adjustments multiply specific options by 0.08–1.8

The `logp_old` stored in the rollout is computed from the **post-reweighting** distribution (via `sampled_distribution`). But during PPO training, log-probs are recomputed from **raw masked logits only** — the heuristic reweighting is not applied. This means `π_θ_old(a|s)` at collection time ≠ what PPO thinks `π_θ_old(a|s)` was.

### Fix — Disable Heuristic Reweighting When PPO Is Enabled

The network already has learnable [`action_type_bias`](rl-environment/models/agent.py:446) that replaces the old hard-coded multipliers. The remaining heuristics in `_sample_action` should be disabled when PPO is active.

1. **In [`_get_action_from_network()`](rl-environment/models/agent.py:2291-2319)**: When `self.ppo_enable`, skip computing `action_weight_adjustments` and set `prefer_project_cards = False`.

2. **Alternatively**, convert the remaining heuristics to **additive logit offsets** (add `log(weight)` to logits before softmax) and apply them both at collection time and during PPO training. This is more work but preserves the heuristic behavior.

**Recommended approach**: Disable heuristics when PPO is on (option 1). The network's `action_type_bias` already handles the general case, and the OR-menu adjustments are small enough that the network can learn them.

### Files Changed

| File | Lines | Change |
|------|-------|--------|
| `agent.py` | ~2254-2319 | When `self.ppo_enable`: set `prefer_project_cards = False` and `action_weight_adjustments = None` |

---

## Fix 4: `aux_milestone_logits` Dropped by Normalizers

### Problem

The network's [`forward()`](rl-environment/models/agent.py:465-471) returns `aux_milestone_logits` (raw logits for BCE loss). But both normalizer functions strip it:

- [`_normalize_network_output()` in agent.py](rl-environment/models/agent.py:474-496) — does not include `aux_milestone_logits` in the returned dict
- [`_normalize_network_output()` in ppo.py](rl-environment/models/ppo.py:73-94) — same issue

So in [`optimize_ppo_policy()`](rl-environment/models/ppo.py:363), `out.get("aux_milestone_logits")` is always `None`, and the code falls back to MSE loss instead of the intended BCE loss.

### Fix

Include `aux_milestone_logits` in both normalizer functions.

### Files Changed

| File | Lines | Change |
|------|-------|--------|
| `agent.py` | ~474-496 | Add `aux_milestone_logits` extraction and return |
| `ppo.py` | ~73-94 | Add `aux_milestone_logits` extraction and return |

---

## Execution Order

The fixes are independent and can be applied in any order. Recommended order by impact:

1. **Fix 1** (episode truncation) — highest impact; terminal credit is fundamentally broken for long games
2. **Fix 4** (aux_milestone_logits) — simple one-liner fix, enables intended BCE loss
3. **Fix 2** (temperature mismatch) — important for PPO ratio correctness
4. **Fix 3** (heuristic reweighting) — moderate impact; partially mitigated by `strict_on_policy_sampling`

## Risk Assessment

- **Fix 1**: Low risk. `deque(maxlen=N)` is a drop-in replacement for list with automatic eviction. The `_queue_episode_rollout` already slices `[-max_episode_steps:]` so behavior is consistent.
- **Fix 2**: Low risk. When `strict_on_policy_sampling=True` (default), epsilon is already 0. Making temperature=1.0 is the natural companion.
- **Fix 3**: Medium risk. Disabling heuristics may change early-game behavior. But the network's `action_type_bias` should compensate. Monitor win rates after deployment.
- **Fix 4**: Low risk. Just passing through a key that the network already produces.
