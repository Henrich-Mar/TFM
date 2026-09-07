Yes. I would restart the **action/masking pipeline**, not necessarily the entire project. The key principle should be:

> During training, never hide an invalid state or invalid action with a fallback.

Fallbacks may still exist for a production-playing agent, but they should be logged separately and excluded from teacher/PPO training until the core pipeline is reliable.

# Recommended restart plan

## Phase 0: Freeze the current system

Before changing code:

- Create a new branch, for example:
  - `action-pipeline-rebuild`
- Freeze the current branch as a known reference.
- Save:
  - Existing teacher samples.
  - Existing game logs.
  - Server responses.
  - Action payloads.
  - Rejected-action logs.
  - PPO rollout files.
- Record the current commit hash.
- Disable long PPO training runs while rebuilding the action system.

Create a short document containing:

```text
Current action dimension:
Current action families:
Current action index ranges:
Current card-selection behavior:
Current fallback behavior:
Current server rejection behavior:
Current terminal-state behavior:
```

Do not delete the old implementation yet. Keep it available for comparison.

---

# Phase 1: Define the action contract

Before writing new code, define one canonical representation for every action.

## 1. Define action families

For example:

```python
ActionFamily = Literal[
    "project",
    "standard_project",
    "convert_resource",
    "payment",
    "card_selection",
    "startup_selection",
    "pass",
]
```

Every action should have:

```python
@dataclass(frozen=True)
class Action:
    action_id: int
    family: str
    payload: dict
    description: str
```

Example:

```python
Action(
    action_id=537,
    family="card_selection",
    payload={
        "type": "card",
        "cards": ["Card A", "Card C"],
    },
    description="Select Card A and Card C",
)
```

The important point is that the policy should eventually select an `action_id`, but the action ID must have one canonical payload representation.

## 2. Create one action registry

Do not scatter action ranges throughout the code.

```python
ACTION_RANGES = {
    "payment": range(400, 408),
    "card_selection": range(520, 600),
    "startup_selection": range(850, 882),
}
```

Add validation:

```python
def validate_action_ranges(action_ranges: dict[str, range]) -> None:
    ranges = list(action_ranges.items())

    for i, (name_a, range_a) in enumerate(ranges):
        for name_b, range_b in ranges[i + 1:]:
            assert set(range_a).isdisjoint(set(range_b)), (
                f"Action ranges overlap: {name_a} and {name_b}"
            )
```

Also verify:

- Every generated action has exactly one family.
- Every action family has a decoder.
- Every decoded action has a server payload.
- No action ID is duplicated.

## 3. Define legal-action semantics

You need an explicit distinction between:

```text
Legal gameplay state:
- One or more legal actions exist.

Terminal state:
- No actions should be sampled.

Invalid state:
- The environment claims to be active, but no legal action exists.

Fallback state:
- The agent could not produce a legal policy action.
```

Do not treat these as equivalent.

A useful result type is:

```python
@dataclass
class LegalActionSet:
    status: Literal["active", "terminal", "invalid"]
    actions: list[Action]
    reason: str | None = None
```

Rules:

- `active` requires at least one action.
- `terminal` must not enter the policy-loss batch.
- `invalid` must raise an error or be quarantined.
- `fallback` is not a legal-action status.

---

# Phase 2: Rebuild teacher samples first

Your idea of starting with teacher samples and manual tests is good. Do this before PPO.

## 1. Store complete teacher examples

Each teacher sample should include:

```json
{
  "sample_id": "game_001_turn_14",
  "state": {},
  "raw_waiting_for": {},
  "canonical_prompt": {},
  "legal_actions": [],
  "selected_action_id": 537,
  "selected_action_payload": {
    "type": "card",
    "cards": ["Card A", "Card C"]
  },
  "action_mask": [],
  "action_source": "teacher",
  "terminal": false,
  "fallback_used": false,
  "server_response": {},
  "valid": true
}
```

Do not store only the selected action. Store enough information to reconstruct and audit the mask.

## 2. Reject samples with fallback behavior

For the initial rebuild, samples should be rejected if:

- A fallback action was used.
- The selected action was not in the legal-action set.
- The legal-action set was empty for an active state.
- The decoded payload differs from the intended payload.
- The server rejected the action.
- The prompt could not be parsed.
- The action ID was outside the declared range.
- The state was terminal but an action was selected.

Use a strict validation function:

```python
def validate_teacher_sample(sample: TeacherSample) -> None:
    assert not sample.fallback_used
    assert not sample.terminal
    assert sample.selected_action_id in sample.legal_action_ids
    assert sample.selected_action_id < len(sample.action_mask)
    assert sample.action_mask[sample.selected_action_id]
```

## 3. Build a teacher-sample dashboard or report

At minimum, report:

- Total samples.
- Valid samples.
- Invalid samples.
- Empty legal masks.
- Invalid selected actions.
- Decoder failures.
- Server rejections.
- Nested-prompt failures.
- Card combinations exceeding the old limit.
- Fallback count.

The goal is to make the following number equal to zero:

```text
invalid samples used for training
```

Do not optimize the model while this number is nonzero.

---

# Phase 3: Rebuild legal-action generation

This should be the source of truth for both teacher collection and PPO.

## 1. Create a single legal-action function

Use one function for all callers:

```python
def enumerate_legal_actions(
    state: GameState,
    waiting_for: dict,
) -> LegalActionSet:
    ...
```

It should be used by:

- Teacher data generation.
- Manual inspection tools.
- PPO rollout collection.
- Inference.
- Server round-trip tests.
- Debug logs.

Avoid having separate implementations for:

- Mask generation.
- Action decoding.
- Teacher action enumeration.
- Human-readable action display.

Those implementations will eventually disagree.

## 2. Make empty masks fail loudly

For active prompts:

```python
if status == "active" and not actions:
    raise EmptyLegalActionError(
        f"No legal actions for active prompt: {waiting_for!r}"
    )
```

For terminal prompts:

```python
if status == "terminal":
    return LegalActionSet(
        status="terminal",
        actions=[],
    )
```

Do not use:

```python
mask[:] = True
```

Do not randomly select an action when enumeration fails.

## 3. Add explicit pass/no-op only if the game supports it

If the game genuinely allows a pass action, represent it explicitly:

```python
Action(
    action_id=12,
    family="pass",
    payload={"type": "pass"},
    description="Pass",
)
```

Do not use a pass action as a generic error recovery mechanism.

---

# Phase 4: Fix card-selection handling

## 1. Build a canonical prompt traversal utility

Nested prompt handling should be centralized.

```python
def find_card_selection_prompts(prompt: dict) -> list[dict]:
    ...
```

It should recursively inspect:

- Root prompt.
- `options`.
- `responses`.
- `response`.
- Nested `or` branches.
- Nested card-selection branches.

Then both enumeration and decoding should use the same resolved prompt.

## 2. Represent card selections by card identity

Internally, use something like:

```python
CardSelection = tuple[str, ...]
```

For example:

```python
("Card A", "Card C")
```

Only convert that selection to a bitmask or action ID at the final policy boundary.

The flow should be:

```text
server prompt
    ↓
canonical card list
    ↓
legal card selections
    ↓
Action objects with card names
    ↓
action IDs and neural-network mask
```

Not:

```text
integer mask
    ↓
guess card names later
```

## 3. Remove the fixed 80-combination assumption

Prefer this order of solutions:

### Best option: dynamic action catalog

Generate all legal combinations and assign action IDs from a dynamic catalog for that prompt.

### More scalable option: autoregressive selection

Predict:

1. First card.
2. Second card or stop.
3. Third card or stop.
4. Continue until complete.

### Temporary option: increase the limit

If you cannot redesign immediately:

- Increase the limit.
- Add telemetry for truncated combinations.
- Fail the sample if legal combinations are truncated.
- Never silently omit combinations.

For example:

```python
if len(generated_combinations) > MAX_COMBINATIONS:
    raise CardCombinationOverflowError(
        f"Generated {len(generated_combinations)} combinations; "
        f"limit is {MAX_COMBINATIONS}"
    )
```

Do not train on truncated action spaces.

## 4. Add combination completeness tests

For a card list and selection bounds, compare generated combinations against an independent implementation using `itertools.combinations`.

Test:

- One-card selections.
- Multiple-card selections.
- Minimum and maximum bounds.
- Disabled cards.
- Nested prompts.
- Duplicate names, if possible.
- Large hands.
- Empty hands.
- Invalid selection bounds.

---

# Phase 5: Add mask and neural-network protections

## 1. Validate every mask before model execution

```python
def validate_action_mask(mask: torch.Tensor, terminal: torch.Tensor) -> None:
    active = ~terminal
    has_action = mask.any(dim=-1)

    if not torch.all(has_action[active]):
        raise RuntimeError("Active state has no legal actions")
```

## 2. Reject all-false masks in the network

The network should never silently produce a uniform distribution.

```python
if not action_mask.any(dim=-1).all():
    raise RuntimeError("Cannot run policy with an empty action mask")
```

For terminal states, skip the policy network entirely.

## 3. Use a numerically safe masked distribution

Make sure the selected action is legal:

```python
distribution = Categorical(logits=masked_logits)

action = distribution.sample()

assert action_mask[
    torch.arange(action.shape[0], device=action.device),
    action,
].all()
```

Also test:

- One legal action.
- Several legal actions.
- All actions legal.
- Empty mask.
- Terminal state.
- Batch containing both active and terminal states.

---

# Phase 6: Add rollout consistency checks

Every rollout transition should contain:

```python
@dataclass
class RolloutStep:
    observation: dict
    action_id: int
    action_mask: list[bool]
    legal_action_ids: list[int]
    action_payload: dict
    action_source: str
    log_probability: float
    value: float
    reward: float
    terminal: bool
```

Validate immediately when adding the step:

```python
def validate_rollout_step(step: RolloutStep) -> None:
    if step.terminal:
        return

    assert step.action_id in step.legal_action_ids
    assert step.action_mask[step.action_id]
    assert step.action_source in {
        "policy",
        "teacher",
        "random",
        "fallback",
    }
```

Do not wait until PPO optimization to find malformed samples.

## Add a batch-level check

Before computing PPO loss:

```python
active_rows = ~batch.terminal

selected_is_legal = batch.action_mask[
    torch.arange(batch_size),
    batch.actions,
]

assert selected_is_legal[active_rows].all()
```

Also assert:

- Action IDs are within bounds.
- Mask dimensions match policy output dimensions.
- No active sample has an empty mask.
- No terminal sample contributes policy loss.
- Old log probabilities are finite.
- New log probabilities are finite.
- Advantages are finite.

---

# Phase 7: Separate policy actions from fallback actions

This is important for your training concern.

## 1. Label action provenance

Use an explicit source:

```python
ActionSource = Literal[
    "teacher",
    "policy",
    "random_exploration",
    "decoder_fallback",
    "retry_fallback",
    "pass_fallback",
]
```

## 2. Exclude fallback actions initially

For the first clean training runs:

```python
trainable = rollout.action_source.isin({
    "teacher",
    "policy",
})
```

Do not include:

- Decoder fallback.
- Random retry.
- Generic pass.
- Actions selected after server rejection.

You can retain them for diagnostics.

## 3. Treat server rejection as a hard data-quality event

When the server rejects an action:

1. Record the raw state.
2. Record the generated legal actions.
3. Record the selected action.
4. Record the payload.
5. Record the server error.
6. Mark the transition invalid.
7. Do not add the fallback replacement to the same PPO transition.

A possible structure:

```json
{
  "event": "action_rejected",
  "state_id": "game_001_turn_14",
  "policy_action_id": 537,
  "policy_payload": {},
  "fallback_action_id": 12,
  "fallback_payload": {},
  "training_eligible": false
}
```

## 4. Start with fallback count equal to zero

Your training gate should be:

```text
fallback actions in training batch: 0
```

Later, you can decide whether certain fallback types are safe to include, but not during the initial rebuild.

---

# Phase 8: Build manual inspection tools

You should be able to inspect a state without running the model.

Create a command such as:

```bash
python tools/inspect_state.py \
  --sample samples/game_001_turn_14.json
```

It should print:

```text
Prompt type: card_selection
Terminal: false
Number of legal actions: 16

Action 520:
  family: card_selection
  cards: ["Card A"]

Action 521:
  family: card_selection
  cards: ["Card B"]

Action 522:
  family: card_selection
  cards: ["Card A", "Card B"]

...
```

Also print:

```text
Selected action:
  action_id: 522
  payload: {"type": "card", "cards": ["Card A", "Card B"]}

Selected action is legal: yes
Fallback used: no
```

Useful manual commands:

```bash
python tools/inspect_state.py --sample samples/state.json
python tools/validate_teacher_samples.py --input samples/
python tools/check_action_ranges.py
python tools/check_card_combinations.py
python tools/replay_server_action.py --sample samples/state.json
```

Manual inspection is especially valuable for:

- Nested prompts.
- Payment prompts.
- Startup selection.
- Multi-card selection.
- End-of-game states.
- Server responses after rejected actions.

---

# Phase 9: Add server round-trip tests

These tests should operate on actual serialized prompts and payloads.

## Card-selection round trip

For each generated card-selection action:

1. Load a real or deterministic game state.
2. Generate the legal action catalog.
3. Decode each action ID.
4. Submit the payload.
5. Confirm the server accepts it.
6. Confirm the resulting prompt is valid.

Test:

- Single-card selection.
- Multi-card selection.
- Minimum selection count.
- Maximum selection count.
- Nested card prompts.
- Large card lists.
- Disabled cards.
- Multiple branches.

## Payment round trip

Test:

- MegaCredits only.
- Steel discounts.
- Titanium discounts.
- Mixed payments.
- Insufficient resources.
- Exact payment.
- Overpayment.
- Alias normalization.
- Card-specific payment rules.

The test should distinguish:

```text
Generated legal action accepted by server: pass
Generated legal action rejected by server: failure
```

Do not make the test pass by retrying with a fallback.

---

# Phase 10: Reintroduce PPO gradually

Do not immediately launch full training.

## Stage 1: Teacher-only validation

Use only teacher samples.

Requirements:

- No fallback samples.
- Every selected action is legal.
- Every payload round-trips.
- No empty active masks.
- No decoder mismatch.
- No action-range errors.

## Stage 2: One-step policy inference

Run the model on held-out teacher states.

Measure:

- Percentage of sampled actions that are legal.
- Percentage of payloads accepted by the server.
- Percentage of empty masks.
- Percentage of decoder failures.
- Percentage of fallback attempts.

Target:

```text
illegal sampled actions: 0
server rejections: 0
fallback actions: 0
```

## Stage 3: Short rollout

Run very short games with PPO disabled.

Collect diagnostics only.

Do not train yet.

## Stage 4: PPO with strict filtering

Train only on:

- Valid active states.
- Legal policy actions.
- Accepted server actions.
- No fallback transitions.
- Finite rewards and log probabilities.

## Stage 5: Longer training

Only after the short-run metrics remain clean should you start longer runs.

---

# Suggested implementation order

This is the order I would use:

1. Freeze the current branch and save all logs.
2. Define the canonical `Action` and `LegalActionSet` types.
3. Build the action registry and range validator.
4. Implement one canonical legal-action enumerator.
5. Implement strict empty-mask behavior.
6. Rebuild teacher-sample generation.
7. Add teacher-sample validation.
8. Add manual state inspection.
9. Fix nested prompt traversal.
10. Rebuild card-selection enumeration.
11. Remove or explicitly detect combination truncation.
12. Add action decoding tests.
13. Add all-false network-mask checks.
14. Add rollout action/mask assertions.
15. Add action-source tracking.
16. Exclude fallback transitions from training.
17. Add card and payment server round-trip tests.
18. Run teacher-only validation.
19. Run policy inference without PPO.
20. Reintroduce PPO in short, controlled experiments.

---

# Definition of done

I would consider the rebuild ready for PPO only when all of these are true:

- No active prompt produces an empty legal-action set.
- Terminal prompts are explicitly identified.
- No all-false mask reaches the network.
- Every sampled training action is present in its mask.
- Every action ID maps to exactly one payload.
- Every payload maps back to the same action semantics.
- Nested card prompts are handled.
- Card combinations are not silently truncated.
- Action ranges are validated at startup.
- Fallback actions are not included in PPO batches.
- Teacher samples contain no rejected actions.
- Card-selection payloads pass server round-trip tests.
- Payment payloads pass server round-trip tests.
- Invalid samples are quarantined rather than repaired silently.
- A short rollout produces zero fallback actions and zero server rejections.

The most important change is to replace the current philosophy of:

```text
If something goes wrong, create a usable action.
```

with:

```text
If something goes wrong, preserve the failure, reject the sample, and investigate it.
```

That will temporarily reduce the amount of training data, but the remaining data will be much more trustworthy.