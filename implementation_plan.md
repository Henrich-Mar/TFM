You should not expect this to self-correct just by running more generations. There are real code-level biases and at least one action-design bug that can push agents toward standard projects.

What I found:

Hardcoded standard-project preference in sampling.
agent.py (line 726) multiplies standard-project probability by 1.5, and agent.py (line 644) also upweights “standard project” OR-options. This alone biases policy selection.

Multi-card buy/keep selection is encoded incorrectly.
action_decoder.py (line 1538) exposes actions as PLAY_CARD(0..n-1) for selection prompts, but action_decoder.py (line 777) interprets action index as a bitmask when max_cards > 1. So the policy cannot represent most subsets correctly (research/buy phases become crippled).

Card quality signal is too weak in state encoding.
state_encoder.py (line 267) encodes hand mostly as aggregates (count/cost/tag counts), not per-card utility. So the model has little ability to learn “which specific card is best now”.

Reward is terminal-only and coarse.
scoring.py (line 31) uses rank/VP at end only. No signal for resource utilization or card-engine quality during the game.

Payment-path mismatches have occurred and should be guarded.
You have underpay rejects in logs (example: rl-coordinator.log.1 (line 38636) onward includes repeated project-card payment issues). Even if partially improved now, this needs hard validation.

Detailed patch plan (ordered):

Patch action bias first (low risk, immediate effect).
Edit agent.py.
Set standard-project multiplier from 1.5 to 1.0 (or 0.9 temporarily) in _sample_action.
Set OR-menu “standard project” weight from 1.5 to 1.0 in _get_action_from_network.
Success criterion: standard-project action share drops materially within 3-5 gens without VP collapse.

Fix multi-select card action encoding (high impact).
Edit action_decoder.py.
Add a dedicated action range for card-subset masks in buy/keep flows (instead of PLAY_CARD(i) only).
In get_available_actions, emit valid mask actions for card selection prompts where max > 1.
In build_response_for_input, decode that mask directly to selected card names.
Success criterion: agents can choose any offered card/subset in research; hand quality and card play rate rise.

Harden affordability/payment correctness for projectCard.
Edit action_decoder.py.
Make affordability and payment checks mirror server rules from Player.ts (line 718) and Player.ts (line 1275).
Include reserve-unit constraints in affordability checks.
Reject action generation when payment cannot fully satisfy card cost.
Success criterion: zero "Did not spend enough to pay for card" over a long training window.

Add card-level features to state (so they can learn card value).
Edit state_encoder.py.
Keep aggregate features, but add top-K candidate-card slots (cost, affordability, VP proxy, key tags, resource synergy indicators).
Add explicit steel/titanium spendability pressure features (stock + production + playable-tag opportunities now).
Success criterion: improved conversion of steel/titanium into card plays and higher non-SP VP sources.

Add behavior metrics and gating.
Track per-generation:
card_plays_per_game, standard_project_ratio, steel_spent, titanium_spent, payment_reject_count.
Use these as promotion gates alongside fitness.
Success criterion: no regressions hidden by raw rank/VP.

Retrain with A/B validation.
Run two branches from the same checkpoint:
A = current baseline.
B = patched logic.
Compare after 10-20 generations using same seeds/opponents.
Pick branch only if B improves card usage/resource usage and keeps or improves VP/win rate.