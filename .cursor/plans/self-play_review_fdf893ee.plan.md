---
name: Self-play review
overview: Replace the frozen teacher/champion curriculum with four-seat PPO self-play. All learning seats share one network and one rollout buffer. One seat is occasionally a past checkpoint. Promotion is by match score against the previous checkpoint and the teacher.
todos:
  - id: shared-seats
    content: Add learning-seat copies that share the learner network, optimizer, rollout buffer, and policy version
    status: completed
  - id: lineup
    content: Seat the live policy in all four chairs, swapping one chair for a frozen past checkpoint on 25% of games
    status: completed
  - id: promote
    content: Promote only when the candidate beats the previous checkpoint and does not lose to the teacher
    status: completed
  - id: tests
    content: Replace the stage-0 mix test and lock shared-buffer, frozen-opponent, and lineup behavior
    status: completed
isProject: false
---

# Four-seat PPO self-play

The training game is the current policy against itself. No Monte Carlo tree search. Teacher, random, and the frozen champion leave the training lineup. They stay as evaluation baselines.

## Lineup

Change [`V2SelfPlayRunner._opponents`](rl-environment/training/v2_self_play.py) and [`_reserve_selfplay_game`](rl-environment/training/v2_self_play.py). V4 already calls this runner from [`training/v4_self_play.py`](rl-environment/training/v4_self_play.py), so v2, v3, and v4 all pick up the same loop.

Every game has four seats:

- Default: four stochastic copies of the live policy. `train_from_self_play=True`, `ppo_enable=True`, `deterministic_actions=False`.
- On 25% of games, and only when `history` is non-empty, replace exactly one seat with a frozen past champion (`ppo_enable=False`, argmax). That seat does not write rollouts.
- Until the first promotion, `history` is empty, so all four seats are the live policy.

Do not put the same `RLAgent` object in four seats. Concurrent games and four chairs would share one recurrent-state map and one in-flight episode. Build a seat agent per chair that aliases the leader:

- `network` and `optimizer` are the leader's objects, so one PPO step updates every chair.
- Rollout appends go to `leader.rollout_buffer`.
- `policy_version` stamped on each step is read from the leader at queue time, so `PPO_STRICT_ON_POLICY` still drops stale episodes after `optimize_from_rollout_buffer` increments the leader.

The hook belongs on `RLAgent` in [`rl-environment/models/agent.py`](rl-environment/models/agent.py), next to `_queue_episode_rollout` (around line 4214) and `policy_version` (around line 976). Seat agents must not call `optimize_from_rollout_buffer`. Only the leader does, after the batch, as today.

`_refresh_frozen_pools` keeps the historical pool only. Drop `teacher_pool`, `random_pool`, and `champion_pool` from training. The champion file remains the last promoted checkpoint used by the benchmark.

## Promotion

[`_evaluate_and_promote`](rl-environment/training/v2_self_play.py) currently promotes stage 0 on the random gate plus the champion regression gate, and stage 1 on the teacher gate plus that same regression gate. Replace both with one rule, on whichever stage is running:

- Benchmark the candidate against the previous checkpoint (`baseline="champion"`, the existing pairwise gate: score at least 0.50, no server rejections).
- Benchmark the candidate against the teacher (`baseline="teacher"`, the existing stage-1 gate: Wilson lower bound on first place above 0.25, no rejections).
- Promote only if both pass. On promote, archive the old champion into `history` (keep the last 8) and copy the candidate to `champion.pth`.

Stage 0 versus stage 1 stays a ruleset switch (`players_beginner`, game options). It no longer changes who sits down.

Leave the behavior-cloning pretrain gate in [`training/v4_gates.py`](rl-environment/training/v4_gates.py) alone. It decides whether PPO is allowed to start. It is not the self-play promotion test.

## Tests and docs

Update [`rl-environment/tests/test_v2_self_play_concurrency.py`](rl-environment/tests/test_v2_self_play_concurrency.py):

- Delete `test_stage_zero_opponent_mix_includes_champion_teacher_and_random`.
- A game with empty history has four learning seats sharing the leader network and buffer.
- A forced 25% draw with non-empty history has three learning seats and one frozen historical seat.
- A step queued by a seat lands on the leader buffer with the leader's `policy_version`.
- After a fake leader optimize, the historical seat's weights are unchanged.

Update the single-learner paragraph in [`README-RL-V2.md`](README-RL-V2.md) (section 6) so it describes four-seat self-play instead of frozen teacher/random/champion opponents.

## Out of scope

No MCTS. No second network. No change to the 15.0M `TerraformingMarsNetwork` architecture. No change to rank reward or step shaping.
