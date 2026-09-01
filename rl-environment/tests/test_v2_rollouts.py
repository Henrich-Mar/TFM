from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ppo import PPORolloutStep, _compute_gae_returns
from models.agent import RLAgent
from models.rollout_store import RolloutShardStore


def _step(episode: str, index: int, done: bool = False) -> PPORolloutStep:
    return PPORolloutStep(
        state_bundle={}, action=0, logp_old=0.0, value_old=0.0,
        reward=1.0 if done else 0.0, done=done, legal_actions=[0],
        episode_id=episode, step_index=index, policy_version=0, terminal=done,
    )


def test_complete_episode_store_never_splits_target(tmp_path: Path) -> None:
    store = RolloutShardStore(str(tmp_path), "v2", shard_max_steps=2)
    store.append_episode([_step("a", idx, idx == 3) for idx in range(4)])
    store.append_episode([_step("b", idx, idx == 2) for idx in range(3)])
    first = store.pop_complete_episodes(2)
    assert len(first) == 4
    assert {item.episode_id for item in first} == {"a"}
    second = store.pop_complete_episodes(2)
    assert len(second) == 3
    assert {item.episode_id for item in second} == {"b"}


def test_episode_store_rejects_tail_fragments(tmp_path: Path) -> None:
    store = RolloutShardStore(str(tmp_path), "v2")
    fragment = [_step("fragment", 4, False), _step("fragment", 5, True)]
    with pytest.raises(ValueError, match="contiguous"):
        store.append_episode(fragment)
    assert store.queued_step_count() == 0


def test_gae_does_not_cross_episode_boundary() -> None:
    payload = _compute_gae_returns(
        rewards=torch.tensor([0.0, 1.0, 0.0, -1.0]),
        dones=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        values=torch.zeros(4),
        gamma=1.0,
        gae_lambda=1.0,
        episode_ids=["a", "a", "b", "b"],
    )
    assert torch.allclose(payload["returns"], torch.tensor([1.0, 1.0, -1.0, -1.0]))


def test_nonterminal_episode_uses_explicit_bootstrap() -> None:
    payload = _compute_gae_returns(
        rewards=torch.tensor([0.0]),
        dones=torch.tensor([0.0]),
        values=torch.tensor([0.2]),
        gamma=1.0,
        gae_lambda=1.0,
        episode_ids=["truncated"],
        bootstrap_values=torch.tensor([0.7]),
    )
    assert torch.allclose(payload["returns"], torch.tensor([0.7]))


def test_old_policy_versions_are_discarded_as_complete_episodes() -> None:
    agent = RLAgent.__new__(RLAgent)
    old = [_step("old", index, index == 1) for index in range(2)]
    current = [_step("current", index, index == 2) for index in range(3)]
    for step in current:
        step.policy_version = 1
    agent.rollout_buffer = deque(old + current)
    agent.rollout_shard_store = None
    agent.strict_on_policy_sampling = True
    agent.policy_version = 1

    steps, filtered = agent._take_rollout_steps_locked(1, "v1")

    assert filtered == 2
    assert len(steps) == 3
    assert {step.episode_id for step in steps} == {"current"}
    assert not agent.rollout_buffer
