import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.rollout_store import RolloutShardStore


def _step(action: int) -> dict:
    return {
        "state_bundle": {
            "world_tokens": np.zeros((1, 4), dtype=np.float32),
            "world_token_types": np.zeros((1,), dtype=np.int64),
            "world_mask": np.ones((1,), dtype=bool),
            "hand_tokens": np.zeros((1, 4), dtype=np.float32),
            "hand_mask": np.ones((1,), dtype=bool),
            "action_tokens": np.zeros((2, 4), dtype=np.float32),
            "action_mask": np.ones((2,), dtype=bool),
            "action_indices": np.array([0, 1], dtype=np.int64),
            "action_positions": np.array([0, 1], dtype=np.int64),
            "global_scalars": np.zeros((4,), dtype=np.float32),
        },
        "action": int(action),
        "action_index": int(action),
        "logp_old": float(-action),
        "value_old": float(action) / 10.0,
        "reward": float(action) / 100.0,
        "done": False,
        "legal_actions": [0, 1],
        "phase_index": 0,
        "recurrent_state": np.zeros((0,), dtype=np.float32),
        "aux_targets": np.zeros((1,), dtype=np.float32),
        "aux_predictions": np.zeros((1,), dtype=np.float32),
        "state_schema_version": "v1",
    }


def test_rollout_shard_store_persists_partial_consumption(tmp_path: Path) -> None:
    store = RolloutShardStore(
        root_dir=str(tmp_path / "rollouts"),
        agent_id="agent-1",
        shard_max_steps=3,
    )

    appended = store.append_steps([_step(i) for i in range(7)])

    assert appended == 7
    assert store.queued_step_count() == 7

    reopened = RolloutShardStore(
        root_dir=str(tmp_path / "rollouts"),
        agent_id="agent-1",
        shard_max_steps=3,
    )
    assert reopened.queued_step_count() == 7

    taken = reopened.pop_steps(5)

    assert [step["action"] for step in taken] == [0, 1, 2, 3, 4]
    assert reopened.queued_step_count() == 2

    reopened_again = RolloutShardStore(
        root_dir=str(tmp_path / "rollouts"),
        agent_id="agent-1",
        shard_max_steps=3,
    )
    assert reopened_again.queued_step_count() == 2

    tail = reopened_again.pop_steps(5)

    assert [step["action"] for step in tail] == [5, 6]
    assert reopened_again.queued_step_count() == 0


def test_rollout_shard_store_clear_removes_queued_steps(tmp_path: Path) -> None:
    store = RolloutShardStore(
        root_dir=str(tmp_path / "rollouts"),
        agent_id="agent-2",
        shard_max_steps=2,
    )
    store.append_steps([_step(i) for i in range(4)])

    cleared = store.clear()

    assert cleared == 4
    assert store.queued_step_count() == 0
    assert not list((tmp_path / "rollouts" / "agent_agent-2").glob("rollout_*.pkl.gz"))


def test_rollout_store_quarantines_mismatched_policy_without_deleting(tmp_path: Path) -> None:
    store = RolloutShardStore(str(tmp_path), "learner")
    current = [
        SimpleNamespace(
            episode_id="current", step_index=index, terminal=index == 1,
            state_schema_version="v1", policy_version=6,
        )
        for index in range(2)
    ]
    stale = [
        SimpleNamespace(
            episode_id="stale", step_index=index, terminal=index == 2,
            state_schema_version="v1", policy_version=7,
        )
        for index in range(3)
    ]
    store.append_episode(current)
    store.append_episode(stale)

    result = store.quarantine_incompatible("v1", policy_version=6)

    assert result["steps"] == 3
    assert result["shards"] == 1
    assert store.queued_step_count() == 2
    assert len(list((store.agent_dir / "quarantine_incompatible").glob("rollout_*.pkl.gz"))) == 1
