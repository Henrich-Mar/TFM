from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.replay_store import SEARCH_REPLAY_SCHEMA_VERSION, SearchReplayStore  # noqa: E402
from training.search_distill import load_search_records, validate_search_record  # noqa: E402


def _bundle(actions: int = 2) -> dict:
    return {
        "world_tokens": np.zeros((1, 64), dtype=np.float32),
        "world_token_types": np.ones((1,), dtype=np.int64),
        "world_mask": np.ones((1,), dtype=np.bool_),
        "hand_tokens": np.zeros((0, 64), dtype=np.float32),
        "hand_mask": np.zeros((0,), dtype=np.bool_),
        "action_tokens": np.zeros((actions, 64), dtype=np.float32),
        "action_mask": np.ones((actions,), dtype=np.bool_),
        "action_indices": np.arange(actions, dtype=np.int64),
        "action_positions": np.arange(actions, dtype=np.int64),
        "global_scalars": np.zeros((16,), dtype=np.float32),
    }


def _record(policy_version: int = 7) -> dict:
    return {
        "policy_version": policy_version,
        "planner_bundle": _bundle(),
        "phase_index": 1,
        "recurrent_state": [0.0] * 4,
        "policy_target": [0.75, 0.25],
        "mcts": {"simulations_selected": 8},
    }


def test_incomplete_episode_is_discarded(tmp_path):
    store = SearchReplayStore(tmp_path)
    store.record_decision("g1", "p1", _record())
    assert store.pending_count() == 1
    assert store.finish_episode("g1", "p1", completed=False) is None
    assert store.pending_count() == 0
    assert store.shard_paths() == []


def test_completed_episode_is_atomic_and_loadable(tmp_path):
    store = SearchReplayStore(tmp_path)
    store.record_decision("g1", "p1", _record())
    path = store.finish_episode(
        "g1",
        "p1",
        completed=True,
        value_target=0.8,
        outcome={"rank": 1, "vp": 80},
    )
    assert path is not None and path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    payload = SearchReplayStore.read_shard(path)
    assert payload["schema_version"] == SEARCH_REPLAY_SCHEMA_VERSION
    assert payload["records"][0]["value_target"] == 0.8
    records = load_search_records(tmp_path)
    assert len(records) == 1
    assert validate_search_record(records[0]) is records[0]


def test_bounded_window_removes_oldest_shards(tmp_path):
    store = SearchReplayStore(tmp_path, max_shards=2)
    for index in range(3):
        store.record_decision(f"g{index}", "p1", _record())
        store.finish_episode(f"g{index}", "p1", completed=True, value_target=float(index))
    assert len(store.shard_paths()) == 2
