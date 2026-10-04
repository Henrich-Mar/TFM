from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.agent import RLAgent  # noqa: E402
from search.replay_store import SearchReplayStore  # noqa: E402
from training.search_distill import validate_search_record  # noqa: E402


def _bundle(actions: int = 3) -> dict:
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


def _teacher_meta(chosen_index: int = 601, *, with_scores: bool = True) -> dict:
    """Mirror of the meta produced by the external decision-policy branch."""
    descriptors = [
        {"family": "play_card", "action_index": 1, "action_position": 0},
        {"family": "fund_award", "action_index": 601, "action_position": 1},
        {"family": "pass", "action_index": 800, "action_position": 2},
    ]
    meta = {
        "action_source": "teacher",
        "phase_index": 2,
        "recurrent_state": [0.0] * 4,
        "action_descriptors": descriptors,
        "legal_actions": [1, 601, 800],
        "chosen_action_index": chosen_index,
        "chosen_action_position": 1,
        "planner_bundle": _bundle(len(descriptors)),
    }
    if with_scores:
        meta["external_policy"] = {
            "scores": [
                {"action_position": 0, "action_index": 1, "probability": 0.2},
                {"action_position": 1, "action_index": 601, "probability": 0.6},
                {"action_position": 2, "action_index": 800, "probability": 0.2},
            ]
        }
    return meta


def _frozen_teacher(tmp_path: Path) -> RLAgent:
    agent = RLAgent(agent_id="teacher-test")
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.teacher_replay_store = SearchReplayStore(tmp_path)
    return agent


def test_teacher_award_decision_is_donated(tmp_path):
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    agent._maybe_stage_teacher_donation("g1", "p1", _teacher_meta(chosen_index=601))
    assert store.pending_count() == 1

    path = store.finish_episode("g1", "p1", completed=True, value_target=1.0, outcome={"completed": True})
    assert path is not None
    payload = SearchReplayStore.read_shard(path)
    record = payload["records"][0]
    validate_search_record(record)
    assert record["donation_source"] == "teacher"
    assert record["donation_family"] == "fund_award"
    # Teacher posterior is preserved rather than collapsed to one-hot.
    assert record["policy_target"] == [0.2, 0.6, 0.2]
    assert record["chosen_action_index"] == 601


def test_donation_falls_back_to_one_hot_without_scores(tmp_path):
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    agent._maybe_stage_teacher_donation("g1", "p1", _teacher_meta(with_scores=False))
    path = store.finish_episode("g1", "p1", completed=True, value_target=0.0, outcome={"completed": True})
    record = SearchReplayStore.read_shard(path)["records"][0]
    validate_search_record(record)
    assert record["policy_target"] == [0.0, 1.0, 0.0]
    assert record["donation_family"] == "fund_award"


def test_learner_never_donates(tmp_path):
    """The learner's own data already reaches PPO; donating it would double-count."""
    agent = _frozen_teacher(tmp_path)
    agent.train_from_self_play = True
    store = agent.teacher_replay_store
    agent._maybe_stage_teacher_donation("g1", "p1", _teacher_meta())
    assert store.pending_count() == 0


def test_non_teacher_source_is_not_donated(tmp_path):
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    meta = _teacher_meta()
    meta["action_source"] = "policy"
    agent._maybe_stage_teacher_donation("g1", "p1", meta)
    assert store.pending_count() == 0


def test_target_width_must_match_action_tokens(tmp_path):
    """A misaligned target would silently train the wrong action slot."""
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    meta = _teacher_meta()
    meta["planner_bundle"] = _bundle(7)  # token/action mismatch
    agent._maybe_stage_teacher_donation("g1", "p1", meta)
    assert store.pending_count() == 0


def test_incomplete_episode_is_discarded(tmp_path):
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    agent._maybe_stage_teacher_donation("g1", "p1", _teacher_meta())
    assert store.finish_episode("g1", "p1", completed=False) is None
    assert store.pending_count() == 0


def test_chosen_position_resolved_by_action_index(tmp_path):
    """action_position is not required to equal the descriptor list index."""
    agent = _frozen_teacher(tmp_path)
    store = agent.teacher_replay_store
    meta = _teacher_meta(chosen_index=800)
    meta["chosen_action_position"] = 99  # deliberately wrong list index
    agent._maybe_stage_teacher_donation("g1", "p1", meta)
    path = store.finish_episode("g1", "p1", completed=True, value_target=0.0, outcome={})
    record = SearchReplayStore.read_shard(path)["records"][0]
    validate_search_record(record)
    assert record["chosen_action_position"] == 2
    assert record["donation_family"] == "pass"