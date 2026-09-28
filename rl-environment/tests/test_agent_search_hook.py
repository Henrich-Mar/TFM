"""RLAgent._make_move search hook: use, fallback, and PPO exclusion."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.agent import AgentConfig, RLAgent  # noqa: E402
from models.planner_common import PLANNER_GLOBAL_DIM, PLANNER_TOKEN_DIM  # noqa: E402
from search.search_agent import SearchDecision  # noqa: E402


class _GameStub:
    game_id = "game-1"
    rl_seed = None

    def __init__(self):
        self.sent_actions = []

    async def send_player_input(self, _player_id, action_input):
        self.sent_actions.append(action_input)
        return True

    def peek_cached_state(self, _player_id):
        return None


def _make_agent() -> RLAgent:
    with patch(
        "models.agent.require_backend_info",
        return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
    ):
        agent = RLAgent(AgentConfig())
    agent.post_move_sleep_sec = 0.0
    agent.failure_pause_sec = 0.0
    agent.initial_cards_reject_pause_sec = 0.0
    agent.train_from_self_play = False
    agent.state_encoder.encode = lambda _state, _turn=0, _descriptors=None: {
        "world_tokens": np.zeros((2, PLANNER_TOKEN_DIM), dtype=np.float32),
        "world_token_types": np.asarray([1, 2], dtype=np.int64),
        "world_mask": np.asarray([True, True], dtype=np.bool_),
        "hand_tokens": np.zeros((0, PLANNER_TOKEN_DIM), dtype=np.float32),
        "hand_mask": np.zeros((0,), dtype=np.bool_),
        "action_tokens": np.zeros((2, PLANNER_TOKEN_DIM), dtype=np.float32),
        "action_mask": np.asarray([True, True], dtype=np.bool_),
        "action_indices": np.asarray([5, 9], dtype=np.int64),
        "action_positions": np.asarray([0, 1], dtype=np.int64),
        "global_scalars": np.zeros((PLANNER_GLOBAL_DIM,), dtype=np.float32),
    }
    agent.action_decoder = type(
        "DecoderStub",
        (),
        {
            "action_types": {"PASS": 900},
            "get_legal_action_descriptors": lambda self, _state: [
                {"action_index": 5, "action_position": 0, "decoded_action": {"type": "or", "index": 0}},
                {"action_index": 9, "action_position": 1, "decoded_action": {"type": "or", "index": 1}},
            ],
            "decode_action": lambda self, index, _state: {"type": "or", "index": int(index)},
        },
    )()
    return agent


_PLAYER_STATE = {
    "id": "p1",
    "thisPlayer": {"id": "p1"},
    "players": [{"id": "p1"}],
    "game": {"phase": "action"},
    "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
}


class _RecordingSearchPolicy:
    def __init__(self, decision=None, raises=None):
        self.calls = []
        self.decision = decision
        self.raises = raises

    async def decide(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.decision


def _search_decision() -> SearchDecision:
    meta = {
        "phase_index": 1,
        "recurrent_state": [0.0] * 4,
        "recurrent_state_out": [0.25] * 4,
        "aux_targets": {},
        "aux_predictions": [],
        "available_actions_raw": [5, 9],
        "available_actions_filtered": [5, 9],
        "action_descriptors": [],
        "chosen_action_position": 1,
        "chosen_action_label": "act-9",
        "sampled_from_policy": False,
        "action_source": "mcts-search",
        "exclude_from_rollout": True,
        "server_accepted": False,
        "fallback_used": False,
        "value_old": 0.4,
        "legal_actions": [5, 9],
        "logp_old": -1.0,
        "policy_temperature": 1.0,
        "mcts": {"mode": "lookahead", "valid_samples": 16},
    }
    return SearchDecision(
        decoded_action={"type": "or", "index": 1, "response": {"type": "option"}},
        action_index=9,
        chosen_position=1,
        meta=meta,
    )


def test_search_hit_uses_payload_and_skips_rollout_recording():
    agent = _make_agent()
    policy = _RecordingSearchPolicy(decision=_search_decision())
    agent.search_policy = policy
    game = _GameStub()
    episode_steps = []

    accepted = asyncio.run(agent._make_move(game, "p1", dict(_PLAYER_STATE), episode_steps))

    assert accepted is True
    assert game.sent_actions == [{"type": "or", "index": 1, "response": {"type": "option"}}]
    assert episode_steps == []
    assert int(agent.decision_stats.get("mcts_search_actions", 0)) == 1
    assert int(agent.decision_stats.get("epsilon_random_actions", 0)) == 0
    assert int(agent.decision_stats.get("policy_sampled_actions", 0)) == 0
    stored = agent._recurrent_hidden_by_player.get("p1")
    assert stored is not None
    assert torch.allclose(stored, torch.full_like(stored, 0.25))
    assert policy.calls[0]["player_id"] == "p1"
    assert [row["action_index"] for row in policy.calls[0]["action_descriptors"]] == [5, 9]


def test_search_decline_falls_back_to_network():
    agent = _make_agent()
    agent.search_policy = _RecordingSearchPolicy(decision=None)
    sentinel = {"type": "or", "index": 0}

    async def _fake_network(*_args, **_kwargs):
        return dict(sentinel), 5, True, {"available_actions_raw": [5]}

    agent._get_action_from_network = _fake_network
    game = _GameStub()
    episode_steps = []

    accepted = asyncio.run(agent._make_move(game, "p1", dict(_PLAYER_STATE), episode_steps))

    assert accepted is True
    assert game.sent_actions == [sentinel]
    assert len(episode_steps) == 1
    assert episode_steps[0]["action_source"] == "policy"


def test_search_exception_falls_back_to_network():
    agent = _make_agent()
    agent.search_policy = _RecordingSearchPolicy(raises=RuntimeError("simulator exploded"))
    sentinel = {"type": "or", "index": 1}

    async def _fake_network(*_args, **_kwargs):
        return dict(sentinel), 9, True, {"available_actions_raw": [9]}

    agent._get_action_from_network = _fake_network
    game = _GameStub()

    accepted = asyncio.run(agent._make_move(game, "p1", dict(_PLAYER_STATE), []))

    assert accepted is True
    assert game.sent_actions == [sentinel]


def test_incomplete_search_decision_is_ignored():
    agent = _make_agent()
    decision = _search_decision()
    decision.meta = {}  # malformed contract
    game = _GameStub()

    async def _fake_network(*_args, **_kwargs):
        return {"type": "or", "index": 0}, 5, True, {"available_actions_raw": [5]}

    agent._get_action_from_network = _fake_network
    agent.search_policy = _RecordingSearchPolicy(decision=decision)

    accepted = asyncio.run(agent._make_move(game, "p1", dict(_PLAYER_STATE), []))

    assert accepted is True
    assert game.sent_actions == [{"type": "or", "index": 0}]


def test_no_search_policy_is_a_noop():
    agent = _make_agent()

    async def _fake_network(*_args, **_kwargs):
        return {"type": "or", "index": 0}, 5, True, {"available_actions_raw": [5]}

    agent._get_action_from_network = _fake_network
    game = _GameStub()
    accepted = asyncio.run(agent._make_move(game, "p1", dict(_PLAYER_STATE), []))
    assert accepted is True
