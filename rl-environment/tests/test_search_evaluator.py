"""Evaluator tests: priors normalization, live-memory isolation, batch parity."""
from __future__ import annotations

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
from search.evaluator import EvalItem, PositionEvaluator  # noqa: E402


def _make_agent() -> RLAgent:
    with patch(
        "models.agent.require_backend_info",
        return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
    ):
        agent = RLAgent(AgentConfig())
    return agent


def _descriptor(index: int, position: int) -> dict:
    return {
        "action_index": index,
        "action_position": position,
        "family": "select_option",
        "label": f"action-{index}",
        "decoded_action": {"type": "or", "index": index},
        "token_features": np.zeros((PLANNER_TOKEN_DIM,), dtype=np.float32),
    }


def _bundle(action_indices: list[int]) -> dict:
    count = len(action_indices)
    return {
        "world_tokens": np.zeros((2, PLANNER_TOKEN_DIM), dtype=np.float32),
        "world_token_types": np.asarray([1, 2], dtype=np.int64),
        "world_mask": np.asarray([True, True], dtype=np.bool_),
        "hand_tokens": np.zeros((0, PLANNER_TOKEN_DIM), dtype=np.float32),
        "hand_mask": np.zeros((0,), dtype=np.bool_),
        "action_tokens": np.zeros((count, PLANNER_TOKEN_DIM), dtype=np.float32),
        "action_mask": np.ones((count,), dtype=np.bool_),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
        "action_positions": np.arange(count, dtype=np.int64),
        "global_scalars": np.zeros((PLANNER_GLOBAL_DIM,), dtype=np.float32),
        "terminal": False,
    }


def _stub_agent(agent: RLAgent, action_indices: list[int]) -> None:
    descriptors = [_descriptor(index, position) for position, index in enumerate(action_indices)]
    agent.action_decoder = type(
        "DecoderStub",
        (),
        {"get_legal_action_descriptors": lambda self, _state: list(descriptors)},
    )()
    agent.state_encoder = type(
        "EncoderStub",
        (),
        {"encode": lambda self, _state, _turn=0, _descriptors=None: _bundle(action_indices)},
    )()
    agent._extract_phase_index = lambda _state: 1


def test_priors_are_renormalized_over_legal_actions():
    agent = _make_agent()
    _stub_agent(agent, [3, 17, 42])
    evaluator = PositionEvaluator(agent)
    item = EvalItem(player_state={"waitingFor": {"type": "or", "options": [{}, {}]}}, player_id="p1")
    result = evaluator.evaluate(item)
    assert len(result.probabilities) == 3
    assert abs(sum(result.probabilities) - 1.0) < 1e-4
    assert all(0.0 <= p <= 1.0 for p in result.probabilities)
    assert [d["action_index"] for d in result.descriptors] == [3, 17, 42]
    assert isinstance(result.value, float)
    assert result.recurrent_out is not None
    assert int(result.recurrent_out.numel()) == int(agent.network.recurrent_size)
    assert not result.recurrent_out.requires_grad


def test_evaluate_batch_matches_single_item_forwards():
    agent = _make_agent()
    _stub_agent(agent, [3, 17, 42])
    evaluator = PositionEvaluator(agent)
    state = {"waitingFor": {"type": "or", "options": [{}, {}]}}
    rec = torch.ones(int(agent.network.recurrent_size))
    items = [
        EvalItem(player_state=state, player_id="p1", recurrent_in=rec),
        EvalItem(player_state=state, player_id="p1", recurrent_in=rec),
    ]
    batch = evaluator.evaluate_batch(items)
    single = evaluator.evaluate(items[0])
    assert batch[0] is not None and batch[1] is not None
    assert abs(batch[0].value - single.value) < 1e-5
    assert abs(batch[1].value - single.value) < 1e-5
    for a, b in zip(batch[0].probabilities, single.probabilities):
        assert abs(a - b) < 1e-5


def test_live_recurrent_memory_is_never_mutated():
    agent = _make_agent()
    _stub_agent(agent, [5, 6])
    evaluator = PositionEvaluator(agent)
    size = int(agent.network.recurrent_size)
    live = torch.ones(size)
    agent._recurrent_hidden_by_player["pA"] = live
    memory = evaluator.clone_recurrent_map(["pA", "pB"])
    assert torch.allclose(memory["pA"], torch.ones(size))
    assert torch.allclose(memory["pB"], torch.zeros(size))
    # Mutating the branch copy must not touch the live map.
    memory["pA"][0] = 99.0
    assert float(agent._recurrent_hidden_by_player["pA"][0]) == 1.0
    item = EvalItem(
        player_state={"waitingFor": {"type": "or", "options": [{}, {}]}},
        player_id="pA",
        recurrent_in=memory["pA"],
    )
    result = evaluator.evaluate(item)
    assert result is not None
    assert float(agent._recurrent_hidden_by_player["pA"][0]) == 1.0
    PositionEvaluator.update_memory(memory, "pA", result.recurrent_out)
    assert not torch.allclose(memory["pA"], torch.ones(size))


def test_branches_with_illegal_prompt_are_dropped_not_raised():
    agent = _make_agent()
    agent.action_decoder = type(
        "DecoderStub",
        (),
        {"get_legal_action_descriptors": lambda self, _state: []},
    )()
    evaluator = PositionEvaluator(agent)
    outcomes = evaluator.evaluate_batch(
        [EvalItem(player_state={"waitingFor": {"type": "option"}}, player_id="p1")]
    )
    assert outcomes == [None]


def test_forward_exception_falls_back_to_per_item_without_killing_batch():
    agent = _make_agent()
    _stub_agent(agent, [3, 17])
    evaluator = PositionEvaluator(agent)
    state = {"waitingFor": {"type": "or", "options": [{}, {}]}}
    items = [EvalItem(player_state=state, player_id="p1"), EvalItem(player_state=state, player_id="p1")]
    original = evaluator._forward_batch
    calls = {"n": 0}

    def flaky(rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory simulation")
        return original(rows)

    evaluator._forward_batch = flaky
    outcomes = evaluator.evaluate_batch(items)
    assert all(outcome is not None for outcome in outcomes)
