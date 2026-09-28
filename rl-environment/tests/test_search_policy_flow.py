"""End-to-end SearchPolicy.decide flow with fake agent, client, and evaluator."""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.config import SearchConfig  # noqa: E402
from search.evaluator import EvalResult  # noqa: E402
from search.search_agent import SearchPolicy  # noqa: E402
from search.simulator_client import (  # noqa: E402
    BranchResult,
    SearchClientStats,
    SearchUnavailableError,
    StartResult,
)

ROOT_PLAYER = "p1"


class FakeAgent:
    def __init__(self) -> None:
        self._turn_action_count_by_player = {ROOT_PLAYER: 1}
        self._recurrent_hidden_by_player = {ROOT_PLAYER: torch.ones(4)}

    def _extract_phase_index(self, _state):
        return 1

    def _get_recurrent_state_for_player(self, player_id):
        return torch.zeros(4)

    def _sync_forward_and_probs(self, planner_state, phase_index, recurrent_state):
        count = int(planner_state["action_tokens"].shape[0])
        probs = torch.tensor([[0.5, 0.3, 0.2][:count]], dtype=torch.float32)
        return (
            torch.log(probs.transpose(0, 1)),
            torch.tensor([[0.11]]),
            torch.full((4,), 0.5),
            None,
            probs,
        )

    def _compute_aux_targets(self, _state):
        return {}

    def _describe_action(self, action_index, _state):
        return f"act-{action_index}"


class FakeEvaluatorFactory:
    VALUE_MAP = {"leaf0": 0.1, "leaf1": 0.2, "leaf2": 0.9}

    def __init__(self, agent) -> None:
        self.agent = agent

    def clone_recurrent_map(self, player_ids):
        return {pid: torch.zeros(4) for pid in player_ids}

    def evaluate_batch(self, items):
        outcomes = []
        for item in items:
            tag = str(item.player_state.get("_tag", ""))
            value = FakeEvaluatorFactory.VALUE_MAP.get(tag, 0.0)
            outcomes.append(
                EvalResult(
                    descriptors=[{"action_index": 900, "action_position": 0, "decoded_action": {"type": "option"}}],
                    probabilities=[1.0],
                    value=float(value),
                    recurrent_out=torch.zeros(4),
                    phase_index=1,
                )
            )
        return outcomes


class FakePolicyClient:
    base_url = "http://fake:8080"
    instances: List["FakePolicyClient"] = []

    def __init__(self, base_url, token=None, timeout_sec=30.0, stats=None):
        self.stats = stats or SearchClientStats()
        self.closed_sessions: List[str] = []
        self.aclose_calls = 0
        FakePolicyClient.instances.append(self)

    async def start(self, player_id):
        return StartResult(
            session_id="ses-42",
            root_digest="a" * 64,
            root_player_id=player_id,
            observation={
                "id": player_id,
                "players": [{"id": player_id}, {"id": "p2"}],
                "thisPlayer": {"id": player_id, "megacredits": 10},
                "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
            },
            lowercase_mc=True,
        )

    async def replay(self, session_id, branches):
        results = []
        for branch in branches:
            results.append(
                BranchResult(
                    branch_id=branch["branchId"],
                    status="next_prompt",
                    applied_steps=1,
                    state_digest="x",
                    next_prompt=(ROOT_PLAYER, {"_tag": f"leaf{self._position(branch['branchId'])}", "waitingFor": {"type": "or", "options": [{}, {}]}},),
                )
            )
        return results

    @staticmethod
    def _position(branch_id) -> int:
        parts = str(branch_id).split("-")
        # la-<candidate>-<det> versus mcts-<sim>-<depth>-<candidate>
        if parts[0] == "mcts":
            return int(parts[3])
        return int(parts[1])

    async def close(self, session_id):
        self.closed_sessions.append(session_id)
        return True

    async def aclose(self):
        self.aclose_calls += 1


def _policy(cfg: SearchConfig) -> SearchPolicy:
    return SearchPolicy(FakeAgent(), cfg)


def _planner_state():
    return {
        "world_tokens": np.zeros((3, 64), dtype=np.float32),
        "hand_tokens": np.zeros((2, 64), dtype=np.float32),
        "action_tokens": np.zeros((3, 64), dtype=np.float32),
    }


def _descriptors() -> List[dict]:
    return [
        {"action_index": 500 + i, "action_position": i, "label": f"c{i}", "decoded_action": {"type": "or", "index": i}}
        for i in range(3)
    ]


def _run(policy: SearchPolicy):
    return asyncio.run(
        policy.decide(
            game_instance=type("G", (), {"base_url": "http://fake:8080"})(),
            player_id=ROOT_PLAYER,
            player_state={
                "id": ROOT_PLAYER,
                "players": [{"id": ROOT_PLAYER}, {"id": "p2"}],
                "thisPlayer": {"id": ROOT_PLAYER},
                "game": {"phase": "action"},
                "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
            },
            planner_state=_planner_state(),
            action_descriptors=_descriptors(),
            raw_available_actions=[500, 501, 502],
        )
    )


def _patched(fn):
    def runner():
        with patch("search.search_agent.SearchClient", FakePolicyClient), patch(
            "search.search_agent.PositionEvaluator", FakeEvaluatorFactory
        ):
            return fn()

    return runner


def test_lookahead_decision_metadata_shape():
    def body():
        FakePolicyClient.instances.clear()
        policy = _policy(SearchConfig(enabled=True, mode="lookahead", top_k=3, determinizations=2))
        decision = _run(policy)
        assert decision is not None
        assert decision.action_index == 502
        assert decision.decoded_action == {"type": "or", "index": 2}
        meta = decision.meta
        assert meta["action_source"] == "mcts-search"
        assert meta["sampled_from_policy"] is False
        assert meta["exclude_from_rollout"] is True
        assert meta["chosen_action_position"] == 2
        assert meta["value_old"] == pytest.approx(0.11)
        assert meta["logp_old"] == pytest.approx(math.log(0.2), abs=1e-6)
        assert meta["recurrent_state_out"] == [0.5, 0.5, 0.5, 0.5]
        assert meta["mcts"]["mode"] == "lookahead"
        assert meta["mcts"]["valid_samples"] == 6
        assert sum(meta["search_policy_target"]) == pytest.approx(1.0)
        assert len(meta["search_policy_target"]) == 3
        assert meta["chosen_action_position"] == 2
        assert policy.stats["searched"] == 1
        client = FakePolicyClient.instances[-1]
        assert client.closed_sessions == ["ses-42"]
        assert client.aclose_calls == 0
        asyncio.run(policy.aclose())
        assert client.aclose_calls == 1

    _patched(body)()


def test_puct_mode_dispatch():
    def body():
        FakePolicyClient.instances.clear()
        FakeEvaluatorFactory.VALUE_MAP = {"leaf0": 0.9, "leaf1": 0.2, "leaf2": 0.1}
        try:
            cfg = SearchConfig(
                enabled=True,
                mode="puct",
                top_k=3,
                determinizations=1,
                simulations_per_move=4,
                max_root_turns_depth=1,
                puct_c=0.0,
            )
            policy = _policy(cfg)
            decision = _run(policy)
            assert decision is not None
            assert decision.meta["mcts"]["mode"] == "puct"
            assert decision.action_index == 500
        finally:
            FakeEvaluatorFactory.VALUE_MAP = {"leaf0": 0.1, "leaf1": 0.2, "leaf2": 0.9}

    _patched(body)()


def test_unavailable_service_falls_back_to_policy():
    class UnavailableClient(FakePolicyClient):
        async def start(self, player_id):
            raise SearchUnavailableError("session_capacity", "full")

    def body():
        with patch("search.search_agent.SearchClient", UnavailableClient), patch(
            "search.search_agent.PositionEvaluator", FakeEvaluatorFactory
        ):
            policy = _policy(SearchConfig(enabled=True))
            decision = _run(policy)
        assert decision is None
        assert policy.stats["fallback_unavailable"] == 1

    body()


def test_non_strategic_server_root_falls_back():
    class BadRootClient(FakePolicyClient):
        async def start(self, player_id):
            result = await super().start(player_id)
            result.observation = {"waitingFor": {"type": "option"}}
            return result

    def body():
        with patch("search.search_agent.SearchClient", BadRootClient), patch(
            "search.search_agent.PositionEvaluator", FakeEvaluatorFactory
        ):
            policy = _policy(SearchConfig(enabled=True))
            decision = _run(policy)
        assert decision is None
        assert policy.stats["fallback_unavailable"] == 1

    body()


def test_snapshot_stats_shape():
    def body():
        policy = _policy(SearchConfig(enabled=True))
        stats = policy.snapshot_stats()
        for key in (
            "enabled",
            "mode",
            "searched_decisions",
            "search_fallbacks",
            "mean_replay_batch_sec",
            "applied_inputs_per_sec",
            "client_failures",
        ):
            assert key in stats

    _patched(body)()
