"""Flat lookahead tests against a scripted simulation service."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.config import SearchConfig  # noqa: E402
from search.evaluator import EvalResult  # noqa: E402
from search.lookahead import decide_initial_cards, decide_lookahead  # noqa: E402
from search.simulator_client import BranchResult, SearchClient  # noqa: E402

ROOT_PLAYER = "p1"
OPPONENT = "p2"

CONT_DESCRIPTORS = [
    {"action_index": 200, "action_position": 0, "label": "opt-a", "decoded_action": {"type": "option"}},
    {"action_index": 201, "action_position": 1, "label": "opt-b", "decoded_action": {"type": "option"}},
]


def _descriptor(index: int, position: int, payload: dict) -> dict:
    return {
        "action_index": index,
        "action_position": position,
        "label": f"candidate-{index}",
        "decoded_action": payload,
    }


def _strategic_observation(tag: str) -> dict:
    return {
        "id": ROOT_PLAYER,
        "players": [{"id": ROOT_PLAYER}, {"id": OPPONENT}],
        "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
        "_tag": tag,
    }


class ScriptedClient(SearchClient):
    """Replays canned branch results and records every request batch."""

    def __init__(self, responder) -> None:
        super().__init__("http://fake:8080", token="t", timeout_sec=5.0)
        self.responder = responder
        self.batches: List[List[dict]] = []

    async def replay(self, session_id, branches):  # type: ignore[override]
        self.batches.append([dict(row) for row in branches])
        return self.responder(self.batches)


class ScriptedEvaluator:
    """Value per observation tag; continuation prompts expose two options."""

    def __init__(self, value_by_tag: dict, descriptors: Optional[List[dict]] = None) -> None:
        self.value_by_tag = value_by_tag
        self.descriptors = descriptors or CONT_DESCRIPTORS

    def clone_recurrent_map(self, player_ids):
        return {pid: torch.zeros(4) for pid in player_ids}

    def evaluate_batch(self, items):
        outcomes = []
        for item in items:
            tag = str(item.player_state.get("_tag", ""))
            value = self.value_by_tag.get(tag, 0.0)
            outcomes.append(
                EvalResult(
                    descriptors=list(self.descriptors),
                    probabilities=[0.5, 0.5],
                    value=float(value),
                    recurrent_out=torch.zeros(4),
                    phase_index=1,
                )
            )
        return outcomes

    def evaluate(self, item):
        return self.evaluate_batch([item])[0]


def _config(**overrides) -> SearchConfig:
    cfg = SearchConfig(enabled=True, mode="lookahead", top_k=3, determinizations=2, seed=7)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.normalize()
    return cfg


def _candidates() -> List[dict]:
    return [
        _descriptor(500, 0, {"type": "or", "index": 0, "response": {"type": "option"}}),
        _descriptor(501, 1, {"type": "or", "index": 1, "response": {"type": "option"}}),
        _descriptor(502, 2, {"type": "or", "index": 2, "response": {"type": "option"}}),
    ]


def _memory():
    return {ROOT_PLAYER: torch.zeros(4), OPPONENT: torch.zeros(4)}


async def _decide(responder, value_by_tag, cfg=None, candidates=None, priors=None, lowercase_mc=False):
    cfg = cfg or _config()
    candidates = candidates if candidates is not None else _candidates()
    priors = priors if priors is not None else [0.5, 0.3, 0.2]
    client = ScriptedClient(responder)
    return await decide_lookahead(
        agent=None,
        client=client,
        base_url="http://fake:8080",
        session_id="ses-1",
        root_player_id=ROOT_PLAYER,
        root_state=_strategic_observation("root"),
        candidates=candidates,
        priors=priors,
        branch_memory=_memory(),
        turn_counts={},
        config=cfg,
        lowercase_mc=lowercase_mc,
        evaluator=ScriptedEvaluator(value_by_tag),
    ), client


def test_best_mean_value_action_is_chosen():
    def responder(batches):
        results = []
        for branch in batches[-1]:
            position = int(branch["branchId"].split("-")[1])
            results.append(
                BranchResult(
                    branch_id=branch["branchId"],
                    status="next_prompt",
                    applied_steps=1,
                    state_digest="x",
                    next_prompt=(ROOT_PLAYER, _strategic_observation(f"leaf{position}")),
                )
            )
        return results

    outcome, client = asyncio.run(
        _decide(responder, {"leaf0": 0.1, "leaf1": 0.2, "leaf2": 0.9})
    )
    assert outcome.chosen_position == 2
    assert outcome.chosen_index == 502
    assert outcome.valid_samples == 6
    assert len(client.batches) == 1
    assert len(client.batches[0]) == 3 * 2
    by_id = {row["position"]: row for row in outcome.candidates}
    assert by_id[2]["mean_value"] == 0.9
    assert by_id[2]["visits"] == 2


def test_rollback_continuations_sample_opponent_policy_and_grow_path():
    def responder(batches):
        results = []
        for branch in batches[-1]:
            if len(branch["steps"]) == 1:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="next_prompt",
                        applied_steps=1,
                        state_digest="x",
                        next_prompt=(OPPONENT, {"id": OPPONENT, "players": [], "waitingFor": {"type": "option"}, "_tag": "opp"}),
                    )
                )
            else:
                position = int(branch["branchId"].split("-")[1])
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="next_prompt",
                        applied_steps=len(branch["steps"]),
                        state_digest="x",
                        next_prompt=(ROOT_PLAYER, _strategic_observation(f"leaf{position}")),
                    )
                )
        return results

    outcome, client = asyncio.run(
        _decide(responder, {"leaf0": 0.0, "leaf1": 0.0, "leaf2": 0.5})
    )
    assert outcome.chosen_position == 2
    assert len(client.batches) == 2
    second_round = client.batches[1]
    for branch in second_round:
        assert branch["steps"][0]["playerId"] == ROOT_PLAYER
        assert branch["steps"][1]["playerId"] == OPPONENT
        assert branch["steps"][1]["input"]["type"] == "option"


def test_rejected_and_boundary_samples_are_dropped_from_mean():
    def responder(batches):
        results = []
        for branch in batches[-1]:
            parts = branch["branchId"].split("-")
            position, det = int(parts[1]), int(parts[2])
            if det == 0:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="rejected",
                        applied_steps=0,
                        state_digest="x",
                        error_code="input_rejected",
                        error_step_index=0,
                        error_message="illegal",
                    )
                )
            else:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="next_prompt",
                        applied_steps=1,
                        state_digest="x",
                        next_prompt=(ROOT_PLAYER, _strategic_observation(f"leaf{position}")),
                    )
                )
        return results

    outcome, _ = asyncio.run(_decide(responder, {"leaf0": 1.0, "leaf1": 0.5, "leaf2": 0.0}))
    assert outcome.invalid_samples == 3
    assert outcome.valid_samples == 3
    for row in outcome.candidates:
        assert row["visits"] == 1
    assert outcome.chosen_position == 0


def test_terminal_branch_uses_v2_terminal_reward():
    def responder(batches):
        results = []
        for branch in batches[-1]:
            position = int(branch["branchId"].split("-")[1])
            if position == 2:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="terminal",
                        applied_steps=1,
                        state_digest="x",
                        terminal_players=[
                            {"playerId": ROOT_PLAYER, "rank": 1, "vp": 100},
                            {"playerId": OPPONENT, "rank": 2, "vp": 60},
                        ],
                    )
                )
            else:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="next_prompt",
                        applied_steps=1,
                        state_digest="x",
                        next_prompt=(ROOT_PLAYER, _strategic_observation(f"leaf{position}")),
                    )
                )
        return results

    outcome, _ = asyncio.run(_decide(responder, {"leaf0": 0.2, "leaf1": 0.2}))
    assert outcome.chosen_position == 2
    # v2 terminal reward: rank 1 -> 1.0 plus capped VP-margin bonus.
    assert outcome.candidates[2]["mean_value"] > 1.0
    assert outcome.valid_samples == 6


def test_initial_cards_score_the_root_view_and_do_not_continue():
    portfolios = [
        _descriptor(850, 0, {"type": "initialCards", "responses": [{"type": "card", "cards": ["Credicor"]}]}),
        _descriptor(851, 1, {"type": "initialCards", "responses": [{"type": "card", "cards": ["Thorgate"]}]}),
    ]

    def responder(batches):
        assert len(batches) == 1
        results = []
        for branch in batches[-1]:
            assert branch["mode"] == "exact"
            assert "determinizationSeed" not in branch
            assert len(branch["steps"]) == 1
            position = int(branch["branchId"].split("-")[1])
            results.append(
                BranchResult(
                    branch_id=branch["branchId"],
                    status="boundary",
                    applied_steps=1,
                    state_digest="x",
                    root_observation={"id": ROOT_PLAYER, "players": [], "waitingFor": None, "_tag": f"corp{position}"},
                )
            )
        return results

    cfg = _config(determinizations=8, top_k=2)
    client = ScriptedClient(responder)
    outcome = asyncio.run(
        decide_initial_cards(
            agent=None,
            client=client,
            base_url="http://fake:8080",
            session_id="ses-1",
            root_player_id=ROOT_PLAYER,
            root_state={"waitingFor": {"type": "initialCards"}},
            candidates=portfolios,
            priors=[0.8, 0.2],
            branch_memory=_memory(),
            turn_counts={},
            config=cfg,
            lowercase_mc=False,
            evaluator=ScriptedEvaluator({"corp0": 0.1, "corp1": 0.9}),
        )
    )
    assert outcome is not None
    assert outcome.mode == "portfolio"
    assert outcome.chosen_position == 1
    assert outcome.candidates[1]["mean_value"] == 0.9
    assert outcome.valid_samples == 2
    assert len(client.batches) == 1


def test_all_invalid_returns_no_signal():
    def responder(batches):
        return [
            BranchResult(
                branch_id=branch["branchId"],
                status="boundary",
                applied_steps=0,
                state_digest="x",
            )
            for branch in batches[-1]
        ]

    outcome, _ = asyncio.run(_decide(responder, {}))
    assert outcome.chosen_position == -1
    assert outcome.valid_samples == 0


def test_payment_payload_adapts_to_server_variant():
    def responder(batches):
        return [
            BranchResult(
                branch_id=branch["branchId"],
                status="next_prompt",
                applied_steps=1,
                state_digest="x",
                next_prompt=(ROOT_PLAYER, _strategic_observation(f"leaf{int(branch['branchId'].split('-')[1])}")),
            )
            for branch in batches[-1]
        ]

    candidates = [
        _descriptor(500, 0, {"type": "or", "index": 0, "response": {"type": "projectCard", "card": "X", "payment": {"megaCredits": 5}}}),
        _descriptor(501, 1, {"type": "or", "index": 1, "response": {"type": "option"}}),
        _descriptor(502, 2, {"type": "or", "index": 2, "response": {"type": "option"}}),
    ]
    _, client = asyncio.run(
        _decide(responder, {"leaf0": 0.9, "leaf1": 0.1, "leaf2": 0.1}, candidates=candidates, lowercase_mc=True)
    )
    first_step = client.batches[0][0]["steps"][0]["input"]
    payment = first_step["response"]["payment"]
    assert "megacredits" in payment
    assert "megaCredits" not in payment
    assert payment["heat"] == 0
