"""PUCT tree tests: selection math, expansion, determinism, rejection handling.

The tree persists only the searching player's own macro-actions; every
simulation determinizes and samples continuations freshly, rollout failures
cost an edge a consecutive failure without polluting Q, and the search loop
batches one replay call per round.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.config import SearchConfig  # noqa: E402
from search.evaluator import EvalResult  # noqa: E402
from search.mcts import decide_puct  # noqa: E402
from search.node import SearchEdge, SearchNode  # noqa: E402
from search.simulator_client import BranchResult, SearchClient  # noqa: E402

ROOT_PLAYER = "p1"


def _descriptor(index: int) -> dict:
    return {"action_index": index, "action_position": 0, "label": f"a{index}", "decoded_action": {"type": "option"}}


class FakeClient(SearchClient):
    def __init__(self, responder) -> None:
        super().__init__("http://fake:8080", token="t", timeout_sec=5.0)
        self.responder = responder
        self.calls: List[int] = []
        self.last_steps: List[int] = []

    async def replay(self, session_id, branches):  # type: ignore[override]
        self.calls.append(len(branches))
        self.last_steps = [len(branch["steps"]) for branch in branches]
        return self.responder(branches)


class FakeEvaluator:
    def __init__(self, value_by_tag: dict) -> None:
        self.value_by_tag = value_by_tag

    def clone_recurrent_map(self, player_ids):
        return {pid: torch.zeros(4) for pid in player_ids}

    def evaluate_batch(self, items):
        outcomes = []
        for item in items:
            tag = str(item.player_state.get("_tag", ""))
            outcomes.append(
                EvalResult(
                    descriptors=[_descriptor(900), _descriptor(901)],
                    probabilities=[0.6, 0.4],
                    value=float(self.value_by_tag.get(tag, 0.0)),
                    recurrent_out=torch.zeros(4),
                    phase_index=1,
                )
            )
        return outcomes

    def evaluate(self, item):
        return self.evaluate_batch([item])[0]


def _strategic(tag: str) -> dict:
    return {
        "id": ROOT_PLAYER,
        "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
        "_tag": tag,
    }


def _leaf_responder(value_by_tag: dict, reject_positions=(), reject_sims=()):
    """Position comes from the branch id; depth is the number of own steps."""

    def responder(branches):
        results = []
        for branch in branches:
            parts = str(branch["branchId"]).split("-")
            position = int(parts[3]) if len(parts) > 3 else 0
            depth = len(branch["steps"])
            sim_index = int(parts[1]) if len(parts) > 1 else 0
            if (depth, position) in reject_positions or (sim_index, position) in reject_sims:
                results.append(
                    BranchResult(
                        branch_id=branch["branchId"],
                        status="rejected",
                        applied_steps=0,
                        state_digest="x",
                        error_code="input_rejected",
                        error_step_index=0,
                        error_message="illegal under this determinization",
                    )
                )
                continue
            results.append(
                BranchResult(
                    branch_id=branch["branchId"],
                    status="next_prompt",
                    applied_steps=len(branch["steps"]),
                    state_digest="x",
                    next_prompt=(ROOT_PLAYER, _strategic(f"d{depth}p{position}")),
                )
            )
        return results

    return responder


def _run(value_by_tag, config, reject_positions=(), reject_sims=(), candidates=None, priors=(0.6, 0.4)):
    candidates = candidates if candidates is not None else [_descriptor(100), _descriptor(200)]
    client = FakeClient(_leaf_responder(value_by_tag, reject_positions, reject_sims))
    outcome = asyncio.run(
        decide_puct(
            agent=None,
            client=client,
            base_url="http://fake:8080",
            session_id="ses-1",
            root_player_id=ROOT_PLAYER,
            root_state=_strategic("root"),
            candidates=candidates,
            priors=list(priors),
            root_value=0.0,
            branch_memory={ROOT_PLAYER: torch.zeros(4)},
            turn_counts={},
            config=config,
            lowercase_mc=False,
            evaluator=FakeEvaluator(value_by_tag),
        )
    )
    setattr(outcome, "_client", client) if outcome is not None else None
    return outcome, client


# ---------------------------------------------------------------------------
# node math
# ---------------------------------------------------------------------------


def test_puct_score_prefers_high_exploration_for_unvisited_edges():
    node = SearchNode(depth=0, path=[], observation={}, value_estimate=0.0)
    hot = SearchEdge(0, 100, "hot", prior=0.4, first_step={"playerId": "p1", "input": {"type": "option"}})
    hot.visits, hot.value_sum = 10, 8.0
    cold = SearchEdge(1, 200, "cold", prior=0.6, first_step={"playerId": "p1", "input": {"type": "option"}})
    node.edges = [hot, cold]
    node.visits, node.value_sum = 10, 8.0
    # c * P * sqrt(10) / (1 + 0) dominates Q(=0.8) difference.
    selected = node.puct(c_puct=1.5, fpu=0.0)
    assert selected is cold


def test_puct_zero_constant_is_pure_exploitation():
    node = SearchNode(depth=0, path=[], observation={}, value_estimate=0.0)
    good = SearchEdge(0, 100, "good", prior=0.1, first_step={"playerId": "p1", "input": {}})
    good.visits, good.value_sum = 3, 2.4
    bad = SearchEdge(1, 200, "bad", prior=0.9, first_step={"playerId": "p1", "input": {}})
    bad.visits, bad.value_sum = 3, 0.3
    node.edges = [good, bad]
    assert node.puct(c_puct=0.0) is good


def test_select_guarantees_first_visit_coverage():
    node = SearchNode(depth=0, path=[], observation={}, value_estimate=-0.5)
    hot = SearchEdge(0, 100, "hot", prior=0.9, first_step={"playerId": "p1", "input": {}})
    hot.visits, hot.value_sum = 10, 9.0
    cold = SearchEdge(1, 200, "cold", prior=0.1, first_step={"playerId": "p1", "input": {}})
    node.edges = [hot, cold]
    node.visits, node.value_sum = 10, 9.0
    # Even pure exploitation must visit the unexplored edge first.
    assert node.select(c_puct=0.0) is cold
    cold.visits = 1
    assert node.select(c_puct=0.0) is hot


def test_puct_unvisited_edges_use_fpu_not_zero():
    node = SearchNode(depth=0, path=[], observation={}, value_estimate=0.5)
    visited = SearchEdge(0, 100, "visited", prior=0.5, first_step={"playerId": "p1", "input": {}})
    visited.visits, visited.value_sum = 4, 0.8  # Q = 0.2
    fresh = SearchEdge(1, 200, "fresh", prior=0.01, first_step={"playerId": "p1", "input": {}})
    node.edges = [visited, fresh]
    node.visits = 4
    # FPU 0.5 with a tiny prior must beat Q 0.2 with no exploration bonus.
    assert node.puct(c_puct=0.0, fpu=0.5) is fresh


def test_dead_edges_are_never_selected():
    node = SearchNode(depth=0, path=[], observation={}, value_estimate=0.0)
    dead = SearchEdge(0, 100, "dead", prior=0.9, first_step={"playerId": "p1", "input": {}})
    dead.dead = True
    live = SearchEdge(1, 200, "live", prior=0.05, first_step={"playerId": "p1", "input": {}})
    node.edges = [dead, live]
    assert node.puct(c_puct=1.5) is live
    assert node.visit_distribution()[0]["dead"] is True


def test_failures_kill_only_after_the_cap():
    edge = SearchEdge(0, 100, "edge", prior=0.5, first_step={"playerId": "p1", "input": {}})
    assert edge.record_failure(3) is False
    assert edge.record_failure(3) is False
    assert edge.dead is False
    assert edge.record_failure(3) is True
    assert edge.dead is True
    assert edge.q == 0.0  # failures never write into value_sum


def test_tree_serializes_nested_child_edges():
    root = SearchNode(depth=0, path=[], observation={}, value_estimate=0.0)
    edge = SearchEdge(0, 100, "line", prior=0.5, first_step={"playerId": "p1", "input": {}})
    child = SearchNode(depth=1, path=[{"playerId": "p1", "input": {}}], observation={}, value_estimate=0.3)
    child.edges = [
        SearchEdge(0, 300, "grand", prior=0.7, first_step={"playerId": "p1", "input": {}}),
    ]
    edge.target = child
    edge.visits, edge.value_sum = 5, 1.5
    root.edges = [edge]
    tree = root.tree()
    assert tree[0]["visits"] == 5
    assert tree[0]["children"][0]["action_index"] == 300
    # Depth cap hides grandchildren beyond max_depth levels.
    assert "children" not in tree[0]["children"][0]


# ---------------------------------------------------------------------------
# full searches
# ---------------------------------------------------------------------------


def test_exploitation_search_converges_on_better_line():
    outcome, _client = _run({"d1p0": 0.9, "d1p1": 0.1}, _config())
    assert outcome is not None
    assert outcome.chosen_index == 100
    assert outcome.root_visits == 8
    root_row = next(row for row in outcome.candidates if row["action_index"] == 100)
    assert root_row["visits"] >= 6  # the other edge keeps its forced-coverage visit
    other_row = next(row for row in outcome.candidates if row["action_index"] == 200)
    assert other_row["visits"] >= 1


def test_search_batches_replays_per_round():
    outcome, client = _run({"d1p0": 0.9, "d1p1": 0.1}, _config())
    assert outcome is not None
    # Sequential sims used to be one replay call each; rounds batch them.
    assert len(client.calls) < outcome.simulations_selected
    assert max(client.calls) >= 2


def test_leaf_batch_caps_each_selection_round():
    outcome, client = _run(
        {"d1p0": 0.9, "d1p1": 0.1},
        _config(simulations_per_move=8, leaf_batch=3),
    )
    assert outcome is not None
    assert max(client.calls) <= 3
    assert outcome.simulations_selected == 8


def test_early_stop_when_visit_winner_cannot_be_caught():
    outcome, _ = _run(
        {"d1p0": 0.9, "d1p1": 0.1},
        _config(simulations_per_move=20, leaf_batch=2, early_stop=True, early_stop_min_simulations=4),
    )
    assert outcome is not None
    assert outcome.early_stopped is True
    assert outcome.simulations_selected < outcome.simulation_budget


def test_search_is_deterministic_for_same_seed():
    first, _ = _run({"d1p0": 0.5, "d1p1": 0.25}, _config(puct_c=1.5))
    second, _ = _run({"d1p0": 0.5, "d1p1": 0.25}, _config(puct_c=1.5))
    assert first is not None and second is not None
    assert [row["visits"] for row in first.candidates] == [row["visits"] for row in second.candidates]
    assert [row["prior"] for row in first.candidates] == [row["prior"] for row in second.candidates]


def test_root_noise_perturbs_priors_without_touching_raw():
    outcome, _ = _run({"d1p0": 0.5, "d1p1": 0.25}, _config(root_noise_alpha=0.3, root_noise_weight=0.5))
    assert outcome is not None
    for row in outcome.candidates:
        assert row["prior_raw"] is not None
        assert 0.0 <= row["prior"] <= 1.0
    assert any(row["prior"] != row["prior_raw"] for row in outcome.candidates)


def test_root_noise_disabled_keeps_priors():
    outcome, _ = _run({"d1p0": 0.5, "d1p1": 0.25}, _config(root_noise_alpha=0.0))
    assert outcome is not None
    for row in outcome.candidates:
        assert row["prior"] == row["prior_raw"]


def test_depth_two_expands_child_nodes_and_backprops():
    outcome, _ = _run(
        {"d1p0": 0.3, "d1p1": 0.3, "d2p0": 0.8, "d2p1": 0.2},
        _config(max_root_turns_depth=2, puct_c=0.0, simulations_per_move=6),
    )
    assert outcome is not None
    assert outcome.root_visits == 6
    assert sum(row["visits"] for row in outcome.candidates) == 6
    best = next(row for row in outcome.candidates if row["action_index"] == 100)
    assert best["children"], "depth-2 child edges must expand and match by label"
    assert outcome.chosen_index == 100


def test_rejected_line_is_killed_and_traffic_moves():
    outcome, _ = _run(
        {"d1p1": 0.4},
        _config(puct_c=0.0, edge_kill_failures=1),
        reject_positions={(1, 0)},
    )
    assert outcome is not None
    dead_row = next(row for row in outcome.candidates if row["action_index"] == 100)
    live_row = next(row for row in outcome.candidates if row["action_index"] == 200)
    assert dead_row["dead"] is True
    assert outcome.root_visits == 7  # the invalid sim is not counted as a visit
    assert live_row["visits"] == outcome.root_visits
    assert live_row["q"] > 0.0
    assert outcome.invalid_rollouts == 1
    assert outcome.killed_edges == 1
    assert outcome.chosen_index == 200


def test_transient_rejection_does_not_pollute_value_and_success_resets_failure_streak():
    # The coverage round visits edges prior-first: edge 0 (prior 0.6) is the
    # first slot. Its first sim is rejected; the kill cap must absorb it and
    # Q must reflect only the later successful rollouts, not a parent-mean
    # backup.
    outcome, _ = _run(
        {"d1p0": 0.6, "d1p1": 0.1},
        _config(puct_c=0.0, edge_kill_failures=3, simulations_per_move=6),
        reject_sims={(0, 0)},
    )
    assert outcome is not None
    edge0 = next(row for row in outcome.candidates if row["action_index"] == 100)
    assert edge0["dead"] is False
    assert edge0["failures"] == 0
    assert edge0["visits"] >= 1
    assert edge0["q"] == edge0["q"] and edge0["q"] >= 0.5
    assert outcome.killed_edges == 0


def test_every_line_dead_returns_no_signal():
    outcome, _ = _run({}, _config(edge_kill_failures=1), reject_positions={(1, 0), (1, 1)})
    assert outcome is not None
    assert outcome.chosen_position == -1
    assert outcome.invalid_rollouts == 2
    assert outcome.killed_edges == 2


def test_no_candidates_returns_no_signal():
    outcome, _ = _run(
        {"d1p0": 0.5},
        _config(),
        candidates=[{"action_index": 100, "action_position": 0, "label": "x", "decoded_action": None}],
    )
    assert outcome is not None
    assert outcome.chosen_position == -1
    assert outcome.simulations_selected == 0


def test_adaptive_early_stop_triggers_on_small_budget():
    cfg = _config(
        simulations_per_move=8,
        leaf_batch=1,
        early_stop=True,
        early_stop_min_simulations=16,  # Should adapt down to budget // 2 = 4
        puct_c=0.0,
    )
    outcome, _ = _run(
        {"d1p0": 0.9, "d1p1": -0.9},
        cfg,
        priors=(0.8, 0.2),
    )
    assert outcome is not None
    assert outcome.early_stopped is True
    assert outcome.simulations_selected < 8


def _config(**overrides) -> SearchConfig:
    cfg = SearchConfig(
        enabled=True,
        mode="puct",
        top_k=2,
        determinizations=1,
        simulations_per_move=8,
        max_root_turns_depth=1,
        puct_c=0.0,
        seed=3,
        root_noise_alpha=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.normalize()
    return cfg
