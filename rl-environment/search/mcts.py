"""Phase B: batched PUCT tree search over root-turn macro-actions.

Each round selects up to ``min(simulations_left, 64)`` root edges (guaranteed
first-visit coverage first, then PUCT with round-provisional visits), rolls
them out concurrently as one replay batch, and backs real leaf/terminal values
up the trail.  Only the searching player's own macro-actions persist in the
tree: every simulation determinizes freshly and re-samples opponent
continuations, so a recorded opponent path is never replayed under a different
hidden deal.  Depth-2 descent matches a child edge to the current leaf
descriptors by action payload, falling back to a unique label; an unmatched
child edge simply ends the branch at the leaf value.  Rollout failures cost
the deepest trail edge a consecutive failure and never enter Q; only
``edge_kill_failures`` consecutive failures kill an edge.  A generation
boundary or step-cap stop that already has a root-player value is backed up
as a leaf instead of a failure.  Dirichlet noise on
the root priors keeps exploration alive against degenerate policies.
"""
from __future__ import annotations

import json
import logging
import math
import random
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import SearchConfig
from .evaluator import PositionEvaluator
from .node import SearchEdge, SearchNode
from .rollout import Branch, run_batched
from .simulator_client import SearchClient, adapt_step_input

logger = logging.getLogger("rl.search")


@dataclass
class MctsOutcome:
    chosen_position: int
    chosen_index: int
    root_visits: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    invalid_rollouts: int = 0
    simulations_selected: int = 0
    valid_rollouts: int = 0
    killed_edges: int = 0
    simulation_budget: int = 0
    early_stopped: bool = False
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    bootstrapped: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        selected = max(1, int(self.simulations_selected))
        return {
            "mode": "puct",
            "simulations": int(self.root_visits),
            "simulations_selected": int(self.simulations_selected),
            "valid_rollouts": int(self.valid_rollouts),
            "invalid_rollouts": int(self.invalid_rollouts),
            "invalid_rate": round(float(self.invalid_rollouts) / selected, 4),
            "killed_edges": int(self.killed_edges),
            "simulation_budget": int(self.simulation_budget),
            "early_stopped": bool(self.early_stopped),
            "invalid_reasons": dict(self.invalid_reasons),
            "bootstrapped": dict(self.bootstrapped),
            "candidates": self.candidates,
        }


def _canonical_input(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(sorted((str(k), str(v)) for k, v in payload.items()))


def _candidate_edges(
    node: SearchNode,
    descriptors: List[Dict[str, Any]],
    probabilities: List[float],
    root_player_id: str,
    config: SearchConfig,
    lowercase_mc: bool,
) -> List[SearchEdge]:
    ranked = sorted(
        range(len(descriptors)),
        key=lambda position: probabilities[position] if position < len(probabilities) else 0.0,
        reverse=True,
    )[: int(config.top_k)]
    edges: List[SearchEdge] = []
    for position in ranked:
        descriptor = descriptors[position]
        payload = descriptor.get("decoded_action")
        if not isinstance(payload, dict) or not payload:
            continue
        adapted = adapt_step_input(dict(payload), lowercase_mc)
        prior = float(probabilities[position]) if position < len(probabilities) else 0.0
        edges.append(
            SearchEdge(
                position=int(position),
                action_index=int(descriptor.get("action_index", -1)),
                label=str(descriptor.get("label", "") or ""),
                prior=prior,
                prior_raw=prior,
                first_step={
                    "playerId": str(root_player_id),
                    "input": adapted,
                },
                action_key=_canonical_input(adapted),
            )
        )
    return edges


class MctsSearch:
    def __init__(
        self,
        agent: Any,
        client: SearchClient,
        base_url: str,
        session_id: str,
        root_player_id: str,
        config: SearchConfig,
        lowercase_mc: bool,
        evaluator: Optional[PositionEvaluator] = None,
    ) -> None:
        self.agent = agent
        self.client = client
        self.base_url = base_url
        self.session_id = str(session_id)
        self.root_player_id = str(root_player_id)
        self.config = config
        self.lowercase_mc = bool(lowercase_mc)
        self.evaluator = evaluator or PositionEvaluator(agent)
        self.invalid_rollouts = 0
        self.valid_rollouts = 0
        self.simulations_selected = 0
        self.killed_edges = 0
        self.invalid_reasons: Counter[str] = Counter()
        self.bootstrapped: Counter[str] = Counter()
        self.simulation_budget = 0
        self.early_stopped = False

    def build_root(
        self,
        root_state: Dict[str, Any],
        descriptors: List[Dict[str, Any]],
        probabilities: List[float],
        value: float,
        memory: Dict[str, Any],
        turn_counts: Dict[str, int],
        root_digest: str = "",
    ) -> SearchNode:
        root = SearchNode(
            depth=0,
            path=[],
            observation=root_state,
            value_estimate=float(value),
            memory={pid: tensor.clone() for pid, tensor in memory.items()},
            turn_counts=dict(turn_counts),
            state_digest=str(root_digest or ""),
        )
        root.edges = _candidate_edges(
            root,
            descriptors,
            probabilities,
            self.root_player_id,
            self.config,
            self.lowercase_mc,
        )
        return root

    # ------------------------------------------------------------------
    # root exploration noise
    # ------------------------------------------------------------------

    def apply_root_noise(self, root: SearchNode) -> None:
        cfg = self.config
        alpha = float(cfg.root_noise_alpha)
        weight = float(cfg.root_noise_weight)
        if alpha <= 0.0 or weight <= 0.0 or len(root.edges) < 2:
            return
        key = f"{self.session_id}:{self.root_player_id}:{cfg.seed}".encode("utf-8", errors="replace")
        rng = random.Random(zlib.crc32(key))
        draws = [max(1e-9, rng.gammavariate(alpha, 1.0)) for _ in root.edges]
        total = sum(draws)
        for edge, draw in zip(root.edges, draws):
            raw = edge.prior_raw if edge.prior_raw is not None else edge.prior
            edge.prior = (1.0 - weight) * float(raw) + weight * float(draw) / total

    # ------------------------------------------------------------------
    # batched search loop
    # ------------------------------------------------------------------

    async def search(self, root: SearchNode, simulation_budget: Optional[int] = None) -> None:
        self.apply_root_noise(root)
        cfg = self.config
        budget = int(cfg.simulations_per_move if simulation_budget is None else simulation_budget)
        self.simulation_budget = max(0, budget)
        remaining = self.simulation_budget
        while remaining > 0:
            for edge in root.edges:
                edge.probes = 0
            root.probes = 0
            live = root.live_edges()
            if not live:
                break
            unvisited = [
                edge for edge in live if int(edge.visits) + int(edge.probes) == 0
            ]
            # Until every live edge has a real value, keep rounds fair (one
            # pick per edge); afterwards spend whole batches on the PUCT
            # choice instead of one sequential replay per simulation.
            if unvisited:
                round_slots = min(remaining, len(live), int(cfg.leaf_batch))
            else:
                round_slots = min(remaining, int(cfg.leaf_batch))
            picks: List[SearchEdge] = []
            for _slot in range(round_slots):
                edge = self._select(root)
                if edge is None:
                    break
                picks.append(edge)
            if not picks:
                break
            branches = [
                self._edge_branch(root, edge, self.simulations_selected + index)
                for index, edge in enumerate(picks)
            ]
            self.simulations_selected += len(picks)
            await run_batched(
                agent=self.agent,
                client=self.client,
                base_url=self.base_url,
                session_id=self.session_id,
                root_player_id=self.root_player_id,
                branches=branches,
                config=cfg,
                lowercase_mc=self.lowercase_mc,
                evaluator=self.evaluator,
            )
            self._backprop(branches)
            remaining -= len(picks)
            if self._winner_is_locked(root, remaining):
                self.early_stopped = True
                break

    def _winner_is_locked(self, root: SearchNode, remaining: int) -> bool:
        """Stop only when no allocation of the remaining visits can change first place."""
        cfg = self.config
        min_sims = min(int(cfg.early_stop_min_simulations), max(4, self.simulation_budget // 2))
        if not cfg.early_stop or remaining <= 0 or root.visits < min_sims:
            return False
        live = sorted(root.live_edges(), key=lambda edge: int(edge.visits), reverse=True)
        if not live:
            return False
        if len(live) == 1:
            return True
        return int(live[0].visits) > int(live[1].visits) + int(remaining)

    def _select(self, root: SearchNode) -> Optional[SearchEdge]:
        edge = root.select(float(self.config.puct_c), fpu=root.value_estimate)
        if edge is None:
            return None
        edge.probes += 1
        root.probes += 1
        return edge

    def _edge_branch(self, root: SearchNode, edge: SearchEdge, sim_index: int) -> Branch:
        seed = int(self.config.seed) * 1_000_003 + sim_index * 1009 + int(edge.position) * 3 + 1
        return Branch(
            branch_id=f"mcts-{sim_index}-1-{int(edge.position)}",
            mode="determinized",
            determinization_seed=int(seed),
            steps=[dict(edge.first_step)],
            memory={pid: tensor.clone() for pid, tensor in root.memory.items()},
            turn_counts=dict(root.turn_counts),
            on_own_decision=self._make_handler(root),
            trail=[edge],
            nodes=[root],
            state_digest=str(root.state_digest or ""),
        )

    def _backprop(self, branches: List[Branch]) -> None:
        cfg = self.config
        for branch in branches:
            value = branch.value
            if branch.status in {"leaf", "terminal"} and value is not None:
                self.valid_rollouts += 1
                if branch.bootstrap_reason:
                    self.bootstrapped[str(branch.bootstrap_reason)] += 1
                for node in branch.nodes:
                    node.record(float(value))
                for edge in branch.trail:
                    edge.record(float(value))
                continue
            self.invalid_rollouts += 1
            self.invalid_reasons[str(branch.invalid_reason or branch.status or "unknown")] += 1
            if branch.trail:
                edge = branch.trail[-1]
                if edge.record_failure(cfg.edge_kill_failures):
                    self.killed_edges += 1

    # ------------------------------------------------------------------
    # tree descent at the root player's strategic prompts
    # ------------------------------------------------------------------

    def _make_handler(self, root: SearchNode):
        cfg = self.config

        def handler(
            branch: Branch,
            state: Dict[str, Any],
            result: Any,
        ) -> Optional[Dict[str, Any]]:
            if len(branch.trail) >= int(cfg.max_root_turns_depth):
                return None
            parent_edge = branch.trail[-1]
            parent_node = branch.nodes[-1]
            child = parent_edge.target
            if child is None:
                child = self._expand_child(parent_node, state, branch, result)
                parent_edge.target = child
            matched = self._match_edges(child, result.descriptors)
            if not matched:
                return None
            edge = child.select(float(cfg.puct_c), fpu=child.value_estimate, pool=matched)
            if edge is None:
                return None
            branch.trail.append(edge)
            branch.nodes.append(child)
            return {
                "playerId": str(self.root_player_id),
                "input": dict(edge.first_step["input"]),
            }

        return handler

    def _expand_child(
        self,
        parent: SearchNode,
        state: Dict[str, Any],
        branch: Branch,
        result: Any,
    ) -> SearchNode:
        memory = {pid: tensor.clone() for pid, tensor in branch.memory.items()}
        if result.recurrent_out is not None:
            memory[self.root_player_id] = result.recurrent_out.reshape(-1).clone()
        child = SearchNode(
            depth=int(parent.depth) + 1,
            path=list(branch.steps),
            observation=dict(state or {}),
            value_estimate=float(result.value),
            memory=memory,
            turn_counts=dict(branch.turn_counts),
            state_digest=str(getattr(branch, "state_digest", "") or ""),
        )
        if child.depth < int(self.config.max_root_turns_depth):
            child.edges = _candidate_edges(
                child,
                result.descriptors,
                result.probabilities,
                self.root_player_id,
                self.config,
                self.lowercase_mc,
            )
        return child

    def _match_edges(self, child: SearchNode, descriptors: List[Dict[str, Any]]) -> List[SearchEdge]:
        """Edges whose action is still legal here: unique canonical payload,
        otherwise a unique label."""
        by_payload: Dict[str, List[Dict[str, Any]]] = {}
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for descriptor in descriptors:
            payload = descriptor.get("decoded_action")
            if isinstance(payload, dict) and payload:
                key = _canonical_input(adapt_step_input(dict(payload), self.lowercase_mc))
                by_payload.setdefault(key, []).append(descriptor)
            label = str(descriptor.get("label", "") or "")
            if label:
                by_label.setdefault(label, []).append(descriptor)
        matched: List[SearchEdge] = []
        for edge in child.live_edges():
            if len(by_payload.get(edge.action_key, [])) == 1:
                matched.append(edge)
            elif edge.label and len(by_label.get(edge.label, [])) == 1:
                matched.append(edge)
        return matched


async def decide_puct(
    agent: Any,
    client: SearchClient,
    base_url: str,
    session_id: str,
    root_player_id: str,
    root_state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    priors: List[float],
    root_value: float,
    branch_memory: Dict[str, Any],
    turn_counts: Dict[str, int],
    config: SearchConfig,
    lowercase_mc: bool,
    evaluator: Optional[PositionEvaluator] = None,
    simulation_budget: Optional[int] = None,
    root_digest: str = "",
) -> Optional[MctsOutcome]:
    search = MctsSearch(
        agent=agent,
        client=client,
        base_url=base_url,
        session_id=session_id,
        root_player_id=root_player_id,
        config=config,
        lowercase_mc=lowercase_mc,
        evaluator=evaluator,
    )
    root = search.build_root(
        root_state,
        candidates,
        priors,
        root_value,
        branch_memory,
        turn_counts,
        root_digest=root_digest,
    )
    if not root.edges:
        return MctsOutcome(chosen_position=-1, chosen_index=-1, root_visits=0)
    await search.search(root, simulation_budget=simulation_budget)
    live = root.live_edges()
    chosen = max(live, key=lambda edge: (int(edge.visits), float(edge.q), float(edge.prior))) if live else None
    if chosen is None or chosen.visits <= 0:
        return MctsOutcome(
            chosen_position=-1,
            chosen_index=-1,
            root_visits=int(root.visits),
            candidates=root.tree(),
            invalid_rollouts=int(search.invalid_rollouts),
            simulations_selected=int(search.simulations_selected),
            valid_rollouts=int(search.valid_rollouts),
            killed_edges=int(search.killed_edges),
            simulation_budget=int(search.simulation_budget),
            early_stopped=bool(search.early_stopped),
            invalid_reasons=dict(search.invalid_reasons),
            bootstrapped=dict(search.bootstrapped),
        )
    if config.selection == "temperature" and len(live) > 1:
        temperature = max(1e-3, float(config.temperature))
        weights = [math.pow(max(1e-9, float(edge.visits)), 1.0 / temperature) for edge in live]
        rng = random.Random(f"puct-select:{config.seed}:{session_id}")
        chosen = rng.choices(live, weights=weights, k=1)[0]
    return MctsOutcome(
        chosen_position=int(chosen.position),
        chosen_index=int(chosen.action_index),
        root_visits=int(root.visits),
        candidates=root.tree(),
        invalid_rollouts=int(search.invalid_rollouts),
        simulations_selected=int(search.simulations_selected),
        valid_rollouts=int(search.valid_rollouts),
        killed_edges=int(search.killed_edges),
        simulation_budget=int(search.simulation_budget),
        early_stopped=bool(search.early_stopped),
        invalid_reasons=dict(search.invalid_reasons),
        bootstrapped=dict(search.bootstrapped),
    )
