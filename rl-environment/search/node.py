"""PUCT tree nodes for root-turn macro-action search.

Tree nodes represent the searching player's own top-level decisions.  A node
stores the macro-actions (own first steps) that were legal at its creation,
never a serialized child state: prompt callbacks are not serializable and,
more importantly, a continuation path recorded under one determinization is
usually illegal under another.  Each simulation therefore re-rolls the game
between two tree nodes with a fresh determinization seed and fresh policy
sampled continuations; the tree persists only the searching player's own
actions, which determinization is guaranteed to preserve.  Edges roll up a
consecutive-failure counter: a rollout failure costs the edge a failure, and
only after ``kill_after`` consecutive failures does the edge go dead.  Failures
never write into ``value_sum``, so a transient rejection cannot pollute Q with
a parent-mean backup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch


@dataclass
class SearchEdge:
    position: int
    action_index: int
    label: str
    prior: float
    first_step: Dict[str, Any]
    action_key: str = ""
    prior_raw: Optional[float] = None
    target: Optional["SearchNode"] = None
    visits: int = 0
    value_sum: float = 0.0
    failures: int = 0
    probes: int = 0
    dead: bool = False

    @property
    def q(self) -> float:
        return float(self.value_sum / self.visits) if self.visits > 0 else 0.0

    def record(self, value: float) -> None:
        self.visits += 1
        self.value_sum += float(value)
        # Failure pruning is intentionally consecutive: any successful sample
        # proves the edge remains executable under at least one fresh rollout.
        self.failures = 0

    def record_failure(self, kill_after: int) -> bool:
        """Count a rollout failure; return True when this call killed the edge."""
        self.failures += 1
        if not self.dead and self.failures >= max(1, int(kill_after)):
            self.dead = True
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": int(self.position),
            "action_index": int(self.action_index),
            "label": self.label,
            "prior": round(float(self.prior), 6),
            "prior_raw": None if self.prior_raw is None else round(float(self.prior_raw), 6),
            "visits": int(self.visits),
            "q": round(float(self.q), 6),
            "failures": int(self.failures),
            "dead": bool(self.dead),
            "expanded": self.target is not None,
        }


@dataclass
class SearchNode:
    depth: int
    path: List[Dict[str, Any]]
    observation: Dict[str, Any]
    value_estimate: float
    memory: Dict[str, torch.Tensor] = field(default_factory=dict)
    turn_counts: Dict[str, int] = field(default_factory=dict)
    edges: List[SearchEdge] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    probes: int = 0
    state_digest: str = ""

    @property
    def q(self) -> float:
        return float(self.value_sum / self.visits) if self.visits > 0 else self.value_estimate

    def record(self, value: float) -> None:
        self.visits += 1
        self.value_sum += float(value)

    def live_edges(self) -> List[SearchEdge]:
        return [edge for edge in self.edges if not edge.dead]

    def unvisited_edges(self) -> List[SearchEdge]:
        return [edge for edge in self.live_edges() if edge.visits + edge.probes == 0]

    def puct(
        self,
        c_puct: float,
        fpu: Optional[float] = None,
        pool: Optional[Iterable[SearchEdge]] = None,
    ) -> Optional[SearchEdge]:
        edges = list(pool) if pool is not None else self.live_edges()
        edges = [edge for edge in edges if not edge.dead]
        if not edges:
            return None
        parent_total = max(1, int(self.visits) + int(self.probes))
        root = math.sqrt(float(parent_total))
        fallback = self.value_estimate if fpu is None else float(fpu)
        scored: List[tuple] = []
        for edge in edges:
            visits = int(edge.visits) + int(edge.probes)
            exploitation = float(edge.q) if visits > 0 else fallback
            exploration = c_puct * float(edge.prior) * root / (1.0 + float(visits))
            scored.append((exploitation + exploration, edge))
        best = max(scored, key=lambda item: (item[0], float(item[1].prior), -int(item[1].position)))
        return best[1]

    def select(
        self,
        c_puct: float,
        fpu: Optional[float] = None,
        pool: Optional[Iterable[SearchEdge]] = None,
    ) -> Optional[SearchEdge]:
        """Guaranteed first-visit coverage, then PUCT over live edges."""
        edges = list(pool) if pool is not None else self.live_edges()
        edges = [edge for edge in edges if not edge.dead]
        if not edges:
            return None
        unvisited = [
            edge for edge in edges if int(edge.visits) + int(edge.probes) == 0
        ]
        if unvisited:
            return max(unvisited, key=lambda edge: (float(edge.prior), -int(edge.position)))
        return self.puct(c_puct, fpu=fpu, pool=edges)

    def visit_distribution(self) -> List[Dict[str, Any]]:
        return [edge.as_dict() for edge in self.edges]

    def tree(self, max_depth: int = 3, depth: int = 1) -> List[Dict[str, Any]]:
        """Recursive edge view with nested child trees, for human inspection."""
        rows: List[Dict[str, Any]] = []
        for edge in self.edges:
            row = edge.as_dict()
            if depth < max_depth and edge.target is not None:
                row["children"] = edge.target.tree(max_depth=max_depth, depth=depth + 1)
            rows.append(row)
        return rows
