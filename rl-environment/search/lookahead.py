"""Phase A: flat deterministic-lookahead over the top-K policy macro-actions.

``top_k x determinizations`` branches are rolled out to the root player's next
top-level decision (or terminal), and the action with the best mean value wins.
This is the smallest loop that proves cloning, determinization, action replay,
and value evaluation end to end before the PUCT tree is enabled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import SearchConfig
from .evaluator import PositionEvaluator
from .rollout import Branch, run_batched
from .simulator_client import SearchClient, adapt_step_input

logger = logging.getLogger("rl.search")


@dataclass
class LookaheadOutcome:
    chosen_position: int
    chosen_index: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    valid_samples: int = 0
    invalid_samples: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": "lookahead",
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "candidates": self.candidates,
        }


async def decide_lookahead(
    agent: Any,
    client: SearchClient,
    base_url: str,
    session_id: str,
    root_player_id: str,
    root_state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    priors: List[float],
    branch_memory: Dict[str, Any],
    turn_counts: Dict[str, int],
    config: SearchConfig,
    lowercase_mc: bool,
    evaluator: Optional[PositionEvaluator] = None,
) -> Optional[LookaheadOutcome]:
    """Rank ``candidates`` (top-K descriptors with matching ``priors``) by rollout value."""
    if not candidates:
        return None
    mode = "determinized"
    branches: List[Branch] = []
    for position, descriptor in enumerate(candidates):
        payload = descriptor.get("decoded_action")
        if not isinstance(payload, dict) or not payload:
            continue
        adapted = adapt_step_input(dict(payload), lowercase_mc)
        for det_index in range(int(config.determinizations)):
            seed = int(config.seed) * 1_000_003 + position * 101 + det_index
            branches.append(
                Branch(
                    branch_id=f"la-{position}-{det_index}",
                    mode=mode,
                    determinization_seed=seed,
                    steps=[{"playerId": str(root_player_id), "input": adapted}],
                    memory={pid: tensor.clone() for pid, tensor in branch_memory.items()},
                    turn_counts=dict(turn_counts),
                )
            )
    if not branches:
        return None

    await run_batched(
        agent=agent,
        client=client,
        base_url=base_url,
        session_id=session_id,
        root_player_id=root_player_id,
        branches=branches,
        config=config,
        lowercase_mc=lowercase_mc,
        evaluator=evaluator,
    )

    per_candidate: Dict[int, List[float]] = {}
    invalid = 0
    for branch in branches:
        position = int(branch.branch_id.split("-")[1])
        if branch.status in {"leaf", "terminal"} and branch.value is not None:
            per_candidate.setdefault(position, []).append(float(branch.value))
        else:
            invalid += 1

    scored: List[Dict[str, Any]] = []
    ranked_positions: List[int] = []
    for position, descriptor in enumerate(candidates):
        values = per_candidate.get(position, [])
        mean_value = sum(values) / len(values) if values else None
        scored.append(
            {
                "position": int(position),
                "action_index": int(descriptor.get("action_index", -1)),
                "label": str(descriptor.get("label", "") or ""),
                "prior": float(priors[position]) if position < len(priors) else 0.0,
                "visits": len(values),
                "mean_value": None if mean_value is None else round(float(mean_value), 6),
                "q": 0.0 if mean_value is None else float(mean_value),
            }
        )
        if values:
            ranked_positions.append(position)
    if not ranked_positions:
        return LookaheadOutcome(
            chosen_position=-1,
            chosen_index=-1,
            candidates=scored,
            valid_samples=0,
            invalid_samples=invalid,
        )

    rng = None
    chosen: Optional[int] = None
    if config.selection == "temperature":
        import random

        rng = random.Random(f"select:{config.seed}:{session_id}")
        means = {position: sum(per_candidate[position]) / len(per_candidate[position]) for position in ranked_positions}
        peak = max(means.values())
        weights = []
        for position in ranked_positions:
            weights.append(max(1e-9, pow(2.718281828, (means[position] - peak) / config.temperature)))
        chosen = rng.choices(ranked_positions, weights=weights, k=1)[0]
    else:
        chosen = max(
            ranked_positions,
            key=lambda position: (
                sum(per_candidate[position]) / len(per_candidate[position]),
                priors[position] if position < len(priors) else 0.0,
                -position,
            ),
        )

    return LookaheadOutcome(
        chosen_position=int(chosen),
        chosen_index=int(candidates[chosen].get("action_index", -1)),
        candidates=scored,
        valid_samples=sum(len(values) for values in per_candidate.values()),
        invalid_samples=invalid,
    )
