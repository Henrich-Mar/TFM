"""Policy-driven rollout of search branches through the simulation service.

A branch is an action path from the immutable session root.  Each replay round
re-sends the full grown path for every unsettled branch; the server clones the
root, determinizes, and re-applies it.  All prompt positions in a round are
evaluated in a single batched forward.  When the searching player reaches a
top-level strategic prompt the branch pauses in ``own_pending`` and a
tree-side handler (``Branch.on_own_decision``) decides whether the branch
descends one more own macro-action, burns a policy-sampled post-tree own turn
from the value horizon, or stops.  Without a handler the branch stops as a
``leaf`` and is valued by the evaluation that produced the decision, which is
also reused directly as the leaf value.  ``terminal`` branches are scored with
the terminal reward.  A ``boundary`` or step-limit stop is scored from the
last root-player value when one exists, because generation turnover and the
32-step cap are truncations, not illegal moves.  A truncation with no root
value yet, and every ``rejected`` input, is still dropped.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from models.agent import _get_inference_executor
from scoring import calculate_v2_terminal_reward

from .config import MAX_BRANCHES_PER_BATCH, SearchConfig
from .evaluator import EvalItem, EvalResult, PositionEvaluator
from .prompts import is_strategic_prompt
from .simulator_client import SearchClient, adapt_step_input

logger = logging.getLogger("rl.search")

SETTLED_STATUSES = {"leaf", "terminal", "invalid"}

OwnDecisionHandler = Callable[[("Branch"), Dict[str, Any], EvalResult], Optional[Dict[str, Any]]]


@dataclass
class Branch:
    branch_id: str
    mode: str
    determinization_seed: Optional[int]
    steps: List[Dict[str, Any]]
    memory: Dict[str, torch.Tensor] = field(default_factory=dict)
    turn_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "pending"
    prompt: Optional[Tuple[str, Dict[str, Any]]] = None
    leaf_state: Optional[Dict[str, Any]] = None
    leaf_eval: Optional[Any] = None
    terminal_players: Optional[List[Dict[str, Any]]] = None
    value: Optional[float] = None
    invalid_reason: str = ""
    # Last value head output for the searching player. Used when the server
    # stops the branch at a generation boundary or the step cap.
    last_root_value: Optional[float] = None
    bootstrap_reason: str = ""
    sent_steps: int = 0
    state_digest: str = ""
    # Tree integration: the handler receives (branch, state, eval) and returns
    # a full step dict ({playerId, input}) to descend, or None to stop.
    on_own_decision: Optional[Any] = None
    own_state: Optional[Dict[str, Any]] = None
    own_eval: Optional[Any] = None
    own_policy_turns: int = 0
    trail: List[Any] = field(default_factory=list)
    nodes: List[Any] = field(default_factory=list)
    # Opening portfolio: score the root player's view after this one input
    # and do not play opponents out to the first action menu.
    portfolio_leaf: bool = False

    @property
    def settled(self) -> bool:
        return self.status in SETTLED_STATUSES

    def rng(self, session_id: str) -> random.Random:
        return random.Random(f"{session_id}:{self.branch_id}:{self.determinization_seed}")


def terminal_value_from_players(
    root_player_id: str,
    players: Optional[List[Dict[str, Any]]],
) -> Optional[float]:
    if not players:
        return None
    rows = {str(row.get("playerId", "")): row for row in players if isinstance(row, dict)}
    root_row = rows.get(str(root_player_id))
    if root_row is None:
        return None
    try:
        rank = int(root_row.get("rank", 4) or 4)
        vp = float(root_row.get("vp", 0.0) or 0.0)
        table_mean = sum(float(row.get("vp", 0.0) or 0.0) for row in rows.values()) / max(1, len(rows))
    except (TypeError, ValueError):
        return None
    return float(calculate_v2_terminal_reward(rank, vp, table_mean, True))


class BranchRunner:
    """Advances branch batches; owns evaluation and replay coordination."""

    def __init__(
        self,
        agent: Any,
        evaluator: PositionEvaluator,
        client: SearchClient,
        base_url: str,
        session_id: str,
        root_player_id: str,
        config: SearchConfig,
        lowercase_mc: bool,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.client = client
        self.base_url = base_url
        self.session_id = str(session_id)
        self.root_player_id = str(root_player_id)
        self.config = config
        self.lowercase_mc = bool(lowercase_mc)

    async def run(self, branches: List[Branch]) -> List[Branch]:
        rounds_used = 0
        for _round_no in range(int(self.config.max_replay_rounds)):
            rounds_used += 1
            unsettled = [b for b in branches if not b.settled]
            if not unsettled:
                break
            has_work = any(
                b.prompt is not None or b.status == "own_pending" or len(b.steps) > b.sent_steps
                for b in unsettled
            )
            if not has_work:
                break
            needing_prompt = [b for b in unsettled if not b.settled and b.prompt is not None]
            if needing_prompt:
                loop = asyncio.get_running_loop()
                inference_started = time.perf_counter()
                cache_hits_before = int(getattr(self.evaluator, "cache_hits", 0))
                cache_misses_before = int(getattr(self.evaluator, "cache_misses", 0))
                await loop.run_in_executor(_get_inference_executor(), self._advance_prompts_sync, needing_prompt)
                self.client.stats.inference_sec += time.perf_counter() - inference_started
                self.client.stats.prompt_evaluations += len(needing_prompt)
                self.client.stats.eval_cache_hits += max(
                    0, int(getattr(self.evaluator, "cache_hits", 0)) - cache_hits_before
                )
                self.client.stats.eval_cache_misses += max(
                    0, int(getattr(self.evaluator, "cache_misses", 0)) - cache_misses_before
                )
            self._resolve_own_decisions([b for b in branches if b.status == "own_pending"])
            unsent = [b for b in unsettled if not b.settled and len(b.steps) > b.sent_steps]
            if unsent:
                await self._replay(unsent)
        for branch in branches:
            if not branch.settled:
                reason = (
                    "replay_round_limit"
                    if rounds_used >= int(self.config.max_replay_rounds)
                    else "rollout_stalled"
                )
                if not self._accept_truncated(branch, reason):
                    branch.status = "invalid"
                    branch.invalid_reason = reason
            if branch.status == "terminal":
                branch.value = terminal_value_from_players(self.root_player_id, branch.terminal_players)
                if branch.value is None:
                    branch.status = "invalid"
                    branch.invalid_reason = "terminal_missing_root"
        return branches

    # ------------------------------------------------------------------
    # prompt evaluation: one batched forward for every pending prompt
    # ------------------------------------------------------------------

    def _advance_prompts_sync(self, branches: List[Branch]) -> None:
        cfg = self.config
        rows: List[Tuple[Branch, EvalItem, bool, Dict[str, Any]]] = []
        for branch in branches:
            if branch.settled or branch.prompt is None:
                continue
            prompt_player, prompt_state = branch.prompt
            branch.prompt = None
            strategic_own = str(prompt_player) == self.root_player_id and is_strategic_prompt(prompt_state)
            if not strategic_own and len(branch.steps) >= cfg.max_rollout_steps:
                if not self._accept_truncated(branch, "step_limit"):
                    branch.status = "invalid"
                    branch.invalid_reason = "step_limit"
                continue
            item = EvalItem(
                player_state=prompt_state,
                player_id=str(prompt_player),
                turn_count=int(branch.turn_counts.get(str(prompt_player), 0)),
                recurrent_in=PositionEvaluator.memory_for(branch.memory, str(prompt_player)),
                state_digest=branch.state_digest,
            )
            rows.append((branch, item, strategic_own, prompt_state))
        if not rows:
            return
        results = self.evaluator.evaluate_batch([row[1] for row in rows])
        for (branch, item, strategic_own, prompt_state), result in zip(rows, results):
            if branch.settled:
                continue
            if result is None or not result.descriptors:
                branch.status = "invalid"
                branch.invalid_reason = "evaluation_failed"
                continue
            if str(item.player_id) == self.root_player_id:
                branch.last_root_value = float(result.value)
            if strategic_own:
                # Paused for the tree handler; memory/turn bookkeeping happens
                # only if the branch actually continues past this decision.
                branch.status = "own_pending"
                branch.own_state = prompt_state
                branch.own_eval = result
                continue
            PositionEvaluator.update_memory(branch.memory, item.player_id, result.recurrent_out)
            branch.turn_counts[item.player_id] = int(branch.turn_counts.get(item.player_id, 0)) + 1
            payload = self._sample_payload(branch, result)
            if payload is None:
                branch.status = "invalid"
                branch.invalid_reason = "sampling_failed"
                continue
            branch.steps.append(
                {
                    "playerId": str(item.player_id),
                    "input": adapt_step_input(dict(payload), self.lowercase_mc),
                }
            )
            branch.status = "active"

    def _sample_payload(self, branch: Branch, result: EvalResult) -> Optional[Dict[str, Any]]:
        position = self._sample_position(branch, result.probabilities)
        if position is None:
            return None
        descriptor = result.descriptors[position]
        payload = descriptor.get("decoded_action")
        if not isinstance(payload, dict) or not payload:
            return None
        return payload

    def _sample_position(self, branch: Branch, probabilities: List[float]) -> Optional[int]:
        weights: List[float] = []
        total = 0.0
        for value in probabilities:
            try:
                weight = max(0.0, float(value))
            except (TypeError, ValueError):
                weight = 0.0
            weights.append(weight)
            total += weight
        if total <= 0.0:
            return None
        rng = branch.rng(self.session_id)
        return int(rng.choices(range(len(weights)), weights=weights, k=1)[0])

    # ------------------------------------------------------------------
    # own-decision resolution: descend / horizon / leaf
    # ------------------------------------------------------------------

    def _resolve_own_decisions(self, pending: List[Branch]) -> None:
        for branch in pending:
            if branch.status != "own_pending":
                continue
            try:
                self._resolve_one(branch)
            except Exception as exc:  # a handler bug must never kill the search
                logger.debug("own-decision handler failed on %s", branch.branch_id, exc_info=True)
                branch.status = "invalid"
                branch.invalid_reason = f"own_decision_failed:{type(exc).__name__}"

    def _resolve_one(self, branch: Branch) -> None:
        cfg = self.config
        root = self.root_player_id
        result = branch.own_eval
        step: Optional[Dict[str, Any]] = None
        if branch.on_own_decision is not None and len(branch.steps) + 1 <= cfg.max_rollout_steps:
            step = branch.on_own_decision(branch, branch.own_state or {}, result)
        if isinstance(step, dict) and step.get("input"):
            self._consume_own_turn(branch)
            branch.steps.append(
                {
                    "playerId": str(step.get("playerId", root)),
                    "input": adapt_step_input(dict(step["input"]), self.lowercase_mc),
                }
            )
            branch.status = "active"
            return
        # Tree exhausted: burn post-tree policy-sampled own turns when enabled.
        if (
            cfg.value_horizon > 0
            and branch.own_policy_turns < cfg.value_horizon
            and len(branch.steps) + 1 <= cfg.max_rollout_steps
        ):
            payload = self._sample_payload(branch, result)
            if payload is not None:
                branch.own_policy_turns += 1
                self._consume_own_turn(branch)
                branch.steps.append(
                    {
                        "playerId": str(root),
                        "input": adapt_step_input(dict(payload), self.lowercase_mc),
                    }
                )
                branch.status = "active"
                return
        branch.status = "leaf"
        branch.leaf_state = branch.own_state
        branch.leaf_eval = result
        branch.value = float(result.value)

    def _accept_truncated(self, branch: Branch, reason: str) -> bool:
        """Score a truncated rollout instead of treating it as an illegal move.

        Prefer the last value computed for the searching player during the
        rollout. A pass that ends the generation never returns another root
        prompt, so that value is missing: use the value of the tree position
        that chose the last own action. Lookahead branches have no tree node
        and stay invalid until a root evaluation exists.
        """
        value = branch.last_root_value
        if value is None:
            for node in reversed(branch.nodes):
                estimate = getattr(node, "value_estimate", None)
                if estimate is not None:
                    value = float(estimate)
                    break
        if value is None:
            return False
        branch.status = "leaf"
        branch.value = float(value)
        branch.bootstrap_reason = str(reason)
        branch.invalid_reason = ""
        return True

    def _score_root_observation(self, branch: Branch, observation: Dict[str, Any]) -> None:
        """Value the searching player's own view and stop. Used for one-step portfolios."""
        item = EvalItem(
            player_state=observation,
            player_id=self.root_player_id,
            turn_count=int(branch.turn_counts.get(self.root_player_id, 0)),
            recurrent_in=PositionEvaluator.memory_for(branch.memory, self.root_player_id),
            state_digest=branch.state_digest,
            value_only=True,
        )
        started = time.perf_counter()
        results = self.evaluator.evaluate_batch([item])
        self.client.stats.inference_sec += time.perf_counter() - started
        self.client.stats.prompt_evaluations += 1
        result = results[0] if results else None
        if result is None:
            branch.status = "invalid"
            branch.invalid_reason = "portfolio_evaluation_failed"
            return
        value = float(result.value)
        branch.status = "leaf"
        branch.value = value
        branch.leaf_state = observation
        branch.leaf_eval = result
        branch.last_root_value = value

    def _consume_own_turn(self, branch: Branch) -> None:
        root = self.root_player_id
        PositionEvaluator.update_memory(branch.memory, root, branch.own_eval.recurrent_out)
        branch.turn_counts[root] = int(branch.turn_counts.get(root, 0)) + 1

    # ------------------------------------------------------------------
    # replay transport
    # ------------------------------------------------------------------

    async def _replay(self, branches: List[Branch]) -> None:
        for start in range(0, len(branches), MAX_BRANCHES_PER_BATCH):
            chunk = branches[start:start + MAX_BRANCHES_PER_BATCH]
            requests = [
                SearchClient.build_branch_request(
                    branch_id=branch.branch_id,
                    mode=branch.mode,
                    steps=branch.steps,
                    determinization_seed=branch.determinization_seed,
                )
                for branch in chunk
            ]
            try:
                results = await self.client.replay(self.session_id, requests)
            except Exception as exc:
                for branch in chunk:
                    branch.status = "invalid"
                    branch.invalid_reason = f"replay_error:{type(exc).__name__}"
                raise
            by_id = {result.branch_id: result for result in results}
            for branch in chunk:
                self._apply_result(branch, by_id.get(branch.branch_id))

    def _apply_result(self, branch: Branch, result) -> None:
        if result is None:
            branch.status = "invalid"
            branch.invalid_reason = "missing_result"
            return
        branch.sent_steps = len(branch.steps)
        if getattr(result, "state_digest", None):
            branch.state_digest = str(result.state_digest)
        if branch.portfolio_leaf:
            if result.status == "rejected":
                branch.status = "invalid"
                branch.invalid_reason = f"rejected:{result.error_code or 'unknown'}"
                return
            observation = getattr(result, "root_observation", None)
            if not isinstance(observation, dict):
                branch.status = "invalid"
                branch.invalid_reason = "missing_root_observation"
                return
            self._score_root_observation(branch, observation)
            return
        if result.status == "next_prompt" and result.next_prompt is not None:
            branch.prompt = result.next_prompt
            branch.status = "active"
            return
        if result.status == "terminal":
            branch.terminal_players = result.terminal_players
            branch.status = "terminal"
            return
        if result.status == "boundary":
            # Production, research close, and discard recycling all arrive as
            # the same status. The server does not say which. A branch that
            # already evaluated the searching player keeps that value; a
            # boundary before any root evaluation is still dropped.
            if not self._accept_truncated(branch, "boundary"):
                branch.status = "invalid"
                branch.invalid_reason = "boundary"
            return
        branch.status = "invalid"
        branch.invalid_reason = f"rejected:{result.error_code or 'unknown'}"


async def run_batched(
    agent: Any,
    client: SearchClient,
    base_url: str,
    session_id: str,
    root_player_id: str,
    branches: List[Branch],
    config: SearchConfig,
    lowercase_mc: bool,
    evaluator: Optional[PositionEvaluator] = None,
) -> List[Branch]:
    runner = BranchRunner(
        agent=agent,
        evaluator=evaluator or PositionEvaluator(agent),
        client=client,
        base_url=base_url,
        session_id=session_id,
        root_player_id=root_player_id,
        config=config,
        lowercase_mc=lowercase_mc,
    )
    return await runner.run(branches)
