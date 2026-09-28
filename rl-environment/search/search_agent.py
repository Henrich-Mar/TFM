"""SearchPolicy: the RLAgent-facing entry point for MCTS decisions.

The agent calls ``decide`` at every legal move.  A ``None`` result means
"search is not applicable right now"; the caller then falls back to ordinary
policy sampling.  Search must never break gameplay: every transport, server,
evaluation, and timeout failure degrades to the plain policy.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from models.agent import _get_inference_executor

from .config import SearchConfig
from .evaluator import PositionEvaluator
from .lookahead import decide_lookahead
from .mcts import decide_puct
from .prompts import is_search_root, prompt_type
from .simulator_client import SearchClient, SearchClientStats, SearchServiceError, SearchUnavailableError

logger = logging.getLogger("rl.search")


@dataclass
class SearchDecision:
    decoded_action: Dict[str, Any]
    action_index: int
    chosen_position: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def mcts(self) -> Dict[str, Any]:
        return dict(self.meta.get("mcts", {}))


class SearchPolicy:
    def __init__(
        self,
        agent: Any,
        config: Optional[SearchConfig] = None,
        replay_store: Optional[Any] = None,
        replay_max_invalid_rate: float = 0.10,
    ) -> None:
        self.agent = agent
        self.config = config if config is not None else SearchConfig.from_env()
        self.config.normalize()
        self.replay_store = replay_store
        self.replay_max_invalid_rate = min(1.0, max(0.0, float(replay_max_invalid_rate)))
        self.stats: Dict[str, int] = {
            "eligible_roots": 0,
            "searched": 0,
            "fallback_ineligible": 0,
            "fallback_unavailable": 0,
            "fallback_error": 0,
            "fallback_no_signal": 0,
            "bypassed_forced": 0,
        }
        self.rollout_stats: Dict[str, int] = {
            "simulations_selected": 0,
            "valid_rollouts": 0,
            "invalid_rollouts": 0,
            "killed_edges": 0,
        }
        self.rollout_invalid_reasons: Dict[str, int] = {}
        self.unavailable_by_prompt: Dict[str, int] = {}
        self.client_stats: Dict[str, Any] = {
            "failures": {},
            "starts": 0,
            "replay_batches": 0,
            "applied_steps": 0,
            "new_steps_applied": 0,
            "reused_steps": 0,
            "replay_payload_bytes": 0,
            "replay_time_sec": 0.0,
            "inference_sec": 0.0,
            "prompt_evaluations": 0,
            "eval_cache_hits": 0,
            "eval_cache_misses": 0,
            "requested_branches": 0,
            "requested_path_steps": 0,
            "max_path_steps": 0,
        }
        self._batch_latency: List[float] = []
        self._total_sec: float = 0.0
        self._search_count: int = 0
        self._client: Optional[SearchClient] = None

    def snapshot_stats(self) -> Dict[str, Any]:
        mean_batch = sum(self._batch_latency) / len(self._batch_latency) if self._batch_latency else 0.0
        replay_time = float(self.client_stats.get("replay_time_sec", 0.0) or 0.0)
        applied = int(self.client_stats.get("applied_steps", 0) or 0)
        new_steps = int(self.client_stats.get("new_steps_applied", 0) or 0)
        return {
            "enabled": bool(self.config.enabled),
            "mode": self.config.mode,
            "top_k": int(self.config.top_k),
            "determinizations": int(self.config.determinizations),
            "simulations_per_move": int(self.config.simulations_per_move),
            "max_root_turns_depth": int(self.config.max_root_turns_depth),
            "adaptive_simulations": bool(self.config.adaptive_simulations),
            "root_prompt_types": str(self.config.root_prompt_types),
            "searched_decisions": int(self.stats.get("searched", 0)),
            "search_fallbacks": int(
                self.stats.get("fallback_ineligible", 0)
                + self.stats.get("fallback_unavailable", 0)
                + self.stats.get("fallback_error", 0)
                + self.stats.get("fallback_no_signal", 0)
            ),
            "mean_replay_batch_sec": round(float(mean_batch), 4),
            "mean_search_sec": round(self._total_sec / max(1, self.stats.get("eligible_roots", 0)), 4),
            "applied_inputs_per_sec": round(applied / replay_time, 2) if replay_time > 0 else 0.0,
            "new_transitions_per_sec": round(new_steps / replay_time, 2) if replay_time > 0 else 0.0,
            "replay_amplification": round(applied / new_steps, 3) if new_steps > 0 else 0.0,
            "replay_batches": int(self.client_stats.get("replay_batches", 0)),
            "reused_steps": int(self.client_stats.get("reused_steps", 0)),
            "replay_payload_bytes": int(self.client_stats.get("replay_payload_bytes", 0)),
            "inference_sec": round(float(self.client_stats.get("inference_sec", 0.0)), 4),
            "prompt_evaluations": int(self.client_stats.get("prompt_evaluations", 0)),
            "eval_cache_hits": int(self.client_stats.get("eval_cache_hits", 0)),
            "eval_cache_misses": int(self.client_stats.get("eval_cache_misses", 0)),
            "mean_requested_path_steps": round(
                int(self.client_stats.get("requested_path_steps", 0))
                / max(1, int(self.client_stats.get("requested_branches", 0))),
                3,
            ),
            "max_path_steps": int(self.client_stats.get("max_path_steps", 0)),
            "rollout": self.rollout_snapshot(),
            "rollout_invalid_reasons": dict(self.rollout_invalid_reasons),
            "client_failures": dict(self.client_stats.get("failures", {})),
            "unavailable_by_prompt": dict(self.unavailable_by_prompt),
            "details": dict(self.stats),
        }

    def rollout_snapshot(self) -> Dict[str, Any]:
        selected = max(1, int(self.rollout_stats["simulations_selected"]))
        return {
            "simulations_selected": int(self.rollout_stats["simulations_selected"]),
            "valid_rollouts": int(self.rollout_stats["valid_rollouts"]),
            "invalid_rollouts": int(self.rollout_stats["invalid_rollouts"]),
            "invalid_rate": round(float(self.rollout_stats["invalid_rollouts"]) / selected, 4),
            "killed_edges": int(self.rollout_stats["killed_edges"]),
        }

    def _absorb_outcome(self, outcome: Any) -> None:
        try:
            summary = outcome.summary() if hasattr(outcome, "summary") else {}
            self.rollout_stats["simulations_selected"] += int(
                summary.get("simulations_selected", 0)
                or summary.get("valid_samples", 0) + summary.get("invalid_samples", 0)
            )
            self.rollout_stats["valid_rollouts"] += int(
                summary.get("valid_rollouts", summary.get("valid_samples", 0)) or 0
            )
            self.rollout_stats["invalid_rollouts"] += int(
                summary.get("invalid_rollouts", summary.get("invalid_samples", 0)) or 0
            )
            self.rollout_stats["killed_edges"] += int(summary.get("killed_edges", 0) or 0)
            for reason, count in dict(summary.get("invalid_reasons", {}) or {}).items():
                key = str(reason or "unknown")
                self.rollout_invalid_reasons[key] = int(self.rollout_invalid_reasons.get(key, 0)) + int(count or 0)
        except Exception:
            logger.debug("Could not absorb search outcome stats", exc_info=True)

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    async def decide(
        self,
        *,
        game_instance: Any,
        player_id: str,
        player_state: Dict[str, Any],
        planner_state: Dict[str, Any],
        action_descriptors: List[Dict[str, Any]],
        raw_available_actions: Optional[List[int]] = None,
    ) -> Optional[SearchDecision]:
        cfg = self.config
        if not cfg.enabled or not action_descriptors:
            return None
        if not is_search_root(player_state, cfg.root_prompt_types):
            self.stats["fallback_ineligible"] += 1
            return None
        if len(action_descriptors) <= 1:
            self.stats["bypassed_forced"] += 1
            return None
        self.stats["eligible_roots"] += 1
        started = time.perf_counter()
        client: Optional[SearchClient] = None
        self._session_id: Optional[str] = None
        try:
            base_url = str(getattr(game_instance, "base_url", "") or "").rstrip("/")
            if not base_url:
                self.stats["fallback_error"] += 1
                return None
            if self._client is None or self._client.base_url != base_url:
                if self._client is not None:
                    await self._client.aclose()
                self._client = SearchClient(base_url, timeout_sec=cfg.request_timeout_sec)
            # Decisions are sequential for one seat. Reset only counters while
            # retaining the aiohttp session and its keep-alive connection.
            self._client.stats = SearchClientStats()
            client = self._client
            return await asyncio.wait_for(
                self._decide_inner(client, game_instance, player_id, player_state, planner_state, action_descriptors, raw_available_actions),
                timeout=cfg.decide_timeout_sec,
            )
        except asyncio.TimeoutError:
            self.stats["fallback_unavailable"] += 1
            self._note_unavailable("timeout", player_state)
            logger.debug("MCTS search timed out for player %s", player_id)
            return None
        except (SearchUnavailableError, SearchServiceError) as exc:
            self.stats["fallback_unavailable"] += 1
            self._note_unavailable(getattr(exc, "code", "unknown"), player_state)
            logger.debug("MCTS search unavailable for player %s: %s", player_id, exc)
            return None
        except Exception as exc:
            self.stats["fallback_error"] += 1
            logger.warning("MCTS search failed for player %s; using policy", player_id, exc_info=True)
            return None
        finally:
            if client is not None:
                self._merge_client_stats(client.stats)
                if self._session_id:
                    try:
                        await asyncio.shield(client.close(self._session_id))
                    except Exception:
                        pass
            self._total_sec += time.perf_counter() - started

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _decide_inner(
        self,
        client: SearchClient,
        game_instance: Any,
        player_id: str,
        player_state: Dict[str, Any],
        planner_state: Dict[str, Any],
        action_descriptors: List[Dict[str, Any]],
        raw_available_actions: Optional[List[int]],
    ) -> Optional[SearchDecision]:
        cfg = self.config
        agent = self.agent
        search_cfg = cfg
        try:
            generation = int(((player_state.get("game") or {}).get("generation", 0)) or 0)
        except (TypeError, ValueError):
            generation = 0
        if (
            cfg.selection == "temperature"
            and cfg.temperature_until_generation > 0
            and generation > cfg.temperature_until_generation
        ):
            search_cfg = replace(cfg, selection="argmax")
        root = await client.start(player_id)
        self._session_id = root.session_id
        if str(root.root_player_id) != str(player_id):
            self.stats["fallback_error"] += 1
            return None
        if not is_search_root(root.observation, cfg.root_prompt_types):
            self.stats["fallback_unavailable"] += 1
            self._note_unavailable("root_not_reconstructible", root.observation)
            return None

        loop = asyncio.get_running_loop()
        phase_index = int(agent._extract_phase_index(player_state))
        recurrent_in = agent._get_recurrent_state_for_player(str(player_id))
        policy_logits, value_tensor, recurrent_out, _aux, probs_tensor = await loop.run_in_executor(
            _get_inference_executor(),
            agent._sync_forward_and_probs,
            planner_state,
            phase_index,
            recurrent_in,
        )
        probabilities = [float(p) for p in probs_tensor.reshape(-1).tolist()][: len(action_descriptors)]
        if len(probabilities) != len(action_descriptors):
            probabilities = list(probabilities) + [0.0] * (len(action_descriptors) - len(probabilities))
        root_value = float(value_tensor.reshape(-1)[0].item()) if hasattr(value_tensor, "reshape") else 0.0

        ranked = sorted(range(len(action_descriptors)), key=lambda i: probabilities[i], reverse=True)[: int(cfg.top_k)]
        candidates = [action_descriptors[i] for i in ranked]
        candidate_priors = [probabilities[i] for i in ranked]

        evaluator = PositionEvaluator(agent)
        memory = evaluator.clone_recurrent_map(self._participant_ids(root.observation, player_id))
        turn_counts: Dict[str, int] = {}
        for player_key in memory:
            turn_counts[player_key] = int(agent._turn_action_count_by_player.get(player_key, 0))

        outcome: Any = None
        if cfg.mode == "puct":
            simulation_budget = search_cfg.simulation_budget(len(action_descriptors))
            outcome = await decide_puct(
                agent=agent,
                client=client,
                base_url=client.base_url,
                session_id=root.session_id,
                root_player_id=player_id,
                root_state=player_state,
                candidates=candidates,
                priors=candidate_priors,
                root_value=root_value,
                branch_memory=memory,
                turn_counts=turn_counts,
                config=search_cfg,
                lowercase_mc=root.lowercase_mc,
                evaluator=evaluator,
                simulation_budget=simulation_budget,
                root_digest=root.root_digest,
            )
        else:
            outcome = await decide_lookahead(
                agent=agent,
                client=client,
                base_url=client.base_url,
                session_id=root.session_id,
                root_player_id=player_id,
                root_state=player_state,
                candidates=candidates,
                priors=candidate_priors,
                branch_memory=memory,
                turn_counts=turn_counts,
                config=search_cfg,
                lowercase_mc=root.lowercase_mc,
                evaluator=evaluator,
            )

        if outcome is None:
            self.stats["fallback_no_signal"] += 1
            return None
        self._absorb_outcome(outcome)
        if int(getattr(outcome, "chosen_position", -1)) < 0:
            self.stats["fallback_no_signal"] += 1
            return None
        chosen_position_local = int(outcome.chosen_position)
        descriptor = candidates[chosen_position_local]
        payload = descriptor.get("decoded_action")
        if not isinstance(payload, dict) or not payload:
            self.stats["fallback_no_signal"] += 1
            return None
        action_index = int(descriptor.get("action_index", -1))
        global_position = int(ranked[chosen_position_local])
        prior = float(candidate_priors[chosen_position_local]) if chosen_position_local < len(candidate_priors) else 0.0
        policy_target = [0.0] * len(action_descriptors)
        visit_total = 0.0
        for row in list(outcome.summary().get("candidates", []) or []):
            local_position = int(row.get("position", -1) or 0)
            visits = max(0.0, float(row.get("visits", 0) or 0))
            if 0 <= local_position < len(ranked):
                policy_target[int(ranked[local_position])] += visits
                visit_total += visits
        if visit_total <= 0.0:
            policy_target[global_position] = 1.0
        else:
            policy_target = [float(value) / visit_total for value in policy_target]

        meta: Dict[str, Any] = {
            "phase_index": phase_index,
            "recurrent_state": recurrent_in.detach().cpu().reshape(-1).tolist(),
            "recurrent_state_out": (
                recurrent_out.detach().cpu().reshape(-1).tolist()
                if recurrent_out is not None and hasattr(recurrent_out, "reshape")
                else recurrent_in.detach().cpu().reshape(-1).tolist()
            ),
            "aux_targets": agent._compute_aux_targets(player_state),
            "aux_predictions": [],
            "rare_state_weight": 1.0,
            "available_actions_raw": [int(a) for a in (raw_available_actions or [])],
            "available_actions_filtered": [int(row.get("action_index", -1)) for row in action_descriptors],
            "action_descriptors": list(action_descriptors),
            "chosen_action_position": int(global_position),
            "chosen_action_index": int(action_index),
            "chosen_action_label": agent._describe_action(action_index, player_state),
            "sampled_from_policy": False,
            "action_source": "mcts-search",
            "exclude_from_rollout": True,
            "server_accepted": False,
            "fallback_used": False,
            "value_old": float(root_value),
            "legal_actions": [int(row.get("action_index", -1)) for row in action_descriptors],
            "logp_old": float(math.log(max(1e-8, prior))),
            "policy_temperature": 1.0,
            "search_selection": search_cfg.selection,
            "mcts": outcome.summary(),
            "search_telemetry": client.stats.snapshot(),
            "search_policy_target": policy_target,
            "search_candidate_coverage": float(len(candidates)) / max(1, len(action_descriptors)),
            "policy_version": int(getattr(agent, "policy_version", 0) or 0),
            "bundle_summary": {
                "world_token_count": int(planner_state["world_tokens"].shape[0]),
                "hand_token_count": int(planner_state["hand_tokens"].shape[0]),
                "action_token_count": int(planner_state["action_tokens"].shape[0]),
                "legal_action_count": int(len(action_descriptors)),
            },
        }

        self.stats["searched"] += 1
        self._search_count += 1
        if cfg.stats_log_every > 0 and self._search_count % int(cfg.stats_log_every) == 0:
            logger.info("MCTS search stats: %s", self.snapshot_stats())
        return SearchDecision(
            decoded_action=dict(payload),
            action_index=action_index,
            chosen_position=int(global_position),
            meta=meta,
        )

    def record_accepted(self, game_id: str, player_id: str, meta: Dict[str, Any]) -> None:
        """Stage one accepted search target for commit at terminal completion."""
        if self.replay_store is None:
            return
        target = list(meta.get("search_policy_target", []) or [])
        planner_bundle = meta.get("planner_bundle")
        if not target or planner_bundle is None:
            return
        self.replay_store.record_decision(
            game_id,
            player_id,
            {
                "agent_id": str(getattr(self.agent, "id", "") or ""),
                "state_schema_version": str(getattr(self.agent, "state_schema_version", "") or ""),
                "policy_version": int(meta.get("policy_version", getattr(self.agent, "policy_version", 0)) or 0),
                "decision_sequence": int(meta.get("decision_sequence", 0) or 0),
                "planner_bundle": planner_bundle,
                "phase_index": int(meta.get("phase_index", 0) or 0),
                "recurrent_state": list(meta.get("recurrent_state", []) or []),
                "action_descriptors": list(meta.get("action_descriptors", []) or []),
                "legal_actions": list(meta.get("legal_actions", []) or []),
                "chosen_action_position": int(meta.get("chosen_action_position", 0) or 0),
                "chosen_action_index": int(meta.get("chosen_action_index", -1) or -1),
                "policy_target": target,
                "root_value": float(meta.get("value_old", 0.0) or 0.0),
                "mcts": dict(meta.get("mcts", {}) or {}),
                "search_telemetry": dict(meta.get("search_telemetry", {}) or {}),
                "search_candidate_coverage": float(meta.get("search_candidate_coverage", 0.0) or 0.0),
                "search_config": {
                    "mode": self.config.mode,
                    "top_k": int(self.config.top_k),
                    "determinizations": int(self.config.determinizations),
                    "simulations_per_move": int(self.config.simulations_per_move),
                    "max_root_turns_depth": int(self.config.max_root_turns_depth),
                    "selection": str(meta.get("search_selection", self.config.selection)),
                    "temperature": float(self.config.temperature),
                    "seed": int(self.config.seed),
                },
            },
        )

    def finish_episode(
        self,
        game_id: str,
        player_id: str,
        outcome: Dict[str, Any],
        value_target: Optional[float],
    ) -> Optional[Any]:
        if self.replay_store is None:
            return None
        selected = int(self.rollout_stats.get("simulations_selected", 0) or 0)
        invalid = int(self.rollout_stats.get("invalid_rollouts", 0) or 0)
        invalid_rate = float(invalid) / max(1, selected)
        unhealthy_fallbacks = (
            int(self.stats.get("fallback_unavailable", 0) or 0)
            + int(self.stats.get("fallback_error", 0) or 0)
            + int(self.stats.get("fallback_no_signal", 0) or 0)
        )
        if invalid_rate > self.replay_max_invalid_rate or unhealthy_fallbacks > 0:
            self.replay_store.discard_episode(game_id, player_id)
            return None
        return self.replay_store.finish_episode(
            game_id,
            player_id,
            completed=bool(outcome.get("completed", False)),
            value_target=value_target,
            outcome=outcome,
        )

    def discard_episode(self, game_id: str, player_id: str) -> None:
        if self.replay_store is not None:
            self.replay_store.discard_episode(game_id, player_id)

    def _note_unavailable(self, code: str, state: Dict[str, Any]) -> None:
        try:
            key = f"{code or 'unknown'}:{prompt_type(state) or 'unknown'}"
            self.unavailable_by_prompt[key] = int(self.unavailable_by_prompt.get(key, 0)) + 1
        except Exception:
            logger.debug("Could not attribute search fallback", exc_info=True)

    @staticmethod
    def _participant_ids(observation: Dict[str, Any], root_player_id: str) -> List[str]:
        ids = [str(root_player_id)]
        for row in observation.get("players", []) or []:
            if isinstance(row, dict):
                player_key = str(row.get("id", "") or "")
                if player_key and player_key not in ids:
                    ids.append(player_key)
        this_player = observation.get("thisPlayer")
        if isinstance(this_player, dict):
            player_key = str(this_player.get("id", "") or "")
            if player_key and player_key not in ids:
                ids.append(player_key)
        return ids

    def _merge_client_stats(self, stats: Any) -> None:
        try:
            self.client_stats["starts"] = int(self.client_stats.get("starts", 0)) + int(stats.starts)
            self.client_stats["replay_batches"] = int(self.client_stats.get("replay_batches", 0)) + int(stats.replays)
            self.client_stats["applied_steps"] = int(self.client_stats.get("applied_steps", 0)) + int(stats.applied_steps)
            self.client_stats["new_steps_applied"] = int(self.client_stats.get("new_steps_applied", 0)) + int(stats.new_steps_applied)
            self.client_stats["reused_steps"] = int(self.client_stats.get("reused_steps", 0)) + int(stats.reused_steps)
            self.client_stats["replay_payload_bytes"] = int(self.client_stats.get("replay_payload_bytes", 0)) + int(stats.replay_payload_bytes)
            self.client_stats["replay_time_sec"] = float(self.client_stats.get("replay_time_sec", 0.0)) + sum(stats.replay_batch_sec)
            self.client_stats["inference_sec"] = float(self.client_stats.get("inference_sec", 0.0)) + float(stats.inference_sec)
            self.client_stats["prompt_evaluations"] = int(self.client_stats.get("prompt_evaluations", 0)) + int(stats.prompt_evaluations)
            self.client_stats["eval_cache_hits"] = int(self.client_stats.get("eval_cache_hits", 0)) + int(stats.eval_cache_hits)
            self.client_stats["eval_cache_misses"] = int(self.client_stats.get("eval_cache_misses", 0)) + int(stats.eval_cache_misses)
            self.client_stats["requested_branches"] = int(self.client_stats.get("requested_branches", 0)) + int(stats.requested_branches)
            self.client_stats["requested_path_steps"] = int(self.client_stats.get("requested_path_steps", 0)) + int(stats.requested_path_steps)
            self.client_stats["max_path_steps"] = max(
                int(self.client_stats.get("max_path_steps", 0)), int(stats.max_path_steps)
            )
            failures = self.client_stats.setdefault("failures", {})
            for code, count in stats.failures.items():
                failures[code] = int(failures.get(code, 0)) + int(count)
            self._batch_latency.extend(stats.replay_batch_sec[-100:])
            self._batch_latency = self._batch_latency[-2000:]
        except Exception:
            logger.debug("Could not merge search client stats", exc_info=True)
