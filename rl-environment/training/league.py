"""
League pools used by RL-first self-play orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class LeagueConfig:
    enabled: bool = True
    historical_ratio: float = 0.4
    exploiter_ratio: float = 0.2
    snapshot_interval: int = 5


class LeagueManager:
    def __init__(self, config: LeagueConfig):
        self.config = config
        self.main_pool: List[str] = []
        self.historical_pool: List[Dict[str, Any]] = []
        self.exploiter_pool: List[str] = []

    @staticmethod
    def _dedupe_ids(agent_ids: Sequence[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for agent_id in agent_ids:
            token = str(agent_id or "")
            if not token or token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def order_population_for_matchmaking(
        self,
        population: Sequence[Any],
        generation: Optional[int] = None,
    ) -> List[Any]:
        """
        Build a stable population order influenced by league pools.
        This affects tournament seatings while still evaluating every current agent.
        """
        ordered_population = list(population)
        if not ordered_population or not bool(self.config.enabled):
            return ordered_population

        id_to_agent = {agent.id: agent for agent in ordered_population}
        total_agents = len(ordered_population)
        if total_agents <= 1:
            return ordered_population

        # Pull only IDs that still exist in the current population.
        main_ids = [agent_id for agent_id in self._dedupe_ids(self.main_pool) if agent_id in id_to_agent]
        exploiter_ids = [agent_id for agent_id in self._dedupe_ids(self.exploiter_pool) if agent_id in id_to_agent]

        historical_candidates: List[str] = []
        for snapshot in reversed(self.historical_pool[-20:]):
            for agent_id in list(snapshot.get("top_agent_ids", []) or []):
                token = str(agent_id or "")
                if token and token in id_to_agent:
                    historical_candidates.append(token)
        historical_ids = self._dedupe_ids(historical_candidates)

        # Keep all current agents in rotation; pool targets only control priority order.
        hist_ratio = max(0.0, min(1.0, float(self.config.historical_ratio)))
        exploiter_ratio = max(0.0, min(1.0, float(self.config.exploiter_ratio)))
        hist_target = max(0, min(total_agents, int(round(total_agents * hist_ratio))))
        exploiter_target = max(0, min(total_agents - hist_target, int(round(total_agents * exploiter_ratio))))
        main_target = max(0, total_agents - hist_target - exploiter_target)

        prioritized_ids: List[str] = []

        def _append_with_limit(source: List[str], limit: int):
            if limit <= 0:
                return
            taken = 0
            for agent_id in source:
                if agent_id in prioritized_ids:
                    continue
                prioritized_ids.append(agent_id)
                taken += 1
                if taken >= limit:
                    break

        _append_with_limit(main_ids, main_target)
        _append_with_limit(exploiter_ids, exploiter_target)
        _append_with_limit(historical_ids, hist_target)

        remaining_ids = [agent.id for agent in ordered_population if agent.id not in prioritized_ids]
        if remaining_ids and generation is not None:
            offset = int(generation) % len(remaining_ids)
            remaining_ids = remaining_ids[offset:] + remaining_ids[:offset]
        prioritized_ids.extend(remaining_ids)

        return [id_to_agent[agent_id] for agent_id in prioritized_ids if agent_id in id_to_agent]

    def get_state(self) -> Dict[str, Any]:
        return {
            "main_pool": list(self.main_pool),
            "historical_pool": list(self.historical_pool),
            "exploiter_pool": list(self.exploiter_pool),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        payload = dict(state or {})
        self.main_pool = self._dedupe_ids(list(payload.get("main_pool", []) or []))
        self.exploiter_pool = self._dedupe_ids(list(payload.get("exploiter_pool", []) or []))

        historical: List[Dict[str, Any]] = []
        for snapshot in list(payload.get("historical_pool", []) or []):
            if not isinstance(snapshot, dict):
                continue
            generation = int(snapshot.get("generation", -1))
            top_ids = self._dedupe_ids(list(snapshot.get("top_agent_ids", []) or []))
            historical.append(
                {
                    "generation": generation,
                    "top_agent_ids": top_ids,
                }
            )
        self.historical_pool = historical[-200:]

    def update_generation(
        self,
        generation: int,
        population: Sequence[Any],
        fitness_scores: Sequence[float],
    ) -> Dict[str, Any]:
        if not population:
            self.main_pool = []
            self.exploiter_pool = []
            return self.get_metrics()

        ranked_indices = sorted(
            range(len(population)),
            key=lambda idx: float(fitness_scores[idx]) if idx < len(fitness_scores) else 0.0,
            reverse=True,
        )
        ranked_ids = [population[idx].id for idx in ranked_indices]

        main_count = max(1, int(len(population) * (1.0 - float(self.config.historical_ratio))))
        exploiter_count = max(1, int(len(population) * float(self.config.exploiter_ratio)))
        self.main_pool = ranked_ids[:main_count]
        self.exploiter_pool = ranked_ids[-exploiter_count:]

        if (
            self.config.enabled
            and int(self.config.snapshot_interval) > 0
            and (int(generation) % int(self.config.snapshot_interval) == 0)
        ):
            self.historical_pool.append(
                {
                    "generation": int(generation),
                    "top_agent_ids": ranked_ids[:max(1, min(8, len(ranked_ids)))],
                }
            )
            if len(self.historical_pool) > 200:
                self.historical_pool = self.historical_pool[-200:]

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "league/enabled": bool(self.config.enabled),
            "league/main_pool_size": int(len(self.main_pool)),
            "league/historical_pool_size": int(len(self.historical_pool)),
            "league/exploiter_pool_size": int(len(self.exploiter_pool)),
            "league/matchmaking_ready": bool(self.main_pool or self.exploiter_pool or self.historical_pool),
            "league/latest_historical_generation": (
                int(self.historical_pool[-1]["generation"]) if self.historical_pool else -1
            ),
        }
