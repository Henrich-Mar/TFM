"""
Coordinator-facing PPO optimization cycle helpers.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


async def optimize_population_with_ppo(
    population: Sequence[Any],
    target_rollout_steps: int,
) -> Dict[str, Any]:
    if not population:
        return {
            "ppo/agents_optimized": 0,
            "rollout/steps_collected": 0,
            "rollout/schema_filtered": 0,
        }

    rollout_budget = max(0, int(target_rollout_steps))
    aggregate: Dict[str, List[float]] = {}
    optimized_agents = 0
    rollout_steps = 0
    schema_filtered = 0

    ppo_agents = [agent for agent in population if hasattr(agent, "optimize_from_rollout_buffer")]
    if not ppo_agents:
        return {
            "ppo/agents_optimized": 0,
            "rollout/steps_collected": 0,
            "rollout/schema_filtered": 0,
        }

    total_available_steps = 0
    for agent in ppo_agents:
        try:
            if hasattr(agent, "get_rollout_buffer_size"):
                total_available_steps += max(0, int(agent.get_rollout_buffer_size()))
        except Exception:
            continue
    min_steps_per_agent = max(0, int(os.getenv("PPO_MIN_STEPS_PER_AGENT", "1024")))
    if min_steps_per_agent > 0 and total_available_steps > 0:
        budget_floor = min(total_available_steps, int(min_steps_per_agent * len(ppo_agents)))
        rollout_budget = max(rollout_budget, budget_floor)

    remaining_budget = rollout_budget
    remaining_agents = len(ppo_agents)

    for agent in ppo_agents:
        if remaining_budget <= 0:
            continue
        per_agent_budget = max(1, remaining_budget // max(1, remaining_agents))
        try:
            if hasattr(agent, "get_rollout_buffer_size"):
                available_for_agent = max(0, int(agent.get_rollout_buffer_size()))
                if available_for_agent > 0:
                    per_agent_budget = min(per_agent_budget, available_for_agent)
        except Exception:
            pass
        metrics = await agent.optimize_from_rollout_buffer(max_steps=per_agent_budget)
        remaining_agents = max(0, remaining_agents - 1)
        if not metrics:
            continue
        steps_used = max(0, int(metrics.get("rollout/steps", 0)))
        filtered = max(0, int(metrics.get("rollout/schema_filtered", 0)))
        if steps_used > 0:
            optimized_agents += 1
        rollout_steps += steps_used
        schema_filtered += filtered
        remaining_budget = max(0, remaining_budget - steps_used)
        for key, value in metrics.items():
            if key in ("rollout/steps", "rollout/schema_filtered"):
                continue
            normalized_key = str(key)
            normalized_value = value
            # Legacy compatibility: older agents may still emit a bool-like early-stop key.
            if normalized_key == "ppo/early_stop_kl":
                normalized_key = "ppo/early_stop_kl_ratio"
                normalized_value = 1.0 if bool(value) else 0.0
            aggregate.setdefault(normalized_key, []).append(float(normalized_value))

    merged = {key: _mean(values) for key, values in aggregate.items()}
    # Compatibility mirror for API consumers that still read this key.
    if "ppo/early_stop_kl_ratio" in merged:
        merged["ppo/early_stop_kl"] = bool(float(merged["ppo/early_stop_kl_ratio"]) > 0.0)
    merged["ppo/agents_optimized"] = int(optimized_agents)
    merged["rollout/steps_collected"] = int(rollout_steps)
    merged["rollout/schema_filtered"] = int(schema_filtered)
    return merged
