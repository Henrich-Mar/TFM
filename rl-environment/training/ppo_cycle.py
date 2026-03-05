"""
Coordinator-facing PPO optimization cycle helpers.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import fcntl  # Linux-only; used for cross-process GPU mutex.
except Exception:  # pragma: no cover - non-Linux fallback
    fcntl = None


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off")


def _safe_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


async def _acquire_ppo_gpu_mutex_if_enabled() -> Any:
    """
    Acquire a cross-process PPO GPU lock (shared across coordinators).

    Env controls:
      - PPO_GPU_MUTEX_ENABLE=1 to enable.
      - PPO_GPU_MUTEX_PATH=/app/rl-models-global/locks/ppo_gpu.lock
      - PPO_GPU_MUTEX_TIMEOUT_SEC=1800
      - PPO_GPU_MUTEX_POLL_SEC=0.2
    """
    if not _env_flag("PPO_GPU_MUTEX_ENABLE", "1"):
        return None
    if fcntl is None:
        logger.warning("PPO_GPU_MUTEX_ENABLE=1 but fcntl is unavailable; skipping global GPU mutex.")
        return None

    lock_path = str(
        os.getenv("PPO_GPU_MUTEX_PATH", "/app/rl-models-global/locks/ppo_gpu.lock")
    ).strip() or "/app/rl-models-global/locks/ppo_gpu.lock"
    timeout_sec = max(0.0, _safe_env_float("PPO_GPU_MUTEX_TIMEOUT_SEC", 1800.0))
    poll_sec = max(0.05, _safe_env_float("PPO_GPU_MUTEX_POLL_SEC", 0.2))

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    lock_file = open(lock_path, "a+", encoding="utf-8")
    start = asyncio.get_running_loop().time()

    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except BlockingIOError:
            if timeout_sec > 0.0 and (asyncio.get_running_loop().time() - start) >= timeout_sec:
                lock_file.close()
                raise TimeoutError(
                    f"Timed out waiting for PPO GPU mutex after {timeout_sec:.1f}s "
                    f"(path={lock_path})"
                )
            await asyncio.sleep(poll_sec)


def _release_ppo_gpu_mutex(lock_file: Any) -> None:
    if lock_file is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_file.close()
    except Exception:
        pass


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
    exception_count = 0
    agents_with_buffer = 0

    ppo_agents = [agent for agent in population if hasattr(agent, "optimize_from_rollout_buffer")]
    if not ppo_agents:
        return {
            "ppo/agents_optimized": 0,
            "rollout/steps_collected": 0,
            "rollout/schema_filtered": 0,
            "ppo/exception_count": 0,
            "ppo/agents_with_rollout_buffer": 0,
            "rollout/steps_available_before": 0,
        }

    total_available_steps = 0
    for agent in ppo_agents:
        try:
            if hasattr(agent, "get_rollout_buffer_size"):
                agent_steps = max(0, int(agent.get_rollout_buffer_size()))
                total_available_steps += agent_steps
                if agent_steps > 0:
                    agents_with_buffer += 1
        except Exception:
            continue
    min_steps_per_agent = max(0, int(os.getenv("PPO_MIN_STEPS_PER_AGENT", "1024")))
    if min_steps_per_agent > 0 and total_available_steps > 0:
        budget_floor = min(total_available_steps, int(min_steps_per_agent * len(ppo_agents)))
        rollout_budget = max(rollout_budget, budget_floor)

    # Build (agent, budget) list for parallel execution.
    remaining_budget = rollout_budget
    remaining_agents = len(ppo_agents)
    work: List[Tuple[Any, int]] = []
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
        work.append((agent, per_agent_budget))
        remaining_agents = max(0, remaining_agents - 1)

    if not work:
        return {
            "ppo/agents_optimized": 0,
            "rollout/steps_collected": 0,
            "rollout/schema_filtered": 0,
            "ppo/exception_count": 0,
            "ppo/agents_with_rollout_buffer": int(agents_with_buffer),
            "rollout/steps_available_before": int(total_available_steps),
        }

    # Run PPO updates in parallel (limited concurrency to avoid GPU OOM).
    parallel = max(1, min(int(os.getenv("PPO_PARALLEL_AGENTS", "4")), len(work)))
    sem = asyncio.Semaphore(parallel)

    async def _run_one(agent: Any, budget: int) -> Dict[str, Any]:
        lock_file = None
        async with sem:
            try:
                lock_file = await _acquire_ppo_gpu_mutex_if_enabled()
                return await agent.optimize_from_rollout_buffer(max_steps=budget)
            finally:
                _release_ppo_gpu_mutex(lock_file)

    results = await asyncio.gather(
        *[_run_one(agent, budget) for agent, budget in work],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            exception_count += 1
            logger.error(
                "PPO update task failed; continuing with remaining agents. "
                "error_type=%s error=%s",
                type(result).__name__,
                str(result),
                exc_info=(type(result), result, result.__traceback__),
            )
            continue
        metrics = result
        if not metrics:
            continue
        steps_used = max(0, int(metrics.get("rollout/steps", 0)))
        filtered = max(0, int(metrics.get("rollout/schema_filtered", 0)))
        if steps_used > 0:
            optimized_agents += 1
        rollout_steps += steps_used
        schema_filtered += filtered
        for key, value in metrics.items():
            if key in ("rollout/steps", "rollout/schema_filtered"):
                continue
            normalized_key = str(key)
            normalized_value = value
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
    merged["ppo/exception_count"] = int(exception_count)
    merged["ppo/agents_with_rollout_buffer"] = int(agents_with_buffer)
    merged["rollout/steps_available_before"] = int(total_available_steps)

    if int(rollout_steps) == 0 and int(total_available_steps) > 0:
        logger.warning(
            "PPO collected zero steps despite buffered rollout data: "
            "available_steps=%d agents_with_buffer=%d exceptions=%d",
            int(total_available_steps),
            int(agents_with_buffer),
            int(exception_count),
        )
    return merged
