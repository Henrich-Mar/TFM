#!/usr/bin/env python3
"""
Generate a cloud-focused RL compose file with dynamic TM server count and concurrency.

The generator reads a base compose file (default: docker-compose.rl_hard.yml),
keeps most rl-coordinator environment values, and replaces capacity-sensitive
values using host RAM/CPU heuristics.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DYNAMIC_ENV_KEYS = {
    "GAME_SERVERS",
    "PUBLIC_TM_MAP",
    "PUBLIC_TM_SERVER_ID_MAP",
    "PUBLIC_TM_URL",
    "INTERNAL_TM_URL",
    "GLOBAL_GAME_CONCURRENCY",
    "TOURNAMENT_CONCURRENCY",
    "MAX_ACTIVE_GAMES_PER_SERVER",
    "TM_HTTP_CONNECTOR_LIMIT",
    "TM_HTTP_CONNECTOR_LIMIT_PER_HOST",
    "TM_HTTP_DNS_CACHE_TTL_SEC",
    "POPULATION_SIZE",
    "TOURNAMENT_SIZE",
    "GAMES_PER_EVAL",
    "PPO_ROLLOUT_STEPS",
    "PPO_MIN_STEPS_PER_AGENT",
    "PPO_BUFFER_MAX_STEPS",
    "AGENT_POLL_INTERVAL_SEC",
    "AGENT_FAILURE_PAUSE_SEC",
    "AGENT_POST_MOVE_SLEEP_SEC",
    "AGENT_INFERENCE_BATCH_DEADLINE_MS",
    "TM_SEND_INPUT_INITIAL_CARDS_JITTER_MS",
    "AGENT_INFERENCE_THREADS",
    "TM_SERVER_SLOT_WAIT_TIMEOUT_SEC",
    "TM_CREATE_GAME_RETRY_ATTEMPTS",
    "TM_HTTP_REQUEST_TOTAL_TIMEOUT_SEC",
    "TM_GET_STATE_RETRY_ATTEMPTS",
    "TM_HTTP_FORCE_CLOSE_CONNECTIONS",
    "TM_RECYCLE_SESSION_ON_DISCONNECT",
    "TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS",
    "TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL",
}


ESSENTIAL_ENV_DEFAULTS = [
    "REDIS_URL=redis://redis:6379/0",
    "POSTGRES_URL=postgresql://tfm:tfm_password@postgres:5432/tfm_rl",
    "POPULATION_SIZE=16",
    "GENERATIONS=5000",
    "GAMES_PER_EVAL=6",
    "TM_GAME_TIMEOUT_SEC=420",
    "TM_FAST_MODE_OPTION=1",
    "RL_MODELS_DIR=/app/rl-models",
    "RL_CHECKPOINT_DIR=/app/rl-models/checkpoints",
    "TM_CARD_METADATA_PATH=/app/card_metadata.json",
]


@dataclass
class CapacityPlan:
    total_ram_mb: int
    cpu_count: int
    server_count: int
    server_mem_mb: int
    node_heap_mb: int
    games_per_server: int
    global_game_concurrency: int
    tournament_concurrency: int
    http_limit: int
    http_limit_per_host: int


@dataclass
class TrainingPlan:
    profile: str
    population_size: int
    tournament_size: int
    games_per_eval: int
    expected_games_per_generation: int
    game_instances_per_generation: int
    ppo_rollout_steps: int
    ppo_min_steps_per_agent: int
    ppo_buffer_max_steps: int


def _read_total_ram_mb() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return max(1024, int(parts[1]) // 1024)
    # Non-Linux fallback
    return max(1024, int(os.getenv("TOTAL_RAM_MB", "16384")))


def _read_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))

def _auto_num_coordinators(
    global_game_concurrency: int,
    profile: str,
    coordinator_games_target: int,
) -> int:
    """
    Pick how many coordinator replicas to run so each one handles at most
    coordinator_games_target concurrent games.

    A single Python async event loop saturates at ~600-800 HTTP req/s.
    With poll_interval=0.03s, target=20 gives ≤667 req/s per coordinator.
    """
    if coordinator_games_target <= 0:
        coordinator_games_target = 20 if profile == "saturate" else 28
    return max(1, math.ceil(global_game_concurrency / coordinator_games_target))


def _pick_games_per_server(total_ram_mb: int, cpu_count: int, override: int | None) -> int:
    if override and override > 0:
        return int(override)
    if total_ram_mb >= 196_608 or cpu_count >= 48:
        return 6
    if total_ram_mb >= 98_304 or cpu_count >= 24:
        return 5
    if total_ram_mb >= 49_152 or cpu_count >= 16:
        return 4
    return 3


def build_capacity_plan(args: argparse.Namespace) -> CapacityPlan:
    total_ram_mb = _read_total_ram_mb()
    cpu_count = _read_cpu_count()

    reserve_mb = args.reserve_mb
    if reserve_mb <= 0:
        reserve_mb = _clamp(int(total_ram_mb * 0.10), 6144, 16384)

    server_mem_mb = max(512, int(args.server_mem_mb))
    infra_overhead_mb = max(1024, int(args.infra_overhead_mb))

    usable_for_servers_mb = max(1024, total_ram_mb - reserve_mb - infra_overhead_mb)
    max_by_ram = max(1, usable_for_servers_mb // server_mem_mb)
    max_by_cpu = max(1, int(cpu_count * args.cpu_server_ratio))

    server_upper = max(1, min(max_by_ram, max_by_cpu, args.max_servers))
    server_count = _clamp(server_upper, args.min_servers, args.max_servers)

    node_heap_mb = int(args.node_heap_mb)
    if node_heap_mb <= 0:
        # Keep headroom for non-V8 memory and spikes.
        node_heap_mb = max(384, min(server_mem_mb - 320, 2048))

    games_per_server = _pick_games_per_server(total_ram_mb, cpu_count, args.games_per_server)
    global_game_concurrency = max(1, server_count * games_per_server)
    computed_tournament = max(1, min(global_game_concurrency, cpu_count * 2))
    tournament_concurrency = (
        max(1, int(args.tournament_concurrency))
        if int(args.tournament_concurrency) > 0
        else computed_tournament
    )

    default_http_limit_per_host = max(32, games_per_server * 12)
    configured_http_limit_per_host = max(0, int(args.http_connector_limit_per_host))
    http_limit_per_host = (
        configured_http_limit_per_host
        if configured_http_limit_per_host > 0
        else default_http_limit_per_host
    )

    default_http_limit = _clamp(server_count * http_limit_per_host, 512, 8192)
    configured_http_limit = max(0, int(args.http_connector_limit))
    http_limit = configured_http_limit if configured_http_limit > 0 else default_http_limit

    return CapacityPlan(
        total_ram_mb=total_ram_mb,
        cpu_count=cpu_count,
        server_count=server_count,
        server_mem_mb=server_mem_mb,
        node_heap_mb=node_heap_mb,
        games_per_server=games_per_server,
        global_game_concurrency=global_game_concurrency,
        tournament_concurrency=tournament_concurrency,
        http_limit=http_limit,
        http_limit_per_host=http_limit_per_host,
    )


def _extract_env_from_base_compose(base_compose_path: Path) -> List[str]:
    if not base_compose_path.exists():
        return []
    text = base_compose_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_coordinator = False
    in_environment = False
    env_items: List[str] = []

    for raw in lines:
        if re.match(r"^\s{2}rl-coordinator:\s*$", raw):
            in_coordinator = True
            in_environment = False
            continue
        if in_coordinator and re.match(r"^\s{2}[A-Za-z0-9_-]+:\s*$", raw):
            # Next top-level service
            break
        if in_coordinator and re.match(r"^\s{4}environment:\s*$", raw):
            in_environment = True
            continue

        if not in_environment:
            continue
        if re.match(r"^\s{6}-\s", raw):
            item = raw.split("-", 1)[1].strip()
            if item:
                env_items.append(item)
            continue
        if re.match(r"^\s{6}#", raw) or not raw.strip():
            continue
        # Environment block ended
        break

    return env_items


def _merge_env_lists(*lists: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for items in lists:
        for item in items:
            key = item.split("=", 1)[0].strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _env_list_to_map(env_items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in env_items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip()
    return out


def _apply_env_overrides(env_items: Iterable[str], overrides: Dict[str, str]) -> List[str]:
    items = list(env_items)
    positions: Dict[str, int] = {}
    for idx, item in enumerate(items):
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if key:
            positions[key] = idx
    for key, value in overrides.items():
        token = f"{key}={value}"
        if key in positions:
            items[positions[key]] = token
        else:
            positions[key] = len(items)
            items.append(token)
    return items


def _coordinator_role_env_overrides(coord_index: int) -> Dict[str, str]:
    # coord_index is 0-based.
    if int(coord_index) <= 0:
        return {
            "PPO_ENABLE": "1",
            "SAVE_TOP_K": "2",
            "SAVE_EVERY_N_GENERATIONS": "1",
            "MAX_SAVED_GENERATIONS": "30",
            "TRAINING_POOL_EXTRA_CHECKPOINTS": "/app/rl-models-global/champion/current/champion.pth",
        }
    return {
        "PPO_ENABLE": "0",
        "SAVE_TOP_K": "1",
        "SAVE_EVERY_N_GENERATIONS": "3",
        "MAX_SAVED_GENERATIONS": "15",
        "FIXED_BENCHMARK_ENABLED": "0",
    }


def _build_orchestrator_coord_sources(num_coordinators: int) -> str:
    return ",".join(
        f"coord-{idx}=/app/coord-models/coord-{idx}"
        for idx in range(1, max(1, int(num_coordinators)) + 1)
    )


def _safe_int_from_map(env_map: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(env_map.get(key, str(default))).strip())
    except Exception:
        return int(default)


def build_training_plan(
    args: argparse.Namespace,
    capacity: CapacityPlan,
    base_env_map: Dict[str, str],
) -> TrainingPlan:
    profile = str(args.training_profile or "balanced").strip().lower()
    if profile not in ("balanced", "saturate"):
        profile = "balanced"

    base_population_size = max(4, _safe_int_from_map(base_env_map, "POPULATION_SIZE", 16))
    base_tournament_size = max(4, _safe_int_from_map(base_env_map, "TOURNAMENT_SIZE", 4))
    base_games_per_eval = max(1, _safe_int_from_map(base_env_map, "GAMES_PER_EVAL", 6))
    base_rollout_steps = max(8192, _safe_int_from_map(base_env_map, "PPO_ROLLOUT_STEPS", 65536))
    base_min_steps_per_agent = max(256, _safe_int_from_map(base_env_map, "PPO_MIN_STEPS_PER_AGENT", 1024))
    base_buffer_max_steps = max(65536, _safe_int_from_map(base_env_map, "PPO_BUFFER_MAX_STEPS", 600000))

    population_size = int(args.population_size) if int(args.population_size) > 0 else base_population_size
    tournament_size = int(args.tournament_size) if int(args.tournament_size) > 0 else base_tournament_size
    tournament_size = max(4, tournament_size)

    if int(args.games_per_eval) > 0:
        games_per_eval = int(args.games_per_eval)
    else:
        factor = float(args.games_per_eval_factor)
        if factor <= 0.0:
            factor = 10.0 if profile == "saturate" else 1.0
        scaled = max(1, int(round(base_games_per_eval * factor)))

        target_waves = float(args.target_game_waves)
        if target_waves <= 0.0:
            target_waves = 2.0 if profile == "saturate" else 1.0
        min_instances = int(math.ceil(float(capacity.global_game_concurrency) * target_waves))
        min_games_per_eval = int(
            math.ceil((float(min_instances) * 4.0) / max(1.0, float(population_size)))
        )
        games_per_eval = max(1, scaled, min_games_per_eval)

    expected_games_per_generation = int(population_size * games_per_eval)
    game_instances_per_generation = int(math.ceil(float(expected_games_per_generation) / 4.0))

    ppo_min_steps_per_agent = (
        int(args.ppo_min_steps_per_agent)
        if int(args.ppo_min_steps_per_agent) > 0
        else base_min_steps_per_agent
    )
    ppo_min_steps_per_agent = max(256, ppo_min_steps_per_agent)

    if int(args.ppo_rollout_steps) > 0:
        ppo_rollout_steps = int(args.ppo_rollout_steps)
    else:
        rollout_factor = float(args.ppo_rollout_factor)
        if rollout_factor <= 0.0:
            rollout_factor = 2.0 if profile == "saturate" else 1.0

        scaled_rollout = max(8192, int(round(base_rollout_steps * rollout_factor)))
        floor_from_agent_min = int(population_size * ppo_min_steps_per_agent)
        if profile == "saturate":
            floor_from_agent_min *= 2
        floor_from_eval_volume = int(expected_games_per_generation * (48 if profile == "saturate" else 24))
        ppo_rollout_steps = max(8192, scaled_rollout, floor_from_agent_min, floor_from_eval_volume)
        ppo_rollout_steps = min(400000, ppo_rollout_steps)

    if int(args.ppo_buffer_max_steps) > 0:
        ppo_buffer_max_steps = int(args.ppo_buffer_max_steps)
    else:
        multiplier = 5 if profile == "saturate" else 4
        ppo_buffer_max_steps = max(base_buffer_max_steps, ppo_rollout_steps * multiplier)
        ppo_buffer_max_steps = min(1_200_000, ppo_buffer_max_steps)
    ppo_buffer_max_steps = max(ppo_rollout_steps * 2, ppo_buffer_max_steps)

    return TrainingPlan(
        profile=profile,
        population_size=population_size,
        tournament_size=tournament_size,
        games_per_eval=games_per_eval,
        expected_games_per_generation=expected_games_per_generation,
        game_instances_per_generation=game_instances_per_generation,
        ppo_rollout_steps=ppo_rollout_steps,
        ppo_min_steps_per_agent=ppo_min_steps_per_agent,
        ppo_buffer_max_steps=ppo_buffer_max_steps,
    )


def _build_dynamic_env(
    capacity: CapacityPlan,
    training: TrainingPlan,
    args: argparse.Namespace,
    public_host: str,
    base_port: int,
    server_range: Optional[Tuple[int, int]] = None,
    rl_models_subdir: str = "rl-models",
    num_coordinators: int = 1, 
) -> List[str]:
    if server_range is None:
        start_idx, end_idx = 1, capacity.server_count
    else:
        start_idx, end_idx = server_range
    subset_count = end_idx - start_idx + 1

    game_servers = ",".join(f"tfm-server-{i}:8080" for i in range(start_idx, end_idx + 1))
    public_tm_map = ",".join(
        f"tfm-server-{i}:8080=http://{public_host}:{base_port + i - 1}"
        for i in range(start_idx, end_idx + 1)
    )
    public_tm_server_id_map = ",".join(
        f"tfm-server-{i}:8080=tfm-server-{i}"
        for i in range(start_idx, end_idx + 1)
    )
    public_tm_url = ",".join(
        f"http://{public_host}:{base_port + i - 1}"
        for i in range(start_idx, end_idx + 1)
    )
    internal_tm_url = ",".join(
        f"http://tfm-server-{i}:8080"
        for i in range(start_idx, end_idx + 1)
    )

    subset_global = subset_count * capacity.games_per_server
    subset_tournament = max(1, min(subset_global, capacity.cpu_count * 2))
    subset_http = max(256, int(capacity.http_limit * subset_count / max(1, capacity.server_count)))
    subset_http_per_host = max(16, int(capacity.http_limit_per_host * subset_count / max(1, capacity.server_count)))

    env_items = [
        f"GAME_SERVERS={game_servers}",
        f"PUBLIC_TM_MAP={public_tm_map}",
        f"PUBLIC_TM_SERVER_ID_MAP={public_tm_server_id_map}",
        f"PUBLIC_TM_URL={public_tm_url}",
        f"INTERNAL_TM_URL={internal_tm_url}",
        f"TOURNAMENT_CONCURRENCY={subset_tournament}",
        f"GLOBAL_GAME_CONCURRENCY={subset_global}",
        f"MAX_ACTIVE_GAMES_PER_SERVER={capacity.games_per_server}",
        f"TM_HTTP_CONNECTOR_LIMIT={subset_http}",
        f"TM_HTTP_CONNECTOR_LIMIT_PER_HOST={subset_http_per_host}",
        f"TM_HTTP_DNS_CACHE_TTL_SEC=300",
        f"POPULATION_SIZE={training.population_size}",
        f"TOURNAMENT_SIZE={training.tournament_size}",
        f"GAMES_PER_EVAL={training.games_per_eval}",
        f"PPO_ROLLOUT_STEPS={training.ppo_rollout_steps}",
        f"PPO_MIN_STEPS_PER_AGENT={training.ppo_min_steps_per_agent}",
        f"PPO_BUFFER_MAX_STEPS={training.ppo_buffer_max_steps}",
    ]

    poll_interval = float(args.agent_poll_interval_sec)
    if poll_interval < 0:
        poll_interval = 0.03 if training.profile == "saturate" else -1.0
    if poll_interval >= 0:
        env_items.append(f"AGENT_POLL_INTERVAL_SEC={poll_interval:.3f}")

    failure_pause = float(args.agent_failure_pause_sec)
    if failure_pause < 0:
        failure_pause = 0.05 if training.profile == "saturate" else -1.0
    if failure_pause >= 0:
        env_items.append(f"AGENT_FAILURE_PAUSE_SEC={failure_pause:.3f}")

    post_move_sleep = float(args.agent_post_move_sleep_sec)
    if post_move_sleep < 0:
        post_move_sleep = 0.0 if training.profile == "saturate" else -1.0
    if post_move_sleep >= 0:
        env_items.append(f"AGENT_POST_MOVE_SLEEP_SEC={post_move_sleep:.3f}")

    batch_deadline = float(args.agent_inference_batch_deadline_ms)
    if batch_deadline > 0:
        env_items.append(f"AGENT_INFERENCE_BATCH_DEADLINE_MS={batch_deadline:.1f}")

    initial_cards_jitter_ms = int(args.initial_cards_jitter_ms)
    if initial_cards_jitter_ms < 0:
        if training.profile == "saturate":
            initial_cards_jitter_ms = 1200 if int(num_coordinators) > 1 else 800
        else:
            initial_cards_jitter_ms = -1
    if initial_cards_jitter_ms >= 0:
        env_items.append(f"TM_SEND_INPUT_INITIAL_CARDS_JITTER_MS={initial_cards_jitter_ms}")

    send_input_retries = int(args.send_input_transport_retry_attempts)
    if send_input_retries > 0:
        env_items.append(f"TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS={send_input_retries}")
    elif training.profile == "saturate":
        env_items.append("TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS=3")

    send_input_initial_retries = int(args.send_input_transport_retry_attempts_initial)
    if send_input_initial_retries > 0:
        env_items.append(
            f"TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL={send_input_initial_retries}"
        )
    elif training.profile == "saturate":
        env_items.append("TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL=7")

    inference_threads = int(args.agent_inference_threads)
    if inference_threads >= 0:
        env_items.append(f"AGENT_INFERENCE_THREADS={inference_threads}")
    elif training.profile == "saturate":
        # Auto: give each coordinator a fair slice of host CPUs so PyTorch
        # inference never blocks the async event loop.
        auto_threads = _clamp(
            capacity.cpu_count // max(1, num_coordinators),
            minimum=4,
            maximum=32,
        )
        env_items.append(f"AGENT_INFERENCE_THREADS={auto_threads}")

    slot_wait_sec = int(args.server_slot_wait_timeout_sec)
    if slot_wait_sec > 0:
        env_items.append(f"TM_SERVER_SLOT_WAIT_TIMEOUT_SEC={slot_wait_sec}")
    elif training.profile == "saturate":
        env_items.append("TM_SERVER_SLOT_WAIT_TIMEOUT_SEC=90")

    create_retries = int(args.create_game_retry_attempts)
    if create_retries > 0:
        env_items.append(f"TM_CREATE_GAME_RETRY_ATTEMPTS={create_retries}")
    elif training.profile == "saturate":
        env_items.append("TM_CREATE_GAME_RETRY_ATTEMPTS=6")

    http_req_timeout = int(args.http_request_timeout_sec)
    if http_req_timeout > 0:
        env_items.append(f"TM_HTTP_REQUEST_TOTAL_TIMEOUT_SEC={http_req_timeout}")
    elif training.profile == "saturate":
        env_items.append("TM_HTTP_REQUEST_TOTAL_TIMEOUT_SEC=120")

    get_state_retries = int(args.get_state_retry_attempts)
    if get_state_retries > 0:
        env_items.append(f"TM_GET_STATE_RETRY_ATTEMPTS={get_state_retries}")
    elif training.profile == "saturate":
        env_items.append("TM_GET_STATE_RETRY_ATTEMPTS=5")

    tm_game_timeout = int(args.tm_game_timeout_sec)
    if tm_game_timeout > 0:
        env_items.append(f"TM_GAME_TIMEOUT_SEC={tm_game_timeout}")
    elif training.profile == "saturate":
        env_items.append("TM_GAME_TIMEOUT_SEC=800")

    # Connection reuse: 0 = keep-alive (better async I/O throughput), 1 = close after each request
    force_close = int(args.tm_http_force_close_connections)
    env_items.append(f"TM_HTTP_FORCE_CLOSE_CONNECTIONS={1 if force_close else 0}")

    # Session recycle on disconnect can amplify churn under high concurrency.
    recycle_on_disconnect = int(args.tm_recycle_session_on_disconnect)
    env_items.append(f"TM_RECYCLE_SESSION_ON_DISCONNECT={1 if recycle_on_disconnect else 0}")

    if rl_models_subdir != "rl-models":
        env_items.append(f"RL_MODELS_DIR=/app/{rl_models_subdir}")
        env_items.append(f"RL_CHECKPOINT_DIR=/app/{rl_models_subdir}/checkpoints")

    return env_items


def _render_compose(
    plan: CapacityPlan,
    env_items: List[str],
    base_port: int,
    coordinator_base_port: int = 5000,
    num_coordinators: int = 1,
    coord_envs: Optional[List[List[str]]] = None,
    coordinators_share_gpu: bool = False,
    gpu_index: int = 0,
) -> str:
    lines: List[str] = []
    lines.append("services:")

    # Reserve the last 3 cores for coordinator / redis / postgres.
    # Each game server is pinned to exactly one core so Node.js event loops
    # never migrate between cores and thrash each other's L2/L3 cache.
    pinnable_cores = max(1, plan.cpu_count - 3)

    for i in range(1, plan.server_count + 1):
        host_port = base_port + i - 1
        core_pin = (i - 1) % pinnable_cores   # wraps if server_count > pinnable_cores

        # NODE_OPTIONS only allows a restricted set of flags (V8 security policy).
        # --optimize-for-size, --gc-interval, --expose-gc are all blocked.
        # Only --max-old-space-size is safe and reliable here.
        node_options = f"--max-old-space-size={plan.node_heap_mb}"

        lines.extend(
            [
                f"  tfm-server-{i}:",
                "    build:",
                "      context: ./terraforming-mars",
                "      dockerfile: ../Dockerfile.rl",
                "    ports:",
                f"      - \"{host_port}:8080\"",
                # Pin to a single core — eliminates cross-core cache thrashing.
                f"    cpuset: \"{core_pin}\"",
                "    environment:",
                "      - NODE_ENV=production",
                "      - COMPRESS_COMPLETED_GAMES_DAYS=0",
                "      - TM_FAST_MODE_OPTION=1",        # headless/fast mode on the server itself
                f"      - SERVER_ID=tfm-server-{i}",
                f"      - NODE_OPTIONS={node_options}",
                # 4 libuv threads is plenty for game simulation (no heavy disk/DNS I/O).
                # 16 threads × 58 servers = 928 threads thrashing 61 cores.
                "      - UV_THREADPOOL_SIZE=4",
                "    tmpfs:",
                "      - /usr/src/app/db:uid=100,gid=65533,mode=0775",
                "    labels:",
                "      - autoheal=true",
                "    healthcheck:",
                "      test:",
                "        - CMD-SHELL",
                # wget is ~10× cheaper than spawning a full `node -e` process.
                # 58 servers × old healthcheck = 58 Node.js forks every 15 s.
                "        - wget -qO- http://127.0.0.1:8080/ >/dev/null 2>&1",
                "      interval: 30s",       # was 15s — halves healthcheck overhead
                "      timeout: 5s",
                "      retries: 3",
                "      start_period: 45s",
                f"    mem_limit: {plan.server_mem_mb}m",
                "    restart: unless-stopped",
                "",
            ]
        )

    lines.extend(
        [
            "  autoheal:",
            "    image: willfarrell/autoheal:1.2.0",
            "    environment:",
            "      - AUTOHEAL_CONTAINER_LABEL=autoheal",
            "      - AUTOHEAL_INTERVAL=15",
            "      - AUTOHEAL_START_PERIOD=120",
            "      - AUTOHEAL_DEFAULT_STOP_TIMEOUT=20",
            "    volumes:",
            "      - /var/run/docker.sock:/var/run/docker.sock",
            "    restart: unless-stopped",
            "    depends_on:",
        ]
    )
    for i in range(1, plan.server_count + 1):
        lines.append(f"      - tfm-server-{i}")

    lines.extend(
        [
            "",
            "  redis:",
            "    image: redis:alpine",
            # Disable RDB snapshots and AOF — RL training state is ephemeral;
            # persistence only causes periodic disk I/O spikes.
            "    command: redis-server --save \"\" --appendonly no",
            "    ports:",
            "      - \"6379:6379\"",
            "    restart: unless-stopped",
            "",
            "  postgres:",
            "    image: postgres:15-alpine",
            "    environment:",
            "      - POSTGRES_USER=tfm",
            "      - POSTGRES_PASSWORD=tfm_password",
            "      - POSTGRES_DB=tfm_rl",
            # Tune Postgres for write-heavy RL workload.
            # synchronous_commit=off is the single biggest win: eliminates the
            # per-transaction fsync wait (tiny data-loss risk on crash is fine
            # for ephemeral training data).
            "    command: >",
            "      postgres",
            "        -c synchronous_commit=off",
            "        -c shared_buffers=256MB",
            "        -c work_mem=8MB",
            "        -c wal_buffers=16MB",
            "        -c checkpoint_completion_target=0.9",
            "        -c max_wal_size=1GB",
            "        -c wal_compression=on",
            "        -c log_min_duration_statement=5000",
            "    ports:",
            "      - \"5432:5432\"",
            "    restart: unless-stopped",
            "    volumes:",
            "      - postgres_data_cloud:/var/lib/postgresql/data",
            "",
        ]
    )

    use_shared_gpu = coordinators_share_gpu and num_coordinators > 1
    gpu_device_block = (
        [
            "            - driver: nvidia",
            f"              device_ids: ['{gpu_index}']",
            "              capabilities: [gpu]",
        ]
        if use_shared_gpu
        else [
            "            - driver: nvidia",
            "              count: 1",
            "              capabilities: [gpu]",
        ]
    )

    if num_coordinators > 1 and coord_envs is not None and len(coord_envs) == num_coordinators:
        for c in range(num_coordinators):
            coord_name = f"rl-coordinator-{c + 1}"
            models_subdir = f"rl-models-coord-{c + 1}"
            coord_port = int(coordinator_base_port) + c
            lines.extend(
                [
                    "",
                    f"  {coord_name}:",
                    "    build:",
                    "      context: ./rl-environment",
                    "      dockerfile: Dockerfile.training",
                    "    restart: unless-stopped",
                    "    deploy:",
                    "      resources:",
                    "        reservations:",
                    "          devices:",
                ]
            )
            lines.extend(gpu_device_block)
            lines.extend(
                [
                    "    volumes:",
                    "      - ./rl-environment:/app",
                    f"      - ./{models_subdir}:/app/{models_subdir}",
                    "      - ./rl-models-global:/app/rl-models-global",
                    "      - ./card_metadata.json:/app/card_metadata.json:ro",
                    # tmpfs for logs avoids host-disk I/O under high-concurrency
                    # saturate mode. Use a bind mount if you need log persistence.
                    "    tmpfs:",
                    "      - /app/logs:mode=0775",
                    "    environment:",
                ]
            )
            for item in coord_envs[c]:
                lines.append(f"      - {item}")
            start_srv = (c * plan.server_count) // num_coordinators + 1
            end_srv = ((c + 1) * plan.server_count) // num_coordinators
            lines.append("    depends_on:")
            lines.append("      - redis")
            lines.append("      - postgres")
            for i in range(start_srv, end_srv + 1):
                lines.append(f"      - tfm-server-{i}")
            lines.append(f'    ports:')
            lines.append(f'      - "{coord_port}:5000"')

        all_game_servers = ",".join(f"tfm-server-{i}:8080" for i in range(1, plan.server_count + 1))
        coord_sources = _build_orchestrator_coord_sources(num_coordinators)
        lines.extend(
            [
                "",
                "  rl-champion-orchestrator:",
                "    build:",
                "      context: ./rl-environment",
                "      dockerfile: Dockerfile.training",
                "    command: python champion_orchestrator.py",
                "    restart: unless-stopped",
                "    volumes:",
                "      - ./rl-environment:/app",
                "      - ./rl-models-global:/app/rl-models-global",
                "      - ./card_metadata.json:/app/card_metadata.json:ro",
            ]
        )
        for idx in range(1, num_coordinators + 1):
            lines.append(f"      - ./rl-models-coord-{idx}:/app/coord-models/coord-{idx}")
        lines.extend(
            [
                "    tmpfs:",
                "      - /app/logs:mode=0775",
                "    environment:",
                "      - GAME_SERVERS=" + all_game_servers,
                "      - AGENT_INFERENCE_DEVICE=cpu",
                "      - ORCH_TRAINER_COORD_ID=coord-1",
                "      - ORCH_COORD_SOURCES=" + coord_sources,
                "      - ORCH_OUTPUT_ROOT=/app/rl-models-global",
                "      - ORCH_POLL_INTERVAL_SEC=20",
                "      - ORCH_TRIGGER_EVERY_N_GENS=1",
                "      - ORCH_TOP_K_PER_COORD=2",
                "      - ORCH_GAMES_PER_CANDIDATE=6",
                "      - ORCH_GLOBAL_GAME_CONCURRENCY=2",
                "      - ORCH_TOURNAMENT_CONCURRENCY=2",
                "      - ORCH_MIN_GAMES_FOR_PROMOTION=24",
                "      - ORCH_MIN_COMPLETION_RATE=0.90",
                "      - ORCH_WIN_RATE_MARGIN=0.03",
                "      - ORCH_KEEP_GENERATIONS_TRAINER=30",
                "      - ORCH_KEEP_GENERATIONS_WORKER=15",
                "      - TOURNAMENT_CONCURRENCY=2",
                "      - GLOBAL_GAME_CONCURRENCY=2",
                "    depends_on:",
                "      - redis",
                "      - postgres",
            ]
        )
        for i in range(1, plan.server_count + 1):
            lines.append(f"      - tfm-server-{i}")
        for idx in range(1, num_coordinators + 1):
            lines.append(f"      - rl-coordinator-{idx}")
    else:
        coord_env = coord_envs[0] if coord_envs else env_items
        lines.extend(
            [
                "",
                "  rl-coordinator:",
                "    build:",
                "      context: ./rl-environment",
                "      dockerfile: Dockerfile.training",
                "    restart: unless-stopped",
                "    deploy:",
                "      resources:",
                "        reservations:",
                "          devices:",
                "            - driver: nvidia",
                "              count: 1",
                "              capabilities: [gpu]",
                "    volumes:",
                "      - ./rl-environment:/app",
                "      - ./rl-models:/app/rl-models",
                "      - ./rl-models-global:/app/rl-models-global",
                "      - ./card_metadata.json:/app/card_metadata.json:ro",
                # tmpfs for logs avoids host-disk I/O on every log write under high concurrency.
                "    tmpfs:",
                "      - /app/logs:mode=0775",
                "    environment:",
            ]
        )
        for item in coord_env:
            lines.append(f"      - {item}")
        lines.extend(
            [
                "    depends_on:",
            ]
        )
        for i in range(1, plan.server_count + 1):
            lines.append(f"      - tfm-server-{i}")
        lines.extend(
            [
                "      - redis",
                "      - postgres",
                "    ports:",
                f'      - "{int(coordinator_base_port)}:5000"',
            ]
        )

    lines.extend(
        [
            "",
            "volumes:",
            "  postgres_data_cloud:",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dynamic cloud compose for RL training.")
    parser.add_argument("--base-compose", default="docker-compose.rl_hard.yml")
    parser.add_argument("--output", default="docker-compose.rl_cloud.generated.yml")
    parser.add_argument("--public-host", default=os.getenv("PUBLIC_HOST", "localhost"))
    parser.add_argument("--base-port", type=int, default=int(os.getenv("TM_BASE_PORT", "8081")))
    parser.add_argument(
        "--coordinator-base-port",
        type=int,
        default=int(os.getenv("RL_COORDINATOR_BASE_PORT", "5000")),
        help="First host port for coordinator dashboard API (multi: base..base+N-1, single: base)",
    )
    parser.add_argument("--min-servers", type=int, default=int(os.getenv("RL_MIN_SERVERS", "4")))
    parser.add_argument("--max-servers", type=int, default=int(os.getenv("RL_MAX_SERVERS", "128")))
    parser.add_argument("--server-mem-mb", type=int, default=int(os.getenv("RL_SERVER_MEM_MB", "1400")))
    parser.add_argument("--node-heap-mb", type=int, default=int(os.getenv("RL_NODE_HEAP_MB", "0")))
    parser.add_argument("--games-per-server", type=int, default=int(os.getenv("RL_GAMES_PER_SERVER", "0")))
    parser.add_argument(
        "--http-connector-limit",
        type=int,
        default=int(os.getenv("RL_HTTP_CONNECTOR_LIMIT", "0")),
    )
    parser.add_argument(
        "--http-connector-limit-per-host",
        type=int,
        default=int(os.getenv("RL_HTTP_CONNECTOR_LIMIT_PER_HOST", "0")),
    )
    parser.add_argument("--reserve-mb", type=int, default=int(os.getenv("RL_RESERVE_MB", "0")))
    parser.add_argument("--infra-overhead-mb", type=int, default=int(os.getenv("RL_INFRA_OVERHEAD_MB", "4096")))
    parser.add_argument("--cpu-server-ratio", type=float, default=float(os.getenv("RL_CPU_SERVER_RATIO", "2.0")))
    parser.add_argument(
        "--training-profile",
        choices=["balanced", "saturate"],
        default=os.getenv("RL_TRAINING_PROFILE", "balanced"),
    )
    parser.add_argument("--population-size", type=int, default=int(os.getenv("RL_POPULATION_SIZE", "0")))
    parser.add_argument("--tournament-size", type=int, default=int(os.getenv("RL_TOURNAMENT_SIZE", "0")))
    parser.add_argument("--games-per-eval", type=int, default=int(os.getenv("RL_GAMES_PER_EVAL", "0")))
    parser.add_argument("--games-per-eval-factor", type=float, default=float(os.getenv("RL_GAMES_PER_EVAL_FACTOR", "0")))
    parser.add_argument("--target-game-waves", type=float, default=float(os.getenv("RL_TARGET_GAME_WAVES", "0")))
    parser.add_argument("--ppo-rollout-steps", type=int, default=int(os.getenv("RL_PPO_ROLLOUT_STEPS", "0")))
    parser.add_argument("--ppo-rollout-factor", type=float, default=float(os.getenv("RL_PPO_ROLLOUT_FACTOR", "0")))
    parser.add_argument(
        "--ppo-min-steps-per-agent",
        type=int,
        default=int(os.getenv("RL_PPO_MIN_STEPS_PER_AGENT", "0")),
    )
    parser.add_argument(
        "--ppo-buffer-max-steps",
        type=int,
        default=int(os.getenv("RL_PPO_BUFFER_MAX_STEPS", "0")),
    )
    parser.add_argument(
        "--agent-poll-interval-sec",
        type=float,
        default=float(os.getenv("RL_AGENT_POLL_INTERVAL_SEC", "-1")),
    )
    parser.add_argument(
        "--agent-failure-pause-sec",
        type=float,
        default=float(os.getenv("RL_AGENT_FAILURE_PAUSE_SEC", "-1")),
    )
    parser.add_argument(
        "--agent-post-move-sleep-sec",
        type=float,
        default=float(os.getenv("RL_AGENT_POST_MOVE_SLEEP_SEC", "-1")),
    )
    parser.add_argument(
        "--agent-inference-batch-deadline-ms",
        type=float,
        default=float(os.getenv("RL_AGENT_INFERENCE_BATCH_DEADLINE_MS", "1.0")),
    )
    parser.add_argument(
        "--initial-cards-jitter-ms",
        type=int,
        default=int(os.getenv("RL_INITIAL_CARDS_JITTER_MS", "-1")),
    )
    parser.add_argument(
        "--tournament-concurrency",
        type=int,
        default=int(os.getenv("RL_TOURNAMENT_CONCURRENCY", "0")),
        help="Override tournament concurrency (0 = auto from capacity)",
    )
    parser.add_argument(
        "--agent-inference-threads",
        type=int,
        default=int(os.getenv("RL_AGENT_INFERENCE_THREADS", "-1")),
        help="Thread pool size for agent inference (0 = auto, -1 = use agent default)",
    )
    parser.add_argument(
        "--server-slot-wait-timeout-sec",
        type=int,
        default=int(os.getenv("RL_TM_SERVER_SLOT_WAIT_TIMEOUT_SEC", "0")),
        help="Timeout when waiting for server slot (0 = use base/saturate default)",
    )
    parser.add_argument(
        "--create-game-retry-attempts",
        type=int,
        default=int(os.getenv("RL_TM_CREATE_GAME_RETRY_ATTEMPTS", "0")),
        help="Retries for create_game (0 = use base/saturate default)",
    )
    parser.add_argument(
        "--http-request-timeout-sec",
        type=int,
        default=int(os.getenv("RL_TM_HTTP_REQUEST_TIMEOUT_SEC", "0")),
        help="HTTP request total timeout (0 = use base/saturate default)",
    )
    parser.add_argument(
        "--get-state-retry-attempts",
        type=int,
        default=int(os.getenv("RL_TM_GET_STATE_RETRY_ATTEMPTS", "0")),
        help="get_player_state retries (0 = use base/saturate default)",
    )
    parser.add_argument(
        "--tm-game-timeout-sec",
        type=int,
        default=int(os.getenv("RL_TM_GAME_TIMEOUT_SEC", "0")),
        help="Game timeout in seconds (0 = use default 420); increase if games cancelled unexpectedly",
    )
    parser.add_argument(
        "--tm-http-force-close-connections",
        type=int,
        default=int(os.getenv("RL_TM_HTTP_FORCE_CLOSE_CONNECTIONS", "0")),
        help="1=close HTTP connection after each request (base default); 0=keep-alive for better async I/O (recommended for cloud)",
    )
    parser.add_argument(
        "--tm-recycle-session-on-disconnect",
        type=int,
        default=int(os.getenv("RL_TM_RECYCLE_SESSION_ON_DISCONNECT", "0")),
        help="1=recycle shared aiohttp session on disconnect errors; 0=disable (recommended for high-concurrency cloud)",
    )
    parser.add_argument(
        "--send-input-transport-retry-attempts",
        type=int,
        default=int(os.getenv("RL_TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS", "0")),
        help="Retries for send_player_input transport errors (0 = use profile default)",
    )
    parser.add_argument(
        "--send-input-transport-retry-attempts-initial",
        type=int,
        default=int(os.getenv("RL_TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL", "0")),
        help="Retries for initialCards transport errors (0 = use profile default)",
    )
    parser.add_argument(
        "--num-coordinators",
        type=int,
        default=int(os.getenv("RL_NUM_COORDINATORS", "0")),
        help="Number of coordinator replicas (0 = auto from --coordinator-games-target)",
    )

    parser.add_argument(
    "--coordinator-games-target",
    type=int,
    default=int(os.getenv("RL_COORDINATOR_GAMES_TARGET", "0")),
    help="Max concurrent games per coordinator when auto-computing count (0 = profile default: 20/28)",
    )
    parser.add_argument(
        "--coordinators-share-gpu",
        type=int,
        default=int(os.getenv("RL_COORDINATORS_SHARE_GPU", "0")),
        help="When 1 and num_coordinators>1, all coordinators share one GPU via device_ids (for pipeline bottleneck)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=int(os.getenv("RL_GPU_INDEX", "0")),
        help="GPU index to use when coordinators share GPU (e.g. 0 -> device_ids: ['0'])",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capacity = build_capacity_plan(args)
    # Auto-compute num_coordinators unless explicitly overridden
    profile = str(args.training_profile or "balanced").strip().lower()
    explicit_num_coordinators = max(0, int(args.num_coordinators))
    if explicit_num_coordinators > 0:
        num_coordinators = explicit_num_coordinators
    else:
        coordinator_games_target = int(args.coordinator_games_target)
        if coordinator_games_target <= 0:
            coordinator_games_target = 20 if profile == "saturate" else 28
        num_coordinators = _auto_num_coordinators(
            global_game_concurrency=capacity.global_game_concurrency,
            profile=profile,
            coordinator_games_target=coordinator_games_target,
        )

    base_env = _extract_env_from_base_compose(Path(args.base_compose))
    base_env_map = _env_list_to_map(base_env)
    training = build_training_plan(args, capacity=capacity, base_env_map=base_env_map)
    base_env_filtered = [
        item
        for item in base_env
        if item.split("=", 1)[0].strip() not in DYNAMIC_ENV_KEYS
    ]

    if num_coordinators > 1:
        coord_envs = []
        for c in range(num_coordinators):
            start_srv = (c * capacity.server_count) // num_coordinators + 1
            end_srv = ((c + 1) * capacity.server_count) // num_coordinators
            if start_srv > end_srv:
                continue
            models_subdir = f"rl-models-coord-{c + 1}"
            dynamic_env = _build_dynamic_env(
                capacity=capacity,
                training=training,
                args=args,
                public_host=args.public_host,
                base_port=args.base_port,
                server_range=(start_srv, end_srv),
                rl_models_subdir=models_subdir,
                num_coordinators=num_coordinators,
            )
            coord_env = _merge_env_lists(dynamic_env, base_env_filtered, ESSENTIAL_ENV_DEFAULTS)
            coord_env = _apply_env_overrides(coord_env, _coordinator_role_env_overrides(c))
            coord_envs.append(coord_env)
        env_items = coord_envs[0]
    else:
        dynamic_env = _build_dynamic_env(
            capacity=capacity,
            training=training,
            args=args,
            public_host=args.public_host,
            base_port=args.base_port,
            num_coordinators=num_coordinators,
        )
        env_items = _merge_env_lists(dynamic_env, base_env_filtered, ESSENTIAL_ENV_DEFAULTS)
        coord_envs = None

    coordinators_share_gpu = bool(args.coordinators_share_gpu)
    compose_text = _render_compose(
        capacity,
        env_items=env_items,
        base_port=args.base_port,
        coordinator_base_port=max(1, int(args.coordinator_base_port)),
        num_coordinators=num_coordinators,
        coord_envs=coord_envs,
        coordinators_share_gpu=coordinators_share_gpu,
        gpu_index=max(0, int(args.gpu_index)),
    )
    with Path(args.output).open("w", encoding="utf-8", newline="\n") as f:
        f.write(compose_text)

    print("Generated compose file:", args.output)
    print(
        "Host capacity: RAM=%d MB, CPUs=%d -> servers=%d, games/server=%d, global_concurrency=%d"
        % (
            capacity.total_ram_mb,
            capacity.cpu_count,
            capacity.server_count,
            capacity.games_per_server,
            capacity.global_game_concurrency,
        )
    )
    print(
        "Per server: mem_limit=%d MB, node_heap=%d MB | HTTP limits: total=%d per_host=%d"
        % (
            capacity.server_mem_mb,
            capacity.node_heap_mb,
            capacity.http_limit,
            capacity.http_limit_per_host,
        )
    )
    print(
        "Training profile=%s: POPULATION_SIZE=%d, TOURNAMENT_SIZE=%d, GAMES_PER_EVAL=%d "
        "(expected agent-games/gen=%d, game instances/gen=%d)"
        % (
            training.profile,
            training.population_size,
            training.tournament_size,
            training.games_per_eval,
            training.expected_games_per_generation,
            training.game_instances_per_generation,
        )
    )
    print(
        "PPO: PPO_ROLLOUT_STEPS=%d, PPO_MIN_STEPS_PER_AGENT=%d, PPO_BUFFER_MAX_STEPS=%d"
        % (
            training.ppo_rollout_steps,
            training.ppo_min_steps_per_agent,
            training.ppo_buffer_max_steps,
        )
    )
    games_per_coord = capacity.global_game_concurrency / num_coordinators
    poll = args.agent_poll_interval_sec if args.agent_poll_interval_sec >= 0 else (0.03 if profile == "saturate" else 0.2)
    print(
        "Coordinators: %d (%.1f games each) — ~%.0f req/s per event loop"
        % (num_coordinators, games_per_coord, games_per_coord / max(0.001, poll))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
