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
from typing import Dict, Iterable, List


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
    "POPULATION_SIZE",
    "TOURNAMENT_SIZE",
    "GAMES_PER_EVAL",
    "PPO_ROLLOUT_STEPS",
    "PPO_MIN_STEPS_PER_AGENT",
    "PPO_BUFFER_MAX_STEPS",
    "AGENT_POLL_INTERVAL_SEC",
    "AGENT_FAILURE_PAUSE_SEC",
    "AGENT_POST_MOVE_SLEEP_SEC",
    "TM_SEND_INPUT_INITIAL_CARDS_JITTER_MS",
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
    tournament_concurrency = max(1, min(global_game_concurrency, cpu_count * 2))

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
) -> List[str]:
    game_servers = ",".join(f"tfm-server-{i}:8080" for i in range(1, capacity.server_count + 1))
    public_tm_map = ",".join(
        f"tfm-server-{i}:8080=http://{public_host}:{base_port + i - 1}"
        for i in range(1, capacity.server_count + 1)
    )
    public_tm_server_id_map = ",".join(
        f"tfm-server-{i}:8080=tfm-server-{i}"
        for i in range(1, capacity.server_count + 1)
    )
    public_tm_url = ",".join(
        f"http://{public_host}:{base_port + i - 1}"
        for i in range(1, capacity.server_count + 1)
    )
    internal_tm_url = ",".join(
        f"http://tfm-server-{i}:8080"
        for i in range(1, capacity.server_count + 1)
    )

    env_items = [
        f"GAME_SERVERS={game_servers}",
        f"PUBLIC_TM_MAP={public_tm_map}",
        f"PUBLIC_TM_SERVER_ID_MAP={public_tm_server_id_map}",
        f"PUBLIC_TM_URL={public_tm_url}",
        f"INTERNAL_TM_URL={internal_tm_url}",
        f"TOURNAMENT_CONCURRENCY={capacity.tournament_concurrency}",
        f"GLOBAL_GAME_CONCURRENCY={capacity.global_game_concurrency}",
        f"MAX_ACTIVE_GAMES_PER_SERVER={capacity.games_per_server}",
        f"TM_HTTP_CONNECTOR_LIMIT={capacity.http_limit}",
        f"TM_HTTP_CONNECTOR_LIMIT_PER_HOST={capacity.http_limit_per_host}",
        f"POPULATION_SIZE={training.population_size}",
        f"TOURNAMENT_SIZE={training.tournament_size}",
        f"GAMES_PER_EVAL={training.games_per_eval}",
        f"PPO_ROLLOUT_STEPS={training.ppo_rollout_steps}",
        f"PPO_MIN_STEPS_PER_AGENT={training.ppo_min_steps_per_agent}",
        f"PPO_BUFFER_MAX_STEPS={training.ppo_buffer_max_steps}",
    ]

    poll_interval = float(args.agent_poll_interval_sec)
    if poll_interval < 0:
        poll_interval = 0.08 if training.profile == "saturate" else -1.0
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

    initial_cards_jitter_ms = int(args.initial_cards_jitter_ms)
    if initial_cards_jitter_ms < 0:
        initial_cards_jitter_ms = 250 if training.profile == "saturate" else -1
    if initial_cards_jitter_ms >= 0:
        env_items.append(f"TM_SEND_INPUT_INITIAL_CARDS_JITTER_MS={initial_cards_jitter_ms}")

    return env_items


def _render_compose(plan: CapacityPlan, env_items: List[str], base_port: int) -> str:
    lines: List[str] = []
    lines.append("services:")

    for i in range(1, plan.server_count + 1):
        host_port = base_port + i - 1
        lines.extend(
            [
                f"  tfm-server-{i}:",
                "    build:",
                "      context: ./terraforming-mars",
                "      dockerfile: ../Dockerfile.rl",
                "    ports:",
                f"      - \"{host_port}:8080\"",
                "    environment:",
                "      - NODE_ENV=production",
                "      - COMPRESS_COMPLETED_GAMES_DAYS=0",
                f"      - SERVER_ID=tfm-server-{i}",
                f"      - NODE_OPTIONS=--max-old-space-size={plan.node_heap_mb}",
                "    tmpfs:",
                "      - /usr/src/app/db:uid=100,gid=65533,mode=0775",
                "    labels:",
                "      - autoheal=true",
                "    healthcheck:",
                "      test:",
                "        - CMD-SHELL",
                "        - node -e \"require('http').get('http://127.0.0.1:8080/',(r)=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))\"",
                "      interval: 15s",
                "      timeout: 8s",
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
            "    ports:",
            "      - \"5432:5432\"",
            "    restart: unless-stopped",
            "    volumes:",
            "      - postgres_data_cloud:/var/lib/postgresql/data",
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
            "      - ./rl-logs:/app/logs",
            "      - ./card_metadata.json:/app/card_metadata.json:ro",
            "    environment:",
        ]
    )

    for item in env_items:
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
            "      - \"5000:5000\"",
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
    parser.add_argument("--min-servers", type=int, default=int(os.getenv("RL_MIN_SERVERS", "4")))
    parser.add_argument("--max-servers", type=int, default=int(os.getenv("RL_MAX_SERVERS", "48")))
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
        "--initial-cards-jitter-ms",
        type=int,
        default=int(os.getenv("RL_INITIAL_CARDS_JITTER_MS", "-1")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capacity = build_capacity_plan(args)

    base_env = _extract_env_from_base_compose(Path(args.base_compose))
    base_env_map = _env_list_to_map(base_env)
    training = build_training_plan(args, capacity=capacity, base_env_map=base_env_map)
    base_env_filtered = [
        item
        for item in base_env
        if item.split("=", 1)[0].strip() not in DYNAMIC_ENV_KEYS
    ]
    dynamic_env = _build_dynamic_env(
        capacity=capacity,
        training=training,
        args=args,
        public_host=args.public_host,
        base_port=args.base_port,
    )
    env_items = _merge_env_lists(dynamic_env, base_env_filtered, ESSENTIAL_ENV_DEFAULTS)

    compose_text = _render_compose(capacity, env_items=env_items, base_port=args.base_port)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
