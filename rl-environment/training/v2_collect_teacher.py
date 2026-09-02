"""Generate fresh base-game teacher demonstrations for TFM RL v2."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy
from tournament_manager import TournamentManager
from training.teacher_dataset import (
    TeacherDatasetRecorder,
    TeacherDatasetStore,
    load_reserved_benchmark_seeds,
)
from v2_runtime import initialize_v2_runtime


async def collect(
    dataset_dir: str,
    games: int,
    stage: int,
    seed_start: int,
    serve_api: bool = False,
    api_host: str = "0.0.0.0",
    api_port: int = 5000,
) -> dict:
    initialize_v2_runtime()
    reserved = load_reserved_benchmark_seeds()
    seeds = [int(seed_start) + idx for idx in range(max(1, int(games)))]
    overlap = reserved.intersection(seeds)
    if overlap:
        raise RuntimeError(f"teacher seeds overlap reserved benchmark seeds: {sorted(overlap)}")
    store = TeacherDatasetStore(dataset_dir)
    recorder = TeacherDatasetRecorder(store)
    servers = [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
    cluster = GameServerCluster(servers)
    manager = TournamentManager(cluster)
    api_server = None
    api_task = None
    if serve_api:
        import uvicorn
        from api.server import app

        api_server = uvicorn.Server(
            uvicorn.Config(app, host=str(api_host), port=int(api_port), log_level="info")
        )
        api_task = asyncio.create_task(api_server.serve())
        while not api_server.started and not api_task.done():
            await asyncio.sleep(0.05)
        if api_task.done():
            await api_task
    agents = []
    for seat in range(4):
        agent = RLAgent(
            agent_id=f"teacher-v1-seat-{seat}",
            decision_policy=HeuristicTeacherPolicy(seed=seed_start + seat, temperature=0.18, sample=True),
            decision_recorder=recorder,
        )
        agent.train_from_self_play = False
        agent.config.train_from_self_play = False
        agent.ppo_enable = False
        agents.append(agent)
    completed = 0
    failed = 0
    try:
        for idx, seed in enumerate(seeds):
            game_number = idx + 1
            game_started_at = time.monotonic()
            print(
                f"[teacher] starting game {game_number}/{len(seeds)} (seed={seed})",
                flush=True,
            )
            result = await manager._run_single_game(
                agents,
                tournament_id=f"v2_teacher_stage{stage}_{idx}",
                game_seed=seed,
                players_beginner=(int(stage) == 0),
            )
            if bool(result.completed):
                completed += 1
                outcome = "completed"
            else:
                failed += 1
                outcome = f"failed: {result.error_message or 'unknown error'}"
            total_decisions_so_far = sum(
                int(getattr(agent.decision_policy, "decisions", 0)) for agent in agents
            )
            print(
                f"[teacher] {game_number}/{len(seeds)} {outcome} "
                f"game_id={result.game_id} "
                f"elapsed={time.monotonic() - game_started_at:.1f}s "
                f"completed={completed} failed={failed} "
                f"decisions={total_decisions_so_far}",
                flush=True,
            )
    finally:
        await cluster.close()
        if api_server is not None and api_task is not None:
            api_server.should_exit = True
            await api_task
    fallback_decisions = sum(int(getattr(agent.decision_policy, "fallbacks", 0)) for agent in agents)
    total_decisions = sum(int(getattr(agent.decision_policy, "decisions", 0)) for agent in agents)
    rejection_count = sum(int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in agents)
    report = {
        "games": len(seeds),
        "completed": completed,
        "failed": failed,
        "completion_rate": completed / len(seeds),
        "teacher_decisions": total_decisions,
        "teacher_fallback_rate": (fallback_decisions / total_decisions) if total_decisions else 0.0,
        "teacher/fallback": fallback_decisions,
        "teacher/fallback_rate": (fallback_decisions / total_decisions) if total_decisions else 0.0,
        "rejection_count": rejection_count,
        "dataset_counts": store.counts(),
    }
    report["smoke_gate_passed"] = bool(
        len(seeds) >= 100
        and report["completion_rate"] >= 0.99
        and report["teacher_fallback_rate"] < 0.05
        and rejection_count == 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TFM RL v2 heuristic-teacher games")
    parser.add_argument("--dataset", default=os.getenv("V2_TEACHER_DATASET_DIR", "/app/v2/teacher-dataset"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument(
        "--serve-api",
        action="store_true",
        help="Serve Decision Explainer in this process so capture requests reach the teacher agents",
    )
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=5000)
    args = parser.parse_args()
    expected_options = f"game_options.v2_stage{args.stage}.json"
    if expected_options not in str(os.getenv("GAME_OPTIONS_FILE", "")):
        raise RuntimeError(f"GAME_OPTIONS_FILE must point to {expected_options}")
    print(
        json.dumps(
            asyncio.run(
                collect(
                    args.dataset,
                    args.games,
                    args.stage,
                    args.seed_start,
                    serve_api=args.serve_api,
                    api_host=args.api_host,
                    api_port=args.api_port,
                )
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
