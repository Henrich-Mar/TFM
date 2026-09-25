"""Seat-rotated reachability teacher evaluation, before collecting any new data."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import mean

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy
from tournament_manager import TournamentManager
from training.v2_benchmark import mean_interval
from v2_runtime import initialize_v2_runtime


async def evaluate(output: str, games: int = 20, seed_start: int = 940000) -> dict:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    if games < 20 or games % 4:
        raise ValueError("games must be at least 20 and divisible by four for seat rotation")
    if seed_start < 940000 or seed_start + games > 950000:
        raise ValueError("teacher A/B seeds must stay in 940000..949999")
    initialize_v2_runtime()
    cluster = GameServerCluster([s.strip() for s in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if s.strip()])
    cluster.base_game_options = json.loads((Path(__file__).resolve().parents[1] / "game_options.v2_stage1.json").read_text())
    manager = TournamentManager(cluster)
    ranks = []
    rejections = 0
    results = []
    try:
        for offset, seed in enumerate(range(seed_start, seed_start + games)):
            seat = offset % 4
            candidate = RLAgent(agent_id=f"reachability-{seed}-{seat}", decision_policy=HeuristicTeacherPolicy(seed, sample=False, reachability=True))
            opponents = [RLAgent(agent_id=f"old-{seed}-{seat}-{i}", decision_policy=HeuristicTeacherPolicy(seed + i + 1, sample=False, reachability=False)) for i in range(3)]
            lineup = list(opponents)
            lineup.insert(seat, candidate)
            for agent in lineup:
                agent.train_from_self_play = False
                agent.config.train_from_self_play = False
                agent.ppo_enable = False
            result = await manager._run_single_game(
                lineup, tournament_id=f"v4_teacher_ab_{seed}_{seat}", game_seed=seed, players_beginner=False,
            )
            rejections += sum(int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in lineup)
            if result.completed:
                rank = int(next(row for row in result.players if row.get("agent_id") == candidate.id)["rank"])
                ranks.append(rank)
                results.append({"seed": seed, "seat": seat, "rank": rank})
            print(f"[teacher-ab] seed={seed} seat={seat} completed={result.completed} mean_rank={mean(ranks) if ranks else 0:.3f}", flush=True)
    finally:
        await cluster.close()
    low, high = mean_interval([float(rank) for rank in ranks])
    observed = mean(ranks) if ranks else 4.0
    decision = "incomplete" if len(ranks) != games or rejections else (
        "collect" if observed < 2.35 else "retune" if observed > 2.65 else "run_40_more"
    )
    report = {
        "schema_version": "v4.teacher_ab.v1", "baseline": "old_teacher", "stage": 1,
        "planned_games": games, "completed_games": len(ranks), "rejection_count": rejections,
        "mean_rank": observed, "mean_rank_lower_95": low, "mean_rank_upper_95": high,
        "decision": decision, "games": results,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=940000)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args.output, args.games, args.seed_start)), indent=2))
