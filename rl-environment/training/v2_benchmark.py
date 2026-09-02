"""Fixed-seed, seat-rotated acceptance benchmark for TFM RL v2."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy, RandomLegalPolicy
from tournament_manager import TournamentManager
from v2_runtime import initialize_v2_runtime


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = float(successes) / float(trials)
    denom = 1.0 + (z * z / trials)
    center = p + (z * z / (2.0 * trials))
    spread = z * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials)))
    return max(0.0, (center - spread) / denom), min(1.0, (center + spread) / denom)


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    return wilson_interval(successes, trials, z)[0]


def mean_interval(values: List[float], z: float = 1.959963984540054) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    center = mean(values)
    if len(values) < 2:
        return float(center), float(center)
    half_width = float(z) * (stdev(values) / math.sqrt(len(values)))
    return float(center - half_width), float(center + half_width)


def _load_seeds(path: Optional[str] = None) -> List[int]:
    source = Path(path).expanduser().resolve() if path else Path(__file__).resolve().parents[1] / "benchmark_seeds.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    seeds = [int(item) for item in payload.get("seeds", [])]
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise RuntimeError("v2 benchmark requires exactly 30 unique reserved seeds")
    return seeds


def _frozen_neural(checkpoint: str, agent_id: str) -> RLAgent:
    agent = RLAgent(agent_id=agent_id)
    agent.load_model(checkpoint)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.deterministic_actions = True
    return agent


def _baseline_agents(kind: str, seed: int, champion: Optional[str] = None) -> List[RLAgent]:
    agents: List[RLAgent] = []
    for idx in range(3):
        if kind == "random":
            agent = RLAgent(agent_id=f"random-{idx}", decision_policy=RandomLegalPolicy(seed + idx))
        elif kind == "teacher":
            agent = RLAgent(agent_id=f"teacher-{idx}", decision_policy=HeuristicTeacherPolicy(seed + idx, sample=False))
        elif kind == "champion" and champion:
            agent = _frozen_neural(champion, f"champion-{idx}")
        else:
            raise ValueError(f"invalid baseline: {kind}")
        agent.train_from_self_play = False
        agent.config.train_from_self_play = False
        agent.ppo_enable = False
        agents.append(agent)
    return agents


async def benchmark(
    checkpoint: str,
    baseline: str,
    stage: int,
    output_dir: str,
    seeds_path: Optional[str] = None,
    champion: Optional[str] = None,
) -> Dict[str, Any]:
    initialize_v2_runtime()
    candidate = _frozen_neural(checkpoint, "v2-candidate")
    seeds = _load_seeds(seeds_path)
    cluster = GameServerCluster([item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()])
    options_path = Path(__file__).resolve().parents[1] / f"game_options.v2_stage{int(stage)}.json"
    cluster.base_game_options = json.loads(options_path.read_text(encoding="utf-8"))
    manager = TournamentManager(cluster)
    ranks: List[int] = []
    vp_margins: List[float] = []
    pairwise_points = 0.0
    pairwise_trials = 0
    completed = 0
    opponents = _baseline_agents(baseline, seeds[0], champion=champion)
    all_agents = [candidate, *opponents]
    rejection_count_before = sum(
        int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in all_agents
    )
    try:
        for seed in seeds:
            for candidate_seat in range(4):
                # Keep stochastic RandomLegal baselines reproducible across complete
                # benchmark reruns instead of depending on prior games in this process.
                for opponent_index, opponent in enumerate(opponents):
                    policy = getattr(opponent, "decision_policy", None)
                    if isinstance(policy, RandomLegalPolicy):
                        policy.rng.seed((int(seed) * 1009) + (candidate_seat * 17) + opponent_index)
                lineup: List[RLAgent] = list(opponents)
                lineup.insert(candidate_seat, candidate)
                result = await manager._run_single_game(
                    lineup,
                    tournament_id=f"v2_benchmark_{baseline}_{seed}_{candidate_seat}",
                    game_seed=seed,
                    players_beginner=(int(stage) == 0),
                )
                if not bool(result.completed):
                    continue
                completed += 1
                candidate_row = next(row for row in result.players if str(row.get("agent_id")) == candidate.id)
                rank = int(candidate_row.get("rank", 4) or 4)
                vp = float(candidate_row.get("victory_points", 0.0) or 0.0)
                table_mean = mean(float(row.get("victory_points", 0.0) or 0.0) for row in result.players)
                ranks.append(rank)
                vp_margins.append(vp - table_mean)
                for opponent_row in result.players:
                    if str(opponent_row.get("agent_id")) == candidate.id:
                        continue
                    opponent_rank = int(opponent_row.get("rank", 4) or 4)
                    pairwise_points += 1.0 if rank < opponent_rank else (0.5 if rank == opponent_rank else 0.0)
                    pairwise_trials += 1
    finally:
        await cluster.close()
    total = len(seeds) * 4
    wins = sum(1 for rank in ranks if rank == 1)
    first_place_rate = wins / completed if completed else 0.0
    rejection_count = sum(
        int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in all_agents
    ) - rejection_count_before
    wilson_low, wilson_high = wilson_interval(wins, completed)
    rank_low, rank_high = mean_interval([float(item) for item in ranks])
    vp_low, vp_high = mean_interval(vp_margins)
    if baseline == "random" and int(stage) == 0:
        gate_passed = completed >= math.ceil(0.99 * total) and rejection_count == 0 and first_place_rate >= 0.55
    elif baseline == "teacher" and int(stage) == 1:
        gate_passed = completed >= math.ceil(0.99 * total) and rejection_count == 0 and wilson_lower(wins, completed) > 0.25
    else:
        gate_passed = completed >= math.ceil(0.99 * total) and rejection_count == 0 and (pairwise_points / max(1, pairwise_trials)) >= 0.50
    report = {
        "schema_version": "tfm_rl_v2.benchmark.v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "baseline": baseline,
        "stage": int(stage),
        "planned_games": total,
        "completed_games": completed,
        "completion_rate": completed / total,
        "rejection_count": rejection_count,
        "first_place_rate": first_place_rate,
        "first_place_wilson_lower_95": wilson_low,
        "first_place_wilson_upper_95": wilson_high,
        "mean_rank": mean(ranks) if ranks else 4.0,
        "mean_rank_lower_95": rank_low,
        "mean_rank_upper_95": rank_high,
        "mean_relative_vp_margin": mean(vp_margins) if vp_margins else 0.0,
        "mean_relative_vp_margin_lower_95": vp_low,
        "mean_relative_vp_margin_upper_95": vp_high,
        "pairwise_score": pairwise_points / max(1, pairwise_trials),
        "gate_passed": bool(gate_passed),
        "seeds": seeds,
        "seat_rotations": 4,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_token = Path(checkpoint).stem.replace(" ", "_")
    target = output / f"benchmark_{checkpoint_token}_stage{stage}_{baseline}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed TFM RL v2 acceptance benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", choices=("random", "teacher", "champion"), required=True)
    parser.add_argument("--champion")
    parser.add_argument("--stage", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--output", default=os.getenv("V2_BENCHMARK_DIR", "/app/v2/benchmarks"))
    args = parser.parse_args()
    if args.baseline == "champion" and not args.champion:
        parser.error("--champion is required for champion baseline")
    print(json.dumps(asyncio.run(benchmark(args.checkpoint, args.baseline, args.stage, args.output, args.seeds, args.champion)), indent=2))


if __name__ == "__main__":
    main()
