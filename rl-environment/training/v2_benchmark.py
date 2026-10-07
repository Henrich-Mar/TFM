"""Fixed-seed, seat-rotated acceptance benchmark for TFM RL v2."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.award_override import parse_award_override
from models.decision_policy import (
    AwardFundingTeacherPolicy,
    HeuristicTeacherPolicy,
    RandomLegalPolicy,
)
from search import SearchConfig, SearchPolicy
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


def _teacher_wilson_floor() -> float:
    """Absolute teacher floor; relative-to-champion protection is applied separately."""
    try:
        value = float(os.getenv("BENCHMARK_TEACHER_MIN_WILSON_LOWER", "0.15"))
    except (TypeError, ValueError):
        value = 0.15
    return min(1.0, max(0.0, value))


def _load_seeds(path: Optional[str] = None) -> List[int]:
    source = Path(path).expanduser().resolve() if path else Path(__file__).resolve().parents[1] / "benchmark_seeds.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    seeds = [int(item) for item in payload.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"benchmark seed file must contain non-empty unique seeds: {source}")
    if path is None and len(seeds) != 30:
        raise RuntimeError("v2 promotion benchmark requires exactly 30 unique reserved seeds")
    return seeds


def _frozen_neural(checkpoint: str, agent_id: str, stochastic: bool = False) -> RLAgent:
    agent = RLAgent(agent_id=agent_id)
    agent.load_model(checkpoint)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    # ``ppo_enable`` also selects the behavior-policy path: PPO sampling omits
    # contextual heuristic reweighting so stored/reconstructed log-probs match.
    # Frozen agents cannot train because ``train_from_self_play`` is false, so
    # keep PPO behavior enabled for stochastic evaluation without collecting
    # rollouts or updating weights.
    agent.ppo_enable = bool(stochastic)
    agent.deterministic_actions = not stochastic
    if stochastic:
        # Match strict on-policy self-play sampling: no epsilon-random moves and
        # temperature 1.0, regardless of the checkpoint's exploration schedule.
        agent.config.epsilon = 0.0
        agent.config.temperature = 1.0
        agent.policy_temperature_cap = 1.0
        agent.policy_temperature_floor = 1.0
    return agent


def _bind_frozen_neural(leader: RLAgent, agent_id: str, stochastic: bool = False) -> RLAgent:
    """A frozen seat with its own episode memory on ``leader``'s weights.

    Concurrent workers then share one network and one inference batcher instead
    of loading a private copy of the same checkpoint each.
    """
    agent = RLAgent(agent_id=agent_id, config=leader.config)
    agent.bind_shared_learner(leader)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = bool(stochastic)
    agent.deterministic_actions = not stochastic
    agent.policy_temperature_cap = leader.policy_temperature_cap
    agent.policy_temperature_floor = leader.policy_temperature_floor
    return agent


def _award_gate_floor() -> float:
    """Minimum mean awards a candidate must fund per game against the funder."""
    try:
        value = float(os.getenv("BENCHMARK_AWARD_MIN_FUNDED", "0.10"))
    except (TypeError, ValueError):
        value = 0.10
    return max(0.0, value)


def _baseline_agents(
    kind: str,
    seed: int,
    champion: Optional[str] = None,
    stochastic: bool = False,
    champion_leader: Optional[RLAgent] = None,
) -> List[RLAgent]:
    agents: List[RLAgent] = []
    for idx in range(3):
        if kind == "random":
            agent = RLAgent(agent_id=f"random-{idx}", decision_policy=RandomLegalPolicy(seed + idx))
        elif kind == "teacher":
            agent = RLAgent(agent_id=f"teacher-{idx}", decision_policy=HeuristicTeacherPolicy(seed + idx, sample=False))
        elif kind == "award_teacher":
            # Award-aware opponent. Promotion against the passive teacher is
            # blind to the cheap-award exploit, so this baseline exists to
            # measure it directly.
            agent = RLAgent(
                agent_id=f"award-teacher-{idx}",
                decision_policy=AwardFundingTeacherPolicy(seed + idx, sample=False),
            )
        elif kind == "champion" and champion:
            if champion_leader is not None:
                agent = _bind_frozen_neural(champion_leader, f"champion-{seed}-{idx}", stochastic=stochastic)
            else:
                agent = _frozen_neural(champion, f"champion-{idx}", stochastic=stochastic)
        else:
            raise ValueError(f"invalid baseline: {kind}")
        agent.train_from_self_play = False
        agent.config.train_from_self_play = False
        # Neural baselines already select the correct behavior path in
        # ``_frozen_neural``. External random/teacher policies never use PPO.
        if kind != "champion":
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
    report_label: Optional[str] = None,
    stochastic: bool = False,
    candidate_stochastic: bool = False,
    award_override: Optional[str] = None,
    random_ma: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the seat-rotated benchmark.

    ``stochastic`` makes both the candidate and neural champion sample, which
    is the symmetric policy comparison used by promotion and rollback gates.
    ``candidate_stochastic`` makes only the candidate sample, matching
    candidate training against a deterministic heuristic teacher.
    ``award_override`` is an ``AwardOverrideRule`` spec applied to the
    candidate only, for hybrid "would funding help?" experiments.
    ``random_ma`` ("Limited synergy" / "Full random") draws random awards and
    milestones per game, as in the random-MA self-play cohort.
    """
    initialize_v2_runtime()
    is_v3 = str(os.getenv("TFM_RL_V3", "0")).strip().lower() in {"1", "true", "yes", "on"}
    version = "v3" if is_v3 else "v2"
    candidate_samples = bool(stochastic or candidate_stochastic)
    opponent_samples = bool(stochastic)
    candidate = _frozen_neural(checkpoint, f"{version}-candidate", stochastic=candidate_samples)
    if is_v3:
        candidate.set_v3_feature_scale(1.0)
    override_rule = parse_award_override(award_override)
    candidate.award_override = override_rule
    if override_rule is not None:
        print(f"[benchmark] award override enabled config={override_rule.config()}", flush=True)
    search_config = SearchConfig.from_env()
    if search_config.enabled:
        candidate.search_policy = SearchPolicy(candidate, search_config)
        print(
            f"[benchmark] MCTS search enabled mode={search_config.mode} "
            f"top_k={search_config.top_k} determinizations={search_config.determinizations} "
            f"simulations={search_config.simulations_per_move} depth={search_config.max_root_turns_depth}",
            flush=True,
        )
    seeds = _load_seeds(seeds_path)
    cluster = GameServerCluster([item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()])
    try:
        requested_concurrency = int(os.getenv("BENCHMARK_CONCURRENCY", str(len(cluster.servers))))
    except (TypeError, ValueError):
        requested_concurrency = len(cluster.servers)
    per_server_capacity = max(1, int(getattr(cluster, "max_active_games_per_server", 0) or 1))
    available_server_slots = len(cluster.servers) * per_server_capacity
    concurrency = max(1, min(len(seeds) * 4, available_server_slots, requested_concurrency))
    options_path = Path(__file__).resolve().parents[1] / f"game_options.{version}_stage{int(stage)}.json"
    cluster.base_game_options = json.loads(options_path.read_text(encoding="utf-8"))
    manager = TournamentManager(cluster)
    ranks: List[int] = []
    vp_margins: List[float] = []
    candidate_awards_funded: List[float] = []
    candidate_award_vp: List[float] = []
    opponent_awards_funded: List[float] = []
    opponent_award_vp: List[float] = []
    candidate_milestones: List[float] = []
    games_with_awards_funded = 0
    pairwise_points = 0.0
    pairwise_trials = 0
    completed = 0
    # Each worker owns its baseline agents. This keeps RandomLegalPolicy RNG
    # state deterministic while games run concurrently. The frozen candidate
    # is safely shared, just like the learner in concurrent self-play.
    champion_leader = (
        _frozen_neural(champion, "champion-shared", stochastic=opponent_samples)
        if baseline == "champion" and champion
        else None
    )
    shared = {"champion_leader": champion_leader} if champion_leader is not None else {}
    opponent_pools = [
        _baseline_agents(
            baseline,
            seeds[0] + worker_index,
            champion=champion,
            stochastic=opponent_samples,
            **shared,
        )
        for worker_index in range(concurrency)
    ]
    all_agents = [candidate, *(agent for pool in opponent_pools for agent in pool)]
    total = len(seeds) * 4
    benchmark_started_at = time.monotonic()
    print(
        f"[benchmark] started baseline={baseline} stage={stage} "
        f"games={total} concurrency={concurrency} checkpoint={Path(checkpoint).name} "
        f"candidate={'sample' if candidate_samples else 'argmax'} "
        f"opponents={'sample' if opponent_samples else 'argmax'}",
        flush=True,
    )
    rejection_count_before = sum(
        int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in all_agents
    )
    try:
        jobs = [
            ((seed_index * 4) + candidate_seat + 1, seed, candidate_seat)
            for seed_index, seed in enumerate(seeds)
            for candidate_seat in range(4)
        ]

        async def _run_worker(worker_index: int) -> None:
            nonlocal completed, pairwise_points, pairwise_trials
            nonlocal games_with_awards_funded
            opponents = opponent_pools[worker_index]
            for game_number, seed, candidate_seat in jobs[worker_index::concurrency]:
                game_started_at = time.monotonic()
                print(
                    f"[benchmark] starting baseline={baseline} game={game_number}/{total} "
                    f"seed={seed} candidate_seat={candidate_seat} worker={worker_index + 1}/{concurrency}",
                    flush=True,
                )
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
                    tournament_id=f"{version}_benchmark_{baseline}_{seed}_{candidate_seat}",
                    game_seed=seed,
                    players_beginner=(int(stage) == 0),
                    game_option_overrides=(
                        {"randomMA": str(random_ma), "includeFanMA": False, "modularMA": False}
                        if random_ma
                        else None
                    ),
                )
                if not bool(result.completed):
                    print(
                        f"[benchmark] incomplete baseline={baseline} game={game_number}/{total} "
                        f"seed={seed} elapsed={time.monotonic() - game_started_at:.1f}s",
                        flush=True,
                    )
                    continue
                completed += 1
                candidate_row = next(row for row in result.players if str(row.get("agent_id")) == candidate.id)
                rank = int(candidate_row.get("rank", 4) or 4)
                vp = float(candidate_row.get("victory_points", 0.0) or 0.0)
                table_mean = mean(float(row.get("victory_points", 0.0) or 0.0) for row in result.players)
                ranks.append(rank)
                vp_margins.append(vp - table_mean)
                table_funded = int(candidate_row.get("awards_funded_table", 0) or 0)
                games_with_awards_funded += 1 if table_funded > 0 else 0
                candidate_awards_funded.append(float(candidate_row.get("awards_funded", 0) or 0))
                candidate_award_vp.append(float(candidate_row.get("vp_awards", 0) or 0))
                candidate_milestones.append(float(candidate_row.get("milestones_claimed", 0) or 0))
                for opponent_row in result.players:
                    if str(opponent_row.get("agent_id")) == candidate.id:
                        continue
                    opponent_rank = int(opponent_row.get("rank", 4) or 4)
                    pairwise_points += 1.0 if rank < opponent_rank else (0.5 if rank == opponent_rank else 0.0)
                    pairwise_trials += 1
                    opponent_awards_funded.append(float(opponent_row.get("awards_funded", 0) or 0))
                    opponent_award_vp.append(float(opponent_row.get("vp_awards", 0) or 0))
                running_wins = sum(1 for item in ranks if item == 1)
                print(
                    f"[benchmark] completed baseline={baseline} game={game_number}/{total} "
                    f"seed={seed} rank={rank} vp={vp:.1f} margin={vp - table_mean:+.1f} "
                    f"wins={running_wins}/{completed} elapsed={time.monotonic() - game_started_at:.1f}s "
                    f"total_elapsed={time.monotonic() - benchmark_started_at:.1f}s",
                    flush=True,
                )
        await asyncio.gather(*(_run_worker(worker_index) for worker_index in range(concurrency)))
    finally:
        await cluster.close()
    wins = sum(1 for rank in ranks if rank == 1)
    first_place_rate = wins / completed if completed else 0.0
    rejection_count = sum(
        int(agent.get_behavior_stats().get("policy_rejections", 0)) for agent in all_agents
    ) - rejection_count_before
    wilson_low, wilson_high = wilson_interval(wins, completed)
    rank_low, rank_high = mean_interval([float(item) for item in ranks])
    vp_low, vp_high = mean_interval(vp_margins)
    teacher_wilson_floor = _teacher_wilson_floor()
    award_gate_floor = _award_gate_floor()
    mean_candidate_awards = mean(candidate_awards_funded) if candidate_awards_funded else 0.0
    award_funding_rate = (
        games_with_awards_funded / completed if completed else 0.0
    )
    award_awareness_gate_passed = mean_candidate_awards >= award_gate_floor
    if baseline == "random" and int(stage) == 0:
        gate_passed = completed >= math.ceil(0.99 * total) and rejection_count == 0 and first_place_rate >= 0.55
    elif baseline == "teacher":
        gate_passed = (
            completed >= math.ceil(0.99 * total)
            and rejection_count == 0
            and wilson_low > teacher_wilson_floor
        )
    elif baseline == "award_teacher":
        # Against an award-aware opponent the candidate must both hold its own
        # pairwise score and actually commit to awards. A candidate that never
        # funds loses 5 VP per 8 MC to any human who does, so funding is gated
        # explicitly instead of being inferred from total VP.
        gate_passed = (
            completed >= math.ceil(0.99 * total)
            and rejection_count == 0
            and (pairwise_points / max(1, pairwise_trials)) >= 0.50
            and award_awareness_gate_passed
        )
    else:
        gate_passed = completed >= math.ceil(0.99 * total) and rejection_count == 0 and (pairwise_points / max(1, pairwise_trials)) >= 0.50
    report = {
        "schema_version": f"tfm_rl_{version}.benchmark.v1",
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
        "award_telemetry": {
            "candidate_mean_awards_funded": mean_candidate_awards,
            "candidate_mean_award_vp": mean(candidate_award_vp) if candidate_award_vp else 0.0,
            "candidate_mean_milestones_claimed": (
                mean(candidate_milestones) if candidate_milestones else 0.0
            ),
            "opponent_mean_awards_funded": (
                mean(opponent_awards_funded) / 3.0 if opponent_awards_funded else 0.0
            ),
            "opponent_mean_award_vp": mean(opponent_award_vp) / 3.0 if opponent_award_vp else 0.0,
            "games_with_any_award_funded": games_with_awards_funded,
            "award_funding_game_rate": award_funding_rate,
            "awareness_gate_passed": bool(award_awareness_gate_passed),
        },
        "gate_thresholds": (
            {
                "teacher_first_place_wilson_lower_min": teacher_wilson_floor,
            }
            if baseline == "teacher"
            else {"min_mean_awards_funded": award_gate_floor}
            if baseline == "award_teacher"
            else {}
        ),
        "seeds": seeds,
        "seat_rotations": 4,
        "concurrency": concurrency,
        "report_label": str(report_label or "promotion"),
        "action_selection": (
            "sample" if opponent_samples else ("candidate_sample" if candidate_samples else "argmax")
        ),
        "search": (
            candidate.search_policy.snapshot_stats()
            if getattr(candidate, "search_policy", None) is not None
            else {"enabled": False}
        ),
        "random_ma": str(random_ma) if random_ma else None,
        "award_override": (
            override_rule.snapshot() if override_rule is not None else {"enabled": False}
        ),
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_token = Path(checkpoint).stem.replace(" ", "_")
    label_token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(report_label or "").strip()
    )
    label_suffix = f"_{label_token}" if label_token else ""
    target = output / f"benchmark_{checkpoint_token}_stage{stage}_{baseline}{label_suffix}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[benchmark] complete baseline={baseline} stage={stage} "
        f"completed={completed}/{total} first_place_rate={first_place_rate:.3f} "
        f"pairwise_score={report['pairwise_score']:.3f} rejections={rejection_count} "
        f"awards_funded={mean_candidate_awards:.3f} "
        f"award_vp={mean(candidate_award_vp) if candidate_award_vp else 0.0:.2f} "
        f"gate_passed={bool(gate_passed)} elapsed={time.monotonic() - benchmark_started_at:.1f}s",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed TFM RL v2 acceptance benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline",
        choices=("random", "teacher", "award_teacher", "champion"),
        required=True,
    )
    parser.add_argument("--champion")
    parser.add_argument("--stage", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--report-label")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample both candidate and neural champion actions for a symmetric policy comparison",
    )
    parser.add_argument(
        "--candidate-stochastic",
        action="store_true",
        help="sample only the candidate while keeping the teacher or other baseline deterministic",
    )
    parser.add_argument(
        "--award-override",
        help=(
            "candidate-only award funding rule, e.g. "
            "'min_lead=1,min_generation=3,max_cost=8,reserve_mc=0,max_own_awards=1' or 'default'"
        ),
    )
    parser.add_argument(
        "--random-ma",
        choices=("Limited synergy", "Full random"),
        help="draw random awards/milestones for every benchmark game",
    )
    parser.add_argument("--output", default=os.getenv("V2_BENCHMARK_DIR", "/app/v2/benchmarks"))
    args = parser.parse_args()
    if args.baseline == "champion" and not args.champion:
        parser.error("--champion is required for champion baseline")
    print(
        json.dumps(
            asyncio.run(
                benchmark(
                    args.checkpoint,
                    args.baseline,
                    args.stage,
                    args.output,
                    args.seeds,
                    args.champion,
                    args.report_label,
                    stochastic=args.stochastic,
                    candidate_stochastic=args.candidate_stochastic,
                    award_override=args.award_override,
                    random_ma=args.random_ma,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
