"""Seat one reachability teacher against three old teachers.

Twenty seeds can see a large rank gap and cannot see a small one. Collection is
allowed only when the new teacher is clearly under 2.5. The band 2.35-2.65 is
noise and asks for 40 more seeds. Above 2.5, the bonus is wrong.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy
from tournament_manager import TournamentManager
from training.v2_benchmark import mean_interval
from v2_runtime import assert_stage_allowed


def teacher_ab_verdict(mean_rank: float | None, finished_games: int) -> str:
    if mean_rank is None or finished_games <= 0:
        return "extend"
    if float(mean_rank) > 2.5:
        return "stop"
    if finished_games >= 60 and float(mean_rank) < 2.5:
        return "collect"
    if float(mean_rank) < 2.35:
        return "collect"
    return "extend"


def assert_teacher_ab_allows_collection(path: str) -> Dict[str, Any]:
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        raise RuntimeError(
            "stage-1 collection is blocked until the reachability teacher beats the old teacher: "
            f"missing {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary") or report
    verdict = str(summary.get("verdict") or "")
    if verdict != "collect":
        raise RuntimeError(
            "stage-1 collection is blocked until the reachability teacher is clearly under rank 2.5; "
            f"verdict={verdict or 'missing'} mean_rank={summary.get('mean_rank')}"
        )
    return summary


def _teacher(agent_id: str, seed: int, reachability: bool) -> RLAgent:
    agent = RLAgent(
        agent_id=agent_id,
        decision_policy=HeuristicTeacherPolicy(seed, sample=False, reachability=reachability),
    )
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    return agent


async def run_ab(games: int, seed_start: int, output: str) -> Dict[str, Any]:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V2_ALLOW_STAGE1", "1")
    assert_stage_allowed(1, context="teacher A/B")
    options_path = Path(__file__).resolve().parents[1] / "game_options.v2_stage1.json"
    cluster = GameServerCluster(
        [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
    )
    cluster.base_game_options = json.loads(options_path.read_text(encoding="utf-8"))
    manager = TournamentManager(cluster)
    rows: List[Dict[str, Any]] = []
    jobs = [(seed_start + index, index % 4, index) for index in range(games)]
    concurrency = max(1, min(len(cluster.servers), games))

    async def _play(seed: int, seat: int, index: int) -> None:
        started = time.monotonic()
        candidate = _teacher("reach-teacher", seed, True)
        opponents = [_teacher(f"old-teacher-{idx}", seed + 1 + idx, False) for idx in range(3)]
        lineup = list(opponents)
        lineup.insert(seat, candidate)
        print(f"[teacher-ab] start seed={seed} seat={seat} game={index + 1}/{games}", flush=True)
        result = await manager._run_single_game(
            lineup,
            tournament_id=f"v4_teacher_ab_{seed}_{seat}",
            game_seed=seed,
            players_beginner=False,
        )
        completed = bool(result.completed)
        rank = None
        if completed:
            row = next(item for item in result.players if str(item.get("agent_id")) == "reach-teacher")
            rank = int(row.get("rank", 4) or 4)
        record = {
            "seed": seed,
            "seat": seat,
            "completed": completed,
            "rank": rank,
            "elapsed_sec": round(time.monotonic() - started, 1),
        }
        rows.append(record)
        print(json.dumps(record), flush=True)

    async def _worker(offset: int) -> None:
        for seed, seat, index in jobs[offset::concurrency]:
            await _play(seed, seat, index)

    try:
        await asyncio.gather(*(_worker(offset) for offset in range(concurrency)))
    finally:
        await cluster.close()
    finished = [row["rank"] for row in rows if row["completed"] and row["rank"] is not None]
    center = mean(finished) if finished else None
    low, high = mean_interval([float(item) for item in finished]) if finished else (None, None)
    summary = {
        "seed_start": seed_start,
        "games": games,
        "finished_games": len(finished),
        "mean_rank": center,
        "mean_rank_lower_95": low,
        "mean_rank_upper_95": high,
        "verdict": teacher_ab_verdict(center, len(finished)),
    }
    payload = {"summary": summary, "games": rows}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="New reachability teacher versus three old teachers")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=940000)
    parser.add_argument("--output", default="/app/v4/diagnostics/teacher_ab.json")
    summary = asyncio.run(run_ab(parser.parse_args().games, parser.parse_args().seed_start, parser.parse_args().output))
    if summary["verdict"] != "collect":
        raise SystemExit(2 if summary["verdict"] == "stop" else 3)


if __name__ == "__main__":
    main()
