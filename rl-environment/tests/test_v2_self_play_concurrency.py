from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v2_self_play import V2SelfPlayRunner
from models.decision_policy import RandomLegalPolicy


class _FakeLearner:
    def get_behavior_stats(self) -> dict:
        return {"total_decisions": 0}


class _FakeManager:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[dict] = []

    async def _run_single_game(self, lineup, tournament_id, game_seed, players_beginner) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(
            {
                "lineup": lineup,
                "tournament_id": tournament_id,
                "seed": game_seed,
                "players_beginner": players_beginner,
            }
        )
        await asyncio.sleep(0)
        self.active -= 1


def test_selfplay_batch_runs_configured_number_of_games_concurrently() -> None:
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 3
    runner.game_count = 0
    runner.seed_cursor = 10
    runner.reserved_benchmark_seeds = {12}
    runner.stage = 0
    runner.decision_offset = 0
    runner.learner = _FakeLearner()
    runner.manager = _FakeManager()
    runner._opponents = lambda seed: [f"opponent-{seed}-{seat}" for seat in range(3)]

    results = asyncio.run(runner._run_selfplay_batch())

    assert [game.number for game, _elapsed in results] == [1, 2, 3]
    assert [game.seed for game, _elapsed in results] == [10, 11, 13]
    assert runner.manager.max_active == 3
    assert [call["seed"] for call in runner.manager.calls] == [10, 11, 13]
    assert all(call["players_beginner"] for call in runner.manager.calls)


def test_stage_zero_opponent_mix_includes_champion_teacher_and_random(monkeypatch) -> None:
    class _FixedRandom:
        def __init__(self, seed: int) -> None:
            self.values = iter((0.10, 0.30, 0.75))

        def random(self) -> float:
            return next(self.values)

    runner = object.__new__(V2SelfPlayRunner)
    runner.stage = 0
    runner.random_pool = [
        SimpleNamespace(id=f"random-{seat}", decision_policy=RandomLegalPolicy(seed=seat))
        for seat in range(3)
    ]
    runner.teacher_pool = [
        SimpleNamespace(id=f"teacher-{seat}", decision_policy=None)
        for seat in range(3)
    ]
    runner.champion_pool = [
        SimpleNamespace(id=f"champion-{seat}", decision_policy=None)
        for seat in range(3)
    ]
    monkeypatch.setattr("training.v2_self_play.random.Random", _FixedRandom)

    opponents = runner._opponents(123)

    assert [opponent.id for opponent in opponents] == ["random-0", "teacher-1", "champion-2"]
