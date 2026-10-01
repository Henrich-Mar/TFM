from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from models.agent import RLAgent
from training import v2_self_play
from training.v2_self_play import V2SelfPlayRunner, _frozen_checkpoint_agent


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


class _CheckpointLearner:
    def __init__(self, policy_version: int = 0) -> None:
        self.policy_version = policy_version

    def save_model(self, path: str) -> None:
        Path(path).write_bytes(b"checkpoint")


def test_cli_can_explicitly_start_stage_one_from_random_weights(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class _FakeRunner:
        def __init__(self, checkpoint, root, interval, seed, initial_stage, from_scratch) -> None:
            captured.update(
                checkpoint=checkpoint,
                root=root,
                interval=interval,
                seed=seed,
                stage=initial_stage,
                from_scratch=from_scratch,
            )

        async def run(self, max_decisions: int) -> None:
            captured["max_decisions"] = max_decisions

    monkeypatch.setattr(v2_self_play, "V2SelfPlayRunner", _FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v2_self_play",
            "--from-scratch",
            "--root",
            str(tmp_path),
            "--stage",
            "1",
            "--max-decisions",
            "123",
        ],
    )

    v2_self_play.main()

    assert captured["checkpoint"] is None
    assert captured["from_scratch"] is True
    assert captured["stage"] == 1
    assert captured["max_decisions"] == 123


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
    runner._lineup = lambda seed: [f"seat-{seed}-{seat}" for seat in range(4)]

    results = asyncio.run(runner._run_selfplay_batch())

    assert [game.number for game, _elapsed in results] == [1, 2, 3]
    assert [game.seed for game, _elapsed in results] == [10, 11, 13]
    assert runner.manager.max_active == 3
    assert [call["seed"] for call in runner.manager.calls] == [10, 11, 13]
    assert all(call["players_beginner"] for call in runner.manager.calls)


def _runner_with_seats(seats: list) -> V2SelfPlayRunner:
    runner = object.__new__(V2SelfPlayRunner)
    runner.history = []
    runner.historical_pool = []
    runner.champion_pool = []
    runner.teacher_pool = []
    runner.learning_seats = seats
    runner._seat_cursor = 0
    runner._historical_cursor = 0
    runner._champion_cursor = 0
    runner._teacher_cursor = 0
    runner.lineup_live_fraction = 1.0
    runner.lineup_champion_fraction = 0.0
    runner.lineup_teacher_fraction = 0.0
    return runner


def test_empty_history_seats_four_copies_of_the_live_policy() -> None:
    seats = [
        SimpleNamespace(
            id=f"self-{seat}",
            network="shared-net",
            rollout_buffer="shared-buf",
            train_from_self_play=True,
            ppo_enable=True,
        )
        for seat in range(4)
    ]
    lineup = _runner_with_seats(seats)._lineup(5)

    assert [seat.id for seat in lineup] == ["self-0", "self-1", "self-2", "self-3"]
    assert len({id(seat) for seat in lineup}) == 4
    assert all(seat.network == "shared-net" and seat.rollout_buffer == "shared-buf" for seat in lineup)
    assert all(seat.train_from_self_play and seat.ppo_enable for seat in lineup)


def test_champion_draw_seats_one_learner_against_three_frozen_champions(monkeypatch) -> None:
    class _FixedRandom:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.50

        def randrange(self, stop: int) -> int:
            assert stop == 4
            return 2

    seats = [
        SimpleNamespace(id=f"self-{seat}", train_from_self_play=True, ppo_enable=True)
        for seat in range(4)
    ]
    runner = _runner_with_seats(seats)
    runner.lineup_live_fraction = 0.40
    runner.lineup_champion_fraction = 0.40
    runner.lineup_teacher_fraction = 0.20
    runner.champion_pool = [
        SimpleNamespace(id=f"champion-{idx}", train_from_self_play=False, ppo_enable=False)
        for idx in range(3)
    ]
    monkeypatch.setattr("training.v2_self_play.random.Random", _FixedRandom)

    lineup = runner._lineup(9)

    assert lineup[2].id == "self-0"
    assert [seat.id for seat in lineup if not seat.train_from_self_play] == [
        "champion-0", "champion-1", "champion-2"
    ]
    assert [seat.id for seat in lineup if seat.train_from_self_play] == ["self-0"]


def test_teacher_draw_seats_one_learner_against_three_teachers(monkeypatch) -> None:
    class _TeacherDraw:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.95

        def randrange(self, stop: int) -> int:
            assert stop == 4
            return 1

    seats = [SimpleNamespace(id=f"self-{seat}", train_from_self_play=True) for seat in range(4)]
    runner = _runner_with_seats(seats)
    runner.lineup_live_fraction = 0.40
    runner.lineup_champion_fraction = 0.40
    runner.lineup_teacher_fraction = 0.20
    runner.teacher_pool = [
        SimpleNamespace(id=f"teacher-{idx}", train_from_self_play=False)
        for idx in range(3)
    ]
    monkeypatch.setattr("training.v2_self_play.random.Random", _TeacherDraw)

    lineup = runner._lineup(9)

    assert [seat.id for seat in lineup] == ["teacher-0", "self-0", "teacher-1", "teacher-2"]


def test_learning_seat_rollout_uses_leader_buffer_and_policy_version(monkeypatch) -> None:
    monkeypatch.setenv("PPO_DISK_SHARDING_ENABLE", "0")
    monkeypatch.setenv("PPO_ENABLE", "1")
    leader = RLAgent(agent_id="leader")
    leader.policy_version = 4
    leader.rollout_buffer.clear()
    seat = RLAgent(agent_id="seat-0")
    seat.bind_shared_learner(leader)
    step = {
        "state_bundle": {
            "action_mask": np.asarray([True]),
            "action_indices": np.asarray([3]),
        },
        "action_position": 0,
        "action_index": 3,
        "logp_old": -0.2,
        "value_old": 0.1,
        "reward": 0.0,
        "legal_actions": [3],
        "action_source": "policy",
        "server_accepted": True,
        "fallback_used": False,
    }

    asyncio.run(seat._queue_episode_rollout([step], 1.0))

    assert seat.network is leader.network
    assert seat.optimizer is leader.optimizer
    assert seat.rollout_buffer is leader.rollout_buffer
    assert len(leader.rollout_buffer) == 1
    assert int(leader.rollout_buffer[0].policy_version) == 4
    assert asyncio.run(seat.optimize_from_rollout_buffer()) == {}
    assert leader.policy_version == 4


def test_historical_seat_weights_stay_fixed_when_leader_updates(tmp_path: Path) -> None:
    leader = RLAgent(agent_id="leader")
    checkpoint = tmp_path / "champion.pth"
    leader.save_model(str(checkpoint))
    frozen = _frozen_checkpoint_agent(str(checkpoint), "historical-0")
    before = frozen.network.action_logit_head.weight.detach().clone()

    with torch.no_grad():
        leader.network.action_logit_head.weight.add_(1.0)
    leader.policy_version += 1

    assert torch.equal(frozen.network.action_logit_head.weight, before)
    assert frozen.train_from_self_play is False
    assert frozen.ppo_enable is False
    assert frozen.deterministic_actions is True


def test_unpromoted_history_is_not_loaded_into_training_pool(monkeypatch) -> None:
    def fake_frozen(path: str, agent_id: str):
        return SimpleNamespace(path=path, id=agent_id)

    monkeypatch.setattr("training.v2_self_play._frozen_checkpoint_agent", fake_frozen)
    monkeypatch.setattr("training.v2_self_play._shared_teacher_pool", lambda count, seed: [])
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 2
    runner.history = [f"champ-{idx}.pth" for idx in range(8)]
    runner.load_unpromoted_history = False
    runner.champion_path = Path("missing-champion.pth")
    runner.is_v3 = False
    runner._refresh_frozen_pools()

    assert runner.historical_pool == []


def test_trusted_pools_allocate_three_opponents_per_concurrent_game(monkeypatch, tmp_path: Path) -> None:
    def fake_frozen(path: str, agent_id: str):
        return SimpleNamespace(path=path, id=agent_id)

    champion = tmp_path / "champion.pth"
    champion.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "training.v2_self_play._shared_champion_pool",
        lambda path, count: [SimpleNamespace(path=path, id=f"champion-{idx}") for idx in range(count)],
    )
    monkeypatch.setattr(
        "training.v2_self_play._shared_teacher_pool",
        lambda count, seed: [SimpleNamespace(id=f"teacher-{idx}") for idx in range(count)],
    )
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 4
    runner.history = ["only.pth"]
    runner.load_unpromoted_history = False
    runner.champion_path = champion
    runner.is_v3 = False
    runner._refresh_frozen_pools()

    assert len(runner.champion_pool) == 12
    assert len(runner.teacher_pool) == 12


def test_screen_requires_completion_no_rejections_and_baseline_threshold() -> None:
    runner = object.__new__(V2SelfPlayRunner)
    runner.screen_min_completion_ratio = 0.90
    runner.screen_teacher_min_first_place_rate = 0.25
    runner.screen_champion_min_pairwise_score = 0.50
    base = {"planned_games": 32, "completed_games": 32, "rejection_count": 0}

    assert runner._screen_report_promising({**base, "first_place_rate": 0.25}, "teacher")
    assert not runner._screen_report_promising({**base, "first_place_rate": 0.249}, "teacher")
    assert runner._screen_report_promising({**base, "pairwise_score": 0.50}, "champion")
    assert not runner._screen_report_promising(
        {**base, "completed_games": 28, "pairwise_score": 1.0},
        "champion",
    )
    assert not runner._screen_report_promising(
        {**base, "rejection_count": 1, "first_place_rate": 1.0},
        "teacher",
    )


def test_history_snapshot_is_independent_of_promotion_cadence(tmp_path: Path) -> None:
    runner = object.__new__(V2SelfPlayRunner)
    runner.learner = _CheckpointLearner(policy_version=8)
    runner.checkpoints = tmp_path
    runner.history = []
    runner.history_snapshot_interval_updates = 8
    runner.last_history_snapshot_policy_version = 0
    refreshes: list[bool] = []
    runner._refresh_frozen_pools = lambda: refreshes.append(True)

    archived = runner._maybe_archive_history_snapshot(100_000)

    assert archived is not None
    assert Path(archived).is_file()
    assert runner.history == [archived]
    assert runner.last_history_snapshot_policy_version == 8
    assert refreshes == []
    runner.learner.policy_version = 9
    assert runner._maybe_archive_history_snapshot(112_000) is None


def test_failed_screen_skips_both_full_promotion_benchmarks(monkeypatch, tmp_path: Path) -> None:
    runner = object.__new__(V2SelfPlayRunner)
    runner.learner = _CheckpointLearner()
    runner.checkpoints = tmp_path / "checkpoints"
    runner.checkpoints.mkdir()
    runner.benchmarks = tmp_path / "benchmarks"
    runner.latest_learner_path = runner.checkpoints / "latest_learner.pth"
    runner.champion_path = runner.checkpoints / "champion.pth"
    runner.champion_path.write_bytes(b"champion")
    runner.screen_seed_path = tmp_path / "screen-seeds.json"
    runner.stage = 1
    runner.screen_min_completion_ratio = 0.90
    runner.screen_teacher_min_first_place_rate = 0.25
    runner.screen_champion_min_pairwise_score = 0.50
    calls: list[tuple[str, str | None]] = []

    async def fake_benchmark(checkpoint, baseline, stage, output, **kwargs):
        del checkpoint, stage, output
        calls.append((baseline, kwargs.get("report_label")))
        return {
            "planned_games": 32,
            "completed_games": 32,
            "rejection_count": 0,
            "first_place_rate": 0.20 if baseline == "teacher" else 1.0,
            "pairwise_score": 1.0,
            "gate_passed": False,
        }

    monkeypatch.setattr(v2_self_play, "benchmark", fake_benchmark)

    report = asyncio.run(runner._evaluate_and_promote(100_000))

    assert calls == [("teacher", "screen"), ("champion", "screen")]
    assert report["screen_passed"] is False
    assert report["teacher"] is None
    assert report["regression"] is None
    assert report["promoted"] is False


def test_failed_full_teacher_gate_skips_full_champion_gate(monkeypatch, tmp_path: Path) -> None:
    runner = object.__new__(V2SelfPlayRunner)
    runner.learner = _CheckpointLearner()
    runner.checkpoints = tmp_path / "checkpoints"
    runner.checkpoints.mkdir()
    runner.benchmarks = tmp_path / "benchmarks"
    runner.latest_learner_path = runner.checkpoints / "latest_learner.pth"
    runner.champion_path = runner.checkpoints / "champion.pth"
    runner.champion_path.write_bytes(b"champion")
    runner.screen_seed_path = tmp_path / "screen-seeds.json"
    runner.stage = 1
    runner.screen_min_completion_ratio = 0.90
    runner.screen_teacher_min_first_place_rate = 0.25
    runner.screen_champion_min_pairwise_score = 0.50
    calls: list[tuple[str, str | None]] = []

    async def fake_benchmark(checkpoint, baseline, stage, output, **kwargs):
        del checkpoint, stage, output
        calls.append((baseline, kwargs.get("report_label")))
        is_screen = kwargs.get("report_label") == "screen"
        return {
            "planned_games": 32 if is_screen else 120,
            "completed_games": 32 if is_screen else 120,
            "rejection_count": 0,
            "first_place_rate": 0.30,
            "pairwise_score": 0.75,
            "gate_passed": False,
        }

    monkeypatch.setattr(v2_self_play, "benchmark", fake_benchmark)

    report = asyncio.run(runner._evaluate_and_promote(100_000))

    assert calls == [("teacher", "screen"), ("champion", "screen"), ("teacher", None)]
    assert report["screen_passed"] is True
    assert report["teacher"]["gate_passed"] is False
    assert report["regression"] is None
    assert report["promoted"] is False


def _promotion_runner(tmp_path: Path) -> V2SelfPlayRunner:
    runner = object.__new__(V2SelfPlayRunner)
    runner.learner = _CheckpointLearner()
    runner.checkpoints = tmp_path / "checkpoints"
    runner.checkpoints.mkdir()
    runner.benchmarks = tmp_path / "benchmarks"
    runner.latest_learner_path = runner.checkpoints / "latest_learner.pth"
    runner.champion_path = runner.checkpoints / "champion.pth"
    runner.champion_path.write_bytes(b"champion")
    runner.screen_seed_path = tmp_path / "screen-seeds.json"
    runner.stage = 1
    runner.screen_min_completion_ratio = 0.90
    runner.screen_teacher_min_first_place_rate = 0.25
    runner.screen_champion_min_pairwise_score = 0.50
    runner.history = []
    runner._refresh_frozen_pools = lambda: None
    return runner


def _passing_benchmark(calls: list, teacher_pairwise: float):
    async def fake_benchmark(checkpoint, baseline, stage, output, **kwargs):
        del checkpoint, stage, output
        calls.append((baseline, kwargs.get("report_label")))
        is_screen = kwargs.get("report_label") == "screen"
        return {
            "planned_games": 32 if is_screen else 120,
            "completed_games": 32 if is_screen else 120,
            "rejection_count": 0,
            "first_place_rate": 0.35,
            "pairwise_score": teacher_pairwise if baseline == "teacher" else 0.80,
            "mean_relative_vp_margin": 4.0,
            "seeds": [1, 2, 3],
            "gate_passed": True,
        }

    return fake_benchmark


def test_candidate_worse_than_champion_against_teacher_is_not_promoted(monkeypatch, tmp_path: Path) -> None:
    runner = _promotion_runner(tmp_path)
    runner.champion_teacher_report = {"pairwise_score": 0.708, "seeds": [1, 2, 3]}
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(v2_self_play, "benchmark", _passing_benchmark(calls, teacher_pairwise=0.681))

    report = asyncio.run(runner._evaluate_and_promote(100_000))

    assert calls == [("teacher", "screen"), ("champion", "screen"), ("teacher", None)]
    assert report["teacher_relative"]["passed"] is False
    assert report["teacher_relative"]["paired"] is True
    assert report["regression"] is None
    assert report["promoted"] is False
    assert runner.champion_path.read_bytes() == b"champion"


def test_promotion_records_new_champion_teacher_baseline(monkeypatch, tmp_path: Path) -> None:
    runner = _promotion_runner(tmp_path)
    runner.champion_teacher_report = {"pairwise_score": 0.70, "seeds": [1, 2, 3]}
    runner.champion_promotions = 1
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(v2_self_play, "benchmark", _passing_benchmark(calls, teacher_pairwise=0.72))

    report = asyncio.run(runner._evaluate_and_promote(100_000))

    assert report["promoted"] is True
    assert report["teacher_relative"]["passed"] is True
    assert runner.champion_teacher_report["pairwise_score"] == pytest.approx(0.72)
    assert runner.champion_path.read_bytes() == b"checkpoint"


def test_history_draw_seats_three_distinct_snapshots(monkeypatch) -> None:
    class _HistoryDraw:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.70

        def randrange(self, stop: int) -> int:
            return 3

        def sample(self, population, count: int):
            return list(population)[:count]

    seats = [SimpleNamespace(id=f"self-{seat}", train_from_self_play=True) for seat in range(4)]
    runner = _runner_with_seats(seats)
    runner.lineup_live_fraction = 0.40
    runner.lineup_champion_fraction = 0.20
    runner.lineup_history_fraction = 0.20
    runner.lineup_teacher_fraction = 0.20
    runner.historical_seats_by_snapshot = [
        [SimpleNamespace(id=f"hist-{snap}-{seat}", train_from_self_play=False) for seat in range(2)]
        for snap in range(4)
    ]
    runner._historical_snapshot_cursors = [0] * 4
    monkeypatch.setattr("training.v2_self_play.random.Random", _HistoryDraw)

    first = runner._lineup(9)
    second = runner._lineup(10)

    assert [seat.id for seat in first] == ["hist-0-0", "hist-1-0", "hist-2-0", "self-0"]
    assert [seat.id for seat in second] == ["hist-0-1", "hist-1-1", "hist-2-1", "self-1"]
    assert runner._lineup_kind(9) == "history"


def test_history_draw_falls_back_to_champion_without_three_snapshots() -> None:
    runner = _runner_with_seats([])
    runner.lineup_live_fraction = 0.40
    runner.lineup_champion_fraction = 0.20
    runner.lineup_history_fraction = 0.20
    runner.lineup_teacher_fraction = 0.20
    runner.historical_seats_by_snapshot = [[SimpleNamespace(id="hist-0-0")]]

    assert runner._draw_lineup_kind(0.70) == "champion"
    assert runner._draw_lineup_kind(0.90) == "teacher"


def test_history_pool_shares_weights_per_snapshot_and_reuses_loaded_snapshots(monkeypatch) -> None:
    loads: list[str] = []

    def fake_frozen(path: str, agent_id: str):
        loads.append(path)
        return SimpleNamespace(path=path, id=agent_id)

    monkeypatch.setattr("training.v2_self_play._frozen_checkpoint_agent", fake_frozen)
    monkeypatch.setattr(
        "training.v2_self_play._bind_frozen_seat",
        lambda leader, agent_id: SimpleNamespace(path=leader.path, id=agent_id),
    )
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 4
    runner.load_unpromoted_history = True
    runner.is_v3 = False
    runner.history = ["a.pth", "b.pth", "c.pth"]
    runner._refresh_history_pool()

    assert [len(seats) for seats in runner.historical_seats_by_snapshot] == [4, 4, 4]
    assert len(runner.historical_pool) == 12
    assert loads == ["a.pth", "b.pth", "c.pth"]

    runner.history = ["a.pth", "b.pth", "c.pth", "d.pth"]
    runner._refresh_history_pool()

    assert loads == ["a.pth", "b.pth", "c.pth", "d.pth"]
    assert runner.historical_snapshot_paths == ["a.pth", "b.pth", "c.pth", "d.pth"]


def test_catastrophic_champion_screen_restores_trusted_policy(tmp_path: Path) -> None:
    class _RollbackLearner:
        def __init__(self) -> None:
            self.policy_version = 12
            self.games_played = 0
            self.loaded: list[str] = []

        async def quarantine_rollout_buffer(self, label: str) -> dict:
            assert label == "rollback_policy_000012"
            return {"steps": 55, "path": str(tmp_path / "quarantine")}

        def load_model(self, path: str) -> None:
            self.loaded.append(path)
            self.policy_version = 7

    runner = object.__new__(V2SelfPlayRunner)
    runner.learner = _RollbackLearner()
    runner.champion_path = tmp_path / "champion.pth"
    runner.champion_path.write_bytes(b"champion")
    runner.champion_promotions = 1
    runner.consecutive_champion_screen_failures = 0
    runner.screen_champion_min_pairwise_score = 0.50
    runner.rollback_pairwise_floor = 0.35
    runner.rollback_failure_limit = 2
    runner.lifetime_games = 100
    runner.learning_seats = []
    runner._rebind_learning_seats = lambda: None
    runner._refresh_frozen_pools = lambda: None

    event = asyncio.run(
        runner._maybe_rollback_from_screen({"pairwise_score": 0.20}, decisions=900_000)
    )

    assert event["applied"] is True
    assert event["quarantined_rollout_steps"] == 55
    assert runner.learner.loaded == [str(runner.champion_path)]
    assert runner.learner.policy_version == 13
    assert runner.learner.games_played == 100
    assert runner.consecutive_champion_screen_failures == 0


def _rollback_runner(tmp_path: Path, loaded: list) -> V2SelfPlayRunner:
    runner = object.__new__(V2SelfPlayRunner)
    runner.champion_path = tmp_path / "champion.pth"
    runner.champion_path.write_bytes(b"champion")
    runner.benchmarks = tmp_path / "benchmarks"
    runner.stage = 1
    runner.champion_promotions = 1
    runner.consecutive_champion_screen_failures = 1
    runner.screen_min_completion_ratio = 0.90
    runner.screen_champion_min_pairwise_score = 0.50
    runner.rollback_pairwise_floor = 0.35
    runner.rollback_failure_limit = 2
    runner.rollback_confirm_full = True
    runner.rollback_confirm_pairwise = 0.45

    async def fake_rollback(decisions: int, reason: str) -> dict:
        loaded.append(reason)
        return {"applied": True, "reason": reason}

    runner._rollback_to_champion = fake_rollback
    return runner


def _confirmation_benchmark(calls: list, pairwise: float):
    async def fake_benchmark(checkpoint, baseline, stage, output, **kwargs):
        del checkpoint, stage, output
        calls.append((baseline, kwargs.get("seeds_path"), kwargs.get("report_label")))
        return {"planned_games": 120, "completed_games": 120, "rejection_count": 0, "pairwise_score": pairwise}

    return fake_benchmark


def test_low_screen_is_cleared_when_full_champion_benchmark_disagrees(monkeypatch, tmp_path: Path) -> None:
    rollbacks: list[str] = []
    calls: list = []
    runner = _rollback_runner(tmp_path, rollbacks)
    monkeypatch.setattr(v2_self_play, "benchmark", _confirmation_benchmark(calls, pairwise=0.52))

    event = asyncio.run(
        runner._maybe_rollback_from_screen({"pairwise_score": 0.297}, 1_800_000, candidate_path=tmp_path / "c.pth")
    )

    assert calls == [("champion", None, "rollback_confirm")]
    assert event["applied"] is False
    assert event["reason"] == "full_benchmark_cleared"
    assert rollbacks == []
    assert runner.consecutive_champion_screen_failures == 0


def test_low_screen_rolls_back_when_full_champion_benchmark_confirms(monkeypatch, tmp_path: Path) -> None:
    rollbacks: list[str] = []
    calls: list = []
    runner = _rollback_runner(tmp_path, rollbacks)
    monkeypatch.setattr(v2_self_play, "benchmark", _confirmation_benchmark(calls, pairwise=0.40))

    event = asyncio.run(
        runner._maybe_rollback_from_screen({"pairwise_score": 0.45}, 1_800_000, candidate_path=tmp_path / "c.pth")
    )

    assert len(calls) == 1
    assert event["applied"] is True
    assert event["confirmation"]["pairwise_score"] == pytest.approx(0.40)
    assert rollbacks == ["champion_screen_failed_2_times_confirmed_full_pairwise_0.400"]


def test_shared_seat_reward_schedule_uses_leader_global_games(monkeypatch) -> None:
    monkeypatch.setenv("PPO_SHAPING_INITIAL_COEF", "0.20")
    monkeypatch.setenv("PPO_SHAPING_FINAL_COEF", "0.05")
    monkeypatch.setenv("PPO_SHAPING_ANNEAL_GAMES", "3000")
    leader = RLAgent(agent_id="leader")
    seat = RLAgent(agent_id="seat")
    seat.bind_shared_learner(leader)
    seat.games_played = 0
    leader.games_played = 1500

    assert seat._current_reward_shaping_coef() == pytest.approx(0.125)


def test_ppo_metrics_record_lineups_and_reset_update_window(tmp_path: Path) -> None:
    learner = SimpleNamespace(
        policy_version=17,
        _current_reward_shaping_coef=lambda: 0.12,
        _current_ppo_entropy_coef=lambda: 0.003,
    )
    runner = object.__new__(V2SelfPlayRunner)
    runner.version = "v2"
    runner.learner = learner
    runner.lifetime_games = 250
    runner.ppo_metrics_path = tmp_path / "ppo_updates.jsonl"
    runner.lineup_games_since_update = {"live": 4, "champion": 3, "teacher": 1, "search": 0}

    runner._append_ppo_metrics({"ppo/approx_kl": 0.008}, decisions=900_000, elapsed=12.5)

    payload = json.loads(runner.ppo_metrics_path.read_text(encoding="utf-8"))
    assert payload["policy_version"] == 17
    assert payload["decisions"] == 900_000
    assert payload["lineups"] == {"live": 4, "champion": 3, "teacher": 1, "search": 0}
    assert payload["ppo/approx_kl"] == pytest.approx(0.008)
    assert all(value == 0 for value in runner.lineup_games_since_update.values())
