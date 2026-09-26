from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.agent import RLAgent
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
    runner.learning_seats = seats
    runner._seat_cursor = 0
    runner._historical_cursor = 0
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


def test_quarter_draw_replaces_one_seat_with_frozen_history(monkeypatch) -> None:
    class _FixedRandom:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.10

        def randrange(self, stop: int) -> int:
            assert stop == 4
            return 2

    seats = [
        SimpleNamespace(id=f"self-{seat}", train_from_self_play=True, ppo_enable=True)
        for seat in range(4)
    ]
    runner = _runner_with_seats(seats)
    runner.history = ["past.pth"]
    runner.historical_pool = [
        SimpleNamespace(id="historical-0", train_from_self_play=False, ppo_enable=False)
    ]
    monkeypatch.setattr("training.v2_self_play.random.Random", _FixedRandom)

    lineup = runner._lineup(9)

    assert lineup[2].id == "historical-0"
    assert lineup[2].train_from_self_play is False
    assert lineup[2].ppo_enable is False
    assert [seat.id for seat in lineup if seat.train_from_self_play] == ["self-0", "self-1", "self-3"]


def test_history_keeps_four_learning_seats_when_the_draw_misses(monkeypatch) -> None:
    class _Miss:
        def __init__(self, seed: int) -> None:
            del seed

        def random(self) -> float:
            return 0.25

    seats = [SimpleNamespace(id=f"self-{seat}", train_from_self_play=True) for seat in range(4)]
    runner = _runner_with_seats(seats)
    runner.history = ["past.pth"]
    runner.historical_pool = [SimpleNamespace(id="historical-0", train_from_self_play=False)]
    monkeypatch.setattr("training.v2_self_play.random.Random", _Miss)

    lineup = runner._lineup(9)

    assert [seat.id for seat in lineup] == ["self-0", "self-1", "self-2", "self-3"]


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


def test_historical_pool_loads_every_retained_checkpoint(monkeypatch) -> None:
    def fake_frozen(path: str, agent_id: str):
        return SimpleNamespace(path=path, id=agent_id)

    monkeypatch.setattr("training.v2_self_play._frozen_checkpoint_agent", fake_frozen)
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 2
    runner.history = [f"champ-{idx}.pth" for idx in range(8)]
    runner.is_v3 = False
    runner._refresh_frozen_pools()

    assert [seat.path for seat in runner.historical_pool] == [f"champ-{idx}.pth" for idx in range(8)]
    seated = [runner._take_historical_seat().path for _ in range(8)]
    assert seated == [f"champ-{idx}.pth" for idx in range(8)]


def test_historical_pool_duplicates_a_short_window_for_concurrent_games(monkeypatch) -> None:
    def fake_frozen(path: str, agent_id: str):
        return SimpleNamespace(path=path, id=agent_id)

    monkeypatch.setattr("training.v2_self_play._frozen_checkpoint_agent", fake_frozen)
    runner = object.__new__(V2SelfPlayRunner)
    runner.selfplay_concurrency = 4
    runner.history = ["only.pth"]
    runner.is_v3 = False
    runner._refresh_frozen_pools()

    seated = [runner._take_historical_seat() for _ in range(4)]
    assert [seat.path for seat in seated] == ["only.pth"] * 4
    assert len({id(seat) for seat in seated}) == 4
