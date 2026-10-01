from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import v2_benchmark


class _FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id

    def get_behavior_stats(self) -> dict:
        return {"policy_rejections": 0}


def test_frozen_stochastic_agent_uses_exact_ppo_behavior_path(monkeypatch) -> None:
    class _Config:
        train_from_self_play = True
        epsilon = 0.05
        temperature = 1.2

    class _Agent:
        def __init__(self, agent_id: str) -> None:
            self.id = agent_id
            self.config = _Config()
            self.train_from_self_play = True
            self.ppo_enable = True
            self.deterministic_actions = False
            self.policy_temperature_cap = 0.75
            self.policy_temperature_floor = 0.75

        def load_model(self, checkpoint: str) -> None:
            self.checkpoint = checkpoint

    monkeypatch.setattr(v2_benchmark, "RLAgent", _Agent)

    sampled = v2_benchmark._frozen_neural("candidate.pth", "candidate", stochastic=True)
    greedy = v2_benchmark._frozen_neural("champion.pth", "champion", stochastic=False)

    assert sampled.train_from_self_play is False
    assert sampled.config.train_from_self_play is False
    assert sampled.ppo_enable is True
    assert sampled.deterministic_actions is False
    assert sampled.config.epsilon == 0.0
    assert sampled.config.temperature == 1.0
    assert sampled.policy_temperature_cap == 1.0
    assert sampled.policy_temperature_floor == 1.0
    assert greedy.ppo_enable is False
    assert greedy.deterministic_actions is True


def test_stochastic_champion_baseline_preserves_ppo_behavior_path(monkeypatch) -> None:
    agents = [
        SimpleNamespace(
            ppo_enable=True,
            config=SimpleNamespace(train_from_self_play=True),
        )
        for _ in range(3)
    ]

    monkeypatch.setattr(
        v2_benchmark,
        "_frozen_neural",
        lambda checkpoint, agent_id, stochastic=False: agents.pop(0),
    )

    baseline = v2_benchmark._baseline_agents(
        "champion",
        seed=101,
        champion="champion.pth",
        stochastic=True,
    )

    assert all(agent.ppo_enable is True for agent in baseline)
    assert all(agent.train_from_self_play is False for agent in baseline)
    assert all(agent.config.train_from_self_play is False for agent in baseline)


def test_benchmark_uses_all_configured_server_slots(monkeypatch, tmp_path: Path) -> None:
    active = 0
    max_active = 0

    class _FakeCluster:
        def __init__(self, servers) -> None:
            self.servers = [object() for _ in servers]
            self.max_active_games_per_server = 2
            self.base_game_options = {}

        async def close(self) -> None:
            return None

    class _FakeManager:
        def __init__(self, cluster) -> None:
            self.cluster = cluster

        async def _run_single_game(self, lineup, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            players = [
                {
                    "agent_id": agent.id,
                    "rank": 1 if agent.id == "v2-candidate" else 2,
                    "victory_points": 100.0 if agent.id == "v2-candidate" else 80.0,
                }
                for agent in lineup
            ]
            return SimpleNamespace(completed=True, players=players)

    monkeypatch.setenv("GAME_SERVERS", "one,two,three,four")
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "8")
    monkeypatch.setattr(v2_benchmark, "initialize_v2_runtime", lambda: {})
    monkeypatch.setattr(v2_benchmark, "_load_seeds", lambda path=None: [101, 202])
    monkeypatch.setattr(
        v2_benchmark,
        "_frozen_neural",
        lambda checkpoint, agent_id, stochastic=False: _FakeAgent(agent_id),
    )
    monkeypatch.setattr(
        v2_benchmark,
        "_baseline_agents",
        lambda kind, seed, champion=None, stochastic=False: [
            _FakeAgent(f"{kind}-{seed}-{index}") for index in range(3)
        ],
    )
    monkeypatch.setattr(v2_benchmark, "GameServerCluster", _FakeCluster)
    monkeypatch.setattr(v2_benchmark, "TournamentManager", _FakeManager)

    report = asyncio.run(
        v2_benchmark.benchmark(
            checkpoint=str(tmp_path / "candidate.pth"),
            baseline="random",
            stage=0,
            output_dir=str(tmp_path / "reports"),
        )
    )

    assert max_active == 8
    assert report["concurrency"] == 8
    assert report["completed_games"] == 8
    assert report["gate_passed"] is True
    assert report["action_selection"] == "argmax"


def test_stochastic_report_records_sample_selection(monkeypatch, tmp_path: Path) -> None:
    class _FakeCluster:
        def __init__(self, servers) -> None:
            self.servers = [object() for _ in servers]
            self.max_active_games_per_server = 1
            self.base_game_options = {}

        async def close(self) -> None:
            return None

    class _FakeManager:
        def __init__(self, cluster) -> None:
            del cluster

        async def _run_single_game(self, lineup, **kwargs):
            players = [
                {
                    "agent_id": agent.id,
                    "rank": 1 if agent.id == "v2-candidate" else 2,
                    "victory_points": 100.0 if agent.id == "v2-candidate" else 80.0,
                }
                for agent in lineup
            ]
            return SimpleNamespace(completed=True, players=players)

    monkeypatch.setenv("GAME_SERVERS", "one")
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "1")
    monkeypatch.setattr(v2_benchmark, "initialize_v2_runtime", lambda: {})
    monkeypatch.setattr(v2_benchmark, "_load_seeds", lambda path=None: [101])
    monkeypatch.setattr(
        v2_benchmark,
        "_frozen_neural",
        lambda checkpoint, agent_id, stochastic=False: _FakeAgent(agent_id),
    )
    monkeypatch.setattr(
        v2_benchmark,
        "_baseline_agents",
        lambda kind, seed, champion=None, stochastic=False: [
            _FakeAgent(f"{kind}-{index}") for index in range(3)
        ],
    )
    monkeypatch.setattr(v2_benchmark, "GameServerCluster", _FakeCluster)
    monkeypatch.setattr(v2_benchmark, "TournamentManager", _FakeManager)

    report = asyncio.run(
        v2_benchmark.benchmark(
            checkpoint=str(tmp_path / "candidate.pth"),
            baseline="champion",
            stage=1,
            output_dir=str(tmp_path / "reports"),
            champion=str(tmp_path / "champion.pth"),
            report_label="stochastic",
            stochastic=True,
        )
    )

    assert report["action_selection"] == "sample"
    assert report["report_label"] == "stochastic"


def test_candidate_stochastic_keeps_opponents_greedy(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class _FakeCluster:
        def __init__(self, servers) -> None:
            self.servers = [object() for _ in servers]
            self.max_active_games_per_server = 1
            self.base_game_options = {}

        async def close(self) -> None:
            return None

    class _FakeManager:
        def __init__(self, cluster) -> None:
            del cluster

        async def _run_single_game(self, lineup, **kwargs):
            players = [
                {
                    "agent_id": agent.id,
                    "rank": 1 if agent.id == "v2-candidate" else 2,
                    "victory_points": 100.0 if agent.id == "v2-candidate" else 80.0,
                }
                for agent in lineup
            ]
            return SimpleNamespace(completed=True, players=players)

    monkeypatch.setenv("GAME_SERVERS", "one")
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "1")
    monkeypatch.setattr(v2_benchmark, "initialize_v2_runtime", lambda: {})
    monkeypatch.setattr(v2_benchmark, "_load_seeds", lambda path=None: [101])

    def fake_frozen(checkpoint, agent_id, stochastic=False):
        captured[agent_id] = stochastic
        return _FakeAgent(agent_id)

    def fake_baseline(kind, seed, champion=None, stochastic=False):
        captured["opponent_stochastic"] = stochastic
        return [_FakeAgent(f"{kind}-{index}") for index in range(3)]

    monkeypatch.setattr(v2_benchmark, "_frozen_neural", fake_frozen)
    monkeypatch.setattr(v2_benchmark, "_baseline_agents", fake_baseline)
    monkeypatch.setattr(v2_benchmark, "GameServerCluster", _FakeCluster)
    monkeypatch.setattr(v2_benchmark, "TournamentManager", _FakeManager)

    report = asyncio.run(
        v2_benchmark.benchmark(
            checkpoint=str(tmp_path / "candidate.pth"),
            baseline="champion",
            stage=1,
            output_dir=str(tmp_path / "reports"),
            champion=str(tmp_path / "champion.pth"),
            candidate_stochastic=True,
        )
    )

    assert captured["v2-candidate"] is True
    assert captured["opponent_stochastic"] is False
    assert report["action_selection"] == "candidate_sample"
