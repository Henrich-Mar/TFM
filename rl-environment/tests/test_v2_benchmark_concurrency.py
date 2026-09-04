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
    monkeypatch.setattr(v2_benchmark, "_frozen_neural", lambda checkpoint, agent_id: _FakeAgent(agent_id))
    monkeypatch.setattr(
        v2_benchmark,
        "_baseline_agents",
        lambda kind, seed, champion=None: [_FakeAgent(f"{kind}-{seed}-{index}") for index in range(3)],
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
