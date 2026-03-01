import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

if "aiohttp" not in sys.modules:
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = object
    aiohttp_stub.ClientTimeout = object
    aiohttp_stub.TCPConnector = object
    aiohttp_stub.ClientError = Exception
    sys.modules["aiohttp"] = aiohttp_stub

from tournament_manager import TournamentManager


@dataclass
class _FakeAgent:
    id: str
    telemetry: Any

    async def play_game(self, _game_instance, _player_name):
        return self.telemetry


class _FakeGameInstance:
    def __init__(self, game_id: str, final_state: Dict[str, Any]):
        self.game_id = game_id
        self._final_state = final_state
        self.base_url = "http://internal"

    def get_public_game_url(self) -> str:
        return f"http://public/game?id={self.game_id}"

    async def get_final_state(self) -> Dict[str, Any]:
        return self._final_state

    async def cleanup(self) -> None:
        return None


class _FakeCluster:
    def __init__(self):
        self.recent_games: List[Dict[str, str]] = []
        self._counter = 0

    async def create_game(self, game_id: str, player_names: List[str], game_options: Dict[str, Any]):
        _ = game_options
        self._counter += 1
        final_players = []
        for idx, name in enumerate(player_names):
            vp_total = 100 - (idx * 10)
            final_players.append(
                {
                    "name": name,
                    "terraformRating": 20 - idx,
                    "megaCredits": 10 + idx,
                    "victoryPointsBreakdown": {
                        "total": vp_total,
                        "terraformRating": 20 - idx,
                        "milestones": 5,
                        "awards": 3,
                        "greenery": 4,
                        "city": 2,
                        "victoryPoints": 6,
                    },
                    "citiesCount": 3,
                }
            )
        final_state = {"players": final_players, "game": {"generation": 9}}
        return _FakeGameInstance(game_id=f"g{self._counter}", final_state=final_state)


def test_tournament_manager_merges_agent_play_game_telemetry() -> None:
    agents = [
        _FakeAgent(
            id="agent-a",
            telemetry={
                "agent_id": "agent-a",
                "draft_decisions": 10,
                "draft_decisions_low_hand_ev": 4,
                "hate_draft_picks": 7,
                "hate_draft_picks_low_hand_ev": 3,
                "hate_draft_rate": 0.7,
                "hate_draft_rate_low_hand_ev": 0.75,
            },
        ),
        _FakeAgent(id="agent-b", telemetry={"agent_id": "agent-b", "draft_decisions": 8, "hate_draft_picks": 2, "hate_draft_rate": 0.25}),
        _FakeAgent(id="agent-c", telemetry={"agent_id": "agent-c", "draft_decisions": 6, "hate_draft_picks": 1, "hate_draft_rate": 1.0 / 6.0}),
        _FakeAgent(id="agent-d", telemetry={"agent_id": "agent-d", "draft_decisions": 5, "hate_draft_picks": 0, "hate_draft_rate": 0.0}),
    ]

    manager = TournamentManager(game_cluster=_FakeCluster())
    result = asyncio.run(manager._run_single_game(agents=agents, tournament_id="tid"))

    assert result.completed is True
    assert len(result.players) == 4
    players_by_agent = {row["agent_id"]: row for row in result.players}
    assert players_by_agent["agent-a"]["draft_decisions"] == 10
    assert players_by_agent["agent-a"]["hate_draft_picks_low_hand_ev"] == 3
    assert players_by_agent["agent-a"]["hate_draft_rate"] == pytest.approx(0.7)
    assert players_by_agent["agent-b"]["draft_decisions_low_hand_ev"] == 0
    assert players_by_agent["agent-b"]["hate_draft_rate_low_hand_ev"] == pytest.approx(0.0)
    assert "rank" in players_by_agent["agent-a"]
    assert "victory_points" in players_by_agent["agent-a"]


def test_tournament_manager_defaults_missing_telemetry_to_zero() -> None:
    agents = [
        _FakeAgent(id="agent-a", telemetry=None),
        _FakeAgent(id="agent-b", telemetry={"agent_id": "agent-b"}),
        _FakeAgent(id="agent-c", telemetry={"agent_id": "agent-c", "draft_decisions": 3}),
        _FakeAgent(id="agent-d", telemetry={"agent_id": "agent-d", "hate_draft_picks": 2}),
    ]

    manager = TournamentManager(game_cluster=_FakeCluster())
    result = asyncio.run(manager._run_single_game(agents=agents, tournament_id="tid"))

    assert result.completed is True
    for row in result.players:
        assert "draft_decisions" in row
        assert "draft_decisions_low_hand_ev" in row
        assert "hate_draft_picks" in row
        assert "hate_draft_picks_low_hand_ev" in row
        assert "hate_draft_rate" in row
        assert "hate_draft_rate_low_hand_ev" in row
        assert row["draft_decisions"] >= 0
        assert row["hate_draft_picks"] >= 0
