from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_interface import GameServerCluster


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def read(self) -> bytes:
        return b""


class _FakeSession:
    def __init__(self, capability_statuses: list[int]) -> None:
        self.capability_statuses = iter(capability_statuses)
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.get_calls.append(url)
        return _FakeResponse(next(self.capability_statuses))

    def post(self, url: str, **_kwargs) -> _FakeResponse:
        self.post_calls.append(url)
        raise AssertionError("a failed capability preflight must not restart any server")


def test_idle_recycle_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("V2_RECYCLE_IDLE_SERVERS", raising=False)
    monkeypatch.delenv("RL_CONTROL_TOKEN", raising=False)

    async def exercise() -> bool:
        cluster = GameServerCluster(["tfm-server-1:8080"])
        return await cluster.recycle_idle_servers()

    assert asyncio.run(exercise()) is False


def test_idle_recycle_refuses_to_restart_while_a_game_is_tracked(monkeypatch) -> None:
    monkeypatch.setenv("V2_RECYCLE_IDLE_SERVERS", "1")
    monkeypatch.setenv("RL_CONTROL_TOKEN", "test-token")

    async def exercise() -> tuple[bool, bool]:
        cluster = GameServerCluster(["tfm-server-1:8080"])
        cluster.servers[0].active_games = 1
        result = await cluster.recycle_idle_servers()
        return result, cluster.servers[0].healthy

    result, healthy = asyncio.run(exercise())
    assert result is False
    assert healthy is True


def test_idle_recycle_never_partially_restarts_a_mixed_server_cluster(monkeypatch) -> None:
    monkeypatch.setenv("V2_RECYCLE_IDLE_SERVERS", "1")
    monkeypatch.setenv("RL_CONTROL_TOKEN", "test-token")

    async def exercise() -> tuple[bool, _FakeSession, list[bool]]:
        cluster = GameServerCluster(["tfm-server-1:8080", "tfm-server-2:8080"])
        session = _FakeSession([200, 404])
        cluster.ensure_session = lambda timeout_total=None: session
        result = await cluster.recycle_idle_servers()
        return result, session, [server.healthy for server in cluster.servers]

    result, session, healthy = asyncio.run(exercise())
    assert result is False
    assert len(session.get_calls) == 2
    assert session.post_calls == []
    assert healthy == [True, True]
