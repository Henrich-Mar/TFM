"""Contract tests for the Python search client against tfm.rl-search.v1."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.simulator_client import (  # noqa: E402
    SCHEMA_VERSION,
    SearchClient,
    SearchProtocolError,
    SearchServiceError,
    SearchUnavailableError,
)


def _start_payload(**overrides):
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "sessionId": "ses-1",
        "rootDigest": "d" * 64,
        "rootPlayerId": "p1",
        "observation": {
            "id": "p1",
            "players": [{"id": "p1", "megacredits": 12}, {"id": "p2", "megacredits": 30}],
            "thisPlayer": {"id": "p1", "megacredits": 12},
            "waitingFor": {"type": "or", "options": [{"title": "a"}, {"title": "b"}]},
        },
    }
    payload.update(overrides)
    return payload


class _FakeResponseCM:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def read(self):
        return self._body.encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeHttpSession:
    closed = False

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    def post(self, url, headers=None, data=None):
        self.calls.append((url, headers, json.loads(data)))
        status, body = self.queue.pop(0)
        return _FakeResponseCM(status, body)

    async def close(self):
        pass


def _client(queue, token="tok"):
    client = SearchClient("http://server:8080", token=token, timeout_sec=5.0)
    client._session = _FakeHttpSession(queue)
    return client


def test_start_parses_session_and_normalizes_observation():
    client = _client([(200, json.dumps(_start_payload()))])
    result = asyncio.run(client.start("p1"))
    url, headers, payload = client._session.calls[0]
    assert url == "http://server:8080/api/rl/search/start"
    assert payload == {"playerId": "p1"}
    assert headers["x-rl-control-token"] == "tok"
    assert result.session_id == "ses-1"
    assert result.root_player_id == "p1"
    # Upstream lowercase megacredits keeps the legacy camelCase alias and is
    # detected for payment adaptation, mirroring the live game client.
    assert result.observation["thisPlayer"]["megaCredits"] == 12
    assert result.lowercase_mc is True
    assert client.stats.starts == 1


def test_start_requires_schema_version():
    client = _client([(200, json.dumps(_start_payload(schemaVersion="tfm.rl-search.v0")))])
    with pytest.raises(SearchProtocolError):
        asyncio.run(client.start("p1"))


def test_start_rejects_non_object_observation():
    client = _client([(200, json.dumps(_start_payload(observation="nope")))])
    with pytest.raises(SearchProtocolError):
        asyncio.run(client.start("p1"))


def test_start_rejects_missing_session_id():
    client = _client([(200, json.dumps(_start_payload(sessionId="")))])
    with pytest.raises(SearchProtocolError):
        asyncio.run(client.start("p1"))


def test_structured_error_body_maps_to_unavailable_by_code():
    client = _client(
        [
            (
                400,
                json.dumps(
                    {"schemaVersion": SCHEMA_VERSION, "error": {"code": "session_capacity", "message": "At most 32"}}
                ),
            )
        ]
    )
    with pytest.raises(SearchUnavailableError) as caught:
        asyncio.run(client.start("p1"))
    assert caught.value.code == "session_capacity"
    assert client.stats.failures["session_capacity"] == 1


def test_http_404_search_service_disabled_is_unavailable():
    client = _client([(404, "<html>404 Not Found</html>")])
    with pytest.raises(SearchUnavailableError) as caught:
        asyncio.run(client.start("p1"))
    assert caught.value.code == "http_404"
    assert caught.value.status == 404


def test_unauthorized_is_unavailable():
    client = _client([(401, "not authorized")], token="wrong")
    with pytest.raises(SearchUnavailableError) as caught:
        asyncio.run(client.start("p1"))
    assert caught.value.code == "unauthorized"


def test_determinization_disabled_error_is_unavailable():
    client = _client(
        [
            (
                400,
                json.dumps(
                    {
                        "schemaVersion": SCHEMA_VERSION,
                        "error": {"code": "determinization_disabled", "message": "gate not passed"},
                    }
                ),
            )
        ]
    )
    branches = [SearchClient.build_branch_request("b", "determinized", [], determinization_seed=3)]
    with pytest.raises(SearchUnavailableError):
        asyncio.run(client.replay("ses-1", branches))


def test_missing_token_refuses_requests():
    saved = os.environ.pop("RL_CONTROL_TOKEN", None)
    try:
        client = SearchClient("http://server:8080", token=None)
        with pytest.raises(SearchUnavailableError) as caught:
            asyncio.run(client.start("p1"))
        assert caught.value.code == "missing_token"
    finally:
        if saved is not None:
            os.environ["RL_CONTROL_TOKEN"] = saved


def test_replay_parses_all_statuses():
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "sessionId": "ses-1",
        "results": [
            {
                "branchId": "b0",
                "status": "next_prompt",
                "appliedSteps": 1,
                "stateDigest": "a" * 64,
                "nextPrompt": {
                    "playerId": "p2",
                    "observation": {"id": "p2", "players": [], "waitingFor": {"type": "option"}},
                },
            },
            {
                "branchId": "b1",
                "status": "terminal",
                "appliedSteps": 4,
                "stateDigest": "b" * 64,
                "terminal": {
                    "players": [
                        {"playerId": "p1", "rank": 1, "vp": 90},
                        {"playerId": "p2", "rank": 2, "vp": 70},
                    ]
                },
            },
            {
                "branchId": "b2",
                "status": "rejected",
                "appliedSteps": 0,
                "stateDigest": "c" * 64,
                "error": {"code": "input_rejected", "stepIndex": 0, "message": "not enough MC"},
            },
            {
                "branchId": "b3",
                "status": "boundary",
                "appliedSteps": 2,
                "stateDigest": "d" * 64,
            },
        ],
    }
    client = _client([(200, json.dumps(body))])
    results = asyncio.run(
        client.replay(
            "ses-1",
            [SearchClient.build_branch_request("b0", "exact", [{"playerId": "p1", "input": {"type": "option"}}])],
        )
    )
    by_id = {row.branch_id: row for row in results}
    assert by_id["b0"].next_prompt[0] == "p2"
    assert by_id["b0"].usable
    assert by_id["b1"].terminal_players[0]["rank"] == 1
    assert by_id["b2"].error_code == "input_rejected"
    assert by_id["b2"].error_step_index == 0
    assert by_id["b3"].status == "boundary"
    assert client.stats.replays == 1
    assert client.stats.applied_steps == 7


def test_replay_rejects_unknown_branch_status():
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "sessionId": "ses-1",
        "results": [{"branchId": "b0", "status": "exploded", "appliedSteps": 0, "stateDigest": "x"}],
    }
    client = _client([(200, json.dumps(body))])
    with pytest.raises(SearchProtocolError):
        asyncio.run(client.replay("ses-1", [SearchClient.build_branch_request("b0", "exact", [])]))


def test_replay_validates_limits_before_http():
    client = _client([])
    branches = [SearchClient.build_branch_request(f"b{i}", "exact", []) for i in range(65)]
    with pytest.raises(SearchServiceError):
        asyncio.run(client.replay("ses-1", branches))
    with pytest.raises(SearchServiceError):
        asyncio.run(
            client.replay(
                "ses-1",
                [
                    SearchClient.build_branch_request(
                        "b0", "exact", [{"playerId": "p1", "input": {"type": "option"}}] * 33
                    )
                ],
            )
        )
    assert client._session.calls == []


def test_branch_request_carries_determinization_seed_only_when_determinized():
    det = SearchClient.build_branch_request("b0", "determinized", [], determinization_seed=17)
    exact = SearchClient.build_branch_request("b1", "exact", [])
    assert det == {"branchId": "b0", "mode": "determinized", "steps": [], "determinizationSeed": 17}
    assert "determinizationSeed" not in exact


def test_close_reports_unknown_session_without_error():
    client = _client([(200, json.dumps({"schemaVersion": SCHEMA_VERSION, "sessionId": "gone", "closed": False}))])
    assert asyncio.run(client.close("gone")) is False
