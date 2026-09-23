"""Opt-in live server acceptance tests for canonical card/payment actions.

Set ``TFM_LIVE_ROUNDTRIP_CASES`` to a JSON file before running this module.
The file must contain independent disposable game/player cases, for example::

    {
      "cases": [
        {
          "base_url": "http://127.0.0.1:8080",
          "player_id": "player-id",
          "family": "card_subset"
        },
        {
          "base_url": "http://127.0.0.1:8081",
          "player_id": "player-id",
          "family": "select_payment",
          "action_id": 401
        }
      ]
    }

Each case fetches the live player view, selects the requested family (or the
explicit action ID), decodes it through ``ActionDecoder``, and posts it back
to the same server.  Cases should use separate disposable games because a
successful input advances the game state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from models.action_decoder import ActionDecoder  # noqa: E402
from tfm_schema import (  # noqa: E402
    adapt_outbound_payment_schema,
    normalize_inbound_player_schema,
    uses_lowercase_payment_mc,
)


def _http_json(
    method: str,
    url: str,
    payload: Dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Tuple[int, Any, str]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = raw
            return int(response.status), parsed, raw
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), raw, raw


def _extract_run_id(player_state: Dict[str, Any]) -> str:
    for key in ("runId", "runID", "run_id"):
        value = player_state.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for scope_key in ("game", "thisPlayer", "waitingFor"):
        scoped = player_state.get(scope_key)
        if not isinstance(scoped, dict):
            continue
        for key in ("runId", "runID", "run_id"):
            value = scoped.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _case_url(case: Dict[str, Any], path: str) -> str:
    base_url = str(case.get("base_url", "")).rstrip("/")
    player_id = str(case.get("player_id", ""))
    if not base_url or not player_id:
        raise AssertionError("live round-trip cases require base_url and player_id")
    return f"{base_url}{path}?{urlencode({'id': player_id})}"


def _iter_cases(fixture: Any) -> Iterable[Dict[str, Any]]:
    cases = fixture.get("cases") if isinstance(fixture, dict) else fixture
    if not isinstance(cases, list) or not cases:
        raise AssertionError("live round-trip fixture must contain a non-empty cases list")
    for case in cases:
        if not isinstance(case, dict):
            raise AssertionError("each live round-trip case must be an object")
        yield case


def test_live_server_accepts_canonical_card_and_payment_actions() -> None:
    fixture_name = os.getenv("TFM_LIVE_ROUNDTRIP_CASES", "").strip()
    if not fixture_name:
        pytest.skip("set TFM_LIVE_ROUNDTRIP_CASES to run live server round-trips")

    fixture_path = Path(fixture_name)
    if not fixture_path.is_file():
        raise AssertionError(f"live round-trip fixture does not exist: {fixture_path}")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    timeout = float(os.getenv("TFM_LIVE_ROUNDTRIP_TIMEOUT_SEC", "10"))

    for case_number, case in enumerate(_iter_cases(fixture), start=1):
        status, player_state, raw = _http_json(
            "GET",
            _case_url(case, "/api/player"),
            timeout=timeout,
        )
        assert status == 200, f"case {case_number}: GET /api/player returned {status}: {raw[:1000]}"
        assert isinstance(player_state, dict), f"case {case_number}: player view was not an object"
        lowercase_payment_mc = uses_lowercase_payment_mc(player_state)
        player_state = normalize_inbound_player_schema(player_state)

        decoder = ActionDecoder()
        legal = decoder.enumerate_legal_actions(player_state)
        assert legal.status == "active", f"case {case_number}: invalid live catalog: {legal.reason}"

        expected_family = str(case.get("family", "")).strip()
        requested_action_id = case.get("action_id")
        if requested_action_id is not None:
            selected = [action for action in legal.actions if action.action_id == int(requested_action_id)]
        elif expected_family:
            selected = [action for action in legal.actions if action.family == expected_family]
        else:
            selected = [action for action in legal.actions if action.family in {"card_subset", "select_payment"}]

        assert selected, (
            f"case {case_number}: no legal action matched family={expected_family!r} "
            f"action_id={requested_action_id!r}; families={[action.family for action in legal.actions]}"
        )
        action = selected[0]
        if expected_family:
            assert action.family == expected_family, (
                f"case {case_number}: action {action.action_id} decoded as {action.family!r}, "
                f"expected {expected_family!r}"
            )

        wire_payload = dict(action.payload)
        run_id = str(case.get("run_id") or _extract_run_id(player_state)).strip()
        if run_id:
            wire_payload["runId"] = run_id
        wire_payload = adapt_outbound_payment_schema(wire_payload, lowercase_payment_mc)

        status, next_player_state, raw = _http_json(
            "POST",
            _case_url(case, "/player/input"),
            payload=wire_payload,
            timeout=timeout,
        )
        assert status == 200, (
            f"case {case_number}: server rejected action {action.action_id} "
            f"({action.family}): HTTP {status}: {raw[:1500]}\nPayload: {wire_payload}"
        )
        assert isinstance(next_player_state, dict), (
            f"case {case_number}: successful input did not return a player view: {raw[:1500]}"
        )
        next_legal = decoder.enumerate_legal_actions(next_player_state)
        if next_legal.status in {"active", "terminal"}:
            continue
        # After a successful input this seat often has no waitingFor while another
        # player is active. That is a valid multiplayer hand-off, not a catalog bug.
        phase = str(((next_player_state.get("game") or {}) if isinstance(next_player_state.get("game"), dict) else {}).get("phase") or "")
        no_prompt = (
            next_legal.status == "invalid"
            and str(next_legal.reason or "") == "active state has no waitingFor prompt"
            and not isinstance(next_player_state.get("waitingFor"), dict)
            and phase not in {"", "end"}
        )
        assert no_prompt, (
            f"case {case_number}: resulting player view has an invalid prompt: {next_legal.reason}"
        )
