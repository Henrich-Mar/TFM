from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_contract import (
    ACTION_BASES,
    ACTION_RANGES,
    Action,
    LegalActionSet,
    action_namespace,
    validate_action_ranges,
)
from models.action_decoder import ActionDecoder, ActionEnumerationError


def test_current_action_namespaces_are_disjoint() -> None:
    validate_action_ranges(ACTION_RANGES)
    assert action_namespace(ACTION_BASES["play_card"]) == "play_card"
    assert action_namespace(ACTION_BASES["startup_selection"]) == "startup_selection"
    assert action_namespace(999) is None


def test_overlapping_action_namespaces_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_action_ranges({"first": range(10, 20), "second": range(19, 30)})


def test_action_requires_registered_id_and_payload() -> None:
    action = Action(
        action_id=520,
        family="card_selection",
        payload={"type": "card", "cards": ["Card A"]},
        description="Select Card A",
    )
    assert action_namespace(action.action_id) == "card_selection"

    with pytest.raises(ValueError, match="outside the action registry"):
        Action(action_id=999, family="pass", payload={"type": "pass"}, description="Pass")
    with pytest.raises(TypeError, match="payload"):
        Action(action_id=520, family="card_selection", payload=[], description="Select")


def test_legal_action_set_distinguishes_active_terminal_and_invalid() -> None:
    action = Action(action_id=12, family="play_card", payload={"type": "card"}, description="Play")
    assert LegalActionSet(status="active", actions=[action]).action_ids == [12]
    assert LegalActionSet(status="terminal", actions=[]).action_ids == []
    assert LegalActionSet(status="invalid", actions=[], reason="missing prompt").reason == "missing prompt"

    with pytest.raises(ValueError, match="active"):
        LegalActionSet(status="active", actions=[])
    with pytest.raises(ValueError, match="terminal"):
        LegalActionSet(status="terminal", actions=[action])
    with pytest.raises(ValueError, match="reason"):
        LegalActionSet(status="invalid", actions=[])


def test_decoder_distinguishes_terminal_invalid_and_active_states() -> None:
    decoder = ActionDecoder()
    assert decoder.enumerate_legal_actions({"game": {"phase": "end"}}).status == "terminal"
    missing = decoder.enumerate_legal_actions({"game": {"phase": "action"}})
    assert missing.status == "invalid"
    assert "waitingFor" in str(missing.reason)

    active = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "selectPlayer", "players": [{"name": "A2"}]},
    })
    assert active.status == "active"
    assert active.actions[0].payload == {"type": "player", "player": {"name": "A2"}}


def test_pass_requires_explicit_prompt_permission() -> None:
    decoder = ActionDecoder()
    allowed = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "selectPlayer", "players": [], "canPass": True},
    })
    assert allowed.status == "active"
    assert allowed.action_ids == [ACTION_BASES["pass"]]
    assert allowed.actions[0].payload == {"type": "pass"}

    denied = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "selectPlayer", "players": []},
    })
    assert denied.status == "invalid"
    with pytest.raises(ActionEnumerationError, match="explicitly allow pass"):
        decoder.decode_action(ACTION_BASES["pass"], {
            "waitingFor": {"type": "selectPlayer", "players": []},
        })


def test_unknown_prompt_decoder_failure_and_duplicate_ids_are_invalid(monkeypatch) -> None:
    decoder = ActionDecoder()
    unknown = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "futurePrompt"},
    })
    assert unknown.status == "invalid"
    assert "unsupported prompt" in str(unknown.reason)

    monkeypatch.setattr(decoder, "_get_available_action_indices", lambda _state: [600, 600])
    duplicate = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "selectPlayer", "players": [{"name": "A2"}]},
    })
    assert duplicate.status == "invalid"
    assert "duplicate action IDs" in str(duplicate.reason)

    def explode(_state):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(decoder, "_get_available_action_indices", explode)
    failed = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "selectPlayer", "players": [{"name": "A2"}]},
    })
    assert failed.status == "invalid"
    assert failed.reason == "decoder exploded"
