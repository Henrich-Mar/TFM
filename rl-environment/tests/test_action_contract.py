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
