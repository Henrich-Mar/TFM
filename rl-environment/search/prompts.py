"""Prompt classification for search roots and rollout continuations."""
from __future__ import annotations

from typing import Any, Dict, Optional

STRATEGIC_PROMPT_TYPE = "or"
SUPPORTED_ROOT_PROMPT_TYPES = {"or", "card", "space"}


def waiting_for(player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(player_state, dict):
        return {}
    waiting = player_state.get("waitingFor")
    return waiting if isinstance(waiting, dict) else {}


def prompt_type(player_state: Optional[Dict[str, Any]]) -> str:
    return str(waiting_for(player_state).get("type", "") or "").strip().lower()


def strategic_options(player_state: Optional[Dict[str, Any]]) -> list:
    """Options of a top-level ``or`` prompt, or an empty list for any other type."""
    waiting = waiting_for(player_state)
    if str(waiting.get("type", "") or "").strip().lower() != STRATEGIC_PROMPT_TYPE:
        return []
    options = waiting.get("options")
    return options if isinstance(options, list) else []


def is_strategic_prompt(player_state: Optional[Dict[str, Any]]) -> bool:
    """A root is searchable only at a top-level action-selection prompt.

    Forced choices, payments, placements, and other continuation prompts stay
    policy-only, matching the Phase 2 macro-action design.
    """
    return len(strategic_options(player_state)) >= 2


def is_search_root(player_state: Optional[Dict[str, Any]], root_prompt_types: str = "or") -> bool:
    """Root gate for ``SearchPolicy.decide``, wider than rollout leaves.

    ``root_prompt_types`` is a comma-separated list. Research ``card`` and
    Action-phase ``space`` roots are supported in addition to ``or``; rollout
    leaves and continuations always use :func:`is_strategic_prompt` regardless.
    """
    allowed = {
        t.strip().lower()
        for t in str(root_prompt_types or "or").split(",")
        if t.strip().lower() in SUPPORTED_ROOT_PROMPT_TYPES
    }
    if allowed == {STRATEGIC_PROMPT_TYPE}:
        return is_strategic_prompt(player_state)
    waiting = waiting_for(player_state)
    prompt = str(waiting.get("type", "") or "").strip().lower()
    if prompt not in allowed:
        return False
    phase = str(((player_state or {}).get("game", {}) or {}).get("phase", "") or "").strip().lower()
    if prompt == STRATEGIC_PROMPT_TYPE:
        options = waiting.get("options")
        return isinstance(options, list) and len(options) >= 2
    if prompt == "space":
        spaces = waiting.get("availableSpaces", waiting.get("spaces", []))
        return phase == "action" and isinstance(spaces, list) and len(spaces) >= 2
    if prompt == "card":
        cards = waiting.get("cards", [])
        if phase != "research" or not isinstance(cards, list) or not cards:
            return False
        try:
            minimum = max(0, int(waiting.get("min", 0) or 0))
            maximum = min(len(cards), int(waiting.get("max", len(cards)) or 0))
        except (TypeError, ValueError):
            return False
        if maximum < minimum:
            return False
        if minimum < maximum:
            return True
        # With one exact subset size, there is a choice only when at least two
        # different combinations exist (0 < k < number of offered cards).
        return 0 < minimum < len(cards)
    return False


def is_terminal_prompt(player_state: Optional[Dict[str, Any]]) -> bool:
    waiting = waiting_for(player_state)
    if not waiting:
        return True
    waiting_type = prompt_type(player_state)
    return waiting_type in {"", "nothing", "pass"} and not waiting.get("options")
