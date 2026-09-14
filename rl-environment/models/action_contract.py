"""Canonical action metadata shared by decoding, masking, and training.

The current server protocol uses a fixed integer action namespace.  Some
parts of that namespace are contextual (for example, ``600-699`` can select
an award or a player), so the registry describes protocol namespaces rather
than pretending that an integer has one meaning without a game prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional


ActionFamily = Literal[
    "play_card",
    "standard_project",
    "select_option",
    "select_space",
    "select_payment",
    "select_amount",
    "card_selection",
    "target_selection",
    "special",
    "select_colony",
    "select_party",
    "select_delegate",
    "select_global_event",
    "select_underground_token",
    "startup_fallback",
    "startup_selection",
    "select_resource",
    "and",
    "select_policy",
    "pass",
    "end_turn",
    # These semantic families are already emitted by ActionDecoder descriptors.
    "fund_award",
    "select_player",
    "convert_plants",
    "convert_heat",
    "sell_patents",
    "card_subset",
    "card_prompt",
    "claim_milestone",
    "ares_global_parameters",
    "startup_plan",
    "other",
]


ACTION_FAMILIES = frozenset(
    {
        "play_card",
        "standard_project",
        "select_option",
        "select_space",
        "select_payment",
        "select_amount",
        "card_selection",
        "target_selection",
        "special",
        "select_colony",
        "select_party",
        "select_delegate",
        "select_global_event",
        "select_underground_token",
        "startup_fallback",
        "startup_selection",
        "select_resource",
        "and",
        "select_policy",
        "pass",
        "end_turn",
        "fund_award",
        "select_player",
        "convert_plants",
        "convert_heat",
        "sell_patents",
        "card_subset",
        "card_prompt",
        "claim_milestone",
        "ares_global_parameters",
        "startup_plan",
        "other",
    }
)


# These ranges are half-open, matching Python's range semantics.  They are
# protocol namespaces, not a promise that every ID in a namespace is legal in
# every prompt.  Prompt-specific legality remains the decoder's responsibility.
ACTION_RANGES: Dict[str, range] = {
    "play_card": range(0, 100),
    "standard_project": range(100, 200),
    "select_option": range(200, 300),
    "select_space": range(300, 400),
    "select_payment": range(400, 500),
    # Keep the amount namespace disjoint from the existing card-mask namespace.
    "select_amount": range(500, 520),
    "card_selection": range(520, 600),
    # Contextual 600-699 namespace:
    # - 600-649: fund-award leaves (and select-player prompts, which never
    #   coexist with the action-menu award branch)
    # - 650-699: claim-milestone leaves (can coexist with awards on the same OR)
    "target_selection": range(600, 700),
    "special": range(700, 720),
    "select_colony": range(720, 730),
    "select_party": range(730, 740),
    "select_delegate": range(740, 750),
    "select_global_event": range(750, 760),
    "select_underground_token": range(760, 770),
    "startup_fallback": range(800, 801),
    "ares_global_parameters": range(810, 811),
    "select_resource": range(820, 830),
    "and": range(830, 831),
    "select_policy": range(840, 850),
    "startup_selection": range(850, 882),
    "pass": range(900, 950),
    "end_turn": range(950, 951),
}


ACTION_BASES = {name: action_range.start for name, action_range in ACTION_RANGES.items()}
CARD_SELECTION_MASK_LIMIT = len(ACTION_RANGES["card_selection"])
STARTUP_PLAN_LIMIT = len(ACTION_RANGES["startup_selection"])
PAYMENT_ACTION_VARIANTS = 8


def validate_action_ranges(action_ranges: Mapping[str, range]) -> None:
    """Validate that named action namespaces are non-empty and disjoint."""
    ranges = list(action_ranges.items())
    for name, action_range in ranges:
        if not isinstance(action_range, range) or not action_range:
            raise ValueError(f"Action range must be a non-empty range: {name}")

    for index, (name_a, range_a) in enumerate(ranges):
        for name_b, range_b in ranges[index + 1 :]:
            if set(range_a).intersection(range_b):
                raise ValueError(f"Action ranges overlap: {name_a} and {name_b}")


def action_namespace(action_id: int) -> Optional[str]:
    """Return the registered protocol namespace for an action ID."""
    normalized = int(action_id)
    for name, action_range in ACTION_RANGES.items():
        if normalized in action_range:
            return name
    return None


@dataclass(frozen=True)
class Action:
    """One decoded action with a stable ID, family, and server payload."""

    action_id: int
    family: ActionFamily
    payload: Dict[str, Any]
    description: str

    def __post_init__(self) -> None:
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, int):
            raise TypeError("action_id must be an integer")
        if action_namespace(self.action_id) is None:
            raise ValueError(f"action_id is outside the action registry: {self.action_id}")
        if self.family not in ACTION_FAMILIES:
            raise ValueError(f"unknown action family: {self.family!r}")
        if not isinstance(self.payload, dict):
            raise TypeError("action payload must be a dict")
        if not str(self.description or "").strip():
            raise ValueError("action description is required")


@dataclass
class LegalActionSet:
    """Prompt legality with explicit active, terminal, and invalid states."""

    status: Literal["active", "terminal", "invalid"]
    actions: List[Action]
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"active", "terminal", "invalid"}:
            raise ValueError(f"unknown legal-action status: {self.status!r}")
        if not isinstance(self.actions, list) or not all(isinstance(item, Action) for item in self.actions):
            raise TypeError("actions must be a list of Action values")
        action_ids = [item.action_id for item in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("legal actions must not contain duplicate action IDs")
        if self.status == "active" and not self.actions:
            raise ValueError("active legal-action sets require at least one action")
        if self.status == "terminal" and self.actions:
            raise ValueError("terminal legal-action sets must not contain actions")
        if self.status == "invalid" and not str(self.reason or "").strip():
            raise ValueError("invalid legal-action sets require a reason")

    @property
    def action_ids(self) -> List[int]:
        return [item.action_id for item in self.actions]


validate_action_ranges(ACTION_RANGES)
