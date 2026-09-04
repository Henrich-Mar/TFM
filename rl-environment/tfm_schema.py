"""Compatibility helpers for Terraforming Mars player and payment models.

The upstream API changed its public M€ fields to ``megacredits`` and
``megacreditProduction``.  The policy was trained with the older camel-case
names, so preserve those aliases internally and emit the upstream wire format.
"""
from copy import deepcopy
from typing import Any, Dict, Iterator


PAYMENT_RESOURCE_KEYS = (
    "megacredits", "heat", "steel", "titanium", "plants", "microbes",
    "floaters", "lunaArchivesScience", "spireScience", "seeds",
    "auroraiData", "graphene", "kuiperAsteroids",
)


def iter_player_models(player_state: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    this_player = player_state.get("thisPlayer")
    if isinstance(this_player, dict):
        yield this_player
    for player in player_state.get("players", []) or []:
        if isinstance(player, dict):
            yield player


def normalize_inbound_player_schema(player_state: Dict[str, Any]) -> Dict[str, Any]:
    """Provide legacy aliases without changing data retained from the server."""
    for player in iter_player_models(player_state):
        if "megaCredits" not in player and "megacredits" in player:
            player["megaCredits"] = player["megacredits"]
        if "megaCreditProduction" not in player and "megacreditProduction" in player:
            player["megaCreditProduction"] = player["megacreditProduction"]
    return player_state


def uses_lowercase_payment_mc(player_state: Dict[str, Any]) -> bool:
    return any(
        "megacredits" in player and "megaCredits" not in player
        for player in iter_player_models(player_state)
    )


def adapt_outbound_payment_schema(input_data: Dict[str, Any], lowercase_mc: bool) -> Dict[str, Any]:
    """Rewrite nested payment objects for the connected server's API version.

    Newer upstream validation also requires a complete payment object.  Add
    every supported resource only for that version, leaving legacy servers'
    smaller payloads untouched.
    """
    payload = deepcopy(input_data)

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        payment = value.get("payment")
        if isinstance(payment, dict):
            if lowercase_mc:
                if "megaCredits" in payment:
                    payment["megacredits"] = payment.pop("megaCredits")
                for key in PAYMENT_RESOURCE_KEYS:
                    payment.setdefault(key, 0)
            elif "megacredits" in payment:
                payment["megaCredits"] = payment.pop("megacredits")
        for nested in value.values():
            visit(nested)

    visit(payload)
    return payload
