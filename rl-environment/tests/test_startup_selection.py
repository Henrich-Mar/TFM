import sys
from typing import Any, Dict, List


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

import models.action_decoder as action_decoder_module
from models.action_decoder import (
    ActionDecoder,
    _card_keep_cost,
    _card_starting_megacredits,
    _enumerate_startup_plan_payloads,
    _select_initial_card_names,
)


def _startup_waiting_for() -> Dict[str, Any]:
    return {
        "type": "initialCards",
        "options": [
            {
                "title": "Select corporation",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {"name": "United Nations Mars Initiative", "calculatedCost": 0},
                    {"name": "Celestic", "calculatedCost": 0},
                    {"name": "Polyphemos", "calculatedCost": 0},
                ],
                "min": 1,
                "max": 1,
            },
            {
                "title": "Select 2 Prelude cards",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {"name": "Business Empire", "calculatedCost": 0},
                    {"name": "Loan", "calculatedCost": 0},
                    {"name": "Acquired Space Agency", "calculatedCost": 0},
                    {"name": "Recession", "calculatedCost": 0},
                ],
                "min": 2,
                "max": 2,
            },
            {
                "title": "Select initial cards to buy",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {"name": "Underground Detonations", "calculatedCost": 6},
                    {"name": "Space Mirrors", "calculatedCost": 3},
                    {"name": "Colonist Shuttles", "calculatedCost": 12},
                    {"name": "Water Splitting Plant", "calculatedCost": 12},
                    {"name": "Venus Governor", "calculatedCost": 4},
                    {"name": "Local Shading", "calculatedCost": 4},
                    {"name": "Tycho Road Network", "calculatedCost": 15},
                    {"name": "Darkside Observatory", "calculatedCost": 12},
                    {"name": "Ishtar Expedition", "calculatedCost": 6},
                    {"name": "Local Heat Trapping", "calculatedCost": 1},
                ],
                "min": 0,
                "max": 10,
            },
        ],
    }


def _corp_economics_metadata() -> Dict[str, Dict[str, Any]]:
    return {
        "United Nations Mars Initiative": {
            "startingMegaCredits": 40,
            "cardCost": 3,
            "tags": ["Earth"],
            "category": "corporation",
        },
        "Celestic": {
            "startingMegaCredits": 42,
            "cardCost": 3,
            "tags": ["Venus"],
            "category": "corporation",
        },
        "Polyphemos": {
            "startingMegaCredits": 50,
            "cardCost": 5,
            "tags": ["Jovian"],
            "category": "corporation",
        },
    }


def _player_state(waiting_for: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "waitingFor": waiting_for,
        "thisPlayer": {
            "megaCredits": 0,
            "cardCost": 3,
            "steel": 0,
            "titanium": 0,
            "steelValue": 2,
            "titaniumValue": 3,
        },
        "game": {"temperature": -30, "oceans": 0, "venusScaleLevel": 0, "moon": {}},
    }


def _response_cards(response_payload: Dict[str, Any], idx: int) -> List[str]:
    responses = response_payload.get("responses", []) or []
    if idx >= len(responses):
        return []
    return list((responses[idx] or {}).get("cards", []) or [])


def test_select_initial_cards_cap_zero_selects_none() -> None:
    option = {
        "title": "Select initial cards to buy",
        "type": "card",
        "cards": [
            {"name": "A", "calculatedCost": 4},
            {"name": "B", "calculatedCost": 5},
        ],
        "min": 0,
        "max": 2,
    }
    selected = _select_initial_card_names(
        option=option,
        role="project",
        project_tag_counts={},
        cap=0,
        force_minimum=0,
    )
    assert selected == []


def test_metadata_lookup_for_corporation_economics(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    assert _card_starting_megacredits({"name": "United Nations Mars Initiative"}, default=0) == 40
    assert _card_keep_cost({"name": "United Nations Mars Initiative"}, default=3) == 3
    assert _card_starting_megacredits({"name": "Celestic"}, default=0) == 42
    assert _card_keep_cost({"name": "Celestic"}, default=3) == 3
    assert _card_starting_megacredits({"name": "Polyphemos"}, default=0) == 50
    assert _card_keep_cost({"name": "Polyphemos"}, default=3) == 5


def test_startup_bundle_generator_respects_project_keep_legality(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _startup_waiting_for()
    player_state = _player_state(waiting_for)

    plans = _enumerate_startup_plan_payloads(waiting_for, player_state, max_plans=32)
    assert plans, "expected at least one startup plan"

    economics = {
        "United Nations Mars Initiative": (40, 3),
        "Celestic": (42, 3),
        "Polyphemos": (50, 5),
    }
    max_cards = int(waiting_for["options"][2]["max"])
    min_cards = int(waiting_for["options"][2]["min"])
    offered_project_names = {c["name"] for c in waiting_for["options"][2]["cards"]}

    for payload in plans:
        corp_cards = _response_cards(payload, 0)
        project_cards = _response_cards(payload, 2)
        assert len(corp_cards) == 1
        corp_name = corp_cards[0]
        start_mc, keep_cost = economics[corp_name]
        legal_max = min(max_cards, start_mc // keep_cost)
        assert min_cards <= len(project_cards) <= legal_max
        assert len(project_cards) == len(set(project_cards))
        assert set(project_cards).issubset(offered_project_names)


def test_startup_decode_keeps_more_than_one_card_when_legal(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _startup_waiting_for()
    player_state = _player_state(waiting_for)
    decoder = ActionDecoder()

    available_actions = decoder.get_available_actions(player_state)
    startup_actions = [a for a in available_actions if 850 <= int(a) < 882]
    assert 800 in available_actions
    assert startup_actions, "expected startup bundle actions to be exposed"

    startup_response = decoder.decode_action(startup_actions[0], player_state)
    assert startup_response is not None
    assert startup_response.get("type") == "initialCards"
    assert len(_response_cards(startup_response, 0)) == 1
    assert len(_response_cards(startup_response, 1)) == 2
    assert len(_response_cards(startup_response, 2)) > 1
