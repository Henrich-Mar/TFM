import sys
from typing import Any, Dict, List

import pytest

if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

import models.action_decoder as action_decoder_module
from models.action_decoder import (
    ActionDecoder,
    ActionEnumerationError,
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
        "Mining Guild": {
            "startingMegaCredits": 30,
            "cardCost": 3,
            "tags": ["Building", "Building"],
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


def _stage1_waiting_for() -> Dict[str, Any]:
    return {
        "type": "initialCards",
        "options": [
            {
                "title": "Select corporation",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {
                        "name": "United Nations Mars Initiative",
                        "calculatedCost": 0,
                        "tags": ["Earth"],
                    },
                    {
                        "name": "Mining Guild",
                        "calculatedCost": 0,
                        "tags": ["Building"],
                    },
                ],
                "min": 1,
                "max": 1,
            },
            {
                "title": "Select initial cards to buy",
                "buttonLabel": "Save",
                "type": "card",
                "cards": [
                    {"name": "Local Heat Trapping", "calculatedCost": 1, "tags": ["Building"]},
                    {"name": "Space Mirrors", "calculatedCost": 3, "tags": ["Science"]},
                    {"name": "Underground Detonations", "calculatedCost": 6, "tags": ["Building"]},
                    {"name": "Water Splitting Plant", "calculatedCost": 12, "tags": ["Building"]},
                    {"name": "Colonist Shuttles", "calculatedCost": 12, "tags": ["Earth"]},
                    {"name": "Tycho Road Network", "calculatedCost": 15, "tags": ["Building"]},
                    {"name": "Darkside Observatory", "calculatedCost": 12, "tags": ["Science"]},
                    {"name": "Ishtar Expedition", "calculatedCost": 6, "tags": ["Venus"]},
                ],
                "min": 0,
                "max": 10,
            },
        ],
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


def test_startup_decode_quarantines_an_overflowing_action_space(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _startup_waiting_for()
    player_state = _player_state(waiting_for)
    decoder = ActionDecoder()

    catalog = decoder.enumerate_legal_actions(player_state)
    assert catalog.status == "invalid"
    assert catalog.actions == []
    assert "more than 32 legal plans" in str(catalog.reason)
    with pytest.raises(ActionEnumerationError, match="more than 32 legal plans"):
        decoder.get_available_actions(player_state)


def test_startup_plan_does_not_fill_the_full_keep_limit_with_mediocre_cards(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _startup_waiting_for()
    waiting_for["options"][2]["cards"] = [
        {"name": f"Mediocre {idx}", "calculatedCost": 24}
        for idx in range(10)
    ]
    plans = _enumerate_startup_plan_payloads(waiting_for, _player_state(waiting_for), max_plans=8)

    assert plans
    assert len(_response_cards(plans[0], 2)) == 0


def test_stage1_startup_catalog_stays_within_plan_limit(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _stage1_waiting_for()
    player_state = _player_state(waiting_for)
    decoder = ActionDecoder()

    catalog = decoder.enumerate_legal_actions(player_state)
    assert catalog.status == "active"
    assert 1 <= len(catalog.actions) <= 32
    assert all(action.family == "startup_plan" for action in catalog.actions)
    assert all("Keep:" in action.description for action in catalog.actions)


def test_startup_plan_tokens_differ_by_corp_and_keeps(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _stage1_waiting_for()
    player_state = _player_state(waiting_for)
    decoder = ActionDecoder()

    def _token(payload: Dict[str, Any]):
        labels = decoder._descriptor_labels(850, waiting_for, "startup_plan", payload, player_state)
        return labels, decoder._build_action_token(
            player_state=player_state,
            action_index=850,
            family="startup_plan",
            label_info=labels,
            decoded_action=payload,
        )

    empty_payload = {
        "type": "initialCards",
        "responses": [
            {"type": "card", "cards": ["United Nations Mars Initiative"]},
            {"type": "card", "cards": []},
        ],
    }
    keep_payload = {
        "type": "initialCards",
        "responses": [
            {"type": "card", "cards": ["United Nations Mars Initiative"]},
            {"type": "card", "cards": ["Local Heat Trapping", "Underground Detonations"]},
        ],
    }
    other_corp_payload = {
        "type": "initialCards",
        "responses": [
            {"type": "card", "cards": ["Mining Guild"]},
            {"type": "card", "cards": ["Local Heat Trapping", "Underground Detonations"]},
        ],
    }
    empty_labels, empty_token = _token(empty_payload)
    keep_labels, keep_token = _token(keep_payload)
    other_labels, other_token = _token(other_corp_payload)

    assert "Keep: no cards" in empty_labels["label"]
    assert "Local Heat Trapping" in keep_labels["label"]
    assert keep_labels["card_name"] == "United Nations Mars Initiative"
    assert other_labels["card_name"] == "Mining Guild"
    assert empty_token.tolist() != keep_token.tolist()
    assert keep_token.tolist() != other_token.tolist()
    assert empty_token.shape == keep_token.shape

    descriptors = decoder.get_legal_action_descriptors(player_state)
    assert len(descriptors) >= 2
    tokens = {tuple(row["token_features"].tolist()) for row in descriptors}
    assert len(tokens) > 1


def test_startup_keep_roi_does_not_prefer_filling_the_cash_cap(monkeypatch) -> None:
    monkeypatch.setattr(action_decoder_module, "_CARD_META_CACHE", _corp_economics_metadata())
    waiting_for = _stage1_waiting_for()
    plans = _enumerate_startup_plan_payloads(waiting_for, _player_state(waiting_for), max_plans=8)
    assert plans
    top_keeps = _response_cards(plans[0], 1)
    unmi_start_mc, unmi_keep_cost = 40, 3
    mining_start_mc, mining_keep_cost = 30, 3
    corp = _response_cards(plans[0], 0)[0]
    start_mc, keep_cost = (unmi_start_mc, unmi_keep_cost) if corp == "United Nations Mars Initiative" else (mining_start_mc, mining_keep_cost)
    legal_cap = start_mc // keep_cost
    assert len(top_keeps) < legal_cap
