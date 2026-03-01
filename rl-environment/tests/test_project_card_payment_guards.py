import sys


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from models.action_decoder import ActionDecoder, build_response_for_input


def _payment_zero(payment: dict) -> bool:
    return all(int(v or 0) == 0 for v in (payment or {}).values())


def test_project_card_prompt_never_emits_zero_payment_for_unaffordable_large_convoy() -> None:
    player_state = {
        "thisPlayer": {
            "megaCredits": 0,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
            "plants": 0,
            "steelValue": 2,
            "titaniumValue": 3,
        }
    }
    base_waiting_for = {
        "cards": [{"name": "Large Convoy", "calculatedCost": 36}],
        "paymentOptions": {"heat": False, "lunaTradeFederationTitanium": True},
    }

    for prompt_type in ["projectCard", "selectProjectCardToPlay"]:
        waiting_for = dict(base_waiting_for)
        waiting_for["type"] = prompt_type
        response = build_response_for_input(waiting_for, action_index=0, player_state=player_state)

        assert response is not None
        assert not (
            response.get("type") == "projectCard"
            and _payment_zero(response.get("payment", {}))
        )


def test_project_card_falls_back_to_affordable_candidate_payment() -> None:
    player_state = {
        "thisPlayer": {
            "megaCredits": 5,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
            "plants": 0,
            "steelValue": 2,
            "titaniumValue": 3,
        }
    }
    waiting_for = {
        "type": "projectCard",
        "cards": [
            {"name": "Large Convoy", "calculatedCost": 36},
            {"name": "Mining Rights", "calculatedCost": 5},
        ],
        "paymentOptions": {"heat": False, "lunaTradeFederationTitanium": True},
    }

    response = build_response_for_input(waiting_for, action_index=0, player_state=player_state)

    assert response is not None
    assert response.get("type") == "projectCard"
    assert response.get("card") == "Mining Rights"
    assert int(response.get("payment", {}).get("megaCredits", 0) or 0) > 0


def test_get_available_actions_project_card_omits_unaffordable_entries() -> None:
    decoder = ActionDecoder()
    player_state = {
        "thisPlayer": {
            "megaCredits": 0,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
            "plants": 0,
            "steelValue": 2,
            "titaniumValue": 3,
        },
        "waitingFor": {
            "type": "projectCard",
            "cards": [{"name": "Large Convoy", "calculatedCost": 36}],
            "paymentOptions": {"heat": False, "lunaTradeFederationTitanium": True},
        },
    }

    available = decoder.get_available_actions(player_state)

    assert available == []
