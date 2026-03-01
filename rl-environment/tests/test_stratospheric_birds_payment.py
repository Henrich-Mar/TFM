import sys


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from models.action_decoder import _build_payment_with_options


def test_stratospheric_birds_reserves_one_floater() -> None:
    player_state = {
        "thisPlayer": {
            "megaCredits": 10,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
            "plants": 0,
            "floaters": 2,
            "steelValue": 2,
            "titaniumValue": 3,
        }
    }
    card = {
        "name": "Stratospheric Birds",
        "calculatedCost": 11,
        "tags": {"Venus": 1},
    }
    payment_options = {
        "megaCredits": True,
        "floaters": True,
    }

    payment = _build_payment_with_options(
        player_state=player_state,
        card=card,
        payment_options=payment_options,
        reserve_units={},
        waiting_for={},
    )

    assert payment is not None
    assert int(payment.get("floaters", 0) or 0) <= 1


def test_stratospheric_birds_not_affordable_if_only_all_floaters_work() -> None:
    player_state = {
        "thisPlayer": {
            "megaCredits": 5,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
            "plants": 0,
            "floaters": 2,
            "steelValue": 2,
            "titaniumValue": 3,
        }
    }
    card = {
        "name": "Stratospheric Birds",
        "calculatedCost": 11,
        "tags": {"Venus": 1},
    }
    payment_options = {
        "megaCredits": True,
        "floaters": True,
    }

    payment = _build_payment_with_options(
        player_state=player_state,
        card=card,
        payment_options=payment_options,
        reserve_units={},
        waiting_for={},
    )

    assert payment is None
