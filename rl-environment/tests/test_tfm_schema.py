import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from tfm_schema import (  # noqa: E402
    PAYMENT_RESOURCE_KEYS,
    adapt_outbound_payment_schema,
    normalize_inbound_player_schema,
    uses_lowercase_payment_mc,
)


def test_upstream_player_fields_are_normalized_for_existing_policy() -> None:
    state = {
        "thisPlayer": {"megacredits": 15, "megacreditProduction": 6},
        "players": [{"megacredits": 2, "megacreditProduction": 1}],
    }

    assert uses_lowercase_payment_mc(state) is True
    normalized = normalize_inbound_player_schema(state)
    assert normalized["thisPlayer"]["megaCredits"] == 15
    assert normalized["thisPlayer"]["megaCreditProduction"] == 6


def test_upstream_payment_is_complete_and_uses_megacredits() -> None:
    wire = adapt_outbound_payment_schema(
        {"type": "or", "response": {"type": "projectCard", "payment": {"megaCredits": 15, "titanium": 4}}},
        lowercase_mc=True,
    )
    payment = wire["response"]["payment"]
    assert set(payment) == set(PAYMENT_RESOURCE_KEYS)
    assert payment["megacredits"] == 15
    assert payment["titanium"] == 4
    assert "megaCredits" not in payment
