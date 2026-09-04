import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from standalone_bot import (  # noqa: E402
    _adapt_outbound_payment_schema,
    _normalize_inbound_player_schema,
    _uses_lowercase_payment_mc,
)


def test_legacy_megacredits_schema_is_normalized_and_round_tripped_for_payment() -> None:
    response = {
        "thisPlayer": {"megacredits": 15, "megacreditProduction": 6},
        "players": [{"megacredits": 2, "megacreditProduction": 1}],
    }

    assert _uses_lowercase_payment_mc(response) is True
    normalized = _normalize_inbound_player_schema(response)
    assert normalized["thisPlayer"]["megaCredits"] == 15
    assert normalized["thisPlayer"]["megaCreditProduction"] == 6

    decoded = {
        "type": "or",
        "response": {
            "type": "projectCard",
            "payment": {"megaCredits": 15, "titanium": 4},
        },
    }
    wire = _adapt_outbound_payment_schema(decoded, lowercase_mc=True)
    payment = wire["response"]["payment"]
    assert payment["megacredits"] == 15
    assert "megaCredits" not in payment
    assert payment["titanium"] == 4


def test_current_payment_schema_keeps_camel_case() -> None:
    wire = _adapt_outbound_payment_schema(
        {"type": "projectCard", "payment": {"megacredits": 9}},
        lowercase_mc=False,
    )
    assert wire["payment"] == {"megaCredits": 9}
