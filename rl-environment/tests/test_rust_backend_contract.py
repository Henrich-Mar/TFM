import json
import sys

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")


rust_tfm_rl = pytest.importorskip("rust_tfm_rl")


def test_backend_info_contract() -> None:
    info = rust_tfm_rl.backend_info()
    assert isinstance(info, dict)
    assert info.get("module") == "rust_tfm_rl"
    assert info.get("api_version") == "1.0"
    capabilities = set(info.get("capabilities", []) or [])
    assert {
        "encode_state",
        "estimate_affordability",
        "can_afford_cards",
        "enumerate_card_selection_combos",
        "rank_startup_plans",
        "backend_info",
    }.issubset(capabilities)


def test_encode_state_signature_and_length() -> None:
    payload = {
        "game": {"oxygenLevel": 4, "temperature": -24, "venusScaleLevel": 8, "generation": 3},
        "thisPlayer": {"megaCredits": 35, "steel": 4, "titanium": 2, "terraformRating": 21},
    }
    encoded = rust_tfm_rl.encode_state(json.dumps(payload), 1, 1024)
    assert len(encoded) == 1024
