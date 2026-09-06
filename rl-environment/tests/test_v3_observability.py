import os
import sys

import numpy as np


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from models.action_decoder import ActionDecoder
from models.state_encoder import StateEncoder


def _state() -> dict:
    red = {
        "name": "Red",
        "color": "red",
        "terraformRating": 24,
        "megaCredits": 30,
        "tableau": [],
        "victoryPointsBreakdown": {"total": 0},
    }
    blue = {
        "name": "Blue",
        "color": "blue",
        "terraformRating": 30,
        "megaCredits": 20,
        "megaCreditProduction": 5,
        "tableau": [
            {"name": "Visible Static VP"},
            {"name": "Visible Animals", "resources": 3},
            {"name": "Science Project", "tags": ["science"]},
        ],
        # This remains hidden/zero during the game and must not drive V3.
        "victoryPointsBreakdown": {"total": 0},
    }
    game = {
        "generation": 10,
        "phase": "action",
        "spaces": [
            {"id": "city", "x": 1, "y": 1, "spaceType": "land", "tileType": 2, "color": "blue"},
            {"id": "greenery", "x": 1, "y": 0, "spaceType": "land", "tileType": 0, "color": "blue"},
            {"id": "edge", "x": 1, "y": 2, "spaceType": "land"},
        ],
        "milestones": [{"name": "Mayor", "playerName": "Blue", "playerColor": "blue"}],
        "awards": [
            {
                "name": "Landlord",
                "playerName": "Red",
                "playerColor": "red",
                "scores": [
                    {"playerColor": "blue", "playerScore": 9},
                    {"playerColor": "red", "playerScore": 4},
                ],
            },
            {
                "name": "Banker",
                "scores": [
                    {"playerColor": "blue", "playerScore": 5},
                    {"playerColor": "red", "playerScore": 3},
                ],
            },
        ],
        "temperature": -10,
        "oxygenLevel": 5,
        "oceans": 3,
    }
    return {"thisPlayer": red, "players": [red, blue], "game": game, "waitingFor": {}}


def _encoder(monkeypatch, scale: float) -> StateEncoder:
    monkeypatch.setenv("TFM_RL_V3", "1")
    monkeypatch.setenv("V3_FEATURE_SCALE", str(scale))
    encoder = StateEncoder()
    encoder.card_metadata_by_name.update({
        "Visible Static VP": {"victoryPoints": 2, "tags": ["earth"]},
        "Visible Animals": {
            "resourceType": "Animal",
            "dynamicVictoryPoints": {"points": 1, "target": 1, "itemType": "resource"},
            "vpPerResource": 1,
            "tags": ["animal"],
        },
        "Science Project": {"tags": ["science"]},
    })
    return encoder


def test_public_vp_is_reconstructed_without_hidden_breakdown(monkeypatch) -> None:
    encoder = _encoder(monkeypatch, 1.0)
    state = _state()
    estimate = encoder.estimate_public_vp(state, state["players"][1])

    assert estimate["terraforming"] == 30
    assert estimate["milestones"] == 5
    assert estimate["greenery"] == 1
    assert estimate["city"] == 1
    assert estimate["cards_static"] == 2
    assert estimate["cards_resource"] == 3
    assert estimate["awards_projected"] == 5
    assert estimate["certain"] == 42
    assert estimate["estimate"] == 47


def test_v3_scale_zero_is_exactly_v2_compatible(monkeypatch) -> None:
    state = _state()
    monkeypatch.setenv("TFM_RL_V3", "0")
    v2 = StateEncoder()
    v2.card_metadata_by_name = _encoder(monkeypatch, 0.0).card_metadata_by_name
    monkeypatch.setenv("TFM_RL_V3", "1")
    monkeypatch.setenv("V3_FEATURE_SCALE", "0")
    v3_zero = StateEncoder()
    v3_zero.card_metadata_by_name = v2.card_metadata_by_name

    before = v2.encode(state)
    after = v3_zero.encode(state)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_v3_opponent_token_contains_public_score_and_public_engine(monkeypatch) -> None:
    encoder = _encoder(monkeypatch, 1.0)
    bundle = encoder.encode(_state())
    opponent_token = bundle["world_tokens"][bundle["world_token_types"] == 4][0]

    # Token slot 0 is type; V2 used only the next 16 slots. V3 starts at 17.
    assert opponent_token[17] > 0.0
    assert opponent_token[18] > opponent_token[17]
    assert np.count_nonzero(opponent_token[17:]) >= 10


def test_award_world_and_action_tokens_have_stable_identity(monkeypatch) -> None:
    encoder = _encoder(monkeypatch, 1.0)
    state = _state()
    landlord = encoder._award_token_features(state["game"], state["thisPlayer"], "Landlord")
    banker = encoder._award_token_features(state["game"], state["thisPlayer"], "Banker")
    assert landlord[12:18] != banker[12:18]

    decoder = ActionDecoder()
    landlord_action = decoder._build_action_token(
        state, 600, "fund_award", {"label": "Landlord", "award_name": "Landlord"}, None
    )
    banker_action = decoder._build_action_token(
        state, 601, "fund_award", {"label": "Banker", "award_name": "Banker"}, None
    )
    assert not np.array_equal(landlord_action, banker_action)

    decoder.set_v3_feature_scale(0.0)
    landlord_zero = decoder._build_action_token(
        state, 600, "fund_award", {"label": "Landlord", "award_name": "Landlord"}, None
    )
    banker_zero = decoder._build_action_token(
        state, 601, "fund_award", {"label": "Banker", "award_name": "Banker"}, None
    )
    np.testing.assert_array_equal(landlord_zero, banker_zero)
