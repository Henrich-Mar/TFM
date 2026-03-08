import sys
import types

import numpy as np
import pytest
import torch


def _install_aiohttp_stub() -> None:
    if "aiohttp" in sys.modules:
        return
    aiohttp_stub = types.ModuleType("aiohttp")
    dummy_session = type("DummySession", (), {"closed": False})
    aiohttp_stub.ClientSession = dummy_session
    aiohttp_stub.ClientConnectionError = Exception
    aiohttp_stub.ServerDisconnectedError = Exception
    aiohttp_stub.ClientOSError = Exception
    aiohttp_stub.ClientPayloadError = Exception
    sys.modules["aiohttp"] = aiohttp_stub


def test_transformer_network_output_shapes() -> None:
    _install_aiohttp_stub()
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")

    from models.agent import AgentConfig, TerraformingMarsNetwork

    config = AgentConfig(
        state_size=1024,
        hidden_size=256,
        transformer_enabled=True,
        card_token_dim=8,
        tableau_token_count=8,
        hand_token_count=4,
        opponent_token_count=4,
        transformer_embed_dim=64,
        transformer_heads=4,
        transformer_layers=2,
    )
    network = TerraformingMarsNetwork(config)
    batch = torch.randn(5, 1024)
    output = network(batch)

    assert tuple(output["policy_logits"].shape) == (5, 1000)
    assert tuple(output["value"].shape) == (5, 1)
    assert tuple(output["aux_predictions"].shape) == (5, 74)


def test_state_encoder_injects_card_token_segment() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
        state_size=1024,
        card_token_dim=8,
        tableau_token_count=8,
        hand_token_count=4,
        opponent_token_count=4,
    )
    player_state = {
        "game": {"generation": 3, "oxygenLevel": 4, "temperature": -20},
        "thisPlayer": {
            "id": "p1",
            "color": "red",
            "megaCredits": 30,
            "steel": 5,
            "titanium": 3,
            "steelValue": 2,
            "titaniumValue": 3,
            "tableau": [
                {"name": "Mars University", "tags": ["Science"], "cost": 12},
                {"name": "Space Elevator", "tags": ["Building", "Space"], "cost": 27},
            ],
        },
        "cardsInHand": [
            {"name": "Olympus Conference", "tags": ["Science"], "cost": 10},
            {"name": "Ganymede Colony", "tags": ["Jovian", "Space"], "cost": 20},
        ],
        "players": [
            {"id": "p1", "color": "red", "tableau": []},
            {
                "id": "p2",
                "color": "blue",
                "tableau": [{"name": "Saturn Systems", "tags": ["Jovian", "Space"], "cost": 12}],
            },
        ],
    }

    try:
        encoded = encoder.encode(player_state)
    except RuntimeError as exc:
        if "rust_tfm_rl" in str(exc):
            pytest.skip("rust_tfm_rl backend not installed")
        raise
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (1024,)
    card_token_tail = encoded[-128:]
    assert float(np.abs(card_token_tail).sum()) > 0.0


def test_state_encoder_merges_prompt_cards_with_owned_hand_context() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
        state_size=1024,
        card_token_dim=8,
        tableau_token_count=8,
        hand_token_count=4,
        opponent_token_count=4,
    )
    player_state = {
        "thisPlayer": {
            "id": "p1",
            "color": "red",
            "cardsInHand": [],
        },
        "cardsInHand": [
            {"name": "Dust Seals", "cost": 2},
            {"name": "Gyropolis", "cost": 20},
            {"name": "Earth Embassy", "cost": 16},
        ],
        "waitingFor": {
            "type": "card",
            "cards": [
                {"name": "Luna Resort", "cost": 21},
                {"name": "Immigrant City", "cost": 13},
                {"name": "Luna Project Office", "cost": 17},
            ],
        },
    }

    owned = encoder._get_owned_hand_cards(player_state)
    candidates = encoder._get_candidate_hand_cards(player_state)

    assert [card["name"] for card in owned] == ["Dust Seals", "Gyropolis", "Earth Embassy"]
    assert [card["name"] for card in candidates[:3]] == ["Luna Resort", "Immigrant City", "Luna Project Office"]
    assert [card["name"] for card in candidates[3:]] == ["Dust Seals", "Gyropolis", "Earth Embassy"]


def test_state_encoder_rejects_token_layout_larger_than_state() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.state_encoder import StateEncoder

    with pytest.raises(ValueError):
        StateEncoder(
            state_size=128,
            card_token_dim=8,
            tableau_token_count=8,
            hand_token_count=64,
            opponent_token_count=6,
        )
