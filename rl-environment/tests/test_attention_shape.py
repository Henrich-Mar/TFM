import sys
import types

import numpy as np
import pytest
import torch


class _FakeRustModule:
    @staticmethod
    def encode_state(_payload: str, _turn_action_count: int, state_size: int):
        return [0.0] * int(state_size)


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


def test_agent_config_from_env_respects_full_transformer_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")

    from models.agent import AgentConfig

    monkeypatch.setenv("AGENT_STATE_SIZE", "3072")
    monkeypatch.setenv("AGENT_HIDDEN_SIZE", "1024")
    monkeypatch.setenv("AGENT_NUM_LAYERS", "8")
    monkeypatch.setenv("AGENT_RECURRENT_SIZE", "128")
    monkeypatch.setenv("AGENT_PHASE_HEAD_COUNT", "6")
    monkeypatch.setenv("AGENT_CARD_TOKEN_DIM", "20")
    monkeypatch.setenv("AGENT_TABLEAU_TOKEN_COUNT", "8")
    monkeypatch.setenv("AGENT_HAND_TOKEN_COUNT", "64")
    monkeypatch.setenv("AGENT_OPPONENT_TOKEN_COUNT", "6")
    monkeypatch.setenv("AGENT_TRANSFORMER_EMBED_DIM", "256")
    monkeypatch.setenv("AGENT_TRANSFORMER_HEADS", "16")
    monkeypatch.setenv("AGENT_TRANSFORMER_LAYERS", "4")

    config = AgentConfig.from_env()

    assert config.state_size == 3072
    assert config.hidden_size == 1024
    assert config.num_layers == 8
    assert config.card_token_dim == 20
    assert config.hand_token_count == 64
    assert config.transformer_embed_dim == 256
    assert config.transformer_heads == 16
    assert config.transformer_layers == 4


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


def test_space_prompt_features_prefer_greenery_near_own_city(monkeypatch: pytest.MonkeyPatch) -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    import models.state_encoder as state_encoder_module

    monkeypatch.setattr(state_encoder_module, "get_rust_module", lambda required=True: _FakeRustModule())
    encoder = state_encoder_module.StateEncoder(
        state_size=512,
        card_token_dim=8,
        tableau_token_count=0,
        hand_token_count=0,
        opponent_token_count=0,
    )

    player_state = {
        "thisPlayer": {"id": "p1", "color": "red"},
        "game": {
            "spaces": [
                {"id": "ocean-a", "x": 0, "y": 0, "spaceType": "ocean", "tileType": 1},
                {"id": "own-city", "x": 1, "y": 1, "spaceType": "land", "tileType": 2, "color": "red"},
                {"id": "enemy-city", "x": 3, "y": 1, "spaceType": "land", "tileType": 2, "color": "blue"},
                {"id": "good-greenery", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad-greenery", "x": 3, "y": 0, "spaceType": "land"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": "Select space for greenery tile",
            "availableSpaces": [
                {"id": "good-greenery", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad-greenery", "x": 3, "y": 0, "spaceType": "land"},
            ],
        },
    }

    encoded = encoder.encode(player_state)
    start, slot_count, summary_start, _ = encoder._space_feature_layout()
    totals = encoded[start:start + slot_count]
    self_scores = encoded[start + slot_count:start + (2 * slot_count)]
    risk_scores = encoded[start + (3 * slot_count):start + (4 * slot_count)]
    summary = encoded[summary_start:summary_start + encoder._SPACE_SUMMARY_FEATURE_COUNT]

    assert self_scores[0] > self_scores[1]
    assert risk_scores[1] > risk_scores[0]
    assert totals[0] > totals[1]
    assert summary[0] == pytest.approx(1.0)
    assert summary[2] == pytest.approx(1.0)


def test_space_prompt_features_prefer_city_with_existing_greenery_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    import models.state_encoder as state_encoder_module

    monkeypatch.setattr(state_encoder_module, "get_rust_module", lambda required=True: _FakeRustModule())
    encoder = state_encoder_module.StateEncoder(
        state_size=512,
        card_token_dim=8,
        tableau_token_count=0,
        hand_token_count=0,
        opponent_token_count=0,
    )

    player_state = {
        "thisPlayer": {"id": "p1", "color": "red"},
        "game": {
            "spaces": [
                {"id": "green-a", "x": 0, "y": 0, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "green-b", "x": 1, "y": 1, "spaceType": "land", "tileType": 0, "color": "blue"},
                {"id": "premium-city", "x": 1, "y": 0, "spaceType": "land", "bonus": ["2 plants"]},
                {"id": "empty-city", "x": 4, "y": 0, "spaceType": "land"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": "Select space for city tile",
            "availableSpaces": [
                {"id": "premium-city", "x": 1, "y": 0, "spaceType": "land", "bonus": ["2 plants"]},
                {"id": "empty-city", "x": 4, "y": 0, "spaceType": "land"},
            ],
        },
    }

    encoded = encoder.encode(player_state)
    start, slot_count, summary_start, _ = encoder._space_feature_layout()
    totals = encoded[start:start + slot_count]
    self_scores = encoded[start + slot_count:start + (2 * slot_count)]
    summary = encoded[summary_start:summary_start + encoder._SPACE_SUMMARY_FEATURE_COUNT]

    assert self_scores[0] > self_scores[1]
    assert totals[0] > totals[1]
    assert summary[0] == pytest.approx(1.0)
    assert summary[1] == pytest.approx(1.0)


def test_dense_context_injects_awards_and_milestones_into_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    import models.state_encoder as state_encoder_module

    monkeypatch.setattr(state_encoder_module, "get_rust_module", lambda required=True: _FakeRustModule())
    encoder = state_encoder_module.StateEncoder(
        state_size=2048,
        card_token_dim=20,
        tableau_token_count=0,
        hand_token_count=0,
        opponent_token_count=0,
    )

    player_state = {
        "thisPlayer": {
            "id": "p1",
            "name": "alpha",
            "color": "red",
        },
        "players": [
            {"id": "p1", "name": "alpha", "color": "red", "tableau": []},
            {"id": "p2", "name": "beta", "color": "blue", "tableau": []},
        ],
        "game": {
            "phase": "action",
            "generation": 6,
            "milestones": [
                {
                    "name": "Gardener",
                    "playerColor": "",
                    "scores": [
                        {"playerColor": "red", "playerScore": 2},
                        {"playerColor": "blue", "playerScore": 1},
                    ],
                },
            ],
            "awards": [
                {
                    "name": "Landscaper",
                    "playerColor": "red",
                    "scores": [
                        {"playerColor": "red", "playerScore": 6},
                        {"playerColor": "blue", "playerScore": 3},
                    ],
                },
            ],
        },
        "waitingFor": {
            "type": "or",
            "options": [
                {"type": "selectCard", "title": "Standard project"},
                {"type": "or", "title": "Fund an award"},
            ],
        },
    }

    encoded = encoder.encode(player_state, turn_action_count=1)
    awards_start, awards_end = encoder._awards_feature_bounds()

    assert awards_end > awards_start
    assert float(np.abs(encoded[awards_start:awards_end]).sum()) > 0.0
    assert encoded[encoder._RUST_BASE_FEATURE_COUNT] == pytest.approx(0.0)
