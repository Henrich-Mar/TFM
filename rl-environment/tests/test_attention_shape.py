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


def _planner_bundle(batch_action_count: int = 3) -> dict:
    from models.planner_common import PLANNER_GLOBAL_DIM, PLANNER_TOKEN_DIM

    return {
        "world_tokens": np.ones((4, PLANNER_TOKEN_DIM), dtype=np.float32),
        "world_token_types": np.asarray([1, 2, 5, 6], dtype=np.int64),
        "world_mask": np.asarray([True, True, True, True], dtype=np.bool_),
        "hand_tokens": np.ones((2, PLANNER_TOKEN_DIM), dtype=np.float32),
        "hand_mask": np.asarray([True, True], dtype=np.bool_),
        "action_tokens": np.ones((batch_action_count, PLANNER_TOKEN_DIM), dtype=np.float32),
        "action_mask": np.ones((batch_action_count,), dtype=np.bool_),
        "action_indices": np.arange(batch_action_count, dtype=np.int64),
        "action_positions": np.arange(batch_action_count, dtype=np.int64),
        "global_scalars": np.zeros((PLANNER_GLOBAL_DIM,), dtype=np.float32),
    }


def test_transformer_network_output_shapes() -> None:
    _install_aiohttp_stub()
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")

    from models.agent import AgentConfig, TerraformingMarsNetwork
    from models.planner_common import pad_bundle_batch

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
        planner_aux_output_dim=280,
    )
    network = TerraformingMarsNetwork(config)
    batch = pad_bundle_batch([_planner_bundle() for _ in range(5)], torch.device("cpu"))
    output = network(batch)

    assert tuple(output["policy_logits"].shape) == (5, 3)
    assert tuple(output["value"].shape) == (5, 1)
    assert tuple(output["aux_predictions"].shape) == (5, 280)


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
    monkeypatch.setenv("AGENT_PLANNER_AUX_OUTPUT_DIM", "320")

    config = AgentConfig.from_env()

    assert config.state_size == 3072
    assert config.hidden_size == 1024
    assert config.num_layers == 8
    assert config.card_token_dim == 20
    assert config.hand_token_count == 64
    assert config.transformer_embed_dim == 256
    assert config.transformer_heads == 16
    assert config.transformer_layers == 4
    assert config.planner_aux_output_dim == 320


def test_state_encoder_returns_planner_bundle_with_hand_tokens() -> None:
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

    encoded = encoder.encode(player_state)

    assert isinstance(encoded, dict)
    assert encoded["world_tokens"].ndim == 2
    assert encoded["hand_tokens"].shape[0] == 2
    assert float(np.abs(encoded["hand_tokens"]).sum()) > 0.0
    assert 8 in set(encoded["world_token_types"].tolist())


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


def test_board_opportunity_rows_prefer_greenery_near_own_city() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
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
    }

    rows = encoder._collect_board_opportunity_rows(player_state)
    greenery_rows = [row for row in rows if row[1] > 0.5]

    assert greenery_rows
    assert greenery_rows[0][6] >= greenery_rows[-1][6]
    assert greenery_rows[0][7] >= greenery_rows[-1][7]


def test_state_encoder_emits_milestone_and_award_world_tokens() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
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

    token_types = set(encoded["world_token_types"].tolist())
    assert 5 in token_types
    assert 6 in token_types
    assert float(np.abs(encoded["world_tokens"]).sum()) > 0.0
