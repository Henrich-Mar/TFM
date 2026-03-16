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
    from models.planner_common import PlannerConfig

    planner = PlannerConfig()

    return {
        "world_tokens": np.ones((4, planner.token_dim), dtype=np.float32),
        "world_token_types": np.asarray([1, 2, 5, 6], dtype=np.int64),
        "world_mask": np.asarray([True, True, True, True], dtype=np.bool_),
        "hand_tokens": np.ones((2, planner.token_dim), dtype=np.float32),
        "hand_mask": np.asarray([True, True], dtype=np.bool_),
        "action_tokens": np.ones((batch_action_count, planner.token_dim), dtype=np.float32),
        "action_mask": np.ones((batch_action_count,), dtype=np.bool_),
        "action_indices": np.arange(batch_action_count, dtype=np.int64),
        "action_positions": np.arange(batch_action_count, dtype=np.int64),
        "global_scalars": np.zeros((planner.global_dim,), dtype=np.float32),
    }


def test_transformer_network_output_shapes() -> None:
    _install_aiohttp_stub()
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")

    from models.agent import AgentConfig, TerraformingMarsNetwork
    from models.planner_common import pad_bundle_batch

    config = AgentConfig(
        hidden_size=256,
        planner_token_dim=64,
        planner_global_dim=16,
        planner_type_vocab_size=16,
        planner_opportunity_limit=12,
        planner_tableau_limit=8,
        planner_hand_limit=4,
        planner_opponent_limit=4,
        transformer_heads=4,
        transformer_layers=2,
        planner_aux_output_dim=280,
    )
    network = TerraformingMarsNetwork(config)
    batch = pad_bundle_batch(
        [_planner_bundle() for _ in range(5)],
        torch.device("cpu"),
        planner_config=config.planner_config(),
    )
    output = network(batch)

    assert tuple(output["policy_logits"].shape) == (5, 3)
    assert tuple(output["value"].shape) == (5, 1)
    assert tuple(output["aux_predictions"].shape) == (5, 280)


def test_agent_config_from_env_respects_planner_and_transformer_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")

    from models.agent import AgentConfig

    monkeypatch.setenv("AGENT_HIDDEN_SIZE", "1024")
    monkeypatch.setenv("AGENT_RECURRENT_SIZE", "128")
    monkeypatch.setenv("AGENT_PHASE_HEAD_COUNT", "6")
    monkeypatch.setenv("AGENT_PLANNER_TOKEN_DIM", "80")
    monkeypatch.setenv("AGENT_PLANNER_GLOBAL_DIM", "24")
    monkeypatch.setenv("AGENT_PLANNER_TYPE_VOCAB_SIZE", "24")
    monkeypatch.setenv("AGENT_PLANNER_OPPORTUNITY_LIMIT", "10")
    monkeypatch.setenv("AGENT_PLANNER_TABLEAU_LIMIT", "8")
    monkeypatch.setenv("AGENT_PLANNER_HAND_LIMIT", "64")
    monkeypatch.setenv("AGENT_PLANNER_OPPONENT_LIMIT", "6")
    monkeypatch.setenv("AGENT_TRANSFORMER_HEADS", "16")
    monkeypatch.setenv("AGENT_TRANSFORMER_LAYERS", "4")
    monkeypatch.setenv("AGENT_PLANNER_AUX_OUTPUT_DIM", "320")

    config = AgentConfig.from_env()

    assert config.hidden_size == 1024
    assert config.planner_token_dim == 80
    assert config.planner_global_dim == 24
    assert config.planner_type_vocab_size == 24
    assert config.planner_opportunity_limit == 10
    assert config.planner_hand_limit == 64
    assert config.planner_opponent_limit == 6
    assert config.transformer_heads == 16
    assert config.transformer_layers == 4
    assert config.planner_aux_output_dim == 320


def test_state_encoder_returns_planner_bundle_with_hand_tokens() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.planner_common import PlannerConfig
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(PlannerConfig(tableau_limit=8, hand_limit=4, opponent_limit=4))
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
    from models.planner_common import PlannerConfig
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(PlannerConfig(tableau_limit=8, hand_limit=4, opponent_limit=4))
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


def test_state_encoder_respects_planner_token_limits() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.planner_common import PlannerConfig
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(PlannerConfig(tableau_limit=1, hand_limit=2, opponent_limit=1))
    player_state = {
        "thisPlayer": {
            "id": "p1",
            "color": "red",
            "tableau": [
                {"name": "A", "cost": 1},
                {"name": "B", "cost": 2},
            ],
        },
        "cardsInHand": [
            {"name": "H1", "cost": 1},
            {"name": "H2", "cost": 2},
            {"name": "H3", "cost": 3},
        ],
        "players": [
            {"id": "p1", "color": "red", "tableau": []},
            {"id": "p2", "color": "blue", "tableau": []},
            {"id": "p3", "color": "green", "tableau": []},
        ],
    }

    encoded = encoder.encode(player_state)

    assert encoded["hand_tokens"].shape[0] == 2
    assert int(np.count_nonzero(encoded["world_token_types"] == 4)) == 1
    assert int(np.count_nonzero(encoded["world_token_types"] == 8)) == 1


def test_board_opportunity_rows_prefer_greenery_near_own_city() -> None:
    if "rl-environment" not in sys.path:
        sys.path.append("rl-environment")
    from models.planner_common import PlannerConfig
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
        PlannerConfig(tableau_limit=0, hand_limit=0, opponent_limit=0),
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
    from models.planner_common import PlannerConfig
    from models.state_encoder import StateEncoder

    encoder = StateEncoder(
        PlannerConfig(token_dim=80, tableau_limit=0, hand_limit=0, opponent_limit=0),
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
