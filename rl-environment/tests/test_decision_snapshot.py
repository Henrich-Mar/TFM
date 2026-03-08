import os
import sys
from pathlib import Path

import pytest


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from debug_decision_snapshot import (  # noqa: E402
    build_decision_snapshot,
    create_capture_request,
    list_saved_snapshots,
    load_snapshot,
    reset_capture_state,
    save_snapshot,
)


def _make_player_state(prompt_type: str, waiting_for: dict) -> dict:
    return {
        "thisPlayer": {
            "id": "player-red",
            "name": "Agent Red",
            "color": "red",
            "terraformRating": 26,
            "megaCredits": 30,
            "steel": 4,
            "titanium": 2,
            "plants": 6,
            "energy": 3,
            "heat": 8,
            "megaCreditProduction": 5,
            "steelProduction": 1,
            "titaniumProduction": 0,
            "plantProduction": 2,
            "energyProduction": 1,
            "heatProduction": 3,
            "cardsInHand": [
                {"name": "Asteroid Mining", "cost": 30, "tags": ["space"], "victoryPoints": 2},
                {"name": "Open City", "cost": 23, "tags": ["building"], "victoryPoints": 1},
            ],
            "tableau": [
                {"name": "Sponsors", "cost": 6, "tags": ["earth"], "type": "automated"},
                {"name": "Electro Catapult", "cost": 17, "tags": ["building"], "type": "active", "hasAction": True},
            ],
        },
        "players": [
            {"id": "player-red", "name": "Agent Red", "color": "red", "terraformRating": 26, "megaCredits": 30},
            {"id": "player-blue", "name": "Agent Blue", "color": "blue", "terraformRating": 23, "megaCredits": 22, "steel": 3, "titanium": 1},
        ],
        "game": {
            "generation": 8,
            "phase": "action",
            "temperature": -8,
            "oxygenLevel": 9,
            "oceans": 6,
            "venusScaleLevel": 4,
            "boardName": "Tharsis",
            "awards": [
                {
                    "name": "Banker",
                    "playerName": "Agent Blue",
                    "scores": [
                        {"playerName": "Agent Red", "playerColor": "red", "score": 5},
                        {"playerName": "Agent Blue", "playerColor": "blue", "score": 8},
                    ],
                }
            ],
            "milestones": [
                {
                    "name": "Mayor",
                    "scores": [
                        {"playerName": "Agent Red", "playerColor": "red", "score": 2},
                        {"playerName": "Agent Blue", "playerColor": "blue", "score": 1},
                    ],
                }
            ],
        },
        "waitingFor": {
            "type": prompt_type,
            "title": {"message": "Choose your move"},
            **waiting_for,
        },
    }


def _base_action_meta() -> dict:
    return {
        "phase_index": 2,
        "aux_targets": {
            "milestone_claimability": [0.0] * 70,
            "award_ev": 0.4,
            "playable_cards": 2.0,
            "steel_target": 1.0,
            "titanium_target": 0.0,
        },
        "aux_predictions": ([0.0] * 70) + [0.33, 1.5, 0.8, 0.2],
        "rare_state_weight": 1.25,
        "rare_award_funding": 0.0,
        "rare_milestone_timing": 1.0,
        "rare_draft_keep_buy": 0.0,
        "rare_high_cost_payment": 0.0,
        "payment_value_estimate": 11.0,
        "available_actions_raw": [0, 1, 100, 200],
        "available_actions_filtered": [0, 1, 100],
        "legal_actions": [0, 1, 100],
        "chosen_action_label": "PLAY_CARD(Asteroid Mining)",
        "policy_temperature": 1.0,
        "value_old": 0.42,
        "transformer_stats": {
            "enabled": True,
            "token_count": 18,
            "active_token_ratio": 0.66,
            "fusion_share": 0.28,
            "attention_context_norm": 4.1,
        },
        "policy_ranking": [
            {
                "action_index": 0,
                "label": "PLAY_CARD(Asteroid Mining)",
                "decoded_action": {"type": "card", "card": "Asteroid Mining"},
                "raw_probability": 0.41,
                "masked_probability": 0.51,
                "logit": 1.8,
                "chosen": True,
                "legal": True,
            },
            {
                "action_index": 100,
                "label": "STANDARD_PROJECT(Aquifer)",
                "decoded_action": {"type": "standardProject", "project": "Aquifer"},
                "raw_probability": 0.30,
                "masked_probability": 0.29,
                "logit": 1.1,
                "chosen": False,
                "legal": True,
            },
        ],
        "prompt_card_rankings": [
            {
                "name": "AI Central",
                "selection_score": 2.4,
                "requirements": [{"tag": "science", "count": 3}],
                "requirement_plan": [
                    {
                        "type": "tag",
                        "label": "science tags",
                        "satisfied": False,
                        "target": 3,
                        "current": 2,
                        "remaining": 1,
                        "remaining_steps": None,
                        "is_max": False,
                        "all_players": False,
                        "count": 3,
                        "next_to": False,
                        "text": "science tags >= 3, now 2, need 1",
                        "advisory_only": False,
                        "server_override": False,
                        "masked_by_server": False,
                    }
                ],
                "plan_summary": "science tags >= 3, now 2, need 1",
                "reachability_score": 0.8,
                "readiness_score": 0.66,
                "all_satisfied": False,
                "blocking_count": 1,
                "server_override": False,
                "masked_by_server": False,
            }
        ],
        "policy_top_actions": [],
    }


def test_snapshot_serialization_supports_prompt_types(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DECISION_SNAPSHOT_DIR", str(tmp_path))
    reset_capture_state()
    request = create_capture_request(note="prompt-coverage")

    prompt_states = {
        "projectCard": _make_player_state("projectCard", {"cards": [{"name": "Asteroid Mining", "cost": 30}, {"name": "Open City", "cost": 23}]}),
        "selectCard": _make_player_state("selectCard", {"cards": [{"name": "Imported Nitrogen", "cost": 23}, {"name": "AI Central", "cost": 21}]}),
        "selectSpace": _make_player_state("selectSpace", {"availableSpaces": [{"id": "M1", "name": "Moon Mine", "spaceType": "moon", "x": 1, "y": 2}, {"id": "T1", "name": "Tharsis 1", "spaceType": "land", "x": 4, "y": 5}]}),
        "selectOption": _make_player_state("selectOption", {"options": [{"type": "award", "title": {"message": "Fund Banker"}}, {"type": "milestone", "title": {"message": "Claim Mayor"}}]}),
        "selectPayment": _make_player_state("selectPayment", {"amount": 23, "min": 0, "max": 3}),
        "selectPlayer": _make_player_state("selectPlayer", {"players": [{"name": "Agent Blue", "color": "blue"}]}),
        "policy": _make_player_state("policy", {"policies": [{"name": "Mars First"}]}),
    }

    for prompt_type, player_state in prompt_states.items():
        snapshot = build_decision_snapshot(
            request=request,
            agent_id="agent-1234",
            game_id="game-1",
            game_url="http://localhost:8081/game?id=game-1",
            player_id="player-red",
            player_state=player_state,
            action_input={"type": "option"},
            action_index=0,
            action_meta=_base_action_meta(),
            sampled_from_policy=True,
            send_outcome="accepted",
            turn_action_count=1,
            state_vector=[0.1, 0.2, 0.3],
        )
        assert snapshot["prompt"]["prompt_type"] == prompt_type
        assert snapshot["state"]["hand"]
        assert snapshot["state"]["tableau"]
        assert "prompt_candidates" in snapshot["state"]
        assert "map_candidates" in snapshot["state"]["prompt_candidates"]
        assert "payment_context" in snapshot["state"]["prompt_candidates"]
        assert "prompt_card_rankings" in snapshot["state"]


def test_snapshot_save_normalizes_numpy_payloads(monkeypatch, tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    monkeypatch.setenv("DECISION_SNAPSHOT_DIR", str(tmp_path))
    reset_capture_state()
    action_meta = _base_action_meta()
    action_meta["transformer_stats"]["attention_map"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    action_meta["aux_targets"]["milestone_claimability"] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    action_meta["policy_top_actions"] = [
        {
            "action_index": np.int64(7),
            "score": np.float32(0.9),
            "weights": np.array([0.6, 0.3, 0.1], dtype=np.float32),
        }
    ]

    saved = save_snapshot(
        build_decision_snapshot(
            request=create_capture_request(note="numpy-normalization"),
            agent_id="agent-numpy",
            game_id="game-1",
            game_url="http://localhost:8081/game?id=game-1",
            player_id="player-red",
            player_state=_make_player_state(
                "projectCard",
                {"cards": [{"name": "Asteroid Mining", "cost": 30}, {"name": "Open City", "cost": 23}]},
            ),
            action_input={"type": "option"},
            action_index=0,
            action_meta=action_meta,
            sampled_from_policy=True,
            send_outcome="accepted",
            turn_action_count=1,
        )
    )

    loaded = load_snapshot(saved["snapshot_id"])
    assert loaded["diagnostics"]["transformer"]["attention_map"][0] == pytest.approx([0.1, 0.2])
    assert loaded["diagnostics"]["transformer"]["attention_map"][1] == pytest.approx([0.3, 0.4])
    assert loaded["aux"]["targets"]["milestone_claimability"] == pytest.approx([0.25, 0.5, 0.75])
    assert loaded["policy"]["top_actions"][0]["action_index"] == 7
    assert loaded["policy"]["top_actions"][0]["score"] == pytest.approx(0.9)
    assert loaded["policy"]["top_actions"][0]["weights"] == pytest.approx([0.6, 0.3, 0.1])


def test_snapshot_uses_or_project_card_options_as_hand_when_cards_in_hand_are_empty(tmp_path: Path) -> None:
    reset_capture_state()
    request = create_capture_request(note="or-project-cards")
    player_state = _make_player_state(
        "or",
        {
            "options": [
                {
                    "type": "card",
                    "title": {"message": "Perform an action from a played card"},
                    "cards": [{"name": "Electro Catapult"}],
                },
                {
                    "type": "projectCard",
                    "title": {"message": "Play project card"},
                    "cards": [
                        {"name": "Venus Governor", "cost": 4},
                        {"name": "Food Factory", "cost": 12},
                    ],
                },
            ]
        },
    )
    player_state["thisPlayer"]["cardsInHand"] = []

    action_meta = _base_action_meta()
    action_meta["prompt_card_rankings"] = []

    snapshot = build_decision_snapshot(
        request=request,
        agent_id="agent-or",
        game_id="game-or",
        game_url="http://localhost:8081/game?id=game-or",
        player_id="player-red",
        player_state=player_state,
        action_input={"type": "or", "index": 1},
        action_index=9,
        action_meta=action_meta,
        sampled_from_policy=True,
        send_outcome="accepted",
        turn_action_count=0,
    )

    assert [card["name"] for card in snapshot["state"]["hand"]] == ["Venus Governor", "Food Factory"]
    assert snapshot["state"]["prompt_candidates"]["options"][1]["title"] == "Play project card"


def test_policy_ranking_contains_chosen_action_and_matching_labels() -> None:
    torch = pytest.importorskip("torch")
    from models.agent import RLAgent  # noqa: E402

    try:
        agent = RLAgent()
    except RuntimeError as exc:
        if "rust_tfm_rl" in str(exc):
            pytest.skip("rust_tfm_rl backend not installed")
        raise
    player_state = _make_player_state(
        "projectCard",
        {"cards": [{"name": "Asteroid Mining", "cost": 30}, {"name": "Open City", "cost": 23}]},
    )
    ranking = agent._build_policy_ranking(
        player_state=player_state,
        available_actions=[0, 1],
        policy_logits=torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32),
        policy_probs=torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32),
        masked_distribution=torch.tensor([0.67, 0.33, 0.0], dtype=torch.float32),
        chosen_action_index=0,
    )
    assert any(row["chosen"] for row in ranking)
    assert ranking[0]["action_index"] == 0
    assert ranking[0]["label"] == agent._describe_action(0, player_state)
    assert ranking[0]["decoded_action"] == agent.action_decoder.decode_action(0, player_state)


def test_snapshot_routes_and_page_render(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: E402
    from api.server import app  # noqa: E402

    monkeypatch.setenv("DECISION_SNAPSHOT_DIR", str(tmp_path))
    reset_capture_state()
    saved = save_snapshot(
        build_decision_snapshot(
            request=create_capture_request(note="api-smoke"),
            agent_id="agent-xyz",
            game_id="game-9",
            game_url="http://localhost:8081/game?id=game-9",
            player_id="player-red",
            player_state=_make_player_state(
                "selectSpace",
                {"availableSpaces": [{"id": "L1", "name": "Lunar Outpost", "spaceType": "moon", "x": 2, "y": 6}]},
            ),
            action_input={"type": "space", "spaceId": "L1"},
            action_index=300,
            action_meta=_base_action_meta(),
            sampled_from_policy=True,
            send_outcome="accepted",
            turn_action_count=0,
            state_vector=None,
        )
    )

    client = TestClient(app)

    list_resp = client.get("/debug/decision-snapshots")
    assert list_resp.status_code == 200
    assert list_resp.json()["snapshots"][0]["snapshot_id"] == saved["snapshot_id"]

    item_resp = client.get(f"/debug/decision-snapshots/{saved['snapshot_id']}")
    assert item_resp.status_code == 200
    assert item_resp.json()["snapshot_id"] == saved["snapshot_id"]

    page_resp = client.get(f"/decision-explainer?snapshot_id={saved['snapshot_id']}")
    assert page_resp.status_code == 200
    assert "Agent Decision Explainer" in page_resp.text
    assert "Arm Next Snapshot" in page_resp.text

    listed = list_saved_snapshots()
    assert listed[0]["snapshot_id"] == saved["snapshot_id"]
    loaded = load_snapshot(saved["snapshot_id"])
    assert loaded["policy"]["chosen_action_label"] == "PLAY_CARD(Asteroid Mining)"
    assert loaded["state"]["prompt_card_rankings"][0]["name"] == "AI Central"
