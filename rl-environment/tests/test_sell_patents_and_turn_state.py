from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_decoder import ActionDecoder, build_response_for_input
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy
from game_interface import GameInstance


def _sell_patents_option() -> dict:
    return {
        "type": "card",
        "title": "Sell patents",
        "min": 1,
        "max": 7,
        "canPass": True,
        "cards": [
            {"name": "Discardable One", "cost": 4},
            {"name": "Discardable Two", "cost": 6},
        ],
    }


def test_sell_patents_is_kept_when_the_server_exposes_enabled_cards(monkeypatch) -> None:
    monkeypatch.setattr("models.action_decoder._can_afford_card", lambda _player, _card: False)
    decoder = ActionDecoder()
    sell_option = _sell_patents_option()
    player_state = {
        "thisPlayer": {"megaCredits": 20, "cardCost": 3},
        "game": {"generation": 5, "phase": "action"},
        "waitingFor": {
            "type": "or",
            "title": "Take your first action",
            "options": [
                {"type": "option", "title": "Pass for this generation"},
                sell_option,
            ],
        },
    }

    available = decoder.get_available_actions(player_state)

    assert 702 in available
    response = build_response_for_input(player_state["waitingFor"], 702, player_state)
    assert response["type"] == "or"
    assert response["index"] == 1
    assert response["response"] == {"type": "card", "cards": ["Discardable One"]}


def test_agent_filter_does_not_remove_server_legal_sell_patents() -> None:
    agent = RLAgent.__new__(RLAgent)
    agent.action_decoder = ActionDecoder()
    player_state = {
        "waitingFor": {
            "type": "or",
            "options": [
                {"type": "projectCard", "title": "Play a project card"},
                _sell_patents_option(),
                {"type": "option", "title": "Pass for this generation"},
            ],
        },
    }

    assert agent._filter_pass_actions([0, 702, 202], player_state) == [0, 702]


def test_turn_action_count_uses_server_actions_taken_this_round() -> None:
    agent = RLAgent.__new__(RLAgent)
    agent._turn_action_count_by_player = {"p-blue": 5}
    agent._last_phase_by_player = {"p-blue": "action"}

    agent._maybe_reset_turn_action_count(
        "p-blue",
        {"game": {"phase": "action"}, "thisPlayer": {"actionsTakenThisRound": 0}},
    )

    assert agent._get_turn_action_count("p-blue") == 0


def test_space_actions_receive_distinct_board_features_and_teacher_scores() -> None:
    player_state = {
        "thisPlayer": {"color": "yellow", "megaCredits": 10},
        "game": {
            "generation": 10,
            "spaces": [
                {"id": "good", "x": 1, "y": 0, "spaceType": "ocean", "bonus": [3]},
                {"id": "bad", "x": 4, "y": 0, "spaceType": "ocean", "bonus": []},
                {"id": "own-city", "x": 1, "y": 1, "spaceType": "land", "tileType": 2, "color": "yellow"},
                {"id": "enemy-city", "x": 4, "y": 1, "spaceType": "land", "tileType": 2, "color": "blue"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": "Select space for ocean from temperature increase",
            "availableSpaces": [
                {"id": "good", "x": 1, "y": 0, "spaceType": "ocean", "bonus": [3]},
                {"id": "bad", "x": 4, "y": 0, "spaceType": "ocean", "bonus": []},
            ],
        },
    }

    descriptors = ActionDecoder().get_legal_action_descriptors(player_state)
    good, bad = descriptors

    assert good["space_features"]["space_id"] == "good"
    assert good["space_features"]["own_city_adjacent"] == 1
    assert good["space_features"]["bonus_value"] > bad["space_features"]["bonus_value"]
    assert good["token_features"].tolist() != bad["token_features"].tolist()
    decision = HeuristicTeacherPolicy(seed=1, sample=False).score_actions(player_state, descriptors)
    assert decision.chosen_action_index == 300


def test_nested_convert_plants_preserves_the_selected_greenery_space() -> None:
    player_state = {
        "thisPlayer": {"color": "yellow", "plants": 8},
        "game": {
            "generation": 4,
            "spaces": [
                {"id": "top-left", "x": 0, "y": 0, "spaceType": "land"},
                {"id": "chosen-greenery", "x": 4, "y": 3, "spaceType": "land", "bonus": [3]},
            ],
        },
        "waitingFor": {
            "type": "or",
            "title": "Take your next action",
            "options": [{
                "type": "space",
                "title": "Convert 8 plants into greenery",
                "availableSpaces": [
                    {"id": "top-left", "x": 0, "y": 0, "spaceType": "land"},
                    {"id": "chosen-greenery", "x": 4, "y": 3, "spaceType": "land", "bonus": [3]},
                ],
            }],
        },
    }

    decoder = ActionDecoder()
    assert decoder.get_available_actions(player_state) == [300, 301]

    response = decoder.decode_action(301, player_state)
    assert response == {
        "type": "or",
        "index": 0,
        "response": {"type": "space", "spaceId": "chosen-greenery"},
    }

    descriptors = decoder.get_legal_action_descriptors(player_state)
    assert [row["space_features"]["space_id"] for row in descriptors] == ["top-left", "chosen-greenery"]
    selected_space = descriptors[1]["space_features"]
    assert selected_space["placement_tile_label"] == "Greenery"
    assert selected_space["current_tile_label"] == "Empty"
    assert selected_space["placement_bonuses"] == ["Card"]
    assert selected_space["placement_bonus_summary"] == "Card"


def test_option_message_templates_are_resolved_for_the_annotation_labels() -> None:
    player_state = {
        "thisPlayer": {"name": "A1_teacherv", "color": "red"},
        "players": [
            {"name": "A1_teacherv", "color": "red"},
            {"name": "A2_teacherv", "color": "blue"},
            {"name": "A3_teacherv", "color": "green"},
        ],
        "waitingFor": {
            "type": "or",
            "options": [
                {
                    "type": "option",
                    "title": {
                        "message": "Remove ${0} plants from ${1}",
                        "data": [{"type": 1, "value": "3"}, {"type": 2, "value": "blue"}],
                    },
                },
                {"type": "option", "title": "Skip removing plants"},
            ],
        },
    }

    descriptors = ActionDecoder().get_legal_action_descriptors(player_state)

    assert [row["label"] for row in descriptors] == [
        "Remove 3 plants from A2_teacherv",
        "Skip removing plants",
    ]


def test_nested_played_card_actions_keep_the_card_name_in_descriptors() -> None:
    player_state = {
        "thisPlayer": {"megaCredits": 20, "cardCost": 3},
        "waitingFor": {
            "type": "or",
            "title": "Take your action",
            "options": [{
                "type": "card",
                "title": "Perform an action from a played card",
                "selectBlueCardAction": True,
                "min": 1,
                "max": 1,
                "cards": [
                    {"name": "Electro Catapult", "cost": 17, "tags": ["building"]},
                    {"name": "Physics Complex", "cost": 12, "tags": ["science"]},
                ],
            }],
        },
    }

    descriptors = ActionDecoder().get_legal_action_descriptors(player_state)

    assert [row["card_name"] for row in descriptors] == ["Electro Catapult", "Physics Complex"]
    assert [row["label"] for row in descriptors] == [
        "Perform an action from a played card: Electro Catapult",
        "Perform an action from a played card: Physics Complex",
    ]
    assert descriptors[0]["token_features"].tolist() != descriptors[1]["token_features"].tolist()


def test_project_card_branch_wins_over_a_duplicate_sell_patents_card() -> None:
    player_state = {
        "thisPlayer": {"megaCredits": 20},
        "waitingFor": {
            "type": "or",
            "title": "Take your action",
            "options": [
                {
                    "type": "projectCard",
                    "title": "Play project card",
                    "cards": [{"name": "Greenhouses", "cost": 6}],
                },
                {
                    "type": "card",
                    "title": "Sell patents",
                    "min": 1,
                    "max": 1,
                    "cards": [{"name": "Greenhouses", "cost": 6}],
                },
            ],
        },
    }

    descriptor = ActionDecoder().get_legal_action_descriptors(player_state)[0]

    assert descriptor["family"] == "play_card"
    assert descriptor["label"] == "Play project card: Greenhouses"


def test_space_id_candidates_receive_board_features() -> None:
    player_state = {
        "thisPlayer": {"color": "red", "megaCredits": 20},
        "game": {"spaces": [{"id": "04", "x": 5, "y": 0, "spaceType": "ocean", "bonus": [3]}]},
        "waitingFor": {"type": "space", "availableSpaces": ["04"]},
    }

    descriptor = ActionDecoder().get_legal_action_descriptors(player_state)[0]

    assert descriptor["label"] == "Select space"
    assert descriptor["space_features"]["space_id"] == "04"
    assert descriptor["space_features"]["x"] == 5
    assert descriptor["space_features"]["y"] == 0


def test_card_subset_descriptors_name_the_exact_cards_to_buy(monkeypatch) -> None:
    monkeypatch.setattr("models.action_decoder._enumerate_card_selection_masks", lambda *_args: [3, 1, 2, 0])
    player_state = {
        "thisPlayer": {"megaCredits": 20, "cardCost": 3},
        "waitingFor": {
            "type": "card",
            "title": "Select card(s) to buy",
            "min": 0,
            "max": 2,
            "cards": [{"name": "Power Grid", "calculatedCost": 18}, {"name": "Trees", "calculatedCost": 13}],
        },
    }

    descriptors = ActionDecoder().get_legal_action_descriptors(player_state)

    assert {row["label"] for row in descriptors} == {
        "Buy: Power Grid + Trees", "Buy: Power Grid", "Buy: Trees", "Buy: no cards",
    }


def test_guided_annotation_targets_only_the_configured_agent(monkeypatch) -> None:
    agent = RLAgent.__new__(RLAgent)
    agent.id = "teacher-v1-seat-0"

    monkeypatch.setenv("V2_GUIDED_ANNOTATION_AGENT_ID", "teacher-v1-seat-0")
    monkeypatch.setenv("V2_GUIDED_ANNOTATION_TIMEOUT_SEC", "12.5")

    assert agent._guided_annotation_is_enabled() is True
    assert agent._guided_annotation_timeout_sec() == 12.5

    agent.id = "teacher-v1-seat-1"
    assert agent._guided_annotation_is_enabled() is False


def test_guided_annotation_zero_timeout_waits_indefinitely(monkeypatch) -> None:
    agent = RLAgent.__new__(RLAgent)
    monkeypatch.setenv("V2_GUIDED_ANNOTATION_TIMEOUT_SEC", "0")
    monkeypatch.setattr(
        "models.agent.load_snapshot_annotation",
        lambda _snapshot_id: (_ for _ in ()).throw(FileNotFoundError()),
    )

    async def check_wait() -> None:
        task = asyncio.create_task(agent._wait_for_guided_annotation("snapshot-pending"))
        await asyncio.sleep(0.05)
        assert task.done() is False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(check_wait())


def test_guided_annotation_uses_the_teacher_choice_when_the_model_choice_is_rejected() -> None:
    descriptors = [{"action_index": 300}, {"action_index": 301}, {"action_index": 306}]

    selected = RLAgent._guided_annotation_action_index(
        {"accepted_action_indices": [306], "skip": False},
        proposed_action_index=300,
        legal_descriptors=descriptors,
    )

    assert selected == 306


def test_guided_annotation_keeps_a_model_choice_that_the_teacher_accepts() -> None:
    descriptors = [{"action_index": 300}, {"action_index": 306}]

    selected = RLAgent._guided_annotation_action_index(
        {"accepted_action_indices": [300, 306], "skip": False},
        proposed_action_index=300,
        legal_descriptors=descriptors,
    )

    assert selected == 300


def test_guided_annotation_replay_uses_the_original_recorded_choice(monkeypatch) -> None:
    agent = RLAgent.__new__(RLAgent)
    agent.id = "teacher-v1-seat-0"
    game = GameInstance.__new__(GameInstance)
    game.game_id = "recreated-game"
    player_state = {
        "thisPlayer": {"name": "A1_teacherv"},
        "game": {"phase": "action", "generation": 6},
        "waitingFor": {"type": "or", "title": "Choose"},
    }
    monkeypatch.setenv("V2_GUIDED_REPLAY_SOURCE_GAME_ID", "original-game")
    monkeypatch.setattr(
        "models.agent.load_annotation_replay",
        lambda *_args: [{
            "source_snapshot_id": "source-step",
            "accepted_action_indices": [300, 301],
            "selected_action_index": 301,
            "prompt": {
                "player_name": "A1_teacherv", "phase": "action", "generation": 6,
                "prompt_type": "or", "prompt_title": "Choose", "turn_action_count": 0,
            },
        }],
    )

    selected = agent._guided_replay_action_index(game, player_state, [{"action_index": 300}, {"action_index": 301}], 0)

    assert selected == 301
    assert agent._guided_replay_cursor == 0
    agent._confirm_guided_replay_action(game, selected)
    assert agent._guided_replay_cursor == 1


def test_guided_teacher_can_invalidate_a_stale_post_input_player_view() -> None:
    game = GameInstance.__new__(GameInstance)
    game._cached_player_state = {"player-red": {"waitingFor": {"spaces": ["04"]}}}

    game.invalidate_cached_player_state("player-red")

    assert game.peek_cached_state("player-red") is None
