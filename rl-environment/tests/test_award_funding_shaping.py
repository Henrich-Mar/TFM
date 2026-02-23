import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from scoring import calculate_step_reward_decomposition


def _build_state(
    generation: int,
    awards: list[dict],
) -> dict:
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "megaCredits": 30,
            "victoryPointsBreakdown": {
                "terraforming": 0,
                "milestones": 0,
                "awards": 0,
                "city": 0,
                "greenery": 0,
                "cards": 0,
            },
            "tableau": [],
        },
        "game": {
            "generation": generation,
            "awards": awards,
            "milestones": [],
            "temperature": -30,
            "oceans": 0,
            "venusScaleLevel": 0,
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": []},
            {"name": "Agent B", "color": "blue", "tableau": []},
        ],
    }


def test_award_funding_early_low_confidence_is_penalized() -> None:
    before_awards = [
        {
            "name": "Thermalist",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "score": 1},
                {"playerName": "Agent B", "playerColor": "blue", "score": 0},
            ],
        }
    ]
    after_awards = [
        {
            "name": "Thermalist",
            "playerName": "Agent A",
            "playerColor": "red",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "score": 1},
                {"playerName": "Agent B", "playerColor": "blue", "score": 0},
            ],
        }
    ]
    before_state = _build_state(generation=1, awards=before_awards)
    after_state = _build_state(generation=1, awards=after_awards)

    reward = calculate_step_reward_decomposition(
        before_state=before_state,
        after_state=after_state,
        action_input={"type": "option"},
    )
    assert reward["milestones_awards_component"] < 0.0


def test_award_funding_late_clear_lead_is_rewarded() -> None:
    before_awards = [
        {
            "name": "Thermalist",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "score": 16},
                {"playerName": "Agent B", "playerColor": "blue", "score": 4},
            ],
        }
    ]
    after_awards = [
        {
            "name": "Thermalist",
            "playerName": "Agent A",
            "playerColor": "red",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "score": 16},
                {"playerName": "Agent B", "playerColor": "blue", "score": 4},
            ],
        }
    ]
    before_state = _build_state(generation=12, awards=before_awards)
    after_state = _build_state(generation=12, awards=after_awards)

    reward = calculate_step_reward_decomposition(
        before_state=before_state,
        after_state=after_state,
        action_input={"type": "option"},
    )
    assert reward["milestones_awards_component"] > 0.0


def test_award_projection_accepts_player_score_field() -> None:
    before_awards = [
        {
            "name": "Thermalist",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "playerScore": 12},
                {"playerName": "Agent B", "playerColor": "blue", "playerScore": 3},
            ],
        }
    ]
    after_awards = [
        {
            "name": "Thermalist",
            "playerName": "Agent A",
            "playerColor": "red",
            "scores": [
                {"playerName": "Agent A", "playerColor": "red", "playerScore": 12},
                {"playerName": "Agent B", "playerColor": "blue", "playerScore": 3},
            ],
        }
    ]
    before_state = _build_state(generation=11, awards=before_awards)
    after_state = _build_state(generation=11, awards=after_awards)

    reward = calculate_step_reward_decomposition(
        before_state=before_state,
        after_state=after_state,
        action_input={"type": "option"},
    )
    assert reward["milestones_awards_component"] > 0.0

