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


def test_projected_zero_award_funding_is_negative_at_every_cost() -> None:
    for prior_funded_count, expected_cost in enumerate((8, 14, 20)):
        before_awards = [
            {
                "name": f"Previously Funded {idx}",
                "playerName": "Agent A",
                "playerColor": "red",
                "scores": [],
            }
            for idx in range(prior_funded_count)
        ]
        before_awards.append(
            {
                "name": "Thermalist",
                "scores": [
                    {"playerName": "Agent A", "playerColor": "red", "score": 2},
                    {"playerName": "Agent B", "playerColor": "blue", "score": 8},
                    {"playerName": "Agent C", "playerColor": "green", "score": 6},
                ],
            }
        )
        after_awards = [dict(award) for award in before_awards]
        after_awards[-1].update({"playerName": "Agent A", "playerColor": "red"})
        before_state = _build_state(generation=10, awards=before_awards)
        after_state = _build_state(generation=10, awards=after_awards)
        # Simulate the server's provisional award-VP jump that previously
        # overwhelmed the selected-award penalty.
        after_state["thisPlayer"]["victoryPointsBreakdown"]["awards"] = 5

        reward = calculate_step_reward_decomposition(before_state, after_state, {"type": "option"})

        assert expected_cost in (8, 14, 20)
        assert reward["milestones_awards_component"] < 0.0


def test_generic_award_rank_drop_is_penalized_for_any_action() -> None:
    before_state = _build_state(
        generation=10,
        awards=[
            {
                "name": "Future Award",
                "scores": [
                    {"playerName": "Agent A", "playerColor": "red", "score": 10},
                    {"playerName": "Agent B", "playerColor": "blue", "score": 8},
                ],
            }
        ],
    )
    after_state = _build_state(
        generation=10,
        awards=[
            {
                "name": "Future Award",
                "scores": [
                    {"playerName": "Agent A", "playerColor": "red", "score": 7},
                    {"playerName": "Agent B", "playerColor": "blue", "score": 8},
                ],
            }
        ],
    )

    reward = calculate_step_reward_decomposition(
        before_state=before_state,
        after_state=after_state,
        action_input={"type": "option", "title": "Spend contested resource"},
    )

    assert reward["award_rank_drop_after_action"] > 0.0
    assert reward["award_rank_drop_component"] < 0.0
    assert reward["awards_component"] == 0.0
    assert reward["milestones_awards_component"] < 0.0
