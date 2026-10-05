import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

import pytest

from scoring import (
    AWARD_CAP_STEP_PENALTY,
    AWARD_FUND_CAP,
    AWARD_MC_COST_STEP_RATE,
    calculate_step_reward_decomposition,
)


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


def _funded_award(idx: int) -> dict:
    return {
        "name": f"Already Funded {idx}",
        "playerName": "Agent A",
        "playerColor": "red",
        "scores": [],
    }


def _unfunded_award(name: str = "Thermalist") -> dict:
    return {
        "name": name,
        "scores": [
            {"playerName": "Agent A", "playerColor": "red", "playerScore": 12},
            {"playerName": "Agent B", "playerColor": "blue", "playerScore": 3},
        ],
    }


def _fund(award: dict) -> dict:
    return dict(award, playerName="Agent A", playerColor="red")


def test_funding_is_charged_for_the_mc_it_spends() -> None:
    before = [_unfunded_award()]
    reward = calculate_step_reward_decomposition(
        _build_state(generation=11, awards=before),
        _build_state(generation=11, awards=[_fund(before[0])]),
        {"type": "option"},
    )

    # First rung of the ladder is 8 MC.
    assert reward["award_funded_mc"] == 8.0
    assert reward["award_cost_component"] == pytest.approx(-(AWARD_MC_COST_STEP_RATE * 8.0))


def test_award_mc_cost_rises_with_the_price_ladder() -> None:
    charged = []
    for prior_funded in range(3):
        before = [_funded_award(idx) for idx in range(prior_funded)] + [_unfunded_award()]
        after = [_funded_award(idx) for idx in range(prior_funded)] + [_fund(_unfunded_award())]
        reward = calculate_step_reward_decomposition(
            _build_state(generation=11, awards=before),
            _build_state(generation=11, awards=after),
            {"type": "option"},
        )
        charged.append(reward["award_funded_mc"])

    assert charged == [8.0, 14.0, 20.0]


def test_mc_cost_is_charged_even_when_funding_is_the_right_call() -> None:
    """The cost must not ride on the funding bonus weight.

    Awards pay 3-5 VP for 8/14/20 MC and the terminal reward scores VP alone, so
    funding has to carry a cost whether or not we also pay for good timing. If
    these two moved together, retuning the bonus would switch the cost off and
    the policy would drift back to funding every award in the game.
    """
    before = [_unfunded_award()]
    reward = calculate_step_reward_decomposition(
        _build_state(generation=11, awards=before),
        _build_state(generation=11, awards=[_fund(before[0])]),
        {"type": "option"},
    )

    # Leading the award late is the case the bonus exists to reward.
    assert reward["awards_component"] > 0.0
    assert reward["award_cost_component"] < 0.0


def test_award_cap_is_silent_at_the_cap_and_steep_past_it() -> None:
    cap = max(1, AWARD_FUND_CAP)

    at_cap_before = [_funded_award(idx) for idx in range(cap - 1)] + [_unfunded_award()]
    at_cap_after = [_funded_award(idx) for idx in range(cap - 1)] + [_fund(_unfunded_award())]
    at_cap = calculate_step_reward_decomposition(
        _build_state(generation=11, awards=at_cap_before),
        _build_state(generation=11, awards=at_cap_after),
        {"type": "option"},
    )
    assert at_cap["award_cap_component"] == 0.0
    assert at_cap["award_cap_exceeded"] is False

    over_before = [_funded_award(idx) for idx in range(cap)] + [_unfunded_award()]
    over_after = [_funded_award(idx) for idx in range(cap)] + [_fund(_unfunded_award())]
    over = calculate_step_reward_decomposition(
        _build_state(generation=11, awards=over_before),
        _build_state(generation=11, awards=over_after),
        {"type": "option"},
    )
    assert over["award_cap_exceeded"] is True
    assert over["award_cap_component"] <= -(AWARD_CAP_STEP_PENALTY)


def test_award_cap_applies_even_when_the_seat_leads_the_award() -> None:
    """Exceeding the budget is a budget error, not a bad read on the award."""
    cap = max(1, AWARD_FUND_CAP)
    before = [_funded_award(idx) for idx in range(cap)] + [_unfunded_award()]
    after = [_funded_award(idx) for idx in range(cap)] + [_fund(_unfunded_award())]
    reward = calculate_step_reward_decomposition(
        _build_state(generation=11, awards=before),
        _build_state(generation=11, awards=after),
        {"type": "option"},
    )

    assert reward["award_cap_exceeded"] is True
    assert reward["award_cap_component"] < 0.0


def test_non_funding_actions_are_not_charged() -> None:
    state = _build_state(generation=11, awards=[])
    reward = calculate_step_reward_decomposition(state, state, {"type": "pass"})

    assert reward["award_funded_mc"] == 0.0
    assert reward["award_cost_component"] == 0.0
    assert reward["award_cap_component"] == 0.0
