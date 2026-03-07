import sys

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from scoring import calculate_step_reward_decomposition


def _build_draft_state(prompt_cards, opponent_tableau=None, own_tableau=None):
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "megaCredits": 20,
            "steel": 0,
            "titanium": 0,
            "steelValue": 2,
            "titaniumValue": 3,
            "tableau": own_tableau or [],
            "victoryPointsBreakdown": {
                "terraforming": 0,
                "milestones": 0,
                "awards": 0,
                "city": 0,
                "greenery": 0,
                "cards": 0,
            },
        },
        "waitingFor": {
            "type": "card",
            "title": "Draft cards",
            "cards": prompt_cards,
        },
        "game": {
            "generation": 5,
            "milestones": [],
            "awards": [],
            "spaces": [],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": own_tableau or []},
            {"name": "Agent B", "color": "blue", "tableau": opponent_tableau or []},
        ],
    }


def test_hate_draft_diagnostics_are_populated_for_draft_choice() -> None:
    prompt_cards = [
        {"name": "Jovian Export", "cost": 18, "tags": ["Jovian", "Space"], "victoryPoints": 1},
        {"name": "Cheap Build", "cost": 3, "tags": ["Building"], "victoryPoints": 0},
    ]
    before_state = _build_draft_state(
        prompt_cards=prompt_cards,
        opponent_tableau=[{"name": "Opp Jovian", "tags": ["Jovian"]}],
        own_tableau=[{"name": "Own Building", "tags": ["Building"]}],
    )
    reward = calculate_step_reward_decomposition(
        before_state=before_state,
        after_state=before_state,
        action_input={"type": "card", "cards": ["Jovian Export"]},
    )
    assert reward["hate_draft_decision"] == pytest.approx(1.0)
    assert reward["hate_draft_best_keep_ev"] > 0.0
    assert reward["hate_draft_opp_overlap_mean"] >= 0.0
    assert reward["hate_draft_own_overlap_mean"] >= 0.0
    assert reward["hate_draft_deny_self_gap"] == pytest.approx(
        reward["hate_draft_opp_overlap_mean"] - reward["hate_draft_own_overlap_mean"]
    )


def test_hate_draft_low_hand_ev_threshold_flips_around_default() -> None:
    low_ev_state = _build_draft_state(
        prompt_cards=[{"name": "Unplayable", "cost": 40, "tags": [], "victoryPoints": 0}],
    )
    low_reward = calculate_step_reward_decomposition(
        before_state=low_ev_state,
        after_state=low_ev_state,
        action_input={"type": "card", "cards": ["Unplayable"]},
    )
    assert low_reward["hate_draft_decision"] == pytest.approx(1.0)
    assert low_reward["hate_draft_best_keep_ev"] < 0.35
    assert low_reward["hate_draft_low_hand_ev"] == pytest.approx(1.0)

    high_ev_state = _build_draft_state(
        prompt_cards=[{"name": "Free Keep", "cost": 0, "tags": [], "victoryPoints": 0}],
    )
    high_reward = calculate_step_reward_decomposition(
        before_state=high_ev_state,
        after_state=high_ev_state,
        action_input={"type": "card", "cards": ["Free Keep"]},
    )
    assert high_reward["hate_draft_best_keep_ev"] > 0.35
    assert high_reward["hate_draft_low_hand_ev"] == pytest.approx(0.0)


def test_hate_draft_zero_gap_signal_stays_small() -> None:
    state = _build_draft_state(
        prompt_cards=[{"name": "Neutral Card", "cost": 10, "tags": [], "victoryPoints": 0}],
        opponent_tableau=[],
        own_tableau=[],
    )
    reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "card", "cards": ["Neutral Card"]},
    )
    assert reward["hate_draft_bonus_applied"] is False
    assert 0.0 < reward["other_component"] < 0.01


def test_project_card_action_gets_stronger_cards_vp_bonus() -> None:
    state = _build_draft_state(prompt_cards=[])
    reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "projectCard", "card": "Asteroid"},
    )
    assert reward["cards_vp_component"] == pytest.approx(0.22)
