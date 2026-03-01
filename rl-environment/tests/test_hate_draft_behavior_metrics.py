import sys
from types import SimpleNamespace

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from training.fitness import compute_generation_behavior_metrics


def _agent(agent_id: str):
    return SimpleNamespace(id=agent_id)


def _snapshot(games_played: float = 0.0):
    return {
        "games_played": float(games_played),
        "card_play_actions": 0.0,
        "standard_project_actions": 0.0,
        "sell_patents_actions": 0.0,
        "steel_spent": 0.0,
        "titanium_spent": 0.0,
        "project_payment_value_total": 0.0,
        "metal_payment_value_total": 0.0,
        "steel_payment_value_total": 0.0,
        "titanium_payment_value_total": 0.0,
        "hate_draft_picks": 0.0,
        "draft_decisions_total": 0.0,
        "draft_decisions_low_hand_ev": 0.0,
        "hate_draft_picks_low_hand_ev": 0.0,
        "milestone_snipes": 0.0,
        "award_snipes": 0.0,
        "standard_project_counts": {},
    }


def test_generation_hate_draft_rates_and_correlations() -> None:
    population = [_agent("agent-a"), _agent("agent-b")]
    before = {"agent-a": _snapshot(0), "agent-b": _snapshot(0)}
    after = {"agent-a": _snapshot(2), "agent-b": _snapshot(2)}
    tournament_results = [
        {
            "games": [
                {
                    "completed": True,
                    "players": [
                        {
                            "agent_id": "agent-a",
                            "completed": True,
                            "rank": 1,
                            "victory_points": 100,
                            "draft_decisions": 10,
                            "draft_decisions_low_hand_ev": 5,
                            "hate_draft_picks": 8,
                            "hate_draft_picks_low_hand_ev": 4,
                        },
                        {
                            "agent_id": "agent-b",
                            "completed": True,
                            "rank": 4,
                            "victory_points": 70,
                            "draft_decisions": 10,
                            "draft_decisions_low_hand_ev": 5,
                            "hate_draft_picks": 2,
                            "hate_draft_picks_low_hand_ev": 1,
                        },
                    ],
                },
                {
                    "completed": True,
                    "players": [
                        {
                            "agent_id": "agent-a",
                            "completed": True,
                            "rank": 1,
                            "victory_points": 95,
                            "draft_decisions": 10,
                            "draft_decisions_low_hand_ev": 4,
                            "hate_draft_picks": 7,
                            "hate_draft_picks_low_hand_ev": 3,
                        },
                        {
                            "agent_id": "agent-b",
                            "completed": True,
                            "rank": 3,
                            "victory_points": 75,
                            "draft_decisions": 10,
                            "draft_decisions_low_hand_ev": 4,
                            "hate_draft_picks": 3,
                            "hate_draft_picks_low_hand_ev": 1,
                        },
                    ],
                },
            ]
        }
    ]

    generation_metrics, per_agent = compute_generation_behavior_metrics(
        population=population,
        before=before,
        after=after,
        payment_reject_before=0,
        payment_reject_after=0,
        current_generation=12,
        game_cluster=SimpleNamespace(input_reject_count=0, payment_reject_count=0),
        tournament_results=tournament_results,
    )

    assert generation_metrics["draft_decisions_total"] == pytest.approx(40.0)
    assert generation_metrics["draft_decisions_low_hand_ev"] == pytest.approx(18.0)
    assert generation_metrics["hate_draft_picks"] == pytest.approx(20.0)
    assert generation_metrics["hate_draft_picks_low_hand_ev"] == pytest.approx(9.0)
    assert generation_metrics["hate_draft_rate"] == pytest.approx(0.5)
    assert generation_metrics["hate_draft_rate_low_hand_ev"] == pytest.approx(0.5)
    assert generation_metrics["hate_draft_corr_sample_count"] == 4
    assert generation_metrics["hate_draft_corr_valid"] is True
    assert generation_metrics["hate_draft_rate_vs_vp_corr"] > 0.0
    assert generation_metrics["hate_draft_rate_vs_rank_corr"] < 0.0
    assert generation_metrics["hate_draft_rate_vs_win_corr"] > 0.0
    assert per_agent["agent-a"]["hate_draft_rate"] > per_agent["agent-b"]["hate_draft_rate"]


def test_generation_hate_draft_metrics_handle_missing_or_zero_denominators() -> None:
    population = [_agent("agent-a")]
    before = {"agent-a": _snapshot(0)}
    after = {"agent-a": _snapshot(1)}
    tournament_results = [
        {
            "games": [
                {
                    "completed": True,
                    "players": [
                        {
                            "agent_id": "agent-a",
                            "completed": True,
                            "rank": 2,
                            "victory_points": 80,
                        }
                    ],
                }
            ]
        }
    ]

    generation_metrics, per_agent = compute_generation_behavior_metrics(
        population=population,
        before=before,
        after=after,
        payment_reject_before=0,
        payment_reject_after=0,
        current_generation=12,
        game_cluster=SimpleNamespace(input_reject_count=0, payment_reject_count=0),
        tournament_results=tournament_results,
    )

    assert generation_metrics["draft_decisions_total"] == pytest.approx(0.0)
    assert generation_metrics["hate_draft_rate"] == pytest.approx(0.0)
    assert generation_metrics["hate_draft_rate_low_hand_ev"] == pytest.approx(0.0)
    assert generation_metrics["hate_draft_corr_sample_count"] == 0
    assert generation_metrics["hate_draft_corr_valid"] is False
    assert per_agent["agent-a"]["draft_decisions_total"] == pytest.approx(0.0)
    assert per_agent["agent-a"]["hate_draft_rate"] == pytest.approx(0.0)
