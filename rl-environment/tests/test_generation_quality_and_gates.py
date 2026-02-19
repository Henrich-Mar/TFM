import sys
from types import SimpleNamespace


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from training.fitness import PromotionGateConfig, apply_promotion_gates
from training.generation_quality import annotate_generation_validity


def test_generation_quality_guard_invalid_zero_eval() -> None:
    generation_metrics = {
        "total_games_evaluated": 0,
        "frozen_pool/ran": True,
        "frozen_pool/completion_rate": 1.0,
    }
    valid, reasons = annotate_generation_validity(
        generation_metrics=generation_metrics,
        population_size=16,
        games_per_evaluation=6,
        training_pool_enabled=True,
        training_pool_games_per_agent=2,
        min_eval_completion_ratio=0.80,
        min_frozen_pool_completion_rate=0.90,
    )

    assert valid is False
    assert generation_metrics["generation/expected_games_evaluated"] == 128
    assert generation_metrics["generation/min_required_games_evaluated"] == 103
    assert generation_metrics["generation/valid_for_selection"] is False
    assert any("total_games_evaluated_below_threshold" in reason for reason in reasons)


def test_generation_quality_guard_invalid_low_completion() -> None:
    generation_metrics = {
        "total_games_evaluated": 128,
        "frozen_pool/ran": True,
        "frozen_pool/completion_rate": 0.50,
    }
    valid, reasons = annotate_generation_validity(
        generation_metrics=generation_metrics,
        population_size=16,
        games_per_evaluation=6,
        training_pool_enabled=True,
        training_pool_games_per_agent=2,
        min_eval_completion_ratio=0.80,
        min_frozen_pool_completion_rate=0.90,
    )

    assert valid is False
    assert generation_metrics["generation/valid_for_selection"] is False
    assert any("frozen_pool_completion_below_threshold" in reason for reason in reasons)


def test_generation_quality_guard_valid_case() -> None:
    generation_metrics = {
        "total_games_evaluated": 128,
        "frozen_pool/ran": True,
        "frozen_pool/completion_rate": 1.0,
    }
    valid, reasons = annotate_generation_validity(
        generation_metrics=generation_metrics,
        population_size=16,
        games_per_evaluation=6,
        training_pool_enabled=True,
        training_pool_games_per_agent=2,
        min_eval_completion_ratio=0.80,
        min_frozen_pool_completion_rate=0.90,
    )

    assert valid is True
    assert reasons == []
    assert generation_metrics["generation/valid_for_selection"] is True
    assert generation_metrics["generation/invalid_reasons"] == []


def _baseline_behavior() -> dict:
    return {
        "games_played": 10.0,
        "card_plays_per_game": 20.0,
        "standard_project_ratio": 0.15,
        "steel_spent_per_game": 12.0,
        "titanium_spent_per_game": 8.0,
        "project_payment_value_total": 300.0,
        "steel_conversion_efficiency": 0.20,
        "titanium_conversion_efficiency": 0.18,
    }


def _gate_config(max_payment_reject_count: int) -> PromotionGateConfig:
    return PromotionGateConfig(
        enabled=True,
        min_card_plays_per_game=10.0,
        max_standard_project_ratio=0.25,
        min_steel_spent_per_game=4.0,
        min_titanium_spent_per_game=2.0,
        min_steel_conversion_efficiency=0.08,
        min_titanium_conversion_efficiency=0.10,
        max_payment_reject_count=max_payment_reject_count,
        penalty_points=8.0,
        global_payment_penalty_points=2.0,
    )


def test_payment_gate_tolerance() -> None:
    population = [SimpleNamespace(id="agent-1")]
    raw_scores = [150.0]
    per_agent_behavior = {"agent-1": _baseline_behavior()}
    generation_metrics = {"payment_reject_count": 1}

    gated_scores, summary, per_agent_gate = apply_promotion_gates(
        population=population,
        raw_scores=raw_scores,
        per_agent_behavior=per_agent_behavior,
        generation_metrics=generation_metrics,
        gate_config=_gate_config(max_payment_reject_count=2),
    )

    assert summary["global_payment_gate_failed"] is False
    assert summary["pass_rate"] == 1.0
    assert per_agent_gate["agent-1"]["passed"] is True
    assert gated_scores[0] == 150.0


def test_payment_gate_failure_after_tolerance() -> None:
    population = [SimpleNamespace(id="agent-1")]
    raw_scores = [150.0]
    per_agent_behavior = {"agent-1": _baseline_behavior()}
    generation_metrics = {"payment_reject_count": 3}

    gated_scores, summary, per_agent_gate = apply_promotion_gates(
        population=population,
        raw_scores=raw_scores,
        per_agent_behavior=per_agent_behavior,
        generation_metrics=generation_metrics,
        gate_config=_gate_config(max_payment_reject_count=2),
    )

    assert summary["global_payment_gate_failed"] is True
    assert summary["pass_rate"] == 0.0
    assert per_agent_gate["agent-1"]["passed"] is False
    assert gated_scores[0] == 148.0
