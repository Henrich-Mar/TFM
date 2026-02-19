"""
Generation-quality guard helpers for selection/evolution safety.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def expected_games_evaluated_per_generation(
    population_size: int,
    games_per_evaluation: int,
    training_pool_enabled: bool,
    training_pool_games_per_agent: int,
) -> int:
    games_per_agent = max(0, int(games_per_evaluation))
    if training_pool_enabled:
        games_per_agent += max(0, int(training_pool_games_per_agent))
    return max(0, int(population_size)) * games_per_agent


def annotate_generation_validity(
    generation_metrics: Dict[str, Any],
    population_size: int,
    games_per_evaluation: int,
    training_pool_enabled: bool,
    training_pool_games_per_agent: int,
    min_eval_completion_ratio: float,
    min_frozen_pool_completion_rate: float,
) -> Tuple[bool, List[str]]:
    expected_games = expected_games_evaluated_per_generation(
        population_size=population_size,
        games_per_evaluation=games_per_evaluation,
        training_pool_enabled=training_pool_enabled,
        training_pool_games_per_agent=training_pool_games_per_agent,
    )
    bounded_eval_ratio = min(1.0, max(0.0, float(min_eval_completion_ratio)))
    bounded_frozen_ratio = min(1.0, max(0.0, float(min_frozen_pool_completion_rate)))
    min_required_games = int(math.ceil(float(expected_games) * bounded_eval_ratio))
    observed_games = max(0, int(generation_metrics.get("total_games_evaluated", 0) or 0))

    invalid_reasons: List[str] = []
    if observed_games < min_required_games:
        invalid_reasons.append(
            f"total_games_evaluated_below_threshold:{observed_games}<{min_required_games}"
        )

    frozen_pool_ran = bool(generation_metrics.get("frozen_pool/ran", False))
    frozen_pool_completion_rate = float(generation_metrics.get("frozen_pool/completion_rate", 1.0) or 0.0)
    if training_pool_enabled and frozen_pool_ran and frozen_pool_completion_rate < bounded_frozen_ratio:
        invalid_reasons.append(
            f"frozen_pool_completion_below_threshold:{frozen_pool_completion_rate:.3f}<{bounded_frozen_ratio:.3f}"
        )

    is_valid = len(invalid_reasons) == 0
    generation_metrics["generation/expected_games_evaluated"] = int(expected_games)
    generation_metrics["generation/min_required_games_evaluated"] = int(min_required_games)
    generation_metrics["generation/valid_for_selection"] = bool(is_valid)
    generation_metrics["generation/invalid_reasons"] = list(invalid_reasons)
    return is_valid, invalid_reasons
