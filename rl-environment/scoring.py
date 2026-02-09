"""
Shared scoring utilities used by both tournament fitness and policy training.
"""
from typing import Any


def calculate_selection_score(rank: Any, victory_points: Any, completed: Any) -> float:
    """Raw game score used for evolutionary selection."""
    try:
        rank_int = int(rank)
    except Exception:
        rank_int = 4

    try:
        vp = float(victory_points)
    except Exception:
        vp = 0.0

    ranking_points = {
        1: 100.0,
        2: 75.0,
        3: 50.0,
        4: 25.0,
    }.get(rank_int, 0.0)

    vp_bonus = vp * 0.5
    completion_bonus = 10.0 if bool(completed) else -50.0
    return ranking_points + vp_bonus + completion_bonus


def calculate_terminal_reward(rank: Any, victory_points: Any, completed: Any) -> float:
    """
    Normalized policy-learning reward derived from the same objective as selection.
    Keeps optimization stable while preserving score ordering.
    """
    raw = calculate_selection_score(rank, victory_points, completed)
    normalized = (raw - 100.0) / 50.0
    return max(-2.0, min(2.0, float(normalized)))

