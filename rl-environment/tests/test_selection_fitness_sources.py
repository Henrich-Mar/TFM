import sys
from types import SimpleNamespace

import pytest


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from scoring import calculate_selection_score
from training.fitness import calculate_selection_fitness_with_diagnostics


def _population():
    return [SimpleNamespace(id="agent-a"), SimpleNamespace(id="agent-b")]


def _player(agent_id: str, rank: int, victory_points: int):
    return {
        "agent_id": agent_id,
        "rank": rank,
        "victory_points": victory_points,
        "completed": True,
    }


def _tournament_result(source, players):
    payload = {
        "games": [
            {
                "completed": True,
                "game_generation": 10,
                "players": players,
            }
        ]
    }
    if source is not None:
        payload["evaluation_source"] = source
    return payload


def test_selection_excludes_training_pool_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SELECTION_INCLUDE_TRAINING_POOL", raising=False)
    monkeypatch.setenv("SELECTION_INCLUDE_INCOMPLETE_GAMES", "0")

    population = _population()
    main_result = _tournament_result(
        "main",
        [
            _player("agent-a", rank=1, victory_points=90),
            _player("agent-b", rank=4, victory_points=70),
        ],
    )
    training_pool_result = _tournament_result(
        "training_pool",
        [
            _player("agent-a", rank=4, victory_points=70),
            _player("agent-b", rank=1, victory_points=90),
        ],
    )

    fitness_scores, diagnostics = calculate_selection_fitness_with_diagnostics(
        population=population,
        tournament_results=[main_result, training_pool_result],
    )

    expected_a = calculate_selection_score(rank=1, victory_points=90, completed=True, game_generation=10)
    expected_b = calculate_selection_score(rank=4, victory_points=70, completed=True, game_generation=10)
    assert diagnostics["selection/include_training_pool"] is False
    assert diagnostics["selection/main_games_counted"] == 1
    assert diagnostics["selection/training_pool_games_counted"] == 0
    assert fitness_scores[0] == pytest.approx(expected_a)
    assert fitness_scores[1] == pytest.approx(expected_b)


def test_selection_includes_training_pool_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELECTION_INCLUDE_TRAINING_POOL", "1")
    monkeypatch.setenv("SELECTION_INCLUDE_INCOMPLETE_GAMES", "0")

    population = _population()
    main_result = _tournament_result(
        "main",
        [
            _player("agent-a", rank=1, victory_points=90),
            _player("agent-b", rank=4, victory_points=70),
        ],
    )
    training_pool_result = _tournament_result(
        "training_pool",
        [
            _player("agent-a", rank=4, victory_points=70),
            _player("agent-b", rank=1, victory_points=90),
        ],
    )

    fitness_scores, diagnostics = calculate_selection_fitness_with_diagnostics(
        population=population,
        tournament_results=[main_result, training_pool_result],
    )

    main_a = calculate_selection_score(rank=1, victory_points=90, completed=True, game_generation=10)
    train_a = calculate_selection_score(rank=4, victory_points=70, completed=True, game_generation=10)
    main_b = calculate_selection_score(rank=4, victory_points=70, completed=True, game_generation=10)
    train_b = calculate_selection_score(rank=1, victory_points=90, completed=True, game_generation=10)
    assert diagnostics["selection/include_training_pool"] is True
    assert diagnostics["selection/main_games_counted"] == 1
    assert diagnostics["selection/training_pool_games_counted"] == 1
    assert fitness_scores[0] == pytest.approx((main_a + train_a) / 2.0)
    assert fitness_scores[1] == pytest.approx((main_b + train_b) / 2.0)


def test_selection_source_backward_compat_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SELECTION_INCLUDE_TRAINING_POOL", raising=False)
    monkeypatch.setenv("SELECTION_INCLUDE_INCOMPLETE_GAMES", "0")

    population = _population()
    missing_source_result = _tournament_result(
        None,
        [
            _player("agent-a", rank=2, victory_points=80),
            _player("agent-b", rank=3, victory_points=75),
        ],
    )
    training_pool_result = _tournament_result(
        "training_pool",
        [
            _player("agent-a", rank=4, victory_points=65),
            _player("agent-b", rank=1, victory_points=95),
        ],
    )

    fitness_scores, diagnostics = calculate_selection_fitness_with_diagnostics(
        population=population,
        tournament_results=[missing_source_result, training_pool_result],
    )

    expected_a = calculate_selection_score(rank=2, victory_points=80, completed=True, game_generation=10)
    expected_b = calculate_selection_score(rank=3, victory_points=75, completed=True, game_generation=10)
    assert "selection/include_training_pool" in diagnostics
    assert "selection/main_games_counted" in diagnostics
    assert "selection/training_pool_games_counted" in diagnostics
    assert diagnostics["selection/main_games_counted"] == 1
    assert diagnostics["selection/training_pool_games_counted"] == 0
    assert fitness_scores[0] == pytest.approx(expected_a)
    assert fitness_scores[1] == pytest.approx(expected_b)
