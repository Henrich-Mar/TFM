from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.decision_policy import HeuristicTeacherPolicy
from training.v4_gates import beats_humans
from training.v4_teacher_ab import teacher_ab_verdict


def _state(milestones, generation=4, awards=None):
    return {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 40},
        "game": {
            "generation": generation,
            "milestones": milestones,
            "awards": awards or [],
        },
        "waitingFor": {"cards": []},
    }


def _card(name, tags, description="", card_type="Active"):
    return {
        "name": name,
        "cost": 10,
        "victoryPoints": 1,
        "tags": list(tags),
        "description": description,
        "cardType": card_type,
    }


def _play(card):
    return {"card_name": card["name"], "family": "play_card"}


def _score(teacher, state, card):
    state = dict(state)
    state["waitingFor"] = {"cards": [card]}
    return teacher._score_card(state, _play(card))[0]


def test_building_tag_at_seven_of_eight_outscores_an_equal_card() -> None:
    milestone = {
        "name": "Builder",
        "scores": [
            {"playerColor": "red", "playerScore": 7},
            {"playerColor": "blue", "playerScore": 2},
        ],
    }
    state = _state([milestone])
    teacher = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    building = _card("Dome", ["Building"])
    plain = _card("Plain", [])
    assert _score(teacher, state, building) > _score(teacher, state, plain)


def test_claimed_builder_adds_no_reachability_bonus() -> None:
    open_track = {
        "name": "Builder",
        "scores": [{"playerColor": "red", "playerScore": 7}],
    }
    claimed = {**open_track, "playerColor": "blue"}
    card = _card("Dome", ["Building"])
    teacher = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    baseline = HeuristicTeacherPolicy(seed=1, sample=False, reachability=False)
    assert _score(teacher, _state([claimed]), card) == _score(baseline, _state([claimed]), card)


def test_three_claimed_milestones_add_no_bonus() -> None:
    taken = [
        {"name": "Mayor", "playerColor": "blue", "scores": []},
        {"name": "Gardener", "playerColor": "green", "scores": []},
        {"name": "Planner", "playerColor": "yellow", "scores": []},
        {"name": "Builder", "scores": [{"playerColor": "red", "playerScore": 7}]},
    ]
    card = _card("Dome", ["Building"])
    teacher = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    baseline = HeuristicTeacherPolicy(seed=1, sample=False, reachability=False)
    assert _score(teacher, _state(taken), card) == _score(baseline, _state(taken), card)


def test_reachability_flag_off_matches_the_unboosted_score() -> None:
    milestone = {"name": "Builder", "scores": [{"playerColor": "red", "playerScore": 7}]}
    card = _card("Dome", ["Building"])
    state = _state([milestone])
    enabled = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    disabled = HeuristicTeacherPolicy(seed=1, sample=False, reachability=False)
    assert _score(enabled, state, card) > _score(disabled, state, card)
    assert "reach-builder=" in " ".join(enabled._score_card(
        {**state, "waitingFor": {"cards": [card]}},
        _play(card),
    )[1])


def test_unreachable_terraformer_at_generation_13_adds_nothing() -> None:
    milestone = {"name": "Terraformer", "scores": [{"playerColor": "red", "playerScore": 15}]}
    card = _card("Asteroid", [], description="Raise the temperature 1 step")
    teacher = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    baseline = HeuristicTeacherPolicy(seed=1, sample=False, reachability=False)
    state = _state([milestone], generation=13)
    assert _score(teacher, state, card) == _score(baseline, state, card)


def test_funded_award_adds_nothing() -> None:
    award = {
        "name": "Scientist",
        "playerColor": "blue",
        "scores": [
            {"color": "red", "score": 4},
            {"color": "blue", "score": 3},
        ],
    }
    card = _card("Lab", ["Science"])
    teacher = HeuristicTeacherPolicy(seed=1, sample=False, reachability=True)
    baseline = HeuristicTeacherPolicy(seed=1, sample=False, reachability=False)
    assert _score(teacher, _state([], awards=[award]), card) == _score(baseline, _state([], awards=[award]), card)


def test_teacher_ab_collects_only_when_the_gap_is_clear() -> None:
    assert teacher_ab_verdict(2.20, 20) == "collect"
    assert teacher_ab_verdict(2.45, 20) == "extend"
    assert teacher_ab_verdict(2.70, 20) == "stop"
    assert teacher_ab_verdict(2.40, 60) == "collect"


def test_human_beating_requires_forty_games_and_rank_interval_below_par() -> None:
    assert beats_humans(40, 2.49) is True
    assert beats_humans(39, 2.10) is False
    assert beats_humans(40, 2.50) is False
    assert beats_humans(10, 1.5) is False
