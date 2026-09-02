from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.decision_policy import HeuristicTeacherPolicy, RandomLegalPolicy
from models.action_decoder import ActionDecoder


def _descriptor(index: int, family: str, label: str = "") -> dict:
    return {"action_index": index, "action_position": index, "family": family, "label": label, "decoded_action": {}}


def test_teacher_prefers_claimable_milestone_over_pass() -> None:
    teacher = HeuristicTeacherPolicy(seed=7, sample=False)
    result = teacher.score_actions(
        {"thisPlayer": {"megaCredits": 20}, "game": {"generation": 5}, "waitingFor": {}},
        [_descriptor(600, "claim_milestone", "Claim Gardener"), _descriptor(900, "pass", "Pass")],
    )
    assert result.chosen_action_index == 600
    assert not result.used_fallback
    assert abs(sum(item.probability for item in result.actions) - 1.0) < 1e-8


def test_teacher_only_returns_legal_action() -> None:
    descriptors = [_descriptor(701, "convert_heat"), _descriptor(900, "pass")]
    result = HeuristicTeacherPolicy(seed=1, sample=True).score_actions(
        {"thisPlayer": {"heat": 8}, "game": {"temperature": -10}, "waitingFor": {}}, descriptors
    )
    assert result.chosen_action_index in {701, 900}


def test_random_legal_distribution_is_uniform() -> None:
    result = RandomLegalPolicy(seed=1).score_actions({}, [_descriptor(1, "other"), _descriptor(2, "other")])
    assert [item.probability for item in result.actions] == [0.5, 0.5]


@pytest.mark.parametrize(
    "family",
    [
        "play_card", "startup_plan", "card_subset", "card_prompt",
        "claim_milestone", "fund_award", "convert_plants", "convert_heat",
        "select_payment", "select_space", "standard_project", "sell_patents",
        "pass", "select_option", "select_amount",
    ],
)
def test_teacher_action_families_score_deterministically_and_legally(family: str) -> None:
    descriptors = [_descriptor(10, family, "production greenery"), _descriptor(900, "pass", "Pass")]
    state = {
        "thisPlayer": {"megaCredits": 30, "plants": 8, "heat": 8},
        "game": {"generation": 8, "oxygenLevel": 10, "temperature": -10},
        "waitingFor": {"cards": []},
    }
    first = HeuristicTeacherPolicy(seed=2, sample=False).score_actions(state, descriptors)
    second = HeuristicTeacherPolicy(seed=99, sample=False).score_actions(state, descriptors)
    assert first.chosen_action_index in {10, 900}
    assert first.chosen_action_index == second.chosen_action_index
    assert [row.score for row in first.actions] == [row.score for row in second.actions]


def test_unsupported_teacher_prompt_uses_deterministic_fallback_metric() -> None:
    teacher = HeuristicTeacherPolicy(seed=5, sample=True)
    result = teacher.score_actions({}, [_descriptor(41, "future_expansion_prompt")])
    assert result.chosen_action_index == 41
    assert result.used_fallback
    assert teacher.decisions == 1
    assert teacher.fallbacks == 1


def test_real_award_action_range_is_not_misclassified_as_card_subset() -> None:
    decoder = ActionDecoder()
    waiting_for = {
        "type": "or",
        "title": "Take one action",
        "options": [{
            "type": "or",
            "title": "Fund an award",
            "options": [{"type": "option", "title": "Landlord"}],
        }],
    }
    assert decoder._semantic_family(600, waiting_for, {"type": "option"}) == "fund_award"
