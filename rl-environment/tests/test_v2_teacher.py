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


def test_builder_reachability_and_closed_tracks() -> None:
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 30},
        "game": {"generation": 5, "milestones": [{
            "name": "Builder", "scores": [{"color": "red", "score": 7}, {"color": "blue", "score": 3}],
        }]},
        "waitingFor": {"cards": [{"name": "Equal Building", "cost": 10}, {"name": "Equal Other", "cost": 10}]},
    }
    teacher = HeuristicTeacherPolicy(sample=False)
    teacher._reachability_metadata = {
        "Equal Building": {"type": "automated"}, "Equal Other": {"type": "automated"},
    }
    teacher._card_ranker._get_card_tags = lambda name, fallback=None: {"building": name == "Equal Building"}
    building = _descriptor(0, "play_card")
    building["card_name"] = "Equal Building"
    other = _descriptor(1, "play_card")
    other["card_name"] = "Equal Other"
    assert teacher._score_card(state, building)[0] > teacher._score_card(state, other)[0]
    baseline = HeuristicTeacherPolicy(sample=False, reachability=False)
    baseline._card_ranker._get_card_tags = teacher._card_ranker._get_card_tags
    original = baseline._score_card(state, building)[0]
    state["game"]["milestones"][0]["playerColor"] = "blue"
    assert teacher._score_card(state, building)[0] == baseline._score_card(state, building)[0]
    assert baseline._score_card(state, building)[0] == original
    state["game"]["milestones"].extend([{"name": "Mayor", "color": "green"}, {"name": "Gardener", "color": "yellow"}, {"name": "Terraformer", "color": "blue"}])
    state["game"]["milestones"][0].pop("playerColor")
    assert teacher._score_card(state, building)[0] == baseline._score_card(state, building)[0]


def test_reachability_rejects_impossible_terraformer_and_funded_award() -> None:
    teacher = HeuristicTeacherPolicy(sample=False)
    state = {"thisPlayer": {"color": "red"}, "game": {"generation": 13, "milestones": [
        {"name": "Terraformer", "scores": [{"color": "red", "score": 15}]},
    ], "awards": [{"name": "Scientist", "color": "blue", "scores": [
        {"color": "red", "score": 3}, {"color": "blue", "score": 4},
    ]}]}}
    assert teacher._reachability_bonus(state, {"terraformer": 1, "scientist": 1}) == 0
    state["game"]["milestones"] = []
    assert teacher._reachability_bonus(state, {"scientist": 1}) == 0


def test_reachability_applies_to_card_subset_and_standard_project(monkeypatch) -> None:
    import scoring
    monkeypatch.setattr(scoring, "_card_quality", lambda card, player: 1.0)
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 30},
        "game": {"generation": 5, "milestones": [
            {"name": "Builder", "scores": [{"color": "red", "score": 7}]},
            {"name": "Mayor", "scores": [{"color": "red", "score": 2}]},
        ]},
        "waitingFor": {"cards": [{"name": "Building", "cost": 10}]},
    }
    enhanced = HeuristicTeacherPolicy(reachability=True)
    enhanced._reachability_metadata = {"Building": {"type": "automated"}}
    enhanced._card_ranker._get_card_tags = lambda name, fallback=None: {"building": True}
    old = HeuristicTeacherPolicy(reachability=False)
    selected = _descriptor(0, "card_subset")
    selected["decoded_action"] = {"cards": ["Building"]}
    assert enhanced._score_card_subset(state, selected)[0] > old._score_card_subset(state, selected)[0]
    city = _descriptor(1, "standard_project", "City")
    assert enhanced._score_descriptor(state, city)[0] > old._score_descriptor(state, city)[0]


def test_teacher_prefers_leading_award_over_trailing_scientist() -> None:
    teacher = HeuristicTeacherPolicy(seed=3, sample=False)
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 25, "heat": 11},
        "game": {
            "generation": 8,
            "temperature": -10,
            "awards": [
                {
                    "name": "Landlord",
                    "playerName": "A2",
                    "playerColor": "blue",
                    "scores": [],
                },
                {
                    "name": "Scientist",
                    "scores": [
                        {"playerName": "A1", "playerColor": "red", "score": 2},
                        {"playerName": "A2", "playerColor": "blue", "score": 6},
                        {"playerName": "A3", "playerColor": "green", "score": 3},
                    ],
                },
                {
                    "name": "Thermalist",
                    "scores": [
                        {"playerName": "A1", "playerColor": "red", "score": 32},
                        {"playerName": "A2", "playerColor": "blue", "score": 10},
                        {"playerName": "A3", "playerColor": "green", "score": 8},
                    ],
                },
                {
                    "name": "Miner",
                    "scores": [
                        {"playerName": "A1", "playerColor": "red", "score": 0},
                        {"playerName": "A2", "playerColor": "blue", "score": 4},
                        {"playerName": "A3", "playerColor": "green", "score": 2},
                    ],
                },
            ],
        },
        "waitingFor": {},
    }
    descriptors = [
        {"action_index": 200, "action_position": 0, "family": "select_option", "label": "Convert 8 heat into temperature", "decoded_action": {}},
        {"action_index": 600, "action_position": 1, "family": "fund_award", "label": "Scientist", "award_name": "Scientist", "decoded_action": {}},
        {"action_index": 602, "action_position": 2, "family": "fund_award", "label": "Thermalist", "award_name": "Thermalist", "decoded_action": {}},
        {"action_index": 603, "action_position": 3, "family": "fund_award", "label": "Miner", "award_name": "Miner", "decoded_action": {}},
    ]
    result = teacher.score_actions(state, descriptors)
    by_index = {row.action_index: row for row in result.actions}
    assert by_index[602].score > by_index[600].score
    assert by_index[602].score > by_index[603].score
    assert result.chosen_action_index in {200, 602}
    assert "projected-vp=5" in by_index[602].reasons
    assert "projected-vp=0" in by_index[600].reasons


def test_teacher_reads_live_award_score_color_fields() -> None:
    """TM FundedAwardModel uses {color, score}, not playerColor/playerScore."""
    teacher = HeuristicTeacherPolicy(seed=4, sample=False)
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 20},
        "game": {
            "generation": 9,
            "awards": [
                {
                    "name": "Scientist",
                    "scores": [
                        {"color": "red", "score": 5},
                        {"color": "blue", "score": 2},
                    ],
                },
                {
                    "name": "Miner",
                    "scores": [
                        {"color": "red", "score": 0},
                        {"color": "blue", "score": 4},
                    ],
                },
            ],
        },
        "waitingFor": {},
    }
    result = teacher.score_actions(
        state,
        [
            {
                "action_index": 600,
                "action_position": 0,
                "family": "fund_award",
                "label": "Scientist",
                "award_name": "Scientist",
                "decoded_action": {},
            },
            {
                "action_index": 601,
                "action_position": 1,
                "family": "fund_award",
                "label": "Miner",
                "award_name": "Miner",
                "decoded_action": {},
            },
            {"action_index": 900, "action_position": 2, "family": "pass", "label": "Pass", "decoded_action": {}},
        ],
    )
    by_index = {row.action_index: row for row in result.actions}
    assert by_index[600].score > by_index[601].score
    assert by_index[600].score > by_index[900].score
    assert result.chosen_action_index == 600
    assert "projected-vp=5" in by_index[600].reasons
    assert "projected-vp=0" in by_index[601].reasons

    teacher = HeuristicTeacherPolicy(seed=11, sample=False)
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 25, "heat": 11},
        "game": {
            "generation": 8,
            "temperature": -10,
            "awards": [
                {
                    "name": "Scientist",
                    "scores": [
                        {"playerName": "A1", "playerColor": "red", "score": 2},
                        {"playerName": "A2", "playerColor": "blue", "score": 6},
                    ],
                }
            ],
        },
        "waitingFor": {},
    }
    result = teacher.score_actions(
        state,
        [
            {"action_index": 200, "action_position": 0, "family": "select_option", "label": "Convert 8 heat into temperature", "decoded_action": {}},
            {"action_index": 600, "action_position": 1, "family": "fund_award", "label": "Scientist", "award_name": "Scientist", "decoded_action": {}},
        ],
    )
    assert result.chosen_action_index == 200
    scientist = next(row for row in result.actions if row.action_index == 600)
    assert scientist.score < 0.0


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
    if family == "claim_milestone":
        descriptors[0].update({"label": "Gardener", "milestone_name": "Gardener"})
        state["thisPlayer"].update({"name": "A1", "color": "red"})
        state["game"]["milestones"] = [{
            "name": "Gardener",
            "scores": [{"color": "red", "score": 4}, {"color": "blue", "score": 2}],
        }]
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


def test_teacher_marks_a_one_action_mask_as_forced_not_confident() -> None:
    result = HeuristicTeacherPolicy(seed=5, sample=False).score_actions(
        {}, [_descriptor(41, "other", "Resolve effect")]
    )

    assert result.is_forced
    assert result.confidence == 0.0


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


def test_milestone_leaves_use_named_650_range_and_not_parent_menu_title() -> None:
    decoder = ActionDecoder()
    waiting_for = {
        "type": "or",
        "title": "Take your next action",
        "options": [
            {
                "type": "or",
                "title": "Claim a milestone",
                "options": [
                    {"type": "option", "title": "Builder"},
                    {"type": "option", "title": "Mayor"},
                ],
            },
            {
                "type": "or",
                "title": "Fund an award",
                "options": [
                    {"type": "option", "title": "Landlord"},
                    {"type": "option", "title": "Scientist"},
                ],
            },
            {"type": "option", "title": "Pass for this generation"},
        ],
    }
    state = {"thisPlayer": {"megaCredits": 20}, "game": {"generation": 8}, "waitingFor": waiting_for}
    actions = decoder.get_available_actions(state)
    assert 650 in actions
    assert 651 in actions
    assert 600 in actions
    assert 601 in actions
    # Parent claim menu must not remain as a bare SELECT_OPTION leaf.
    claim_option_index = 200  # SELECT_OPTION + 0
    assert claim_option_index not in actions

    builder = decoder._descriptor_labels(650, waiting_for, "claim_milestone", {"type": "option"}, state)
    mayor = decoder._descriptor_labels(651, waiting_for, "claim_milestone", {"type": "option"}, state)
    landlord = decoder._descriptor_labels(600, waiting_for, "fund_award", {"type": "option"}, state)
    assert builder["milestone_name"] == "Builder"
    assert builder["label"] == "Builder"
    assert mayor["milestone_name"] == "Mayor"
    assert landlord["award_name"] == "Landlord"
    assert decoder._semantic_family(650, waiting_for, {"type": "option"}) == "claim_milestone"
    assert decoder._semantic_family(651, waiting_for, {"type": "option"}) == "claim_milestone"

    decoded = decoder.decode_action(650, state)
    assert decoded == {
        "type": "or",
        "index": 0,
        "response": {"type": "or", "index": 0, "response": {"type": "option"}},
    }
    decoded_mayor = decoder.decode_action(651, state)
    assert decoded_mayor["response"]["index"] == 1


def test_teacher_startup_prefers_keep_roi_over_cash_drain() -> None:
    waiting_for = {
        "type": "initialCards",
        "options": [
            {
                "title": "Select corporation",
                "type": "card",
                "cards": [{"name": "United Nations Mars Initiative", "tags": ["Earth"]}],
                "min": 1,
                "max": 1,
            },
            {
                "title": "Select initial cards to buy",
                "type": "card",
                "cards": [
                    {"name": "Local Heat Trapping", "calculatedCost": 1, "tags": ["Building"]},
                    {"name": "Tycho Road Network", "calculatedCost": 24, "tags": ["Building"]},
                ],
                "min": 0,
                "max": 10,
            },
        ],
    }
    state = {
        "thisPlayer": {"megaCredits": 0, "cardCost": 3},
        "game": {"generation": 1},
        "waitingFor": waiting_for,
    }
    cheap = {
        "action_index": 850,
        "action_position": 1,
        "family": "startup_plan",
        "label": "United Nations Mars Initiative | Keep: Local Heat Trapping",
        "decoded_action": {
            "type": "initialCards",
            "responses": [
                {"type": "card", "cards": ["United Nations Mars Initiative"]},
                {"type": "card", "cards": ["Local Heat Trapping"]},
            ],
        },
    }
    expensive = {
        "action_index": 851,
        "action_position": 0,
        "family": "startup_plan",
        "label": "United Nations Mars Initiative | Keep: Tycho Road Network",
        "decoded_action": {
            "type": "initialCards",
            "responses": [
                {"type": "card", "cards": ["United Nations Mars Initiative"]},
                {"type": "card", "cards": ["Tycho Road Network"]},
            ],
        },
    }
    result = HeuristicTeacherPolicy(seed=3, sample=False).score_actions(state, [expensive, cheap])
    assert result.chosen_action_index == 850
    cheap_score = next(item.score for item in result.actions if item.action_index == 850)
    expensive_score = next(item.score for item in result.actions if item.action_index == 851)
    assert cheap_score > expensive_score
