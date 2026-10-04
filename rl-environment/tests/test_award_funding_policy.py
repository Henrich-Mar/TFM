from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.decision_policy import AwardFundingTeacherPolicy, HeuristicTeacherPolicy  # noqa: E402


def _state(generation: int = 9, awards=None, mc: float = 40.0) -> dict:
    return {
        "game": {"generation": generation, "awards": list(awards or []), "milestones": []},
        "thisPlayer": {"color": "red", "name": "me", "megaCredits": mc},
        "players": [
            {"color": "red", "name": "me"},
            {"color": "blue", "name": "b"},
            {"color": "green", "name": "g"},
            {"color": "yellow", "name": "y"},
        ],
        "waitingFor": {"title": "Select an action", "options": []},
    }


def _descriptor(name: str) -> dict:
    return {
        "family": "fund_award",
        "award_name": name,
        "label": name,
        "action_index": 600,
        "action_position": 0,
    }


LEAD_FIRST = [{"name": "Banker", "scores": [
    {"color": "red", "score": 42}, {"color": "blue", "score": 30},
    {"color": "green", "score": 22}, {"color": "yellow", "score": 18}]}]
SECOND_PLACE = [{"name": "Banker", "scores": [
    {"color": "blue", "score": 43}, {"color": "red", "score": 42}]}]
THIRD_PLACE = [{"name": "Banker", "scores": [
    {"color": "blue", "score": 60}, {"color": "green", "score": 55},
    {"color": "red", "score": 42}]}]
LADDER_TWO = [
    {"name": "Banker", "color": "blue", "scores": [
        {"color": "red", "score": 42}, {"color": "blue", "score": 30}]},
    {"name": "Scientist", "scores": [
        {"color": "red", "score": 9}, {"color": "blue", "score": 4}]},
]
LADDER_THREE = LADDER_TWO + [
    {"name": "Thermalist", "scores": [
        {"color": "red", "score": 6}, {"color": "blue", "score": 2}]},
]


def test_funds_a_leading_first_award():
    score, reasons, fallback = AwardFundingTeacherPolicy()._score_fund_award(
        _state(awards=LEAD_FIRST), _descriptor("Banker"))
    assert not fallback
    assert score > 0.0
    # The exploit the baseline exists to model: 8 MC buys 5 VP while leading.
    assert any("cheap-award band" in reason for reason in reasons)


def test_refuses_when_in_no_paid_place():
    score, _reasons, _fallback = AwardFundingTeacherPolicy()._score_fund_award(
        _state(awards=THIRD_PLACE), _descriptor("Banker"))
    assert score < 0.0


def test_second_place_funding_is_opt_in():
    state = _state(awards=SECOND_PLACE)
    strict = AwardFundingTeacherPolicy(fund_second_place=False)._score_fund_award(
        state, _descriptor("Banker"))[0]
    loose = AwardFundingTeacherPolicy(fund_second_place=True)._score_fund_award(
        state, _descriptor("Banker"))[0]
    assert strict < 0.0
    assert loose > strict


def test_reserve_mc_prevents_starving_card_play():
    """Funding must not strip the MC needed to keep playing cards."""
    score, reasons, _ = AwardFundingTeacherPolicy()._score_fund_award(
        _state(awards=LEAD_FIRST, mc=15.0), _descriptor("Banker"))
    assert score < 0.0
    assert any("reserve" in reason for reason in reasons)


def test_cheap_award_band_rejects_the_expensive_ladder_rung():
    policy = AwardFundingTeacherPolicy(max_award_cost=8)
    score, reasons, _ = policy._score_fund_award(
        _state(awards=LADDER_TWO), _descriptor("Scientist"))
    assert score < 0.0
    assert any("cheap-award band" in reason for reason in reasons)
    # Widening the band admits it: 14 MC for a 5 VP lead is the second rung.
    wider = AwardFundingTeacherPolicy(max_award_cost=14)._score_fund_award(
        _state(awards=LADDER_TWO), _descriptor("Scientist"))[0]
    assert wider > 0.0


def test_already_funded_award_is_rejected():
    funded = [{"name": "Banker", "color": "blue", "scores": [
        {"color": "red", "score": 42}, {"color": "blue", "score": 30}]}]
    policy = AwardFundingTeacherPolicy()
    assert policy._score_fund_award(_state(awards=funded), _descriptor("Banker"))[0] < 0.0
    assert HeuristicTeacherPolicy()._score_fund_award(
        _state(awards=funded), _descriptor("Banker"))[0] < 0.0


def test_card_play_is_rewarded_for_building_a_fundable_category():
    """The exploit is two-stage: build the category, then buy the award."""
    leading = _state(generation=6, awards=[{"name": "Scientist", "scores": [
        {"color": "red", "score": 7}, {"color": "blue", "score": 3}]}], mc=40.0)
    trailing = _state(generation=6, awards=[{"name": "Scientist", "scores": [
        {"color": "blue", "score": 9}, {"color": "green", "score": 8},
        {"color": "red", "score": 2}]}], mc=40.0)
    descriptor = {"family": "play_card", "card_name": "AI Central",
                  "action_index": 1, "action_position": 0}

    base = HeuristicTeacherPolicy()._score_card(leading, descriptor)[0]
    funded_lead, lead_reasons = AwardFundingTeacherPolicy()._score_card(leading, descriptor)
    funded_trail = AwardFundingTeacherPolicy()._score_card(trailing, descriptor)[0]

    assert funded_lead > base
    assert any("builds-fundable-scientist" in reason for reason in lead_reasons)
    # No lead, no category bonus: the card is scored exactly as the base teacher.
    assert funded_trail == AwardFundingTeacherPolicy(category_goal_bonus=0.0)._score_card(
        trailing, descriptor)[0]


def test_beats_ordinary_alternatives_when_leading():
    """Funding must outrank a normal card play, or the baseline never acts."""
    policy = AwardFundingTeacherPolicy()
    state = _state(generation=9, awards=LEAD_FIRST)
    state["waitingFor"] = {"title": "Select a card to play", "cards": [
        {"name": "AI Central", "calculatedCost": 21, "victoryPoints": 3}]}
    fund = policy._score_fund_award(state, _descriptor("Banker"))[0]
    card = policy._score_card(
        state,
        {"family": "play_card", "card_name": "AI Central", "action_index": 1, "action_position": 0},
    )[0]
    assert fund > card