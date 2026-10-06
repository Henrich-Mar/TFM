from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.award_override import AwardOverrideRule, parse_award_override  # noqa: E402


def _state(generation: int = 6, awards=None, mc: float = 30.0) -> dict:
    return {
        "game": {"generation": generation, "awards": list(awards or []), "milestones": []},
        "thisPlayer": {"color": "red", "name": "me", "megaCredits": mc},
        "players": [
            {"color": "red", "name": "me"},
            {"color": "blue", "name": "b"},
            {"color": "green", "name": "g"},
            {"color": "yellow", "name": "y"},
        ],
    }


def _fund(name: str, index: int) -> dict:
    return {
        "family": "fund_award",
        "award_name": name,
        "label": name,
        "action_index": index,
        "decoded_action": {"type": "or", "index": 1, "response": {"type": "or", "index": index - 600}},
    }


PLAY = {"family": "play_card", "action_index": 10, "decoded_action": {"type": "card"}}


def _award(name: str, own: float, opp: float, funder: str = "") -> dict:
    row = {"name": name, "scores": [{"color": "red", "score": own}, {"color": "blue", "score": opp}]}
    if funder:
        row["color"] = funder
    return row


def test_fires_on_leading_award_and_picks_largest_lead():
    awards = [_award("Banker", 10, 8), _award("Miner", 9, 3)]
    rule = AwardOverrideRule(min_lead=1)
    chosen = rule.choose(_state(awards=awards), [PLAY, _fund("Banker", 600), _fund("Miner", 601)], 10)
    assert chosen is not None and chosen["award_name"] == "Miner"
    snap = rule.snapshot()
    assert snap["fired"] == 1 and snap["fired_policy_already_chose_it"] == 0


def test_defers_without_fund_option_or_lead():
    rule = AwardOverrideRule(min_lead=2)
    assert rule.choose(_state(awards=[_award("Banker", 10, 9)]), [PLAY, _fund("Banker", 600)]) is None
    assert rule.choose(_state(awards=[_award("Banker", 10, 2)]), [PLAY]) is None
    assert rule.snapshot()["decisions_with_fund_option"] == 1


@pytest.mark.parametrize(
    "kwargs,state_kwargs",
    [
        ({"min_generation": 7}, {"generation": 6}),
        ({"reserve_mc": 5}, {"mc": 12}),
        ({}, {"mc": 7}),
    ],
)
def test_respects_generation_and_money(kwargs, state_kwargs):
    rule = AwardOverrideRule(min_lead=1, **kwargs)
    state = _state(awards=[_award("Banker", 10, 2)], **state_kwargs)
    assert rule.choose(state, [_fund("Banker", 600)]) is None


def test_cost_ladder_and_own_award_cap():
    awards = [_award("Banker", 10, 2, funder="blue"), _award("Miner", 9, 3)]
    # One award already funded: next costs 14, above the default 8 band.
    assert AwardOverrideRule().choose(_state(awards=awards), [_fund("Miner", 601)]) is None
    assert AwardOverrideRule(max_cost=14).choose(_state(awards=awards), [_fund("Miner", 601)]) is not None
    own = [_award("Banker", 10, 2, funder="red"), _award("Miner", 9, 3)]
    assert AwardOverrideRule(max_cost=14, max_own_awards=1).choose(_state(awards=own), [_fund("Miner", 601)]) is None
    assert AwardOverrideRule(max_cost=14, max_own_awards=2).choose(_state(awards=own), [_fund("Miner", 601)]) is not None


def test_parse_spec():
    assert parse_award_override(None) is None
    assert parse_award_override("off") is None
    assert parse_award_override("default") == AwardOverrideRule()
    rule = parse_award_override("min_lead=0, min_generation=5,max_cost=14,reserve_mc=3,max_own_awards=2")
    assert rule.config() == {
        "min_lead": 0.0, "min_generation": 5, "max_cost": 14.0, "reserve_mc": 3.0, "max_own_awards": 2,
        "hand_weight": 0.0,
    }
    with pytest.raises(ValueError):
        parse_award_override("lead=1")


class _StubTeacher:
    """Card track deltas keyed by card name, without loading card metadata."""

    DELTAS = {"Heat Card": {"heat": 3.0}, "City Card": {"tiles": 1.0}}

    def _card_track_delta(self, name, card, *, include_planner):
        return dict(self.DELTAS.get(name, {}))


def test_hand_weight_extends_lead_with_cards_in_hand():
    state = _state(awards=[_award("Thermalist", 10, 9)])
    state["cardsInHand"] = [{"name": "Heat Card"}, {"name": "City Card"}, "Heat Card"]
    blind = AwardOverrideRule(min_lead=3)
    assert blind.choose(state, [_fund("Thermalist", 600)]) is None
    aware = AwardOverrideRule(min_lead=3, hand_weight=0.5)
    aware._teacher = _StubTeacher()
    # lead 1 + 0.5 * (3 + 3) heat in hand = 4 >= 3
    assert aware.choose(state, [_fund("Thermalist", 600)]) is not None
    # The hand never makes a trailing award fundable.
    behind = _state(awards=[_award("Thermalist", 8, 9)])
    behind["cardsInHand"] = [{"name": "Heat Card"}] * 5
    assert aware.choose(behind, [_fund("Thermalist", 600)]) is None
