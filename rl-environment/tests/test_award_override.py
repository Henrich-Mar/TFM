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
        "hand_weight": 0.0, "init_focus_weight": 0.0, "init_top_k": 8,
    }
    with pytest.raises(ValueError):
        parse_award_override("lead=1")


def test_hand_weight_extends_lead_with_cards_in_hand():
    state = _state(awards=[_award("Thermalist", 10, 9)])
    state["cardsInHand"] = [{"name": "Heat Card"}, {"name": "City Card"}, "Heat Card"]
    blind = AwardOverrideRule(min_lead=3)
    assert blind.choose(state, [_fund("Thermalist", 600)]) is None
    aware = AwardOverrideRule(min_lead=3, hand_weight=0.5)
    aware._card_vectors = {"Heat Card": {"stock:heat": 3.0}, "City Card": {"tiles": 1.0}}
    # lead 1 + 0.5 * (3 + 3) heat in hand = 4 >= 3
    assert aware.choose(state, [_fund("Thermalist", 600)]) is not None
    # The hand never makes a trailing award fundable.
    behind = _state(awards=[_award("Thermalist", 8, 9)])
    behind["cardsInHand"] = [{"name": "Heat Card"}] * 5
    assert aware.choose(behind, [_fund("Thermalist", 600)]) is None


def _plan(index: int, corp: str, keep: list) -> dict:
    return {
        "family": "startup_plan",
        "action_index": index,
        "decoded_action": {
            "type": "initialCards",
            "responses": [{"type": "card", "cards": [corp]}, {"type": "card", "cards": list(keep)}],
        },
    }


def _startup_state() -> dict:
    state = _state(generation=1)
    state["game"]["awards"] = [{"name": "Scientist", "scores": []}, {"name": "Banker", "scores": []}]
    state["game"]["milestones"] = [{"name": "Builder", "scores": []}]
    state["waitingFor"] = {"type": "initialCards", "options": []}
    return state


def _steering_rule(**kwargs) -> AwardOverrideRule:
    rule = AwardOverrideRule(**kwargs)
    rule._card_vectors = {
        "Corp": {},
        "Lab": {"tag:science": 2.0},
        "Lab2": {"tag:science": 2.0},
        "Filler": {},
        "Factory": {"tag:building": 1.0},
    }
    return rule


def test_startup_focus_rewards_one_committed_track():
    rule = _steering_rule()
    state = _startup_state()
    focused = rule.startup_focus(state, _plan(800, "Corp", ["Lab", "Lab2"]))
    spread = rule.startup_focus(state, _plan(801, "Corp", ["Lab", "Factory"]))
    empty = rule.startup_focus(state, _plan(802, "Corp", ["Filler"]))
    assert focused == pytest.approx(0.8)  # 4 science tags / 5
    assert spread == pytest.approx(0.4 + 0.5 * (1.0 / 7.5))
    assert empty == 0.0


def test_startup_steering_picks_focused_plan_within_policy_top_k():
    state = _startup_state()
    plans = [_plan(800, "Corp", ["Filler"]), _plan(801, "Corp", ["Lab", "Lab2"]), _plan(802, "Corp", ["Lab", "Lab2"])]
    probs = [0.6, 0.3, 0.1]
    rule = _steering_rule(init_focus_weight=2.0, init_top_k=2)
    chosen = rule.choose(state, plans, 800, position_probs=probs)
    # log(0.3) + 1.6 beats log(0.6); the third plan is outside the top 2.
    assert chosen["action_index"] == 801
    assert rule.snapshot()["startup_overridden"] == 1
    # Weight 0 leaves startup to the policy entirely.
    assert AwardOverrideRule().choose(state, plans, 800, position_probs=probs) is None
    # A tiny weight keeps the policy's favourite.
    weak = _steering_rule(init_focus_weight=0.1)
    assert weak.choose(state, plans, 800, position_probs=probs)["action_index"] == 800
    assert weak.snapshot()["startup_overridden"] == 0


def test_random_ma_names_are_steered():
    from models.award_tracks import card_features, plan_totals, track_for, track_value

    assert track_for("A. Engineer") is track_for("aengineer")
    assert track_for("V. Electrician") is not None
    assert track_for("Desert Settler") is None  # board geometry, not card-derivable
    rule = AwardOverrideRule()
    rule._card_vectors = {"Corp": {}, "Gen": {"tag:power": 2.0, "prod:energy": 2.0}, "Filler": {}}
    state = _startup_state()
    state["game"]["awards"] = [{"name": "Electrician"}, {"name": "Venuphile"}]
    state["game"]["milestones"] = [{"name": "Energizer"}]
    focus = rule.startup_focus(state, _plan(800, "Corp", ["Gen"]))
    assert focus == pytest.approx(2.0 / 5.0 + 0.5 * (2.0 / 6.0))

    features = card_features({
        "name": "Lunar-ish", "type": "automated", "tags": ["earth", "power"], "cost": 22,
        "description": "Increase your energy production 2 steps.", "requirements": [{"oceans": 3}],
    })
    assert features["tag:power"] == 1.0 and features["cost20"] == 1.0 and features["has_req"] == 1.0
    corp = card_features({"name": "Helion", "type": "corporation", "tags": ["space"],
                          "description": "You start with 3 heat production and 42 M€."})
    assert corp["prod:heat"] == 3.0
    event = card_features({"name": "Ev", "type": "event", "tags": ["space"], "cost": 5})
    assert "tag:space" not in event and event["event"] == 1.0
    totals = plan_totals({"tag:space": 3.0, "tag:earth": 1.0})
    assert totals["distinct_tags"] == 2.0 and totals["max_tag"] == 3.0
    assert track_value(track_for("Curator")[0], totals) == 3.0
