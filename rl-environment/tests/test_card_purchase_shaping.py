import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from scoring import calculate_step_reward_decomposition


def _state(cards, hand, mc=30):
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "megaCredits": mc,
            "cardCost": 3,
            "tableau": [],
            "cardsInHand": hand,
            "victoryPointsBreakdown": {},
        },
        "waitingFor": {
            "type": "card",
            "title": "Select card(s) to buy",
            "buttonLabel": "Buy",
            "min": 0,
            "max": len(cards),
            "cards": cards,
        },
        "game": {"generation": 5, "awards": [], "milestones": []},
        "players": [{"name": "Agent A", "color": "red", "tableau": []}],
    }


def test_buying_four_mediocre_cards_is_negative_shaping() -> None:
    cards = [{"name": f"Mediocre {idx}", "calculatedCost": 24} for idx in range(4)]
    before_state = _state(cards, [])
    after_state = _state(cards, cards, mc=18)

    reward = calculate_step_reward_decomposition(
        before_state, after_state, {"type": "card", "cards": [card["name"] for card in cards]}
    )

    assert reward["other_component"] < 0.0


def test_buying_one_high_value_card_is_positive_shaping() -> None:
    card = {"name": "Strong Science Project", "calculatedCost": 8, "victoryPoints": 3, "tags": ["Science"]}
    before_state = _state([card], [])
    after_state = _state([card], [card], mc=27)

    reward = calculate_step_reward_decomposition(
        before_state, after_state, {"type": "card", "cards": [card["name"]]}
    )

    assert reward["other_component"] > 0.0


def test_nested_or_card_purchase_is_scored_without_decoder_helpers() -> None:
    card = {"name": "Strong Science Project", "calculatedCost": 8, "victoryPoints": 3, "tags": ["Science"]}
    before_state = _state([card], [])
    before_state["waitingFor"] = {
        "type": "or",
        "options": [before_state["waitingFor"]],
    }
    after_state = _state([card], [card], mc=27)

    reward = calculate_step_reward_decomposition(
        before_state,
        after_state,
        {"type": "or", "index": 0, "response": {"type": "card", "cards": [card["name"]]}},
    )

    assert reward["other_component"] > 0.0
