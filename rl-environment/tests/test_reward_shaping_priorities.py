import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from scoring import calculate_step_reward_decomposition


def _vp_breakdown() -> dict:
    return {
        "terraforming": 0,
        "milestones": 0,
        "awards": 0,
        "city": 0,
        "greenery": 0,
        "cards": 0,
    }


def _endgame_card_state() -> dict:
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "megaCredits": 25,
            "steel": 0,
            "titanium": 0,
            "steelValue": 2,
            "titaniumValue": 3,
            "tableau": [],
            "victoryPointsBreakdown": _vp_breakdown(),
        },
        "cardsInHand": [
            {"name": "Livestock", "cost": 13, "victoryPoints": 3, "tags": ["Animal"]},
            {"name": "Sponsors", "cost": 6, "victoryPoints": 0, "tags": ["Earth"]},
        ],
        "game": {
            "generation": 12,
            "spaces": [],
            "awards": [],
            "milestones": [],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": []},
            {"name": "Agent B", "color": "blue", "tableau": []},
        ],
    }


def _placement_state(waiting_title: str) -> dict:
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "victoryPointsBreakdown": _vp_breakdown(),
        },
        "game": {
            "generation": 9,
            "awards": [],
            "milestones": [],
            "spaces": [
                {"id": "own-city", "x": 1, "y": 1, "spaceType": "land", "tileType": 2, "color": "red"},
                {"id": "green-a", "x": 1, "y": 2, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "enemy-city", "x": 4, "y": 1, "spaceType": "land", "tileType": 2, "color": "blue"},
                {"id": "good", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad", "x": 5, "y": 0, "spaceType": "land"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": waiting_title,
            "availableSpaces": [
                {"id": "good", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad", "x": 5, "y": 0, "spaceType": "land"},
            ],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": []},
            {"name": "Agent B", "color": "blue", "tableau": []},
        ],
    }


def _city_placement_state() -> dict:
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "victoryPointsBreakdown": _vp_breakdown(),
        },
        "game": {
            "generation": 9,
            "awards": [],
            "milestones": [],
            "spaces": [
                {"id": "green-a", "x": 0, "y": 0, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "green-b", "x": 1, "y": 1, "spaceType": "land", "tileType": 0, "color": "blue"},
                {"id": "good", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad", "x": 4, "y": 0, "spaceType": "land"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": "Select space for city tile",
            "availableSpaces": [
                {"id": "good", "x": 1, "y": 0, "spaceType": "land"},
                {"id": "bad", "x": 4, "y": 0, "spaceType": "land"},
            ],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": []},
            {"name": "Agent B", "color": "blue", "tableau": []},
        ],
    }


def _future_city_preservation_state() -> dict:
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "victoryPointsBreakdown": _vp_breakdown(),
        },
        "game": {
            "generation": 10,
            "awards": [],
            "milestones": [],
            "spaces": [
                {"id": "green-a", "x": 1, "y": 0, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "green-b", "x": 2, "y": 0, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "green-c", "x": 0, "y": 1, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "own-city", "x": 0, "y": 2, "spaceType": "land", "tileType": 2, "color": "red"},
                {"id": "seal-hub", "x": 1, "y": 1, "spaceType": "land"},
                {"id": "preserve", "x": 1, "y": 2, "spaceType": "land"},
            ],
        },
        "waitingFor": {
            "type": "space",
            "title": "Select space for greenery tile",
            "availableSpaces": [
                {"id": "seal-hub", "x": 1, "y": 1, "spaceType": "land"},
                {"id": "preserve", "x": 1, "y": 2, "spaceType": "land"},
            ],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": []},
            {"name": "Agent B", "color": "blue", "tableau": []},
        ],
    }


def _with_placed_tile(state: dict, space_id: str, tile_type: int) -> dict:
    copied_spaces = []
    for space in state["game"]["spaces"]:
        if space["id"] == space_id:
            updated = dict(space)
            updated["tileType"] = tile_type
            updated["color"] = "red"
            copied_spaces.append(updated)
        else:
            copied_spaces.append(dict(space))
    return {
        **state,
        "game": {
            **state["game"],
            "spaces": copied_spaces,
        },
    }


def _draft_state() -> dict:
    prompt_cards = [
        {
            "name": "Research Outpost",
            "cost": 18,
            "victoryPoints": 1,
            "tags": ["Science", "Building"],
            "hasAction": True,
            "description": "Ongoing effect. Increase your card draw options.",
        },
        {
            "name": "Jovian Export",
            "cost": 18,
            "victoryPoints": 1,
            "tags": ["Jovian", "Space"],
        },
    ]
    return {
        "thisPlayer": {
            "name": "Agent A",
            "color": "red",
            "megaCredits": 24,
            "steel": 3,
            "titanium": 3,
            "steelValue": 2,
            "titaniumValue": 3,
            "tableau": [{"name": "Mars University", "tags": ["Science"]}],
            "victoryPointsBreakdown": _vp_breakdown(),
        },
        "waitingFor": {
            "type": "card",
            "title": "Draft cards",
            "cards": prompt_cards,
        },
        "game": {
            "generation": 6,
            "spaces": [],
            "awards": [],
            "milestones": [],
        },
        "players": [
            {"name": "Agent A", "color": "red", "tableau": [{"name": "Mars University", "tags": ["Science"]}]},
            {"name": "Agent B", "color": "blue", "tableau": [{"name": "Saturn Systems", "tags": ["Jovian"]}]},
        ],
    }


def test_endgame_prefers_affordable_vp_card_over_low_vp_play() -> None:
    state = _endgame_card_state()
    vp_reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "projectCard", "card": "Livestock"},
    )
    non_vp_reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "projectCard", "card": "Sponsors"},
    )

    assert vp_reward["cards_vp_component"] > non_vp_reward["cards_vp_component"]
    assert (vp_reward["cards_vp_component"] - non_vp_reward["cards_vp_component"]) > 0.15


def test_greenery_and_city_rewards_prefer_higher_local_board_value() -> None:
    greenery_before = _placement_state("Select space for greenery tile")
    greenery_good = calculate_step_reward_decomposition(
        before_state=greenery_before,
        after_state=_with_placed_tile(greenery_before, "good", 0),
        action_input={"type": "space", "spaceId": "good"},
    )
    greenery_bad = calculate_step_reward_decomposition(
        before_state=greenery_before,
        after_state=_with_placed_tile(greenery_before, "bad", 0),
        action_input={"type": "space", "spaceId": "bad"},
    )

    city_before = _city_placement_state()
    city_good = calculate_step_reward_decomposition(
        before_state=city_before,
        after_state=_with_placed_tile(city_before, "good", 2),
        action_input={"type": "space", "spaceId": "good"},
    )
    city_bad = calculate_step_reward_decomposition(
        before_state=city_before,
        after_state=_with_placed_tile(city_before, "bad", 2),
        action_input={"type": "space", "spaceId": "bad"},
    )

    assert greenery_good["other_component"] > greenery_bad["other_component"]
    assert city_good["other_component"] > city_bad["other_component"]


def test_greenery_reward_penalizes_destroying_best_future_city_hub() -> None:
    before = _future_city_preservation_state()
    seal_hub = calculate_step_reward_decomposition(
        before_state=before,
        after_state=_with_placed_tile(before, "seal-hub", 0),
        action_input={"type": "space", "spaceId": "seal-hub"},
    )
    preserve_hub = calculate_step_reward_decomposition(
        before_state=before,
        after_state=_with_placed_tile(before, "preserve", 0),
        action_input={"type": "space", "spaceId": "preserve"},
    )

    assert preserve_hub["city_future_component"] > seal_hub["city_future_component"]
    assert preserve_hub["city_future_component"] > 0.0
    assert seal_hub["city_future_component"] < 0.0


def test_draft_penalizes_pure_denial_when_engine_keep_is_clear() -> None:
    state = _draft_state()
    engine_reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "card", "cards": ["Research Outpost"]},
    )
    denial_reward = calculate_step_reward_decomposition(
        before_state=state,
        after_state=state,
        action_input={"type": "card", "cards": ["Jovian Export"]},
    )

    assert engine_reward["other_component"] > denial_reward["other_component"]
