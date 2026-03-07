import sys
from pathlib import Path


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from models.requirement_planning import RequirementPlanner  # noqa: E402
from models.state_encoder import StateEncoder  # noqa: E402


def _base_player_state() -> dict:
    return {
        "thisPlayer": {
            "id": "p-red",
            "name": "Red",
            "color": "red",
            "megaCredits": 5,
            "steel": 4,
            "titanium": 0,
            "plants": 3,
            "energy": 0,
            "heat": 2,
            "megaCreditProduction": 2,
            "steelProduction": 1,
            "titaniumProduction": 0,
            "plantProduction": 0,
            "energyProduction": 0,
            "heatProduction": 1,
            "terraformRating": 23,
            "coloniesCount": 1,
            "tags": {
                "science": 2,
                "building": 1,
                "earth": 1,
                "wild": 1,
            },
            "tableau": [
                {"name": "Floater Card", "resources": 2, "resourceType": "Floater"},
                {"name": "Animal Card", "resources": 1, "resourceType": "Animal"},
            ],
            "underworldData": {
                "corruption": 2,
                "tokens": ["a", "b", "c"],
            },
            "alliedParty": {
                "partyName": "Unity",
            },
        },
        "players": [
            {
                "id": "p-red",
                "color": "red",
                "coloniesCount": 1,
                "tableau": [{"name": "Floater Card", "resources": 2, "resourceType": "Floater"}],
                "tags": {"science": 2, "wild": 1},
                "underworldData": {"corruption": 2, "tokens": ["a", "b", "c"]},
            },
            {
                "id": "p-blue",
                "color": "blue",
                "coloniesCount": 2,
                "tableau": [{"name": "Other Floater", "resources": 3, "resourceType": "Floater"}],
                "tags": {"science": 1, "wild": 2},
                "underworldData": {"corruption": 1, "tokens": ["x"]},
            },
        ],
        "game": {
            "temperature": -20,
            "oxygenLevel": 6,
            "oceans": 2,
            "venusScaleLevel": 4,
            "spaces": [
                {"id": "03", "x": 0, "y": 0, "spaceType": "land", "tileType": 2, "color": "red"},
                {"id": "04", "x": 1, "y": 0, "spaceType": "ocean", "tileType": 1},
                {"id": "05", "x": 0, "y": 1, "spaceType": "land", "tileType": 0, "color": "red"},
                {"id": "06", "x": 2, "y": 0, "spaceType": "land", "tileType": 2, "color": "blue"},
            ],
            "moon": {
                "habitatRate": 2,
                "miningRate": 1,
                "logisticsRate": 3,
                "spaces": [
                    {"id": "m01", "x": 0, "y": 0, "spaceType": "land", "tileType": 30, "color": "red"},
                    {"id": "m02", "x": 1, "y": 0, "spaceType": "land", "tileType": 29, "color": "red"},
                    {"id": "m03", "x": 2, "y": 0, "spaceType": "land", "tileType": 31, "color": "blue"},
                ],
            },
            "turmoil": {
                "ruling": "Scientists",
                "chairman": "red",
                "parties": [
                    {
                        "name": "Scientists",
                        "partyLeader": "red",
                        "delegates": [{"color": "red", "number": 2}],
                    },
                    {
                        "name": "Unity",
                        "partyLeader": "blue",
                        "delegates": [{"color": "red", "number": 1}],
                    },
                ],
            },
        },
        "waitingFor": {"type": "projectCard", "cards": []},
    }


def test_requirement_planner_global_tag_and_composite_rows() -> None:
    planner = RequirementPlanner()
    state = _base_player_state()
    plan = planner.evaluate_requirements(
        [
            {"oceans": 4, "count": 4},
            {"oxygen": 4, "count": 4, "max": True},
            {"temperature": -10, "count": -10},
            {"tag": "science", "count": 3},
        ],
        state,
    )
    assert plan["blocking_count"] == 3
    rows = {row["type"]: row for row in plan["requirement_plan"]}
    assert rows["oceans"]["remaining"] == 2
    assert rows["oxygen"]["satisfied"] is False
    assert rows["oxygen"]["remaining"] == 2
    assert rows["temperature"]["remaining_steps"] == 5
    assert rows["tag"]["remaining"] == 0


def test_requirement_planner_political_moon_underworld_and_resource_types() -> None:
    planner = RequirementPlanner()
    state = _base_player_state()
    plan = planner.evaluate_requirements(
        [
            {"party": "Scientists"},
            {"chairman": {}},
            {"partyLeader": 1},
            {"habitatRate": 2},
            {"miningTiles": 1},
            {"roadTiles": 1, "all": True},
            {"corruption": 2},
            {"undergroundTokens": 4},
            {"resourceTypes": 5},
        ],
        state,
    )
    rows = {row["type"]: row for row in plan["requirement_plan"]}
    assert rows["party"]["satisfied"] is True
    assert rows["chairman"]["satisfied"] is True
    assert rows["partyLeader"]["satisfied"] is True
    assert rows["habitatRate"]["satisfied"] is True
    assert rows["miningTiles"]["satisfied"] is True
    assert rows["roadTiles"]["satisfied"] is True
    assert rows["corruption"]["satisfied"] is True
    assert rows["undergroundTokens"]["satisfied"] is False
    assert rows["undergroundTokens"]["remaining"] == 1
    assert rows["resourceTypes"]["satisfied"] is True


def test_requirement_planner_next_to_and_advisory_mismatch_flags() -> None:
    planner = RequirementPlanner()
    state = _base_player_state()
    playable = planner.evaluate_card(
        {
            "name": "Aqueduct Systems",
            "isDisabled": False,
            "requirements": [{"cities": 1, "nextTo": True, "count": 1}, {"oceans": 4, "count": 4}],
        },
        state,
    )
    masked = planner.evaluate_card(
        {
            "name": "Ready Card",
            "isDisabled": True,
            "requirements": [{"cities": 1, "nextTo": True, "count": 1}, {"oceans": 1, "count": 1}],
        },
        state,
    )
    advisory = planner.evaluate_card(
        {
            "name": "Crash Site Cleanup",
            "isDisabled": False,
            "requirements": [{"plantsRemoved": True}],
        },
        state,
    )
    city_row = [row for row in playable["requirement_plan"] if row["type"] == "cities"][0]
    assert city_row["satisfied"] is True
    assert playable["server_override"] is True
    assert masked["masked_by_server"] is True
    assert advisory["requirement_plan"][0]["advisory_only"] is True


def test_prompt_card_ranking_prefers_satisfied_and_smaller_requirement_gaps(monkeypatch) -> None:
    encoder = StateEncoder()
    monkeypatch.setattr(encoder, "_estimate_affordability_for_card", lambda player, card, tags=None: 1.0)
    state = _base_player_state()
    state["waitingFor"] = {
        "type": "projectCard",
        "cards": [
            {"name": "Ready Science", "cost": 20, "tags": ["science"], "requirements": [{"tag": "science", "count": 3}]},
            {"name": "Need Oceans", "cost": 20, "tags": ["science"], "requirements": [{"oceans": 3, "count": 3}]},
            {"name": "Far Oceans", "cost": 20, "tags": ["science"], "requirements": [{"oceans": 7, "count": 7}]},
        ],
    }
    ranked = encoder.build_prompt_card_rankings(state)
    assert [row["name"] for row in ranked[:3]] == ["Ready Science", "Need Oceans", "Far Oceans"]

