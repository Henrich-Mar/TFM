"""
Tests that the agent sees and can select all standard projects,
including the new Venus and Moon expansion projects.
"""
import sys

if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from models.action_decoder import ActionDecoder


def _make_standard_project_cards(names: list[str]) -> list[dict]:
    """Build card payloads for standard project prompts."""
    return [{"name": n, "isDisabled": False} for n in names]


def test_action_decoder_has_all_standard_projects() -> None:
    """Verify ActionDecoder knows all standard projects including Venus/Moon."""
    decoder = ActionDecoder()
    sp = decoder.standard_projects

    # Classic 6
    assert "Power Plant:SP" in sp
    assert "Asteroid:SP" in sp
    assert "Aquifer" in sp
    assert "Greenery" in sp
    assert "City" in sp
    assert "Colony" in sp

    # Venus expansion
    assert "Air Scrapping" in sp
    assert "Buffer Gas" in sp

    # Moon expansion
    assert any("Road Infrastructure" in n for n in sp)
    assert any("Lunar Mine" in n for n in sp)
    assert any("Lunar Habitat" in n for n in sp)

    assert len(sp) >= 11, f"Expected at least 11 standard projects, got {len(sp)}"


def test_agent_sees_extended_standard_projects_in_available_actions() -> None:
    """When the game offers extended SP list, agent gets indices for all enabled projects."""
    decoder = ActionDecoder()
    # Simulate game offering: classic 6 + Venus 2 + Moon 3 = 11 projects
    cards = _make_standard_project_cards([
        "Power Plant:SP", "Asteroid:SP", "Aquifer", "Greenery", "City", "Colony",
        "Air Scrapping", "Buffer Gas",
        "Road Infrastructure", "Lunar Mine", "Lunar Habitat",
    ])

    player_state = {
        "waitingFor": {
            "type": "card",
            "title": "Select a standard project",
            "cards": cards,
            "canPass": True,
        },
        "thisPlayer": {"megaCredits": 100, "steel": 0, "titanium": 0},
        "game": {
            "temperature": -20,
            "oceans": 0,
            "venusScaleLevel": 0,
            "moon": {"logisticsRate": 0, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    sp_base = decoder.action_types["STANDARD_PROJECT"]

    # Should see STANDARD_PROJECT(0) through STANDARD_PROJECT(10)
    sp_indices = [a - sp_base for a in available if sp_base <= a < sp_base + 20]
    assert len(sp_indices) >= 11, (
        f"Expected ≥11 standard project actions, got {len(sp_indices)}: {available}"
    )
    assert set(sp_indices) >= set(range(11)), (
        f"Missing SP indices 0-10, got: {sp_indices}"
    )


def test_agent_decodes_extended_standard_project_action() -> None:
    """Decode action index for a non-classic standard project (e.g. Air Scrapping)."""
    decoder = ActionDecoder()
    cards = _make_standard_project_cards([
        "Power Plant:SP", "Asteroid:SP", "Aquifer", "Greenery", "City", "Colony",
        "Air Scrapping", "Buffer Gas",
    ])

    player_state = {
        "waitingFor": {
            "type": "card",
            "title": "Select a standard project",
            "cards": cards,
            "canPass": True,
        },
        "thisPlayer": {"megaCredits": 100},
        "game": {"temperature": -20, "oceans": 0, "venusScaleLevel": 0},
    }

    # Action index 106 = STANDARD_PROJECT + 6 → Air Scrapping
    action = decoder.decode_action(106, player_state)
    assert action is not None
    assert action.get("type") == "card"
    chosen = action.get("cards", [])
    assert chosen == ["Air Scrapping"], f"Expected ['Air Scrapping'], got {chosen}"


def test_agent_keeps_offered_venus_projects_available() -> None:
    """The decoder should trust offered Venus projects instead of locally pruning them."""
    decoder = ActionDecoder()
    cards = _make_standard_project_cards([
        "Power Plant:SP", "Aquifer", "Greenery", "City",
        "Air Scrapping", "Buffer Gas",
    ])

    player_state = {
        "waitingFor": {
            "type": "card",
            "title": "Select a standard project",
            "cards": cards,
            "canPass": True,
        },
        "thisPlayer": {"megaCredits": 100},
        "game": {
            "temperature": -20,
            "oceans": 0,
            "venusScaleLevel": 30,  # Maxed
            "moon": {"logisticsRate": 0, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    sp_base = decoder.action_types["STANDARD_PROJECT"]
    sp_indices = [a - sp_base for a in available if sp_base <= a < sp_base + 20]

    assert set(sp_indices) >= set(range(6)), (
        f"Expected all offered Venus standard projects to remain selectable, got {sp_indices}"
    )


def test_agent_keeps_offered_moon_projects_available() -> None:
    """Moon tile projects should remain selectable if the server still offers them."""
    decoder = ActionDecoder()
    cards = _make_standard_project_cards([
        "Power Plant:SP", "Aquifer", "Greenery", "City",
        "Road Infrastructure", "Lunar Mine", "Lunar Habitat",
    ])

    player_state = {
        "waitingFor": {
            "type": "card",
            "title": "Select a standard project",
            "cards": cards,
            "canPass": True,
        },
        "thisPlayer": {"megaCredits": 100},
        "game": {
            "temperature": -20,
            "oceans": 0,
            "venusScaleLevel": 0,
            "moon": {"logisticsRate": 8, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    sp_base = decoder.action_types["STANDARD_PROJECT"]
    sp_indices = [a - sp_base for a in available if sp_base <= a < sp_base + 20]

    assert set(sp_indices) >= set(range(7)), (
        f"Expected all offered Moon standard projects to remain selectable, got {sp_indices}"
    )


def test_agent_filters_convert_heat_option_when_temperature_is_maxed() -> None:
    """Plain option prompts should not expose convert heat after temperature is maxed."""
    decoder = ActionDecoder()

    player_state = {
        "waitingFor": {
            "type": "option",
            "options": [
                {"title": {"message": "Convert heat"}},
                {"title": {"message": "Pass for this generation"}},
            ],
        },
        "game": {
            "temperature": 8,
            "oxygenLevel": 10,
            "oceans": 5,
            "venusScaleLevel": 12,
            "moon": {"logisticsRate": 0, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    select_option_base = decoder.action_types["SELECT_OPTION"]

    assert (select_option_base + 0) not in available
    assert (select_option_base + 1) in available


def test_agent_filters_standalone_temperature_option_in_or_prompt_when_maxed() -> None:
    """OR fallbacks should not surface dead temperature-only options at max temperature."""
    decoder = ActionDecoder()

    player_state = {
        "waitingFor": {
            "type": "or",
            "options": [
                {"type": "option", "title": "Increase temperature"},
                {"type": "option", "title": "Increase oxygen"},
            ],
        },
        "game": {
            "temperature": 8,
            "oxygenLevel": 10,
            "oceans": 5,
            "venusScaleLevel": 12,
            "moon": {"logisticsRate": 0, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    select_option_base = decoder.action_types["SELECT_OPTION"]

    assert (select_option_base + 0) not in available
    assert (select_option_base + 1) in available


def test_agent_filters_add_ocean_option_when_oceans_are_maxed() -> None:
    """Standalone ocean-only options should be masked after oceans are maxed out."""
    decoder = ActionDecoder()

    player_state = {
        "waitingFor": {
            "type": "option",
            "options": [
                {"title": {"message": "Add an ocean"}},
                {"title": {"message": "Increase oxygen"}},
            ],
        },
        "game": {
            "temperature": -6,
            "oxygenLevel": 10,
            "oceans": 9,
            "venusScaleLevel": 12,
            "moon": {"logisticsRate": 0, "miningRate": 0, "habitatRate": 0},
        },
    }

    available = decoder.get_available_actions(player_state)
    select_option_base = decoder.action_types["SELECT_OPTION"]

    assert (select_option_base + 0) not in available
    assert (select_option_base + 1) in available
