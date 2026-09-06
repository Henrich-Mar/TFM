import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from models.action_decoder import ActionDecoder
from models.agent import RLAgent


def _endgame_state() -> dict:
    return {
        "thisPlayer": {"megaCredits": 50},
        "game": {"temperature": 8, "oceans": 9, "oxygenLevel": 10},
        "waitingFor": {
            "type": "or",
            "options": [
                {
                    "type": "card",
                    "title": "Standard project",
                    "cards": [
                        {"name": "Asteroid:SP"},
                        {"name": "Aquifer"},
                    ],
                },
                {"type": "pass", "title": "Pass for this generation"},
            ],
        },
    }


def test_maxed_temperature_and_oceans_standard_projects_are_masked() -> None:
    decoder = ActionDecoder()
    actions = decoder.get_available_actions(_endgame_state())

    assert 100 not in actions
    assert 101 not in actions
    assert 201 in actions


def test_server_legal_pass_is_not_removed_when_other_actions_exist() -> None:
    agent = RLAgent.__new__(RLAgent)
    agent.action_decoder = ActionDecoder()
    state = _endgame_state()
    actions = agent.action_decoder.get_available_actions(state)

    assert agent._filter_pass_actions(actions, state) == actions
