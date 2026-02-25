import sys
import os

# Add the rl-environment directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.agent import RLAgent, AgentConfig
from models.action_decoder import ActionDecoder
import unittest

class TestActionLogging(unittest.TestCase):
    def setUp(self):
        # We don't want to load actual weights or do network setup
        config = AgentConfig()
        # Mocking some env vars that RLAgent.__init__ might use
        os.environ["PPO_BATCH_SIZE"] = "32"
        self.agent = RLAgent(config)
        self.agent.action_decoder = ActionDecoder()

    def test_describe_play_card(self):
        player_state = {
            "waitingFor": {
                "type": "card",
                "cards": [{"name": "Microbe Technology"}]
            }
        }
        desc = self.agent._describe_action(0, player_state)
        self.assertEqual(desc, "PLAY_CARD(Microbe Technology)")

    def test_describe_standard_project(self):
        player_state = {"waitingFor": {"type": "card"}}
        # STANDARD_PROJECT(3) -> Greenery (index 3 in action_decoder.standard_projects)
        desc = self.agent._describe_action(103, player_state)
        self.assertEqual(desc, "STANDARD_PROJECT(Greenery)")

    def test_describe_select_option(self):
        player_state = {
            "waitingFor": {
                "type": "or",
                "options": [
                    {"type": "projectCard", "title": "Play Card"},
                    {"type": "standardProject", "title": "Standard Project"}
                ]
            }
        }
        desc = self.agent._describe_action(201, player_state)
        self.assertEqual(desc, "SELECT_OPTION(Standard Project)")

    def test_describe_award(self):
        player_state = {
            "waitingFor": {
                "type": "or",
                "options": [
                    {
                        "type": "or",
                        "title": {"message": "Fund an award"},
                        "options": [
                            {"title": "Banker"},
                            {"title": "Scientist"}
                        ]
                    }
                ]
            }
        }
        # Award selection (600+)
        desc = self.agent._describe_action(600, player_state)
        self.assertEqual(desc, "AWARD(Banker)")

    def test_describe_player(self):
        player_state = {
            "waitingFor": {
                "type": "selectPlayer",
                "players": [{"name": "Player 2"}]
            }
        }
        # Player selection (600+)
        desc = self.agent._describe_action(600, player_state)
        self.assertEqual(desc, "PLAYER(Player 2)")

    def test_describe_special(self):
        player_state = {"waitingFor": {"type": "card"}}
        self.assertEqual(self.agent._describe_action(700, player_state), "CONVERT_PLANTS")
        self.assertEqual(self.agent._describe_action(701, player_state), "CONVERT_HEAT")
        self.assertEqual(self.agent._describe_action(702, player_state), "SELL_PATENTS")
        self.assertEqual(self.agent._describe_action(900, player_state), "PASS")

if __name__ == "__main__":
    unittest.main()
