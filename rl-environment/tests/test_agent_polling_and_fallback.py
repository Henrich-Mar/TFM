import asyncio
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add the rl-environment directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.agent import AgentConfig, RLAgent


class _GameStub:
    def __init__(self):
        self.sent_actions = []

    async def send_player_input(self, _player_id, action_input):
        self.sent_actions.append(action_input)
        # Reject policy action, accept fallback action.
        return len(self.sent_actions) >= 2

    def get_public_game_url(self):
        return "http://localhost:8081/game?id=test"

    def get_public_player_api_url(self, _player_id):
        return "http://localhost:8081/api/player?id=test"

    def get_internal_player_api_url(self, _player_id):
        return "http://tm-server-1:8080/api/player?id=test"


class _DecoderStub:
    action_types = {"PASS": 900}

    def get_available_actions(self, _player_state):
        raise AssertionError("get_available_actions should not be called on fallback when raw actions are cached")

    def decode_action(self, action_index, _player_state):
        return {"type": "selectOption", "index": int(action_index)}

    def _create_pass_action(self):
        return {"type": "pass"}


class TestAgentPollingAndFallback(unittest.TestCase):
    def setUp(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def tearDown(self):
        asyncio.set_event_loop(None)
        self._loop.close()

    def test_poll_interval_switches_between_idle_and_active(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_ACTIVE_POLL_INTERVAL_SEC": "0.03",
                "AGENT_IDLE_POLL_INTERVAL_SEC": "0.12",
            },
            clear=False,
        ):
            with patch(
                "models.agent.require_backend_info",
                return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
            ):
                agent = RLAgent(AgentConfig())

        self.assertAlmostEqual(agent._poll_interval_for_state({"waitingFor": {"type": "card"}}), 0.03, places=6)
        self.assertAlmostEqual(agent._poll_interval_for_state({"waitingFor": {}}), 0.12, places=6)
        self.assertAlmostEqual(agent._poll_interval_for_state({"waitingFor": None}), 0.12, places=6)
        self.assertAlmostEqual(agent._poll_interval_for_state({}), 0.12, places=6)

    def test_fallback_reuses_cached_raw_actions_without_recomputing(self):
        with patch(
            "models.agent.require_backend_info",
            return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
        ):
            agent = RLAgent(AgentConfig())

        agent.state_encoder.encode = lambda _state, _turn=0: np.zeros((agent.config.state_size,), dtype=np.float32)
        agent.action_decoder = _DecoderStub()
        agent.post_move_sleep_sec = 0.0
        agent.failure_pause_sec = 0.0
        agent.initial_cards_reject_pause_sec = 0.0
        agent.max_fallback_attempts = 1
        agent.max_fallback_random_retries_per_prompt = 1

        async def _fake_get_action(*_args, **_kwargs):
            return {"type": "policyAction"}, 1, True, {"available_actions_raw": [5]}

        agent._get_action_from_network = _fake_get_action
        game = _GameStub()
        player_state = {"waitingFor": {"type": "or", "options": [{"title": "Take action"}]}}

        result = asyncio.run(agent._make_move(game, "player-1", player_state, []))

        self.assertTrue(result)
        self.assertEqual(len(game.sent_actions), 2)
        self.assertEqual(str(game.sent_actions[0].get("type", "")), "policyAction")
        self.assertEqual(int(game.sent_actions[1].get("index", -1)), 5)

    def test_compute_aux_targets_uses_or_project_card_options(self):
        with patch(
            "models.agent.require_backend_info",
            return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
        ):
            agent = RLAgent(AgentConfig())

        player_state = {
            "thisPlayer": {
                "color": "red",
                "megaCredits": 21,
                "steel": 3,
                "titanium": 9,
                "steelProduction": 1,
                "titaniumProduction": 2,
                "steelValue": 2,
                "titaniumValue": 3,
            },
            "game": {"milestones": [], "awards": []},
            "waitingFor": {
                "type": "or",
                "options": [
                    {
                        "type": "projectCard",
                        "title": "Play project card",
                        "cards": [
                            {"name": "Venus Governor", "cost": 4, "tags": ["earth"]},
                            {"name": "Food Factory", "cost": 12, "tags": ["building"]},
                        ],
                    }
                ],
            },
        }

        targets = agent._compute_aux_targets(player_state)
        self.assertAlmostEqual(targets["playable_cards"], 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
