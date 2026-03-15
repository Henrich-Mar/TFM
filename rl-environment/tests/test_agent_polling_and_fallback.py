import asyncio
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add the rl-environment directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.agent import AgentConfig, RLAgent
from models.planner_common import PLANNER_GLOBAL_DIM, PLANNER_OPPORTUNITY_LIMIT, PLANNER_TOKEN_DIM, planner_aux_layout
from models.state_encoder import StateEncoder


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

    def get_legal_action_descriptors(self, _player_state):
        return [
            {
                "action_index": 5,
                "action_position": 0,
                "token_features": np.zeros((PLANNER_TOKEN_DIM,), dtype=np.float32),
            }
        ]

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

    def test_legacy_startup_selection_defaults_to_autosubmit(self):
        env = dict(os.environ)
        env["AGENT_STARTUP_PLAN_SELECTION"] = "legacy"
        env.pop("AGENT_STARTUP_AUTOSUBMIT", None)
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "models.agent.require_backend_info",
                return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
            ):
                agent = RLAgent(AgentConfig())

        self.assertTrue(agent.startup_autosubmit)

    def test_explicit_startup_autosubmit_off_overrides_legacy_default(self):
        env = dict(os.environ)
        env["AGENT_STARTUP_PLAN_SELECTION"] = "legacy"
        env["AGENT_STARTUP_AUTOSUBMIT"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "models.agent.require_backend_info",
                return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
            ):
                agent = RLAgent(AgentConfig())

        self.assertFalse(agent.startup_autosubmit)

    def test_fallback_reuses_cached_raw_actions_without_recomputing(self):
        with patch(
            "models.agent.require_backend_info",
            return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
        ):
            agent = RLAgent(AgentConfig())

        agent.state_encoder.encode = lambda _state, _turn=0, _descriptors=None: {
            "world_tokens": np.zeros((2, PLANNER_TOKEN_DIM), dtype=np.float32),
            "world_token_types": np.asarray([1, 2], dtype=np.int64),
            "world_mask": np.asarray([True, True], dtype=np.bool_),
            "hand_tokens": np.zeros((0, PLANNER_TOKEN_DIM), dtype=np.float32),
            "hand_mask": np.zeros((0,), dtype=np.bool_),
            "action_tokens": np.zeros((1, PLANNER_TOKEN_DIM), dtype=np.float32),
            "action_mask": np.asarray([True], dtype=np.bool_),
            "action_indices": np.asarray([5], dtype=np.int64),
            "action_positions": np.asarray([0], dtype=np.int64),
            "global_scalars": np.zeros((PLANNER_GLOBAL_DIM,), dtype=np.float32),
        }
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
                "plants": 7,
                "heat": 8,
                "steelProduction": 1,
                "titaniumProduction": 2,
                "steelValue": 2,
                "titaniumValue": 3,
            },
            "game": {
                "generation": 10,
                "milestones": [],
                "awards": [],
            },
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
        self.assertIn("planner_vector", targets)
        vector = np.asarray(targets["planner_vector"], dtype=np.float32)
        layout = planner_aux_layout(
            num_milestones=len(StateEncoder._ALL_MILESTONES),
            num_awards=len(StateEncoder._ALL_AWARDS),
            opportunity_limit=PLANNER_OPPORTUNITY_LIMIT,
        )
        self.assertGreater(vector.size, 200)
        self.assertGreater(float(vector[layout["carry_save_plants_value"].start]), 0.0)
        self.assertGreater(float(vector[layout["carry_save_heat_value"].start]), 0.0)


if __name__ == "__main__":
    unittest.main()
