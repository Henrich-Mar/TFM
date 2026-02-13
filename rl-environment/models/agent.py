"""
RL Agent - Neural network model for playing Terraforming Mars
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict
from collections import deque

from game_interface import GameInstance
from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
from scoring import calculate_terminal_reward, calculate_step_reward
import random
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    state_size: int = 512
    hidden_size: int = 256
    num_layers: int = 3
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    epsilon: float = 0.05  # Reduced for more policy-driven behavior
    temperature: float = 1.2  # Slightly increased for more exploration
    max_thinking_time: float = 5.0  # Max seconds to think per move
    train_from_self_play: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    max_episode_steps: int = 512

class TerraformingMarsNetwork(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # State encoder - processes game state
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        
        # Value head - estimates position value
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()
        )
        
        # Action policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1000),  # Large action space
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: return policy and value"""
        x = self.state_encoder(state)
        
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        return policy_logits, value
# Removed conflicting Agent class - using RLAgent instead
        
class RLAgent:
    def __init__(self, config: AgentConfig = None, agent_id: str = None):
        self.id = agent_id or str(uuid.uuid4())
        self.config = config or AgentConfig()
        
        # Neural network
        self.network = TerraformingMarsNetwork(self.config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        # Use eval mode for inference-only tournaments (disables dropout, speeds up forward pass)
        self.network.eval()
        
        # Game interaction components
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        
        # Self-play learning configuration (env overrides)
        self.train_from_self_play = (
            self.config.train_from_self_play and os.getenv("SELF_PLAY_LEARNING", "1") != "0"
        )
        self.self_play_reward_scale = float(os.getenv("SELF_PLAY_REWARD_SCALE", "1.0"))
        self.training_lock = asyncio.Lock()
        self.poll_interval_sec = float(os.getenv("AGENT_POLL_INTERVAL_SEC", "0.2"))
        self.post_move_sleep_sec = float(os.getenv("AGENT_POST_MOVE_SLEEP_SEC", "0.0"))
        self.failure_pause_sec = float(os.getenv("AGENT_FAILURE_PAUSE_SEC", "0.0"))
        self.stuck_log_cooldown_sec = float(os.getenv("AGENT_STUCK_LOG_COOLDOWN_SEC", "5.0"))
        self._last_stuck_log_by_player: Dict[str, float] = {}
        
        # Performance tracking
        self.games_played = 0
        self.total_victory_points = 0
        self.wins = 0
        self.decision_stats: Dict[str, Any] = {
            'total_decisions': 0,
            'policy_attempts': 0,
            'policy_successes': 0,
            'policy_rejections': 0,
            'policy_sampled_actions': 0,
            'epsilon_random_actions': 0,
            'fallback_decisions': 0,
            'fallback_random_attempts': 0,
            'fallback_random_successes': 0,
            'fallback_passes': 0,
            'no_available_actions': 0,
            'sum_available_actions': 0,
            'available_action_observations': 0,
            'action_type_counts': {},
            'standard_project_counts': {},
            'card_play_actions': 0,
            'steel_spent': 0,
            'titanium_spent': 0,
        }
        
    async def play_game(self, game_instance: GameInstance, player_name: str):
        """Play a complete game"""
        episode_steps: List[Tuple[np.ndarray, int, float]] = []
        try:
            # Join the game
            player_id = await game_instance.join_player(player_name)
            logger.info(f"Agent {self.id[:8]} joined game as {player_name} (ID: {player_id})")
            logger.info(
                "Agent %s debug links: game=%s player_api(public)=%s player_api(internal)=%s",
                self.id[:8],
                game_instance.get_public_game_url(),
                game_instance.get_public_player_api_url(player_id),
                game_instance.get_internal_player_api_url(player_id),
            )

            # Dedicated startup flow for corporation/prelude/initial card selection.
            await self._run_initial_setup(game_instance, player_id)
             
            # Game loop
            while True:
                # Get current game state
                player_state = await game_instance.get_player_state(player_id)
                
                # Check if game is over
                if player_state.get('game', {}).get('phase') == 'end':
                    break
                
                # If we are waiting for input, make a move.
                # Otherwise, we wait for our turn.
                if player_state.get('waitingFor'):
                    await self._make_move(game_instance, player_id, player_state, episode_steps)
                
                # Wait before polling again to avoid busy-waiting
                await self._sleep_if_needed(self.poll_interval_sec)
            
            # Record game completion
            self.games_played += 1
            final_state = await game_instance.get_final_state()
            game_outcome = await self._record_game_result(final_state, player_name, game_instance)

            # Sparse-reward policy/value update from self-play trajectory
            if self.train_from_self_play and game_outcome.get("completed", False):
                reward = self._compute_terminal_reward(
                    game_outcome.get("rank", 4),
                    game_outcome.get("vp", 0),
                    game_outcome.get("completed", False),
                )
                reward *= self.self_play_reward_scale
                await self._train_from_episode(episode_steps, reward)
            
        except Exception as e:
            logger.error(f"Agent {self.id[:8]} failed during game: {e}")
            raise

    async def _sleep_if_needed(self, seconds: float):
        delay = max(0.0, float(seconds or 0.0))
        if delay > 0.0:
            await asyncio.sleep(delay)

    async def _run_initial_setup(self, game_instance: GameInstance, player_id: str, max_attempts: int = 12) -> bool:
        """
        Attempt to submit initial setup choices once near game start.
        Returns True only when startup selections were successfully submitted.
        """
        attempts = max(1, int(max_attempts))
        for _ in range(attempts):
            try:
                player_state = await game_instance.get_player_state(player_id)
            except Exception:
                await self._sleep_if_needed(self.poll_interval_sec)
                continue

            waiting_for = player_state.get('waitingFor', {}) if isinstance(player_state, dict) else {}
            waiting_type = str(waiting_for.get('type', ''))
            if waiting_type in ['initialCards', 'selectInitialCards']:
                setup_action = self.action_decoder.build_initial_setup_response(player_state)
                if not setup_action:
                    return False
                logger.info(f"Agent {self.id[:8]} startup setup action: {setup_action}")
                if await game_instance.send_player_input(player_id, setup_action):
                    self._record_action_choice(800, setup_action, player_state)
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return True
                await self._sleep_if_needed(max(self.failure_pause_sec, self.poll_interval_sec))
                continue

            # If startup input is not currently visible, let normal loop handle the rest.
            if waiting_for:
                return False

            await self._sleep_if_needed(self.poll_interval_sec)

        return False

    def _log_stuck_context(self, game_instance: GameInstance, player_id: str, player_state: Dict[str, Any], reason: str):
        now = time.monotonic()
        last = self._last_stuck_log_by_player.get(player_id, 0.0)
        if (now - last) < max(0.0, self.stuck_log_cooldown_sec):
            return
        self._last_stuck_log_by_player[player_id] = now

        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        waiting_type = waiting_for.get('type', 'unknown')
        try:
            waiting_preview = json.dumps(waiting_for, ensure_ascii=True, separators=(',', ':'))
        except Exception:
            waiting_preview = str(waiting_for)
        if len(waiting_preview) > 1200:
            waiting_preview = waiting_preview[:1200] + "...(truncated)"

        logger.warning(
            "Agent %s stuck (%s). waitingFor.type=%s, GameURL=%s, PlayerAPI(public)=%s, PlayerAPI(internal)=%s, waitingFor=%s",
            self.id[:8],
            reason,
            waiting_type,
            game_instance.get_public_game_url(),
            game_instance.get_public_player_api_url(player_id),
            game_instance.get_internal_player_api_url(player_id),
            waiting_preview,
        )
    
    async def _make_move(
        self,
        game_instance: GameInstance,
        player_id: str,
        player_state: Dict[str, Any],
        episode_steps: List[Tuple[np.ndarray, int, float]],
    ):
        """Make a single move in the game, with robust fallbacks."""
        try:
            state_vector = self.state_encoder.encode(player_state)
            self._bump_decision_stat('total_decisions')
            
            # Log what we're waiting for
            waiting_for = player_state.get('waitingFor', {})
            waiting_type = waiting_for.get('type', 'unknown')
            logger.info(f"Agent {self.id[:8]} making move for input type: {waiting_type}")

            if waiting_type in ['initialCards', 'selectInitialCards']:
                initial_action = self.action_decoder.build_initial_setup_response(player_state)
                if initial_action:
                    self._bump_decision_stat('policy_attempts')
                    self._bump_decision_stat('policy_sampled_actions')
                    logger.info(f"Agent {self.id[:8]} attempting startup setup action: {initial_action}")
                    if await game_instance.send_player_input(player_id, initial_action):
                        self._bump_decision_stat('policy_successes')
                        self._record_action_choice(800, initial_action, player_state)
                        logger.info(f"Agent {self.id[:8]} startup setup action succeeded")
                        await self._sleep_if_needed(self.post_move_sleep_sec)
                        return
                    self._bump_decision_stat('policy_rejections')
                    self._log_stuck_context(game_instance, player_id, player_state, "startup_setup_rejected")
                    await self._sleep_if_needed(self.failure_pause_sec)

            # 1. Try a policy-driven action
            policy_action, policy_action_idx, sampled_from_policy = await self._get_action_from_network(
                state_vector, player_state, force_random=False
            )
            tried_action_indices = set()
            if policy_action:
                self._bump_decision_stat('policy_attempts')
                if sampled_from_policy:
                    self._bump_decision_stat('policy_sampled_actions')
                else:
                    self._bump_decision_stat('epsilon_random_actions')
                logger.info(f"Agent {self.id[:8]} attempting policy action: {policy_action}")
                if await game_instance.send_player_input(player_id, policy_action):
                    self._bump_decision_stat('policy_successes')
                    if policy_action_idx is not None:
                        self._record_action_choice(int(policy_action_idx), policy_action, player_state)
                    logger.info(f"Agent {self.id[:8]} policy action succeeded {policy_action}")
                    if sampled_from_policy and policy_action_idx is not None:
                        if len(episode_steps) < self.config.max_episode_steps:
                            step_reward = 0.0
                            if self.train_from_self_play:
                                try:
                                    post_action_state = await game_instance.get_player_state(player_id)
                                except Exception:
                                    post_action_state = None
                                step_reward = calculate_step_reward(
                                    before_state=player_state,
                                    after_state=post_action_state,
                                    action_input=policy_action,
                                )
                            episode_steps.append((state_vector.astype(np.float32), int(policy_action_idx), float(step_reward)))
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return  # Success
                else:
                    self._bump_decision_stat('policy_rejections')
                    if policy_action_idx is not None:
                        tried_action_indices.add(int(policy_action_idx))
                    logger.warning(f"Agent {self.id[:8]} policy action was rejected by game")
                    self._log_stuck_context(game_instance, player_id, player_state, "policy_action_rejected")
                    await self._sleep_if_needed(self.failure_pause_sec)

            logger.warning(f"Policy action failed for agent {self.id[:8]}. Trying random actions.")
            self._bump_decision_stat('fallback_decisions')

            # 2. Try a broader set of alternative actions, excluding already-rejected choices.
            pass_base = int(self.action_decoder.action_types.get('PASS', 900))
            raw_available_actions = self.action_decoder.get_available_actions(player_state)
            can_legally_pass = any(int(a) >= pass_base for a in raw_available_actions)

            available_actions = self._filter_pass_actions(raw_available_actions, player_state)
            available_actions = [a for a in available_actions if int(a) not in tried_action_indices]
            if not available_actions and not can_legally_pass:
                # In mandatory selection flows we cannot pass; retry non-pass actions even if tried.
                available_actions = [a for a in self._filter_pass_actions(raw_available_actions, player_state) if int(a) < pass_base]

            if not available_actions:
                # If no actions are available, pass only when pass is legal.
                self._bump_decision_stat('no_available_actions')
                if can_legally_pass:
                    self._bump_decision_stat('fallback_passes')
                    self._record_action_choice(pass_base)
                    self._log_stuck_context(game_instance, player_id, player_state, "no_available_actions_pass")
                    await game_instance.send_player_input(player_id, self.action_decoder._create_pass_action())
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                else:
                    self._log_stuck_context(game_instance, player_id, player_state, "no_available_actions_no_pass")
                    await self._sleep_if_needed(self.failure_pause_sec or self.poll_interval_sec)
                return
                 
            random.shuffle(available_actions)
            max_attempts = min(len(available_actions), 12)
             
            for i in range(max_attempts):
                random_action_idx = available_actions[i]
                random_action = self.action_decoder.decode_action(random_action_idx, player_state)
                 
                if random_action:
                    self._bump_decision_stat('fallback_random_attempts')
                    if await game_instance.send_player_input(player_id, random_action):
                        self._bump_decision_stat('fallback_random_successes')
                        self._record_action_choice(int(random_action_idx), random_action, player_state)
                        logger.info(f"Random action succeeded for agent {self.id[:8]}.")
                        await self._sleep_if_needed(self.post_move_sleep_sec)
                        return  # Success

            if can_legally_pass:
                logger.warning(f"All random actions failed for agent {self.id[:8]}. Passing.")
                self._bump_decision_stat('fallback_passes')
                self._record_action_choice(pass_base)
                self._log_stuck_context(game_instance, player_id, player_state, "all_random_actions_failed_pass")
                await game_instance.send_player_input(player_id, self.action_decoder._create_pass_action())
                await self._sleep_if_needed(self.post_move_sleep_sec)
                return

            # Mandatory prompt and no legal pass: never send invalid pass input.
            fallback_candidates = [a for a in self._filter_pass_actions(raw_available_actions, player_state) if int(a) < pass_base]
            if fallback_candidates:
                fallback_action_idx = int(fallback_candidates[0])
                fallback_action = self.action_decoder.decode_action(fallback_action_idx, player_state)
                if fallback_action and await game_instance.send_player_input(player_id, fallback_action):
                    self._bump_decision_stat('fallback_random_successes')
                    self._record_action_choice(fallback_action_idx, fallback_action, player_state)
                    logger.info(f"Mandatory fallback action succeeded for agent {self.id[:8]}.")
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return

            self._log_stuck_context(game_instance, player_id, player_state, "all_random_actions_failed_no_pass")
            await self._sleep_if_needed(self.failure_pause_sec or self.poll_interval_sec)
            return

        except Exception as e:
            logger.error(f"Error making move for agent {self.id[:8]}: {e}", exc_info=True)

    def _filter_pass_actions(self, available_actions: List[int], player_state: Dict[str, Any]) -> List[int]:
        if not available_actions:
            return available_actions
        pass_base = self.action_decoder.action_types.get('PASS', 900)
        non_pass_actions = [a for a in available_actions if a < pass_base]
        if not non_pass_actions:
            return available_actions

        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        waiting_type = str(waiting_for.get('type', ''))

        # Keep pass available during explicit selection flows (draft/research/buy/keep).
        if waiting_type in ['card', 'selectCard', 'projectCard', 'selectProjectCardToPlay', 'initialCards']:
            title = waiting_for.get('title', '')
            if isinstance(title, dict):
                title = title.get('message', '')
            title_l = str(title).lower()
            button_label = str(waiting_for.get('buttonLabel', '') or '').lower()
            if (
                waiting_for.get('showOnlyInLearnerMode', False)
                or waiting_for.get('selectBlueCardAction', False)
                or 'prelude' in title_l
                or 'research' in title_l
                or 'draft' in title_l
                or 'select' in title_l
                or button_label in ['keep', 'buy', 'select', 'choose', 'discard', 'confirm', 'ok', 'save']
                or (
                    'min' in waiting_for
                    and 'max' in waiting_for
                    and button_label in ['keep', 'buy', 'select', 'choose', 'confirm', 'ok', 'save', 'research']
                )
            ):
                return available_actions

        if waiting_for.get('type') == 'or':
            options = waiting_for.get('options', [])
            select_option_base = self.action_decoder.action_types.get('SELECT_OPTION', 200)
            filtered = list(non_pass_actions)
            pass_option_actions = set()
            sell_option_actions = set()
            for i, option in enumerate(options):
                title = option.get('title', '')
                if isinstance(title, dict):
                    title = title.get('message', '')
                title_l = str(title).lower()
                if 'pass' in title_l:
                    pass_action = select_option_base + i
                    if pass_action in filtered:
                        filtered.remove(pass_action)
                    pass_option_actions.add(pass_action)
                if 'sell patents' in title_l:
                    sell_option_actions.add(select_option_base + i)

            def _is_sell_patents_action(action_idx: int) -> bool:
                return int(action_idx) == 702 or int(action_idx) in sell_option_actions

            non_pass_non_pass_option = [a for a in non_pass_actions if int(a) not in pass_option_actions]
            productive_actions = [a for a in non_pass_non_pass_option if not _is_sell_patents_action(int(a))]

            # If the only alternative to pass is sell patents, keep pass.
            if not productive_actions:
                return available_actions
            return filtered if filtered else available_actions

        # If the only non-pass action is sell patents, keep pass to avoid forced selling.
        if non_pass_actions and all(int(a) == 702 for a in non_pass_actions):
            return available_actions

        return non_pass_actions

    def _bump_decision_stat(self, key: str, amount: int = 1):
        self.decision_stats[key] = int(self.decision_stats.get(key, 0)) + int(amount)

    def _categorize_action(self, action_index: int) -> str:
        pass_base = int(self.action_decoder.action_types.get('PASS', 900))
        mask_base = int(self.action_decoder.action_types.get('SELECT_CARD_MASK', -1))
        mask_limit = int(getattr(self.action_decoder, 'card_selection_mask_limit', 0) or 0)
        if action_index >= pass_base:
            return 'pass'
        if action_index < 100:
            return 'play_card'
        if action_index < 200:
            return 'standard_project'
        if action_index < 300:
            return 'select_option'
        if mask_base >= 0 and mask_limit > 0 and mask_base <= action_index < (mask_base + mask_limit):
            return 'card_selection_mask'
        if action_index == 700:
            return 'convert_plants'
        if action_index == 701:
            return 'convert_heat'
        if action_index == 702:
            return 'sell_patents'
        return 'other'

    def _extract_standard_project_name(
        self,
        action_index: int,
        action_input: Optional[Dict[str, Any]],
        player_state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Best-effort extraction of selected standard project name for telemetry."""
        known_names = set(getattr(self.action_decoder, "standard_projects", []) or [])

        # Direct payload naming.
        if action_input:
            if action_input.get('type') == 'standardProject':
                project = str(action_input.get('project', '') or '').strip()
                return project or None
            if action_input.get('type') == 'card':
                selected = action_input.get('cards', []) or []
                if selected:
                    candidate = str(selected[0]).strip()
                    if candidate in known_names:
                        return candidate
            if action_input.get('type') == 'projectCard':
                candidate = str(action_input.get('card', '') or '').strip()
                if candidate in known_names:
                    return candidate

        waiting_for = (player_state or {}).get('waitingFor', {}) if player_state else {}
        input_type = str(waiting_for.get('type', ''))
        title = waiting_for.get('title', '')
        if isinstance(title, dict):
            title = title.get('message', '')
        title_l = str(title).lower()

        if input_type in ['card', 'selectCard'] and 'standard project' in title_l:
            cards = waiting_for.get('cards', []) or []
            normalized = int(action_index)
            if 100 <= normalized < 200:
                normalized -= 100
            if 0 <= normalized < len(cards):
                name = str(cards[normalized].get('name', '') or '').strip()
                if name:
                    return name
            enabled = [c for c in cards if not c.get('isDisabled', False)]
            if enabled:
                name = str(enabled[0].get('name', '') or '').strip()
                return name or None
            return None

        if input_type == 'or':
            options = waiting_for.get('options', []) or []
            for option in options:
                opt_title = option.get('title', '')
                if isinstance(opt_title, dict):
                    opt_title = opt_title.get('message', '')
                opt_title_l = str(opt_title).lower()
                if option.get('type') in ['card', 'selectCard'] and 'standard project' in opt_title_l:
                    cards = option.get('cards', []) or []
                    normalized = int(action_index)
                    if 100 <= normalized < 200:
                        normalized -= 100
                    if 0 <= normalized < len(cards):
                        name = str(cards[normalized].get('name', '') or '').strip()
                        if name:
                            return name
                    enabled = [c for c in cards if not c.get('isDisabled', False)]
                    if enabled:
                        name = str(enabled[0].get('name', '') or '').strip()
                        return name or None
                    break
        return None

    def _record_action_choice(
        self,
        action_index: int,
        action_input: Optional[Dict[str, Any]] = None,
        player_state: Optional[Dict[str, Any]] = None,
    ):
        counts = self.decision_stats.setdefault('action_type_counts', {})
        category = self._categorize_action(int(action_index))
        counts[category] = int(counts.get(category, 0)) + 1
        if category == 'standard_project':
            project_name = self._extract_standard_project_name(int(action_index), action_input, player_state)
            if project_name:
                project_counts = self.decision_stats.setdefault('standard_project_counts', {})
                project_counts[project_name] = int(project_counts.get(project_name, 0)) + 1

        # Track project card plays and resource spend, including nested OR/AND payloads.
        if not isinstance(action_input, dict):
            return

        stack: List[Dict[str, Any]] = [action_input]
        while stack:
            payload = stack.pop()
            if not isinstance(payload, dict):
                continue

            action_type = str(payload.get('type', '') or '')
            if category != 'standard_project' and (
                action_type == 'projectCard'
                or (action_type == 'card' and 'card' in payload)
            ):
                self._bump_decision_stat('card_play_actions')

            payment = payload.get('payment')
            if isinstance(payment, dict):
                try:
                    steel_units = int(payment.get('steel', 0) or 0)
                except Exception:
                    steel_units = 0
                try:
                    titanium_units = int(payment.get('titanium', 0) or 0)
                except Exception:
                    titanium_units = 0
                if steel_units > 0:
                    self._bump_decision_stat('steel_spent', steel_units)
                if titanium_units > 0:
                    self._bump_decision_stat('titanium_spent', titanium_units)

            nested_response = payload.get('response')
            if isinstance(nested_response, dict):
                stack.append(nested_response)
            nested_responses = payload.get('responses')
            if isinstance(nested_responses, list):
                for item in nested_responses:
                    if isinstance(item, dict):
                        stack.append(item)
    
    async def _get_action_from_network(self, state_vector: np.ndarray, 
                                    player_state: Dict[str, Any], force_random: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool]:
        """Get action from neural network"""
        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            
            with torch.no_grad():
                policy_logits, value = self.network(state_tensor)
                
                # Apply temperature for exploration
                policy_logits = policy_logits / self.config.temperature
                policy_probs = F.softmax(policy_logits, dim=-1)
            
            # Get available actions
            available_actions = self.action_decoder.get_available_actions(player_state)
            available_actions = self._filter_pass_actions(available_actions, player_state)
            waiting_for = player_state.get('waitingFor', {})
            self._bump_decision_stat('available_action_observations')
            self._bump_decision_stat('sum_available_actions', len(available_actions))
            prefer_project_cards = False
            waiting_type = str(waiting_for.get('type', '') or '').lower()
            if waiting_type in ['projectcard', 'selectprojectcardtoplay']:
                prefer_project_cards = True
            elif waiting_type == 'or':
                options = waiting_for.get('options', [])
                has_project_card_option = any(
                    str(option.get('type', '') or '') in ['projectCard', 'selectProjectCardToPlay']
                    for option in options
                )
                def _opt_title_l(option: Dict[str, Any]) -> str:
                    raw = option.get('title', '')
                    if isinstance(raw, dict):
                        raw = raw.get('message', '')
                    return str(raw or '').lower()
                has_standard_project_option = any(
                    str(option.get('type', '') or '') in ['card', 'selectCard']
                    and 'standard project' in _opt_title_l(option)
                    for option in options
                )
                prefer_project_cards = bool(has_project_card_option and has_standard_project_option)
               
            if not available_actions:
                return None, None, False
                 
            # Log available action types for debugging
            action_types = []
            option_titles = waiting_for.get('options', []) if waiting_for.get('type') == 'or' else []
            card_mask_base = int(self.action_decoder.action_types.get('SELECT_CARD_MASK', -1))
            card_mask_limit = int(getattr(self.action_decoder, 'card_selection_mask_limit', 0) or 0)
            for action_idx in available_actions:
                if action_idx < 100:
                    action_types.append(f"PLAY_CARD({action_idx})")
                elif action_idx < 200:
                    action_types.append(f"STANDARD_PROJECT({action_idx-100})")
                elif action_idx >= 200 and action_idx < 300:  # SELECT_OPTION range
                    option_idx = action_idx - 200
                    option_name = str(option_idx)
                    if option_idx < len(option_titles):
                        title = option_titles[option_idx].get('title', '')
                        if isinstance(title, dict):
                            title = title.get('message', '')
                        title = str(title).strip()
                        option_name = title if title else option_titles[option_idx].get('type', option_name)
                    action_types.append(f"SELECT_OPTION_{option_name}({action_idx})")
                elif card_mask_base >= 0 and card_mask_limit > 0 and card_mask_base <= action_idx < (card_mask_base + card_mask_limit):
                    action_types.append(f"CARD_SELECTION_MASK({action_idx - card_mask_base})")
                elif action_idx == 700:
                    action_types.append("CONVERT_PLANTS")
                elif action_idx == 701:
                    action_types.append("CONVERT_HEAT")
                elif action_idx == 702:
                    action_types.append("SELL_PATENTS")
                elif action_idx >= 900:
                    action_types.append("PASS")
                else:
                    action_types.append(f"OTHER({action_idx})")
            
            logger.info(f"Available actions: {action_types}")
             
            # Optional: adjust weights for OR menus based on option titles to avoid passing
            action_weight_adjustments = None
            if waiting_for and waiting_for.get('type') == 'or':
                options = waiting_for.get('options', [])
                adjustments = {}
                select_option_base = 200
                for i, opt in enumerate(options):
                    title = opt.get('title', '')
                    if isinstance(title, dict):
                        title = title.get('message', '')
                    title_l = str(title).lower()
                    idx = select_option_base + i
                    # Downweight passing and selling patents
                    if 'pass' in title_l:
                        adjustments[idx] = 0.2
                    elif 'sell patents' in title_l:
                        adjustments[idx] = 0.4
                    # Upweight productive actions
                    elif 'play project card' in title_l:
                        adjustments[idx] = 1.8
                    elif 'perform an action' in title_l or 'take action' in title_l:
                        # Blue card actions entry
                        adjustments[idx] = 1.7
                    elif 'standard project' in title_l:
                        adjustments[idx] = 0.85
                    elif 'convert heat' in title_l or 'convert 8 heat' in title_l or 'convert plants' in title_l:
                        adjustments[idx] = 1.3
                if adjustments:
                    action_weight_adjustments = adjustments

            # Ensure eval mode for inference
            try:
                self.network.eval()
            except Exception:
                pass

            # Sample action based on policy
            action_index, sampled_from_policy = self._sample_action(
                policy_probs.squeeze(),
                available_actions,
                force_random=force_random,
                action_weight_adjustments=action_weight_adjustments,
                prefer_project_cards=prefer_project_cards,
            )
            
            # Convert to game input
            action_input = self.action_decoder.decode_action(action_index, player_state)
            
            return action_input, action_index, sampled_from_policy
            
        except Exception as e:
            logger.error(f"Error getting action from network: {e}")
            return None, None, False
    
    def _sample_action(
        self,
        policy_probs: torch.Tensor,
        available_actions: List[int],
        force_random: bool = False,
        action_weight_adjustments: Optional[Dict[int, float]] = None,
        prefer_project_cards: bool = False,
    ) -> Tuple[int, bool]:
        """Sample action from policy, restricted to available actions"""
        # Reduce epsilon-greedy for more policy-driven behavior
        if force_random or np.random.random() < max(0.05, self.config.epsilon * 0.5):
            return np.random.choice(available_actions), False

        # Mask unavailable actions (strict mask; never sample hidden actions).
        masked_probs = torch.zeros_like(policy_probs)
        valid_actions = [
            int(action_idx)
            for action_idx in available_actions
            if 0 <= int(action_idx) < len(policy_probs)
        ]
        if not valid_actions:
            return np.random.choice(available_actions), False
        for action_idx in valid_actions:
            masked_probs[action_idx] = torch.clamp(policy_probs[action_idx], min=0.0)
        
        # Renormalize
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback to uniform if all probabilities are zero
            for action_idx in valid_actions:
                if action_idx < len(masked_probs):
                    masked_probs[action_idx] = 1.0
            masked_probs /= masked_probs.sum()

        # Prefer productive engine-building decisions over repetitive low-ceiling lines.
        pass_action_base = 900  # From action_types['PASS']
        sell_patents_action = 702  # Sell patents action
        select_option_base = 200  # From action_types['SELECT_OPTION']
        has_play_card_action = any(0 <= int(a) < 100 for a in valid_actions)
        has_standard_project_action = any(100 <= int(a) < 200 for a in valid_actions)
        play_card_actions = [int(a) for a in valid_actions if 0 <= int(a) < 100]
        standard_project_actions = [int(a) for a in valid_actions if 100 <= int(a) < 200]

        # Aggressive bias: when both branches are legal in the action phase,
        # force a project-card attempt with configurable probability.
        if prefer_project_cards and play_card_actions and standard_project_actions:
            try:
                priority_prob = float(os.getenv("PLAY_CARD_PRIORITY_PROB", "0.90"))
            except Exception:
                priority_prob = 0.90
            priority_prob = max(0.0, min(1.0, priority_prob))
            if np.random.random() < priority_prob:
                return int(np.random.choice(play_card_actions)), True
        
        for i, action_idx in enumerate(valid_actions):
            if action_idx >= pass_action_base:
                masked_probs[action_idx] *= 0.3  # Reduce pass action probability
            elif action_idx == sell_patents_action:
                masked_probs[action_idx] *= 0.5  # Reduce sell patents probability to encourage diversity
            elif action_idx < 100:  # Play project card
                masked_probs[action_idx] *= 1.55
            elif action_idx >= 100 and action_idx < 200:  # Standard projects
                masked_probs[action_idx] *= 0.75
            elif action_idx == 700:  # Convert plants
                masked_probs[action_idx] *= 1.3  # Increase convert plants probability
            elif action_idx == 701:  # Convert heat
                masked_probs[action_idx] *= 1.3  # Increase convert heat probability
            elif action_idx >= select_option_base and action_idx < select_option_base + 100:  # SELECT_OPTION range
                # Keep option actions near-neutral; contextual boosts are applied via titles.
                masked_probs[action_idx] *= 1.0

        # When both lines are available, nudge policy toward card execution.
        if has_play_card_action and has_standard_project_action:
            for action_idx in valid_actions:
                if action_idx < 100:
                    masked_probs[action_idx] *= 1.35
                elif 100 <= action_idx < 200:
                    masked_probs[action_idx] *= 0.45
        
        # Apply contextual adjustments (e.g., OR menu titles)
        if action_weight_adjustments:
            for action_idx, mult in action_weight_adjustments.items():
                if action_idx in valid_actions and action_idx < len(masked_probs):
                    masked_probs[action_idx] *= float(mult)

        # Keep numerical stability only on valid actions.
        for action_idx in valid_actions:
            masked_probs[action_idx] += 1e-8

        # Renormalize after adjustment
        total_prob = float(masked_probs.sum().item())
        if total_prob <= 0:
            return np.random.choice(valid_actions), False
        masked_probs = masked_probs / total_prob

        # Sample from policy
        try:
            return torch.multinomial(masked_probs, 1).item(), True
        except RuntimeError:
            # Fallback to non-pass action if possible
            non_pass_actions = [a for a in available_actions if a < pass_action_base]
            if non_pass_actions:
                return np.random.choice(non_pass_actions), False
            return np.random.choice(available_actions), False
    
    async def _record_game_result(self, final_state: Dict[str, Any], player_name: str, game_instance: GameInstance) -> Dict[str, Any]:
        """Record the result of a completed game"""
        try:
            # Find our player in the final state
            our_player = None
            for player in final_state.get('players', []):
                if player.get('name') == player_name:
                    our_player = player
                    break
            
            vp = 0
            rank = 4

            # Prefer authoritative data from the JSON player view to avoid parsing dynamic HTML
            try:
                our_player_id = (our_player or {}).get('id')
                if our_player_id and game_instance and getattr(game_instance, 'session', None):
                    # Use the same base URL/session as the game instance for reliable in-cluster access
                    internal_base = getattr(game_instance, 'base_url', os.getenv('INTERNAL_TM_URL', os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')))
                    async with game_instance.session.get(f"{internal_base}/api/player", params={'id': our_player_id}) as r2:
                        if r2.status == 200:
                            view = await r2.json()
                            players_view = view.get('players', []) or []
                            # Sort like the UI: by total VP desc, then megacredits desc
                            def vp_total(p: Dict[str, Any]) -> int:
                                return int(((p.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0))
                            def mc_val(p: Dict[str, Any]) -> int:
                                return int(p.get('megaCredits', 0) or 0)
                            sorted_players = sorted(players_view, key=lambda p: (vp_total(p), mc_val(p)), reverse=True)
                            for idx, p in enumerate(sorted_players, start=1):
                                if p.get('name') == player_name:
                                    vp = vp_total(p)
                                    rank = idx
                                    break
                        else:
                            logger.warning(f"Agent {self.id[:8]} failed to fetch player view JSON (HTTP {r2.status}).")
            except Exception as _:
                pass



            # Final fallback to whatever is available in the final state
            if our_player and vp == 0:
                vp = int(((our_player.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0))
                # Rank may not be present; keep previous value if missing
                rank = int(our_player.get('rank', rank) or rank)

            # Update aggregates
            self.total_victory_points += int(vp)
            if int(rank) == 1:
                self.wins += 1
            logger.info(f"Agent {self.id[:8]} finished rank {rank} with {vp} VP")
            return {"completed": True, "rank": int(rank), "vp": int(vp)}
        
        except Exception as e:
            logger.error(f"Error recording game result: {e}")
            return {"completed": False, "rank": 4, "vp": 0}

    def _compute_terminal_reward(self, rank: int, vp: int, completed: bool = True) -> float:
        """Convert game outcome into a bounded terminal reward for policy updates."""
        return calculate_terminal_reward(rank=rank, victory_points=vp, completed=completed)

    async def _train_from_episode(self, episode_steps: List[Tuple[np.ndarray, int, float]], terminal_reward: float):
        """Policy/value update from one self-play episode with terminal reward."""
        if not episode_steps:
            return

        # Protect optimizer/network updates when the same agent appears in concurrent games.
        async with self.training_lock:
            steps = episode_steps[-max(1, self.config.max_episode_steps):]
            states = torch.from_numpy(np.stack([step[0] for step in steps], axis=0)).float()
            actions = torch.tensor([int(step[1]) for step in steps], dtype=torch.long)
            step_rewards = [
                float(step[2]) if len(step) >= 3 else 0.0
                for step in steps
            ]

            # Dense+terminal reward: back-propagate step shaping and final outcome.
            running_return = float(terminal_reward)
            returns = []
            for immediate_reward in reversed(step_rewards):
                running_return = float(immediate_reward) + (float(self.config.discount_factor) * running_return)
                returns.append(running_return)
            returns.reverse()
            returns_t = torch.tensor(returns, dtype=torch.float32)
            if returns_t.numel() > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

            self.network.train()
            policy_logits, values = self.network(states)
            policy_logits = policy_logits / max(float(self.config.temperature), 1e-3)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            values = values.squeeze(-1)

            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            policy_loss = -(chosen_log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            total_loss = (
                policy_loss
                + float(self.config.value_loss_coef) * value_loss
                - float(self.config.entropy_coef) * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.max_grad_norm))
            self.optimizer.step()
            self.network.eval()

            logger.info(
                f"Agent {self.id[:8]} self-play update: "
                f"steps={len(steps)}, reward={terminal_reward:.3f}, shaped_sum={sum(step_rewards):.3f}, "
                f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
            )
    
    def get_fitness_score(self) -> float:
        """Calculate fitness score for evolutionary selection"""
        if self.games_played == 0:
            return 0.0
        
        avg_vp = self.total_victory_points / self.games_played
        win_rate = self.wins / self.games_played
        
        # Fitness combines average VP and win rate
        fitness = avg_vp * 0.7 + win_rate * 100 * 0.3
        return fitness

    def get_behavior_stats(self) -> Dict[str, Any]:
        """Return action-decision telemetry used by the monitoring dashboard."""
        total_decisions = int(self.decision_stats.get('total_decisions', 0))
        policy_attempts = int(self.decision_stats.get('policy_attempts', 0))
        policy_successes = int(self.decision_stats.get('policy_successes', 0))
        policy_rejections = int(self.decision_stats.get('policy_rejections', 0))
        policy_sampled_actions = int(self.decision_stats.get('policy_sampled_actions', 0))
        epsilon_random_actions = int(self.decision_stats.get('epsilon_random_actions', 0))
        fallback_decisions = int(self.decision_stats.get('fallback_decisions', 0))
        fallback_random_attempts = int(self.decision_stats.get('fallback_random_attempts', 0))
        fallback_random_successes = int(self.decision_stats.get('fallback_random_successes', 0))
        fallback_passes = int(self.decision_stats.get('fallback_passes', 0))
        no_available_actions = int(self.decision_stats.get('no_available_actions', 0))
        sum_available_actions = int(self.decision_stats.get('sum_available_actions', 0))
        available_action_observations = int(self.decision_stats.get('available_action_observations', 0))
        card_play_actions = int(self.decision_stats.get('card_play_actions', 0))
        steel_spent = int(self.decision_stats.get('steel_spent', 0))
        titanium_spent = int(self.decision_stats.get('titanium_spent', 0))

        def _ratio(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator > 0 else 0.0

        action_counts = dict(self.decision_stats.get('action_type_counts', {}))
        standard_project_counts = dict(self.decision_stats.get('standard_project_counts', {}))
        total_recorded_actions = sum(int(v) for v in action_counts.values())
        action_mix = {
            key: _ratio(int(value), total_recorded_actions)
            for key, value in action_counts.items()
        }
        total_standard_projects = sum(int(v) for v in standard_project_counts.values())
        standard_project_mix = {
            key: _ratio(int(value), total_standard_projects)
            for key, value in standard_project_counts.items()
        }

        return {
            'total_decisions': total_decisions,
            'policy_attempts': policy_attempts,
            'policy_successes': policy_successes,
            'policy_rejections': policy_rejections,
            'policy_sampled_actions': policy_sampled_actions,
            'epsilon_random_actions': epsilon_random_actions,
            'fallback_decisions': fallback_decisions,
            'fallback_random_attempts': fallback_random_attempts,
            'fallback_random_successes': fallback_random_successes,
            'fallback_passes': fallback_passes,
            'no_available_actions': no_available_actions,
            'avg_available_actions': _ratio(sum_available_actions, available_action_observations),
            'policy_success_rate': _ratio(policy_successes, policy_attempts),
            'policy_sample_rate': _ratio(policy_sampled_actions, total_decisions),
            'epsilon_random_rate': _ratio(epsilon_random_actions, total_decisions),
            'fallback_decision_rate': _ratio(fallback_decisions, total_decisions),
            'fallback_random_success_rate': _ratio(fallback_random_successes, fallback_random_attempts),
            'fallback_pass_rate': _ratio(fallback_passes, total_decisions),
            'card_play_actions': card_play_actions,
            'card_plays_per_game': _ratio(card_play_actions, int(self.games_played)),
            'steel_spent': steel_spent,
            'steel_spent_per_game': _ratio(steel_spent, int(self.games_played)),
            'titanium_spent': titanium_spent,
            'titanium_spent_per_game': _ratio(titanium_spent, int(self.games_played)),
            'action_counts': action_counts,
            'action_mix': action_mix,
            'standard_project_counts': standard_project_counts,
            'standard_project_mix': standard_project_mix,
        }
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate network weights for evolutionary training"""
        with torch.no_grad():
            for param in self.network.parameters():
                if np.random.random() < mutation_rate:
                    # Add Gaussian noise
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
    
    def crossover(self, other_agent: 'RLAgent') -> 'RLAgent':
        """Create offspring by crossing over with another agent"""
        # Create new agent
        child_config = AgentConfig(
            state_size=self.config.state_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            learning_rate=np.random.choice([self.config.learning_rate, other_agent.config.learning_rate]),
            epsilon=np.random.uniform(
                min(self.config.epsilon, other_agent.config.epsilon),
                max(self.config.epsilon, other_agent.config.epsilon)
            ),
            temperature=np.random.uniform(
                min(self.config.temperature, other_agent.config.temperature),
                max(self.config.temperature, other_agent.config.temperature)
            )
        )
        
        child = RLAgent(child_config)
        
        # Mix network weights
        with torch.no_grad():
            for child_param, parent1_param, parent2_param in zip(
                child.network.parameters(),
                self.network.parameters(),
                other_agent.network.parameters()
            ):
                # Random crossover point for each parameter
                crossover_mask = torch.rand_like(child_param) < 0.5
                child_param.data = torch.where(crossover_mask, parent1_param.data, parent2_param.data)
        
        return child
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'games_played': self.games_played,
            'total_victory_points': self.total_victory_points,
            'wins': self.wins
        }, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # Our checkpoints include optimizer/config metadata, so we need full unpickling
        # for trusted local artifacts.
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Backward compatibility for torch versions without weights_only argument.
            checkpoint = torch.load(path, map_location='cpu')

        # Restore saved AgentConfig so policy behavior and training hyperparameters
        # remain consistent after resume/load.
        config_payload = checkpoint.get('config', {})
        if isinstance(config_payload, dict):
            defaults = asdict(AgentConfig())
            merged_config = defaults.copy()
            for key, value in config_payload.items():
                if key in defaults:
                    merged_config[key] = value
            self.config = AgentConfig(**merged_config)

        # Rebuild model/optimizer from restored config before loading state dicts.
        self.network = TerraformingMarsNetwork(self.config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        self.network.load_state_dict(checkpoint['network_state_dict'])

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state from {path}: {e}")

        self.network.eval()
        self.train_from_self_play = (
            self.config.train_from_self_play and os.getenv("SELF_PLAY_LEARNING", "1") != "0"
        )
        self.games_played = int(checkpoint.get('games_played', 0))
        self.total_victory_points = int(checkpoint.get('total_victory_points', 0))
        self.wins = int(checkpoint.get('wins', 0))
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            'id': self.id,
            'config': asdict(self.config),
            'games_played': self.games_played,
            'total_victory_points': self.total_victory_points,
            'wins': self.wins,
            'fitness_score': self.get_fitness_score()
        }
    

    
