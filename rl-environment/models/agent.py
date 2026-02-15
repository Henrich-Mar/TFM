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
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict
from collections import deque

from game_interface import GameInstance, ServerTransportError
from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
from scoring import calculate_terminal_reward, calculate_step_reward
import random
import aiohttp

from .ppo import PPORolloutStep, PPOHyperParameters, optimize_ppo_policy

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
        
        # State encoder - processes game state
        layers = []
        # Input layer
        layers.append(nn.Linear(config.state_size, config.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(max(1, config.num_layers - 2)):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            
        # Final shared layer before heads
        layers.append(nn.Linear(config.hidden_size, config.hidden_size))
        layers.append(nn.ReLU())
        
        self.state_encoder = nn.Sequential(*layers)
        
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
        # PPO optimizer LR is decoupled from evolutionary config learning_rate by default.
        self.ppo_learning_rate = self._safe_env_float("PPO_LEARNING_RATE", 3e-4)
        self.ppo_lr_min = self._safe_env_float(
            "PPO_LR_MIN",
            max(1e-6, float(self.ppo_learning_rate) * 0.2),
        )
        self.ppo_lr_max = self._safe_env_float(
            "PPO_LR_MAX",
            max(float(self.ppo_lr_min), float(self.ppo_learning_rate) * 5.0),
        )
        self.ppo_lr_adapt_up = self._safe_env_float("PPO_LR_ADAPT_UP", 1.03)
        self.ppo_lr_adapt_down = self._safe_env_float("PPO_LR_ADAPT_DOWN", 0.85)
        
        # Neural network
        self.network = TerraformingMarsNetwork(self.config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.ppo_learning_rate)
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
        self.ppo_enable = str(os.getenv("PPO_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.ppo_rollout_steps = self._safe_env_int("PPO_ROLLOUT_STEPS", 8192)
        self.ppo_buffer_max_steps = self._safe_env_int("PPO_BUFFER_MAX_STEPS", 65536)
        self.state_schema_version = str(os.getenv("STATE_SCHEMA_VERSION", "v1")).strip() or "v1"
        self.strict_on_policy_sampling = str(os.getenv("PPO_STRICT_ON_POLICY", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.exploration_decay_games = max(1, self._safe_env_int("EXPLORATION_DECAY_GAMES", 200))
        self.policy_epsilon_cap = max(0.0, self._safe_env_float("POLICY_EPSILON_CAP", 0.02))
        self.policy_epsilon_floor = max(0.0, self._safe_env_float("POLICY_EPSILON_FLOOR", 0.001))
        self.policy_temperature_cap = max(1e-3, self._safe_env_float("POLICY_TEMPERATURE_CAP", 1.0))
        self.policy_temperature_floor = max(1e-3, self._safe_env_float("POLICY_TEMPERATURE_FLOOR", 0.75))
        self.project_card_priority_weight = max(0.1, self._safe_env_float("PLAY_CARD_PRIORITY_WEIGHT", 1.2))
        self.max_fallback_attempts = max(1, self._safe_env_int("MAX_FALLBACK_ACTION_ATTEMPTS", 6))
        self.rejected_action_memory_size = max(64, self._safe_env_int("REJECTED_ACTION_MEMORY_SIZE", 2048))
        self.ppo_hparams = PPOHyperParameters(
            clip_eps=self._safe_env_float("PPO_CLIP_EPS", 0.2),
            value_clip_eps=self._safe_env_float("PPO_VALUE_CLIP_EPS", 0.2),
            gamma=self._safe_env_float("PPO_GAMMA", 0.99),
            gae_lambda=self._safe_env_float("PPO_GAE_LAMBDA", 0.95),
            epochs=self._safe_env_int("PPO_EPOCHS", 4),
            minibatch_size=self._safe_env_int("PPO_MINIBATCH_SIZE", 1024),
            entropy_coef=self._safe_env_float("PPO_ENTROPY_COEF", 0.01),
            value_coef=self._safe_env_float("PPO_VALUE_COEF", 0.5),
            max_grad_norm=self._safe_env_float("PPO_MAX_GRAD_NORM", 1.0),
            target_kl=self._safe_env_float("PPO_TARGET_KL", 0.02),
        )
        self.rollout_buffer: deque[PPORolloutStep] = deque(maxlen=max(1, int(self.ppo_buffer_max_steps)))
        self.poll_interval_sec = float(os.getenv("AGENT_POLL_INTERVAL_SEC", "0.2"))
        self.post_move_sleep_sec = float(os.getenv("AGENT_POST_MOVE_SLEEP_SEC", "0.0"))
        self.failure_pause_sec = float(os.getenv("AGENT_FAILURE_PAUSE_SEC", "0.0"))
        self.stuck_log_cooldown_sec = float(os.getenv("AGENT_STUCK_LOG_COOLDOWN_SEC", "5.0"))
        self._last_stuck_log_by_player: Dict[str, float] = {}
        self._rejected_actions_by_prompt: Dict[str, set[int]] = {}
        self._rejected_action_prompt_order: deque[str] = deque()
        
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
            'action_mask_observations': 0,
            'action_legal_count_total': 0,
            'action_rejected_by_server': 0,
            'policy_actions_blocked_by_reject_cache': 0,
        }

    @staticmethod
    def _safe_env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    @staticmethod
    def _safe_env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return int(default)

    def _exploration_progress(self) -> float:
        games = max(0, int(self.games_played))
        decay_games = max(1, int(self.exploration_decay_games))
        return max(0.0, min(1.0, float(games) / float(decay_games)))

    def _effective_policy_epsilon(self, force_random: bool = False) -> float:
        if force_random:
            return 1.0
        raw = max(0.0, float(self.config.epsilon))
        capped = min(raw, float(self.policy_epsilon_cap))
        floor = min(capped, max(0.0, float(self.policy_epsilon_floor)))
        progress = self._exploration_progress()
        decayed = floor + ((capped - floor) * (1.0 - progress))
        if self.strict_on_policy_sampling and self.ppo_enable:
            return 0.0
        return max(0.0, float(decayed))

    def _effective_policy_temperature(self) -> float:
        raw = max(1e-3, float(self.config.temperature))
        capped = min(raw, max(1e-3, float(self.policy_temperature_cap)))
        floor = min(capped, max(1e-3, float(self.policy_temperature_floor)))
        progress = self._exploration_progress()
        decayed = floor + ((capped - floor) * (1.0 - progress))
        return max(1e-3, float(decayed))

    def _build_prompt_signature(self, player_state: Dict[str, Any]) -> str:
        waiting_for = player_state.get('waitingFor', {}) if isinstance(player_state, dict) else {}
        if not isinstance(waiting_for, dict):
            return ''

        cards = waiting_for.get('cards', []) or []
        card_signature = []
        for card in cards[:16]:
            if not isinstance(card, dict):
                continue
            card_signature.append(
                {
                    "name": str(card.get("name", "") or ""),
                    "cost": int(card.get("calculatedCost", card.get("cost", 0)) or 0),
                    "reserveUnits": card.get("reserveUnits", {}) if isinstance(card.get("reserveUnits", {}), dict) else {},
                }
            )

        options = waiting_for.get('options', []) or []
        option_signature = []
        for option in options[:16]:
            if not isinstance(option, dict):
                continue
            title = option.get("title", "")
            if isinstance(title, dict):
                title = title.get("message", "")
            option_signature.append(
                {
                    "type": str(option.get("type", "") or ""),
                    "title": str(title or ""),
                    "buttonLabel": str(option.get("buttonLabel", "") or ""),
                }
            )

        player = player_state.get('thisPlayer', {}) if isinstance(player_state, dict) else {}
        player_budget = {}
        if isinstance(player, dict):
            for key in ("megaCredits", "steel", "titanium", "heat", "plants"):
                player_budget[key] = int(player.get(key, 0) or 0)

        title = waiting_for.get("title", "")
        if isinstance(title, dict):
            title = title.get("message", "")

        signature_payload = {
            "type": str(waiting_for.get("type", "") or ""),
            "title": str(title or ""),
            "buttonLabel": str(waiting_for.get("buttonLabel", "") or ""),
            "amount": int(waiting_for.get("amount", 0) or 0),
            "min": int(waiting_for.get("min", 0) or 0),
            "max": int(waiting_for.get("max", 0) or 0),
            "canPass": bool(waiting_for.get("canPass", False)),
            "paymentOptions": waiting_for.get("paymentOptions", {}) if isinstance(waiting_for.get("paymentOptions", {}), dict) else {},
            "reserveUnits": waiting_for.get("reserveUnits", {}) if isinstance(waiting_for.get("reserveUnits", {}), dict) else {},
            "cards": card_signature,
            "options": option_signature,
            "playerBudget": player_budget,
        }
        try:
            serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            serialized = str(signature_payload)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return str(digest)

    def _prompt_cache_key(self, player_id: str, player_state: Dict[str, Any]) -> str:
        signature = self._build_prompt_signature(player_state)
        if not signature:
            return ''
        return f"{str(player_id)}:{signature}"

    def _get_rejected_actions_for_prompt(self, player_id: str, player_state: Dict[str, Any]) -> set[int]:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return set()
        return set(int(a) for a in self._rejected_actions_by_prompt.get(cache_key, set()))

    def _remember_rejected_action(self, player_id: str, player_state: Dict[str, Any], action_index: int) -> None:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return
        if cache_key not in self._rejected_actions_by_prompt:
            self._rejected_actions_by_prompt[cache_key] = set()
            self._rejected_action_prompt_order.append(cache_key)
        self._rejected_actions_by_prompt[cache_key].add(int(action_index))
        self._prune_rejected_action_cache()

    def _clear_rejected_action(self, player_id: str, player_state: Dict[str, Any], action_index: int) -> None:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return
        blocked = self._rejected_actions_by_prompt.get(cache_key)
        if not blocked:
            return
        blocked.discard(int(action_index))
        if not blocked:
            self._rejected_actions_by_prompt.pop(cache_key, None)

    def _prune_rejected_action_cache(self) -> None:
        max_entries = max(64, int(self.rejected_action_memory_size))
        while len(self._rejected_actions_by_prompt) > max_entries:
            if not self._rejected_action_prompt_order:
                self._rejected_actions_by_prompt.clear()
                return
            oldest_key = self._rejected_action_prompt_order.popleft()
            self._rejected_actions_by_prompt.pop(oldest_key, None)
        while len(self._rejected_action_prompt_order) > (max_entries * 2):
            self._rejected_action_prompt_order.popleft()
        
    async def play_game(self, game_instance: GameInstance, player_name: str):
        """Play a complete game"""
        episode_steps: List[Dict[str, Any]] = []
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

            # Queue PPO trajectory for coordinator-driven optimization.
            if self.train_from_self_play and game_outcome.get("completed", False):
                reward = self._compute_terminal_reward(
                    game_outcome.get("rank", 4),
                    game_outcome.get("vp", 0),
                    game_outcome.get("completed", False),
                )
                reward *= self.self_play_reward_scale
                if self.ppo_enable:
                    await self._queue_episode_rollout(episode_steps, reward)
                else:
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
            except Exception as e:
                if isinstance(e, ServerTransportError):
                    raise
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
        episode_steps: List[Dict[str, Any]],
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
            policy_action, policy_action_idx, sampled_from_policy, action_meta = await self._get_action_from_network(
                state_vector,
                player_state,
                player_id=player_id,
                force_random=False,
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
                        self._clear_rejected_action(player_id, player_state, int(policy_action_idx))
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
                            if action_meta is not None:
                                episode_steps.append(
                                    {
                                        "state": state_vector.astype(np.float32),
                                        "action": int(policy_action_idx),
                                        "reward": float(step_reward),
                                        "logp_old": float(action_meta.get("logp_old", 0.0)),
                                        "value_old": float(action_meta.get("value_old", 0.0)),
                                        "legal_actions": list(action_meta.get("legal_actions", [])),
                                    }
                                )
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return  # Success
                else:
                    self._bump_decision_stat('policy_rejections')
                    self._bump_decision_stat('action_rejected_by_server')
                    if policy_action_idx is not None:
                        action_idx = int(policy_action_idx)
                        tried_action_indices.add(action_idx)
                        self._remember_rejected_action(player_id, player_state, action_idx)
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
            prompt_rejected_actions = self._get_rejected_actions_for_prompt(player_id, player_state)
            if prompt_rejected_actions:
                candidate_actions = [a for a in available_actions if int(a) not in prompt_rejected_actions]
                if candidate_actions:
                    blocked_count = max(0, int(len(available_actions) - len(candidate_actions)))
                    if blocked_count > 0:
                        self._bump_decision_stat('policy_actions_blocked_by_reject_cache', blocked_count)
                    available_actions = candidate_actions
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
            max_attempts = min(len(available_actions), int(self.max_fallback_attempts))
             
            for i in range(max_attempts):
                random_action_idx = available_actions[i]
                random_action = self.action_decoder.decode_action(random_action_idx, player_state)
                 
                if random_action:
                    self._bump_decision_stat('fallback_random_attempts')
                    if await game_instance.send_player_input(player_id, random_action):
                        self._bump_decision_stat('fallback_random_successes')
                        self._record_action_choice(int(random_action_idx), random_action, player_state)
                        self._clear_rejected_action(player_id, player_state, int(random_action_idx))
                        logger.info(f"Random action succeeded for agent {self.id[:8]}.")
                        await self._sleep_if_needed(self.post_move_sleep_sec)
                        return  # Success
                    self._remember_rejected_action(player_id, player_state, int(random_action_idx))

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
                    self._clear_rejected_action(player_id, player_state, fallback_action_idx)
                    logger.info(f"Mandatory fallback action succeeded for agent {self.id[:8]}.")
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return
                if fallback_action:
                    self._remember_rejected_action(player_id, player_state, fallback_action_idx)

            self._log_stuck_context(game_instance, player_id, player_state, "all_random_actions_failed_no_pass")
            await self._sleep_if_needed(self.failure_pause_sec or self.poll_interval_sec)
            return

        except Exception as e:
            if isinstance(e, ServerTransportError):
                raise
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
            # When at least one non-sell productive action exists, hide sell-patents
            # choices so they are only used as a last-resort liquidity tool.
            filtered_non_sell = [a for a in filtered if not _is_sell_patents_action(int(a))]
            return filtered_non_sell if filtered_non_sell else (filtered if filtered else available_actions)

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
    
    async def _get_action_from_network(
        self,
        state_vector: np.ndarray,
        player_state: Dict[str, Any],
        player_id: Optional[str] = None,
        force_random: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool, Optional[Dict[str, Any]]]:
        """Get action from neural network"""
        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
             
            with torch.no_grad():
                policy_logits, value = self.network(state_tensor)

                # Use decayed/capped exploration temperature for action selection.
                policy_temperature = self._effective_policy_temperature()
                policy_logits = policy_logits / max(policy_temperature, 1e-3)
                policy_probs = F.softmax(policy_logits, dim=-1)
            
            # Get available actions
            available_actions = self.action_decoder.get_available_actions(player_state)
            available_actions = self._filter_pass_actions(available_actions, player_state)
            if player_id:
                rejected_actions = self._get_rejected_actions_for_prompt(player_id, player_state)
                if rejected_actions:
                    filtered_actions = [a for a in available_actions if int(a) not in rejected_actions]
                    if filtered_actions:
                        blocked_count = max(0, int(len(available_actions) - len(filtered_actions)))
                        if blocked_count > 0:
                            self._bump_decision_stat('policy_actions_blocked_by_reject_cache', blocked_count)
                        available_actions = filtered_actions
            waiting_for = player_state.get('waitingFor', {})
            self._bump_decision_stat('available_action_observations')
            self._bump_decision_stat('sum_available_actions', len(available_actions))
            self._bump_decision_stat('action_mask_observations')
            self._bump_decision_stat('action_legal_count_total', len(available_actions))
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
                return None, None, False, None
                 
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
                        adjustments[idx] = 0.08
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
            action_index, sampled_from_policy, sampled_distribution = self._sample_action(
                policy_probs.squeeze(),
                available_actions,
                force_random=force_random,
                action_weight_adjustments=action_weight_adjustments,
                prefer_project_cards=prefer_project_cards,
            )
             
            # Convert to game input
            action_input = self.action_decoder.decode_action(action_index, player_state)
            action_meta: Optional[Dict[str, Any]] = None
            if sampled_from_policy and sampled_distribution is not None:
                action_prob = float(sampled_distribution[int(action_index)].item())
                action_meta = {
                    "logp_old": float(np.log(max(1e-8, action_prob))),
                    "value_old": float(value.squeeze().item()),
                    "legal_actions": [int(a) for a in available_actions],
                    "policy_temperature": float(policy_temperature),
                }

            return action_input, action_index, sampled_from_policy, action_meta
             
        except Exception as e:
            logger.error(f"Error getting action from network: {e}")
            return None, None, False, None
    
    def _sample_action(
        self,
        policy_probs: torch.Tensor,
        available_actions: List[int],
        force_random: bool = False,
        action_weight_adjustments: Optional[Dict[int, float]] = None,
        prefer_project_cards: bool = False,
    ) -> Tuple[int, bool, Optional[torch.Tensor]]:
        """Sample action from policy, restricted to available actions"""
        effective_epsilon = self._effective_policy_epsilon(force_random=force_random)
        if np.random.random() < float(effective_epsilon):
            return np.random.choice(available_actions), False, None

        # Mask unavailable actions (strict mask; never sample hidden actions).
        masked_probs = torch.zeros_like(policy_probs)
        valid_actions = [
            int(action_idx)
            for action_idx in available_actions
            if 0 <= int(action_idx) < len(policy_probs)
        ]
        if not valid_actions:
            return np.random.choice(available_actions), False, None
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

        for i, action_idx in enumerate(valid_actions):
            if action_idx >= pass_action_base:
                masked_probs[action_idx] *= 0.3  # Reduce pass action probability
            elif action_idx == sell_patents_action:
                masked_probs[action_idx] *= 0.08  # Strongly discourage routine sell-patents usage
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
        if prefer_project_cards and play_card_actions and standard_project_actions:
            for action_idx in valid_actions:
                if action_idx < 100:
                    masked_probs[action_idx] *= float(self.project_card_priority_weight)
        
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
            return np.random.choice(valid_actions), False, None
        masked_probs = masked_probs / total_prob

        # Sample from policy
        try:
            return torch.multinomial(masked_probs, 1).item(), True, masked_probs
        except RuntimeError:
            # Fallback to non-pass action if possible
            non_pass_actions = [a for a in available_actions if a < pass_action_base]
            if non_pass_actions:
                return np.random.choice(non_pass_actions), False, None
            return np.random.choice(available_actions), False, None
    
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

    async def _queue_episode_rollout(self, episode_steps: List[Dict[str, Any]], terminal_reward: float):
        """Queue rollout transitions for coordinator-driven PPO optimization."""
        if not episode_steps:
            return

        steps = episode_steps[-max(1, self.config.max_episode_steps):]
        rollout_steps: List[PPORolloutStep] = []
        for idx, step in enumerate(steps):
            try:
                state_vector = np.asarray(step.get("state"), dtype=np.float32).reshape(-1)
                if int(state_vector.size) != int(self.config.state_size):
                    continue
                if not np.isfinite(state_vector).all():
                    continue
                reward = float(step.get("reward", 0.0))
                if idx == len(steps) - 1:
                    reward += float(terminal_reward)
                rollout_steps.append(
                    PPORolloutStep(
                        state=state_vector,
                        action=int(step.get("action", 0)),
                        logp_old=float(step.get("logp_old", 0.0)),
                        value_old=float(step.get("value_old", 0.0)),
                        reward=reward,
                        done=(idx == len(steps) - 1),
                        legal_actions=[int(a) for a in step.get("legal_actions", [])],
                        state_schema_version=self.state_schema_version,
                    )
                )
            except Exception:
                continue

        if not rollout_steps:
            return

        async with self.training_lock:
            for step in rollout_steps:
                self.rollout_buffer.append(step)

        logger.info(
            "Agent %s queued PPO rollout: steps=%d terminal_reward=%.3f buffer_size=%d",
            self.id[:8],
            len(rollout_steps),
            float(terminal_reward),
            len(self.rollout_buffer),
        )

    async def optimize_from_rollout_buffer(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run PPO optimization from buffered rollout data."""
        if not self.ppo_enable:
            return {}

        take = int(max_steps) if max_steps is not None else int(self.ppo_rollout_steps)
        take = max(1, take)
        expected_schema_version = str(self.state_schema_version or "v1")
        schema_filtered = 0

        async with self.training_lock:
            if not self.rollout_buffer:
                return {}
            steps: List[PPORolloutStep] = []
            while self.rollout_buffer and len(steps) < take:
                candidate = self.rollout_buffer.popleft()
                if str(getattr(candidate, "state_schema_version", "")) != expected_schema_version:
                    schema_filtered += 1
                    continue
                steps.append(candidate)

            if not steps:
                return {"rollout/steps": 0, "rollout/schema_filtered": int(schema_filtered)}

            metrics = optimize_ppo_policy(
                network=self.network,
                optimizer=self.optimizer,
                steps=steps,
                ppo=self.ppo_hparams,
                policy_temperature=max(float(self._effective_policy_temperature()), 1e-3),
            )
            if metrics:
                metrics.update(self._adapt_ppo_learning_rate(float(metrics.get("ppo/approx_kl", 0.0))))
                metrics["ppo/learning_rate"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
                metrics["ppo/target_kl"] = float(self.ppo_hparams.target_kl)
            self.network.eval()

        metrics["rollout/schema_filtered"] = int(schema_filtered)
        if metrics:
            logger.info(
                "Agent %s PPO update: steps=%d policy_loss=%.4f value_loss=%.4f approx_kl=%.4f",
                self.id[:8],
                int(metrics.get("rollout/steps", 0)),
                float(metrics.get("ppo/policy_loss", 0.0)),
                float(metrics.get("ppo/value_loss", 0.0)),
                float(metrics.get("ppo/approx_kl", 0.0)),
            )
        return metrics

    def _adapt_ppo_learning_rate(self, approx_kl: float) -> Dict[str, float]:
        target_kl = float(self.ppo_hparams.target_kl)
        prev_lr = float(self.optimizer.param_groups[0].get("lr", float(self.ppo_learning_rate)))

        # Keep learning-rate adaptation deterministic and bounded.
        up_factor = max(1.0, float(self.ppo_lr_adapt_up))
        down_factor = min(1.0, max(1e-6, float(self.ppo_lr_adapt_down)))
        min_lr = max(1e-7, float(self.ppo_lr_min))
        max_lr = max(min_lr, float(self.ppo_lr_max))

        next_lr = prev_lr
        if target_kl > 0.0:
            if approx_kl > (1.5 * target_kl):
                next_lr = prev_lr * down_factor
            elif approx_kl < (0.5 * target_kl):
                next_lr = prev_lr * up_factor

        next_lr = max(min_lr, min(max_lr, float(next_lr)))
        if abs(next_lr - prev_lr) > 1e-12:
            for group in self.optimizer.param_groups:
                group["lr"] = float(next_lr)

        return {
            "ppo/learning_rate_prev": float(prev_lr),
            "ppo/learning_rate_next": float(next_lr),
            "ppo/lr_adjustment_applied": 1.0 if abs(next_lr - prev_lr) > 1e-12 else 0.0,
        }

    def get_rollout_buffer_size(self) -> int:
        return int(len(self.rollout_buffer))

    async def _train_from_episode(self, episode_steps: List[Dict[str, Any]], terminal_reward: float):
        """Policy/value update from one self-play episode with terminal reward."""
        if not episode_steps:
            return

        # Protect optimizer/network updates when the same agent appears in concurrent games.
        async with self.training_lock:
            steps = episode_steps[-max(1, self.config.max_episode_steps):]
            states = torch.from_numpy(np.stack([np.asarray(step.get("state"), dtype=np.float32) for step in steps], axis=0)).float()
            actions = torch.tensor([int(step.get("action", 0)) for step in steps], dtype=torch.long)
            step_rewards = [
                float(step.get("reward", 0.0))
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
        action_mask_observations = int(self.decision_stats.get('action_mask_observations', 0))
        action_legal_count_total = int(self.decision_stats.get('action_legal_count_total', 0))
        action_rejected_by_server = int(self.decision_stats.get('action_rejected_by_server', 0))
        policy_actions_blocked_by_reject_cache = int(self.decision_stats.get('policy_actions_blocked_by_reject_cache', 0))

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
            'action_legal_count_mean': _ratio(action_legal_count_total, action_mask_observations),
            'action_mask_coverage_rate': _ratio(action_mask_observations, total_decisions),
            'action_rejected_by_server': action_rejected_by_server,
            'policy_actions_blocked_by_reject_cache': policy_actions_blocked_by_reject_cache,
            'reject_cache_entries': int(len(self._rejected_actions_by_prompt)),
            'rollout_buffer_size': int(len(self.rollout_buffer)),
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
        max_epsilon = max(0.0, self._safe_env_float("EVOLUTION_MAX_EPSILON", 0.12))
        min_epsilon = max(0.0, self._safe_env_float("EVOLUTION_MIN_EPSILON", 0.001))
        max_temperature = max(1e-3, self._safe_env_float("EVOLUTION_MAX_TEMPERATURE", 1.2))
        min_temperature = max(1e-3, self._safe_env_float("EVOLUTION_MIN_TEMPERATURE", 0.6))
        sampled_epsilon = np.random.uniform(
            min(self.config.epsilon, other_agent.config.epsilon),
            max(self.config.epsilon, other_agent.config.epsilon),
        )
        sampled_temperature = np.random.uniform(
            min(self.config.temperature, other_agent.config.temperature),
            max(self.config.temperature, other_agent.config.temperature),
        )
        clamped_epsilon = float(max(min_epsilon, min(max_epsilon, sampled_epsilon)))
        clamped_temperature = float(max(min_temperature, min(max_temperature, sampled_temperature)))

        # Create new agent
        child_config = AgentConfig(
            state_size=self.config.state_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            learning_rate=np.random.choice([self.config.learning_rate, other_agent.config.learning_rate]),
            epsilon=clamped_epsilon,
            temperature=clamped_temperature,
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
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.ppo_learning_rate)

        self.network.load_state_dict(checkpoint['network_state_dict'])

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state from {path}: {e}")
        # Keep PPO optimizer LR env-driven even after restoring optimizer state.
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.ppo_learning_rate)

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
    

    
