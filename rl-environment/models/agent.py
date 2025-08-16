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
from typing import Dict, Any, List, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict
from collections import deque

from game_interface import GameInstance
from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
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
        
        # Training data
        self.replay_buffer = []
        self.current_episode_data = []
        
        # Performance tracking
        self.games_played = 0
        self.total_victory_points = 0
        self.wins = 0
        
    async def play_game(self, game_instance: GameInstance, player_name: str):
        """Play a complete game"""
        try:
            # Join the game
            player_id = await game_instance.join_player(player_name)
            logger.info(f"Agent {self.id[:8]} joined game as {player_name} (ID: {player_id})")
            
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
                    await self._make_move(game_instance, player_id, player_state)
                
                # Wait before polling again to avoid busy-waiting
                await asyncio.sleep(0.2)
            
            # Record game completion
            self.games_played += 1
            final_state = await game_instance.get_final_state()
            await self._record_game_result(final_state, player_name, game_instance)
            
        except Exception as e:
            logger.error(f"Agent {self.id[:8]} failed during game: {e}")
            raise
    
    async def _make_move(self, game_instance: GameInstance, player_id: str, player_state: Dict[str, Any]):
        """Make a single move in the game, with robust fallbacks."""
        try:
            state_vector = self.state_encoder.encode(player_state)
            
            # Log what we're waiting for
            waiting_for = player_state.get('waitingFor', {})
            waiting_type = waiting_for.get('type', 'unknown')
            logger.info(f"Agent {self.id[:8]} making move for input type: {waiting_type}")

            # 1. Try a policy-driven action
            policy_action = await self._get_action_from_network(state_vector, player_state, force_random=False)
            if policy_action:
                logger.info(f"Agent {self.id[:8]} attempting policy action: {policy_action}")
                if await game_instance.send_player_input(player_id, policy_action):
                    logger.info(f"Agent {self.id[:8]} policy action succeeded")
                    # await asyncio.sleep(2)  # Slow down agent: wait 2 seconds after move
                    return  # Success
                else:
                    logger.warning(f"Agent {self.id[:8]} policy action was rejected by game")

            logger.warning(f"Policy action failed for agent {self.id[:8]}. Trying random actions.")

            # 2. Try up to 3 different random actions
            available_actions = self.action_decoder.get_available_actions(player_state)
            if not available_actions:
                # If no actions are available, just pass.
                await game_instance.send_player_input(player_id, self.action_decoder._create_pass_action())
                # await asyncio.sleep(2)  # Slow down agent: wait 10 seconds after move
                return
                
            random.shuffle(available_actions)
            
            for i in range(min(3, len(available_actions))):
                random_action_idx = available_actions[i]
                random_action = self.action_decoder.decode_action(random_action_idx, player_state)
                
                if random_action:
                    if await game_instance.send_player_input(player_id, random_action):
                        logger.info(f"Random action succeeded for agent {self.id[:8]}.")
                        # await asyncio.sleep(2)  # Slow down agent: wait 10 seconds after move
                        return  # Success

            logger.warning(f"All random actions failed for agent {self.id[:8]}. Passing.")

            # 3. If all else fails, pass
            await game_instance.send_player_input(player_id, self.action_decoder._create_pass_action())
            # await asyncio.sleep(2)  # Slow down agent: wait 10 seconds after move

        except Exception as e:
            logger.error(f"Error making move for agent {self.id[:8]}: {e}", exc_info=True)
    
    async def _get_action_from_network(self, state_vector: np.ndarray, 
                                    player_state: Dict[str, Any], force_random: bool = False) -> Optional[Dict[str, Any]]:
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
            
            if not available_actions:
                return None
                
            # Log available action types for debugging
            action_types = []
            for action_idx in available_actions:
                if action_idx < 100:
                    action_types.append(f"PLAY_CARD({action_idx})")
                elif action_idx < 200:
                    action_types.append(f"STANDARD_PROJECT({action_idx-100})")
                elif action_idx >= 200 and action_idx < 210:  # SELECT_OPTION range
                    option_idx = action_idx - 200
                    option_names = ["CARD_ACTION", "PLAY_PROJECT_CARD", "FUND_AWARD", "STANDARD_PROJECTS", "PASS", "SELL_PATENTS"]
                    if option_idx < len(option_names):
                        action_types.append(f"SELECT_OPTION_{option_names[option_idx]}({action_idx})")
                    else:
                        action_types.append(f"SELECT_OPTION_{option_idx}({action_idx})")
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
            
            logger.debug(f"Available actions: {action_types}")
            
            # Optional: adjust weights for OR menus based on option titles to avoid passing
            action_weight_adjustments = None
            waiting_for = player_state.get('waitingFor', {})
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
                        adjustments[idx] = 1.5
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
            action_index = self._sample_action(
                policy_probs.squeeze(),
                available_actions,
                force_random=force_random,
                action_weight_adjustments=action_weight_adjustments,
            )
            
            # Convert to game input
            action_input = self.action_decoder.decode_action(action_index, player_state)
            
            # Additional safety check: if this is a card play action and we have no money, prefer pass
            if action_input and action_input.get('type') in ['card', 'projectCard']:
                player = player_state.get('thisPlayer', {})
                player_mc = player.get('megaCredits', 0)
                if player_mc <= 0:
                    logger.info(f"Agent has no money ({player_mc} MC), preferring pass over card play")
                    # Try to find a pass action
                    pass_actions = [a for a in available_actions if a >= 900]  # PASS action base
                    if pass_actions:
                        pass_action_index = pass_actions[0]
                        action_input = self.action_decoder.decode_action(pass_action_index, player_state)
            
            return action_input
            
        except Exception as e:
            logger.error(f"Error getting action from network: {e}")
            return None
    
    def _sample_action(self, policy_probs: torch.Tensor, available_actions: List[int], force_random: bool = False, action_weight_adjustments: Optional[Dict[int, float]] = None) -> int:
        """Sample action from policy, restricted to available actions"""
        # Reduce epsilon-greedy for more policy-driven behavior
        if force_random or np.random.random() < max(0.05, self.config.epsilon * 0.5):
            return np.random.choice(available_actions)

        # Mask unavailable actions
        masked_probs = torch.zeros_like(policy_probs)
        for action_idx in available_actions:
            if action_idx < len(policy_probs):
                masked_probs[action_idx] = policy_probs[action_idx]
        
        # Add small epsilon to prevent zero probabilities
        epsilon = 1e-8
        masked_probs += epsilon
        
        # Renormalize
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback to uniform if all probabilities are zero
            for action_idx in available_actions:
                if action_idx < len(masked_probs):
                    masked_probs[action_idx] = 1.0
            masked_probs /= masked_probs.sum()

        # Prefer diverse actions - reduce probability of repetitive actions
        pass_action_base = 900  # From action_types['PASS']
        sell_patents_action = 702  # Sell patents action
        select_option_base = 200  # From action_types['SELECT_OPTION']
        
        for i, action_idx in enumerate(available_actions):
            if action_idx >= pass_action_base:
                masked_probs[action_idx] *= 0.3  # Reduce pass action probability
            elif action_idx == sell_patents_action:
                masked_probs[action_idx] *= 0.5  # Reduce sell patents probability to encourage diversity
            elif action_idx >= 100 and action_idx < 200:  # Standard projects
                masked_probs[action_idx] *= 1.2  # Increase standard project probability
            elif action_idx == 700:  # Convert plants
                masked_probs[action_idx] *= 1.3  # Increase convert plants probability
            elif action_idx == 701:  # Convert heat
                masked_probs[action_idx] *= 1.3  # Increase convert heat probability
            elif action_idx >= select_option_base and action_idx < select_option_base + 10:  # SELECT_OPTION range
                # Encourage diverse option selection, but avoid always picking the same option
                option_idx = action_idx - select_option_base
                if option_idx == 5:  # Sell patents option (index 5 in the OR structure)
                    masked_probs[action_idx] *= 0.4  # Strongly reduce sell patents option
                elif option_idx == 4:  # Pass option (index 4 in the OR structure)
                    masked_probs[action_idx] *= 0.6  # Reduce pass option
                elif option_idx == 3:  # Standard projects option (index 3 in the OR structure)
                    masked_probs[action_idx] *= 1.3  # Strongly encourage standard projects
                elif option_idx == 2:  # Fund award option (index 2 in the OR structure)
                    masked_probs[action_idx] *= 1.2  # Encourage award funding
                elif option_idx == 1:  # Play project card option (index 1 in the OR structure)
                    masked_probs[action_idx] *= 1.6  # Encourage playing project cards
                elif option_idx == 0:  # Play card action option (index 0 in the OR structure)
                    masked_probs[action_idx] *= 1.5  # Encourage playing card actions
        
        # Apply contextual adjustments (e.g., OR menu titles)
        if action_weight_adjustments:
            for action_idx, mult in action_weight_adjustments.items():
                if action_idx < len(masked_probs):
                    masked_probs[action_idx] *= float(mult)

        # Renormalize after adjustment
        masked_probs = masked_probs / masked_probs.sum()

        # Sample from policy
        try:
            return torch.multinomial(masked_probs, 1).item()
        except RuntimeError:
            # Fallback to non-pass action if possible
            non_pass_actions = [a for a in available_actions if a < pass_action_base]
            if non_pass_actions:
                return np.random.choice(non_pass_actions)
            return np.random.choice(available_actions)
    
    async def _record_game_result(self, final_state: Dict[str, Any], player_name: str, game_instance: GameInstance):
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
        
        except Exception as e:
            logger.error(f"Error recording game result: {e}")
    
    def get_fitness_score(self) -> float:
        """Calculate fitness score for evolutionary selection"""
        if self.games_played == 0:
            return 0.0
        
        avg_vp = self.total_victory_points / self.games_played
        win_rate = self.wins / self.games_played
        
        # Fitness combines average VP and win rate
        fitness = avg_vp * 0.7 + win_rate * 100 * 0.3
        return fitness
    
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
        checkpoint = torch.load(path, map_location='cpu')
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games_played = checkpoint.get('games_played', 0)
        self.total_victory_points = checkpoint.get('total_victory_points', 0)
        self.wins = checkpoint.get('wins', 0)
    
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
    

    