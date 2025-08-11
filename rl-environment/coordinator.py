"""
RL Coordinator - Manages tournaments, agent evolution, and training
"""
import sys
import os

import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from tournament_manager import TournamentManager
from agent_evolution import EvolutionManager
from game_interface import GameServerCluster
from models.agent import RLAgent
from metrics.tracker import MetricsTracker
from api.server import start_api_server
from logging_setup import setup_logging

# Configure logging with rotation (env toggles)
LOG_DIR = os.getenv('RL_LOG_DIR', '/app/rl-logs')
LOG_MAX_MB = int(os.getenv('RL_LOG_MAX_MB', '5'))
LOG_BACKUPS = int(os.getenv('RL_LOG_BACKUPS', '5'))
setup_logging(
    log_dir=LOG_DIR,
    filename='rl-coordinator.log',
    max_bytes=LOG_MAX_MB * 1024 * 1024,
    backup_count=LOG_BACKUPS,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class CoordinatorConfig:
    game_servers: List[str]
    population_size: int = 16
    tournament_size: int = 16
    generations: int = 50
    games_per_evaluation: int = 1
    mutation_rate: float = 0.1
    elite_percentage: float = 0.2
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://postgres:password@localhost:5432/rl_metrics"

class RLCoordinator:
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.game_cluster = GameServerCluster(config.game_servers)
        self.tournament_manager = TournamentManager(self.game_cluster)
        self.evolution_manager = EvolutionManager(config)
        self.metrics_tracker = MetricsTracker(config.postgres_url)
        self.current_generation = 0
        self.population: List[RLAgent] = []
        # Last evaluation fitness (used for selection) per agent id
        self.last_eval_fitness: Dict[str, float] = {}
        # Track created games for dashboard linking
        self.recent_games: List[Dict[str, str]] = []  # {game_id, url}
        # Track recent end screens for dashboard
        self.recent_end_screens: List[Dict[str, List[str]]] = []  # {game_id, end_screens}
        
    async def initialize(self):
        """Initialize the RL system"""
        logger.info("Initializing RL Coordinator...")
        
        # Initialize database
        await self.metrics_tracker.initialize()
        
        # Test game server connections
        await self.game_cluster.health_check()
        
        # Create initial population
        self.population = await self.evolution_manager.create_initial_population(
            self.config.population_size
        )
        
        logger.info(f"Created initial population of {len(self.population)} agents")
    
    async def run_evolution_cycle(self):
        """Main evolution loop"""
        logger.info("Starting evolution cycle...")
        
        for generation in range(self.config.generations):
            self.current_generation = generation
            logger.info(f"=== Generation {generation + 1}/{self.config.generations} ===")
            
            # Evaluate current population
            fitness_scores = await self.evaluate_population()
            
            # Record metrics
            await self.metrics_tracker.record_generation(
                generation, self.population, fitness_scores
            )

            # Save best agents FROM the evaluated population (before evolving)
            await self.save_generation_models(generation, fitness_scores)

            # Evolve population for the next generation
            self.population = await self.evolution_manager.evolve_population(
                self.population, fitness_scores
            )
            
            logger.info(f"Generation {generation + 1} complete. Best fitness: {max(fitness_scores):.2f}")
    
    async def evaluate_population(self) -> List[float]:
        """Evaluate entire population through tournaments"""
        logger.info(f"Evaluating population of {len(self.population)} agents...")
        
        # Create tournament brackets
        tournaments = self.tournament_manager.create_tournaments(
            self.population, 
            self.config.tournament_size,
            self.config.games_per_evaluation
        )
        
        # Run tournaments in parallel
        all_results = await asyncio.gather(*[
            self.tournament_manager.run_tournament(tournament)
            for tournament in tournaments
        ])
        
        # Calculate fitness scores
        fitness_scores = self._calculate_fitness_scores(all_results)
        # Persist mapping for API/dashboard and saving configs
        try:
            self.last_eval_fitness = {
                agent.id: float(fitness_scores[i]) for i, agent in enumerate(self.population)
            }
        except Exception:
            self.last_eval_fitness = {}
        
        return fitness_scores
    
    def _calculate_fitness_scores(self, tournament_results: List[Dict]) -> List[float]:
        """Calculate fitness scores based on tournament performance"""
        agent_scores = {agent.id: 0.0 for agent in self.population}
        agent_games = {agent.id: 0 for agent in self.population}
        
        for tournament_result in tournament_results:
            for game_result in tournament_result['games']:
                for player_result in game_result['players']:
                    agent_id = player_result['agent_id']
                    
                    # Points based on ranking
                    ranking_points = [100, 75, 50, 25][player_result['rank'] - 1]
                    
                    # Victory points bonus
                    vp_bonus = player_result['victory_points'] * 0.5
                    
                    # Completion bonus (for not timing out/crashing)
                    completion_bonus = 10 if player_result['completed'] else -50
                    
                    total_score = ranking_points + vp_bonus + completion_bonus
                    agent_scores[agent_id] += total_score
                    agent_games[agent_id] += 1
        
        # Average scores and handle agents with no games
        fitness_scores = []
        for agent in self.population:
            if agent_games[agent.id] > 0:
                avg_score = agent_scores[agent.id] / agent_games[agent.id]
            else:
                avg_score = 0.0  # Penalty for not playing
            fitness_scores.append(avg_score)
        
        return fitness_scores
    
    async def save_generation_models(self, generation: int, fitness_scores: List[float]):
        """Save models from current generation"""
        save_dir = f"/app/rl-models/generation_{generation}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save top 10% of agents
        num_to_save = max(1, int(len(self.population) * 0.1))
        
        # Get fitness scores to find top agents
        top_indices = sorted(range(len(fitness_scores)), 
                           key=lambda i: fitness_scores[i], 
                           reverse=True)[:num_to_save]
        
        for i, idx in enumerate(top_indices):
            agent = self.population[idx]
            model_path = f"{save_dir}/agent_{i}_fitness_{fitness_scores[idx]:.2f}.pth"
            agent.save_model(model_path)
            
            # Save agent config
            config_path = f"{save_dir}/agent_{i}_config.json"
            cfg = agent.get_config()
            # Include evaluation fitness used for selection for clarity
            try:
                cfg['eval_fitness'] = float(fitness_scores[idx])
            except Exception:
                cfg['eval_fitness'] = None
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        logger.info(f"Saved top {num_to_save} agents from generation {generation}")

async def main():
    """Main entry point"""
    # Parse environment variables
    game_servers = os.getenv('GAME_SERVERS', 'localhost:8080').split(',')
    
    config = CoordinatorConfig(
        game_servers=game_servers,
        population_size=int(os.getenv('POPULATION_SIZE', '4')),
        tournament_size=int(os.getenv('TOURNAMENT_SIZE', '4')),
        generations=int(os.getenv('GENERATIONS', '1000')),
        games_per_evaluation=int(os.getenv('GAMES_PER_EVAL', '1')),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
        postgres_url=os.getenv('POSTGRES_URL', 'postgresql://postgres:password@localhost:5432/rl_metrics')
    )
    
    coordinator = RLCoordinator(config)
    
    # Start API server for monitoring
    api_task = asyncio.create_task(start_api_server(coordinator))
    
    try:
        # Initialize and run evolution
        await coordinator.initialize()
        await coordinator.run_evolution_cycle()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        api_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())