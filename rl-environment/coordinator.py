"""
RL Coordinator - Manages tournaments, agent evolution, and training
"""
import sys
import os
import glob
import uuid
import shutil

import asyncio
import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import json
import random

from tournament_manager import TournamentManager
from agent_evolution import EvolutionManager
from game_interface import GameServerCluster
from models.agent import RLAgent
from metrics.tracker import MetricsTracker
from api.server import start_api_server
from logging_setup import setup_logging
from scoring import calculate_selection_score

# Configure logging with rotation (env toggles)
LOG_DIR = os.getenv('RL_LOG_DIR', '/app/rl-logs')
LOG_MAX_MB = int(os.getenv('RL_LOG_MAX_MB', '5'))
LOG_BACKUPS = int(os.getenv('RL_LOG_BACKUPS', '5'))
LOG_LEVEL_NAME = str(os.getenv("LOG_LEVEL", "INFO")).strip().upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
setup_logging(
    log_dir=LOG_DIR,
    filename='rl-coordinator.log',
    max_bytes=LOG_MAX_MB * 1024 * 1024,
    backup_count=LOG_BACKUPS,
    level=LOG_LEVEL,
)
logger = logging.getLogger(__name__)

# Optional override for very chatty per-move loggers.
AGENT_LOG_LEVEL_NAME = str(os.getenv("AGENT_LOG_LEVEL", "")).strip().upper()
if AGENT_LOG_LEVEL_NAME:
    agent_level = getattr(logging, AGENT_LOG_LEVEL_NAME, None)
    if isinstance(agent_level, int):
        logging.getLogger("models.agent").setLevel(agent_level)
        logging.getLogger("models.action_decoder").setLevel(agent_level)
        logger.info("Applied AGENT_LOG_LEVEL=%s to models.agent and models.action_decoder", AGENT_LOG_LEVEL_NAME)
    else:
        logger.warning("Invalid AGENT_LOG_LEVEL=%s. Expected Python logging level name.", AGENT_LOG_LEVEL_NAME)

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
        # Keep background tasks alive (e.g., human-vs-AI matches started from dashboard)
        self.background_tasks: Set[asyncio.Task] = set()
        # Pause functionality
        self.paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Initially not paused
        # Crash-safe resume settings
        self.resume_training_enabled = os.getenv("RESUME_TRAINING", "1") != "0"
        self.checkpoint_root = os.getenv(
            "RL_CHECKPOINT_DIR",
            os.path.join(self._default_models_root(), "checkpoints"),
        )
        self.checkpoint_state_path = os.path.join(self.checkpoint_root, "state.json")
        self.checkpoint_population_dir = os.path.join(self.checkpoint_root, "population")

    def _resolve_global_game_concurrency(self) -> int:
        configured = str(os.getenv("GLOBAL_GAME_CONCURRENCY", "0")).strip()
        if configured:
            try:
                configured_value = int(configured)
                if configured_value > 0:
                    return configured_value
            except Exception:
                pass

        try:
            per_server = int(os.getenv("GLOBAL_GAME_CONCURRENCY_PER_SERVER", "4"))
        except Exception:
            per_server = 4
        per_server = max(1, per_server)
        return max(1, len(self.config.game_servers) * per_server)
        
    async def initialize(self):
        """Initialize the RL system"""
        logger.info("Initializing RL Coordinator...")
        
        # Initialize database
        await self.metrics_tracker.initialize()
        
        # Test game server connections
        await self.game_cluster.health_check()

        resumed = False
        if self.resume_training_enabled:
            resumed = self._load_training_checkpoint()
            if not resumed:
                resumed = self._load_population_from_latest_saved_generation()

        if resumed:
            if len(self.population) > self.config.population_size:
                logger.warning(
                    "Checkpoint population (%d) larger than configured size (%d). Trimming.",
                    len(self.population),
                    self.config.population_size,
                )
                self.population = self.population[:self.config.population_size]
            elif len(self.population) < self.config.population_size:
                missing = self.config.population_size - len(self.population)
                logger.warning(
                    "Checkpoint population (%d) smaller than configured size (%d). Adding %d fresh agents.",
                    len(self.population),
                    self.config.population_size,
                    missing,
                )
                self.population.extend(
                    await self.evolution_manager.create_initial_population(missing)
                )
            logger.info(
                "Resumed training at generation %d with %d agents",
                self.current_generation,
                len(self.population),
            )
            return

        # No checkpoint found or resume disabled: start fresh.
        self.current_generation = 0
        self.population = await self.evolution_manager.create_initial_population(
            self.config.population_size
        )
        logger.info(f"Created initial population of {len(self.population)} agents")
        self._save_training_checkpoint(next_generation=0)
    
    async def run_evolution_cycle(self):
        """Main evolution loop"""
        logger.info("Starting evolution cycle...")

        generation = max(0, int(self.current_generation))
        if generation >= self.config.generations:
            logger.info(
                "Checkpoint generation %d is already at/above configured GENERATIONS=%d. Nothing to run.",
                generation,
                self.config.generations,
            )
            return

        while generation < self.config.generations:
            # Check for pause
            await self.pause_event.wait()

            self.current_generation = generation
            logger.info(f"=== Generation {generation + 1}/{self.config.generations} ===")

            try:
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

                next_generation = generation + 1
                self._save_training_checkpoint(next_generation=next_generation)
                self.current_generation = next_generation
                logger.info(
                    f"Generation {generation + 1} complete. Best fitness: {max(fitness_scores):.2f}"
                )
                generation = next_generation
            except Exception:
                logger.exception(
                    "Generation %d failed. Keeping current population and retrying after backoff.",
                    generation + 1,
                )
                await asyncio.sleep(5)
    
    async def evaluate_population(self) -> List[float]:
        """Evaluate entire population through tournaments"""
        logger.info(f"Evaluating population of {len(self.population)} agents...")
        
        # Create tournament brackets
        tournaments = self.tournament_manager.create_tournaments(
            self.population, 
            self.config.tournament_size,
            self.config.games_per_evaluation
        )
        
        # Run tournaments in parallel with a global cap on in-flight games.
        global_game_limit = self._resolve_global_game_concurrency()
        global_game_semaphore = asyncio.Semaphore(global_game_limit)
        logger.info(
            "Evaluation concurrency: GLOBAL_GAME_CONCURRENCY=%d, TOURNAMENT_CONCURRENCY=%s",
            global_game_limit,
            os.getenv("TOURNAMENT_CONCURRENCY", "3"),
        )
        all_results = await asyncio.gather(*[
            self.tournament_manager.run_tournament(
                tournament,
                global_game_semaphore=global_game_semaphore,
            )
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

                    total_score = calculate_selection_score(
                        rank=player_result.get('rank', 4),
                        victory_points=player_result.get('victory_points', 0),
                        completed=player_result.get('completed', False),
                    )
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

    @staticmethod
    def _checkpoint_agent_index(path: str) -> int:
        try:
            base = os.path.basename(path)
            return int(base.replace("agent_", "").replace(".pth", ""))
        except Exception:
            return sys.maxsize

    def _save_training_checkpoint(self, next_generation: int):
        """Persist full population so crashes can resume from the last completed generation."""
        if not self.resume_training_enabled:
            return

        os.makedirs(self.checkpoint_root, exist_ok=True)
        tmp_population_dir = os.path.join(
            self.checkpoint_root,
            f"population.tmp.{uuid.uuid4().hex}",
        )
        tmp_state_path = os.path.join(self.checkpoint_root, "state.tmp.json")
        previous_population_dir = os.path.join(self.checkpoint_root, "population.prev")

        try:
            os.makedirs(tmp_population_dir, exist_ok=True)
            for idx, agent in enumerate(self.population):
                agent.save_model(os.path.join(tmp_population_dir, f"agent_{idx}.pth"))

            state_payload = {
                "version": 1,
                "next_generation": int(next_generation),
                "population_size": len(self.population),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            with open(tmp_state_path, "w", encoding="utf-8") as f:
                json.dump(state_payload, f, indent=2)

            if os.path.isdir(previous_population_dir):
                shutil.rmtree(previous_population_dir, ignore_errors=True)
            if os.path.isdir(self.checkpoint_population_dir):
                os.replace(self.checkpoint_population_dir, previous_population_dir)
            os.replace(tmp_population_dir, self.checkpoint_population_dir)
            os.replace(tmp_state_path, self.checkpoint_state_path)
            if os.path.isdir(previous_population_dir):
                shutil.rmtree(previous_population_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed saving training checkpoint: {e}")
            try:
                if os.path.isdir(tmp_population_dir):
                    shutil.rmtree(tmp_population_dir, ignore_errors=True)
                if os.path.exists(tmp_state_path):
                    os.remove(tmp_state_path)
            except Exception:
                pass

    def _load_training_checkpoint(self) -> bool:
        """Load full population + next generation pointer from disk."""
        if not os.path.exists(self.checkpoint_state_path):
            return False

        try:
            with open(self.checkpoint_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            next_generation = int(state.get("next_generation", 0))
        except Exception as e:
            logger.warning(f"Failed loading checkpoint state file: {e}")
            return False

        model_paths = sorted(
            glob.glob(os.path.join(self.checkpoint_population_dir, "agent_*.pth")),
            key=self._checkpoint_agent_index,
        )
        if not model_paths:
            logger.warning(
                "Checkpoint state exists but no population models found in %s",
                self.checkpoint_population_dir,
            )
            return False

        loaded_population: List[RLAgent] = []
        for model_path in model_paths:
            try:
                agent = RLAgent()
                agent.load_model(model_path)
                loaded_population.append(agent)
            except Exception as e:
                logger.warning(f"Failed loading checkpoint model {model_path}: {e}")

        if not loaded_population:
            return False

        self.population = loaded_population
        self.current_generation = max(0, min(next_generation, self.config.generations))
        return True

    def _load_population_from_latest_saved_generation(self) -> bool:
        """Fallback resume path: bootstrap population from latest generation_* saves."""
        try:
            models_root = self._default_models_root()
            generations = self._available_generations(models_root)
            if not generations:
                return False
            latest_generation = generations[-1]
            checkpoints = self._all_generation_checkpoints(models_root, latest_generation)
            if not checkpoints:
                return False

            loaded_population: List[RLAgent] = []
            for checkpoint in checkpoints[: self.config.population_size]:
                agent = RLAgent()
                agent.load_model(checkpoint)
                loaded_population.append(agent)

            if not loaded_population:
                return False

            self.population = loaded_population
            self.current_generation = max(
                0,
                min(int(latest_generation) + 1, self.config.generations),
            )
            logger.info(
                "Bootstrapped resume from generation_%d using %d saved agents",
                latest_generation,
                len(loaded_population),
            )
            return True
        except Exception as e:
            logger.warning(f"Failed fallback resume from generation_* saves: {e}")
            return False

    async def run_debug_game(self, agent_ids: List[str] = None) -> Dict[str, Any]:
        """Run a single debug game with 4 agents for easy debugging"""
        logger.info("Starting debug game...")
        
        # Use provided agents or select first 4 from population
        if agent_ids:
            selected_agents = [agent for agent in self.population if agent.id in agent_ids]
            if len(selected_agents) < 4:
                # Pad with random agents if needed
                while len(selected_agents) < 4:
                    selected_agents.append(random.choice(self.population))
        else:
            # Use first 4 agents from population
            selected_agents = self.population[:4]
            if len(selected_agents) < 4:
                logger.error(f"Not enough agents in population ({len(self.population)}). Need at least 4.")
                return {"error": "Not enough agents in population"}
        
        # Limit to exactly 4 agents
        selected_agents = selected_agents[:4]
        
        logger.info(f"Running debug game with agents: {[agent.id[:8] for agent in selected_agents]}")
        
        try:
            # Create a single game
            game_result = await self.tournament_manager._run_single_game(selected_agents, "debug_game")
            
            # Record the game URL
            try:
                public_base = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
                game_url = f"{public_base}/game?id={game_result.game_id}"
                self.recent_games.append({"game_id": game_result.game_id, "url": game_url})
            except Exception as e:
                logger.warning(f"Failed to record game URL: {e}")
            
            return {
                "success": True,
                "game_id": game_result.game_id,
                "game_url": game_url if 'game_url' in locals() else None,
                "agents": [agent.id[:8] for agent in selected_agents],
                "result": self.tournament_manager._game_result_to_dict(game_result)
            }
            
        except Exception as e:
            logger.error(f"Debug game failed: {e}")
            return {"error": str(e)}

    def _track_background_task(self, task: asyncio.Task):
        """Track fire-and-forget task lifecycle to avoid premature GC and surface errors."""
        self.background_tasks.add(task)

        def _on_done(done_task: asyncio.Task):
            self.background_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception as e:
                logger.error(f"Background task failed: {e}")

        task.add_done_callback(_on_done)

    def _default_models_root(self) -> str:
        env_path = os.getenv("RL_MODELS_DIR")
        if env_path:
            return env_path
        candidates = [
            "/app/rl-models",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "rl-models")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl-models")),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return candidates[0]

    @staticmethod
    def _checkpoint_fitness_from_name(path: str) -> float:
        base = os.path.basename(path)
        try:
            return float(base.split("_fitness_")[-1].replace(".pth", ""))
        except Exception:
            return -1.0

    def _available_generations(self, models_root: str) -> List[int]:
        generations: List[int] = []
        for path in glob.glob(os.path.join(models_root, "generation_*")):
            base = os.path.basename(path)
            try:
                generations.append(int(base.split("_", 1)[1]))
            except Exception:
                continue
        return sorted(set(generations))

    def _all_generation_checkpoints(self, models_root: str, generation: int) -> List[str]:
        pattern = os.path.join(models_root, f"generation_{generation}", "agent_*_fitness_*.pth")
        return sorted(glob.glob(pattern), key=self._checkpoint_fitness_from_name, reverse=True)

    def _checkpoint_for_agent_index(self, models_root: str, generation: int, agent_index: int) -> Optional[str]:
        pattern = os.path.join(models_root, f"generation_{generation}", f"agent_{int(agent_index)}_fitness_*.pth")
        matches = sorted(glob.glob(pattern), key=self._checkpoint_fitness_from_name, reverse=True)
        return matches[0] if matches else None

    async def _monitor_human_vs_ai_game(self, game_instance: Any, bot_agents: List[RLAgent], bot_names: List[str]):
        """Run bot tasks to completion while a human plays, then cleanup and collect links."""
        try:
            bot_tasks = [
                asyncio.create_task(agent.play_game(game_instance, bot_name))
                for agent, bot_name in zip(bot_agents, bot_names)
            ]
            await asyncio.gather(*bot_tasks, return_exceptions=True)
            final_state = await game_instance.get_final_state()
            end_screens: List[str] = []
            for player in final_state.get('players', []):
                player_id = player.get('id')
                if player_id:
                    public_base = game_instance.get_public_game_url().split("/game?id=", 1)[0]
                    end_screens.append(f"{public_base}/the-end?id={player_id}")
            if end_screens:
                self.recent_end_screens.append({"game_id": game_instance.game_id, "end_screens": end_screens})
        except Exception as e:
            logger.warning(f"Human-vs-AI monitor task ended with error: {e}")
        finally:
            try:
                await game_instance.cleanup()
            except Exception:
                pass

    async def run_human_vs_generation_game(
        self,
        generation: Optional[int] = None,
        random_generation: bool = True,
        human_name: str = "You",
        bot_count: int = 3,
        agent_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a game with one human and bots loaded from saved checkpoints."""
        models_root = self._default_models_root()
        rng = random.Random(seed)

        if generation is None:
            if not random_generation:
                return {"error": "generation must be provided when random_generation is false"}
            generations = self._available_generations(models_root)
            if not generations:
                return {"error": f"No generations found under {models_root}"}
            generation = rng.choice(generations)

        generation = int(generation)
        all_checkpoints = self._all_generation_checkpoints(models_root, generation)
        if not all_checkpoints:
            return {"error": f"No checkpoints found for generation {generation} under {models_root}"}

        selected_checkpoints: List[str] = []
        if agent_indices:
            for idx in agent_indices:
                ckpt = self._checkpoint_for_agent_index(models_root, generation, int(idx))
                if ckpt is None:
                    return {
                        "error": f"No checkpoint found for generation {generation}, agent index {idx}"
                    }
                selected_checkpoints.append(ckpt)
        else:
            take = min(bot_count, len(all_checkpoints))
            selected_checkpoints.extend(rng.sample(all_checkpoints, k=take))

        while len(selected_checkpoints) < bot_count:
            selected_checkpoints.append(rng.choice(all_checkpoints))
        selected_checkpoints = selected_checkpoints[:bot_count]

        bot_agents: List[RLAgent] = []
        for ckpt in selected_checkpoints:
            agent = RLAgent()
            agent.load_model(ckpt)
            # Dashboard test games should be inference-only.
            agent.train_from_self_play = False
            bot_agents.append(agent)

        bot_names = [f"AI_{agent.id[:8]}" for agent in bot_agents]
        player_names = [human_name] + bot_names
        game_instance = await self.game_cluster.create_game(
            game_id=str(uuid.uuid4()),
            player_names=player_names,
            game_options={
                'soloMode': False,
                'fastModeOption': False,
                'removeNegativeGlobalEventsOption': True,
                'undoOption': False,
            }
        )
        human_player_id = await game_instance.join_player(human_name)
        game_url = game_instance.get_public_game_url()
        player_url = game_instance.get_public_player_url(human_player_id)
        self.recent_games.append({"game_id": game_instance.game_id, "url": game_url})

        monitor_task = asyncio.create_task(self._monitor_human_vs_ai_game(game_instance, bot_agents, bot_names))
        self._track_background_task(monitor_task)

        return {
            "success": True,
            "game_id": game_instance.game_id,
            "generation": generation,
            "game_url": game_url,
            "player_url": player_url,
            "human_name": human_name,
            "bot_names": bot_names,
            "checkpoints": selected_checkpoints,
        }
    
    def pause_training(self):
        """Pause the training process"""
        self.paused = True
        self.pause_event.clear()
        logger.info("Training paused")
    
    def resume_training(self):
        """Resume the training process"""
        self.paused = False
        self.pause_event.set()
        logger.info("Training resumed")
    
    def is_paused(self) -> bool:
        """Check if training is currently paused"""
        return self.paused

    async def shutdown(self):
        """Release coordinator resources for clean process shutdown."""
        # Stop background tasks launched from API endpoints.
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()
        if self.background_tasks:
            await asyncio.gather(*list(self.background_tasks), return_exceptions=True)
            self.background_tasks.clear()

        # Close shared aiohttp session to avoid unclosed session warnings.
        try:
            await self.game_cluster.close()
        except Exception as e:
            logger.warning(f"Failed to close game server cluster session: {e}")

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

    api_task: Optional[asyncio.Task] = None
    evolution_task: Optional[asyncio.Task] = None
    try:
        await coordinator.initialize()

        # Start API server and evolution loop concurrently.
        api_task = asyncio.create_task(start_api_server(coordinator))
        evolution_task = asyncio.create_task(coordinator.run_evolution_cycle())

        done, _pending = await asyncio.wait(
            {api_task, evolution_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if evolution_task in done:
            # If training completed (e.g., reached GENERATIONS), keep API alive.
            exc = evolution_task.exception()
            if exc is not None:
                raise exc
            logger.info("Evolution cycle finished. Keeping API server alive.")
            await api_task
        elif api_task in done:
            # API should not exit during normal operation.
            exc = api_task.exception()
            if exc is not None:
                raise exc
            raise RuntimeError("API server stopped unexpectedly")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Coordinator main loop failed: {e}")
        raise
    finally:
        for task in (evolution_task, api_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *[task for task in (evolution_task, api_task) if task is not None],
            return_exceptions=True,
        )
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
