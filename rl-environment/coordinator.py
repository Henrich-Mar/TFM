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
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
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
from training.fitness import (
    PromotionGateConfig,
    apply_promotion_gates,
    calculate_selection_fitness,
    compute_generation_behavior_metrics,
    snapshot_population_behavior,
)
from training.ppo_cycle import optimize_population_with_ppo
from training.league import LeagueConfig, LeagueManager

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
        # Last evaluation fitness maps.
        # last_eval_fitness is the value used for selection (may include gating penalties).
        self.last_eval_fitness: Dict[str, float] = {}
        self.last_raw_eval_fitness: Dict[str, float] = {}
        self.last_gated_eval_fitness: Dict[str, float] = {}
        self.last_generation_behavior_metrics: Dict[str, Any] = {}
        self.last_generation_gate: Dict[str, Any] = {}
        self.generation_behavior_history: List[Dict[str, Any]] = []
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
        self.promotion_gate_enabled = str(os.getenv("PROMOTION_GATE_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.gate_min_card_plays_per_game = self._safe_env_float("GATE_MIN_CARD_PLAYS_PER_GAME", 0.60)
        self.gate_max_standard_project_ratio = self._safe_env_float("GATE_MAX_STANDARD_PROJECT_RATIO", 0.58)
        self.gate_min_steel_spent_per_game = self._safe_env_float("GATE_MIN_STEEL_SPENT_PER_GAME", 0.25)
        self.gate_min_titanium_spent_per_game = self._safe_env_float("GATE_MIN_TITANIUM_SPENT_PER_GAME", 0.12)
        self.gate_max_payment_reject_count = self._safe_env_int("GATE_MAX_PAYMENT_REJECT_COUNT", 0)
        self.gate_penalty_points = self._safe_env_float("GATE_PENALTY_POINTS", 8.0)
        self.gate_global_payment_penalty_points = self._safe_env_float("GATE_GLOBAL_PAYMENT_PENALTY_POINTS", 6.0)
        self.ppo_enable = str(os.getenv("PPO_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.ppo_rollout_steps = self._safe_env_int("PPO_ROLLOUT_STEPS", 8192)
        self.save_top_k = max(1, self._safe_env_int("SAVE_TOP_K", 3))
        self.training_opponent_pool_enabled = str(os.getenv("TRAINING_OPPONENT_POOL_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.training_pool_games_per_agent = max(0, self._safe_env_int("TRAINING_POOL_GAMES_PER_AGENT", 1))
        self.training_pool_generation_window = max(1, self._safe_env_int("TRAINING_POOL_GENERATION_WINDOW", 8))
        self.training_pool_max_checkpoints = max(3, self._safe_env_int("TRAINING_POOL_MAX_CHECKPOINTS", 48))
        self.training_pool_min_checkpoints = max(3, self._safe_env_int("TRAINING_POOL_MIN_CHECKPOINTS", 3))
        self.fixed_benchmark_enabled = str(os.getenv("FIXED_BENCHMARK_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.fixed_benchmark_interval = max(1, self._safe_env_int("FIXED_BENCHMARK_INTERVAL", 5))
        self.fixed_benchmark_games_per_agent = max(1, self._safe_env_int("FIXED_BENCHMARK_GAMES_PER_AGENT", 2))
        self.fixed_benchmark_agent_count = max(1, self._safe_env_int("FIXED_BENCHMARK_AGENT_COUNT", 1))
        self.fixed_benchmark_pool_size = max(3, self._safe_env_int("FIXED_BENCHMARK_POOL_SIZE", 12))
        self.fixed_benchmark_checkpoints: List[str] = self._resolve_fixed_benchmark_checkpoints()
        self.league_manager = LeagueManager(
            LeagueConfig(
                enabled=str(os.getenv("LEAGUE_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off"),
                historical_ratio=self._safe_env_float("LEAGUE_HISTORICAL_RATIO", 0.4),
                exploiter_ratio=self._safe_env_float("LEAGUE_EXPLOITER_RATIO", 0.2),
                snapshot_interval=self._safe_env_int("LEAGUE_SNAPSHOT_INTERVAL", 5),
            )
        )

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
            self.evolution_manager.generation_count = int(self.current_generation)
            return

        # No checkpoint found or resume disabled: start fresh.
        self.current_generation = 0
        self.evolution_manager.generation_count = 0
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
                raw_fitness_scores = [
                    float(self.last_raw_eval_fitness.get(agent.id, fitness_scores[i]))
                    for i, agent in enumerate(self.population)
                ]
                benchmark_metrics = await self._maybe_run_fixed_benchmark(
                    generation=int(generation),
                    fitness_scores=fitness_scores,
                )
                self.last_generation_behavior_metrics.update(benchmark_metrics)

                # Record metrics
                await self.metrics_tracker.record_generation(
                    generation,
                    self.population,
                    fitness_scores,
                    raw_fitness_scores=raw_fitness_scores,
                    generation_metrics=self.last_generation_behavior_metrics,
                    gating_summary=self.last_generation_gate,
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
        before_behavior = self._snapshot_population_behavior()
        payment_reject_before = int(getattr(self.game_cluster, "payment_reject_count", 0))

        matchmaking_population = self.league_manager.order_population_for_matchmaking(
            self.population,
            generation=int(self.current_generation),
        )
        matchmaking_ordering_applied = [
            agent.id for agent in matchmaking_population
        ] != [agent.id for agent in self.population]
         
        # Create tournament brackets
        tournaments = self.tournament_manager.create_tournaments(
            matchmaking_population,
            self.config.tournament_size,
            self.config.games_per_evaluation,
            shuffle_population=not matchmaking_ordering_applied,
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
        training_pool_results, training_pool_metrics = await self._evaluate_with_training_opponent_pool(
            matchmaking_population,
            global_game_semaphore=global_game_semaphore,
        )
        if training_pool_results:
            all_results.extend(training_pool_results)
        
        # Update Elo ratings based on all played games
        self._update_elo_ratings(all_results)
        
        # Evaluate selection fitness from tournament outcomes.
        raw_fitness_scores = self._calculate_fitness_scores(all_results)

        # RL-first: optimize policies from collected rollouts after rollout collection.
        ppo_metrics: Dict[str, Any] = {}
        if self.ppo_enable:
            ppo_metrics = await optimize_population_with_ppo(
                population=self.population,
                target_rollout_steps=int(self.ppo_rollout_steps),
            )

        after_behavior = self._snapshot_population_behavior()
        payment_reject_after = int(getattr(self.game_cluster, "payment_reject_count", 0))

        generation_metrics, per_agent_behavior = self._compute_generation_behavior_metrics(
            before_behavior,
            after_behavior,
            payment_reject_before,
            payment_reject_after,
            all_results,
        )
        generation_metrics["league/matchmaking_ordering_applied"] = bool(matchmaking_ordering_applied)
        generation_metrics.update(training_pool_metrics)
        selection_fitness_scores, gate_summary, per_agent_gate = self._apply_promotion_gates(
            raw_fitness_scores,
            per_agent_behavior,
            generation_metrics,
        )
        league_metrics = self.league_manager.update_generation(
            generation=int(self.current_generation),
            population=self.population,
            fitness_scores=selection_fitness_scores,
        )
        generation_metrics.update(ppo_metrics)
        generation_metrics.update(league_metrics)

        try:
            self.last_raw_eval_fitness = {
                agent.id: float(raw_fitness_scores[i]) for i, agent in enumerate(self.population)
            }
            self.last_eval_fitness = {
                agent.id: float(selection_fitness_scores[i]) for i, agent in enumerate(self.population)
            }
            self.last_gated_eval_fitness = dict(self.last_eval_fitness)
        except Exception:
            self.last_raw_eval_fitness = {}
            self.last_eval_fitness = {}
            self.last_gated_eval_fitness = {}

        self.last_generation_behavior_metrics = generation_metrics
        self.last_generation_gate = {
            **gate_summary,
            "per_agent": per_agent_gate,
        }
        self.generation_behavior_history.append({
            "generation": int(self.current_generation),
            "metrics": generation_metrics,
            "gate": gate_summary,
        })
        if len(self.generation_behavior_history) > 200:
            self.generation_behavior_history = self.generation_behavior_history[-200:]

        logger.info(
            "Generation %d behavior: card_plays_per_game=%.3f standard_project_ratio=%.3f steel_spent=%d titanium_spent=%d payment_reject_count=%d ppo_steps=%d",
            int(self.current_generation),
            float(generation_metrics.get("card_plays_per_game", 0.0)),
            float(generation_metrics.get("standard_project_ratio", 0.0)),
            int(generation_metrics.get("steel_spent", 0)),
            int(generation_metrics.get("titanium_spent", 0)),
            int(generation_metrics.get("payment_reject_count", 0)),
            int(generation_metrics.get("rollout/steps_collected", 0)),
        )

        return selection_fitness_scores

    def _snapshot_population_behavior(self) -> Dict[str, Dict[str, float]]:
        return snapshot_population_behavior(self.population)

    def _compute_generation_behavior_metrics(
        self,
        before: Dict[str, Dict[str, float]],
        after: Dict[str, Dict[str, float]],
        payment_reject_before: int,
        payment_reject_after: int,
        tournament_results: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        return compute_generation_behavior_metrics(
            population=self.population,
            before=before,
            after=after,
            payment_reject_before=payment_reject_before,
            payment_reject_after=payment_reject_after,
            current_generation=int(self.current_generation),
            game_cluster=self.game_cluster,
            tournament_results=tournament_results,
        )

    def _apply_promotion_gates(
        self,
        raw_scores: List[float],
        per_agent_behavior: Dict[str, Dict[str, float]],
        generation_metrics: Dict[str, Any],
    ) -> Tuple[List[float], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        return apply_promotion_gates(
            population=self.population,
            raw_scores=raw_scores,
            per_agent_behavior=per_agent_behavior,
            generation_metrics=generation_metrics,
            gate_config=PromotionGateConfig(
                enabled=bool(self.promotion_gate_enabled),
                min_card_plays_per_game=float(self.gate_min_card_plays_per_game),
                max_standard_project_ratio=float(self.gate_max_standard_project_ratio),
                min_steel_spent_per_game=float(self.gate_min_steel_spent_per_game),
                min_titanium_spent_per_game=float(self.gate_min_titanium_spent_per_game),
                max_payment_reject_count=int(self.gate_max_payment_reject_count),
                penalty_points=float(self.gate_penalty_points),
                global_payment_penalty_points=float(self.gate_global_payment_penalty_points),
            ),
        )
    
    def _update_elo_ratings(self, tournament_results: List[Dict[str, Any]]):
        """Update Elo ratings based on tournament games."""
        for result in tournament_results:
            for game in result.get("games", []) or []:
                if not game.get("completed"):
                    continue
                players = game.get("players", []) or []
                if len(players) < 2:
                    continue
                
                # Pairwise updates for all players in the game
                for i in range(len(players)):
                    p1 = players[i]
                    id1 = p1.get("agent_id")
                    if not id1: continue
                    
                    for j in range(i + 1, len(players)):
                        p2 = players[j]
                        id2 = p2.get("agent_id")
                        if not id2: continue
                        
                        rank1 = int(p1.get("rank", 4) or 4)
                        rank2 = int(p2.get("rank", 4) or 4)
                        
                        # Lower rank is better (1st place < 2nd place)
                        if rank1 < rank2:
                            score1 = 1.0
                        elif rank1 > rank2:
                            score1 = 0.0
                        else:
                            score1 = 0.5
                        score2 = 1.0 - score1
                        
                        self.metrics_tracker.update_elo(id1, id2, score1, score2)

    def _calculate_fitness_scores(self, tournament_results: List[Dict]) -> List[float]:
        """Calculate fitness scores based on tournament performance"""
        return calculate_selection_fitness(self.population, tournament_results)

    @staticmethod
    def _dedupe_existing_checkpoints(paths: List[str]) -> List[str]:
        deduped: List[str] = []
        seen: Set[str] = set()
        for path in paths:
            normalized = os.path.abspath(str(path))
            if normalized in seen or not os.path.isfile(normalized):
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _resolve_fixed_benchmark_checkpoints(self) -> List[str]:
        models_root = self._default_models_root()
        explicit_raw = str(os.getenv("FIXED_BENCHMARK_CHECKPOINTS", "") or "").strip()
        explicit_refs = [token.strip() for token in explicit_raw.split(",") if token.strip()]
        explicit_paths: List[str] = []
        for ref in explicit_refs:
            candidate = ref
            if not os.path.isabs(candidate):
                candidate = os.path.join(models_root, candidate)
            candidate = os.path.abspath(candidate)
            if os.path.isfile(candidate):
                explicit_paths.append(candidate)
            else:
                logger.warning("Ignoring missing FIXED_BENCHMARK_CHECKPOINTS entry: %s", ref)
        if explicit_paths:
            return self._dedupe_existing_checkpoints(explicit_paths)[: int(self.fixed_benchmark_pool_size)]

        generation_refs = [token.strip() for token in str(os.getenv("FIXED_BENCHMARK_GENERATIONS", "") or "").split(",") if token.strip()]
        checkpoint_candidates: List[str] = []
        if generation_refs:
            for token in generation_refs:
                try:
                    generation = int(token)
                except Exception:
                    continue
                checkpoint_candidates.extend(
                    self._all_generation_checkpoints(models_root, generation)[: max(1, int(self.save_top_k))]
                )
        else:
            checkpoint_candidates.extend(self._all_saved_checkpoints(models_root))

        return self._dedupe_existing_checkpoints(checkpoint_candidates)[: int(self.fixed_benchmark_pool_size)]

    def _resolve_training_opponent_checkpoints(self) -> List[str]:
        models_root = self._default_models_root()
        generations = self._available_generations(models_root)
        if not generations:
            return []

        window = max(1, int(self.training_pool_generation_window))
        candidate_generations = generations[-window:]
        checkpoint_candidates: List[str] = []
        top_per_generation = max(1, int(self.save_top_k))
        for generation in reversed(candidate_generations):
            checkpoint_candidates.extend(
                self._all_generation_checkpoints(models_root, generation)[:top_per_generation]
            )
        max_pool = max(3, int(self.training_pool_max_checkpoints))
        return self._dedupe_existing_checkpoints(checkpoint_candidates)[:max_pool]

    @staticmethod
    def _sample_checkpoint_pool(checkpoint_pool: List[str], sample_size: int, rng: random.Random) -> List[str]:
        if not checkpoint_pool:
            return []
        target = max(1, int(sample_size))
        if len(checkpoint_pool) >= target:
            return list(rng.sample(checkpoint_pool, k=target))
        return [str(rng.choice(checkpoint_pool)) for _ in range(target)]

    def _load_frozen_agent_from_checkpoint(self, checkpoint_path: str, label: str) -> Optional[RLAgent]:
        try:
            agent = RLAgent()
            agent.load_model(checkpoint_path)
            agent.train_from_self_play = False
            agent.config.train_from_self_play = False
            agent.ppo_enable = False
            agent.id = f"{label}_{agent.id[:8]}_{uuid.uuid4().hex[:8]}"
            return agent
        except Exception as e:
            logger.warning("Failed loading frozen checkpoint %s: %s", checkpoint_path, e)
            return None

    @staticmethod
    def _clone_agent_for_evaluation(agent: RLAgent) -> Optional[RLAgent]:
        try:
            clone = RLAgent(config=type(agent.config)(**asdict(agent.config)))
            clone.network.load_state_dict(agent.network.state_dict())
            clone.network.eval()
            clone.train_from_self_play = False
            clone.config.train_from_self_play = False
            clone.ppo_enable = False
            clone.id = str(agent.id)
            return clone
        except Exception:
            return None

    async def _run_single_checkpoint_pool_game(
        self,
        anchor_agent: RLAgent,
        opponent_checkpoints: List[str],
        label: str,
        global_game_semaphore: Optional[asyncio.Semaphore],
        frozen_cache: Optional[Dict[str, RLAgent]] = None,
    ) -> Optional[Any]:
        opponents: List[RLAgent] = []
        for checkpoint_path in opponent_checkpoints:
            template_agent: Optional[RLAgent] = None
            if frozen_cache is not None:
                template_agent = frozen_cache.get(checkpoint_path)
            if template_agent is None:
                template_agent = self._load_frozen_agent_from_checkpoint(checkpoint_path, label=label)
                if template_agent is not None and frozen_cache is not None:
                    frozen_cache[checkpoint_path] = template_agent
            if template_agent is None:
                return None
            frozen_agent = self._clone_agent_for_evaluation(template_agent)
            if frozen_agent is None:
                return None
            opponents.append(frozen_agent)

        if len(opponents) < 3:
            return None

        lineup = [anchor_agent] + opponents[:3]
        tournament_id = f"{label}_{int(self.current_generation)}_{uuid.uuid4().hex[:8]}"
        if global_game_semaphore is not None:
            async with global_game_semaphore:
                return await self.tournament_manager._run_single_game(lineup, tournament_id)
        return await self.tournament_manager._run_single_game(lineup, tournament_id)

    async def _run_checkpoint_pool_matchups(
        self,
        anchor_agents: List[RLAgent],
        checkpoint_pool: List[str],
        games_per_agent: int,
        label: str,
        global_game_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        rounds = max(1, int(games_per_agent))
        planned_games = int(len(anchor_agents) * rounds)
        successful_games = 0
        failed_games = 0
        games: List[Dict[str, Any]] = []
        rng = random.Random(f"{label}:{int(self.current_generation)}")
        # Cache frozen checkpoint loads during this matchup batch to avoid repeated
        # synchronous disk deserialization for every game.
        frozen_cache: Dict[str, RLAgent] = {}

        for _round_idx in range(rounds):
            round_tasks = [
                asyncio.create_task(
                    self._run_single_checkpoint_pool_game(
                        anchor_agent=anchor_agent,
                        opponent_checkpoints=self._sample_checkpoint_pool(checkpoint_pool, 3, rng),
                        label=label,
                        global_game_semaphore=global_game_semaphore,
                        frozen_cache=frozen_cache,
                    )
                )
                for anchor_agent in anchor_agents
            ]
            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)
            for result in round_results:
                if isinstance(result, Exception) or result is None:
                    failed_games += 1
                    continue
                games.append(self.tournament_manager._game_result_to_dict(result))
                if bool(getattr(result, "completed", False)):
                    successful_games += 1
                else:
                    failed_games += 1

        finished_games = successful_games + failed_games
        return {
            "tournament_id": f"{label}_{int(self.current_generation)}_{uuid.uuid4().hex[:8]}",
            "agents": [agent.id for agent in anchor_agents],
            "games": games,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "completed_games": int(finished_games),
            "successful_games": int(successful_games),
            "failed_games": int(failed_games),
            "total_planned_games": int(planned_games),
        }

    @staticmethod
    def _summarize_anchor_results(anchor_agents: List[RLAgent], matchup_result: Dict[str, Any]) -> Dict[str, Any]:
        anchor_ids = [agent.id for agent in anchor_agents]
        stats: Dict[str, Dict[str, float]] = {
            agent_id: {
                "games": 0.0,
                "wins": 0.0,
                "vp_sum": 0.0,
                "rank_sum": 0.0,
                "completed": 0.0,
            }
            for agent_id in anchor_ids
        }

        for game_result in matchup_result.get("games", []) or []:
            if not isinstance(game_result, dict):
                continue
            for player_result in game_result.get("players", []) or []:
                if not isinstance(player_result, dict):
                    continue
                agent_id = str(player_result.get("agent_id", "") or "")
                if agent_id not in stats:
                    continue
                bucket = stats[agent_id]
                bucket["games"] += 1.0
                bucket["vp_sum"] += float(player_result.get("victory_points", 0) or 0)
                bucket["rank_sum"] += float(player_result.get("rank", 4) or 4)
                if int(player_result.get("rank", 4) or 4) == 1:
                    bucket["wins"] += 1.0
                if bool(player_result.get("completed", False)):
                    bucket["completed"] += 1.0

        aggregate = {
            "games": 0.0,
            "wins": 0.0,
            "vp_sum": 0.0,
            "rank_sum": 0.0,
            "completed": 0.0,
        }
        per_agent: Dict[str, Dict[str, float]] = {}
        for agent_id, bucket in stats.items():
            games = float(bucket["games"])
            aggregate["games"] += games
            aggregate["wins"] += float(bucket["wins"])
            aggregate["vp_sum"] += float(bucket["vp_sum"])
            aggregate["rank_sum"] += float(bucket["rank_sum"])
            aggregate["completed"] += float(bucket["completed"])
            per_agent[agent_id] = {
                "games": games,
                "win_rate": (float(bucket["wins"]) / games) if games > 0 else 0.0,
                "avg_vp": (float(bucket["vp_sum"]) / games) if games > 0 else 0.0,
                "avg_rank": (float(bucket["rank_sum"]) / games) if games > 0 else 0.0,
                "completion_rate": (float(bucket["completed"]) / games) if games > 0 else 0.0,
            }

        aggregate_games = float(aggregate["games"])
        aggregate_summary = {
            "games": aggregate_games,
            "win_rate": (float(aggregate["wins"]) / aggregate_games) if aggregate_games > 0 else 0.0,
            "avg_vp": (float(aggregate["vp_sum"]) / aggregate_games) if aggregate_games > 0 else 0.0,
            "avg_rank": (float(aggregate["rank_sum"]) / aggregate_games) if aggregate_games > 0 else 0.0,
            "completion_rate": (float(aggregate["completed"]) / aggregate_games) if aggregate_games > 0 else 0.0,
        }
        return {
            "aggregate": aggregate_summary,
            "per_agent": per_agent,
        }

    async def _evaluate_with_training_opponent_pool(
        self,
        matchmaking_population: List[RLAgent],
        global_game_semaphore: Optional[asyncio.Semaphore],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        metrics: Dict[str, Any] = {
            "frozen_pool/enabled": bool(self.training_opponent_pool_enabled),
            "frozen_pool/ran": False,
            "frozen_pool/checkpoints_available": 0,
            "frozen_pool/games_per_agent": int(self.training_pool_games_per_agent),
            "frozen_pool/min_checkpoints": int(self.training_pool_min_checkpoints),
        }
        if not self.training_opponent_pool_enabled or int(self.training_pool_games_per_agent) <= 0:
            metrics["frozen_pool/skip_reason"] = "disabled"
            return [], metrics

        checkpoint_pool = self._resolve_training_opponent_checkpoints()
        metrics["frozen_pool/checkpoints_available"] = int(len(checkpoint_pool))
        if len(checkpoint_pool) < int(self.training_pool_min_checkpoints):
            metrics["frozen_pool/skip_reason"] = "insufficient_checkpoint_pool"
            return [], metrics

        anchors = list(matchmaking_population or self.population)
        matchup_result = await self._run_checkpoint_pool_matchups(
            anchor_agents=anchors,
            checkpoint_pool=checkpoint_pool,
            games_per_agent=int(self.training_pool_games_per_agent),
            label="training_pool",
            global_game_semaphore=global_game_semaphore,
        )
        summary = self._summarize_anchor_results(anchors, matchup_result)
        aggregate = dict(summary.get("aggregate", {}) or {})

        metrics.update(
            {
                "frozen_pool/ran": True,
                "frozen_pool/games_planned": int(matchup_result.get("total_planned_games", 0)),
                "frozen_pool/games_completed": int(matchup_result.get("completed_games", 0)),
                "frozen_pool/games_successful": int(matchup_result.get("successful_games", 0)),
                "frozen_pool/games_failed": int(matchup_result.get("failed_games", 0)),
                "frozen_pool/win_rate": float(aggregate.get("win_rate", 0.0)),
                "frozen_pool/avg_vp": float(aggregate.get("avg_vp", 0.0)),
                "frozen_pool/avg_rank": float(aggregate.get("avg_rank", 0.0)),
                "frozen_pool/completion_rate": float(aggregate.get("completion_rate", 0.0)),
                "frozen_pool/sample_checkpoints": checkpoint_pool[: min(8, len(checkpoint_pool))],
            }
        )
        return [matchup_result], metrics

    async def _maybe_run_fixed_benchmark(
        self,
        generation: int,
        fitness_scores: List[float],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "benchmark/fixed/enabled": bool(self.fixed_benchmark_enabled),
            "benchmark/fixed/interval": int(self.fixed_benchmark_interval),
            "benchmark/fixed/ran": False,
            "benchmark/fixed/checkpoints_available": int(len(self.fixed_benchmark_checkpoints)),
        }
        if not self.fixed_benchmark_enabled:
            metrics["benchmark/fixed/skip_reason"] = "disabled"
            return metrics
        if (int(generation) + 1) % int(self.fixed_benchmark_interval) != 0:
            metrics["benchmark/fixed/skip_reason"] = "interval_not_reached"
            return metrics

        if not self.fixed_benchmark_checkpoints:
            self.fixed_benchmark_checkpoints = self._resolve_fixed_benchmark_checkpoints()
        metrics["benchmark/fixed/checkpoints_available"] = int(len(self.fixed_benchmark_checkpoints))
        if len(self.fixed_benchmark_checkpoints) < 3:
            metrics["benchmark/fixed/skip_reason"] = "insufficient_checkpoint_pool"
            return metrics

        if not self.population:
            metrics["benchmark/fixed/skip_reason"] = "empty_population"
            return metrics

        rank_indices = sorted(
            range(len(self.population)),
            key=lambda idx: float(fitness_scores[idx]) if idx < len(fitness_scores) else 0.0,
            reverse=True,
        )
        eval_count = max(1, min(len(rank_indices), int(self.fixed_benchmark_agent_count)))
        anchor_agents: List[RLAgent] = []
        for idx in rank_indices[:eval_count]:
            cloned = self._clone_agent_for_evaluation(self.population[idx])
            if cloned is not None:
                anchor_agents.append(cloned)
        if not anchor_agents:
            metrics["benchmark/fixed/skip_reason"] = "anchor_clone_failed"
            return metrics
        global_game_limit = self._resolve_global_game_concurrency()
        global_game_semaphore = asyncio.Semaphore(global_game_limit)
        matchup_result = await self._run_checkpoint_pool_matchups(
            anchor_agents=anchor_agents,
            checkpoint_pool=list(self.fixed_benchmark_checkpoints),
            games_per_agent=int(self.fixed_benchmark_games_per_agent),
            label="fixed_benchmark",
            global_game_semaphore=global_game_semaphore,
        )
        summary = self._summarize_anchor_results(anchor_agents, matchup_result)
        aggregate = dict(summary.get("aggregate", {}) or {})

        metrics.update(
            {
                "benchmark/fixed/ran": True,
                "benchmark/fixed/games_per_agent": int(self.fixed_benchmark_games_per_agent),
                "benchmark/fixed/games_planned": int(matchup_result.get("total_planned_games", 0)),
                "benchmark/fixed/games_completed": int(matchup_result.get("completed_games", 0)),
                "benchmark/fixed/games_successful": int(matchup_result.get("successful_games", 0)),
                "benchmark/fixed/games_failed": int(matchup_result.get("failed_games", 0)),
                "benchmark/fixed/win_rate": float(aggregate.get("win_rate", 0.0)),
                "benchmark/fixed/avg_vp": float(aggregate.get("avg_vp", 0.0)),
                "benchmark/fixed/avg_rank": float(aggregate.get("avg_rank", 0.0)),
                "benchmark/fixed/completion_rate": float(aggregate.get("completion_rate", 0.0)),
                "benchmark/fixed/anchor_agent_ids": [agent.id for agent in anchor_agents],
                "benchmark/fixed/per_agent": summary.get("per_agent", {}),
            }
        )
        logger.info(
            "Fixed benchmark generation %d: agents=%d games=%d win_rate=%.3f avg_vp=%.2f avg_rank=%.2f",
            int(generation),
            len(anchor_agents),
            int(matchup_result.get("completed_games", 0)),
            float(metrics.get("benchmark/fixed/win_rate", 0.0)),
            float(metrics.get("benchmark/fixed/avg_vp", 0.0)),
            float(metrics.get("benchmark/fixed/avg_rank", 0.0)),
        )
        return metrics
    
    async def save_generation_models(self, generation: int, fitness_scores: List[float]):
        """Save models from current generation"""
        save_dir = os.path.join(self._default_models_root(), f"generation_{generation}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save top-K agents so training can sample stronger frozen opponents.
        num_to_save = max(1, min(len(self.population), int(self.save_top_k)))
        
        # Get fitness scores to find top agents
        top_indices = sorted(range(len(fitness_scores)), 
                           key=lambda i: fitness_scores[i], 
                           reverse=True)[:num_to_save]

        gate_per_agent = dict((self.last_generation_gate or {}).get("per_agent", {}) or {})
        gate_summary = dict(self.last_generation_gate or {})
        if "per_agent" in gate_summary:
            gate_summary.pop("per_agent", None)
        
        for i, idx in enumerate(top_indices):
            agent = self.population[idx]
            model_path = f"{save_dir}/agent_{i}_fitness_{fitness_scores[idx]:.2f}.pth"
            agent.save_model(model_path)
            
            # Save agent config
            config_path = f"{save_dir}/agent_{i}_config.json"
            cfg = agent.get_config()
            raw_eval_fitness = self.last_raw_eval_fitness.get(agent.id)
            gated_eval_fitness = self.last_eval_fitness.get(agent.id)
            # Include evaluation fitness used for selection for clarity.
            try:
                cfg['eval_fitness'] = float(gated_eval_fitness if gated_eval_fitness is not None else fitness_scores[idx])
            except Exception:
                cfg['eval_fitness'] = None
            cfg['eval_fitness_raw'] = float(raw_eval_fitness) if raw_eval_fitness is not None else None
            cfg['eval_fitness_gated'] = float(gated_eval_fitness) if gated_eval_fitness is not None else None
            cfg['generation_behavior_metrics'] = dict(self.last_generation_behavior_metrics or {})
            cfg['generation_gate_summary'] = gate_summary
            cfg['promotion_gate'] = gate_per_agent.get(agent.id, {})
            gate_behavior = dict((gate_per_agent.get(agent.id, {}) or {}).get("behavior", {}) or {})
            cfg['end_screen_tracking'] = {
                "vp_terraforming_per_game": float(gate_behavior.get("vp_terraforming_per_game", 0.0)),
                "vp_milestones_per_game": float(gate_behavior.get("vp_milestones_per_game", 0.0)),
                "vp_awards_per_game": float(gate_behavior.get("vp_awards_per_game", 0.0)),
                "vp_greenery_per_game": float(gate_behavior.get("vp_greenery_per_game", 0.0)),
                "vp_city_per_game": float(gate_behavior.get("vp_city_per_game", 0.0)),
                "vp_cards_per_game": float(gate_behavior.get("vp_cards_per_game", 0.0)),
                "vp_total_per_game": float(gate_behavior.get("vp_total_per_game", 0.0)),
                "town_placements_per_game": float(gate_behavior.get("town_placements_per_game", 0.0)),
                "greenery_placements_per_game": float(gate_behavior.get("greenery_placements_per_game", 0.0)),
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)

        generation_summary_path = f"{save_dir}/generation_metrics.json"
        generation_summary = {
            "generation": int(generation),
            "saved_agents": int(num_to_save),
            "top_indices": [int(i) for i in top_indices],
            "selection_fitness_scores": [float(fitness_scores[i]) for i in top_indices],
            "raw_eval_fitness": {
                self.population[i].id: float(self.last_raw_eval_fitness.get(self.population[i].id, fitness_scores[i]))
                for i in top_indices
            },
            "gated_eval_fitness": {
                self.population[i].id: float(self.last_eval_fitness.get(self.population[i].id, fitness_scores[i]))
                for i in top_indices
            },
            "behavior_metrics": dict(self.last_generation_behavior_metrics or {}),
            "gate_summary": gate_summary,
        }
        with open(generation_summary_path, 'w', encoding='utf-8') as f:
            json.dump(generation_summary, f, indent=2)
        
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
                "version": 2,
                "next_generation": int(next_generation),
                "population_size": len(self.population),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "evolution_generation_count": int(self.evolution_manager.generation_count),
                "last_generation_behavior_metrics": dict(self.last_generation_behavior_metrics or {}),
                "last_generation_gate": dict(self.last_generation_gate or {}),
                "league_state": self.league_manager.get_state(),
                "fixed_benchmark_checkpoints": list(self.fixed_benchmark_checkpoints or []),
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
            self.evolution_manager.generation_count = int(
                state.get("evolution_generation_count", next_generation)
            )
            self.last_generation_behavior_metrics = dict(state.get("last_generation_behavior_metrics", {}) or {})
            self.last_generation_gate = dict(state.get("last_generation_gate", {}) or {})
            self.league_manager.load_state(dict(state.get("league_state", {}) or {}))
            restored_benchmark = self._dedupe_existing_checkpoints(
                list(state.get("fixed_benchmark_checkpoints", []) or [])
            )
            if restored_benchmark:
                self.fixed_benchmark_checkpoints = restored_benchmark[: int(self.fixed_benchmark_pool_size)]
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
        self.evolution_manager.generation_count = int(self.current_generation)
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
            self.evolution_manager.generation_count = int(self.current_generation)
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
                public_base = public_base.split(',')[0] if ',' in public_base else public_base
                if '://' not in public_base:
                    public_base = f"http://{public_base}"
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

    def _generation_from_checkpoint(self, checkpoint_path: str) -> Optional[int]:
        try:
            generation_dir = os.path.basename(os.path.dirname(str(checkpoint_path)))
            if generation_dir.startswith("generation_"):
                return int(generation_dir.split("_", 1)[1])
        except Exception:
            return None
        return None

    def _all_saved_checkpoints(self, models_root: str) -> List[str]:
        checkpoints: List[str] = []
        for generation in self._available_generations(models_root):
            checkpoints.extend(self._all_generation_checkpoints(models_root, generation))
        deduped = sorted(set(checkpoints), key=self._checkpoint_fitness_from_name, reverse=True)
        return deduped

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

    async def _start_human_vs_ai_from_checkpoints(
        self,
        generation: Optional[int],
        human_name: str,
        selected_checkpoints: List[str],
    ) -> Dict[str, Any]:
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

        return await self._start_human_vs_ai_from_checkpoints(
            generation=generation,
            human_name=human_name,
            selected_checkpoints=selected_checkpoints,
        )

    async def run_human_vs_best_agent_game(
        self,
        human_name: str = "You",
        bot_count: int = 3,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a human game against the best saved checkpoint found across all generations."""
        models_root = self._default_models_root()
        rng = random.Random(seed)
        all_checkpoints = self._all_saved_checkpoints(models_root)
        if not all_checkpoints:
            return {"error": f"No checkpoints found under {models_root}"}

        best_checkpoint = all_checkpoints[0]
        best_generation = self._generation_from_checkpoint(best_checkpoint)

        selected_checkpoints: List[str] = [best_checkpoint]
        same_generation_pool: List[str] = []
        if best_generation is not None:
            same_generation_pool = [
                path for path in self._all_generation_checkpoints(models_root, best_generation)
                if path != best_checkpoint
            ]
        fallback_pool = [
            path for path in all_checkpoints
            if path != best_checkpoint and path not in same_generation_pool
        ]
        candidate_pool = same_generation_pool + fallback_pool
        while len(selected_checkpoints) < bot_count:
            if candidate_pool:
                selected_checkpoints.append(candidate_pool.pop(0))
            else:
                selected_checkpoints.append(best_checkpoint)

        # Randomize non-primary opponents for variety while keeping the best checkpoint included.
        if len(selected_checkpoints) > 1:
            tail = selected_checkpoints[1:]
            rng.shuffle(tail)
            selected_checkpoints = [selected_checkpoints[0]] + tail

        result = await self._start_human_vs_ai_from_checkpoints(
            generation=best_generation,
            human_name=human_name,
            selected_checkpoints=selected_checkpoints[:bot_count],
        )
        if "error" in result:
            return result
        result["best_checkpoint"] = best_checkpoint
        result["selection_mode"] = "best_overall"
        return result
    
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
        api_restart_delay_sec = 2.0

        # Keep training independent from API liveness; restart API if it exits.
        while True:
            done, _pending = await asyncio.wait(
                {api_task, evolution_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if evolution_task in done:
                exc = evolution_task.exception()
                if exc is not None:
                    raise exc
                logger.info("Evolution cycle finished. Keeping API server alive.")
                while True:
                    try:
                        await api_task
                    except Exception as api_exc:
                        logger.exception(
                            "API server task crashed after evolution completion: %s",
                            api_exc,
                        )
                    logger.error(
                        "API server stopped unexpectedly after evolution completion. Restarting in %.1fs.",
                        api_restart_delay_sec,
                    )
                    await asyncio.sleep(api_restart_delay_sec)
                    api_task = asyncio.create_task(start_api_server(coordinator))

            if api_task in done:
                api_exc = api_task.exception()
                if api_exc is not None:
                    logger.error(
                        "API server task crashed (%s). Restarting in %.1fs while training continues.",
                        str(api_exc),
                        api_restart_delay_sec,
                    )
                else:
                    logger.error(
                        "API server stopped unexpectedly. Restarting in %.1fs while training continues.",
                        api_restart_delay_sec,
                    )
                await asyncio.sleep(api_restart_delay_sec)
                api_task = asyncio.create_task(start_api_server(coordinator))
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
