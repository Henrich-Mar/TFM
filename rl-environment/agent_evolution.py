"""
Agent Evolution - Manages evolutionary algorithm for agent breeding
"""
import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import asyncio
import torch
import os

from models.agent import RLAgent, AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    population_size: int = 32
    elite_percentage: float = 0.2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    diversity_bonus: float = 0.1
    min_generations_before_evolution: int = 5

class EvolutionManager:
    def __init__(self, config):
        self.population_size = config.population_size
        self.elite_percentage = config.elite_percentage
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = getattr(config, 'crossover_rate', 0.7)
        self.diversity_bonus = getattr(config, 'diversity_bonus', 0.1)
        
        self.generation_count = 0
        self.fitness_history = []
        self.diversity_history = []
        self.rl_first_enabled = str(os.getenv("RL_FIRST_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.immigrant_ratio = self._safe_env_float("EVOLUTION_IMMIGRANT_RATIO", 0.10)
        self.immigrant_interval = self._safe_env_int("EVOLUTION_IMMIGRANT_INTERVAL", 3)
        self.immigrant_mutation_rate = self._safe_env_float("EVOLUTION_IMMIGRANT_MUTATION_RATE", 0.30)
        self.gate_parent_filter_enabled = str(os.getenv("EVOLUTION_GATE_PARENT_FILTER_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.gate_failed_replacement_enabled = str(os.getenv("EVOLUTION_GATE_REPLACE_FAILED_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.gate_replacement_mutation_rate = self._safe_env_float("EVOLUTION_GATE_REPLACEMENT_MUTATION_RATE", 0.05)
        self.epsilon_init_min = self._safe_env_float("EVOLUTION_INIT_EPSILON_MIN", 0.01)
        self.epsilon_init_max = self._safe_env_float("EVOLUTION_INIT_EPSILON_MAX", 0.12)
        self.temperature_init_min = self._safe_env_float("EVOLUTION_INIT_TEMPERATURE_MIN", 0.7)
        self.temperature_init_max = self._safe_env_float("EVOLUTION_INIT_TEMPERATURE_MAX", 1.2)

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
        
        
    async def create_initial_population(self, population_size: int) -> List[RLAgent]:
        """Create initial diverse population of agents"""
        population = []
        default_config = AgentConfig()
        
        for i in range(population_size):
            # Create diverse initial configurations
            config = AgentConfig(
                state_size=self._safe_env_int("AGENT_STATE_SIZE", int(default_config.state_size)),
                hidden_size=self._safe_env_int("AGENT_HIDDEN_SIZE", int(default_config.hidden_size)),
                num_layers=self._safe_env_int("AGENT_NUM_LAYERS", int(default_config.num_layers)),
                learning_rate=random.uniform(1e-5, 1e-3),
                epsilon=random.uniform(
                    min(self.epsilon_init_min, self.epsilon_init_max),
                    max(self.epsilon_init_min, self.epsilon_init_max),
                ),
                temperature=random.uniform(
                    min(self.temperature_init_min, self.temperature_init_max),
                    max(self.temperature_init_min, self.temperature_init_max),
                ),
                max_thinking_time=random.uniform(1.0, 10.0)
            )
            
            agent = RLAgent(config)
            
            # Add some initial random mutations to diversify
            agent.mutate(mutation_rate=0.5)
            
            population.append(agent)
            
        logger.info(f"Created initial population of {len(population)} diverse agents")
        return population
    
    async def evolve_population(self,
                              population: List[RLAgent],
                              fitness_scores: List[float],
                              gate_results: Optional[Dict[str, Dict[str, Any]]] = None) -> List[RLAgent]:
        """Evolve population using genetic algorithm"""
        self.generation_count += 1
        
        # Record statistics
        self.fitness_history.append({
            'max': max(fitness_scores),
            'mean': np.mean(fitness_scores),
            'std': np.std(fitness_scores)
        })
        
        # Calculate diversity
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        
        logger.info(f"Generation {self.generation_count} - "
                   f"Max eval fitness: {max(fitness_scores):.2f}, "
                   f"Mean eval fitness: {np.mean(fitness_scores):.2f}, "
                   f"Diversity: {diversity:.3f}")

        if self.rl_first_enabled:
            return await self._evolve_population_rl_first(
                population,
                fitness_scores,
                gate_results=gate_results,
            )
        
        # Apply diversity bonus to fitness scores
        adjusted_fitness = self._apply_diversity_bonus(population, fitness_scores)
        
        # Selection
        elite_agents, breeding_pool = self._selection(
            population,
            adjusted_fitness,
            gate_results=gate_results,
        )
        
        # Create new generation
        new_population = await self._create_new_generation(elite_agents, breeding_pool)
        
        return new_population

    async def _evolve_population_rl_first(
        self,
        population: List[RLAgent],
        fitness_scores: List[float],
        gate_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[RLAgent]:
        """RL-first mode: keep PPO-trained population, inject immigrants periodically."""
        if not population:
            return []

        ranked_indices = sorted(
            range(len(population)),
            key=lambda i: float(fitness_scores[i]) if i < len(fitness_scores) else 0.0,
            reverse=True,
        )
        ranked_population = [population[i] for i in ranked_indices]
        new_population = ranked_population[: self.population_size]
        passed_ranked = ranked_population
        failed_count = 0

        if gate_results and (self.gate_parent_filter_enabled or self.gate_failed_replacement_enabled):
            passed_ranked, failed_ranked = self._partition_ranked_population_by_gate(
                ranked_population,
                gate_results,
            )
            failed_count = len(failed_ranked)
            if passed_ranked:
                if self.gate_failed_replacement_enabled:
                    new_population = list(passed_ranked[: self.population_size])
                    replacement_parent_pool = self._parent_pool_from_ranked(passed_ranked)
                    while len(new_population) < self.population_size:
                        new_population.append(self._create_gate_replacement(replacement_parent_pool))
                else:
                    new_population = ranked_population[: self.population_size]
                logger.info(
                    "RL-first evolution: retained %d gate-passed agents and replaced %d gate-failed slots.",
                    min(len(passed_ranked), self.population_size),
                    max(0, self.population_size - min(len(passed_ranked), self.population_size)),
                )
            else:
                logger.warning(
                    "RL-first evolution: all %d ranked agents failed promotion gates; falling back to pure fitness ordering.",
                    failed_count,
                )

        interval = max(1, int(self.immigrant_interval))
        if (self.generation_count % interval) != 0:
            logger.info(
                "RL-first evolution: no immigrants this generation (interval=%d, gate_failed=%d).",
                interval,
                failed_count,
            )
            return new_population

        immigrant_count = max(1, int(len(new_population) * max(0.0, min(0.5, float(self.immigrant_ratio)))))
        parent_pool_source = passed_ranked if passed_ranked else new_population
        parent_pool = self._parent_pool_from_ranked(parent_pool_source)
        for idx in range(immigrant_count):
            immigrant = self._create_immigrant(parent_pool)
            replace_idx = max(0, len(new_population) - 1 - idx)
            new_population[replace_idx] = immigrant

        logger.info(
            "RL-first evolution: introduced %d immigrants (ratio=%.2f, interval=%d, gate_failed=%d).",
            immigrant_count,
            float(self.immigrant_ratio),
            interval,
            failed_count,
        )
        return new_population

    def _create_immigrant(self, parent_pool: List[RLAgent]) -> RLAgent:
        if len(parent_pool) >= 2 and random.random() < float(self.crossover_rate):
            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)
            if parent1.id != parent2.id:
                child = parent1.crossover(parent2)
            else:
                child = self._clone_agent(parent1)
        else:
            child = self._clone_agent(random.choice(parent_pool))
        child.mutate(mutation_rate=max(0.01, float(self.immigrant_mutation_rate)))
        return child

    def _create_gate_replacement(self, parent_pool: List[RLAgent]) -> RLAgent:
        if not parent_pool:
            raise ValueError("parent_pool must not be empty")
        if len(parent_pool) >= 2 and random.random() < float(self.crossover_rate):
            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)
            if parent1.id != parent2.id:
                child = parent1.crossover(parent2)
            else:
                child = self._clone_agent(parent1)
        else:
            child = self._clone_agent(random.choice(parent_pool))
        child.mutate(mutation_rate=max(0.0, float(self.gate_replacement_mutation_rate)))
        return child

    @staticmethod
    def _gate_passed(agent_id: str, gate_results: Optional[Dict[str, Dict[str, Any]]]) -> bool:
        if not gate_results:
            return True
        gate_info = gate_results.get(agent_id, {})
        if not isinstance(gate_info, dict):
            return True
        return bool(gate_info.get("passed", True))

    def _partition_ranked_population_by_gate(
        self,
        ranked_population: List[RLAgent],
        gate_results: Optional[Dict[str, Dict[str, Any]]],
    ) -> Tuple[List[RLAgent], List[RLAgent]]:
        passed_ranked: List[RLAgent] = []
        failed_ranked: List[RLAgent] = []
        for agent in ranked_population:
            if self._gate_passed(getattr(agent, "id", ""), gate_results):
                passed_ranked.append(agent)
            else:
                failed_ranked.append(agent)
        return passed_ranked, failed_ranked

    @staticmethod
    def _parent_pool_from_ranked(ranked_population: List[RLAgent]) -> List[RLAgent]:
        if not ranked_population:
            return []
        top_count = max(1, len(ranked_population) // 2)
        return ranked_population[:top_count]
    
    def _apply_diversity_bonus(self, population: List[RLAgent], fitness_scores: List[float]) -> List[float]:
        """Apply bonus for genetic diversity"""
        if self.diversity_bonus <= 0:
            return fitness_scores
        
        adjusted_scores = fitness_scores.copy()
        
        # Calculate pairwise distances between agents
        for i, agent_i in enumerate(population):
            diversity_score = 0.0
            
            for j, agent_j in enumerate(population):
                if i != j:
                    distance = self._calculate_agent_distance(agent_i, agent_j)
                    diversity_score += distance
            
            # Normalize by population size
            diversity_score /= len(population) - 1
            
            # Add diversity bonus
            bonus = diversity_score * self.diversity_bonus * max(fitness_scores)
            adjusted_scores[i] += bonus
        
        return adjusted_scores
    
    def _calculate_agent_distance(self, agent1: RLAgent, agent2: RLAgent) -> float:
        """Calculate genetic distance between two agents"""
        distance = 0.0
        param_count = 0
        
        # Compare network parameters
        with torch.no_grad():
            for p1, p2 in zip(agent1.network.parameters(), agent2.network.parameters()):
                p1_cpu = p1.detach().float().cpu()
                p2_cpu = p2.detach().float().cpu()
                param_distance = torch.norm(p1_cpu - p2_cpu).item()
                distance += param_distance
                param_count += 1
        
        # Normalize by number of parameters
        if param_count > 0:
            distance /= param_count
        
        # Add configuration differences
        config_distance = abs(agent1.config.learning_rate - agent2.config.learning_rate)
        config_distance += abs(agent1.config.epsilon - agent2.config.epsilon)
        config_distance += abs(agent1.config.temperature - agent2.config.temperature)
        
        return distance + config_distance
    
    def _calculate_population_diversity(self, population: List[RLAgent]) -> float:
        """Calculate overall population diversity"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_agent_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _selection(
        self,
        population: List[RLAgent],
        fitness_scores: List[float],
        gate_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[RLAgent], List[RLAgent]]:
        """Select elite agents and breeding pool"""
        # Sort by fitness
        sorted_indices = sorted(range(len(fitness_scores)),
                              key=lambda i: fitness_scores[i],
                              reverse=True)
        selection_indices = list(sorted_indices)
        if gate_results and self.gate_parent_filter_enabled:
            passed_indices = [
                idx for idx in sorted_indices
                if self._gate_passed(getattr(population[idx], "id", ""), gate_results)
            ]
            if passed_indices:
                selection_indices = passed_indices
                logger.info(
                    "Evolution selection: filtered parent pool to %d gate-passed agents (excluded %d gate-failed agents).",
                    len(selection_indices),
                    max(0, len(sorted_indices) - len(selection_indices)),
                )
            else:
                logger.warning(
                    "Evolution selection: no gate-passed agents available; falling back to pure fitness ordering."
                )

        # Elite selection
        elite_count = min(len(selection_indices), max(1, int(len(population) * self.elite_percentage)))
        elite_indices = selection_indices[:elite_count]
        elite_agents = [population[i] for i in elite_indices]

        # Breeding pool: top 50% for crossover
        breeding_count = min(len(selection_indices), max(1, len(population) // 2))
        breeding_indices = selection_indices[:breeding_count]
        breeding_pool = [population[i] for i in breeding_indices]
        
        logger.info(f"Selected {len(elite_agents)} elite agents and {len(breeding_pool)} for breeding")
        
        return elite_agents, breeding_pool
    
    async def _create_new_generation(self, 
                                   elite_agents: List[RLAgent], 
                                   breeding_pool: List[RLAgent]) -> List[RLAgent]:
        """Create new generation through crossover and mutation"""
        new_population = []
        
        # Keep elite agents
        for agent in elite_agents:
            new_agent = self._clone_agent(agent)
            new_population.append(new_agent)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(breeding_pool) >= 2:
                # Crossover
                parent1 = random.choice(breeding_pool)
                parent2 = random.choice(breeding_pool)
                
                if parent1.id != parent2.id:  # Ensure different parents
                    child = parent1.crossover(parent2)
                else:
                    child = self._clone_agent(parent1)
            else:
                # Mutation only
                parent = random.choice(breeding_pool)
                child = self._clone_agent(parent)
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                mutation_strength = self._adaptive_mutation_rate()
                child.mutate(mutation_strength)
            
            new_population.append(child)
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        logger.info(f"Created new generation of {len(new_population)} agents")
        return new_population
    
    def _clone_agent(self, agent: RLAgent) -> RLAgent:
        """Create a copy of an agent"""
        # Clone config by value to avoid shared mutable references between agents.
        new_agent = RLAgent(AgentConfig(**asdict(agent.config)))
        
        # Copy network weights
        new_agent.network.load_state_dict(agent.network.state_dict())
        
        return new_agent
    
    def _adaptive_mutation_rate(self) -> float:
        """Calculate adaptive mutation rate based on diversity"""
        base_rate = self.mutation_rate
        
        # If diversity is low, increase mutation
        if self.diversity_history:
            recent_diversity = np.mean(self.diversity_history[-5:])  # Last 5 generations
            if recent_diversity < 0.1:  # Low diversity threshold
                return base_rate * 2.0
            elif recent_diversity > 0.5:  # High diversity
                return base_rate * 0.5
        
        return base_rate
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        if not self.fitness_history:
            return {}
        
        return {
            'generation': self.generation_count,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'current_best_fitness': self.fitness_history[-1]['max'] if self.fitness_history else 0,
            'average_fitness_trend': np.mean([gen['mean'] for gen in self.fitness_history[-10:]]) if len(self.fitness_history) >= 10 else 0,
            'diversity_trend': np.mean(self.diversity_history[-10:]) if len(self.diversity_history) >= 10 else 0
        }
    
    def should_introduce_immigrants(self) -> bool:
        """Determine if we should introduce new random agents (immigration)"""
        if len(self.diversity_history) < 10:
            return False
        
        # Check if diversity has been consistently low
        recent_diversity = np.mean(self.diversity_history[-10:])
        return recent_diversity < 0.05  # Very low diversity threshold
    
    async def introduce_immigrants(self, population: List[RLAgent], num_immigrants: int = None) -> List[RLAgent]:
        """Introduce new random agents to increase diversity"""
        if num_immigrants is None:
            num_immigrants = max(1, len(population) // 10)  # 10% immigrants
        
        # Remove worst performers
        # This would require fitness scores, so for now just replace random agents
        for _ in range(num_immigrants):
            if len(population) > num_immigrants:
                # Replace a random agent
                replace_idx = random.randint(0, len(population) - 1)
                
                # Create new random agent
                default_config = AgentConfig()
                config = AgentConfig(
                    state_size=self._safe_env_int("AGENT_STATE_SIZE", int(default_config.state_size)),
                    hidden_size=self._safe_env_int("AGENT_HIDDEN_SIZE", int(default_config.hidden_size)),
                    num_layers=self._safe_env_int("AGENT_NUM_LAYERS", int(default_config.num_layers)),
                    learning_rate=random.uniform(1e-5, 1e-3),
                    epsilon=random.uniform(
                        min(self.epsilon_init_min, self.epsilon_init_max),
                        max(self.epsilon_init_min, self.epsilon_init_max),
                    ),
                    temperature=random.uniform(
                        min(self.temperature_init_min, self.temperature_init_max),
                        max(self.temperature_init_min, self.temperature_init_max),
                    )
                )
                
                new_agent = RLAgent(config)
                new_agent.mutate(mutation_rate=0.8)  # High mutation for diversity
                
                population[replace_idx] = new_agent
        
        logger.info(f"Introduced {num_immigrants} immigrant agents for diversity")
        return population
    
    
