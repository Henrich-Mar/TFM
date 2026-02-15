"""
Metrics Tracker - Stores training metrics in database
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import json

try:
    import asyncpg
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    HAS_DB_LIBS = True
except ImportError:
    HAS_DB_LIBS = False

from models.agent import RLAgent

logger = logging.getLogger(__name__)

# Database schema (simplified for now)
Base = declarative_base() if HAS_DB_LIBS else None

class MetricsTracker:
    def __init__(self, postgres_url: str):
        self.postgres_url = postgres_url
        self.engine = None
        self.session_factory = None
        
        if not HAS_DB_LIBS:
            logger.warning("Database libraries not available. Using in-memory storage.")
            self.memory_storage = []
    
    async def initialize(self):
        """Initialize database connection"""
        if not HAS_DB_LIBS:
            logger.info("Using in-memory metrics storage")
            return
        
        try:
            # Convert postgres:// to postgresql+asyncpg://
            async_url = self.postgres_url.replace('postgresql://', 'postgresql+asyncpg://')
            
            self.engine = create_async_engine(async_url, echo=False)
            self.session_factory = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Create tables (in a real implementation, use Alembic migrations)
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.info("Falling back to in-memory storage")
            self.memory_storage = []
    
    async def record_generation(self, 
                              generation: int, 
                              population: List[RLAgent], 
                              fitness_scores: List[float],
                              raw_fitness_scores: List[float] = None,
                              generation_metrics: Dict[str, Any] = None,
                              gating_summary: Dict[str, Any] = None):
        """Record generation metrics"""
        timestamp = datetime.now()
        raw_scores = raw_fitness_scores if raw_fitness_scores is not None else fitness_scores
        
        generation_data = {
            'generation': generation,
            'timestamp': timestamp.isoformat(),
            'population_size': len(population),
            'fitness_stats': {
                'max': max(fitness_scores),
                'min': min(fitness_scores),
                'mean': sum(fitness_scores) / len(fitness_scores),
                'std': self._calculate_std(fitness_scores)
            },
            'raw_fitness_stats': {
                'max': max(raw_scores),
                'min': min(raw_scores),
                'mean': sum(raw_scores) / len(raw_scores),
                'std': self._calculate_std(raw_scores)
            },
            'generation_metrics': generation_metrics or {},
            'gating_summary': gating_summary or {},
            'agent_details': [
                {
                    'agent_id': agent.id,
                    'fitness': fitness_scores[i],
                    'raw_fitness': raw_scores[i] if i < len(raw_scores) else fitness_scores[i],
                    'games_played': agent.games_played,
                    'wins': agent.wins,
                    'total_vp': agent.total_victory_points,
                    'behavior': agent.get_behavior_stats() if hasattr(agent, 'get_behavior_stats') else {},
                    'config': {
                        'learning_rate': agent.config.learning_rate,
                        'epsilon': agent.config.epsilon,
                        'temperature': agent.config.temperature,
                        'hidden_size': agent.config.hidden_size
                    },
                    'elo': self.get_elo(agent.id)
                }
                for i, agent in enumerate(population)
            ]
        }
        
        if HAS_DB_LIBS and self.session_factory:
            await self._store_in_database(generation_data)
        else:
            self._store_in_memory(generation_data)
        
        logger.info(f"Recorded metrics for generation {generation}")
    
    async def record_tournament(self, tournament_data: Dict[str, Any]):
        """Record tournament results"""
        timestamp = datetime.now()
        
        tournament_record = {
            'tournament_id': tournament_data['tournament_id'],
            'timestamp': timestamp.isoformat(),
            'agents': tournament_data['agents'],
            'games': tournament_data['games'],
            'duration_seconds': tournament_data['duration_seconds'],
            'completed_games': tournament_data['completed_games']
        }
        if HAS_DB_LIBS and self.session_factory:
            await self._store_tournament_in_database(tournament_record)
        else:
            self._store_in_memory(tournament_record)

    def update_elo(self, agent_id_1: str, agent_id_2: str, score_1: float, score_2: float, k_factor: int = 32):
        """Update Elo ratings for two agents based on a match result."""
        if not hasattr(self, 'elo_ratings'):
            self.elo_ratings: Dict[str, float] = {}

        rating_1 = self.elo_ratings.get(agent_id_1, 1000.0)
        rating_2 = self.elo_ratings.get(agent_id_2, 1000.0)

        expected_1 = 1 / (1 + 10 ** ((rating_2 - rating_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((rating_1 - rating_2) / 400))

        # Handle draws or relative wins based on score
        if score_1 > score_2:
            s_1, s_2 = 1.0, 0.0
        elif score_2 > score_1:
            s_1, s_2 = 0.0, 1.0
        else:
            s_1, s_2 = 0.5, 0.5

        new_rating_1 = rating_1 + k_factor * (s_1 - expected_1)
        new_rating_2 = rating_2 + k_factor * (s_2 - expected_2)

        self.elo_ratings[agent_id_1] = new_rating_1
        self.elo_ratings[agent_id_2] = new_rating_2
        
    def get_elo(self, agent_id: str) -> float:
        if not hasattr(self, 'elo_ratings'):
            return 1000.0
        return self.elo_ratings.get(agent_id, 1000.0)

    async def get_generation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical generation data"""
        if HAS_DB_LIBS and self.session_factory:
            return await self._get_from_database(limit)
        else:
            return self.memory_storage[-limit:] if hasattr(self, 'memory_storage') else []
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _store_in_memory(self, data: Dict[str, Any]):
        """Store data in memory (fallback)"""
        if not hasattr(self, 'memory_storage'):
            self.memory_storage = []
        
        self.memory_storage.append(data)
        
        # Keep only last 1000 records
        if len(self.memory_storage) > 1000:
            self.memory_storage = self.memory_storage[-1000:]
    
    async def _store_in_database(self, data: Dict[str, Any]):
        """Store generation data in database"""
        try:
            async with self.session_factory() as session:
                # In a real implementation, insert into proper tables
                # For now, we'll simulate with logging
                logger.debug(f"Would store in DB: {data['generation']}")
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store in database: {e}")
            self._store_in_memory(data)
    
    async def _store_tournament_in_database(self, data: Dict[str, Any]):
        """Store tournament data in database"""
        try:
            async with self.session_factory() as session:
                # In a real implementation, insert into tournament tables
                logger.debug(f"Would store tournament in DB: {data['tournament_id']}")
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to store tournament in database: {e}")
            self._store_in_memory(data)
    
    async def _get_from_database(self, limit: int) -> List[Dict[str, Any]]:
        """Get data from database"""
        try:
            async with self.session_factory() as session:
                # In a real implementation, query from tables
                # For now, return empty list
                return []
        except Exception as e:
            logger.error(f"Failed to get from database: {e}")
            return self.memory_storage[-limit:] if hasattr(self, 'memory_storage') else []
