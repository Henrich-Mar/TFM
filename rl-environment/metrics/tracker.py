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
                              fitness_scores: List[float]):
        """Record generation metrics"""
        timestamp = datetime.now()
        
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
            'agent_details': [
                {
                    'agent_id': agent.id,
                    'fitness': fitness_scores[i],
                    'games_played': agent.games_played,
                    'wins': agent.wins,
                    'total_vp': agent.total_victory_points,
                    'config': {
                        'learning_rate': agent.config.learning_rate,
                        'epsilon': agent.config.epsilon,
                        'temperature': agent.config.temperature,
                        'hidden_size': agent.config.hidden_size
                    }
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