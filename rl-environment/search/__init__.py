"""AlphaGo MCTS phase 2: Python information-set search consumer.

This package owns the search tree, neural leaf evaluation, and batched
simulation requests against the TypeScript ``/api/rl/search`` service.
The game server remains the authoritative rules simulator.
"""
from .config import SearchConfig
from .search_agent import SearchDecision, SearchPolicy
from .simulator_client import (
    BranchResult,
    SearchClient,
    SearchServiceError,
    SearchUnavailableError,
    StartResult,
)

__all__ = [
    "BranchResult",
    "SearchClient",
    "SearchConfig",
    "SearchDecision",
    "SearchPolicy",
    "SearchServiceError",
    "SearchUnavailableError",
    "StartResult",
]
