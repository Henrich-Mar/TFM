from types import SimpleNamespace

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolution import EvolutionManager


class _FakeAgent:
    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.network = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(
            learning_rate=1e-4 * seed,
            epsilon=0.01 * seed,
            temperature=0.5 * seed,
        )


def test_agent_distance_uses_cpu_safe_parameter_comparison() -> None:
    manager = EvolutionManager(SimpleNamespace(population_size=2, elite_percentage=0.2, mutation_rate=0.1))

    agent1 = _FakeAgent(1)
    agent2 = _FakeAgent(2)

    distance = manager._calculate_agent_distance(agent1, agent2)

    assert isinstance(distance, float)
    assert distance > 0.0
