import asyncio
from types import SimpleNamespace
from typing import Optional, Tuple

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolution import EvolutionManager


class _FakeAgent:
    def __init__(self, seed: int, agent_id: Optional[str] = None, parent_ids: Tuple[str, ...] = ()) -> None:
        torch.manual_seed(seed)
        self.id = agent_id or f"agent-{seed}"
        self.network = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(
            learning_rate=1e-4 * seed,
            epsilon=0.01 * seed,
            temperature=0.5 * seed,
        )
        self.parent_ids = parent_ids
        self.mutation_calls: list[float] = []

    def mutate(self, mutation_rate: float = 0.1) -> None:
        self.mutation_calls.append(float(mutation_rate))

    def crossover(self, other_agent: "_FakeAgent") -> "_FakeAgent":
        return _FakeAgent(
            seed=999,
            agent_id=f"{self.id}-x-{other_agent.id}",
            parent_ids=(self.id, other_agent.id),
        )


def test_agent_distance_uses_cpu_safe_parameter_comparison() -> None:
    manager = EvolutionManager(SimpleNamespace(population_size=2, elite_percentage=0.2, mutation_rate=0.1))

    agent1 = _FakeAgent(1)
    agent2 = _FakeAgent(2)

    distance = manager._calculate_agent_distance(agent1, agent2)

    assert isinstance(distance, float)
    assert distance > 0.0


def test_rl_first_replaces_gate_failed_agents_from_passed_pool(monkeypatch) -> None:
    manager = EvolutionManager(SimpleNamespace(population_size=4, elite_percentage=0.25, mutation_rate=0.1))
    manager.rl_first_enabled = True
    manager.crossover_rate = 0.0
    manager.immigrant_interval = 99
    manager.gate_replacement_mutation_rate = 0.05

    population = [
        _FakeAgent(1, "agent-a"),
        _FakeAgent(2, "agent-b"),
        _FakeAgent(3, "agent-c"),
        _FakeAgent(4, "agent-d"),
    ]
    created: list[_FakeAgent] = []

    def _clone_agent(agent: _FakeAgent) -> _FakeAgent:
        clone = _FakeAgent(100 + len(created), f"{agent.id}-clone-{len(created)}", parent_ids=(agent.id,))
        created.append(clone)
        return clone

    monkeypatch.setattr(manager, "_clone_agent", _clone_agent)

    new_population = asyncio.run(
        manager.evolve_population(
            population,
            [40.0, 30.0, 20.0, 10.0],
            gate_results={
                "agent-a": {"passed": True},
                "agent-b": {"passed": True},
                "agent-c": {"passed": False},
                "agent-d": {"passed": False},
            },
        )
    )

    new_ids = [agent.id for agent in new_population]
    assert new_ids[:2] == ["agent-a", "agent-b"]
    assert "agent-c" not in new_ids
    assert "agent-d" not in new_ids
    assert len(created) == 2
    assert all(clone.parent_ids[0] in {"agent-a", "agent-b"} for clone in created)
    assert all(clone.mutation_calls == [0.05] for clone in created)


def test_selection_filters_gate_failed_agents_from_parent_pool(monkeypatch) -> None:
    manager = EvolutionManager(SimpleNamespace(population_size=4, elite_percentage=0.25, mutation_rate=0.1))
    manager.rl_first_enabled = False
    manager.diversity_bonus = 0.0

    population = [
        _FakeAgent(1, "agent-a"),
        _FakeAgent(2, "agent-b"),
        _FakeAgent(3, "agent-c"),
        _FakeAgent(4, "agent-d"),
    ]
    captured: dict[str, list[_FakeAgent]] = {}

    async def _capture_generation(elite_agents, breeding_pool):
        captured["elite"] = list(elite_agents)
        captured["breeding"] = list(breeding_pool)
        return list(elite_agents) + list(breeding_pool)

    monkeypatch.setattr(manager, "_create_new_generation", _capture_generation)

    asyncio.run(
        manager.evolve_population(
            population,
            [100.0, 90.0, 80.0, 70.0],
            gate_results={
                "agent-a": {"passed": False},
                "agent-b": {"passed": True},
                "agent-c": {"passed": True},
                "agent-d": {"passed": False},
            },
        )
    )

    assert [agent.id for agent in captured["elite"]] == ["agent-b"]
    assert [agent.id for agent in captured["breeding"]] == ["agent-b", "agent-c"]
