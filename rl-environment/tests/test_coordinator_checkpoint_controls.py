import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import asyncio

import pytest


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

import coordinator as coordinator_module  # noqa: E402
from coordinator import RLCoordinator  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_should_save_generation_cadence() -> None:
    coordinator = RLCoordinator.__new__(RLCoordinator)
    coordinator.save_every_n_generations = 3
    assert coordinator._should_save_generation(0) is True
    assert coordinator._should_save_generation(1) is False
    assert coordinator._should_save_generation(2) is False
    assert coordinator._should_save_generation(3) is True


def test_training_pool_extra_checkpoints_accepts_existing_absolute_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "global" / "champion" / "current" / "champion.pth"
    _touch(existing)
    relative_missing = "relative/path/does-not-exist.pth"
    raw = f"{existing},{relative_missing}"
    monkeypatch.setenv("TRAINING_POOL_EXTRA_CHECKPOINTS", raw)

    coordinator = RLCoordinator.__new__(RLCoordinator)
    resolved = coordinator._resolve_training_pool_extra_checkpoints()
    assert resolved == [os.path.abspath(str(existing))]


def test_prune_saved_generation_dirs_keeps_last_and_pinned(tmp_path: Path) -> None:
    models_root = tmp_path / "coord-models"
    for generation in range(6):
        _touch(models_root / f"generation_{generation}" / f"agent_0_fitness_{100 - generation:.2f}.pth")

    coordinator = RLCoordinator.__new__(RLCoordinator)
    coordinator._default_models_root = lambda: str(models_root)

    summary = coordinator._prune_saved_generation_dirs(
        keep_last=2,
        pinned_generations={1},
    )
    assert summary["removed_generations"] == [0, 2, 3]
    assert (models_root / "generation_1").is_dir()
    assert (models_root / "generation_4").is_dir()
    assert (models_root / "generation_5").is_dir()


def test_apply_start_generation_override() -> None:
    coordinator = RLCoordinator.__new__(RLCoordinator)
    coordinator.config = SimpleNamespace(generations=500)
    coordinator.training_start_generation_override = 42
    assert coordinator._apply_start_generation_override(7, "test") == 42
    coordinator.training_start_generation_override = None
    assert coordinator._apply_start_generation_override(7, "test") == 7


@dataclass
class _FakeConfig:
    train_from_self_play: bool = True


class _FakeNetwork:
    def __init__(self) -> None:
        self.loaded_state = {}

    def state_dict(self):
        return {"w": 1}

    def load_state_dict(self, state):
        self.loaded_state = dict(state)

    def eval(self):
        return None


class _FakeAgent:
    def __init__(self, config=None):
        self.config = config or _FakeConfig()
        self.network = _FakeNetwork()
        self.id = f"fake-{id(self)}"
        self.loaded_path = ""
        self.last_mutation_rate = None

    def load_model(self, path: str) -> None:
        self.loaded_path = path
        self.config = _FakeConfig()

    def mutate(self, mutation_rate: float = 0.1) -> None:
        self.last_mutation_rate = float(mutation_rate)


def test_bootstrap_from_checkpoint_seeds_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "champion.pth"
    _touch(checkpoint_path)
    monkeypatch.setattr(coordinator_module, "RLAgent", _FakeAgent)

    coordinator = RLCoordinator.__new__(RLCoordinator)
    coordinator.bootstrap_checkpoint_path = str(checkpoint_path)
    coordinator.bootstrap_population_mode = "mutated_copies"
    coordinator.bootstrap_mutation_rate = 0.12
    coordinator.training_start_generation_override = 0
    coordinator.resume_training_enabled = True
    coordinator.config = SimpleNamespace(population_size=3, generations=500)
    coordinator.evolution_manager = SimpleNamespace(generation_count=0)
    coordinator.metrics_tracker = SimpleNamespace(elo_ratings={"old": 1500.0})
    coordinator.league_manager = SimpleNamespace(load_state=lambda state: None)
    coordinator.fixed_benchmark_pool_size = 3
    coordinator.fixed_benchmark_checkpoints = []
    coordinator.last_eval_fitness = {"old": 1.0}
    coordinator.last_raw_eval_fitness = {"old": 1.0}
    coordinator.last_gated_eval_fitness = {"old": 1.0}
    coordinator.last_generation_behavior_metrics = {"x": 1}
    coordinator.last_generation_gate = {"x": 1}
    coordinator.last_selection_diagnostics = {"x": 1}
    coordinator.generation_behavior_history = [{"generation": 1}]
    coordinator.population = []
    coordinator._resolve_fixed_benchmark_checkpoints = lambda: []
    saved = {}
    coordinator._save_training_checkpoint = lambda next_generation: saved.setdefault("next_generation", next_generation)

    ok = asyncio.run(coordinator._load_population_from_bootstrap_checkpoint())

    assert ok is True
    assert len(coordinator.population) == 3
    assert coordinator.current_generation == 0
    assert coordinator.evolution_manager.generation_count == 0
    assert saved["next_generation"] == 0
    assert coordinator.metrics_tracker.elo_ratings == {}
    clone_mutations = [agent.last_mutation_rate for agent in coordinator.population[1:]]
    assert clone_mutations == [0.12, 0.12]
