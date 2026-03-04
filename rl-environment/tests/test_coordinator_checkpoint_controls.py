import os
import sys
from pathlib import Path

import pytest


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

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
