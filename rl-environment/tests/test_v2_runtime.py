from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2_runtime import initialize_v2_runtime


def test_v2_runtime_isolated_and_refuses_nonempty_checkpoint(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "v2"
    monkeypatch.setenv("TFM_RL_V2", "1")
    monkeypatch.setenv("TFM_RL_V2_ROOT", str(root))
    monkeypatch.setenv("RL_MODELS_DIR", str(root / "models"))
    monkeypatch.setenv("RL_CHECKPOINT_DIR", str(root / "checkpoints"))
    monkeypatch.setenv("PPO_ROLLOUT_SHARD_DIR", str(root / "rollouts"))
    monkeypatch.setenv("V2_TEACHER_DATASET_DIR", str(root / "teacher"))
    monkeypatch.setenv("V2_BENCHMARK_DIR", str(root / "benchmarks"))
    monkeypatch.setenv("V2_METRICS_DIR", str(root / "metrics"))
    monkeypatch.setenv("RESUME_TRAINING", "0")
    paths = initialize_v2_runtime()
    assert Path(paths["models"]).is_dir()
    (root / "checkpoints" / "old.pth").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty checkpoint"):
        initialize_v2_runtime()
