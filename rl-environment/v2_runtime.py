"""Isolation and safety checks for the clean TFM RL v2 experiment."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def _enabled(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def initialize_v2_runtime() -> Dict[str, str]:
    """Create isolated v2 directories and reject accidental legacy resume."""
    if not _enabled("TFM_RL_V2"):
        return {}

    root = Path(os.getenv("TFM_RL_V2_ROOT", "/app/v2")).expanduser().resolve()
    paths = {
        "root": root,
        "models": Path(os.getenv("RL_MODELS_DIR", root / "models")).expanduser().resolve(),
        "checkpoints": Path(os.getenv("RL_CHECKPOINT_DIR", root / "checkpoints")).expanduser().resolve(),
        "rollouts": Path(os.getenv("PPO_ROLLOUT_SHARD_DIR", root / "rollouts")).expanduser().resolve(),
        "teacher": Path(os.getenv("V2_TEACHER_DATASET_DIR", root / "teacher-dataset")).expanduser().resolve(),
        "benchmarks": Path(os.getenv("V2_BENCHMARK_DIR", root / "benchmarks")).expanduser().resolve(),
        "metrics": Path(os.getenv("V2_METRICS_DIR", root / "metrics")).expanduser().resolve(),
        "logs": Path(os.getenv("RL_LOG_DIR", root / "logs")).expanduser().resolve(),
    }
    for name, path in paths.items():
        if name == "root":
            continue
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"v2 path escapes TFM_RL_V2_ROOT: {name}={path}") from exc

    allow_resume = _enabled("V2_ALLOW_RESUME")
    checkpoint_dir = paths["checkpoints"]
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not allow_resume:
        raise RuntimeError(
            f"Refusing to start clean v2 with non-empty checkpoint directory: {checkpoint_dir}. "
            "Set V2_ALLOW_RESUME=1 only for an intentional v2 resume."
        )
    if str(os.getenv("BOOTSTRAP_CHECKPOINT_PATH", "")).strip():
        raise RuntimeError("TFM RL v2 forbids BOOTSTRAP_CHECKPOINT_PATH")
    if _enabled("RESUME_TRAINING", default=False) and not allow_resume:
        raise RuntimeError("TFM RL v2 requires RESUME_TRAINING=0 unless V2_ALLOW_RESUME=1")

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = paths["root"] / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "tfm_rl_v2.runtime.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "fresh_weights": True,
                    "legacy_checkpoint_discovery": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {name: str(path) for name, path in paths.items()}
