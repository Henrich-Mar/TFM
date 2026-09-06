"""Isolation and safety checks for TFM RL v2 and warm-started v3."""
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
    """Create isolated experiment directories and reject accidental resume."""
    is_v3 = _enabled("TFM_RL_V3")
    if not is_v3 and not _enabled("TFM_RL_V2"):
        return {}

    version = "v3" if is_v3 else "v2"
    prefix = "V3" if is_v3 else "V2"
    root_env = "TFM_RL_V3_ROOT" if is_v3 else "TFM_RL_V2_ROOT"
    root = Path(os.getenv(root_env, f"/app/{version}")).expanduser().resolve()
    paths = {
        "root": root,
        "models": Path(os.getenv("RL_MODELS_DIR", root / "models")).expanduser().resolve(),
        "checkpoints": Path(os.getenv("RL_CHECKPOINT_DIR", root / "checkpoints")).expanduser().resolve(),
        "rollouts": Path(os.getenv("PPO_ROLLOUT_SHARD_DIR", root / "rollouts")).expanduser().resolve(),
        "teacher": Path(os.getenv(f"{prefix}_TEACHER_DATASET_DIR", root / "teacher-dataset")).expanduser().resolve(),
        "benchmarks": Path(os.getenv(f"{prefix}_BENCHMARK_DIR", root / "benchmarks")).expanduser().resolve(),
        "metrics": Path(os.getenv(f"{prefix}_METRICS_DIR", root / "metrics")).expanduser().resolve(),
        "logs": Path(os.getenv("RL_LOG_DIR", root / "logs")).expanduser().resolve(),
    }
    for name, path in paths.items():
        if name == "root":
            continue
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"{version} path escapes {root_env}: {name}={path}") from exc

    allow_resume_env = f"{prefix}_ALLOW_RESUME"
    allow_resume = _enabled(allow_resume_env)
    checkpoint_dir = paths["checkpoints"]
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not allow_resume:
        raise RuntimeError(
            f"Refusing to start clean {version} with non-empty checkpoint directory: {checkpoint_dir}. "
            f"Set {allow_resume_env}=1 only for an intentional {version} resume."
        )
    if str(os.getenv("BOOTSTRAP_CHECKPOINT_PATH", "")).strip():
        raise RuntimeError(f"TFM RL {version} forbids BOOTSTRAP_CHECKPOINT_PATH")
    if _enabled("RESUME_TRAINING", default=False) and not allow_resume:
        raise RuntimeError(f"TFM RL {version} requires RESUME_TRAINING=0 unless {allow_resume_env}=1")

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = paths["root"] / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": f"tfm_rl_{version}.runtime.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "fresh_weights": not is_v3,
                    "warm_started": False,
                    "legacy_checkpoint_discovery": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {name: str(path) for name, path in paths.items()}
