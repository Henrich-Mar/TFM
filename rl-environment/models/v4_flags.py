"""Environment switches for the isolated TFM RL v4 experiment."""
from __future__ import annotations

import os


def _enabled(name: str) -> bool:
    return str(os.getenv(name, "0")).strip().lower() in {"1", "true", "yes", "on"}


def v4_enabled() -> bool:
    return _enabled("TFM_RL_V4")


def v3_enabled() -> bool:
    return _enabled("TFM_RL_V3") or v4_enabled()
