#!/usr/bin/env python3
"""
Wrapper for the canonical planner-policy benchmark entrypoint.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "rl-environment" / "benchmark_training.py"
    os.chdir(str(repo_root))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
