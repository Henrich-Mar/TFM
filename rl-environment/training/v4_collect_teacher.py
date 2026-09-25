"""Collect V4 teacher games on the same base-game stage as the existing dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from training.v2_collect_teacher import collect
from training.v4_teacher_ab import assert_teacher_ab_allows_collection


def pin_stage_options(stage: int) -> str:
    """Force the options file for this stage.

    The V4 compose file points at corporate-era stage 1. Teacher rows already
    stored for pretraining were collected with base-game stage 0. Pinning the
    file here keeps a later run from mixing those curricula.
    """
    name = f"game_options.v2_stage{int(stage)}.json"
    path = Path(__file__).resolve().parents[1] / name
    if not path.is_file():
        raise FileNotFoundError(f"V4 teacher collection options file does not exist: {path}")
    os.environ["GAME_OPTIONS_FILE"] = str(path)
    return str(path)


def main() -> None:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    parser = argparse.ArgumentParser(description="Collect TFM RL v4 card-aware teacher games")
    parser.add_argument("--dataset", default=os.getenv("V4_TEACHER_DATASET_DIR", "/app/v4/teacher-dataset-v5"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed-start", type=int, default=920040)
    parser.add_argument(
        "--teacher-ab",
        default=os.getenv("V4_TEACHER_AB_REPORT", "/app/v4/diagnostics/teacher_ab.json"),
        help="Stage 1 collection requires a reachability-teacher A/B verdict of collect",
    )
    args = parser.parse_args()
    if int(args.stage) == 1:
        assert_teacher_ab_allows_collection(args.teacher_ab)
    options = pin_stage_options(args.stage)
    print(f"[teacher] V4 options={options} dataset={args.dataset}", flush=True)
    print(
        json.dumps(
            asyncio.run(collect(args.dataset, args.games, args.stage, args.seed_start)),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
