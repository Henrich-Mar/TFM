"""V4 self-play. PPO stays blocked until the card-aware pretrain gate passes."""
from __future__ import annotations

import argparse
import asyncio
import os

from training.v2_self_play import V2SelfPlayRunner
from training.v4_gates import assert_ppo_unlocked


def main() -> None:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    parser = argparse.ArgumentParser(description="Run TFM RL V4 self-play after the pretrain gate")
    parser.add_argument("--bootstrap-checkpoint", default="/app/v4/pretrain-v2/bc_best.pth")
    parser.add_argument("--root", default=os.getenv("TFM_RL_V4_ROOT", "/app/v4"))
    parser.add_argument("--pretrain-report", default=os.getenv("V4_PRETRAIN_REPORT", "/app/v4/pretrain-v2/pretrain_report.json"))
    parser.add_argument("--max-decisions", type=int, default=1_000_000)
    parser.add_argument("--benchmark-interval", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    os.environ["V4_PRETRAIN_REPORT"] = args.pretrain_report
    assert_ppo_unlocked(args.pretrain_report, args.bootstrap_checkpoint)
    runner = V2SelfPlayRunner(
        args.bootstrap_checkpoint,
        args.root,
        args.benchmark_interval,
        args.seed,
        initial_stage=args.stage,
    )
    asyncio.run(runner.run(args.max_decisions))


if __name__ == "__main__":
    main()
