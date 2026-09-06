"""V3 feature-ramped curriculum using the proven V2 self-play loop."""
from __future__ import annotations

import argparse
import asyncio
import os

from training.v2_self_play import V2SelfPlayRunner


def main() -> None:
    os.environ["TFM_RL_V3"] = "1"
    parser = argparse.ArgumentParser(description="Run warm-started TFM RL V3 curriculum self-play")
    parser.add_argument("--bootstrap-checkpoint", default="/app/v3/bootstrap/warm_start.pth")
    parser.add_argument("--root", default=os.getenv("TFM_RL_V3_ROOT", "/app/v3"))
    parser.add_argument("--max-decisions", type=int, default=1_000_000)
    parser.add_argument("--benchmark-interval", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=600_000)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
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
