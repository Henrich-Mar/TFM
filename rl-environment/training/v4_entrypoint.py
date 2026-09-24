"""Safe V4 container entrypoint. Initialization never starts PPO."""
from __future__ import annotations

import json
import os

from v2_runtime import initialize_v2_runtime


def main() -> None:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    paths = initialize_v2_runtime()
    print(json.dumps({
        "status": "initialized",
        "message": (
            "TFM RL V4 is isolated. Warm-start from /app/v2/pretrain-h512/bc_best.pth, "
            "collect repaired teacher_sample.v5 decisions with target-family coverage, re-encode human "
            "events with two held-out games, run the adaptive gate-aware pretrain, then execute the "
            "checkpoint-bound smoke test. PPO remains blocked until every held-out gate passes."
        ),
        "paths": paths,
        "pretrain": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "batch_size": 128,
            "epochs_max": 10,
            "early_stopping_patience": 2,
            "human_batch_fraction": 0.0,
            "minimum_teacher_decisions": 100000,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
