"""Run and record the checkpoint-bound fixed-seed V4 inference smoke test."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from training.v4_gates import SMOKE_CONFIGURATION, SMOKE_SEED, file_sha256, record_smoke_result


async def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    supplied_configuration = {
        "baseline": "teacher",
        "stage": int(args.stage),
        "candidate_seat": int(args.candidate_seat),
        "deterministic_actions": True,
        "ppo_enabled": False,
    }
    if int(args.seed) != SMOKE_SEED or supplied_configuration != SMOKE_CONFIGURATION:
        raise ValueError(
            f"promotion smoke must use seed={SMOKE_SEED} and configuration={SMOKE_CONFIGURATION}"
        )
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ["TFM_RL_V2"] = "0"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    from training.v2_trace_candidate_game import trace

    trace_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        baseline="teacher",
        stage=args.stage,
        seed=args.seed,
        candidate_seat=args.candidate_seat,
        game_servers=args.game_servers,
    )
    report = await trace(trace_args)
    candidate = report.get("candidate_result") or {}
    behavior = report.get("candidate_behavior") or {}
    if "action_rejected_by_server" not in behavior:
        raise RuntimeError("candidate smoke trace did not record server-rejection telemetry")
    rejected = int(behavior["action_rejected_by_server"] or 0)
    evidence = {
        "completed": bool(report.get("completed", False)),
        "server_rejected_actions": rejected,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": int(args.seed),
        "configuration": supplied_configuration,
        "decision_count": int(report.get("decision_count", 0) or 0),
    }
    if args.trace_output:
        trace_output = Path(args.trace_output)
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-seed V4 candidate smoke without PPO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrain-report", required=True)
    parser.add_argument("--game-servers", required=True)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--trace-output")
    args = parser.parse_args()
    evidence = asyncio.run(run_smoke(args))
    report = record_smoke_result(evidence, args.pretrain_report)
    print(json.dumps({
        "smoke": evidence,
        "ppo_gate_passed": report["ppo_gate_passed"],
        "ppo_gate_reasons": report["ppo_gate_reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
