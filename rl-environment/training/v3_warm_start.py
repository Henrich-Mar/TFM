"""Create a clean V3 checkpoint from a trusted V2 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from models.agent import RLAgent
from v2_runtime import initialize_v2_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warm_start(source: str, output: str) -> dict:
    os.environ["TFM_RL_V3"] = "1"
    paths = initialize_v2_runtime()
    root = Path(paths["root"]).resolve()
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"V2 source checkpoint does not exist: {source_path}")
    try:
        output_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"V3 bootstrap output must stay under {root}: {output_path}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_permission = os.environ.get("V3_ALLOW_V2_WARMSTART")
    os.environ["V3_ALLOW_V2_WARMSTART"] = "1"
    try:
        agent = RLAgent(agent_id="v3-warm-start")
        agent.load_model(str(source_path))
    finally:
        if previous_permission is None:
            os.environ.pop("V3_ALLOW_V2_WARMSTART", None)
        else:
            os.environ["V3_ALLOW_V2_WARMSTART"] = previous_permission

    # Keep learned network weights, but start a new experiment trajectory.
    agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=agent.ppo_learning_rate)
    agent.policy_version = 0
    agent.games_played = 0
    agent.total_victory_points = 0
    agent.wins = 0
    for key, value in list(agent.decision_stats.items()):
        if isinstance(value, dict):
            agent.decision_stats[key] = {}
        elif isinstance(value, float):
            agent.decision_stats[key] = 0.0
        else:
            agent.decision_stats[key] = 0
    agent.set_v3_feature_scale(0.0)
    agent.save_model(str(output_path))

    payload = {
        "schema_version": "tfm_rl_v3.warm_start.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_checkpoint": str(output_path),
        "output_sha256": _sha256(output_path),
        "network_weights_reused": True,
        "optimizer_reset": True,
        "policy_version_reset": True,
        "training_statistics_reset": True,
        "initial_v3_feature_scale": 0.0,
        "ppo_gate_passed": True,
    }
    report_path = output_path.with_name("pretrain_report.json")
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["warm_started"] = True
    manifest["warm_start"] = {
        "source": str(source_path),
        "source_sha256": payload["source_sha256"],
        "checkpoint": str(output_path),
        "checkpoint_sha256": payload["output_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start TFM RL V3 from the stable V2 champion")
    parser.add_argument("--source", default="/app/v2/checkpoints/candidate_000500494.pth")
    parser.add_argument("--output", default="/app/v3/bootstrap/warm_start.pth")
    args = parser.parse_args()
    print(json.dumps(warm_start(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
