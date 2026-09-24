"""Partial warm-start of the V4 card-aware policy from the H512 V2 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from models.agent import AgentConfig, TerraformingMarsNetwork
from models.card_catalog import get_catalog
from v2_runtime import initialize_v2_runtime

CARD_MODULE_PREFIXES = (
    "card_identity.",
    "card_metadata_projection.",
    "card_set_projection.",
    "card_count_embedding.",
    "hand_context_norm.",
)
REQUIRED_ARCHITECTURE = {
    "hidden_size": 512,
    "transformer_layers": 3,
    "transformer_heads": 4,
    "planner_token_dim": 64,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_architecture(config: Dict) -> None:
    for key, expected in REQUIRED_ARCHITECTURE.items():
        observed = int(config.get(key, -1) or -1)
        if observed != expected:
            raise RuntimeError(
                f"V4 warm-start requires source {key}={expected}; checkpoint has {observed}"
            )


def _copy_compatible(source_state: Dict[str, torch.Tensor], network: TerraformingMarsNetwork) -> Tuple[List[str], List[str]]:
    destination = network.state_dict()
    reused: List[str] = []
    initialized: List[str] = []
    for name, parameter in destination.items():
        if name.startswith(CARD_MODULE_PREFIXES):
            initialized.append(name)
            continue
        source = source_state.get(name)
        if source is None or tuple(source.shape) != tuple(parameter.shape):
            raise RuntimeError(f"V4 warm-start cannot reuse incompatible parameter {name}")
        parameter.copy_(source)
        reused.append(name)
    unexpected = sorted(set(source_state) - set(destination))
    if unexpected:
        raise RuntimeError(f"V4 warm-start source contains unexpected parameters: {unexpected}")
    network.load_state_dict(destination)
    return reused, initialized


def warm_start(source: str, output: str) -> dict:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    paths = initialize_v2_runtime()
    root = Path(paths["root"]).resolve()
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"V2 source checkpoint does not exist: {source_path}")
    try:
        output_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"V4 bootstrap output must stay under {root}: {output_path}") from exc

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config_payload = dict(checkpoint.get("config") or {})
    _require_source_architecture(config_payload)
    config = AgentConfig(
        hidden_size=512,
        recurrent_size=int(config_payload.get("recurrent_size", 128) or 128),
        transformer_heads=4,
        transformer_layers=3,
        planner_token_dim=64,
        learning_rate=1e-4,
    )
    network = TerraformingMarsNetwork(config)
    fresh_card_state = {
        name: parameter.detach().clone()
        for name, parameter in network.state_dict().items()
        if name.startswith(CARD_MODULE_PREFIXES)
    }
    reused, initialized = _copy_compatible(dict(checkpoint.get("network_state_dict") or {}), network)
    loaded_card_state = {
        name: parameter.detach().clone()
        for name, parameter in network.state_dict().items()
        if name.startswith(CARD_MODULE_PREFIXES)
    }
    for name, fresh in fresh_card_state.items():
        if not torch.equal(fresh, loaded_card_state[name]):
            raise RuntimeError(f"V4 warm-start overwrote a freshly initialized card parameter: {name}")

    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "tfm_rl_v4.bc_checkpoint.v1",
            "experiment_version": "tfm-rl-v4",
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "hidden_size": 512,
                "recurrent_size": config.recurrent_size,
                "transformer_heads": 4,
                "transformer_layers": 3,
                "planner_token_dim": 64,
            },
            "games_played": 0,
            "total_victory_points": 0,
            "wins": 0,
            "policy_version": 0,
            "card_catalog_sha256": get_catalog().sha256,
            "state_schema_version": "v4-card-aware.v1",
            "fresh_weights": False,
        },
        output_path,
    )
    payload = {
        "schema_version": "tfm_rl_v4.warm_start.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_checkpoint": str(output_path),
        "output_sha256": _sha256(output_path),
        "card_catalog_sha256": get_catalog().sha256,
        "reused_parameters": reused,
        "initialized_parameters": initialized,
        "optimizer_reset": True,
        "policy_version_reset": True,
        "training_statistics_reset": True,
    }
    report_path = output_path.with_name("warm_start_report.json")
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["warm_started"] = True
    manifest["warm_start"] = {
        "source": str(source_path),
        "source_sha256": payload["source_sha256"],
        "checkpoint": str(output_path),
        "checkpoint_sha256": payload["output_sha256"],
        "card_catalog_sha256": payload["card_catalog_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start TFM RL V4 from the H512 V2 behavior-cloning checkpoint")
    parser.add_argument("--source", default="/app/v2/pretrain-h512/bc_best.pth")
    parser.add_argument("--output", default="/app/v4/bootstrap/warm_start.pth")
    args = parser.parse_args()
    print(json.dumps(warm_start(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
