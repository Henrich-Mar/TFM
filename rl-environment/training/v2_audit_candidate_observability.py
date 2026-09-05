"""Measure what a frozen V2 candidate can observe and whether it uses it.

This is intentionally an offline audit.  It only loads an existing checkpoint
and rollout shards; it never starts a game, changes a checkpoint, or mutates a
rollout shard.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from models.agent import AgentConfig, TerraformingMarsNetwork
from models.planner_common import pad_bundle_batch


def _load_checkpoint(path: Path) -> Tuple[TerraformingMarsNetwork, Dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    config = AgentConfig(**{
        key: value
        for key, value in dict(checkpoint.get("config", {})).items()
        if key in AgentConfig.__dataclass_fields__
    })
    network = TerraformingMarsNetwork(config)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()
    return network, checkpoint


def _iter_steps(root: Path) -> Iterable[Any]:
    for path in sorted(root.glob("rollout_*.pkl.gz"), reverse=True):
        with gzip.open(path, "rb") as fh:
            for step in pickle.load(fh):
                yield step


def _copy_bundle(bundle: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in bundle.items()}


def _opponent_rows(bundle: Dict[str, np.ndarray]) -> np.ndarray:
    types = np.asarray(bundle.get("world_token_types", []), dtype=np.int64)
    mask = np.asarray(bundle.get("world_mask", []), dtype=bool)
    return np.flatnonzero((types == 4) & mask)


def _variant(bundle: Dict[str, np.ndarray], kind: str) -> Dict[str, np.ndarray]:
    out = _copy_bundle(bundle)
    rows = _opponent_rows(out)
    if kind == "hide_details":
        # Preserve token count/type. Only opponent values become unavailable.
        out["world_tokens"][rows, 1:] = 0.0
    elif kind == "remove_tokens":
        # Stronger counterfactual: no opponent rows participate in attention.
        out["world_tokens"][rows, :] = 0.0
        out["world_mask"][rows] = False
    elif kind == "max_pressure":
        # Same visible opponents, but each reported resource/production/VP
        # feature is high. This checks response direction, not game realism.
        out["world_tokens"][rows, 1:] = 1.0
    elif kind == "max_vp_only":
        # StateEncoder puts opponent VP in both the resource and trailing
        # summary positions.  This is useful when validating the server's
        # showOtherPlayersVP contract separately from economy information.
        out["world_tokens"][rows, 8] = 1.0
        out["world_tokens"][rows, 16] = 1.0
    else:
        raise ValueError(f"unknown variant: {kind}")
    return out


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _run_policy(
    network: TerraformingMarsNetwork,
    samples: List[Any],
    bundles: List[Dict[str, np.ndarray]],
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = pad_bundle_batch(bundles, device=torch.device("cpu"), planner_config=network.planner_config)
    recurrent = np.zeros((len(samples), network.recurrent_size), dtype=np.float32)
    for row, step in enumerate(samples):
        raw = getattr(step, "recurrent_state", None)
        if raw is None:
            continue
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        recurrent[row, : min(vector.size, network.recurrent_size)] = vector[: network.recurrent_size]
    phases = torch.tensor(
        [int(getattr(step, "phase_index", 0) or 0) for step in samples],
        dtype=torch.long,
    )
    with torch.no_grad():
        output = network(state, phase_indices=phases, recurrent_state=torch.tensor(recurrent))
        logits = output["policy_logits"] / max(float(temperature), 1e-3)
        return (
            torch.softmax(logits, dim=-1).cpu().numpy(),
            output["value"].reshape(-1).cpu().numpy(),
            state["action_mask"].cpu().numpy(),
        )


def _compare(base: np.ndarray, other: np.ndarray, action_mask: np.ndarray) -> Dict[str, Any]:
    tvd: List[float] = []
    kl: List[float] = []
    max_delta: List[float] = []
    flips = 0
    valid_rows = 0
    for row, mask in enumerate(action_mask):
        legal = np.flatnonzero(mask)
        if legal.size < 2:
            continue
        valid_rows += 1
        p = np.clip(base[row, legal], 1e-12, 1.0)
        q = np.clip(other[row, legal], 1e-12, 1.0)
        # The model already masks illegal actions. Normalize defensively.
        p /= p.sum()
        q /= q.sum()
        tvd.append(float(0.5 * np.abs(p - q).sum()))
        kl.append(float(np.sum(p * np.log(p / q))))
        max_delta.append(float(np.abs(p - q).max()))
        flips += int(legal[int(np.argmax(p))] != legal[int(np.argmax(q))])
    return {
        "decision_rows": int(valid_rows),
        "top_action_flip_rate": float(flips / valid_rows) if valid_rows else 0.0,
        "total_variation": _stats(tvd),
        "kl_base_to_variant": _stats(kl),
        "largest_legal_probability_change": _stats(max_delta),
    }


def audit(checkpoint_path: Path, rollout_root: Path, sample_limit: int) -> Dict[str, Any]:
    network, checkpoint = _load_checkpoint(checkpoint_path)
    expected_policy_version = int(checkpoint.get("policy_version", 0) or 0)
    selected: List[Any] = []
    seen_policy_versions: Counter[str] = Counter()
    seen_schema_versions: Counter[str] = Counter()
    for step in _iter_steps(rollout_root):
        seen_policy_versions[str(getattr(step, "policy_version", ""))] += 1
        seen_schema_versions[str(getattr(step, "state_schema_version", ""))] += 1
        bundle = getattr(step, "state_bundle", {})
        if int(getattr(step, "policy_version", -1)) != expected_policy_version:
            continue
        if not isinstance(bundle, dict) or _opponent_rows(bundle).size == 0:
            continue
        if np.asarray(bundle.get("action_mask", []), dtype=bool).sum() < 2:
            continue
        selected.append(step)
        if len(selected) >= max(1, sample_limit):
            break
    if not selected:
        raise RuntimeError(
            f"no usable rollout steps at policy_version={expected_policy_version}; "
            f"observed versions={dict(seen_policy_versions)}"
        )

    base_bundles = [_copy_bundle(step.state_bundle) for step in selected]
    temperature = float(dict(checkpoint.get("config", {})).get("temperature", 1.0) or 1.0)
    base_probs, base_values, action_mask = _run_policy(network, selected, base_bundles, temperature)
    variants = {
        name: _run_policy(network, selected, [_variant(bundle, name) for bundle in base_bundles], temperature)
        for name in ("hide_details", "remove_tokens", "max_pressure", "max_vp_only")
    }
    opponent_counts = [_opponent_rows(bundle).size for bundle in base_bundles]
    return {
        "schema_version": "tfm_rl_v2.candidate_observability_audit.v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_policy_version": expected_policy_version,
        "sample_count": len(selected),
        "rollout_policy_versions_seen": dict(seen_policy_versions),
        "rollout_schema_versions_seen": dict(seen_schema_versions),
        "visible_opponent_tokens": {
            "type_id": 4,
            "tokens_per_sample": _stats([float(value) for value in opponent_counts]),
            "feature_contract": [
                "megaCredits, steel, titanium, plants, energy, heat, TR, current VP",
                "megaCredit/steel/titanium/plant/energy/heat production",
                "tableau count and current VP duplicate",
            ],
        },
        "counterfactuals": {
            name: {
                **_compare(base_probs, probs, action_mask),
                "absolute_value_change": _stats(np.abs(base_values - values).tolist()),
            }
            for name, (probs, values, _mask) in variants.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate reliance on V2 opponent observations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(Path(args.checkpoint), Path(args.rollouts), max(1, int(args.samples)))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
