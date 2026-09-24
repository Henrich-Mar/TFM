"""Read-only V4 checkpoint diagnostics for placement and award imitation misses."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import numpy as np

# The existing epoch-11 shards were written by a newer NumPy whose pickle
# module path is ``numpy._core``. Keep the diagnostic read-only and portable
# across the older local NumPy used by the test environment.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

from models.agent import AgentConfig, TerraformingMarsNetwork
from models.planner_common import pad_bundle_batch
from training.v4_gates import file_sha256


TEACHER_TEMPERATURE = 0.18
PLACEMENT_FIELDS = (
    "total_value", "self_value", "deny_value", "risk_value", "bonus_value",
    "own_city_adjacent", "enemy_city_adjacent", "own_greenery_adjacent",
    "enemy_greenery_adjacent", "ocean_adjacent", "empty_adjacent", "x", "y",
)
AWARD_TAIL_FIELDS = (
    "identity_0", "identity_1", "identity_2", "identity_3", "identity_4", "identity_5",
    "own_score", "opponent_score", "score_gap", "projected_vp", "cost",
    "affordability", "timing", "early_risk",
)


def _iter_raw_samples(dataset: Path, split: str) -> Iterable[Dict[str, Any]]:
    for path in sorted((dataset / split).glob("episode_*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle) or []
        for item in payload:
            if isinstance(item, dict):
                yield item


def _config_from_checkpoint(checkpoint: Mapping[str, Any]) -> AgentConfig:
    defaults = asdict(AgentConfig())
    supplied = checkpoint.get("config") or {}
    if isinstance(supplied, dict):
        defaults.update({key: value for key, value in supplied.items() if key in defaults})
    return AgentConfig(**defaults)


def _nested_action_type(payload: Any) -> str:
    current = payload
    while isinstance(current, dict):
        kind = str(current.get("type", "") or "").lower()
        if kind != "or":
            return kind
        current = current.get("response")
    return ""


def _descriptor_name(descriptor: Mapping[str, Any]) -> str:
    return str(
        descriptor.get("space_id")
        or descriptor.get("award_name")
        or descriptor.get("milestone_name")
        or descriptor.get("card_name")
        or descriptor.get("label")
        or descriptor.get("action_index")
        or "?"
    )


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    if not count:
        return {"count": 0, "top1": 0.0, "top2": 0.0, "top3": 0.0, "mrr": 0.0}
    return {
        "count": count,
        "top1": sum(int(row["target_rank"] <= 1) for row in rows) / count,
        "top2": sum(int(row["target_rank"] <= 2) for row in rows) / count,
        "top3": sum(int(row["target_rank"] <= 3) for row in rows) / count,
        "mrr": sum(1.0 / int(row["target_rank"]) for row in rows) / count,
        "mean_candidates": sum(int(row["candidate_count"]) for row in rows) / count,
        "mean_logit_margin": sum(float(row["logit_margin"]) for row in rows) / count,
        "mean_teacher_score_gap": sum(float(row["teacher_score_gap"]) for row in rows) / count,
    }


def _cohort_report(rows: Sequence[Mapping[str, Any]], family_field: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[family_field])].append(row)
    return {family: _metrics(items) for family, items in sorted(grouped.items())}


def _identity_confusion(rows: Sequence[Mapping[str, Any]], family_field: str) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row[family_field])
        grouped[family][
            f"{_descriptor_name(row['target_descriptor'])}->{_descriptor_name(row['predicted_descriptor'])}"
        ] += 1
    return {family: dict(confusion) for family, confusion in sorted(grouped.items())}


def _placement_report(
    rows: Sequence[Mapping[str, Any]],
    cohort_descriptor: str = "target_descriptor",
) -> Dict[str, Any]:
    tile_rows = [
        row for row in rows
        if _nested_action_type(row[cohort_descriptor].get("decoded_action")) == "space"
    ]
    non_tile = [
        row for row in rows
        if _nested_action_type(row[cohort_descriptor].get("decoded_action")) != "space"
    ]
    misses = [row for row in tile_rows if int(row["target_rank"]) > 1]
    near_ties = [row for row in misses if float(row["teacher_score_gap"]) <= TEACHER_TEMPERATURE]
    deltas: Dict[str, List[float]] = defaultdict(list)
    intent_confusion: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []
    for row in misses:
        target = row["target_descriptor"].get("space_features") or {}
        predicted = row["predicted_descriptor"].get("space_features") or {}
        for field in PLACEMENT_FIELDS:
            try:
                deltas[field].append(float(predicted.get(field, 0.0) or 0.0) - float(target.get(field, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        intent_confusion[f"{target.get('intent', '?')}->{predicted.get('intent', '?')}"] += 1
        if len(examples) < 50:
            examples.append({
                "sample_id": row["sample_id"],
                "rank": row["target_rank"],
                "teacher_score_gap": row["teacher_score_gap"],
                "target": {"name": _descriptor_name(row["target_descriptor"]), **target},
                "predicted": {"name": _descriptor_name(row["predicted_descriptor"]), **predicted},
            })
    tile_metrics = _metrics(tile_rows)
    near_tie_rate = (len(near_ties) / len(misses)) if misses else 1.0
    qualified = bool(tile_rows and tile_metrics["top3"] >= 0.97 and near_tie_rate >= 0.80)
    return {
        "tile": tile_metrics,
        "non_tile": _metrics(non_tile),
        "top1_misses": len(misses),
        "near_tie_misses": len(near_ties),
        "near_tie_rate": near_tie_rate,
        "mean_predicted_minus_target": {
            field: (sum(values) / len(values)) for field, values in sorted(deltas.items()) if values
        },
        "intent_confusion": dict(intent_confusion),
        "miss_examples": examples,
        "qualified_for_top3_gate": qualified,
    }


def _award_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    confusion: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []
    for row in rows:
        target = row["target_descriptor"]
        predicted = row["predicted_descriptor"]
        confusion[f"{_descriptor_name(target)}->{_descriptor_name(predicted)}"] += 1
        if int(row["target_rank"]) > 1 and len(examples) < 50:
            examples.append({
                "sample_id": row["sample_id"],
                "rank": row["target_rank"],
                "teacher_score_gap": row["teacher_score_gap"],
                "target": _award_features(target),
                "predicted": _award_features(predicted),
            })
    return {"metrics": _metrics(rows), "confusion": dict(confusion), "miss_examples": examples}


def _award_features(descriptor: Mapping[str, Any]) -> Dict[str, Any]:
    raw_token = descriptor.get("token_features")
    token = list(raw_token) if raw_token is not None else []
    tail = token[50:64] if len(token) >= 64 else []
    return {
        "name": _descriptor_name(descriptor),
        "family": str(descriptor.get("family", "other") or "other"),
        "encoded": {name: float(tail[index]) for index, name in enumerate(AWARD_TAIL_FIELDS[:len(tail)])},
    }


def diagnose(checkpoint_path: str, dataset_dir: str, split: str = "test", batch_size: int = 128) -> Dict[str, Any]:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    dataset = Path(dataset_dir).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    network = TerraformingMarsNetwork(_config_from_checkpoint(checkpoint))
    network.load_state_dict(checkpoint["network_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device).eval()

    samples = list(_iter_raw_samples(dataset, split))
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(samples), max(1, int(batch_size))):
            batch = samples[start:start + max(1, int(batch_size))]
            bundles = pad_bundle_batch(
                [item["planner_bundle"] for item in batch],
                device=device,
                planner_config=network.planner_config,
            )
            phases = torch.tensor([int(item.get("phase_index", 0)) for item in batch], device=device)
            logits = network(bundles, phase_indices=phases)["policy_logits"].float().cpu()
            for index, sample in enumerate(batch):
                descriptors = list(sample.get("action_descriptors") or [])
                probabilities = [float(value) for value in (sample.get("teacher_probabilities") or [])]
                if not descriptors or len(descriptors) != len(probabilities):
                    continue
                target = max(range(len(probabilities)), key=probabilities.__getitem__)
                chosen = int(sample.get("chosen_action_position", target))
                order = torch.argsort(logits[index, :len(descriptors)], descending=True).tolist()
                predicted = int(order[0])
                rank = int(order.index(target) + 1)
                wrong_best = next((position for position in order if position != target), target)
                logit_margin = float(logits[index, target] - logits[index, wrong_best])
                p_target = max(probabilities[target], 1e-12)
                p_predicted = max(probabilities[predicted], 1e-12)
                teacher_gap = max(0.0, TEACHER_TEMPERATURE * math.log(p_target / p_predicted))
                rows.append({
                    "sample_id": str(sample.get("sample_id", "") or ""),
                    "target_rank": rank,
                    "candidate_count": len(descriptors),
                    "logit_margin": logit_margin,
                    "teacher_score_gap": teacher_gap,
                    "target_family": str(descriptors[target].get("family", "other") or "other"),
                    "executed_family": str(descriptors[chosen].get("family", "other") or "other") if 0 <= chosen < len(descriptors) else "other",
                    "target_descriptor": descriptors[target],
                    "chosen_descriptor": descriptors[chosen] if 0 <= chosen < len(descriptors) else {},
                    "predicted_descriptor": descriptors[predicted],
                })

    target_placement = [row for row in rows if row["target_family"] == "select_space"]
    executed_placement = [row for row in rows if row["executed_family"] == "select_space"]
    target_awards = [row for row in rows if row["target_family"] == "fund_award"]
    executed_awards = [row for row in rows if row["executed_family"] == "fund_award"]
    placement = _placement_report(target_placement)
    return {
        "schema_version": "tfm_rl_v4.checkpoint_diagnostic.v1",
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "dataset": str(dataset),
        "split": split,
        "samples": len(rows),
        "cohorts": {
            "target_family": _cohort_report(rows, "target_family"),
            "executed_family": _cohort_report(rows, "executed_family"),
        },
        "identity_confusion": {
            "target_family": _identity_confusion(rows, "target_family"),
            "executed_family": _identity_confusion(rows, "executed_family"),
        },
        "placement": {
            "target_family": placement,
            # In the legacy executed-family cohort, classify whether the
            # teacher target was actually a tile. This isolates the six rows
            # sampled as select_space whose supervised target was non-tile.
            "executed_family": _placement_report(executed_placement),
            "executed_action_type_counts": dict(Counter(
                _nested_action_type(row["chosen_descriptor"].get("decoded_action")) or "unknown"
                for row in executed_placement
            )),
        },
        "awards": {
            "target_family": _award_report(target_awards),
            "executed_family": _award_report(executed_awards),
        },
        "placement_gate": {
            "qualified": bool(placement["qualified_for_top3_gate"]),
            "mode": "top3" if placement["qualified_for_top3_gate"] else "top1",
            "top3_threshold": 0.97,
            "near_tie_threshold": TEACHER_TEMPERATURE,
            "near_tie_rate_threshold": 0.80,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V4 placement and award misses without changing artifacts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = diagnose(args.checkpoint, args.dataset, args.split, args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "placement_gate": report["placement_gate"],
        "placement": report["placement"]["target_family"]["tile"],
        "awards": report["awards"]["target_family"]["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
