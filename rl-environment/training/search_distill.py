"""Train a candidate from completed information-set MCTS replay shards.

This is intentionally a standalone candidate-training step. It never consumes
or mutates the strict on-policy PPO queue; callers evaluate and promote the
written checkpoint through the normal benchmark gates.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from models.agent import RLAgent
from models.planner_common import ensure_bundle, pad_bundle_batch
from search.replay_store import SEARCH_REPLAY_SCHEMA_VERSION, SearchReplayStore
from v2_runtime import initialize_v2_runtime


def load_search_records(
    replay_dir: str | Path,
    *,
    max_samples: int = 0,
    min_policy_version: int = 0,
) -> List[Dict[str, Any]]:
    root = Path(replay_dir).expanduser()
    paths = sorted(root.glob("search_*.pkl.gz"), reverse=True)
    records: List[Dict[str, Any]] = []
    for path in paths:
        payload = SearchReplayStore.read_shard(path)
        if int(payload.get("policy_version", 0) or 0) < int(min_policy_version):
            continue
        for raw in reversed(list(payload.get("records", []) or [])):
            record = validate_search_record(raw)
            records.append(record)
            if max_samples > 0 and len(records) >= int(max_samples):
                return records
    return records


def validate_search_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema_version") != SEARCH_REPLAY_SCHEMA_VERSION:
        raise ValueError("search record schema mismatch")
    if not bool(raw.get("value_target_valid", False)):
        raise ValueError("search record has no terminal value target")
    bundle = ensure_bundle(raw.get("planner_bundle"))
    target = [float(value) for value in list(raw.get("policy_target", []) or [])]
    action_count = int(bundle["action_tokens"].shape[0])
    if len(target) != action_count or action_count <= 0:
        raise ValueError("search policy target must align with planner action tokens")
    if any(not math.isfinite(value) or value < 0.0 for value in target):
        raise ValueError("search policy target contains an invalid probability")
    total = sum(target)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("search policy target is not normalized")
    value_target = float(raw.get("value_target", 0.0))
    if not math.isfinite(value_target):
        raise ValueError("search value target is not finite")
    return raw


def _batches(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def parse_family_weights(values: Sequence[str]) -> Dict[str, float]:
    """Parse repeatable FAMILY=WEIGHT CLI values used for training upsampling."""
    weights: Dict[str, float] = {}
    for raw in values:
        family, separator, raw_weight = str(raw).partition("=")
        family = family.strip()
        if not separator or not family or not raw_weight.strip():
            raise ValueError(f"family weight must use FAMILY=WEIGHT syntax: {raw!r}")
        if family in weights:
            raise ValueError(f"family weight was specified more than once: {family}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"family weight must be numeric: {raw!r}") from exc
        if not math.isfinite(weight) or weight < 1.0:
            raise ValueError(f"family weight must be finite and at least 1.0: {raw!r}")
        weights[family] = weight
    return weights


def upsample_records_by_family(
    records: Sequence[Dict[str, Any]],
    family_weights: Mapping[str, float],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Return a training-only sample with selected donated families repeated.

    Every source record remains present once. Integer weights are exact;
    fractional weights use deterministic stochastic rounding with ``rng``.
    Validation records must not be passed through this function.
    """
    sampled = list(records)
    for record in records:
        family = str(record.get("donation_family", "") or "")
        weight = float(family_weights.get(family, 1.0))
        extra_weight = max(0.0, weight - 1.0)
        whole_copies = int(math.floor(extra_weight))
        if whole_copies:
            sampled.extend([record] * whole_copies)
        if rng.random() < extra_weight - whole_copies:
            sampled.append(record)
    return sampled


def _donation_family_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        family = str(record.get("donation_family", "") or "")
        if family:
            counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _target_batch(records: Sequence[Dict[str, Any]], action_dim: int, device: torch.device) -> torch.Tensor:
    target = torch.zeros((len(records), action_dim), dtype=torch.float32, device=device)
    for row, record in enumerate(records):
        values = torch.tensor(record["policy_target"], dtype=torch.float32, device=device)
        target[row, : min(action_dim, int(values.numel()))] = values[:action_dim]
    return target


def _recurrent_batch(records: Sequence[Dict[str, Any]], device: torch.device) -> torch.Tensor | None:
    rows = [list(record.get("recurrent_state", []) or []) for record in records]
    widths = {len(row) for row in rows}
    if widths == {0}:
        return None
    if len(widths) != 1 or 0 in widths:
        raise ValueError("search replay recurrent states must have one consistent width")
    return torch.tensor(rows, dtype=torch.float32, device=device)


def run_epoch(
    agent: RLAgent,
    records: Sequence[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch_size: int,
    value_weight: float,
    train: bool,
) -> Dict[str, float]:
    network = agent.network
    network.train(mode=train)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "target_top1": 0.0}
    seen = 0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in _batches(records, batch_size):
            bundles = pad_bundle_batch(
                [record["planner_bundle"] for record in batch],
                device=device,
                planner_config=network.planner_config,
            )
            phases = torch.tensor(
                [int(record.get("phase_index", 0) or 0) for record in batch],
                dtype=torch.long,
                device=device,
            )
            recurrent = _recurrent_batch(batch, device)
            output = network(bundles, phase_indices=phases, recurrent_state=recurrent)
            logits = output["policy_logits"].float()
            targets = _target_batch(batch, int(logits.shape[1]), device)
            policy_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            expected_values = torch.tensor(
                [float(record["value_target"]) for record in batch],
                dtype=torch.float32,
                device=device,
            )
            predicted_values = output["value"].float().reshape(-1)
            value_loss = F.smooth_l1_loss(predicted_values, expected_values)
            loss = policy_loss + float(value_weight) * value_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
            count = len(batch)
            seen += count
            totals["loss"] += float(loss.detach().item()) * count
            totals["policy_loss"] += float(policy_loss.detach().item()) * count
            totals["value_loss"] += float(value_loss.detach().item()) * count
            totals["target_top1"] += float(
                (logits.argmax(dim=-1) == targets.argmax(dim=-1)).float().sum().item()
            )
    if seen <= 0:
        return {key: 0.0 for key in totals}
    return {key: value / seen for key, value in totals.items()}


def distill(
    checkpoint: str,
    replay_dir: str,
    output: str,
    *,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    value_weight: float = 0.25,
    max_samples: int = 0,
    min_policy_version: int = 0,
    validation_fraction: float = 0.1,
    family_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    initialize_v2_runtime()
    records = load_search_records(
        replay_dir,
        max_samples=max_samples,
        min_policy_version=min_policy_version,
    )
    if not records:
        raise RuntimeError("no compatible completed search replay records were found")
    rng = random.Random(20260928)
    rng.shuffle(records)
    validation_count = min(
        max(0, len(records) - 1),
        max(1, round(len(records) * min(0.5, max(0.0, float(validation_fraction))))),
    ) if len(records) > 1 and validation_fraction > 0.0 else 0
    validation = records[:validation_count]
    raw_training = records[validation_count:]
    normalized_family_weights = {
        str(family): float(weight) for family, weight in (family_weights or {}).items()
    }
    for family, weight in normalized_family_weights.items():
        if not family or not math.isfinite(weight) or weight < 1.0:
            raise ValueError(
                f"family weight for {family!r} must be finite and at least 1.0"
            )
    training = upsample_records_by_family(raw_training, normalized_family_weights, rng)

    agent = RLAgent(agent_id="search-distill")
    agent.load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.network.to(device)
    optimizer = torch.optim.AdamW(agent.network.parameters(), lr=float(learning_rate))
    started = time.monotonic()
    history: List[Dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        rng.shuffle(training)
        train_metrics = run_epoch(
            agent,
            training,
            optimizer,
            device,
            batch_size=batch_size,
            value_weight=value_weight,
            train=True,
        )
        validation_metrics = (
            run_epoch(
                agent,
                validation,
                optimizer,
                device,
                batch_size=batch_size,
                value_weight=value_weight,
                train=False,
            )
            if validation
            else {}
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})

    agent.policy_version = int(agent.policy_version) + 1
    # Persist the optimizer that actually produced this candidate so a later
    # resume does not silently restore the source checkpoint's stale moments.
    agent.optimizer = optimizer
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(output_path))
    # Donation coverage is the point of the teacher replay stream, so report it
    # explicitly. A run with zero fund_award records has not addressed the
    # award exploit no matter how good the loss curve looks.
    family_counts: Dict[str, int] = {}
    chosen_family_counts: Dict[str, int] = {}
    for record in records:
        for row in list(record.get("action_descriptors") or []):
            if not isinstance(row, dict):
                continue
            family = str(row.get("family", "") or "")
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
        chosen = str(record.get("donation_family", "") or "")
        if chosen:
            chosen_family_counts[chosen] = chosen_family_counts.get(chosen, 0) + 1
    report = {
        "schema_version": "tfm.search_distill_report.v1",
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output_path),
        "replay_dir": str(Path(replay_dir).expanduser()),
        "records": len(records),
        "training_records": len(raw_training),
        "effective_training_records": len(training),
        "validation_records": len(validation),
        "policy_version": int(agent.policy_version),
        "device": str(device),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "donation_sources": sorted({
            str(record.get("donation_source", "") or "")
            for record in records
            if record.get("donation_source")
        }),
        "legal_family_counts": dict(sorted(family_counts.items())),
        "chosen_family_counts": dict(sorted(chosen_family_counts.items())),
        "family_weights": dict(sorted(normalized_family_weights.items())),
        "raw_training_family_counts": _donation_family_counts(raw_training),
        "effective_training_family_counts": _donation_family_counts(training),
        "history": history,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".search-distill.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill completed MCTS replay shards into a candidate checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-policy-version", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--family-weight",
        action="append",
        default=[],
        metavar="FAMILY=WEIGHT",
        help=(
            "repeat donated records from FAMILY by WEIGHT in the training split only; "
            "may be supplied more than once"
        ),
    )
    args = parser.parse_args()
    try:
        family_weights = parse_family_weights(args.family_weight)
    except ValueError as exc:
        parser.error(str(exc))
    report = distill(
        args.checkpoint,
        args.replay_dir,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        max_samples=args.max_samples,
        min_policy_version=args.min_policy_version,
        validation_fraction=args.validation_fraction,
        family_weights=family_weights,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
