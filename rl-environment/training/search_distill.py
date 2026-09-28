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
from typing import Any, Dict, Iterable, List, Sequence

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
    training = records[validation_count:]

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
    report = {
        "schema_version": "tfm.search_distill_report.v1",
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output_path),
        "replay_dir": str(Path(replay_dir).expanduser()),
        "records": len(records),
        "training_records": len(training),
        "validation_records": len(validation),
        "policy_version": int(agent.policy_version),
        "device": str(device),
        "elapsed_sec": round(time.monotonic() - started, 3),
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
    args = parser.parse_args()
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
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
