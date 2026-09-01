"""Behavior-cloning pretraining for TFM RL v2 teacher datasets."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from models.agent import AgentConfig, TerraformingMarsNetwork
from models.planner_common import pad_bundle_batch
from training.teacher_dataset import TeacherDatasetStore, validate_sample


def _iter_shard_batches(root: Path, split: str, batch_size: int, shuffle: bool) -> Iterable[List[Dict[str, Any]]]:
    paths = sorted((root / split).glob("episode_*.pkl.gz"))
    rng = random.Random(20260901)
    if shuffle:
        rng.shuffle(paths)
    pending: List[Dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rb") as handle:
            items = list(pickle.load(handle) or [])
        if shuffle:
            rng.shuffle(items)
        for item in items:
            validate_sample(item)
            pending.append(item)
            if len(pending) >= batch_size:
                yield pending
                pending = []
    if pending:
        yield pending


def _targets(samples: Sequence[Dict[str, Any]], action_dim: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((len(samples), action_dim), dtype=torch.float32, device=device)
    for row, sample in enumerate(samples):
        probs = torch.tensor(sample["teacher_probabilities"], dtype=torch.float32, device=device)
        out[row, : min(action_dim, int(probs.numel()))] = probs[:action_dim]
    return out


def _run_epoch(
    network: TerraformingMarsNetwork,
    optimizer: torch.optim.Optimizer,
    dataset_root: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    network.train(mode=train)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "human_top3": 0.0,
        "teacher_top1": 0.0,
        "teacher_top3": 0.0,
    }
    count = 0
    human_count = 0
    teacher_count = 0
    planner_config = network.planner_config
    for samples in _iter_shard_batches(dataset_root, split, batch_size, shuffle=train):
        bundles = pad_bundle_batch([item["planner_bundle"] for item in samples], device=device, planner_config=planner_config)
        phase_indices = torch.tensor([int(item.get("phase_index", 0)) for item in samples], dtype=torch.long, device=device)
        output = network(bundles, phase_indices=phase_indices)
        logits = output["policy_logits"]
        targets = _targets(samples, int(logits.shape[1]), device)
        weights = torch.tensor([float(item.get("sample_weight", 1.0)) for item in samples], dtype=torch.float32, device=device)
        target_values = torch.tensor([float(item.get("value_target", 0.0)) for item in samples], dtype=torch.float32, device=device)
        per_row_policy = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        policy_loss = (per_row_policy * weights).sum() / torch.clamp(weights.sum(), min=1.0)
        predicted_values = output["value"].reshape(-1)
        value_loss = F.mse_loss(predicted_values, target_values)
        loss = policy_loss + (0.25 * value_loss)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

        expected = targets.argmax(dim=-1)
        predicted = logits.argmax(dim=-1)
        top1 = (predicted == expected).float()
        topk = logits.topk(k=min(3, int(logits.shape[1])), dim=-1).indices
        top3 = (topk == expected.unsqueeze(1)).any(dim=1).float()
        human_rows = [str(sample.get("source", "")).startswith("human") for sample in samples]
        for idx, is_human in enumerate(human_rows):
            if is_human:
                top1[idx] = float(targets[idx, predicted[idx]].item() > 0.0)
                top3[idx] = float((targets[idx, topk[idx]] > 0.0).any().item())
        batch_count = len(samples)
        totals["loss"] += float(loss.detach().item()) * batch_count
        totals["policy_loss"] += float(policy_loss.detach().item()) * batch_count
        totals["value_loss"] += float(value_loss.detach().item()) * batch_count
        totals["top1"] += float(top1.sum().item())
        totals["top3"] += float(top3.sum().item())
        for idx, is_human in enumerate(human_rows):
            if is_human:
                totals["human_top3"] += float(top3[idx].item())
                human_count += 1
            else:
                totals["teacher_top1"] += float(top1[idx].item())
                totals["teacher_top3"] += float(top3[idx].item())
                teacher_count += 1
        count += batch_count
    if count == 0:
        raise RuntimeError(f"empty teacher dataset split: {split}")
    return {
        "loss": totals["loss"] / count,
        "policy_loss": totals["policy_loss"] / count,
        "value_loss": totals["value_loss"] / count,
        "top1": totals["top1"] / count,
        "top3": totals["top3"] / count,
        "human_top3": (totals["human_top3"] / human_count) if human_count else 0.0,
        "teacher_top1": (totals["teacher_top1"] / teacher_count) if teacher_count else 0.0,
        "teacher_top3": (totals["teacher_top3"] / teacher_count) if teacher_count else 0.0,
        "samples": float(count),
        "human_samples": float(human_count),
        "teacher_samples": float(teacher_count),
    }


def pretrain(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    allow_small_dataset: bool = False,
) -> Dict[str, Any]:
    store = TeacherDatasetStore(dataset_dir)
    counts = store.counts()
    human_total = 0
    teacher_total = 0
    for split in ("train", "validation", "test"):
        for item in store.iter_samples(split):
            if str(item.get("source", "")).startswith("human"):
                human_total += 1
            else:
                teacher_total += 1
    if not allow_small_dataset and teacher_total < 100_000:
        raise RuntimeError(f"v2 pretraining requires at least 100000 teacher samples; found {teacher_total}")
    if not allow_small_dataset and human_total < 100:
        raise RuntimeError(f"v2 pretraining requires at least 100 human labels; found {human_total}")

    config = AgentConfig.from_env()
    network = TerraformingMarsNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(learning_rate))
    history: List[Dict[str, Any]] = []
    best_score = -1.0
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, max(1, int(epochs)) + 1):
        train_metrics = _run_epoch(network, optimizer, store.root, "train", batch_size, device, train=True)
        with torch.no_grad():
            validation_metrics = _run_epoch(network, optimizer, store.root, "validation", batch_size, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        score = float(validation_metrics["top1"] + validation_metrics["top3"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "schema_version": "tfm_rl_v2.bc_checkpoint.v1",
                    "experiment_version": "tfm-rl-v2",
                    "network_state_dict": network.state_dict(),
                    "config": asdict(config),
                    "fresh_weights": True,
                    "validation": validation_metrics,
                    "dataset_counts": counts,
                },
                output / "bc_best.pth",
            )
    best_checkpoint = torch.load(output / "bc_best.pth", map_location=device, weights_only=False)
    network.load_state_dict(best_checkpoint["network_state_dict"])
    with torch.no_grad():
        test_metrics = _run_epoch(network, optimizer, store.root, "test", batch_size, device, train=False)
        human_eval_parts = [
            _run_epoch(network, optimizer, store.root, split, batch_size, device, train=False)
            for split in ("train", "validation", "test")
        ]
    human_eval_count = sum(int(part["human_samples"]) for part in human_eval_parts)
    human_top3_all = (
        sum(float(part["human_top3"]) * int(part["human_samples"]) for part in human_eval_parts) / human_eval_count
        if human_eval_count else 0.0
    )
    passed = bool(
        test_metrics["teacher_top1"] >= 0.85
        and test_metrics["teacher_top3"] >= 0.97
        and human_eval_count >= 100
        and human_top3_all >= 0.80
    )
    report = {
        "schema_version": "tfm_rl_v2.pretrain_report.v1",
        "counts": counts,
        "teacher_samples": teacher_total,
        "human_samples": human_total,
        "history": history,
        "test": test_metrics,
        "human_evaluation": {"samples": human_eval_count, "top3": human_top3_all},
        "ppo_gate_passed": passed,
    }
    (output / "pretrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a fresh TFM RL v2 policy from teacher data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--allow-small-dataset", action="store_true", help="Testing only; bypass 100k/100-label gates")
    args = parser.parse_args()
    report = pretrain(args.dataset, args.output, args.epochs, args.batch_size, args.learning_rate, args.allow_small_dataset)
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
