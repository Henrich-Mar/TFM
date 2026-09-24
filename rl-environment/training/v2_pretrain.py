"""Behavior-cloning pretraining for TFM RL v2 teacher datasets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from models.agent import AgentConfig, TerraformingMarsNetwork
from models.planner_common import pad_bundle_batch
from training.teacher_dataset import TeacherDatasetStore, validate_sample
from v2_runtime import initialize_v2_runtime


def _iter_shard_samples(
    root: Path,
    split: str,
    shuffle: bool,
    human: bool | None = None,
) -> Iterable[Dict[str, Any]]:
    """Yield samples from one split, optionally restricted by their source."""
    paths = sorted((root / split).glob("episode_*.pkl.gz"))
    rng = random.Random(20260901)
    if shuffle:
        rng.shuffle(paths)
    for path in paths:
        with gzip.open(path, "rb") as handle:
            items = list(pickle.load(handle) or [])
        if shuffle:
            rng.shuffle(items)
        for item in items:
            validate_sample(item)
            is_human = str(item.get("source", "")).startswith("human")
            if human is None or is_human == human:
                yield item


def _iter_shard_batches(
    root: Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    human_batch_fraction: float = 0.0,
    human_only: bool | None = None,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield ordinary batches, or source-balanced training batches.

    Human decisions are scarce by design.  A weight alone cannot compensate
    when fewer than one human row appears in most batches, so training mixes a
    small, deterministic stream of repeated human rows into every batch.
    Evaluation always uses the natural dataset distribution.
    """
    if not shuffle or human_batch_fraction <= 0.0:
        pending: List[Dict[str, Any]] = []
        for item in _iter_shard_samples(root, split, shuffle, human=human_only):
            pending.append(item)
            if len(pending) >= batch_size:
                yield pending
                pending = []
        if pending:
            yield pending
        return

    fraction = float(human_batch_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("human_batch_fraction must be between 0 and 1")
    human_samples = list(_iter_shard_samples(root, split, shuffle=True, human=True))
    if not human_samples:
        # Keep the standard path for teacher-only datasets (including small
        # unit-test fixtures).
        yield from _iter_shard_batches(root, split, batch_size, shuffle, 0.0)
        return
    human_per_batch = min(batch_size - 1, max(1, round(batch_size * fraction)))
    teacher_per_batch = batch_size - human_per_batch
    if teacher_per_batch <= 0:
        raise ValueError("batch_size must leave room for teacher samples")
    rng = random.Random(20260901)
    human_offset = 0
    pending = []
    for item in _iter_shard_samples(root, split, shuffle=True, human=False):
        pending.append(item)
        if len(pending) < teacher_per_batch:
            continue
        batch = pending
        pending = []
        batch.extend(human_samples[(human_offset + index) % len(human_samples)] for index in range(human_per_batch))
        human_offset += human_per_batch
        rng.shuffle(batch)
        yield batch
    if pending:
        batch = pending
        batch.extend(human_samples[(human_offset + index) % len(human_samples)] for index in range(human_per_batch))
        rng.shuffle(batch)
        yield batch


def _targets(samples: Sequence[Dict[str, Any]], action_dim: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((len(samples), action_dim), dtype=torch.float32, device=device)
    for row, sample in enumerate(samples):
        probs = torch.tensor(sample["teacher_probabilities"], dtype=torch.float32, device=device)
        out[row, : min(action_dim, int(probs.numel()))] = probs[:action_dim]
    return out


def _target_family(sample: Mapping[str, Any], expected_position: int) -> str:
    descriptors = list(sample.get("action_descriptors", []) or [])
    target = int(sample.get("target_action_position", expected_position))
    if 0 <= target < len(descriptors):
        return str(descriptors[target].get("family", "other") or "other")
    return "other"


def _run_epoch(
    network: TerraformingMarsNetwork,
    optimizer: torch.optim.Optimizer,
    dataset_root: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    train: bool,
    human_batch_fraction: float = 0.0,
    human_only: bool | None = None,
    track_families: bool = False,
) -> Dict[str, Any]:
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
    policy_count = 0
    human_count = 0
    teacher_count = 0
    family_hits: Dict[str, float] = {}
    family_top3_hits: Dict[str, float] = {}
    family_counts: Dict[str, int] = {}
    batch_index = 0
    started_at = time.monotonic()
    planner_config = network.planner_config
    for samples in _iter_shard_batches(
        dataset_root,
        split,
        batch_size,
        shuffle=train,
        human_batch_fraction=human_batch_fraction if train else 0.0,
        human_only=human_only,
    ):
        batch_index += 1
        bundles = pad_bundle_batch([item["planner_bundle"] for item in samples], device=device, planner_config=planner_config)
        phase_indices = torch.tensor([int(item.get("phase_index", 0)) for item in samples], dtype=torch.long, device=device)
        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            output = network(bundles, phase_indices=phase_indices)
        logits = output["policy_logits"].float()
        targets = _targets(samples, int(logits.shape[1]), device)
        weights = torch.tensor([float(item.get("sample_weight", 1.0)) for item in samples], dtype=torch.float32, device=device)
        policy_valid = torch.tensor(
            [bool(item.get("policy_target_valid", not item.get("is_forced", False))) for item in samples],
            dtype=torch.bool,
            device=device,
        )
        target_values = torch.tensor([float(item.get("value_target", 0.0)) for item in samples], dtype=torch.float32, device=device)
        value_valid = torch.tensor(
            [bool(item.get("value_target_valid", True)) for item in samples],
            dtype=torch.bool,
            device=device,
        )
        per_row_policy = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        effective_weights = weights * policy_valid.float()
        policy_loss = (per_row_policy * effective_weights).sum() / torch.clamp(effective_weights.sum(), min=1.0)
        predicted_values = output["value"].float().reshape(-1)
        value_loss = (
            F.mse_loss(predicted_values[value_valid], target_values[value_valid])
            if bool(value_valid.any())
            else predicted_values.sum() * 0.0
        )
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
            if not bool(policy_valid[idx].item()):
                top1[idx] = 0.0
                top3[idx] = 0.0
                continue
            if is_human:
                top1[idx] = float(targets[idx, predicted[idx]].item() > 0.0)
                top3[idx] = float((targets[idx, topk[idx]] > 0.0).any().item())
        batch_count = len(samples)
        totals["loss"] += float(loss.detach().item()) * batch_count
        totals["policy_loss"] += float(policy_loss.detach().item()) * batch_count
        totals["value_loss"] += float(value_loss.detach().item()) * batch_count
        totals["top1"] += float(top1.sum().item())
        totals["top3"] += float(top3.sum().item())
        policy_count += int(policy_valid.sum().item())
        for idx, is_human in enumerate(human_rows):
            if not bool(policy_valid[idx].item()):
                continue
            if is_human:
                totals["human_top3"] += float(top3[idx].item())
                human_count += 1
            else:
                totals["teacher_top1"] += float(top1[idx].item())
                totals["teacher_top3"] += float(top3[idx].item())
                teacher_count += 1
                if track_families:
                    family = _target_family(samples[idx], int(expected[idx].item()))
                    family_hits[family] = family_hits.get(family, 0.0) + float(top1[idx].item())
                    family_top3_hits[family] = family_top3_hits.get(family, 0.0) + float(top3[idx].item())
                    family_counts[family] = family_counts.get(family, 0) + 1
        count += batch_count
        if batch_index == 1 or batch_index % 250 == 0:
            print(
                f"[pretrain] {split} batch={batch_index} samples={count} "
                f"elapsed={time.monotonic() - started_at:.1f}s",
                flush=True,
            )
    if count == 0:
        if human_only:
            # A split can contain teacher rows and no human games. The final
            # human evaluation still visits every split.
            empty = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "top1": 0.0,
                "top3": 0.0,
                "human_top3": 0.0,
                "teacher_top1": 0.0,
                "teacher_top3": 0.0,
                "samples": 0.0,
                "human_samples": 0.0,
                "teacher_samples": 0.0,
            }
            if track_families:
                empty["family_top1"] = {}
                empty["family_top3"] = {}
                empty["family_counts"] = {}
            return empty
        raise RuntimeError(f"empty teacher dataset split: {split}")
    metrics: Dict[str, Any] = {
        "loss": totals["loss"] / count,
        "policy_loss": totals["policy_loss"] / count,
        "value_loss": totals["value_loss"] / count,
        "top1": (totals["top1"] / policy_count) if policy_count else 0.0,
        "top3": (totals["top3"] / policy_count) if policy_count else 0.0,
        "human_top3": (totals["human_top3"] / human_count) if human_count else 0.0,
        "teacher_top1": (totals["teacher_top1"] / teacher_count) if teacher_count else 0.0,
        "teacher_top3": (totals["teacher_top3"] / teacher_count) if teacher_count else 0.0,
        "samples": float(count),
        "human_samples": float(human_count),
        "teacher_samples": float(teacher_count),
    }
    if track_families:
        metrics["family_top1"] = {
            family: (family_hits.get(family, 0.0) / count_value if count_value else 0.0)
            for family, count_value in family_counts.items()
        }
        metrics["family_top3"] = {
            family: (family_top3_hits.get(family, 0.0) / count_value if count_value else 0.0)
            for family, count_value in family_counts.items()
        }
        metrics["family_counts"] = dict(sorted(family_counts.items()))
    return metrics


def pretrain(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    allow_small_dataset: bool = False,
    random_seed: int = 20260901,
    human_batch_fraction: float = 0.125,
    patience: int | None = None,
    experiment: str = "v2",
    track_families: bool = False,
    init_checkpoint: str | None = None,
    placement_gate_mode: str = "top1",
    placement_diagnostic_qualified: bool = False,
) -> Dict[str, Any]:
    runtime_paths = initialize_v2_runtime()
    output = Path(output_dir).expanduser().resolve()
    if runtime_paths:
        runtime_root = Path(runtime_paths["root"]).resolve()
        try:
            output.relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeError(f"v2 pretraining output must stay inside {runtime_root}: {output}") from exc
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"v2 pretraining output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and str(os.getenv("V2_ALLOW_PRETRAIN_OVERWRITE", "0")).lower() not in {
        "1", "true", "yes", "on",
    }:
        raise RuntimeError(
            f"refusing to overwrite non-empty v2 pretraining output: {output}; "
            "set V2_ALLOW_PRETRAIN_OVERWRITE=1 for an intentional restart"
        )
    random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))
    store = TeacherDatasetStore(dataset_dir)
    dataset_audit = store.audit()
    if not bool(dataset_audit.get("valid", False)):
        raise RuntimeError("invalid v2 teacher dataset: " + "; ".join(dataset_audit.get("errors", [])))
    # audit() already traverses every shard and now returns both split and
    # source counts. Reusing them avoids three extra full gzip/pickle passes
    # before the first training epoch.
    counts = dict(dataset_audit["split_counts"])
    source_counts = dict(dataset_audit.get("source_counts", {}))
    human_total = int(source_counts.get("human_preference", source_counts.get("human", 0)))
    teacher_total = int(source_counts.get("teacher", 0))
    if not allow_small_dataset and teacher_total < 100_000:
        raise RuntimeError(f"v2 pretraining requires at least 100000 teacher samples; found {teacher_total}")
    if not allow_small_dataset and human_total < 100:
        raise RuntimeError(f"v2 pretraining requires at least 100 human labels; found {human_total}")
    if human_total and experiment != "v4" and not 0.0 < float(human_batch_fraction) < 1.0:
        raise ValueError("human_batch_fraction must be between 0 and 1 when human labels are present")
    if experiment == "v4" and float(human_batch_fraction) != 0.0:
        raise ValueError("V4 repaired pretraining requires natural-frequency human rows (human_batch_fraction=0)")
    if human_total and int(batch_size) < 2:
        raise ValueError("batch_size must be at least 2 when human labels are present")
    if experiment == "v4" and placement_gate_mode == "top3" and not placement_diagnostic_qualified:
        raise ValueError("V4 select_space top-3 mode requires a qualifying checkpoint diagnostic")
    if experiment == "v4" and not allow_small_dataset:
        target_counts = dataset_audit.get("target_family_counts") or {}
        for split in ("validation", "test"):
            for family in ("select_space", "fund_award", "claim_milestone"):
                observed = int((target_counts.get(split) or {}).get(family, 0) or 0)
                if observed < 100:
                    raise RuntimeError(
                        f"V4 requires at least 100 {family} target-family rows in {split}; found {observed}"
                    )
        human_games = dataset_audit.get("human_game_ids") or {}
        training_human_games = list(human_games.get("train") or [])
        held_out_human_games = list(human_games.get("test") or [])
        validation_human_games = list(human_games.get("validation") or [])
        if (
            len(training_human_games) != 8
            or len(held_out_human_games) != 2
            or validation_human_games
        ):
            raise RuntimeError(
                "V4 requires eight human training games, exactly two held-out human test games, "
                "and none in validation"
            )

    config = AgentConfig.from_env()
    network = TerraformingMarsNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    loaded_init = None
    if init_checkpoint:
        if experiment != "v4":
            raise RuntimeError("init checkpoints are only supported for v4 pretraining")
        from training.v4_pretrain import load_v4_init_weights
        loaded_init = load_v4_init_weights(network, init_checkpoint)
    print(
        f"[pretrain] starting: teacher_samples={teacher_total} human_samples={human_total} device={device} "
        f"hidden_size={config.hidden_size} transformer_layers={config.transformer_layers} "
        f"transformer_heads={config.transformer_heads} recurrent_size={config.recurrent_size} "
        f"planner_token_dim={config.planner_token_dim} "
        f"batch_size={batch_size} amp={'bf16' if device.type == 'cuda' else 'off'} "
        f"init_checkpoint={loaded_init or 'none'}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(learning_rate))
    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_key: tuple[Any, ...] | None = None
    best_epoch = 0
    best_validation_gate_passed = False
    first_qualifying_epoch: int | None = None
    stale_epochs = 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, max(1, int(epochs)) + 1):
        epoch_started_at = time.monotonic()
        print(f"[pretrain] starting epoch {epoch}/{max(1, int(epochs))}", flush=True)
        train_metrics = _run_epoch(
            network,
            optimizer,
            store.root,
            "train",
            batch_size,
            device,
            train=True,
            human_batch_fraction=human_batch_fraction,
            track_families=track_families,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                network, optimizer, store.root, "validation", batch_size, device, train=False, track_families=track_families,
            )
        gate_status: Dict[str, Any] | None = None
        if experiment == "v4":
            from training.v4_gates import validation_candidate_status
            gate_status = validation_candidate_status(
                validation_metrics,
                placement_gate_mode,
                allow_small_dataset=allow_small_dataset,
            )
            selection_key = tuple(gate_status["selection_key"])
            score = float(validation_metrics["teacher_top1"] + validation_metrics["teacher_top3"])
        else:
            score = float(
                validation_metrics["teacher_top1"]
                + validation_metrics["teacher_top3"]
                + validation_metrics["human_top3"]
            )
            selection_key = (score,)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        if gate_status is not None:
            row["validation_gate"] = {
                **gate_status,
                "selection_key": list(gate_status["selection_key"]),
            }
            if bool(gate_status["passed"]) and first_qualifying_epoch is None:
                first_qualifying_epoch = epoch
        history.append(row)

        checkpoint = {
            "schema_version": "tfm_rl_v4.bc_checkpoint.v2" if experiment == "v4" else "tfm_rl_v2.bc_checkpoint.v1",
            "experiment_version": "tfm-rl-v4" if experiment == "v4" else "tfm-rl-v2",
            "network_state_dict": network.state_dict(),
            "config": asdict(config),
            "fresh_weights": experiment != "v4" and loaded_init is None,
            "initialized_from": loaded_init,
            "validation": validation_metrics,
            "selection_score": score,
            "selection_key": list(selection_key),
            "epoch": epoch,
            "validation_gate_passed": bool(gate_status and gate_status["passed"]),
            "dataset_counts": counts,
        }
        if experiment == "v4":
            from models.card_catalog import get_catalog
            checkpoint["card_catalog_sha256"] = get_catalog().sha256
            checkpoint["state_schema_version"] = "v4-card-aware.v1"
            checkpoint["teacher_sample_schema_version"] = "teacher_sample.v5"
            checkpoint["placement_gate_mode"] = placement_gate_mode
            torch.save(checkpoint, output / f"bc_epoch_{epoch:02d}.pth")

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_score = score
            best_epoch = epoch
            best_validation_gate_passed = bool(gate_status and gate_status["passed"])
            stale_epochs = 0
            torch.save(checkpoint, output / "bc_best.pth")
        else:
            stale_epochs += 1
        print(
            f"[pretrain] epoch {epoch}/{max(1, int(epochs))} "
            f"loss={train_metrics['loss']:.4f} "
            f"validation_top1={validation_metrics['top1']:.4f} "
            f"validation_top3={validation_metrics['top3']:.4f} "
            f"elapsed={time.monotonic() - epoch_started_at:.1f}s",
            flush=True,
        )
        if experiment == "v4" and first_qualifying_epoch is not None and epoch > first_qualifying_epoch:
            print("[pretrain] stopped after one confirmation epoch following a qualifying checkpoint", flush=True)
            break
        if patience is not None and stale_epochs >= int(patience):
            print(f"[pretrain] early stop after {stale_epochs} validation epochs without improvement", flush=True)
            break
    best_checkpoint = torch.load(output / "bc_best.pth", map_location=device, weights_only=False)
    network.load_state_dict(best_checkpoint["network_state_dict"])
    with torch.no_grad():
        test_metrics = _run_epoch(
            network,
            optimizer,
            store.root,
            "test",
            batch_size,
            device,
            train=False,
            human_only=False if experiment == "v4" else None,
            track_families=track_families,
        )
        human_eval_parts = (
            [
                _run_epoch(
                    network, optimizer, store.root, split, batch_size, device, train=False,
                    human_only=True, track_families=track_families,
                )
                for split in (("test",) if experiment == "v4" else ("train", "validation", "test"))
            ]
            if human_total
            else []
        )
    human_eval_count = sum(int(part["human_samples"]) for part in human_eval_parts)
    human_top3_all = (
        sum(float(part["human_top3"]) * int(part["human_samples"]) for part in human_eval_parts) / human_eval_count
        if human_eval_count else 0.0
    )
    if experiment == "v4":
        selected_path = output / "bc_best.pth"
        digest = hashlib.sha256()
        with selected_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        selected_sha256 = digest.hexdigest()
        held_out_games = list((dataset_audit.get("human_game_ids") or {}).get("test") or [])
        report = {
            "schema_version": "tfm_rl_v4.pretrain_report.v2",
            "counts": counts,
            "dataset_audit": dataset_audit,
            "teacher_samples": teacher_total,
            "human_samples": human_total,
            "random_seed": int(random_seed),
            "epochs": len(history),
            "epochs_requested": max(1, int(epochs)),
            "human_batch_fraction": float(human_batch_fraction),
            "history": history,
            "test": test_metrics,
            "human_evaluation": {
                "samples": human_eval_count,
                "top3": human_top3_all,
                "held_out_games": held_out_games,
                "split": "test",
            },
            "human_source_game_provenance": {
                "strategy": "whole-game-deterministic-v1",
                "training_games": list((dataset_audit.get("human_game_ids") or {}).get("train") or []),
                "held_out_games": held_out_games,
                "validation_games": list((dataset_audit.get("human_game_ids") or {}).get("validation") or []),
                "sample_weight": 1.0,
                "repeated_within_epoch": False,
            },
            "init_checkpoint": loaded_init,
            "selected_epoch": int(best_epoch),
            "selected_checkpoint": str(selected_path),
            "selected_checkpoint_sha256": selected_sha256,
            "selected_validation_gate_passed": bool(best_validation_gate_passed),
            "placement_gate": {
                "mode": placement_gate_mode,
                "diagnostic_qualified": bool(placement_diagnostic_qualified),
            },
            "duplicate_executable_actions": 0,
            "unresolved_known_card_references": 0,
            "smoke": {
                "completed": None,
                "server_rejected_actions": None,
                "checkpoint_sha256": None,
                "seed": None,
                "configuration": None,
            },
        }
        from training.v4_gates import evaluate_ppo_gate
        passed, reasons = evaluate_ppo_gate(report)
        report["ppo_gate_passed"] = passed
        report["ppo_gate_reasons"] = reasons
    else:
        passed = bool(
            test_metrics["teacher_top1"] >= 0.85
            and test_metrics["teacher_top3"] >= 0.97
            and human_eval_count >= 100
            and human_top3_all >= 0.80
        )
        report = {
            "schema_version": "tfm_rl_v2.pretrain_report.v1",
            "counts": counts,
            "dataset_audit": dataset_audit,
            "teacher_samples": teacher_total,
            "human_samples": human_total,
            "random_seed": int(random_seed),
            "epochs": max(1, int(epochs)),
            "human_batch_fraction": float(human_batch_fraction),
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--human-batch-fraction",
        type=float,
        default=0.125,
        help="Training fraction reserved for repeated human examples (default: 0.125)",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--allow-small-dataset", action="store_true", help="Testing only; bypass 100k/100-label gates")
    args = parser.parse_args()
    report = pretrain(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        allow_small_dataset=args.allow_small_dataset,
        random_seed=args.seed,
        human_batch_fraction=args.human_batch_fraction,
    )
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
