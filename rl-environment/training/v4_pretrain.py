"""Card-aware behavior-cloning pretraining for TFM RL v4."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from training.v2_pretrain import pretrain

# token_from_features writes the type id at index 0 and the 63 action features
# at indices 1..63. The 14-value family tail therefore occupies columns 50:64.
ACTION_TAIL_COLUMN_START = 50
ACTION_TAIL_COLUMN_END = 64


def reinitialize_action_tail(network) -> None:
    """Replace only the projection columns that read the changed family tail."""
    weight = network.action_projection.weight
    if weight.shape[1] < ACTION_TAIL_COLUMN_END:
        raise RuntimeError(
            f"action projection has {weight.shape[1]} inputs; expected at least {ACTION_TAIL_COLUMN_END}"
        )
    bound = 1.0 / math.sqrt(weight.shape[1])
    with torch.no_grad():
        weight[:, ACTION_TAIL_COLUMN_START:ACTION_TAIL_COLUMN_END].uniform_(-bound, bound)


def load_v4_init_weights(network, checkpoint_path: str, *, reinit_action_tail: bool = False) -> str:
    """Copy warm-started V4 weights. The caller keeps a fresh AdamW optimizer."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V4 init checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from models.card_catalog import validate_checkpoint_catalog
    validate_checkpoint_catalog(checkpoint)
    state = checkpoint.get("network_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"V4 init checkpoint is missing network weights: {path}")
    network.load_state_dict(state)
    if reinit_action_tail:
        reinitialize_action_tail(network)
    return str(path)


def pretrain_v4(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    allow_small_dataset: bool = False,
    random_seed: int = 20260923,
    human_batch_fraction: float = 0.0,
    patience: int = 2,
    init_checkpoint: str | None = None,
    placement_gate_mode: str = "top1",
    placement_diagnostic_qualified: bool = False,
) -> dict:
    os.environ["TFM_RL_V4"] = "1"
    os.environ["TFM_RL_V3"] = "1"
    os.environ.setdefault("V3_FEATURE_SCALE", "1")
    output = Path(output_dir).expanduser().resolve()
    baseline = (Path(os.getenv("TFM_RL_V4_ROOT", "/app/v4")).expanduser().resolve() / "pretrain").resolve()
    if output == baseline:
        raise ValueError(f"refusing to overwrite immutable V4 baseline output: {baseline}")
    if not allow_small_dataset and not init_checkpoint:
        raise ValueError("V4 recovery pretraining requires /app/v4/bootstrap/warm_start.pth")
    return pretrain(
        dataset_dir,
        str(output),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        allow_small_dataset=allow_small_dataset,
        random_seed=random_seed,
        human_batch_fraction=human_batch_fraction,
        patience=patience,
        experiment="v4",
        track_families=True,
        init_checkpoint=init_checkpoint,
        reinit_action_tail=True,
        placement_gate_mode=placement_gate_mode,
        placement_diagnostic_qualified=placement_diagnostic_qualified,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the TFM RL v4 card-aware policy")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--human-batch-fraction", type=float, default=0.0)
    parser.add_argument(
        "--placement-diagnostic",
        help="JSON report from training.v4_diagnose_checkpoint; qualifying reports enable the top-3 placement gate",
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        default=os.getenv("V4_INIT_CHECKPOINT", "/app/v4/bootstrap/warm_start.pth"),
        help="Warm-started V4 weights. Optimizer state is not restored.",
    )
    args = parser.parse_args()
    placement_mode = "top1"
    placement_qualified = False
    if args.placement_diagnostic:
        diagnostic = json.loads(Path(args.placement_diagnostic).read_text(encoding="utf-8"))
        placement_qualified = bool((diagnostic.get("placement_gate") or {}).get("qualified", False))
        placement_mode = "top3" if placement_qualified else "top1"
    report = pretrain_v4(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        allow_small_dataset=args.allow_small_dataset,
        random_seed=args.seed,
        human_batch_fraction=args.human_batch_fraction,
        patience=args.patience,
        init_checkpoint=args.init_checkpoint,
        placement_gate_mode=placement_mode,
        placement_diagnostic_qualified=placement_qualified,
    )
    print(report["ppo_gate_passed"])


if __name__ == "__main__":
    main()
