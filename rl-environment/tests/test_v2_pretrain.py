from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.teacher_dataset import SCHEMA_VERSION, TeacherDatasetStore, split_for_episode
from training.v2_pretrain import pretrain


def _bundle() -> dict:
    return {
        "world_tokens": np.zeros((1, 64), dtype=np.float32),
        "world_token_types": np.asarray([1], dtype=np.int64),
        "world_mask": np.asarray([True]),
        "hand_tokens": np.zeros((0, 64), dtype=np.float32),
        "hand_mask": np.zeros((0,), dtype=np.bool_),
        "action_tokens": np.zeros((2, 64), dtype=np.float32),
        "action_mask": np.asarray([True, True]),
        "action_indices": np.asarray([10, 20], dtype=np.int64),
        "action_positions": np.asarray([0, 1], dtype=np.int64),
        "global_scalars": np.zeros((16,), dtype=np.float32),
    }


def _sample(seed: int, game_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "planner_bundle": _bundle(),
        "action_descriptors": [
            {"action_index": 10, "action_position": 0},
            {"action_index": 20, "action_position": 1},
        ],
        "action_indices": [10, 20],
        "teacher_probabilities": [1.0, 0.0],
        "chosen_action_position": 0,
        "phase_index": 0,
        "confidence": 1.0,
        "source": "heuristic-teacher.v1",
        "sample_weight": 1.0,
        "seed": seed,
        "game_id": game_id,
        "value_target": 1.0,
        "rank": 1,
        "vp": 100.0,
        "vp_mean": 80.0,
    }


def test_small_pretrain_writes_v2_checkpoint_and_refuses_implicit_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TFM_RL_V2", raising=False)
    monkeypatch.delenv("V2_ALLOW_PRETRAIN_OVERWRITE", raising=False)
    monkeypatch.setenv("AGENT_HIDDEN_SIZE", "64")
    monkeypatch.setenv("AGENT_RECURRENT_SIZE", "16")
    monkeypatch.setenv("AGENT_TRANSFORMER_HEADS", "4")
    monkeypatch.setenv("AGENT_TRANSFORMER_LAYERS", "1")
    monkeypatch.setenv("AGENT_PLANNER_AUX_OUTPUT_DIM", "32")

    store = TeacherDatasetStore(str(tmp_path / "dataset"))
    split_keys = {}
    for idx in range(100):
        key = f"pretrain:{idx}"
        split_keys.setdefault(split_for_episode(key), key)
        if len(split_keys) == 3:
            break
    for index, split in enumerate(("train", "validation", "test"), start=1):
        store.append_episode(
            f"episode-{split}",
            [_sample(1000 + index, f"game-{split}")],
            split_key=split_keys[split],
        )

    output = tmp_path / "pretrain"
    report = pretrain(
        str(store.root),
        str(output),
        epochs=1,
        batch_size=1,
        allow_small_dataset=True,
        random_seed=7,
    )
    checkpoint = torch.load(output / "bc_best.pth", map_location="cpu", weights_only=False)
    assert checkpoint["experiment_version"] == "tfm-rl-v2"
    assert checkpoint["fresh_weights"] is True
    assert report["random_seed"] == 7
    assert report["dataset_audit"]["valid"] is True

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        pretrain(
            str(store.root),
            str(output),
            epochs=1,
            batch_size=1,
            allow_small_dataset=True,
            random_seed=7,
        )
