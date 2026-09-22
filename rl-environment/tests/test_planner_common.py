from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.planner_common import PLANNER_TOKEN_DIM, pad_bundle_batch


def _bundle(action_count: int, *, terminal: bool = False) -> dict:
    return {
        "world_tokens": np.zeros((1, PLANNER_TOKEN_DIM), dtype=np.float32),
        "world_token_types": np.zeros((1,), dtype=np.int64),
        "world_mask": np.ones((1,), dtype=np.bool_),
        "hand_tokens": np.zeros((0, PLANNER_TOKEN_DIM), dtype=np.float32),
        "hand_mask": np.zeros((0,), dtype=np.bool_),
        "action_tokens": np.zeros((action_count, PLANNER_TOKEN_DIM), dtype=np.float32),
        "action_mask": np.ones((action_count,), dtype=np.bool_),
        "action_indices": np.arange(action_count, dtype=np.int64),
        "action_positions": np.arange(action_count, dtype=np.int64),
        "global_scalars": np.zeros((16,), dtype=np.float32),
        "terminal": terminal,
    }


def test_padding_rejects_empty_active_bundle_instead_of_inventing_legality() -> None:
    with pytest.raises(ValueError, match="zero actions.*terminal"):
        pad_bundle_batch([_bundle(0)], torch.device("cpu"))


def test_padding_preserves_explicit_terminal_empty_bundle() -> None:
    batch = pad_bundle_batch([_bundle(0, terminal=True)], torch.device("cpu"))

    assert batch["action_mask"].shape == (1, 1)
    assert not bool(batch["action_mask"].any())
    assert bool(batch["terminal_mask"].tolist() == [True])
    assert int(batch["action_indices"][0, 0].item()) == -1


def test_padding_does_not_make_terminal_row_legal_when_mixed_with_active_row() -> None:
    batch = pad_bundle_batch(
        [_bundle(2), _bundle(0, terminal=True)],
        torch.device("cpu"),
    )

    assert batch["action_mask"].tolist() == [[True, True], [False, False]]
    assert batch["terminal_mask"].tolist() == [False, True]


def test_padding_rejects_terminal_bundle_with_action_rows() -> None:
    with pytest.raises(ValueError, match="terminal.*action rows"):
        pad_bundle_batch([_bundle(1, terminal=True)], torch.device("cpu"))
