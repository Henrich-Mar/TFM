from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.teacher_dataset import SCHEMA_VERSION, TeacherDatasetStore, source_weight, split_for_episode
from debug_decision_snapshot import load_snapshot_annotation, save_snapshot, save_snapshot_annotation


def _sample() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "planner_bundle": {"action_tokens": np.zeros((2, 64), dtype=np.float32)},
        "teacher_probabilities": [0.75, 0.25],
        "source": "heuristic-teacher.v1",
        "confidence": 0.8,
    }


def test_episode_split_is_stable_and_whole(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    store.append_episode("episode-one", [_sample(), _sample()])
    split = split_for_episode("episode-one")
    assert len(list(store.iter_samples(split))) == 2
    assert sum(store.counts().values()) == 2


def test_all_seats_from_one_seed_share_one_split(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    store.append_episode("seat-a", [_sample()], split_key="seed:123")
    store.append_episode("seat-b", [_sample()], split_key="seed:123")
    expected = split_for_episode("seed:123")
    rows = list(store.iter_samples(expected))
    assert len(rows) == 2
    assert {row["split_key"] for row in rows} == {"seed:123"}
    assert sum(store.counts().values()) == 2


def test_human_and_low_confidence_weights() -> None:
    assert source_weight("human.annotation.v1", 1.0) == 4.0
    assert source_weight("heuristic-teacher.v1", 0.1) == 0.25
    assert source_weight("heuristic-teacher.v1", 0.9) == 1.0


def test_snapshot_annotation_rejects_non_legal_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DECISION_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    saved = save_snapshot({
        "snapshot_id": "ignored",
        "prompt": {"prompt_type": "or"},
        "agent": {"id": "a"},
        "request": {"request_id": "r"},
        "policy": {"legal_actions": [10, 20]},
    })
    snapshot_id = saved["snapshot_id"]
    annotation = save_snapshot_annotation(snapshot_id, [20], note="best")
    assert annotation["accepted_action_indices"] == [20]
    assert load_snapshot_annotation(snapshot_id)["note"] == "best"
    try:
        save_snapshot_annotation(snapshot_id, [99])
    except ValueError:
        pass
    else:
        raise AssertionError("non-legal human label was accepted")
