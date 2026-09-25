from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.agent import generation_is_runaway
from training.teacher_dataset import (
    SCHEMA_VERSION,
    TeacherDatasetRecorder,
    TeacherDatasetStore,
    load_reserved_benchmark_seeds,
    source_weight,
    split_for_episode,
)
from debug_decision_snapshot import load_snapshot_annotation, save_snapshot, save_snapshot_annotation


def _sample(seed: int = 123, game_id: str = "game-123") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "planner_bundle": {
            "action_tokens": np.zeros((2, 64), dtype=np.float32),
            "action_indices": np.asarray([10, 20], dtype=np.int64),
            "action_mask": np.asarray([True, True], dtype=np.bool_),
        },
        "action_descriptors": [
            {"action_index": 10, "action_position": 0, "decoded_action": {"type": "or", "index": 0}},
            {"action_index": 20, "action_position": 1, "decoded_action": {"type": "or", "index": 1}},
        ],
        "action_indices": [10, 20],
        "teacher_probabilities": [0.75, 0.25],
        "chosen_action_position": 0,
        "phase_index": 0,
        "source": "heuristic-teacher.v1",
        "confidence": 0.8,
        "sample_weight": 1.0,
        "seed": seed,
        "game_id": game_id,
        "value_target": 1.0,
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


def test_reserved_benchmark_seed_is_rejected(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    reserved_seed = min(load_reserved_benchmark_seeds())
    with pytest.raises(ValueError, match="reserved benchmark"):
        store.append_episode("leaked-benchmark", [_sample(seed=reserved_seed)])


def test_dataset_audit_detects_game_split_leakage(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    split_keys = {}
    for idx in range(100):
        key = f"probe:{idx}"
        split_keys.setdefault(split_for_episode(key), key)
        if len(split_keys) == 3:
            break
    store.append_episode("seat-a", [_sample(seed=123, game_id="same-game")], split_key=split_keys["train"])
    store.append_episode("seat-b", [_sample(seed=123, game_id="same-game")], split_key=split_keys["test"])
    audit = store.audit()
    assert not audit["valid"]
    assert audit["leaking_seeds"] == [123]
    assert audit["leaking_game_ids"] == ["same-game"]


def test_teacher_shard_write_is_atomic(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    target = store.append_episode("atomic", [_sample()])
    assert target.is_file()
    assert not list(target.parent.glob("*.tmp"))


def test_generation_cap_discards_incomplete_episode(tmp_path: Path) -> None:
    assert generation_is_runaway(30, limit=30) is False
    assert generation_is_runaway(31, limit=30) is True
    store = TeacherDatasetStore(str(tmp_path))
    recorder = TeacherDatasetRecorder(store, seed=960000)
    recorder.record_decision(
        "runaway-game",
        "teacher-v1-seat-0",
        {},
        {
            "external_policy": {
                "scores": [{"probability": 1.0, "action_index": 1}],
                "confidence": 1.0,
            },
            "action_descriptors": [{
                "action_index": 1,
                "family": "standard_project",
                "decoded_action": {"type": "standardProject", "project": "Asteroid"},
            }],
            "chosen_action_position": 0,
            "server_accepted": True,
        },
    )
    assert recorder._pending
    written = recorder.finish_episode("runaway-game", "teacher-v1-seat-0", {"completed": False})
    assert written is None
    assert recorder._pending == {}
    assert list(tmp_path.rglob("episode_*.pkl.gz")) == []


def test_human_and_low_confidence_weights() -> None:
    assert source_weight("human.annotation.v1", 1.0) == 4.0
    assert source_weight("heuristic-teacher.v1", 0.1) == 0.25
    assert source_weight("heuristic-teacher.v1", 0.9) == 1.0
    assert source_weight("heuristic-teacher.v1", 1.0, is_forced=True) == 0.25
    assert source_weight("human.annotation.v1", 1.0, is_forced=True) == 0.25


def test_rejected_or_payload_mismatched_sample_is_quarantined(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    rejected = _sample()
    rejected["server_accepted"] = False
    rejected["training_eligible"] = False
    rejected["validation_errors"] = ["server_not_accepted"]
    with pytest.raises(ValueError, match="not training eligible"):
        store.append_episode("rejected", [rejected])
    assert len(list((tmp_path / "quarantine").glob("episode_*.pkl.gz"))) == 1
    assert sum(store.counts().values()) == 0

    mismatched = _sample(game_id="game-mismatch")
    mismatched["selected_action_payload"] = {"type": "pass"}
    with pytest.raises(ValueError, match="does not match chosen descriptor"):
        store.append_episode("mismatched", [mismatched])
    assert len(list((tmp_path / "quarantine").glob("episode_*.pkl.gz"))) == 2


def test_forced_human_label_is_excluded_from_preference_count(tmp_path: Path) -> None:
    store = TeacherDatasetStore(str(tmp_path))
    sample = _sample()
    sample["source"] = "human.annotation.v1"
    sample["action_source"] = "human_annotation"
    sample["is_forced"] = True
    sample["policy_target_valid"] = False
    store.append_episode("forced-human", [sample])
    audit = store.audit()
    assert audit["source_counts"]["human"] == 1
    assert audit["source_counts"]["human_preference"] == 0


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
