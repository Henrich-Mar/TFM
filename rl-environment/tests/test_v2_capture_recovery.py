from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v2_import_annotations import analyze_annotations, import_annotations
from v2_runtime import assert_stage_allowed, stage1_unlocked


def _descriptor(action_index: int, label: str = "A") -> dict:
    return {
        "action_index": action_index,
        "family": "or",
        "label": label,
        "decoded_action": {"type": "or", "index": action_index},
    }


def _write_guided_pair(
    snapshots: Path,
    annotations: Path,
    *,
    snapshot_id: str,
    game_id: str,
    agent_id: str,
    annotated_at: str,
    descriptors: list,
    chosen_action_index: int,
    is_forced: bool = False,
    decision_sequence: int = 0,
) -> None:
    legal = [int(row["action_index"]) for row in descriptors]
    snapshot = {
        "schema_version": "decision_snapshot.v1",
        "agent": {"id": agent_id},
        "prompt": {
            "game_id": game_id,
            "seed": 10000,
            "phase_index": 0,
            "decision_sequence": decision_sequence,
        },
        "policy": {
            "chosen_action_index": chosen_action_index,
            "legal_actions": legal,
            "action_descriptors": descriptors,
        },
        "diagnostics": {
            "external_policy": {
                "is_forced": is_forced,
                "used_fallback": False,
            }
        },
        "state": {},
    }
    annotation = {
        "snapshot_id": snapshot_id,
        "annotated_at": annotated_at,
        "accepted_action_indices": [chosen_action_index],
        "skip": False,
    }
    (snapshots / f"{snapshot_id}.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (annotations / f"{snapshot_id}.json").write_text(json.dumps(annotation), encoding="utf-8")


def _teacher_row(descriptors: list, chosen_position: int, game_id: str) -> dict:
    count = len(descriptors)
    return {
        "game_id": game_id,
        "seed": 10000,
        "action_indices": [int(row["action_index"]) for row in descriptors],
        "action_descriptors": descriptors,
        "chosen_action_position": chosen_position,
        "planner_bundle": {
            "action_tokens": np.zeros((count, 8), dtype=np.float32),
            "action_mask": np.asarray([True] * count, dtype=np.bool_),
            "action_indices": np.asarray([int(row["action_index"]) for row in descriptors], dtype=np.int64),
        },
        "fallback_used": False,
        "value_target": 1.0,
        "rank": 1,
        "vp": 50.0,
        "vp_mean": 40.0,
    }


def test_stage1_stays_blocked_until_explicitly_unlocked(monkeypatch) -> None:
    monkeypatch.delenv("V2_ALLOW_STAGE1", raising=False)
    assert stage1_unlocked() is False
    assert_stage_allowed(0, context="unit")
    with pytest.raises(RuntimeError, match="Stage 1 remains blocked"):
        assert_stage_allowed(1, context="unit")
    monkeypatch.setenv("V2_ALLOW_STAGE1", "1")
    assert_stage_allowed(1, context="unit")


def test_legacy_recovery_hydrates_and_counts_preference_labels(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    annotations = tmp_path / "annotations"
    teacher = tmp_path / "teacher"
    dataset = tmp_path / "dataset"
    snapshots.mkdir()
    annotations.mkdir()
    (teacher / "test").mkdir(parents=True)

    descriptors_a = [_descriptor(10, "Pass?"), _descriptor(11, "Card")]
    descriptors_b = [_descriptor(20, "Only")]
    _write_guided_pair(
        snapshots,
        annotations,
        snapshot_id="snap-pref",
        game_id="g-recovery",
        agent_id="teacher-v1-seat-0",
        annotated_at="2026-09-14T07:00:00Z",
        descriptors=descriptors_a,
        chosen_action_index=10,
        is_forced=False,
    )
    _write_guided_pair(
        snapshots,
        annotations,
        snapshot_id="snap-forced",
        game_id="g-recovery",
        agent_id="teacher-v1-seat-0",
        annotated_at="2026-09-14T07:00:01Z",
        descriptors=descriptors_b,
        chosen_action_index=20,
        is_forced=True,
    )
    episode = [
        _teacher_row(descriptors_a, 0, "g-recovery"),
        _teacher_row(descriptors_b, 0, "g-recovery"),
    ]
    with gzip.open(teacher / "test" / "episode_recovery_2.pkl.gz", "wb") as handle:
        pickle.dump(episode, handle)

    report = analyze_annotations(
        str(snapshots),
        str(annotations),
        teacher_source=str(teacher),
        game_id="g-recovery",
        agent_id="teacher-v1-seat-0",
    )
    assert report["valid"] is True
    assert report["records"] == 2
    assert report["forced_labels"] == 1
    assert report["preference_labels"] == 1
    assert report["fallback_count"] == 0
    assert report["rejection_count"] == 0
    assert report["illegal_selection_count"] == 0
    assert report["hydrated_bundles"] == 2
    assert report["games"][0]["importable"] is True

    imported = import_annotations(
        str(snapshots),
        str(annotations),
        str(dataset),
        teacher_source=str(teacher),
        game_id="g-recovery",
        agent_id="teacher-v1-seat-0",
        strict=True,
    )
    assert imported["imported"] == 2


def test_incomplete_game_marker_blocks_import(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    annotations = tmp_path / "annotations"
    snapshots.mkdir()
    annotations.mkdir()
    (snapshots / "incomplete_games.json").write_text(
        json.dumps({"games": [{"game_id": "g-bad", "reason": "fragment"}]}),
        encoding="utf-8",
    )
    descriptors = [_descriptor(1)]
    _write_guided_pair(
        snapshots,
        annotations,
        snapshot_id="snap-bad",
        game_id="g-bad",
        agent_id="teacher-v1-seat-0",
        annotated_at="2026-09-14T07:00:00Z",
        descriptors=descriptors,
        chosen_action_index=1,
    )
    report = analyze_annotations(str(snapshots), str(annotations), game_id="g-bad")
    assert report["valid"] is False
    assert any("incomplete_game:g-bad" in item for item in report["errors"])
    assert report["games"][0]["importable"] is False
    assert "fragment" in str(report["games"][0].get("incomplete_reason", ""))


def test_live_g8d2_recovery_counts_when_artifacts_present() -> None:
    snapshots = REPO / "rl-environment" / "debug_snapshots"
    annotations = snapshots / "annotations"
    teacher = REPO / "rl-v2" / "action-test-recovery-20260914" / "teacher-dataset"
    if not snapshots.is_dir() or not annotations.is_dir() or not teacher.is_dir():
        pytest.skip("recovery artifacts are not present on this machine")
    snap_count = len(list(snapshots.glob("*g8d2e4e20f93*.json")))
    ann_count = len(list(annotations.glob("*g8d2e4e20f93*.json")))
    if snap_count != 64 or ann_count != 64:
        pytest.skip(f"unexpected g8d2 capture counts snaps={snap_count} anns={ann_count}")

    report = analyze_annotations(
        str(snapshots),
        str(annotations),
        teacher_source=str(teacher),
        game_id="g8d2e4e20f93",
        agent_id="teacher-v1-seat-0",
    )
    if any(item.startswith("teacher_source_unreadable:") for item in report["errors"]):
        pytest.skip("host Python cannot unpickle recovered teacher shards; run audit inside Docker")

    assert report["records"] == 64
    assert report["forced_labels"] == 1
    assert report["preference_labels"] == 63
    assert report["fallback_count"] == 0
    assert report["rejection_count"] == 0
    assert report["illegal_selection_count"] == 0
    assert report["overflow_count"] == 0
    assert report["valid"] is True
    assert report["games"][0]["importable"] is True
