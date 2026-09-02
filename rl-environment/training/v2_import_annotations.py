"""Convert Decision Explainer annotations into weighted v2 teacher samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.teacher_dataset import SCHEMA_VERSION, TeacherDatasetStore, source_weight


def import_annotations(snapshot_dir: str, annotation_dir: str, dataset_dir: str) -> dict:
    snapshots = Path(snapshot_dir).expanduser().resolve()
    annotations = Path(annotation_dir).expanduser().resolve()
    store = TeacherDatasetStore(dataset_dir)
    imported = 0
    skipped = 0
    invalid = 0
    for annotation_path in sorted(annotations.glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if bool(annotation.get("skip", False)):
            skipped += 1
            continue
        snapshot_id = str(annotation.get("snapshot_id", annotation_path.stem))
        snapshot_path = snapshots / f"{snapshot_id}.json"
        if not snapshot_path.is_file():
            invalid += 1
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        prompt = snapshot.get("prompt", {}) or {}
        snapshot_seed = prompt.get("seed")
        state = snapshot.get("state", {}) or {}
        bundle = state.get("planner_bundle", {}) or {}
        descriptors = list((snapshot.get("policy", {}) or {}).get("action_descriptors", []) or [])
        accepted = {int(item) for item in (annotation.get("accepted_action_indices", []) or [])}
        positions = [idx for idx, row in enumerate(descriptors) if int(row.get("action_index", -1)) in accepted]
        if not bundle or not descriptors or not positions:
            invalid += 1
            continue
        probabilities = [0.0] * len(descriptors)
        for position in positions:
            probabilities[position] = 1.0 / len(positions)
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"human-{snapshot_id}",
            "planner_bundle": bundle,
            "action_descriptors": descriptors,
            "action_indices": [int(row.get("action_index", -1)) for row in descriptors],
            "teacher_probabilities": probabilities,
            "chosen_action_position": int(positions[0]),
            "phase_index": int(prompt.get("phase_index", 0) or 0),
            "confidence": 1.0,
            "source": "human.annotation.v1",
            "sample_weight": source_weight("human.annotation.v1", 1.0),
            "seed": int(snapshot_seed) if snapshot_seed is not None else -1,
            "game_id": str(prompt.get("game_id", "") or ""),
            "value_target": 0.0,
            "rank": 0,
            "vp": 0.0,
            "vp_mean": 0.0,
            "note": str(annotation.get("note", "") or ""),
        }
        split_key = f"seed:{int(snapshot_seed)}" if snapshot_seed is not None else f"human:{snapshot_id}"
        store.append_episode(f"human-{snapshot_id}", [sample], split_key=split_key)
        imported += 1
    return {"imported": imported, "skipped": skipped, "invalid": invalid, "dataset_counts": store.counts()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import human Decision Explainer labels into the v2 dataset")
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(import_annotations(args.snapshots, args.annotations, args.dataset), indent=2))


if __name__ == "__main__":
    main()
