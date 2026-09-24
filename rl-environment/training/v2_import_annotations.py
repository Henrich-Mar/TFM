"""Strictly convert guided Decision Explainer labels into v2 teacher samples."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from training.teacher_dataset import TeacherDatasetStore, active_schema_version, source_weight


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_.") or "unknown"


def _descriptor_indices(snapshot: Dict[str, Any]) -> List[int]:
    return [
        int(row.get("action_index", -1))
        for row in ((snapshot.get("policy", {}) or {}).get("action_descriptors", []) or [])
    ]


def _descriptor_fingerprint(descriptors: Iterable[Dict[str, Any]]) -> str:
    canonical = [
        {
            "action_index": int(row.get("action_index", -1)),
            "family": str(row.get("family", "") or ""),
            "label": str(row.get("label", "") or ""),
            "decoded_action": row.get("decoded_action", {}),
        }
        for row in descriptors
    ]
    return json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _ordered_records(
    snapshot_dir: Path,
    annotation_dir: Path,
    game_id: Optional[str],
    agent_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            snapshot_id = str(annotation.get("snapshot_id", annotation_path.stem) or annotation_path.stem)
            snapshot_path = snapshot_dir / f"{snapshot_id}.json"
            if not snapshot_path.is_file():
                errors.append(f"annotation_without_snapshot:{snapshot_id}")
                continue
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"malformed_annotation:{annotation_path.name}:{exc}")
            continue
        prompt = snapshot.get("prompt", {}) or {}
        snapshot_agent = str((snapshot.get("agent", {}) or {}).get("id", "") or "")
        snapshot_game = str(prompt.get("game_id", "") or "")
        if game_id and snapshot_game != str(game_id):
            continue
        if agent_id and snapshot_agent != str(agent_id):
            continue
        records.append({
            "snapshot_id": snapshot_id,
            "snapshot": snapshot,
            "annotation": annotation,
            "game_id": snapshot_game,
            "agent_id": snapshot_agent,
            "decision_sequence": int(prompt.get("decision_sequence", 0) or 0),
            "annotated_at": str(annotation.get("annotated_at", "") or ""),
        })

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["game_id"], record["agent_id"])].append(record)
    ordered: List[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        if rows and all(int(row.get("decision_sequence", 0)) > 0 for row in rows):
            rows.sort(key=lambda row: (int(row["decision_sequence"]), row["snapshot_id"]))
        else:
            rows.sort(key=lambda row: (row["annotated_at"], row["snapshot_id"]))
        ordered.extend(rows)
    return ordered, errors


def _load_incomplete_game_ids(snapshot_dir: Path) -> Dict[str, str]:
    marker = snapshot_dir / "incomplete_games.json"
    if not marker.is_file():
        return {}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        rows = payload.get("games", payload.get("incomplete_games", []))
    else:
        rows = payload
    marked: Dict[str, str] = {}
    if not isinstance(rows, list):
        return marked
    for row in rows:
        if isinstance(row, str):
            marked[str(row)] = "marked_incomplete"
            continue
        if not isinstance(row, dict):
            continue
        game = str(row.get("game_id", "") or "").strip()
        if not game:
            continue
        marked[game] = str(row.get("reason", "marked_incomplete") or "marked_incomplete")
    return marked


def _teacher_source_read_errors(root: Optional[Path]) -> List[str]:
    if root is None:
        return []
    errors: List[str] = []
    shard_count = 0
    for path in sorted(root.rglob("episode_*.pkl.gz")):
        shard_count += 1
        try:
            with gzip.open(path, "rb") as handle:
                pickle.load(handle)
        except Exception as exc:
            errors.append(f"teacher_source_unreadable:{path.name}:{exc}")
    if shard_count == 0:
        errors.append(f"teacher_source_empty:{root}")
    return errors


def _load_legacy_episodes(root: Optional[Path], game_id: str) -> List[List[Dict[str, Any]]]:
    if root is None:
        return []
    episodes: List[List[Dict[str, Any]]] = []
    for path in sorted(root.rglob("episode_*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as handle:
                rows = list(pickle.load(handle) or [])
        except Exception:
            continue
        if rows and {str(row.get("game_id", "") or "") for row in rows} == {str(game_id)}:
            episodes.append(rows)
    return episodes


def _selected_action(record: Dict[str, Any]) -> Optional[int]:
    annotation = record["annotation"]
    if bool(annotation.get("skip", False)):
        return None
    accepted = sorted({int(item) for item in (annotation.get("accepted_action_indices", []) or [])})
    proposed = int((record["snapshot"].get("policy", {}) or {}).get("chosen_action_index", -1))
    return proposed if proposed in accepted else (accepted[0] if accepted else None)


def _match_legacy_episode(records: List[Dict[str, Any]], episodes: Iterable[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    matches: List[List[Dict[str, Any]]] = []
    for rows in episodes:
        if len(rows) != len(records):
            continue
        valid = True
        for record, row in zip(records, rows):
            indices = _descriptor_indices(record["snapshot"])
            row_indices = [int(item) for item in (row.get("action_indices", []) or [])]
            if indices != row_indices:
                valid = False
                break
            if _descriptor_fingerprint((record["snapshot"].get("policy", {}) or {}).get("action_descriptors", []) or []) != _descriptor_fingerprint(row.get("action_descriptors", []) or []):
                valid = False
                break
            selected = _selected_action(record)
            position = int(row.get("chosen_action_position", -1))
            row_selected = row_indices[position] if 0 <= position < len(row_indices) else None
            if selected is not None and selected != row_selected:
                valid = False
                break
        if valid:
            matches.append(rows)
    if len(matches) > 1:
        raise RuntimeError("legacy guided episode match is ambiguous")
    return matches[0] if matches else None


def analyze_annotations(
    snapshot_dir: str,
    annotation_dir: str,
    teacher_source: Optional[str] = None,
    game_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    snapshots = Path(snapshot_dir).expanduser().resolve()
    annotations = Path(annotation_dir).expanduser().resolve()
    teacher_root = Path(teacher_source).expanduser().resolve() if teacher_source else None
    incomplete_games = _load_incomplete_game_ids(snapshots)
    records, errors = _ordered_records(snapshots, annotations, game_id, agent_id)
    errors.extend(_teacher_source_read_errors(teacher_root))
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["game_id"], record["agent_id"])].append(record)
    snapshot_inventory: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for path in sorted(snapshots.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt = payload.get("prompt", {}) or {}
        snapshot_game = str(prompt.get("game_id", "") or "")
        snapshot_agent = str((payload.get("agent", {}) or {}).get("id", "") or "")
        if game_id and snapshot_game != str(game_id):
            continue
        if agent_id and snapshot_agent != str(agent_id):
            continue
        snapshot_inventory[(snapshot_game, snapshot_agent)].append(path.stem)

    for marked_game, reason in sorted(incomplete_games.items()):
        if game_id and marked_game != str(game_id):
            continue
        matching_keys = [key for key in set(grouped).union(snapshot_inventory) if key[0] == marked_game]
        if matching_keys or (game_id and marked_game == str(game_id)):
            errors.append(f"incomplete_game:{marked_game}:{reason}")

    samples_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    skipped = 0
    forced = 0
    fallback_count = 0
    rejection_count = 0
    hydrated = 0
    illegal_selection_count = 0
    empty_bundle_count = 0
    overflow_count = 0
    missing_annotation_count = 0
    game_reports: List[Dict[str, Any]] = []
    for key in sorted(set(grouped).union(snapshot_inventory)):
        group_records = grouped.get(key, [])
        group_game, group_agent = key
        all_game_snapshots = snapshot_inventory.get(key, [])
        annotated_ids = {record["snapshot_id"] for record in group_records}
        missing_annotations = sorted(set(all_game_snapshots) - annotated_ids)
        if missing_annotations:
            errors.append(f"incomplete_game:{group_game}:{len(missing_annotations)}_missing_annotations")
            missing_annotation_count += len(missing_annotations)
        if group_game in incomplete_games:
            game_reports.append({
                "game_id": group_game,
                "agent_id": group_agent,
                "snapshots": len(all_game_snapshots),
                "annotations": len(group_records),
                "samples": 0,
                "legacy_hydrated": False,
                "missing_annotations": len(missing_annotations),
                "decision_order": [],
                "importable": False,
                "incomplete_reason": incomplete_games[group_game],
            })
            continue

        legacy_episodes = _load_legacy_episodes(teacher_root, group_game)
        legacy = (
            _match_legacy_episode(group_records, legacy_episodes)
            if group_records else None
        )
        if teacher_root is not None and group_records and legacy is None:
            if legacy_episodes:
                errors.append(f"legacy_episode_unmatched:{group_game}:{group_agent}")
            else:
                errors.append(f"legacy_episode_missing:{group_game}")
        group_samples: List[Dict[str, Any]] = []
        for index, record in enumerate(group_records):
            snapshot = record["snapshot"]
            annotation = record["annotation"]
            if bool(annotation.get("skip", False)):
                skipped += 1
                continue
            policy = snapshot.get("policy", {}) or {}
            prompt = snapshot.get("prompt", {}) or {}
            diagnostics = snapshot.get("diagnostics", {}) or {}
            external = diagnostics.get("external_policy", {}) or {}
            execution = snapshot.get("execution", {}) or {}
            descriptors = list(policy.get("action_descriptors", []) or [])
            accepted = sorted({int(item) for item in (annotation.get("accepted_action_indices", []) or [])})
            legal = [int(item) for item in (policy.get("legal_actions", []) or [])]
            invalid_accepted = [item for item in accepted if item not in legal]
            selected = _selected_action(record)
            positions = [idx for idx, descriptor in enumerate(descriptors) if int(descriptor.get("action_index", -1)) in accepted]
            selected_positions = [
                idx for idx, descriptor in enumerate(descriptors)
                if selected is not None and int(descriptor.get("action_index", -1)) == int(selected)
            ]
            row = legacy[index] if legacy is not None else None
            bundle = (snapshot.get("state", {}) or {}).get("planner_bundle", {}) or {}
            if not bundle and row is not None:
                bundle = row.get("planner_bundle", {}) or {}
                hydrated += 1
            server_accepted = execution.get("server_accepted")
            if server_accepted is None and row is not None:
                server_accepted = True
            fallback_used = bool(external.get("used_fallback", False) or (row or {}).get("fallback_used", False))
            is_forced = bool(external.get("is_forced", len(descriptors) == 1))
            forced += int(is_forced)
            fallback_count += int(fallback_used)
            rejection_count += int(server_accepted is False)
            validation_errors: List[str] = []
            if not bundle:
                validation_errors.append("missing_planner_bundle")
                empty_bundle_count += 1
            if not descriptors or not legal:
                validation_errors.append("empty_legal_actions")
            if invalid_accepted or not positions or selected is None:
                validation_errors.append("invalid_human_selection")
                illegal_selection_count += 1
            if server_accepted is not True:
                validation_errors.append("server_not_accepted")
            if fallback_used:
                validation_errors.append("fallback_used")
            snapshot_errors = [
                str(item) for item in (execution.get("validation_errors", []) or [])
            ]
            if any("overflow" in item.lower() for item in snapshot_errors):
                validation_errors.append("action_space_overflow")
                overflow_count += 1
            selected_descriptor = next(
                (
                    item for item in descriptors
                    if selected is not None and int(item.get("action_index", -1)) == int(selected)
                ),
                {},
            )
            sample = {
                "schema_version": active_schema_version(),
                "sample_id": f"human-{record['snapshot_id']}",
                "planner_bundle": bundle,
                "action_descriptors": descriptors,
                "action_indices": [int(row.get("action_index", -1)) for row in descriptors],
                "teacher_probabilities": [
                    (1.0 / len(positions)) if idx in positions else 0.0
                    for idx in range(len(descriptors))
                ],
                "chosen_action_position": int(selected_positions[0]) if selected_positions else -1,
                "phase_index": int(prompt.get("phase_index", 0) or 0),
                "confidence": 1.0,
                "is_forced": is_forced,
                "policy_target_valid": not is_forced,
                "source": "human.annotation.v1",
                "action_source": "human_annotation",
                "sample_weight": source_weight("human.annotation.v1", 1.0, is_forced=is_forced),
                "seed": int(prompt.get("seed")) if prompt.get("seed") is not None else int((row or {}).get("seed", -1)),
                "game_id": group_game,
                "value_target": float((row or {}).get("value_target", 0.0)),
                "value_target_valid": row is not None,
                "rank": int((row or {}).get("rank", 0) or 0),
                "vp": float((row or {}).get("vp", 0.0) or 0.0),
                "vp_mean": float((row or {}).get("vp_mean", 0.0) or 0.0),
                "note": str(annotation.get("note", "") or ""),
                "selected_action_payload": dict(selected_descriptor.get("decoded_action", {}) or {}),
                "server_accepted": server_accepted is True,
                "fallback_used": fallback_used,
                "training_eligible": not validation_errors,
                "validation_errors": validation_errors,
                "source_snapshot_id": record["snapshot_id"],
                "decision_sequence": int(record.get("decision_sequence", 0) or index + 1),
            }
            if validation_errors:
                errors.extend(f"{record['snapshot_id']}:{reason}" for reason in validation_errors)
            group_samples.append(sample)
        samples_by_group[key] = group_samples
        game_reports.append({
            "game_id": group_game,
            "agent_id": group_agent,
            "snapshots": len(all_game_snapshots),
            "annotations": len(group_records),
            "samples": len(group_samples),
            "legacy_hydrated": legacy is not None,
            "missing_annotations": len(missing_annotations),
            "decision_order": [
                int(record.get("decision_sequence", 0) or index + 1)
                for index, record in enumerate(group_records)
            ],
            "importable": bool(group_records) and not missing_annotations and all(
                bool(sample.get("training_eligible", False)) for sample in group_samples
            ),
        })

    for path in sorted((snapshots / "quarantine").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt = payload.get("prompt", {}) or {}
        snapshot_game = str(prompt.get("game_id", "") or "")
        snapshot_agent = str((payload.get("agent", {}) or {}).get("id", "") or "")
        if game_id and snapshot_game != str(game_id):
            continue
        if agent_id and snapshot_agent != str(agent_id):
            continue
        reasons = [
            str(item)
            for item in ((payload.get("execution", {}) or {}).get("validation_errors", []) or [])
        ]
        overflow_count += int(any("overflow" in reason.lower() for reason in reasons))
        errors.append(f"quarantined_snapshot:{path.stem}")

    return {
        "valid": not errors,
        "errors": errors,
        "records": len(records),
        "samples": sum(len(rows) for rows in samples_by_group.values()),
        "skipped": skipped,
        "forced_labels": forced,
        "preference_labels": sum(
            int(bool(sample.get("policy_target_valid", False)))
            for rows in samples_by_group.values()
            for sample in rows
        ),
        "fallback_count": fallback_count,
        "rejection_count": rejection_count,
        "illegal_selection_count": illegal_selection_count,
        "overflow_count": overflow_count,
        "missing_annotation_count": missing_annotation_count,
        "bundle_coverage": {
            "present": sum(len(rows) for rows in samples_by_group.values()) - empty_bundle_count,
            "missing": empty_bundle_count,
        },
        "hydrated_bundles": hydrated,
        "incomplete_games": incomplete_games,
        "games": game_reports,
        "samples_by_group": samples_by_group,
    }


def import_annotations(
    snapshot_dir: str,
    annotation_dir: str,
    dataset_dir: str,
    teacher_source: Optional[str] = None,
    game_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    analysis = analyze_annotations(snapshot_dir, annotation_dir, teacher_source, game_id, agent_id)
    if strict and not bool(analysis["valid"]):
        raise RuntimeError("annotation import failed strict audit: " + "; ".join(analysis["errors"][:20]))
    store = TeacherDatasetStore(dataset_dir)
    imported = 0
    for (group_game, group_agent), samples in analysis.pop("samples_by_group").items():
        eligible = [sample for sample in samples if bool(sample.get("training_eligible", False))]
        ineligible = [sample for sample in samples if not bool(sample.get("training_eligible", False))]
        if ineligible:
            store.quarantine_episode(
                f"human-guided-{_safe_name(group_game)}-{_safe_name(group_agent)}",
                ineligible,
                [reason for sample in ineligible for reason in sample.get("validation_errors", [])],
            )
        if not eligible:
            continue
        seed = eligible[0].get("seed")
        split_key = f"seed:{int(seed)}" if seed is not None and int(seed) >= 0 else f"human-game:{group_game}"
        store.append_episode(
            f"human-guided-{_safe_name(group_game)}-{_safe_name(group_agent)}",
            eligible,
            split_key=split_key,
        )
        imported += len(eligible)
    analysis["imported"] = imported
    analysis["dataset_counts"] = store.counts()
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly import guided human labels into the v2 dataset")
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-source")
    parser.add_argument("--game-id")
    parser.add_argument("--agent-id")
    args = parser.parse_args()
    report = import_annotations(
        args.snapshots,
        args.annotations,
        args.dataset,
        teacher_source=args.teacher_source,
        game_id=args.game_id,
        agent_id=args.agent_id,
        strict=True,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
