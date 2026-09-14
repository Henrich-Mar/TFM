"""Audit award/milestone claim behavior from teacher datasets or debug snapshots.

Reports whether claim/fund actions are legal, taken, named correctly, and timed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GENERIC_MILESTONE_LABELS = {
    "claim a milestone",
    "claim milestone",
    "select a milestone",
}


def _generation(sample: Dict[str, Any], snap: Optional[Dict[str, Any]] = None) -> int:
    for source in (sample, snap or {}):
        if not isinstance(source, dict):
            continue
        prompt = source.get("prompt") or {}
        if isinstance(prompt, dict) and prompt.get("generation") is not None:
            try:
                return int(prompt.get("generation") or 0)
            except (TypeError, ValueError):
                pass
        for key in ("generation", "game_generation"):
            if source.get(key) is not None:
                try:
                    return int(source.get(key) or 0)
                except (TypeError, ValueError):
                    pass
        state = source.get("player_state") or source.get("state") or {}
        if isinstance(state, dict):
            game = state.get("game") or {}
            if isinstance(game, dict) and game.get("generation") is not None:
                try:
                    return int(game.get("generation") or 0)
                except (TypeError, ValueError):
                    pass
            this_player = state.get("this_player") or state.get("thisPlayer") or {}
            if isinstance(this_player, dict) and this_player.get("generation") is not None:
                try:
                    return int(this_player.get("generation") or 0)
                except (TypeError, ValueError):
                    pass
    return 0


def _chosen_descriptor(descriptors: Sequence[Dict[str, Any]], sample: Dict[str, Any]) -> Dict[str, Any]:
    if not descriptors:
        return {}
    pos = sample.get("chosen_action_position")
    if pos is not None:
        try:
            idx = int(pos)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(descriptors):
            return dict(descriptors[idx] or {})
    chosen = sample.get("chosen_action_descriptor")
    if isinstance(chosen, dict) and chosen:
        return dict(chosen)
    action_index = sample.get("chosen_action_index")
    if action_index is not None:
        for row in descriptors:
            try:
                if int(row.get("action_index", -999)) == int(action_index):
                    return dict(row)
            except (TypeError, ValueError):
                continue
    return {}


def _is_generic_milestone_name(value: Any) -> bool:
    text = " ".join(str(value or "").strip().lower().split())
    return (not text) or text in GENERIC_MILESTONE_LABELS or (
        "claim" in text and "milestone" in text and len(text.split()) <= 4
    )


def _iter_teacher_samples(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for split in ("train", "validation", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("episode_*.pkl.gz")):
            with gzip.open(path, "rb") as handle:
                items = list(pickle.load(handle) or [])
            for item in items:
                if isinstance(item, dict):
                    yield split, item


def _iter_snapshot_samples(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for path in sorted(root.glob("*.json")):
        if path.parent.name == "annotations":
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(snap, dict):
            continue
        policy = snap.get("policy") or {}
        descriptors = list(policy.get("action_descriptors") or [])
        chosen = policy.get("chosen_action_descriptor") or {}
        sample = {
            "action_descriptors": descriptors,
            "chosen_action_descriptor": chosen,
            "chosen_action_index": policy.get("chosen_action_index"),
            "chosen_action_position": policy.get("chosen_action_position"),
            "source": "debug_snapshot",
            "prompt": snap.get("prompt") or {},
            "state": snap.get("state") or {},
            "snapshot_id": snap.get("snapshot_id") or path.stem,
            "snapshot_path": str(path),
        }
        yield "snapshot", sample


def _analyze_samples(rows: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    family_chosen: Counter[str] = Counter()
    legal_offer: Counter[str] = Counter()
    claim = {"offered": 0, "chosen": 0, "missed": 0, "generic_names": 0, "named": 0}
    fund = {"offered": 0, "chosen": 0, "missed": 0}
    claim_names: Counter[str] = Counter()
    fund_names: Counter[str] = Counter()
    claim_by_gen: Counter[int] = Counter()
    fund_by_gen: Counter[int] = Counter()
    missed_claims: List[Dict[str, Any]] = []
    generic_claim_examples: List[Dict[str, Any]] = []
    samples = 0
    splits: Counter[str] = Counter()

    for split, sample in rows:
        samples += 1
        splits[split] += 1
        descriptors = [row for row in (sample.get("action_descriptors") or []) if isinstance(row, dict)]
        chosen = _chosen_descriptor(descriptors, sample)
        family = str(chosen.get("family") or "other")
        family_chosen[family] += 1
        legal_families = {str(row.get("family") or "other") for row in descriptors}
        for legal_family in legal_families:
            legal_offer[legal_family] += 1
        generation = _generation(sample)
        claim_opts = [row for row in descriptors if str(row.get("family")) == "claim_milestone"]
        fund_opts = [row for row in descriptors if str(row.get("family")) == "fund_award"]

        if claim_opts:
            claim["offered"] += 1
            for row in claim_opts:
                name = str(row.get("milestone_name") or row.get("label") or "")
                if _is_generic_milestone_name(name):
                    claim["generic_names"] += 1
                    if len(generic_claim_examples) < 12:
                        generic_claim_examples.append(
                            {
                                "name": name,
                                "action_index": row.get("action_index"),
                                "snapshot_id": sample.get("snapshot_id"),
                                "source": sample.get("source"),
                            }
                        )
                else:
                    claim["named"] += 1
            if family == "claim_milestone":
                claim["chosen"] += 1
                claim_names[str(chosen.get("milestone_name") or chosen.get("label") or "?")] += 1
                claim_by_gen[generation] += 1
            else:
                claim["missed"] += 1
                if len(missed_claims) < 12:
                    missed_claims.append(
                        {
                            "chosen_family": family,
                            "chosen_label": chosen.get("label"),
                            "legal_claims": [
                                row.get("milestone_name") or row.get("label") for row in claim_opts
                            ],
                            "generation": generation,
                            "source": sample.get("source"),
                        }
                    )

        if fund_opts:
            fund["offered"] += 1
            if family == "fund_award":
                fund["chosen"] += 1
                fund_names[str(chosen.get("award_name") or chosen.get("label") or "?")] += 1
                fund_by_gen[generation] += 1
            else:
                fund["missed"] += 1

    named_leaf_rate = None
    named_total = int(claim["named"]) + int(claim["generic_names"])
    if named_total:
        named_leaf_rate = float(claim["named"]) / float(named_total)

    return {
        "schema_version": "tfm_rl_v2.awards_milestones_audit.v1",
        "samples": samples,
        "splits": dict(splits),
        "family_chosen_top": family_chosen.most_common(20),
        "claim_milestone_chosen": int(family_chosen.get("claim_milestone", 0)),
        "fund_award_chosen": int(family_chosen.get("fund_award", 0)),
        "legal_offer_claim": int(legal_offer.get("claim_milestone", 0)),
        "legal_offer_fund": int(legal_offer.get("fund_award", 0)),
        "claim_when_legal": claim,
        "fund_when_legal": fund,
        "claim_take_rate": (claim["chosen"] / claim["offered"]) if claim["offered"] else None,
        "fund_take_rate": (fund["chosen"] / fund["offered"]) if fund["offered"] else None,
        "milestone_named_leaf_rate": named_leaf_rate,
        "claim_by_name": claim_names.most_common(30),
        "fund_by_name": fund_names.most_common(30),
        "claim_by_generation": dict(sorted(claim_by_gen.items())),
        "fund_by_generation": dict(sorted(fund_by_gen.items())),
        "missed_claims_sample": missed_claims,
        "generic_milestone_name_sample": generic_claim_examples,
        "valid_milestone_leaves": named_leaf_rate is None or named_leaf_rate >= 0.999,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit award/milestone claim naming and take rates")
    parser.add_argument("--teacher-dataset", help="Teacher dataset root with train/validation/test shards")
    parser.add_argument("--snapshots", help="Debug snapshot directory containing *.json decisions")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--require-named-milestones",
        action="store_true",
        help="Exit non-zero when any claim leaf still uses a generic menu title",
    )
    args = parser.parse_args()
    if not args.teacher_dataset and not args.snapshots:
        raise SystemExit("provide --teacher-dataset and/or --snapshots")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    if args.teacher_dataset:
        root = Path(args.teacher_dataset)
        if not root.exists():
            raise SystemExit(f"teacher dataset not found: {root}")
        rows.extend(_iter_teacher_samples(root))
    if args.snapshots:
        root = Path(args.snapshots)
        if not root.exists():
            raise SystemExit(f"snapshots directory not found: {root}")
        rows.extend(_iter_snapshot_samples(root))

    report = _analyze_samples(rows)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    if args.require_named_milestones and not bool(report.get("valid_milestone_leaves", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
