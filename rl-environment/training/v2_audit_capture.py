"""Audit guided capture artifacts without mutating the teacher dataset."""
from __future__ import annotations

import argparse
import json

from training.v2_import_annotations import analyze_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict guided-capture training eligibility")
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--teacher-source")
    parser.add_argument("--game-id")
    parser.add_argument("--agent-id")
    args = parser.parse_args()
    report = analyze_annotations(
        args.snapshots,
        args.annotations,
        teacher_source=args.teacher_source,
        game_id=args.game_id,
        agent_id=args.agent_id,
    )
    report.pop("samples_by_group", None)
    print(json.dumps(report, indent=2))
    if not bool(report.get("valid", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
