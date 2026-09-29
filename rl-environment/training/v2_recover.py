"""One-shot, recoverable reset of the live PPO learner to a trusted checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from models.agent import RLAgent
from v2_runtime import initialize_v2_runtime


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover(root: str, checkpoint: str, *, force: bool = False) -> Dict[str, Any]:
    paths = initialize_v2_runtime()
    runtime_root = Path(paths["root"]).resolve()
    requested_root = Path(root).expanduser().resolve()
    if requested_root != runtime_root:
        raise RuntimeError(f"--root must exactly match the active runtime root: {runtime_root}")

    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"recovery checkpoint does not exist: {source}")
    latest = runtime_root / "checkpoints" / "latest_learner.pth"
    state_path = runtime_root / "metrics" / "selfplay_state.json"
    progress_path = runtime_root / "metrics" / "selfplay_progress.json"
    if not latest.is_file() or not state_path.is_file():
        raise RuntimeError("recovery requires latest_learner.pth and selfplay_state.json")

    state = _read_json(state_path)
    progress = _read_json(progress_path)
    source_hash = _sha256(source)
    recovery_events = list(state.get("recovery_events", []) or [])
    if recovery_events and recovery_events[-1].get("source_sha256") == source_hash and not force:
        raise RuntimeError("this checkpoint was already applied by the latest recovery; use --force to repeat")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = runtime_root / "recovery-backups" / timestamp
    sequence = 1
    while backup_dir.exists():
        backup_dir = runtime_root / "recovery-backups" / f"{timestamp}-{sequence}"
        sequence += 1
    backup_dir.mkdir(parents=True)
    shutil.copy2(latest, backup_dir / latest.name)
    shutil.copy2(state_path, backup_dir / state_path.name)
    if progress_path.is_file():
        shutil.copy2(progress_path, backup_dir / progress_path.name)

    learner = RLAgent(agent_id="v2-main-learner")
    learner.load_model(str(latest))
    previous_policy_version = int(learner.policy_version)
    quarantine = {"steps": 0, "shards": 0, "path": ""}
    if learner.rollout_shard_store is not None:
        quarantine = learner.rollout_shard_store.quarantine_all(f"recovery_{timestamp}")

    learner.load_model(str(source))
    source_policy_version = int(learner.policy_version)
    learner.policy_version = max(previous_policy_version, source_policy_version) + 1
    decisions = max(
        int(state.get("decisions", 0) or 0),
        int(progress.get("decisions", 0) or 0),
    )
    games = max(
        int(state.get("games", 0) or 0),
        int(progress.get("games", 0) or 0),
        int(getattr(learner, "games_played", 0) or 0),
    )
    seed_cursor = max(
        int(state.get("seed_cursor", 0) or 0),
        int(progress.get("seed_cursor", 0) or 0),
    )
    learner.games_played = games
    temporary_latest = latest.with_name(latest.name + ".recovery.tmp")
    learner.save_model(str(temporary_latest))
    os.replace(temporary_latest, latest)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": source_hash,
        "backup_dir": str(backup_dir),
        "previous_policy_version": previous_policy_version,
        "source_policy_version": source_policy_version,
        "new_policy_version": int(learner.policy_version),
        "decisions": decisions,
        "games": games,
        "quarantined_rollout_steps": int(quarantine.get("steps", 0)),
        "quarantine_path": str(quarantine.get("path", "")),
    }
    recovery_events.append(event)
    state.update(
        {
            "decisions": decisions,
            "games": games,
            "seed_cursor": seed_cursor,
            "policy_version": int(learner.policy_version),
            "consecutive_champion_screen_failures": 0,
            "recovery_events": recovery_events,
        }
    )
    _write_json_atomic(state_path, state)
    progress.update(
        {
            "status": "recovered",
            "decisions": decisions,
            "games": games,
            "seed_cursor": seed_cursor,
            "rollout_buffer": 0,
            "game_seed": None,
            "game_elapsed_sec": None,
            "error": None,
        }
    )
    _write_json_atomic(progress_path, progress)
    _write_json_atomic(backup_dir / "recovery_event.json", event)
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover the v2 learner from a trusted checkpoint")
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.checkpoint, force=args.force), indent=2))


if __name__ == "__main__":
    main()
