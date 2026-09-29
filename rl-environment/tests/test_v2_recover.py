from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import v2_recover


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def quarantine_all(self, label: str) -> dict:
        self.calls.append(label)
        return {"steps": 17, "shards": 2, "path": "/quarantine"}


class _FakeAgent:
    store = _FakeStore()

    def __init__(self, agent_id: str) -> None:
        assert agent_id == "v2-main-learner"
        self.policy_version = 0
        self.games_played = 0
        self.rollout_shard_store = self.store

    def load_model(self, path: str) -> None:
        self.policy_version = 9 if Path(path).name == "latest_learner.pth" else 4

    def save_model(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"policy_version": self.policy_version, "games": self.games_played}),
            encoding="utf-8",
        )


def test_recovery_preserves_progress_and_quarantines_rollouts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "alphago"
    checkpoints = root / "checkpoints"
    metrics = root / "metrics"
    rollouts = root / "rollouts"
    for path in (checkpoints, metrics, rollouts):
        path.mkdir(parents=True)
    latest = checkpoints / "latest_learner.pth"
    source = checkpoints / "candidate_000800867.pth"
    latest.write_bytes(b"latest")
    source.write_bytes(b"trusted")
    (metrics / "selfplay_state.json").write_text(
        json.dumps({"decisions": 100, "games": 8, "seed_cursor": 20, "reports": []}),
        encoding="utf-8",
    )
    (metrics / "selfplay_progress.json").write_text(
        json.dumps({"decisions": 125, "games": 10, "seed_cursor": 24, "rollout_buffer": 17}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        v2_recover,
        "initialize_v2_runtime",
        lambda: {"root": str(root), "checkpoints": str(checkpoints), "metrics": str(metrics), "rollouts": str(rollouts)},
    )
    monkeypatch.setattr(v2_recover, "RLAgent", _FakeAgent)
    _FakeAgent.store.calls.clear()

    event = v2_recover.recover(str(root), str(source))

    state = json.loads((metrics / "selfplay_state.json").read_text(encoding="utf-8"))
    progress = json.loads((metrics / "selfplay_progress.json").read_text(encoding="utf-8"))
    assert event["new_policy_version"] == 10
    assert event["quarantined_rollout_steps"] == 17
    assert state["decisions"] == 125
    assert state["games"] == 10
    assert state["seed_cursor"] == 24
    assert state["policy_version"] == 10
    assert progress["status"] == "recovered"
    assert progress["rollout_buffer"] == 0
    assert _FakeAgent.store.calls and _FakeAgent.store.calls[0].startswith("recovery_")
    assert (Path(event["backup_dir"]) / "latest_learner.pth").is_file()

