import os
import sys
from pathlib import Path


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from api.server import _load_orchestrator_snapshot, _parse_named_urls  # noqa: E402


def test_parse_named_urls_normalizes_http_sources() -> None:
    parsed = _parse_named_urls("coord-1=rl-coordinator-1:5000,coord-2=http://rl-coordinator-2:5000")
    assert parsed == [
        ("coord-1", "http://rl-coordinator-1:5000"),
        ("coord-2", "http://rl-coordinator-2:5000"),
    ]


def test_load_orchestrator_snapshot_reads_state_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "orchestrator_state.json"
    manifest_path = tmp_path / "champion" / "current" / "champion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    state_path.write_text(
        """{
  "status": "idle",
  "updated_at": "2026-03-06T10:00:00Z",
  "last_round_id": "r-42",
  "coord_progress": {
    "coord-1": {"generation_signal": 101, "latest_saved_generation": 100, "next_generation": 101}
  }
}""",
        encoding="utf-8",
    )
    manifest_path.write_text(
        """{
  "winner": {
    "candidate_id": "coord-1:g100:agent_0.pth",
    "coordinator_id": "coord-1",
    "generation": 100,
    "metrics": {"win_rate": 0.75, "avg_vp": 91.5}
  },
  "promotion": {"applied": true, "reason": "margin_passed"}
}""",
        encoding="utf-8",
    )
    monkeypatch.setenv("GLOBAL_DASHBOARD_ORCH_OUTPUT_ROOT", str(tmp_path))

    snapshot = _load_orchestrator_snapshot()
    assert snapshot["summary"]["status"] == "idle"
    assert snapshot["summary"]["last_round_id"] == "r-42"
    assert snapshot["summary"]["winner_coordinator_id"] == "coord-1"
    assert snapshot["summary"]["winner_generation"] == 100
    assert snapshot["summary"]["promotion_applied"] is True
    assert snapshot["summary"]["coord_progress"]["coord-1"]["generation_signal"] == 101
