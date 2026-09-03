from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.human_game_listener import HumanGameListener


def _state(phase: str = "action", step: int = 1) -> dict:
    return {
        "game": {"phase": phase, "generation": 1, "step": step, "spaces": [], "milestones": [], "awards": []},
        "thisPlayer": {
            "id": "player-1",
            "name": "You",
            "color": "blue",
            "actionsTakenThisRound": 0,
            "victoryPointsBreakdown": {"total": 27},
            "tableau": [],
        },
        "players": [
            {"id": "player-1", "name": "You", "color": "blue", "victoryPointsBreakdown": {"total": 27}, "tableau": []},
            {"id": "player-2", "name": "Bot", "color": "red", "victoryPointsBreakdown": {"total": 21}, "tableau": []},
        ],
        "waitingFor": {
            "type": "or",
            "title": "Take an action",
            "options": [{"type": "option", "title": "Pass for this generation", "buttonLabel": "Pass"}],
        },
    }


def test_listener_records_a_human_action_and_finishes_a_whole_game(tmp_path: Path) -> None:
    listener = HumanGameListener(
        str(tmp_path / "dataset"),
        event_dir=str(tmp_path / "events"),
        player_ids=["player-1"],
        completion_grace_sec=60,
    )
    payload = {
        "game_id": "game-human-1",
        "player_id": "player-1",
        "player_name": "You",
        "player_state": _state(),
        "response": {"type": "or", "index": 0, "response": {"type": "option"}},
    }

    result = listener.record(payload)
    assert result["status"] == "recorded"
    assert listener.record(payload)["status"] == "duplicate"
    assert len(list((tmp_path / "events").glob("*.json"))) == 1

    completion = listener.complete_game("game-human-1", _state(phase="end", step=3))
    assert completion["status"] == "queued"
    final_payload = {
        **payload,
        "player_state": _state(step=2),
    }
    finished = listener.record(final_payload)
    assert finished["status"] == "recorded"
    listener._finalize_completed_game("game-human-1")
    rows = [row for split in ("train", "validation", "test") for row in listener.store.iter_samples(split)]
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"human.listener.v1"}
    assert {row["rank"] for row in rows} == {1}
    assert all(row["value_target"] > 0.9 for row in rows)


def test_listener_retains_an_unmapped_raw_action(tmp_path: Path) -> None:
    listener = HumanGameListener(str(tmp_path / "dataset"), event_dir=str(tmp_path / "events"))
    result = listener.record({
        "game_id": "game-human-2",
        "player_id": "player-1",
        "player_state": _state(),
        "response": {"type": "not-a-real-input"},
    })
    assert result["status"] == "unmapped"
    assert listener.stats["unmapped"] == 1
    assert len(list((tmp_path / "events").glob("*.json"))) == 1
