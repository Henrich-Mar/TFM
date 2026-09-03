"""Passive recorder for games played by a human through the TFM web UI."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.action_decoder import ActionDecoder
from models.state_encoder import StateEncoder
from training.teacher_dataset import SCHEMA_VERSION, TeacherDatasetStore, source_weight


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_.") or "unknown"


def _normalise_response(value: Any) -> Any:
    """Remove transport-only fields while retaining the exact choice shape."""
    if isinstance(value, dict):
        return {
            str(key): _normalise_response(item)
            for key, item in sorted(value.items())
            if str(key) != "runId"
        }
    if isinstance(value, list):
        return [_normalise_response(item) for item in value]
    return value


def _response_key(value: Any) -> str:
    return json.dumps(_normalise_response(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class HumanGameListener:
    """Convert non-blocking UI move events into whole-game human demonstrations."""

    def __init__(
        self,
        dataset_dir: str,
        event_dir: Optional[str] = None,
        player_ids: Optional[Iterable[str]] = None,
        player_names: Optional[Iterable[str]] = None,
        completion_grace_sec: float = 2.0,
    ) -> None:
        self.store = TeacherDatasetStore(dataset_dir)
        self.event_root = Path(event_dir or (Path(dataset_dir).resolve().parent / "human-listener-events"))
        self.event_root.mkdir(parents=True, exist_ok=True)
        self.player_ids = {str(value).strip() for value in (player_ids or []) if str(value).strip()}
        self.player_names = {str(value).strip() for value in (player_names or []) if str(value).strip()}
        self.encoder = StateEncoder()
        self.decoder = ActionDecoder(planner_config=self.encoder.planner_config)
        self._pending: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._completed_outcomes: Dict[str, Dict[str, Any]] = {}
        self._finalized_games: set[str] = set()
        self._completion_timers: Dict[str, threading.Timer] = {}
        self.completion_grace_sec = max(0.0, float(completion_grace_sec))
        self._seen_events: set[str] = set()
        self._lock = threading.Lock()
        self.stats: Dict[str, int] = {
            "received": 0,
            "raw_saved": 0,
            "recorded": 0,
            "unmapped": 0,
            "ignored": 0,
            "completed_games": 0,
        }

    @staticmethod
    def _phase_index(state: Dict[str, Any]) -> int:
        phase = str((state.get("game", {}) or {}).get("phase", "") or "").strip().lower()
        return {"research": 0, "drafting": 1, "action": 2, "production": 3, "solar": 4}.get(phase, 5)

    @staticmethod
    def _turn_action_count(state: Dict[str, Any]) -> int:
        player = state.get("thisPlayer", {}) or {}
        try:
            return max(0, int(player.get("actionsTakenThisRound", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _outcome(state: Dict[str, Any], player_id: str) -> Dict[str, float | int]:
        players = [row for row in (state.get("players", []) or []) if isinstance(row, dict)]
        current = state.get("thisPlayer", {}) or {}
        current_color = str(current.get("color", "") or "").strip().lower()
        own = next((row for row in players if str(row.get("id", "") or "") == player_id), current)
        if own is current and current_color:
            own = next((row for row in players if str(row.get("color", "") or "").lower() == current_color), current)

        def score(row: Dict[str, Any]) -> float:
            return float(((row.get("victoryPointsBreakdown", {}) or {}).get("total", 0) or 0))

        own_vp = score(own)
        scores = sorted((score(row) for row in players), reverse=True)
        rank = 1 + sum(1 for value in scores if value > own_vp)
        mean = sum(scores) / len(scores) if scores else own_vp
        return {"rank": int(rank), "vp": float(own_vp), "vp_mean": float(mean)}

    def _write_raw_event(self, payload: Dict[str, Any]) -> None:
        state = payload.get("player_state", {}) or {}
        game = state.get("game", {}) or {}
        fingerprint = hashlib.sha256(_response_key(payload).encode("utf-8")).hexdigest()[:12]
        filename = "_".join(
            (
                _safe_name(payload.get("game_id")),
                _safe_name(payload.get("player_id")),
                _safe_name(game.get("step", "0")),
                fingerprint,
            )
        ) + ".json"
        path = self.event_root / filename
        if path.exists():
            return
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(temp, path)
        self.stats["raw_saved"] += 1

    @staticmethod
    def _selected_descriptor(descriptors: List[Dict[str, Any]], response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        expected = _response_key(response)
        return next(
            (
                row for row in descriptors
                if _response_key(row.get("decoded_action")) == expected
            ),
            None,
        )

    def _finish_game(self, game_id: str, player_id: str, state: Dict[str, Any]) -> Optional[Path]:
        key = (game_id, player_id)
        samples = self._pending.pop(key, [])
        if not samples:
            return None
        outcome = self._outcome(state, player_id)
        rank = int(outcome["rank"])
        vp = float(outcome["vp"])
        vp_mean = float(outcome["vp_mean"])
        rank_reward = {1: 1.0, 2: 0.25, 3: -0.25, 4: -1.0}.get(rank, -1.0)
        value_target = rank_reward + max(-0.1, min(0.1, 0.05 * ((vp - vp_mean) / 20.0)))
        for sample in samples:
            sample.update({"rank": rank, "vp": vp, "vp_mean": vp_mean, "value_target": float(value_target)})
        path = self.store.append_episode(
            episode_id=f"human-listener-{_safe_name(game_id)}-{_safe_name(player_id)}",
            samples=samples,
            split_key=f"human-game:{game_id}",
        )
        self.stats["completed_games"] += 1
        return path

    def record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        game_id = str(payload.get("game_id", "") or "").strip()
        player_id = str(payload.get("player_id", "") or "").strip()
        player_name = str(payload.get("player_name", "") or "").strip()
        state = payload.get("player_state", {}) or {}
        response = payload.get("response", {}) or {}
        if not game_id or not player_id or not isinstance(state, dict) or not isinstance(response, dict):
            raise ValueError("human listener event is missing game_id, player_id, player_state, or response")
        if self.player_ids and player_id not in self.player_ids:
            self.stats["ignored"] += 1
            return {"status": "ignored"}
        if self.player_names and player_name not in self.player_names:
            self.stats["ignored"] += 1
            return {"status": "ignored"}

        event_key = hashlib.sha256(
            f"{game_id}:{player_id}:{_response_key(state.get('game', {}).get('step'))}:{_response_key(response)}".encode("utf-8")
        ).hexdigest()
        with self._lock:
            self.stats["received"] += 1
            if event_key in self._seen_events:
                return {"status": "duplicate"}
            self._seen_events.add(event_key)
            self._write_raw_event(payload)
            if game_id in self._finalized_games:
                self.stats["ignored"] += 1
                return {"status": "late_after_completion"}

            descriptors = self.decoder.get_legal_action_descriptors(state)
            selected = self._selected_descriptor(descriptors, response)
            if selected is None:
                self.stats["unmapped"] += 1
                return {"status": "unmapped", "legal_actions": len(descriptors)}

            selected_position = next(
                index for index, row in enumerate(descriptors)
                if int(row.get("action_index", -1)) == int(selected.get("action_index", -2))
            )
            bundle = self.encoder.encode(state, self._turn_action_count(state), descriptors)
            probabilities = [0.0] * len(descriptors)
            probabilities[selected_position] = 1.0
            sample = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"human-listener-{event_key}",
                "planner_bundle": bundle,
                "action_descriptors": descriptors,
                "action_indices": [int(row.get("action_index", -1)) for row in descriptors],
                "teacher_probabilities": probabilities,
                "chosen_action_position": int(selected_position),
                "phase_index": self._phase_index(state),
                "confidence": 1.0,
                "source": "human.listener.v1",
                "sample_weight": source_weight("human.listener.v1", 1.0),
                "seed": -1,
                "game_id": game_id,
                "value_target": 0.0,
                "rank": 0,
                "vp": 0.0,
                "vp_mean": 0.0,
                "captured_at": str(payload.get("captured_at", "") or _utc_now_iso()),
                "player_name": str(payload.get("player_name", "") or ""),
            }
            self._pending.setdefault((game_id, player_id), []).append(sample)
            self.stats["recorded"] += 1
            result = {
                "status": "recorded",
                "action_index": int(selected.get("action_index", -1)),
                "label": str(selected.get("label", "") or ""),
                "legal_actions": len(descriptors),
            }
            completion_state = payload.get("completion_state")
            if isinstance(completion_state, dict):
                path = self._finish_game(game_id, player_id, completion_state)
                if path is not None:
                    result.update({"game_completed": True, "path": str(path)})
            return result

    def complete(self, game_id: str, player_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            path = self._finish_game(str(game_id), str(player_id), state)
            return {"status": "completed" if path else "empty", "path": str(path) if path else ""}

    def complete_game(self, game_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attach a terminal result to every captured human seat in one game.

        The terminal event can arrive ahead of an in-flight earlier move, so the
        state is retained and applied when that move is eventually recorded.
        """
        game_id = str(game_id or "").strip()
        if not game_id or not isinstance(state, dict):
            raise ValueError("completion event is missing game_id or player_state")
        with self._lock:
            self._completed_outcomes[game_id] = state
            if game_id not in self._completion_timers and game_id not in self._finalized_games:
                timer = threading.Timer(self.completion_grace_sec, self._finalize_completed_game, args=(game_id,))
                timer.daemon = True
                self._completion_timers[game_id] = timer
                timer.start()
            return {"status": "queued", "episodes": 0}

    def _finalize_completed_game(self, game_id: str) -> None:
        with self._lock:
            timer = self._completion_timers.pop(game_id, None)
            if timer is not None:
                timer.cancel()
            state = self._completed_outcomes.get(game_id)
            if state is None or game_id in self._finalized_games:
                return
            for pending_game_id, player_id in list(self._pending):
                if pending_game_id == game_id:
                    self._finish_game(pending_game_id, player_id, state)
            self._finalized_games.add(game_id)
