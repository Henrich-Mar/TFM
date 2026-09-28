"""Atomic, bounded replay storage for search-distillation episodes.

These shards are deliberately separate from PPO rollout shards. A search
episode is committed only after a completed game supplies a terminal value;
incomplete games are discarded and can never become value targets.
"""
from __future__ import annotations

import copy
import gzip
import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SEARCH_REPLAY_SCHEMA_VERSION = "tfm.search_replay.v1"
_SHARD_RE = re.compile(r"^search_(?P<ts>\d{20})_(?P<seq>\d{6})_(?P<count>\d+)\.pkl\.gz$")


class SearchReplayStore:
    """Collect accepted MCTS decisions and atomically commit complete episodes."""

    def __init__(self, root_dir: str | Path, max_shards: int = 2048) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_shards = max(1, int(max_shards))
        self._lock = threading.Lock()
        self._pending: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._next_seq = self._discover_next_seq()

    def record_decision(
        self,
        game_id: str,
        player_id: str,
        record: Dict[str, Any],
    ) -> None:
        key = (str(game_id), str(player_id))
        if not key[0] or not key[1]:
            raise ValueError("search replay records require game_id and player_id")
        target = list(record.get("policy_target", []) or [])
        if not target or abs(sum(float(item) for item in target) - 1.0) > 1e-4:
            raise ValueError("search replay policy_target must be a normalized non-empty distribution")
        item = copy.deepcopy(record)
        item["schema_version"] = SEARCH_REPLAY_SCHEMA_VERSION
        item["game_id"] = key[0]
        item["player_id"] = key[1]
        with self._lock:
            self._pending.setdefault(key, []).append(item)

    def finish_episode(
        self,
        game_id: str,
        player_id: str,
        *,
        completed: bool,
        value_target: Optional[float] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        key = (str(game_id), str(player_id))
        with self._lock:
            records = self._pending.pop(key, [])
            if not completed or value_target is None or not records:
                return None
            value = float(value_target)
            for index, record in enumerate(records):
                record["step_index"] = int(index)
                record["value_target"] = value
                record["value_target_valid"] = True
            payload = {
                "schema_version": SEARCH_REPLAY_SCHEMA_VERSION,
                "game_id": key[0],
                "player_id": key[1],
                "policy_version": int(records[0].get("policy_version", 0) or 0),
                "completed": True,
                "outcome": copy.deepcopy(outcome or {}),
                "records": records,
            }
            path = self._new_path(len(records))
            temporary = path.with_name(path.name + ".tmp")
            with gzip.open(temporary, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
            self._enforce_window()
            return path

    def discard_episode(self, game_id: str, player_id: str) -> int:
        with self._lock:
            return len(self._pending.pop((str(game_id), str(player_id)), []))

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._pending.values())

    def shard_paths(self) -> List[Path]:
        return sorted(path for path in self.root_dir.glob("search_*.pkl.gz") if _SHARD_RE.match(path.name))

    @staticmethod
    def read_shard(path: str | Path) -> Dict[str, Any]:
        with gzip.open(Path(path), "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_REPLAY_SCHEMA_VERSION:
            raise ValueError(f"incompatible search replay shard: {path}")
        if not payload.get("completed") or not isinstance(payload.get("records"), list):
            raise ValueError(f"incomplete search replay shard: {path}")
        return payload

    def _discover_next_seq(self) -> int:
        highest = -1
        for path in self.shard_paths():
            match = _SHARD_RE.match(path.name)
            if match:
                highest = max(highest, int(match.group("seq")))
        return highest + 1

    def _new_path(self, count: int) -> Path:
        timestamp = time.time_ns()
        sequence = self._next_seq
        self._next_seq += 1
        return self.root_dir / f"search_{timestamp:020d}_{sequence:06d}_{int(count)}.pkl.gz"

    def _enforce_window(self) -> None:
        paths = self.shard_paths()
        for path in paths[: max(0, len(paths) - self.max_shards)]:
            path.unlink(missing_ok=True)
