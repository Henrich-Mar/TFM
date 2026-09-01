"""Sharded teacher dataset and live-game recorder for TFM RL v2."""
from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "teacher_sample.v1"


def split_for_episode(episode_id: str) -> str:
    bucket = int(hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def source_weight(source: str, confidence: float) -> float:
    if str(source).startswith("human"):
        return 4.0
    return 1.0 if float(confidence) >= 0.5 else 0.25


def validate_sample(sample: Dict[str, Any]) -> None:
    if str(sample.get("schema_version", "")) != SCHEMA_VERSION:
        raise ValueError("invalid teacher sample schema")
    bundle = sample.get("planner_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("teacher sample missing planner_bundle")
    action_count = int(np.asarray(bundle.get("action_tokens", [])).shape[0])
    probabilities = list(sample.get("teacher_probabilities", []) or [])
    if action_count <= 0 or len(probabilities) != action_count:
        raise ValueError("teacher probabilities do not match action-token count")
    if any(float(item) < 0.0 for item in probabilities):
        raise ValueError("teacher probabilities must be non-negative")
    total = sum(float(item) for item in probabilities)
    if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("teacher probabilities must sum to one")


class TeacherDatasetStore:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir).expanduser().resolve()
        for split in ("train", "validation", "test"):
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def append_episode(
        self,
        episode_id: str,
        samples: Sequence[Dict[str, Any]],
        split_key: Optional[str] = None,
    ) -> Path:
        items = [dict(item) for item in (samples or [])]
        if not items:
            raise ValueError("cannot append empty teacher episode")
        resolved_split_key = str(split_key or episode_id)
        split = split_for_episode(resolved_split_key)
        for step_index, item in enumerate(items):
            item["schema_version"] = SCHEMA_VERSION
            item["episode_id"] = str(episode_id)
            item["step_index"] = int(item.get("step_index", step_index))
            item["split"] = split
            item["split_key"] = resolved_split_key
            validate_sample(item)
        target = self.root / split / f"episode_{episode_id}_{len(items)}.pkl.gz"
        fd, tmp_name = tempfile.mkstemp(prefix="teacher_", suffix=".tmp", dir=str(target.parent))
        os.close(fd)
        try:
            with gzip.open(tmp_name, "wb", compresslevel=3) as handle:
                pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target

    def iter_samples(self, split: str) -> Iterable[Dict[str, Any]]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid dataset split: {split}")
        for path in sorted((self.root / split).glob("episode_*.pkl.gz")):
            with gzip.open(path, "rb") as handle:
                for item in list(pickle.load(handle) or []):
                    validate_sample(item)
                    yield item

    def counts(self) -> Dict[str, int]:
        return {split: sum(1 for _ in self.iter_samples(split)) for split in ("train", "validation", "test")}


@dataclass
class _PendingEpisode:
    episode_id: str
    split_key: str
    game_id: str
    seed: int
    samples: List[Dict[str, Any]] = field(default_factory=list)


class TeacherDatasetRecorder:
    """Thread-safe recorder attached to teacher agents during live games."""

    def __init__(self, store: TeacherDatasetStore, seed: int = 0) -> None:
        self.store = store
        self.default_seed = int(seed)
        self._pending: Dict[str, _PendingEpisode] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(game_id: str, agent_id: str) -> str:
        return f"{game_id}:{agent_id}"

    def record_decision(
        self,
        game_id: str,
        agent_id: str,
        planner_bundle: Dict[str, Any],
        action_meta: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        external = dict(action_meta.get("external_policy", {}) or {})
        score_rows = list(external.get("scores", []) or [])
        probabilities = [float(row.get("probability", 0.0)) for row in score_rows]
        if not probabilities:
            return
        confidence = float(external.get("confidence", 0.0) or 0.0)
        source = str(external.get("version", "heuristic-teacher.v1") or "heuristic-teacher.v1")
        key = self._key(game_id, agent_id)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                resolved_seed = self.default_seed if seed is None else int(seed)
                pending = _PendingEpisode(
                    uuid.uuid4().hex,
                    f"seed:{resolved_seed}",
                    str(game_id),
                    resolved_seed,
                )
                self._pending[key] = pending
            pending.samples.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": uuid.uuid4().hex,
                    "planner_bundle": planner_bundle,
                    "action_descriptors": list(action_meta.get("action_descriptors", []) or []),
                    "action_indices": [int(row.get("action_index", -1)) for row in score_rows],
                    "teacher_probabilities": probabilities,
                    "chosen_action_position": int(action_meta.get("chosen_action_position", 0)),
                    "confidence": confidence,
                    "source": source,
                    "sample_weight": source_weight(source, confidence),
                    "seed": int(pending.seed),
                    "game_id": str(pending.game_id),
                    "value_target": 0.0,
                    "rank": 0,
                    "vp": 0.0,
                    "vp_mean": 0.0,
                }
            )

    def finish_episode(self, game_id: str, agent_id: str, outcome: Dict[str, Any]) -> Optional[Path]:
        key = self._key(game_id, agent_id)
        with self._lock:
            pending = self._pending.pop(key, None)
        if pending is None or not pending.samples or not bool(outcome.get("completed", False)):
            return None
        rank = int(outcome.get("rank", 4) or 4)
        vp = float(outcome.get("vp", 0.0) or 0.0)
        vp_mean = float(outcome.get("vp_mean", vp) or vp)
        rank_reward = {1: 1.0, 2: 0.25, 3: -0.25, 4: -1.0}.get(rank, -1.0)
        value_target = rank_reward + max(-0.1, min(0.1, 0.05 * ((vp - vp_mean) / 20.0)))
        for item in pending.samples:
            item.update({"rank": rank, "vp": vp, "vp_mean": vp_mean, "value_target": float(value_target)})
        return self.store.append_episode(pending.episode_id, pending.samples, split_key=pending.split_key)
