from __future__ import annotations

import gzip
import logging
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_SHARD_NAME_RE = re.compile(r"^rollout_(?P<ts>\d{20})_(?P<seq>\d{6})_(?P<count>\d+)\.pkl\.gz$")


def _safe_agent_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "").strip())
    token = token.strip("._-")
    return token or "default"


class RolloutShardStore:
    """Simple FIFO disk queue for PPO rollout shards."""

    def __init__(self, root_dir: str, agent_id: str, shard_max_steps: int = 2048) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.agent_dir = self.root_dir / f"agent_{_safe_agent_token(agent_id)}"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.shard_max_steps = max(1, int(shard_max_steps))
        self._next_seq = self._discover_next_seq()
        self._queued_steps = self._scan_existing_step_count()

    def queued_step_count(self) -> int:
        return int(self._queued_steps)

    def append_steps(self, steps: Sequence[Any]) -> int:
        items = list(steps or [])
        if not items:
            return 0

        written = 0
        for start in range(0, len(items), self.shard_max_steps):
            chunk = items[start:start + self.shard_max_steps]
            if not chunk:
                continue
            path = self._new_shard_path(len(chunk))
            self._write_payload(path, chunk)
            written += len(chunk)

        self._queued_steps += int(written)
        return int(written)

    def append_episode(self, steps: Sequence[Any]) -> int:
        """Persist one complete episode as one atomic shard."""
        items = list(steps or [])
        if not items:
            return 0
        episode_ids = {str(getattr(step, "episode_id", "") or "") for step in items}
        if len(episode_ids) != 1 or "" in episode_ids:
            raise ValueError("episode shard must contain exactly one non-empty episode_id")
        step_indices = [int(getattr(step, "step_index", -1)) for step in items]
        if step_indices != list(range(len(items))):
            raise ValueError("episode shard step_index values must be contiguous from zero")
        if not bool(getattr(items[-1], "terminal", getattr(items[-1], "done", False))):
            raise ValueError("episode shard must end with a terminal step")
        path = self._new_shard_path(len(items))
        self._write_payload(path, items)
        self._queued_steps += len(items)
        return len(items)

    def pop_complete_episodes(self, target_steps: int) -> List[Any]:
        """Pop whole episode shards until the target is met or the queue is empty."""
        target = max(1, int(target_steps))
        out: List[Any] = []
        for path, meta in self._iter_shards():
            if out and len(out) >= target:
                break
            parsed_count = int(meta[2]) if meta is not None else 0
            try:
                payload = list(self._read_payload(path) or [])
            except Exception as exc:
                logger.warning("Dropping unreadable rollout episode %s: %s", path, exc)
                self._drop_path(path, parsed_count=parsed_count)
                continue
            actual_count = len(payload)
            if actual_count != parsed_count:
                self._queued_steps = max(0, self._queued_steps + actual_count - parsed_count)
            if not payload:
                self._drop_path(path, parsed_count=0)
                continue
            episode_ids = {str(getattr(step, "episode_id", "") or "") for step in payload}
            if len(episode_ids) > 1:
                logger.warning("Dropping mixed-episode rollout shard %s", path)
                self._drop_path(path, parsed_count=actual_count)
                continue
            out.extend(payload)
            self._queued_steps = max(0, self._queued_steps - actual_count)
            self._drop_path(path, parsed_count=0)
        return out

    def pop_steps(self, max_steps: int) -> List[Any]:
        take = max(0, int(max_steps))
        if take <= 0 or self._queued_steps <= 0:
            return []

        out: List[Any] = []
        for path, meta in self._iter_shards():
            if len(out) >= take:
                break

            parsed_count = int(meta[2]) if meta is not None else 0
            try:
                payload = self._read_payload(path)
            except Exception as exc:
                logger.warning("Dropping unreadable rollout shard %s: %s", path, exc)
                self._drop_path(path, parsed_count=parsed_count)
                continue

            shard_steps = list(payload or [])
            actual_count = int(len(shard_steps))
            if actual_count != parsed_count:
                self._queued_steps = max(0, self._queued_steps + (actual_count - parsed_count))

            if actual_count <= 0:
                self._drop_path(path, parsed_count=0)
                continue

            remaining = take - len(out)
            consume = min(remaining, actual_count)
            out.extend(shard_steps[:consume])
            self._queued_steps = max(0, self._queued_steps - consume)

            remainder = shard_steps[consume:]
            if remainder:
                self._rewrite_shard(path, meta=meta, steps=remainder)
            else:
                self._drop_path(path, parsed_count=0)

        return out

    def clear(self) -> int:
        cleared = int(self._queued_steps)
        for path, _meta in self._iter_shards():
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed deleting rollout shard %s", path, exc_info=True)
        self._queued_steps = 0
        return cleared

    def _iter_shards(self) -> List[Tuple[Path, Optional[Tuple[int, int, int]]]]:
        items: List[Tuple[Path, Optional[Tuple[int, int, int]]]] = []
        if not self.agent_dir.exists():
            return items

        for path in self.agent_dir.iterdir():
            if not path.is_file():
                continue
            meta = self._parse_shard_name(path.name)
            if meta is None:
                continue
            items.append((path, meta))
        items.sort(key=lambda item: item[1])
        return items

    def _scan_existing_step_count(self) -> int:
        total = 0
        for _path, meta in self._iter_shards():
            total += int(meta[2]) if meta is not None else 0
        return int(total)

    def _discover_next_seq(self) -> int:
        highest = -1
        for _path, meta in self._iter_shards():
            if meta is None:
                continue
            highest = max(highest, int(meta[1]))
        return int(highest + 1)

    def _new_shard_path(self, step_count: int) -> Path:
        timestamp = int(time.time_ns())
        seq = int(self._next_seq)
        self._next_seq += 1
        return self.agent_dir / f"rollout_{timestamp:020d}_{seq:06d}_{int(step_count)}.pkl.gz"

    def _rewrite_shard(
        self,
        original_path: Path,
        meta: Optional[Tuple[int, int, int]],
        steps: Sequence[Any],
    ) -> None:
        items = list(steps or [])
        if not items:
            self._drop_path(original_path, parsed_count=0)
            return

        if meta is not None:
            ts, seq, _count = meta
            target_path = original_path.with_name(
                f"rollout_{int(ts):020d}_{int(seq):06d}_{len(items)}.pkl.gz"
            )
        else:
            target_path = self._new_shard_path(len(items))

        self._write_payload(target_path, items)
        if target_path != original_path:
            original_path.unlink(missing_ok=True)

    def _drop_path(self, path: Path, parsed_count: int) -> None:
        try:
            path.unlink(missing_ok=True)
        finally:
            if parsed_count > 0:
                self._queued_steps = max(0, self._queued_steps - int(parsed_count))

    def _write_payload(self, path: Path, payload: Sequence[Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="rollout_", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        try:
            with gzip.open(tmp_path, "wb", compresslevel=3) as fh:
                pickle.dump(list(payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _read_payload(self, path: Path) -> List[Any]:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
        return list(payload or [])

    @staticmethod
    def _parse_shard_name(name: str) -> Optional[Tuple[int, int, int]]:
        match = _SHARD_NAME_RE.match(str(name or ""))
        if match is None:
            return None
        return (
            int(match.group("ts")),
            int(match.group("seq")),
            int(match.group("count")),
        )
