"""
Cross-coordinator PPO serialization primitives.

When multiple coordinators share a single GPU (``RL_COORDINATORS_SHARE_GPU=1``),
this module provides file-based synchronization so that:

  1. Only one coordinator runs PPO at a time (exclusive flock).
  2. The champion orchestrator pauses while any coordinator is in PPO.
  3. No coordinator starts new games until *all* coordinators have finished
     PPO for the current round (barrier).

All state lives under ``/app/rl-models-global/locks/`` which is a shared
Docker volume across all coordinator and orchestrator containers.

Gated by env ``PPO_COORDINATOR_SERIALIZE=1`` (default ``0``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None

_DEFAULT_LOCKS_DIR = "/app/rl-models-global/locks"
_COORD_LOCK_FILENAME = "ppo_coordinator.lock"
_PHASE_ACTIVE_FILENAME = "ppo_phase_active"
_ROUND_STATE_FILENAME = "ppo_round_state.json"


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off")


def _safe_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _locks_dir() -> str:
    return str(os.getenv("PPO_COORDINATOR_LOCKS_DIR", _DEFAULT_LOCKS_DIR)).strip() or _DEFAULT_LOCKS_DIR


def is_serialize_enabled() -> bool:
    return _env_flag("PPO_COORDINATOR_SERIALIZE", "0")


# ── 1. Exclusive coordinator PPO lock ────────────────────────────────────

async def acquire_coordinator_ppo_lock(coord_id: str) -> Any:
    """Acquire an exclusive flock so only one coordinator runs PPO at a time.

    Returns the open lock file handle (to be passed to
    ``release_coordinator_ppo_lock``).  Returns ``None`` when serialization
    is disabled or ``fcntl`` is unavailable.
    """
    if not is_serialize_enabled():
        return None
    if fcntl is None:
        logger.warning("PPO_COORDINATOR_SERIALIZE=1 but fcntl unavailable; skipping.")
        return None

    locks = _locks_dir()
    os.makedirs(locks, exist_ok=True)
    lock_path = os.path.join(locks, _COORD_LOCK_FILENAME)
    timeout_sec = max(0.0, _safe_env_float("PPO_COORDINATOR_LOCK_TIMEOUT_SEC", 3600.0))
    poll_sec = max(0.1, _safe_env_float("PPO_COORDINATOR_LOCK_POLL_SEC", 0.5))

    lock_file = open(lock_path, "a+", encoding="utf-8")
    start = asyncio.get_running_loop().time()
    logger.info("Coordinator %s waiting for PPO lock …", coord_id)

    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info("Coordinator %s acquired PPO lock.", coord_id)
            return lock_file
        except BlockingIOError:
            elapsed = asyncio.get_running_loop().time() - start
            if timeout_sec > 0.0 and elapsed >= timeout_sec:
                lock_file.close()
                raise TimeoutError(
                    f"Coordinator {coord_id} timed out waiting for PPO lock "
                    f"after {timeout_sec:.1f}s"
                )
            await asyncio.sleep(poll_sec)


def release_coordinator_ppo_lock(lock_file: Any) -> None:
    if lock_file is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_file.close()
    except Exception:
        pass


# ── 2. PPO phase marker (orchestrator reads this) ────────────────────────

def signal_ppo_phase_active() -> None:
    """Touch a marker file indicating that PPO is in progress globally."""
    if not is_serialize_enabled():
        return
    locks = _locks_dir()
    os.makedirs(locks, exist_ok=True)
    marker = os.path.join(locks, _PHASE_ACTIVE_FILENAME)
    try:
        Path(marker).write_text(str(time.time()), encoding="utf-8")
        logger.info("PPO phase marker set: %s", marker)
    except Exception:
        logger.warning("Failed to write PPO phase marker %s", marker, exc_info=True)


def clear_ppo_phase_active() -> None:
    """Remove the PPO phase marker once all coordinators are done."""
    locks = _locks_dir()
    marker = os.path.join(locks, _PHASE_ACTIVE_FILENAME)
    try:
        if os.path.exists(marker):
            os.remove(marker)
            logger.info("PPO phase marker cleared: %s", marker)
    except Exception:
        logger.warning("Failed to remove PPO phase marker %s", marker, exc_info=True)


def is_ppo_phase_active() -> bool:
    """Return True if the PPO phase marker file exists."""
    locks = _locks_dir()
    return os.path.exists(os.path.join(locks, _PHASE_ACTIVE_FILENAME))


# ── 3. PPO round completion barrier ─────────────────────────────────────

def _round_state_path() -> str:
    return os.path.join(_locks_dir(), _ROUND_STATE_FILENAME)


def _read_round_state() -> dict:
    path = _round_state_path()
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_round_state(state: dict) -> None:
    path = _round_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        Path(tmp).write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.warning("Failed to write PPO round state %s", path, exc_info=True)


def signal_ppo_done(coord_id: str, num_coordinators: int) -> None:
    """Record that *coord_id* has finished PPO.

    When all ``num_coordinators`` have reported, clear the phase marker and
    reset the round state so the next round starts fresh.
    """
    if not is_serialize_enabled():
        return
    state = _read_round_state()
    done_set: list = state.get("done", [])
    if coord_id not in done_set:
        done_set.append(coord_id)
    state["done"] = done_set
    state["num_coordinators"] = int(num_coordinators)
    _write_round_state(state)

    logger.info(
        "Coordinator %s signalled PPO done (%d/%d).",
        coord_id, len(done_set), num_coordinators,
    )

    if len(done_set) >= int(num_coordinators):
        clear_ppo_phase_active()
        _write_round_state({})
        logger.info("All %d coordinators finished PPO. Barrier released.", num_coordinators)


async def wait_ppo_phase_complete(coord_id: str) -> None:
    """Block until the PPO phase marker is gone (all coordinators done).

    Called at the start of ``evaluate_population`` so no coordinator starts
    new games while another is still running PPO.
    """
    if not is_serialize_enabled():
        return
    if not is_ppo_phase_active():
        return

    timeout_sec = max(0.0, _safe_env_float("PPO_BARRIER_TIMEOUT_SEC", 3600.0))
    poll_sec = max(0.2, _safe_env_float("PPO_BARRIER_POLL_SEC", 1.0))
    start = asyncio.get_running_loop().time()
    logger.info("Coordinator %s waiting at PPO barrier …", coord_id)

    while is_ppo_phase_active():
        elapsed = asyncio.get_running_loop().time() - start
        if timeout_sec > 0.0 and elapsed >= timeout_sec:
            logger.warning(
                "Coordinator %s PPO barrier timeout after %.1fs; proceeding.",
                coord_id, timeout_sec,
            )
            return
        await asyncio.sleep(poll_sec)

    logger.info("Coordinator %s PPO barrier cleared.", coord_id)
