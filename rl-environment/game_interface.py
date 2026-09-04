"""
Game Interface - Handles communication with Terraforming Mars game servers
"""
import asyncio
import aiohttp
import json
import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlsplit, urlunsplit

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False


def _json_loads(text: str) -> Any:
    """Parse JSON; use orjson when available for faster parsing."""
    if _USE_ORJSON:
        return orjson.loads(text)
    return json.loads(text)


def _json_loads_bytes(data: bytes) -> Any:
    """Parse JSON from bytes; use orjson when available (avoids extra decode step)."""
    if _USE_ORJSON:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def _json_dumps(obj: Any) -> str:
    """Serialize to JSON string; use orjson when available."""
    if _USE_ORJSON:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _json_dumps_bytes(obj: Any) -> bytes:
    """Serialize to JSON bytes (for HTTP body)."""
    if _USE_ORJSON:
        return orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
logger = logging.getLogger(__name__)

class ServerTransportError(RuntimeError):
    """Raised when transport-level communication with a TM server fails."""


def _exc_summary(exc: Exception) -> str:
    """Return stable exception detail even when str(exc) is empty."""
    try:
        return f"{type(exc).__name__}: {exc!r}"
    except Exception:
        return f"{type(exc).__name__}: <unprintable>"


@dataclass
class GameServer:
    host: str
    port: int
    active_games: int = 0
    healthy: bool = True
    last_health_check: Optional[datetime] = None

class GameInstance:
    _debug_initial_cards_pause_done: bool = False
    _debug_initial_cards_pause_lock: Optional[asyncio.Lock] = None

    def __init__(
        self,
        game_id: str,
        server: GameServer,
        session: aiohttp.ClientSession,
        cluster: Optional['GameServerCluster'] = None,
    ):
        self.game_id = game_id
        self.server = server
        self.session = session
        self.cluster = cluster
        self.player_ids: Dict[str, str] = {}  # player_name -> player_id
        self.base_url = f"http://{server.host}:{server.port}"
        self.spectator_id: Optional[str] = None
        self._latest_run_id_by_player: Dict[str, str] = {}
        # Cache: send_player_input stores the response body here so the next
        # get_player_state call can return it without a network round-trip.
        self._cached_player_state: Dict[str, Dict[str, Any]] = {}  # player_id -> state

    def peek_cached_state(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Return the cached post-action state without consuming it.

        Unlike ``get_player_state`` (which pops the cache entry), this leaves
        the cache intact so the next ``get_player_state`` call still benefits
        from the zero-network-trip fast path.
        """
        return self._cached_player_state.get(str(player_id))

    def invalidate_cached_player_state(self, player_id: str) -> None:
        """Discard a post-input response when the next prompt must be authoritative."""
        self._cached_player_state.pop(str(player_id), None)

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = str(os.getenv(name, "1" if default else "0")).strip().lower()
        return value in ("1", "true", "yes", "on")

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    @staticmethod
    def _extract_run_id_from_state(player_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(player_state, dict):
            return ""
        for key in ("runId", "runID", "run_id"):
            value = player_state.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        for scope_key in ("game", "thisPlayer", "waitingFor"):
            scoped = player_state.get(scope_key)
            if not isinstance(scoped, dict):
                continue
            for key in ("runId", "runID", "run_id"):
                value = scoped.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    def _with_run_id(
        self,
        player_id: str,
        input_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = dict(input_data or {})
        if not self._env_flag("TM_SEND_INPUT_INCLUDE_RUN_ID", default=True):
            return payload
        existing = payload.get("runId")
        if existing is not None and str(existing).strip():
            return payload
        cached = self._latest_run_id_by_player.get(str(player_id), "")
        if cached:
            payload["runId"] = cached
        return payload

    @staticmethod
    def _extract_waiting_type(player_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(player_state, dict):
            return ""
        waiting_for = player_state.get("waitingFor", {})
        if not isinstance(waiting_for, dict):
            return ""
        return str(waiting_for.get("type", "") or "")

    async def _refresh_or_skip_initial_cards_send(
        self,
        player_id: str,
        payload: Dict[str, Any],
    ) -> tuple[bool, Dict[str, Any]]:
        """Optionally verify initial-cards prompt is still active before posting.

        Returns (should_skip_send, possibly_updated_payload).
        """
        if not self._env_flag("TM_VALIDATE_INITIAL_CARDS_BEFORE_SEND", default=False):
            return False, payload
        try:
            player_state = await self.get_player_state(player_id)
        except Exception as e:
            logger.warning("Initial cards pre-send validation failed to read player state for %s: %s", player_id, e)
            return False, payload

        waiting_type = self._extract_waiting_type(player_state)
        if waiting_type not in ("initialCards", "selectInitialCards"):
            if self._env_flag("TM_SKIP_STALE_INITIAL_CARDS_SEND", default=True):
                logger.warning(
                    "Skipping stale initialCards send for player %s. Current waitingFor.type=%s, GameURL=%s",
                    player_id,
                    waiting_type or "<none>",
                    self.get_public_game_url(),
                )
                return True, payload
            return False, payload

        refreshed = dict(payload or {})
        current_run_id = self._extract_run_id_from_state(player_state)
        if current_run_id and str(refreshed.get("runId", "") or "") != current_run_id:
            refreshed["runId"] = current_run_id
            logger.warning(
                "Updated initialCards runId from live player state before send (player=%s, runId=%s).",
                player_id,
                current_run_id,
            )
        return False, refreshed

    async def _maybe_pause_before_first_initial_cards(
        self,
        player_id: str,
        input_type: str,
        payload_full: str,
    ) -> None:
        if input_type not in ("initialCards", "selectInitialCards"):
            return
        if not self._env_flag("TM_DEBUG_PAUSE_ON_FIRST_INITIAL_CARDS", default=False):
            return
        arm_file_raw = str(os.getenv("TM_DEBUG_PAUSE_ARM_FILE", "") or "").strip()
        if arm_file_raw:
            arm_path = Path(arm_file_raw)
            if not arm_path.exists():
                return
            if self._env_flag("TM_DEBUG_PAUSE_CONSUME_ARM_FILE", default=True):
                try:
                    arm_path.unlink()
                except Exception as e:
                    logger.warning("Failed to consume debug pause arm file %s: %s", arm_path.as_posix(), e)

        if GameInstance._debug_initial_cards_pause_lock is None:
            GameInstance._debug_initial_cards_pause_lock = asyncio.Lock()

        async with GameInstance._debug_initial_cards_pause_lock:
            if GameInstance._debug_initial_cards_pause_done:
                return

            release_path = Path(
                str(os.getenv("TM_DEBUG_PAUSE_RELEASE_FILE", "/app/logs/release-initial-cards.pause")).strip()
            )
            poll_sec = max(0.05, self._env_float("TM_DEBUG_PAUSE_POLL_SEC", 0.5))
            timeout_sec = max(0.0, self._env_float("TM_DEBUG_PAUSE_TIMEOUT_SEC", 0.0))
            clear_release_file = self._env_flag("TM_DEBUG_PAUSE_CLEAR_RELEASE_FILE", default=True)

            if clear_release_file and release_path.exists():
                try:
                    release_path.unlink()
                except Exception as e:
                    logger.warning("Failed to clear stale release file %s: %s", release_path.as_posix(), e)

            logger.warning(
                "Debug pause armed before first initialCards send. "
                "Player=%s, GameURL=%s, PlayerAPI(public)=%s, PlayerAPI(internal)=%s, ReleaseFile=%s, TimeoutSec=%.2f",
                player_id,
                self.get_public_game_url(),
                self.get_public_player_api_url(player_id),
                self.get_internal_player_api_url(player_id),
                release_path.as_posix(),
                timeout_sec,
            )

            loop = asyncio.get_running_loop()
            started = loop.time()
            while True:
                if release_path.exists():
                    logger.warning("Debug pause released via %s", release_path.as_posix())
                    if clear_release_file:
                        try:
                            release_path.unlink()
                        except Exception:
                            pass
                    break
                if timeout_sec > 0.0 and (loop.time() - started) >= timeout_sec:
                    logger.warning(
                        "Debug pause timeout reached after %.2fs. Continuing first initialCards send.",
                        timeout_sec,
                    )
                    break
                await asyncio.sleep(poll_sec)

            GameInstance._debug_initial_cards_pause_done = True
    
    def _get_session(self) -> aiohttp.ClientSession:
        """Return an open HTTP session, recreating cluster session if needed."""
        if self.cluster is not None and hasattr(self.cluster, "ensure_session"):
            self.session = self.cluster.ensure_session(timeout_total=None)
        if self.session is None or self.session.closed:
            raise ServerTransportError("HTTP session is closed")
        return self.session

    @staticmethod
    def _cancellation_requested() -> bool:
        """True when current task is being intentionally cancelled."""
        try:
            task = asyncio.current_task()
            if task is None:
                return False
            cancelling = getattr(task, "cancelling", None)
            if callable(cancelling):
                return bool(cancelling())
            return bool(task.cancelled())
        except Exception:
            return False

    @staticmethod
    def _is_transport_error(exc: Exception) -> bool:
        if isinstance(exc, ServerTransportError):
            return True
        if isinstance(
            exc,
            (
                aiohttp.ClientConnectionError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
                ConnectionResetError,
                BrokenPipeError,
            ),
        ):
            return True
        message = str(exc or "").lower()
        return "session is closed" in message

    @staticmethod
    def _is_local_client_session_error(exc: Exception) -> bool:
        message = str(exc or "").lower()
        return (
            "session is closed" in message
            or "connector is closed" in message
            or "can not write request body" in message
        )

    def _raise_transport_error(self, operation: str, exc: Exception):
        if self.cluster is not None and hasattr(self.cluster, "record_runtime_server_failure"):
            try:
                if not self._is_local_client_session_error(exc):
                    self.cluster.record_runtime_server_failure(self.server, exc)
                else:
                    logger.warning(
                        "Skipping server backoff for %s on %s:%s due to local client-session error: %s",
                        operation,
                        self.server.host,
                        self.server.port,
                        exc,
                    )
            except Exception:
                pass
        raise ServerTransportError(
            f"{operation} failed on {self.server.host}:{self.server.port}: {exc}"
        ) from exc

    def _resolve_public_base(self) -> str:
        """Resolve external URL for this game server."""
        mapping_str = os.getenv('PUBLIC_TM_MAP', '')
        public_map: Dict[str, str] = {}
        if mapping_str:
            try:
                for pair in mapping_str.split(','):
                    if not pair or '=' not in pair:
                        continue
                    k, v = pair.split('=', 1)
                    public_map[k.strip()] = v.strip()
            except Exception:
                logger.warning("Failed to parse PUBLIC_TM_MAP; falling back to PUBLIC_TM_URL")
        server_key = f"{self.server.host}:{self.server.port}"
        public_base = public_map.get(server_key)
        if not public_base:
            pub = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
            public_base = pub.split(',')[0] if ',' in pub else pub
        return self._normalize_public_base(public_base)

    @staticmethod
    def _normalize_public_base(value: Any) -> str:
        """Return normalized public base URL: scheme://host[:port] (no path/query/fragment)."""
        fallback = "http://localhost:8081"
        raw = str(value or "").strip()
        if not raw:
            raw = fallback
        # Handle accidental comma-separated env values.
        if "," in raw:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            raw = parts[0] if parts else fallback
        if "://" not in raw:
            raw = f"http://{raw}"
        try:
            parsed = urlsplit(raw)
            scheme = parsed.scheme if parsed.scheme in ("http", "https") else "http"
            netloc = parsed.netloc or parsed.path
            if not netloc:
                return fallback
            return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")
        except Exception:
            return fallback

    def get_public_game_url(self) -> str:
        return f"{self._resolve_public_base()}/game?id={self.game_id}"

    def get_public_player_api_url(self, player_id: str) -> str:
        return f"{self._resolve_public_base()}/api/player?id={player_id}"

    def get_public_player_url(self, player_id: str) -> str:
        return f"{self._resolve_public_base()}/player?id={player_id}"

    def get_internal_player_api_url(self, player_id: str) -> str:
        return f"{self.base_url}/api/player?id={player_id}"
        
    async def join_player(self, player_name: str) -> str:
        """Join a player to the game and return player ID"""
        if player_name in self.player_ids:
            return self.player_ids[player_name]
        
        # Get game state to find player ID
        try:
            session = self._get_session()
            async with session.get(f"{self.base_url}/api/game", 
                                      params={'id': self.game_id}) as response:
                if response.status == 200:
                    game_data = _json_loads_bytes(await response.read())

                    # Find player by name
                    for player in game_data.get('players', []):
                        if player.get('name') == player_name:
                            player_id = player.get('id')
                            self.player_ids[player_name] = player_id
                            return player_id
                    
                    raise ValueError(f"Player {player_name} not found in game")
                else:
                    raise ValueError(f"Failed to get game state: {response.status}")
        except asyncio.CancelledError as e:
            if self._cancellation_requested():
                raise
            self._raise_transport_error("join player", RuntimeError(f"request cancelled: {e}"))
        except Exception as e:
            if self._is_transport_error(e):
                self._raise_transport_error("join player", e)
            logger.error(f"Failed to join player {player_name}: {e}")
            raise
    
    async def get_player_state(self, player_id: str) -> Dict[str, Any]:
        """Get current state for a specific player with retry on transient errors.

        If ``send_player_input`` previously cached a response for this player,
        the cached value is returned immediately (one-shot: the cache entry is
        consumed).  This eliminates one full HTTP round-trip per successful
        action, cutting network I/O roughly in half during gameplay.
        """
        # Fast path: use cached state from the last send_player_input response.
        cached = self._cached_player_state.pop(str(player_id), None)
        if cached is not None:
            return cached

        try:
            max_retries = max(1, int(os.getenv("TM_GET_STATE_RETRY_ATTEMPTS", "3")))
        except Exception:
            max_retries = 3

        last_exc: Optional[Exception] = None
        debug_timing = os.getenv("TM_TRANSPORT_TIMING_DEBUG", "0") == "1"
        for attempt_no in range(1, max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                session = self._get_session()
                if debug_timing:
                    t0 = time.perf_counter()
                async with session.get(f"{self.base_url}/api/player",
                                          params={'id': player_id}) as response:
                    if debug_timing:
                        t1 = time.perf_counter()
                    if response.status == 200:
                        body = await response.read()
                        if debug_timing:
                            t2 = time.perf_counter()
                        player_state = _json_loads_bytes(body)
                        if debug_timing:
                            t3 = time.perf_counter()
                            logger.warning(
                                "TM transport timing get_player_state: connect_ttfb_ms=%.2f body_read_ms=%.2f parse_ms=%.2f total_ms=%.2f host=%s port=%s",
                                (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t3 - t0) * 1000,
                                self.server.host, self.server.port,
                            )
                        run_id = self._extract_run_id_from_state(player_state)
                        if run_id:
                            self._latest_run_id_by_player[str(player_id)] = run_id
                        return player_state
                    else:
                        error = ValueError(f"Failed to get player state: {response.status}")
                        if int(response.status) >= 500:
                            last_exc = error
                            if attempt_no < max_retries:
                                backoff = min(3.0, 0.30 * float(attempt_no))
                                logger.warning(
                                    "get_player_state attempt %d/%d got HTTP %d from %s:%s. "
                                    "Backing off %.2fs.",
                                    attempt_no, max_retries, response.status,
                                    self.server.host, self.server.port, backoff,
                                )
                                await asyncio.sleep(backoff)
                                continue
                            self._raise_transport_error("get player state", error)
                        raise error
            except asyncio.CancelledError as e:
                if self._cancellation_requested():
                    raise
                self._raise_transport_error("get player state", RuntimeError(f"request cancelled: {e}"))
            except Exception as e:
                if self._is_transport_error(e):
                    last_exc = e
                    if attempt_no < max_retries:
                        backoff = min(3.0, 0.30 * float(attempt_no))
                        logger.warning(
                            "get_player_state attempt %d/%d transport error on %s:%s. "
                            "Backing off %.2fs: %s",
                            attempt_no, max_retries,
                            self.server.host, self.server.port, backoff, _exc_summary(e),
                        )
                        await asyncio.sleep(backoff)
                        continue
                    self._raise_transport_error("get player state", e)
                logger.error(f"Failed to get player state for {player_id}: {e!r}")
                raise
            finally:
                try:
                    if self.cluster is not None and hasattr(self.cluster, "record_transport_timing"):
                        self.cluster.record_transport_timing(
                            "get_player_state_sec",
                            time.perf_counter() - attempt_started,
                        )
                except Exception:
                    pass

        if last_exc is not None:
            self._raise_transport_error("get player state", last_exc)
        raise RuntimeError("get_player_state: unexpected fall-through")
    
    async def send_player_input(self, player_id: str, input_data: Dict[str, Any]) -> bool:
        """Send player input to the game.

        On success the TM server response (full PlayerViewModel JSON) is cached
        in ``_cached_player_state[player_id]`` so the next ``get_player_state``
        call returns instantly without a network round-trip.
        """
        prepared_input_data = self._with_run_id(player_id, input_data)
        try:
            payload_full = _json_dumps(prepared_input_data)
        except Exception:
            payload_full = str(prepared_input_data)
        payload_preview = payload_full
        if len(payload_preview) > 800:
            payload_preview = payload_preview[:800] + "...(truncated)"
        input_type = str((prepared_input_data or {}).get('type', 'unknown'))
        is_initial_cards_input = input_type in ('initialCards', 'selectInitialCards')
        try:
            retry_attempts = max(1, int(os.getenv("TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS", "2")))
        except Exception:
            retry_attempts = 2
        if is_initial_cards_input:
            try:
                initial_retry_attempts = max(
                    1,
                    int(os.getenv("TM_SEND_INPUT_TRANSPORT_RETRY_ATTEMPTS_INITIAL", "6")),
                )
            except Exception:
                initial_retry_attempts = 6
            retry_attempts = max(retry_attempts, initial_retry_attempts)
            try:
                initial_jitter_ms = max(
                    0,
                    int(os.getenv("TM_SEND_INPUT_INITIAL_CARDS_JITTER_MS", "250")),
                )
            except Exception:
                initial_jitter_ms = 250
            if initial_jitter_ms > 0:
                await asyncio.sleep(random.uniform(0.0, float(initial_jitter_ms) / 1000.0))
            await self._maybe_pause_before_first_initial_cards(
                player_id=player_id,
                input_type=input_type,
                payload_full=payload_full,
            )
            should_skip, prepared_input_data = await self._refresh_or_skip_initial_cards_send(
                player_id=player_id,
                payload=prepared_input_data,
            )
            if should_skip:
                return True
            try:
                payload_full = _json_dumps(prepared_input_data)
            except Exception:
                payload_full = str(prepared_input_data)
            payload_preview = payload_full
            if len(payload_preview) > 800:
                payload_preview = payload_preview[:800] + "...(truncated)"

        send_input_debug_timing = os.getenv("TM_TRANSPORT_TIMING_DEBUG", "0") == "1"
        for attempt in range(retry_attempts):
            attempt_no = attempt + 1
            attempt_started = time.perf_counter()
            try:
                session = self._get_session()
                if send_input_debug_timing:
                    t0 = time.perf_counter()
                async with session.post(
                    f"{self.base_url}/player/input",
                    params={'id': player_id},
                    data=_json_dumps_bytes(prepared_input_data),
                    headers={'Content-Type': 'application/json'},
                ) as response:
                    if send_input_debug_timing:
                        t1 = time.perf_counter()
                    if response.status == 200:
                        # The TM server returns the full PlayerViewModel JSON
                        # on every successful input.  Cache it so the next
                        # get_player_state() call returns instantly without a
                        # network round-trip.
                        try:
                            body_bytes = await response.read()
                            if send_input_debug_timing:
                                t2 = time.perf_counter()
                            body = _json_loads_bytes(body_bytes)
                            if send_input_debug_timing:
                                t3 = time.perf_counter()
                                logger.warning(
                                    "TM transport timing send_player_input: connect_ttfb_ms=%.2f body_read_ms=%.2f parse_ms=%.2f total_ms=%.2f host=%s port=%s",
                                    (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t3 - t0) * 1000,
                                    self.server.host, self.server.port,
                                )
                            if isinstance(body, dict):
                                run_id = self._extract_run_id_from_state(body)
                                if run_id:
                                    self._latest_run_id_by_player[str(player_id)] = run_id
                                self._cached_player_state[str(player_id)] = body
                        except Exception:
                            pass
                        return True
                    else:
                        response_text = (await response.read()).decode("utf-8")
                        logger.error(
                            "Failed to send input for player %s. Status: %s, Response: %s, "
                            "Input: %s, GameURL: %s, PlayerAPI(public): %s, PlayerAPI(internal): %s",
                            player_id,
                            response.status,
                            response_text,
                            payload_preview,
                            self.get_public_game_url(),
                            self.get_public_player_api_url(player_id),
                            self.get_internal_player_api_url(player_id),
                        )
                        logger.error(
                            "Full input payload for failed player input (%s): %s",
                            input_type,
                            payload_full,
                        )
                        if int(response.status) >= 500:
                            self._raise_transport_error(
                                "send player input",
                                RuntimeError(f"Server returned {response.status}"),
                            )
                        try:
                            if self.cluster is not None and hasattr(self.cluster, "record_input_reject"):
                                self.cluster.record_input_reject(response_text)
                        except Exception:
                            pass
                        return False
            except asyncio.CancelledError as e:
                if self._cancellation_requested():
                    raise
                if attempt_no < retry_attempts:
                    if is_initial_cards_input:
                        backoff = min(4.0, 0.50 * float(attempt_no))
                    else:
                        backoff = min(2.0, 0.10 * float(attempt_no))
                    logger.warning(
                        "Retrying send_player_input after unexpected cancellation on %s:%s "
                        "(attempt %d/%d, input_type=%s): %s",
                        self.server.host,
                        self.server.port,
                        attempt_no,
                        retry_attempts,
                        input_type,
                        _exc_summary(e),
                    )
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        raise
                    continue
                wrapped = RuntimeError(f"request cancelled: {e} [input_type={input_type}, payload={payload_full}]")
                self._raise_transport_error("send player input", wrapped)
            except Exception as e:
                if self._is_transport_error(e):
                    is_retryable = attempt_no < retry_attempts
                    msg_l = str(e or "").lower()
                    is_disconnect = (
                        isinstance(e, (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError,
                                       ConnectionResetError, BrokenPipeError))
                        or "can not write request body" in msg_l
                        or "server disconnected" in msg_l
                    )
                    if is_disconnect and is_retryable:
                        if is_initial_cards_input:
                            backoff = min(4.0, 0.50 * float(attempt_no))
                        else:
                            backoff = min(2.0, 0.10 * float(attempt_no))
                        
                        logger.warning(
                            "Attempt %d/%d failed with transport error for player %s (input_type=%s) on %s:%s. "
                            "Backing off for %.2fs: %s",
                            attempt_no,
                            retry_attempts,
                            player_id,
                            input_type,
                            self.server.host,
                            self.server.port,
                            backoff,
                            _exc_summary(e),
                        )
                        if (
                            self._env_flag("TM_RECYCLE_SESSION_ON_DISCONNECT", default=False)
                            and self.cluster is not None
                            and hasattr(self.cluster, "recycle_session")
                        ):
                            try:
                                await self.cluster.recycle_session()
                            except Exception:
                                pass
                        await asyncio.sleep(backoff)
                        continue
                    logger.error(
                        "Transport failure sending player input (input_type=%s, player_id=%s): payload=%s",
                        input_type,
                        player_id,
                        payload_full,
                    )
                    wrapped = RuntimeError(
                        f"{e} [input_type={input_type}, payload={payload_full}]"
                    )
                    self._raise_transport_error("send player input", wrapped)
                logger.error(f"Failed to send input for player {player_id}: {e!r}")
                return False
            finally:
                try:
                    if self.cluster is not None and hasattr(self.cluster, "record_transport_timing"):
                        self.cluster.record_transport_timing(
                            "send_player_input_sec",
                            time.perf_counter() - attempt_started,
                        )
                except Exception:
                    pass

        return False
    
    async def get_final_state(self) -> Dict[str, Any]:
        """Get final game state after completion"""
        try:
            session = self._get_session()
            async with session.get(f"{self.base_url}/api/game", 
                                      params={'id': self.game_id}) as response:
                if response.status == 200:
                    return _json_loads_bytes(await response.read())
                else:
                    error = ValueError(f"Failed to get final state: {response.status}")
                    if int(response.status) >= 500:
                        self._raise_transport_error("get final state", error)
                    raise error
        except asyncio.CancelledError as e:
            if self._cancellation_requested():
                raise
            self._raise_transport_error("get final state", RuntimeError(f"request cancelled: {e}"))
        except Exception as e:
            if self._is_transport_error(e):
                self._raise_transport_error("get final state", e)
            logger.error(f"Failed to get final state: {e!r}")
            raise
    
    async def cleanup(self):
        """Clean up game resources"""
        # Keep server slot counters consistent across concurrent game cleanup paths.
        if self.cluster is not None:
            await self.cluster.release_server_slot(self.server)
            return
        self.server.active_games = max(0, self.server.active_games - 1)

class GameServerCluster:
    def __init__(self, server_addresses: List[str]):
        self.servers = []
        for address in server_addresses:
            if ':' in address:
                host, port = address.split(':')
                port = int(port)
            else:
                host = address
                port = 8080
            
            self.servers.append(GameServer(host=host, port=port))
        
        self.session = None
        self._server_slot_lock = asyncio.Lock()
        self.max_active_games_per_server = self._parse_int_env(
            "MAX_ACTIVE_GAMES_PER_SERVER",
            0,
            min_value=0,
        )
        self.server_slot_wait_timeout_sec = self._parse_float_env(
            "TM_SERVER_SLOT_WAIT_TIMEOUT_SEC",
            30.0,
            min_value=0.0,
        )
        self.server_slot_retry_sleep_sec = self._parse_float_env(
            "TM_SERVER_SLOT_RETRY_SLEEP_SEC",
            0.05,
            min_value=0.01,
        )
        self.create_game_retry_attempts = self._parse_int_env(
            "TM_CREATE_GAME_RETRY_ATTEMPTS",
            3,
            min_value=1,
        )
        self.create_game_retry_backoff_sec = self._parse_float_env(
            "TM_CREATE_GAME_RETRY_BACKOFF_SEC",
            0.20,
            min_value=0.0,
        )
        self.server_failure_cooldown_sec = self._parse_float_env(
            "TM_SERVER_FAILURE_COOLDOWN_SEC",
            10.0,
            min_value=0.0,
        )
        self.health_check_interval_sec = self._parse_float_env(
            "TM_HEALTH_CHECK_INTERVAL_SEC",
            15.0,
            min_value=0.0,
        )
        self.server_health_timeout_sec = self._parse_float_env(
            "TM_SERVER_HEALTH_TIMEOUT_SEC",
            5.0,
            min_value=0.5,
        )
        self.rl_recycle_enabled = GameInstance._env_flag("V2_RECYCLE_IDLE_SERVERS", default=False)
        self.rl_control_token = str(os.getenv("RL_CONTROL_TOKEN", "")).strip()
        self.rl_recycle_request_timeout_sec = self._parse_float_env(
            "V2_IDLE_SERVER_RECYCLE_REQUEST_TIMEOUT_SEC",
            5.0,
            min_value=0.5,
        )
        self.rl_recycle_ready_timeout_sec = self._parse_float_env(
            "V2_IDLE_SERVER_RECYCLE_READY_TIMEOUT_SEC",
            45.0,
            min_value=5.0,
        )
        self._server_backoff_until: Dict[str, float] = {}
        self._last_health_check_monotonic: float = 0.0
        self._health_check_lock = asyncio.Lock()
        self._recycle_lock = asyncio.Lock()
        self._last_recycle_monotonic: float = 0.0
        # Sessions retired during recycle stay alive briefly so in-flight
        # requests can finish before connectors are closed.
        self._retired_sessions: List[Tuple[aiohttp.ClientSession, float]] = []
        # Optional cross-component scratchpad for latest game URLs
        self.recent_games: List[Dict[str, str]] = []
        self.base_game_options = self._load_game_options()
        self.input_reject_count: int = 0
        self.payment_reject_count: int = 0
        self.transport_timing_totals_sec: Dict[str, float] = {
            "get_player_state_sec": 0.0,
            "send_player_input_sec": 0.0,
        }
        self.transport_timing_counts: Dict[str, int] = {
            "get_player_state_sec": 0,
            "send_player_input_sec": 0,
        }

    @staticmethod
    def _cancellation_requested() -> bool:
        try:
            task = asyncio.current_task()
            if task is None:
                return False
            cancelling = getattr(task, "cancelling", None)
            if callable(cancelling):
                return bool(cancelling())
            return bool(task.cancelled())
        except Exception:
            return False

    @staticmethod
    def _parse_int_env(name: str, default: int, min_value: int = 0) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = int(default)
        return max(int(min_value), value)

    @staticmethod
    def _parse_float_env(name: str, default: float, min_value: float = 0.0) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except Exception:
            value = float(default)
        return max(float(min_value), value)

    @staticmethod
    def _parse_mapping_env(raw_value: Any) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        raw = str(raw_value or "").strip()
        if not raw:
            return mapping
        for pair in raw.split(","):
            piece = pair.strip()
            if not piece or "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            mapping[key] = value
        return mapping

    @staticmethod
    def _server_key(server: GameServer) -> str:
        return f"{server.host}:{server.port}"

    def _mark_server_failure(self, server: GameServer, reason: Any):
        """Temporarily back off a server after transient transport failures."""
        cooldown = max(0.0, float(self.server_failure_cooldown_sec))
        key = self._server_key(server)
        existing_backoff = float(self._server_backoff_until.get(key, 0.0))
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            now = 0.0
        # Avoid log spam when many in-flight games hit the same broken server.
        if existing_backoff > now:
            return
        server.healthy = False
        server.last_health_check = datetime.now()
        if cooldown > 0.0:
            self._server_backoff_until[key] = now + cooldown
        logger.warning(
            "Temporarily backing off server %s for %.1fs after failure: %s",
            key,
            cooldown,
            reason,
        )

    def record_runtime_server_failure(self, server: GameServer, reason: Any):
        """Called by GameInstance when in-game transport failures occur."""
        self._mark_server_failure(server, reason)

    async def _maybe_refresh_health_check(self, force: bool = False):
        """Run health checks on an interval so servers can recover automatically."""
        async with self._health_check_lock:
            interval = max(0.0, float(self.health_check_interval_sec))
            now = asyncio.get_running_loop().time()
            if not force and interval > 0.0 and (now - float(self._last_health_check_monotonic)) < interval:
                return
            self._last_health_check_monotonic = now
            try:
                await self.health_check()
            except asyncio.CancelledError:
                # Propagate true task cancellation; otherwise treat transport-layer
                # cancellation from aiohttp internals as transient noise.
                if self._cancellation_requested():
                    raise
                logger.warning("Periodic health check cancelled unexpectedly; continuing")
            except Exception as e:
                logger.warning("Periodic health check failed: %s", e)

    def _build_session(self, timeout_total: Optional[float] = None) -> aiohttp.ClientSession:
        if timeout_total is None:
            timeout_total = self._parse_float_env("TM_HTTP_REQUEST_TOTAL_TIMEOUT_SEC", 90.0, min_value=10.0)
        timeout_value = max(10.0, float(timeout_total))
        connector_limit = self._parse_int_env("TM_HTTP_CONNECTOR_LIMIT", 256, min_value=0)
        connector_limit_per_host = self._parse_int_env("TM_HTTP_CONNECTOR_LIMIT_PER_HOST", 128, min_value=0)
        force_close = GameInstance._env_flag("TM_HTTP_FORCE_CLOSE_CONNECTIONS", default=False)
        use_dns_cache = GameInstance._env_flag("TM_HTTP_USE_DNS_CACHE", default=True)
        ttl_dns_cache_sec = max(10.0, self._parse_float_env("TM_HTTP_DNS_CACHE_TTL_SEC", 300.0, min_value=0.0))
        connector = aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=connector_limit_per_host,
            force_close=force_close,
            use_dns_cache=use_dns_cache,
            ttl_dns_cache=ttl_dns_cache_sec if use_dns_cache else None,
        )
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_value),
            connector=connector,
        )

    def ensure_session(self, timeout_total: Optional[float] = None) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = self._build_session(timeout_total=timeout_total)
        return self.session

    async def _prewarm_connections(self) -> None:
        """Open connections to each game server to eliminate cold-start TCP handshake.

        Controlled by TM_HTTP_PREWARM_CONNECTIONS (0=disabled, 2-4 recommended).
        Issues lightweight GETs to each server in parallel.
        """
        prewarm = self._parse_int_env("TM_HTTP_PREWARM_CONNECTIONS", 0, min_value=0)
        if prewarm <= 0:
            return
        prewarm = min(prewarm, 8)
        session = self.session
        if session is None or session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=5.0)

        async def _prewarm_one(url: str, server: GameServer) -> None:
            try:
                async with session.get(url, timeout=timeout) as resp:
                    await resp.read()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Prewarm GET to %s:%s failed: %s", server.host, server.port, e)

        async def _prewarm_server(server: GameServer) -> None:
            base = f"http://{server.host}:{server.port}"
            tasks = [_prewarm_one(f"{base}/", server) for _ in range(prewarm)]
            await asyncio.gather(*tasks)

        try:
            await asyncio.gather(*(_prewarm_server(s) for s in self.servers))
            logger.info("Connection prewarm complete: %d requests per server", prewarm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Connection prewarm had errors (non-fatal): %s", e)
        
    async def __aenter__(self):
        self.session = self.ensure_session(timeout_total=None)
        await self._prewarm_connections()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _close_session_quietly(
        self,
        session: Optional[aiohttp.ClientSession],
        *,
        propagate_cancel: bool = False,
    ):
        if session is None or session.closed:
            return
        try:
            await asyncio.shield(session.close())
        except asyncio.CancelledError:
            # Ensure close is still attempted even when caller is cancelling.
            try:
                await session.close()
            except Exception:
                pass
            if propagate_cancel:
                raise
        except Exception:
            pass

    async def _drain_retired_sessions(
        self,
        *,
        force: bool = False,
        now: Optional[float] = None,
        propagate_cancel: bool = False,
    ):
        if not self._retired_sessions:
            return
        if now is None:
            try:
                now = asyncio.get_running_loop().time()
            except RuntimeError:
                now = 0.0

        to_close: List[aiohttp.ClientSession] = []
        keep: List[Tuple[aiohttp.ClientSession, float]] = []
        for retired, retire_after in self._retired_sessions:
            if retired is None or retired.closed:
                continue
            if force or float(retire_after) <= float(now):
                to_close.append(retired)
            else:
                keep.append((retired, float(retire_after)))
        self._retired_sessions = keep

        for retired in to_close:
            await self._close_session_quietly(
                retired,
                propagate_cancel=propagate_cancel,
            )

    async def close(self):
        """Close shared HTTP session used for game/server API calls."""
        async with self._recycle_lock:
            session = self.session
            self.session = None
            if session is not None and not session.closed:
                self._retired_sessions.append((session, 0.0))
            await self._drain_retired_sessions(force=True, propagate_cancel=True)

    async def recycle_session(self):
        """Rotate to a fresh shared session after transport errors.

        Important: do not close the old session immediately. Doing that can
        close connectors still used by in-flight requests from other games.
        Retired sessions are closed after a grace period.
        """
        async with self._recycle_lock:
            try:
                now = asyncio.get_running_loop().time()
            except RuntimeError:
                now = 0.0
            await self._drain_retired_sessions(now=now)
            if (now - self._last_recycle_monotonic) < 2.0:
                return
            self._last_recycle_monotonic = now

            old_session = self.session
            self.session = self._build_session(timeout_total=None)

            if old_session is not None and not old_session.closed:
                request_timeout = self._parse_float_env(
                    "TM_HTTP_REQUEST_TOTAL_TIMEOUT_SEC",
                    90.0,
                    min_value=10.0,
                )
                default_grace_sec = max(30.0, float(request_timeout) + 5.0)
                grace_sec = self._parse_float_env(
                    "TM_RECYCLE_SESSION_GRACE_SEC",
                    default_grace_sec,
                    min_value=5.0,
                )
                self._retired_sessions.append((old_session, now + float(grace_sec)))

                max_retired = self._parse_int_env(
                    "TM_RECYCLED_SESSION_MAX",
                    32,
                    min_value=1,
                )
                while len(self._retired_sessions) > max_retired:
                    overflow_session, _ = self._retired_sessions.pop(0)
                    await self._close_session_quietly(overflow_session)

                logger.warning(
                    "Recycled shared HTTP session; retiring previous connector for %.1fs.",
                    grace_sec,
                )

    def _default_game_options(self) -> Dict[str, Any]:
        return {
            'altVenusBoard': False,
            'aresExtremeVariant': False,
            'bannedCards': [],
            'board': "random all",
            'ceosDraftVariant': False,
            'customCeos': [],
            'customColoniesList': [],
            'customCorporationsList': [],
            'customPreludes': [],
            'draftVariant': True,
            'escapeVelocityBonusSeconds': 2,
            'escapeVelocityMode': False,
            'expansions': {
                'ares': False,
                'ceo': False,
                'colonies': False,
                'community': True,
                'corpera': False,
                'moon': True,
                'pathfinders': False,
                'prelude': True,
                'prelude2': True,
                'promo': False,
                'starwars': False,
                'turmoil': False,
                'underworld': False,
                'venus': True,
            },
            'fastModeOption': False,
            'includeFanMA': False,
            'includedCards': [],
            'initialDraft': False,
            'modularMA': False,
            'moonStandardProjectVariant': False,
            'moonStandardProjectVariant1': False,
            'politicalAgendasExtension': "Standard",
            'preludeDraftVariant': True,
            'randomFirstPlayer': True,
            'randomMA': "No randomization",
            'removeNegativeGlobalEventsOption': False,
            'requiresMoonTrackCompletion': True,
            'requiresVenusTrackCompletion': True,
            'seed': "12345",
            'showOtherPlayersVP': False,
            'showTimers': True,
            'shuffleMapOption': True,
            'solarPhaseOption': False,
            'soloTR': False,
            'startingCeos': 0,
            'startingCorporations': 2,
            'startingPreludes': 4,
            'twoCorpsVariant': True,
            'undoOption': False,
        }

    def _load_game_options(self) -> Dict[str, Any]:
        options_path = os.getenv('GAME_OPTIONS_FILE')
        if not options_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            options_path = os.path.join(base_dir, 'game_options.base_prelude.json')
        try:
            with open(options_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("GAME_OPTIONS_FILE did not contain a JSON object. Using defaults.")
                return self._default_game_options()
            return data
        except FileNotFoundError:
            logger.warning(f"GAME_OPTIONS_FILE not found at {options_path}. Using defaults.")
            return self._default_game_options()
        except Exception as e:
            logger.warning(f"Failed to load GAME_OPTIONS_FILE from {options_path}: {e}. Using defaults.")
            return self._default_game_options()

    def _merge_options(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all game servers"""
        self.ensure_session(timeout_total=None)
        
        results = {}
        health_timeout = aiohttp.ClientTimeout(total=max(0.5, float(self.server_health_timeout_sec)))

        async def _check_server(server: GameServer):
            key = self._server_key(server)
            healthy = False
            failure = None
            try:
                async with self.session.get(f"http://{server.host}:{server.port}/", timeout=health_timeout) as response:
                    healthy = response.status == 200
            except asyncio.CancelledError as e:
                if self._cancellation_requested():
                    raise
                failure = RuntimeError(f"health check cancelled unexpectedly: {e}")
                healthy = False
            except Exception as e:
                failure = e
                healthy = False
            server.healthy = bool(healthy)
            server.last_health_check = datetime.now()
            results[key] = bool(healthy)
            if healthy:
                self._server_backoff_until.pop(key, None)
            elif failure is not None:
                logger.warning("Health check failed for %s: %s", key, _exc_summary(failure))

        await asyncio.gather(*(_check_server(server) for server in self.servers))
        
        healthy_count = sum(results.values())
        logger.info(f"Health check complete: {healthy_count}/{len(self.servers)} servers healthy")
        
        return results

    async def recycle_idle_servers(self) -> bool:
        """Recycle the dedicated RL servers only after every tracked game ends.

        The game database is a container tmpfs mount, so a Docker-managed
        process restart removes old games that cache eviction intentionally
        leaves available for later reload.  The endpoint is opt-in and token
        protected; normal TM clusters remain unaffected.
        """
        if not self.rl_recycle_enabled:
            return False
        if not self.rl_control_token:
            logger.warning("Skipping idle-server recycle: RL_CONTROL_TOKEN is not configured")
            return False

        session = self.ensure_session(timeout_total=None)
        headers = {"x-rl-control-token": self.rl_control_token}
        request_timeout = aiohttp.ClientTimeout(total=float(self.rl_recycle_request_timeout_sec))

        async with self._server_slot_lock:
            active = {self._server_key(server): int(server.active_games) for server in self.servers}
            if any(active.values()):
                logger.info("Skipping idle-server recycle; active games remain: %s", active)
                return False

        async def _check_capability(server: GameServer) -> bool:
            endpoint = f"http://{server.host}:{server.port}/api/rl/recycle"
            try:
                async with session.get(endpoint, headers=headers, timeout=request_timeout) as response:
                    await response.read()
                    if response.status == 200:
                        return True
                    logger.warning(
                        "Idle recycle unavailable on %s (HTTP %s); no servers will restart",
                        self._server_key(server),
                        response.status,
                    )
            except Exception as exc:
                logger.warning(
                    "Idle recycle capability check failed for %s: %s; no servers will restart",
                    self._server_key(server),
                    exc,
                )
            return False

        # Do not permit a partial cluster recycle. This preflight is
        # non-destructive, so older server images safely reject it with 404.
        capable = await asyncio.gather(*(_check_capability(server) for server in self.servers))
        if not all(capable):
            return False

        async with self._server_slot_lock:
            active = {self._server_key(server): int(server.active_games) for server in self.servers}
            if any(active.values()):
                logger.info("Skipping idle-server recycle after capability check; active games remain: %s", active)
                return False
            # Prevent a new reservation from racing the accepted recycle requests.
            for server in self.servers:
                server.healthy = False

        async def _request_recycle(server: GameServer) -> bool:
            endpoint = f"http://{server.host}:{server.port}/api/rl/recycle"
            try:
                async with session.post(endpoint, headers=headers, timeout=request_timeout) as response:
                    await response.read()
                    if response.status == 202:
                        logger.info("Idle recycle accepted by %s", self._server_key(server))
                        return True
                    logger.warning(
                        "Idle recycle rejected by %s with HTTP %s",
                        self._server_key(server),
                        response.status,
                    )
            except Exception as exc:
                logger.warning("Idle recycle request failed for %s: %s", self._server_key(server), exc)
            return False

        accepted = await asyncio.gather(*(_request_recycle(server) for server in self.servers))
        # A request can be accepted even if its response gets lost during the
        # process exit. Always wait for the whole cluster to be healthy again
        # before allowing the next batch, including on a partial failure.
        deadline = asyncio.get_running_loop().time() + float(self.rl_recycle_ready_timeout_sec)
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.5)
            results = await self.health_check()
            if all(results.values()):
                logger.info("All %d RL servers are healthy after idle recycle", len(self.servers))
                return bool(all(accepted))
        logger.warning("Timed out waiting for RL servers after idle recycle")
        return False
    
    def _get_best_server(self) -> GameServer:
        """Get the server with the least load"""
        healthy_servers = [s for s in self.servers if s.healthy]
        
        if not healthy_servers:
            raise RuntimeError("No healthy game servers available")
        
        # Return server with least active games
        return min(healthy_servers, key=lambda s: s.active_games)

    def _pick_best_server_with_capacity(self) -> Optional[GameServer]:
        now = asyncio.get_running_loop().time()
        # Drop expired backoff entries.
        for key, backoff_until in list(self._server_backoff_until.items()):
            if float(backoff_until) <= now:
                self._server_backoff_until.pop(key, None)

        healthy_servers = [s for s in self.servers if s.healthy]
        if not healthy_servers:
            return None

        healthy_servers = [
            s for s in healthy_servers
            if float(self._server_backoff_until.get(self._server_key(s), 0.0)) <= now
        ]
        if not healthy_servers:
            return None

        cap = int(self.max_active_games_per_server)
        if cap > 0:
            healthy_servers = [s for s in healthy_servers if int(s.active_games) < cap]
            if not healthy_servers:
                return None

        return min(healthy_servers, key=lambda s: s.active_games)

    async def _reserve_server_slot(self) -> GameServer:
        deadline = asyncio.get_running_loop().time() + float(self.server_slot_wait_timeout_sec)

        while True:
            async with self._server_slot_lock:
                selected = self._pick_best_server_with_capacity()
                if selected is not None:
                    selected.active_games += 1
                    return selected

            try:
                await self._maybe_refresh_health_check()
            except asyncio.CancelledError:
                if self._cancellation_requested():
                    raise
                logger.warning("Server-slot health refresh cancelled unexpectedly; retrying")

            if asyncio.get_running_loop().time() >= deadline:
                cap = int(self.max_active_games_per_server)
                if cap > 0:
                    raise RuntimeError(
                        f"No game server capacity available (MAX_ACTIVE_GAMES_PER_SERVER={cap})"
                    )
                raise RuntimeError("No healthy game servers available")

            await asyncio.sleep(float(self.server_slot_retry_sleep_sec))

    async def release_server_slot(self, server: GameServer):
        async with self._server_slot_lock:
            server.active_games = max(0, int(server.active_games) - 1)
    
    async def create_game(self, 
                         game_id: str,
                         player_names: List[str],
                         game_options: Dict[str, Any]) -> GameInstance:
        """Create a new game on the best available server"""
        self.ensure_session(timeout_total=None)
        
        # Prepare game creation request with a preset + runtime overrides.
        base_options = self.base_game_options or self._default_game_options()
        create_request = self._merge_options(base_options, game_options or {})
        players_beginner = bool(create_request.pop('_players_beginner', False))
        create_request['players'] = [
            {
                'name': name,
                'color': ['red', 'blue', 'green', 'yellow'][i % 4],
                'beginner': players_beginner,
                'handicap': 0,
                'first': False
            }
            for i, name in enumerate(player_names)
        ]

        attempts = max(1, int(self.create_game_retry_attempts))
        last_error: Optional[Exception] = None

        for attempt_idx in range(attempts):
            await self._maybe_refresh_health_check()
            server = await self._reserve_server_slot()
            slot_reserved = True
            server_key = self._server_key(server)

            try:
                base_url = f"http://{server.host}:{server.port}"

                # Create the game
                async with self.session.post(
                    f"{base_url}/api/creategame",
                    json=create_request,
                    headers={'Content-Type': 'application/json'},
                ) as response:
                    if response.status != 200:
                        response_text = (await response.read()).decode("utf-8")
                        error = RuntimeError(f"Failed to create game: {response.status} - {response_text}")
                        # Do not retry likely caller/config errors.
                        setattr(error, "_retryable", bool(response.status >= 500 or response.status in (408, 429)))
                        raise error

                    # Get the created game data
                    game_data = _json_loads_bytes(await response.read())

                    # Extract game ID from response (authoritative)
                    actual_game_id = game_data.get('id')
                    if not actual_game_id:
                        raise RuntimeError("Game ID not found in response")

                # Create game instance
                game_instance = GameInstance(actual_game_id, server, self.session, cluster=self)
                slot_reserved = False

                # Initialize player IDs
                for player_name in player_names:
                    try:
                        await game_instance.join_player(player_name)
                    except asyncio.CancelledError as e:
                        if self._cancellation_requested():
                            raise
                        logger.warning(
                            "Join player %s was unexpectedly cancelled on %s: %s",
                            player_name,
                            server_key,
                            e,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to join player {player_name}: {e}")

                logger.info(f"Created game {actual_game_id} on {server.host}:{server.port}")
                return game_instance

            except Exception as e:
                last_error = e
                retryable = bool(getattr(e, "_retryable", True))
                if slot_reserved:
                    await self.release_server_slot(server)
                if retryable:
                    self._mark_server_failure(server, e)

                has_next_attempt = (attempt_idx + 1) < attempts
                if retryable and has_next_attempt:
                    backoff_sec = float(self.create_game_retry_backoff_sec) * float(attempt_idx + 1)
                    logger.warning(
                        "Create game attempt %d/%d failed on %s: %s. Retrying after %.2fs",
                        attempt_idx + 1,
                        attempts,
                        server_key,
                        _exc_summary(e),
                        backoff_sec,
                    )
                    if backoff_sec > 0.0:
                        await asyncio.sleep(backoff_sec)
                    continue

                logger.error(f"Failed to create game on {server.host}:{server.port}: {e}")
                raise
            except asyncio.CancelledError as e:
                if slot_reserved:
                    await self.release_server_slot(server)
                if self._cancellation_requested():
                    raise
                wrapped = RuntimeError(f"create game attempt cancelled on {server_key}: {e}")
                last_error = wrapped
                self._mark_server_failure(server, wrapped)
                has_next_attempt = (attempt_idx + 1) < attempts
                if has_next_attempt:
                    backoff_sec = float(self.create_game_retry_backoff_sec) * float(attempt_idx + 1)
                    logger.warning(
                        "Create game attempt %d/%d cancelled unexpectedly on %s: %s. Retrying after %.2fs",
                        attempt_idx + 1,
                        attempts,
                        server_key,
                        e,
                        backoff_sec,
                    )
                    if backoff_sec > 0.0:
                        await asyncio.sleep(backoff_sec)
                    continue
                logger.error("Failed to create game on %s due to unexpected cancellation: %s", server_key, e)
                raise wrapped

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to create game for unknown reason")
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get statistics for all servers"""
        public_map = self._parse_mapping_env(os.getenv("PUBLIC_TM_MAP", ""))
        server_id_map = self._parse_mapping_env(os.getenv("PUBLIC_TM_SERVER_ID_MAP", ""))
        default_public_raw = str(os.getenv("PUBLIC_TM_URL", "http://localhost:8081"))
        default_public_base = (
            default_public_raw.split(",", 1)[0].strip()
            if "," in default_public_raw
            else default_public_raw.strip()
        )
        stats = {
            'total_servers': len(self.servers),
            'healthy_servers': sum(1 for s in self.servers if s.healthy),
            'total_active_games': sum(s.active_games for s in self.servers),
            'input_reject_count': int(self.input_reject_count),
            'payment_reject_count': int(self.payment_reject_count),
            'transport_timing': {
                key: {
                    'calls': int(self.transport_timing_counts.get(key, 0) or 0),
                    'total_sec': float(self.transport_timing_totals_sec.get(key, 0.0) or 0.0),
                    'avg_ms': (
                        (float(self.transport_timing_totals_sec.get(key, 0.0) or 0.0) * 1000.0)
                        / float(max(1, int(self.transport_timing_counts.get(key, 0) or 0)))
                    ),
                }
                for key in sorted(set(self.transport_timing_totals_sec.keys()) | set(self.transport_timing_counts.keys()))
            },
            'servers': []
        }
        
        for server in self.servers:
            server_key = f"{server.host}:{server.port}"
            public_base = GameInstance._normalize_public_base(public_map.get(server_key, default_public_base))
            server_id = str(server_id_map.get(server_key, "")).strip()
            links: Dict[str, str] = {}
            if server_id:
                encoded_server_id = quote_plus(server_id)
                links = {
                    "admin": f"{public_base}/admin?serverId={encoded_server_id}",
                    "games_overview": f"{public_base}/games-overview?serverId={encoded_server_id}",
                    "metrics": f"{public_base}/api/metrics?serverId={encoded_server_id}",
                }
            server_info = {
                'key': server_key,
                'host': server.host,
                'port': server.port,
                'healthy': server.healthy,
                'active_games': server.active_games,
                'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None,
                'public_base': public_base,
                'server_id': server_id if server_id else None,
                'links': links,
            }
            stats['servers'].append(server_info)
        
        return stats

    def record_input_reject(self, response_text: str):
        self.input_reject_count = int(self.input_reject_count) + 1
        response_l = str(response_text or '').lower()
        payment_error_markers = [
            'did not spend enough',
            'pay for card',
            'cannot pay',
            'invalid payment',
            'payment',
            'resources to spend',
            'does not have',
            'cannot afford',
            'm€',
            'megacredit',
        ]
        if any(marker in response_l for marker in payment_error_markers):
            self.payment_reject_count = int(self.payment_reject_count) + 1

    def record_transport_timing(self, timing_key: str, elapsed_sec: float):
        key = str(timing_key or "").strip()
        if not key:
            return
        elapsed = max(0.0, float(elapsed_sec or 0.0))
        self.transport_timing_totals_sec[key] = float(self.transport_timing_totals_sec.get(key, 0.0) or 0.0) + elapsed
        self.transport_timing_counts[key] = int(self.transport_timing_counts.get(key, 0) or 0) + 1
