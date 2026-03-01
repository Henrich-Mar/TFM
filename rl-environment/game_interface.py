"""
Game Interface - Handles communication with Terraforming Mars game servers
"""
import asyncio
import aiohttp
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from urllib.parse import quote_plus, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

class ServerTransportError(RuntimeError):
    """Raised when transport-level communication with a TM server fails."""


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
            dump_path_raw = str(os.getenv("TM_DEBUG_INITIAL_CARDS_DUMP_FILE", "/app/logs/initial-cards-first.json")).strip()

            if dump_path_raw:
                try:
                    dump_path = Path(dump_path_raw)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_path.write_text(payload_full, encoding="utf-8")
                    logger.warning("Initial cards debug payload written to %s", dump_path.as_posix())
                except Exception as e:
                    logger.warning("Failed to write initial cards debug payload dump to %s: %s", dump_path_raw, e)

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
            logger.warning("First initialCards payload snapshot: %s", payload_full)

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
            self.session = self.cluster.ensure_session(timeout_total=60.0)
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
                    game_data = await response.json()
                    
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
        for attempt_no in range(1, max_retries + 1):
            try:
                session = self._get_session()
                async with session.get(f"{self.base_url}/api/player",
                                          params={'id': player_id}) as response:
                    if response.status == 200:
                        player_state = await response.json()
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
                            self.server.host, self.server.port, backoff, e,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    self._raise_transport_error("get player state", e)
                logger.error(f"Failed to get player state for {player_id}: {e!r}")
                raise

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
            payload_full = json.dumps(prepared_input_data, ensure_ascii=True, separators=(',', ':'))
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
            if self._env_flag("TM_DEBUG_LOG_INITIAL_CARDS", default=False):
                logger.warning(
                    "Sending initialCards payload (player=%s, game=%s): %s",
                    player_id,
                    self.get_public_game_url(),
                    payload_full,
                )
            should_skip, prepared_input_data = await self._refresh_or_skip_initial_cards_send(
                player_id=player_id,
                payload=prepared_input_data,
            )
            if should_skip:
                return True
            try:
                payload_full = json.dumps(prepared_input_data, ensure_ascii=True, separators=(',', ':'))
            except Exception:
                payload_full = str(prepared_input_data)
            payload_preview = payload_full
            if len(payload_preview) > 800:
                payload_preview = payload_preview[:800] + "...(truncated)"

        for attempt in range(retry_attempts):
            attempt_no = attempt + 1
            try:
                session = self._get_session()
                async with session.post(f"{self.base_url}/player/input",
                                           params={'id': player_id},
                                           json=prepared_input_data,
                                           headers={'Content-Type': 'application/json'}) as response:
                    if response.status == 200:
                        # The TM server returns the full PlayerViewModel JSON
                        # on every successful input.  Cache it so the next
                        # get_player_state() call returns instantly without a
                        # network round-trip.
                        try:
                            body = await response.json()
                            if isinstance(body, dict):
                                run_id = self._extract_run_id_from_state(body)
                                if run_id:
                                    self._latest_run_id_by_player[str(player_id)] = run_id
                                self._cached_player_state[str(player_id)] = body
                        except Exception:
                            pass
                        return True
                    else:
                        response_text = await response.text()
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
                        e,
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
                            e,
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

        return False
    
    async def get_final_state(self) -> Dict[str, Any]:
        """Get final game state after completion"""
        try:
            session = self._get_session()
            async with session.get(f"{self.base_url}/api/game", 
                                      params={'id': self.game_id}) as response:
                if response.status == 200:
                    return await response.json()
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
        self._server_backoff_until: Dict[str, float] = {}
        self._last_health_check_monotonic: float = 0.0
        self._health_check_lock = asyncio.Lock()
        self._recycle_lock = asyncio.Lock()
        self._last_recycle_monotonic: float = 0.0
        # Optional cross-component scratchpad for latest game URLs
        self.recent_games: List[Dict[str, str]] = []
        self.base_game_options = self._load_game_options()
        self.input_reject_count: int = 0
        self.payment_reject_count: int = 0

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

    def _build_session(self, timeout_total: float = 60.0) -> aiohttp.ClientSession:
        timeout_value = max(1.0, float(timeout_total))
        connector_limit = self._parse_int_env("TM_HTTP_CONNECTOR_LIMIT", 256, min_value=0)
        connector_limit_per_host = self._parse_int_env("TM_HTTP_CONNECTOR_LIMIT_PER_HOST", 128, min_value=0)
        force_close = GameInstance._env_flag("TM_HTTP_FORCE_CLOSE_CONNECTIONS", default=False)
        connector = aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=connector_limit_per_host,
            force_close=force_close,
        )
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_value),
            connector=connector,
        )

    def ensure_session(self, timeout_total: float = 60.0) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = self._build_session(timeout_total=timeout_total)
        return self.session
        
    async def __aenter__(self):
        self.session = self.ensure_session(timeout_total=60.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close shared HTTP session used for game/server API calls."""
        session = self.session
        self.session = None
        if session and not session.closed:
            try:
                await asyncio.shield(session.close())
            except asyncio.CancelledError:
                # Ensure close is still attempted even when caller is cancelling.
                try:
                    await session.close()
                except Exception:
                    pass
                raise

    async def recycle_session(self):
        """Force next request to use a fresh HTTP session after transport errors.

        Uses a lock + 2-second debounce so that concurrent callers don't
        destroy each other's in-flight requests via the shared connector.
        """
        async with self._recycle_lock:
            try:
                now = asyncio.get_running_loop().time()
            except RuntimeError:
                now = 0.0
            if (now - self._last_recycle_monotonic) < 2.0:
                return
            self._last_recycle_monotonic = now
            old_session = self.session
            self.session = None
            if old_session and not old_session.closed:
                try:
                    await asyncio.shield(old_session.close())
                except asyncio.CancelledError:
                    try:
                        await old_session.close()
                    except Exception:
                        pass
                    raise
                except Exception:
                    pass

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
                'community': False,
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
            'twoCorpsVariant': False,
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
        self.ensure_session(timeout_total=60.0)
        
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
                logger.warning(f"Health check failed for {key}: {failure}")

        await asyncio.gather(*(_check_server(server) for server in self.servers))
        
        healthy_count = sum(results.values())
        logger.info(f"Health check complete: {healthy_count}/{len(self.servers)} servers healthy")
        
        return results
    
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
        self.ensure_session(timeout_total=60.0)
        
        # Prepare game creation request with a preset + runtime overrides.
        base_options = self.base_game_options or self._default_game_options()
        create_request = self._merge_options(base_options, game_options or {})
        create_request['players'] = [
            {
                'name': name,
                'color': ['red', 'blue', 'green', 'yellow'][i % 4],
                'beginner': False,
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
                        response_text = await response.text()
                        error = RuntimeError(f"Failed to create game: {response.status} - {response_text}")
                        # Do not retry likely caller/config errors.
                        setattr(error, "_retryable", bool(response.status >= 500 or response.status in (408, 429)))
                        raise error

                    # Get the created game data
                    game_data = await response.json()

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
                        e,
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
