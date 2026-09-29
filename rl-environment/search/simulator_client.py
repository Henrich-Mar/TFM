"""Async HTTP client for the token-protected TypeScript search service.

Wire contract: ``tfm.rl-search.v1`` (``src/common/models/RlSearchModel.ts``).
Every request carries ``x-rl-control-token``; the routes answer 404 while
``RL_SEARCH_ENABLED`` is off.  Observations are raw upstream ``PlayerViewModel``
JSON and are normalized with the same inbound schema aliasing the live game
client uses in ``game_interface.py``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from tfm_schema import (
    adapt_outbound_payment_schema,
    normalize_inbound_player_schema,
    uses_lowercase_payment_mc,
)

logger = logging.getLogger("rl.search")

SCHEMA_VERSION = "tfm.rl-search.v1"
ROUTE_START = "/api/rl/search/start"
ROUTE_REPLAY = "/api/rl/search/replay"
ROUTE_CLOSE = "/api/rl/search/close"

MAX_BRANCHES_PER_BATCH = 64
MAX_STEPS_PER_BRANCH = 32

UNAVAILABLE_CODES = {
    "session_capacity",
    "determinization_disabled",
    "unsupported_ruleset",
    "unsupported_root",
    "unsupported_hidden_zone",
    "root_not_round_trippable",
    "restore_failed",
    "game_not_found",
}


class SearchServiceError(RuntimeError):
    """Structured failure returned by the search endpoints."""

    def __init__(self, code: str, message: str, status: int = 0) -> None:
        super().__init__(f"{code}: {message}")
        self.code = str(code or "unknown")
        self.message = str(message or "")
        self.status = int(status)


class SearchUnavailableError(SearchServiceError):
    """The server cannot service search right now; fall back to the policy."""


class SearchProtocolError(SearchServiceError):
    """The server response violated the tfm.rl-search.v1 contract."""


@dataclass
class StartResult:
    session_id: str
    root_digest: str
    root_player_id: str
    observation: Dict[str, Any]
    lowercase_mc: bool


@dataclass
class BranchResult:
    branch_id: str
    status: str
    applied_steps: int
    state_digest: str
    next_prompt: Optional[Tuple[str, Dict[str, Any]]] = None
    terminal_players: Optional[List[Dict[str, Any]]] = None
    error_code: str = ""
    error_step_index: int = -1
    error_message: str = ""
    new_steps_applied: int = 0
    reused_steps: int = 0
    root_observation: Optional[Dict[str, Any]] = None

    @property
    def usable(self) -> bool:
        return self.status == "next_prompt" or self.status == "terminal"


@dataclass
class SearchClientStats:
    starts: int = 0
    replays: int = 0
    closes: int = 0
    replay_batch_sec: List[float] = field(default_factory=list)
    applied_steps: int = 0
    new_steps_applied: int = 0
    reused_steps: int = 0
    replay_payload_bytes: int = 0
    inference_sec: float = 0.0
    prompt_evaluations: int = 0
    eval_cache_hits: int = 0
    eval_cache_misses: int = 0
    requested_branches: int = 0
    requested_path_steps: int = 0
    max_path_steps: int = 0
    failures: Dict[str, int] = field(default_factory=dict)

    def record_failure(self, code: str) -> None:
        self.failures[code] = int(self.failures.get(code, 0)) + 1

    def mean_batch_sec(self) -> float:
        if not self.replay_batch_sec:
            return 0.0
        return sum(self.replay_batch_sec) / len(self.replay_batch_sec)

    def applied_inputs_per_sec(self) -> float:
        total = sum(self.replay_batch_sec)
        if total <= 0.0:
            return 0.0
        return self.applied_steps / total

    def new_transitions_per_sec(self) -> float:
        total = sum(self.replay_batch_sec)
        if total <= 0.0:
            return 0.0
        return self.new_steps_applied / total

    def replay_amplification(self) -> float:
        if self.new_steps_applied <= 0:
            return 0.0
        return self.applied_steps / self.new_steps_applied

    def snapshot(self) -> Dict[str, Any]:
        return {
            "starts": self.starts,
            "replay_batches": self.replays,
            "closes": self.closes,
            "mean_batch_sec": round(self.mean_batch_sec(), 4),
            "applied_steps": self.applied_steps,
            "new_steps_applied": self.new_steps_applied,
            "reused_steps": self.reused_steps,
            "replay_payload_bytes": self.replay_payload_bytes,
            "applied_inputs_per_sec": round(self.applied_inputs_per_sec(), 2),
            "new_transitions_per_sec": round(self.new_transitions_per_sec(), 2),
            "replay_amplification": round(self.replay_amplification(), 3),
            "inference_sec": round(float(self.inference_sec), 4),
            "prompt_evaluations": int(self.prompt_evaluations),
            "eval_cache_hits": int(self.eval_cache_hits),
            "eval_cache_misses": int(self.eval_cache_misses),
            "mean_requested_path_steps": round(
                self.requested_path_steps / max(1, self.requested_branches), 3
            ),
            "max_path_steps": int(self.max_path_steps),
            "failures": dict(self.failures),
        }


class SearchClient:
    """One client instance per searched decision keeps session lifetimes clear."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout_sec: float = 30.0,
        stats: Optional[SearchClientStats] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.token = token if token is not None else os.getenv("RL_CONTROL_TOKEN", "")
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.stats = stats if stats is not None else SearchClientStats()
        self._session: Optional[aiohttp.ClientSession] = None

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
            )
        return self._session

    async def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            raise SearchUnavailableError("missing_token", "RL_CONTROL_TOKEN is not configured")
        url = f"{self.base_url}{route}"
        headers = {
            "Content-Type": "application/json",
            "x-rl-control-token": self.token,
        }
        session = self._get_session()
        try:
            async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
                body = await response.read()
                text = body.decode("utf-8", errors="replace")
                if response.status >= 400:
                    code = f"http_{response.status}"
                    message = text[:400]
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                            code = str(parsed["error"].get("code", code))
                            message = str(parsed["error"].get("message", message))
                    except ValueError:
                        pass
                    self.stats.record_failure(code)
                    if code in UNAVAILABLE_CODES or response.status == 404 or response.status == 429:
                        raise SearchUnavailableError(code, message, response.status)
                    if response.status == 401:
                        raise SearchUnavailableError("unauthorized", message, response.status)
                    raise SearchServiceError(code, message, response.status)
                try:
                    parsed = json.loads(text)
                except ValueError as exc:
                    raise SearchProtocolError("invalid_json_response", str(exc), response.status) from exc
                if not isinstance(parsed, dict):
                    raise SearchProtocolError("invalid_response", "response is not an object")
                if parsed.get("schemaVersion") != SCHEMA_VERSION:
                    raise SearchProtocolError(
                        "schema_mismatch",
                        f"expected {SCHEMA_VERSION}, got {parsed.get('schemaVersion')!r}",
                    )
                return parsed
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.stats.record_failure("transport_error")
            raise SearchServiceError("transport_error", str(exc)) from exc

    @staticmethod
    def _normalize_observation(observation: Any) -> Dict[str, Any]:
        if not isinstance(observation, dict):
            raise SearchProtocolError("invalid_observation", "observation payload is not an object")
        return normalize_inbound_player_schema(observation)

    async def start(self, player_id: str) -> StartResult:
        payload = {"playerId": str(player_id)}
        data = await self._post(ROUTE_START, payload)
        for key in ("sessionId", "rootDigest", "rootPlayerId"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise SearchProtocolError("invalid_start_response", f"missing {key}")
        raw_observation = data.get("observation")
        if not isinstance(raw_observation, dict):
            raise SearchProtocolError("invalid_observation", "observation payload is not an object")
        # Detect the server payment variant before aliasing adds camelCase keys,
        # matching GameInstance.get_player_state ordering.
        lowercase_mc = bool(uses_lowercase_payment_mc(raw_observation))
        observation = normalize_inbound_player_schema(raw_observation)
        self.stats.starts += 1
        return StartResult(
            session_id=data["sessionId"],
            root_digest=data["rootDigest"],
            root_player_id=data["rootPlayerId"],
            observation=observation,
            lowercase_mc=lowercase_mc,
        )

    async def replay(
        self,
        session_id: str,
        branches: List[Dict[str, Any]],
    ) -> List[BranchResult]:
        if not branches:
            raise SearchServiceError("invalid_branch_count", "replay requires at least one branch")
        if len(branches) > MAX_BRANCHES_PER_BATCH:
            raise SearchServiceError(
                "invalid_branch_count",
                f"a replay batch may contain at most {MAX_BRANCHES_PER_BATCH} branches",
            )
        for branch in branches:
            steps = branch.get("steps")
            if not isinstance(steps, list) or len(steps) > MAX_STEPS_PER_BRANCH:
                raise SearchServiceError(
                    "invalid_step_count",
                    f"branch {branch.get('branchId')!r} exceeds {MAX_STEPS_PER_BRANCH} steps",
                )
        payload = {"sessionId": str(session_id), "branches": list(branches)}
        path_lengths = [len(branch.get("steps", []) or []) for branch in branches]
        self.stats.requested_branches += len(path_lengths)
        self.stats.requested_path_steps += sum(path_lengths)
        self.stats.max_path_steps = max([self.stats.max_path_steps, *path_lengths])
        self.stats.replay_payload_bytes += len(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))
        started = time.perf_counter()
        data = await self._post(ROUTE_REPLAY, payload)
        elapsed = time.perf_counter() - started
        results_raw = data.get("results")
        if not isinstance(results_raw, list):
            raise SearchProtocolError("invalid_replay_response", "results is not a list")
        results: List[BranchResult] = []
        applied = 0
        for item in results_raw:
            if not isinstance(item, dict):
                raise SearchProtocolError("invalid_replay_response", "branch result is not an object")
            result = self._parse_branch_result(item)
            applied += int(result.applied_steps or 0)
            self.stats.new_steps_applied += int(result.new_steps_applied or 0)
            self.stats.reused_steps += int(result.reused_steps or 0)
            results.append(result)
        self.stats.replays += 1
        self.stats.replay_batch_sec.append(float(elapsed))
        self.stats.applied_steps += applied
        return results

    @staticmethod
    def _parse_branch_result(item: Dict[str, Any]) -> BranchResult:
        status = str(item.get("status", "") or "")
        if status not in {"next_prompt", "terminal", "boundary", "rejected"}:
            raise SearchProtocolError("invalid_branch_status", f"unknown status {status!r}")
        next_prompt: Optional[Tuple[str, Dict[str, Any]]] = None
        raw_prompt = item.get("nextPrompt")
        if isinstance(raw_prompt, dict) and raw_prompt.get("observation") is not None:
            next_prompt = (
                str(raw_prompt.get("playerId", "") or ""),
                SearchClient._normalize_observation(raw_prompt.get("observation")),
            )
        terminal_players: Optional[List[Dict[str, Any]]] = None
        raw_terminal = item.get("terminal")
        if isinstance(raw_terminal, dict) and isinstance(raw_terminal.get("players"), list):
            terminal_players = [row for row in raw_terminal["players"] if isinstance(row, dict)]
        error = item.get("error") if isinstance(item.get("error"), dict) else {}
        return BranchResult(
            branch_id=str(item.get("branchId", "") or ""),
            status=status,
            applied_steps=int(item.get("appliedSteps", 0) or 0),
            state_digest=str(item.get("stateDigest", "") or ""),
            next_prompt=next_prompt,
            terminal_players=terminal_players,
            error_code=str(error.get("code", "") or ""),
            error_step_index=int(error.get("stepIndex", -1) if isinstance(error.get("stepIndex"), (int, float)) else -1),
            error_message=str(error.get("message", "") or ""),
            new_steps_applied=int(item.get("newStepsApplied", item.get("appliedSteps", 0)) or 0),
            reused_steps=int(item.get("reusedSteps", 0) or 0),
            root_observation=(
                SearchClient._normalize_observation(item.get("rootObservation"))
                if isinstance(item.get("rootObservation"), dict)
                else None
            ),
        )

    async def close(self, session_id: str) -> bool:
        data = await self._post(ROUTE_CLOSE, {"sessionId": str(session_id)})
        self.stats.closes += 1
        return bool(data.get("closed", False))

    @staticmethod
    def build_branch_request(
        branch_id: str,
        mode: str,
        steps: List[Dict[str, Any]],
        determinization_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        branch: Dict[str, Any] = {
            "branchId": str(branch_id),
            "mode": mode,
            "steps": list(steps),
        }
        if mode == "determinized":
            branch["determinizationSeed"] = int(determinization_seed if determinization_seed is not None else 0)
        return branch


def adapt_step_input(payload: Dict[str, Any], lowercase_mc: bool) -> Dict[str, Any]:
    """Reuse the live client's outbound payment schema adaptation for replay steps."""
    return adapt_outbound_payment_schema(payload, lowercase_mc)
