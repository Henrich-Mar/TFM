"""
API Server for monitoring RL training progress
"""
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from debug_decision_snapshot import (
    create_capture_request,
    list_capture_requests,
    list_saved_snapshots,
    load_snapshot,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Terraforming Mars RL Monitor", version="1.0.0")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global reference to coordinator (set when starting server)
coordinator = None

class DebugGameRequest(BaseModel):
    agent_ids: Optional[List[str]] = None


class HumanVsGenerationRequest(BaseModel):
    generation: Optional[int] = None
    random_generation: bool = True
    human_name: str = "You"
    bot_count: int = 3
    agent_indices: Optional[List[int]] = None
    seed: Optional[int] = None


class HumanVsBestRequest(BaseModel):
    human_name: str = "You"
    bot_count: int = 3
    seed: Optional[int] = None


class DecisionSnapshotCaptureRequest(BaseModel):
    agent_id: Optional[str] = None
    game_id: Optional[str] = None
    player_id: Optional[str] = None
    note: str = ""
    include_state_vector: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        logger.debug("Failed reading JSON file %s", path, exc_info=True)
    return {}


def _parse_named_urls(raw: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for token in [part.strip() for part in str(raw or "").split(",") if part.strip()]:
        if "=" in token:
            name, url = token.split("=", 1)
        else:
            name, url = f"coord-{len(entries) + 1}", token
        label = str(name or "").strip()
        normalized = _normalize_external_url(url)
        if label and normalized:
            entries.append((label, normalized.rstrip("/")))
    return entries


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _summarize_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": _safe_mean(values),
        "max": max(values),
    }


def _normalize_external_url(raw_url: Any) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return ""
    raw = raw.replace("/ui/game?", "/game?")

    if "," in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        preferred = None
        for part in parts:
            low = part.lower()
            if "/game?" in low or "/the-end?" in low or "/player?" in low:
                preferred = part
                break
        raw = preferred or (parts[0] if parts else raw)

    if "://" not in raw:
        raw = f"http://{raw}"

    try:
        parsed = urlsplit(raw)
    except Exception:
        return ""

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    netloc = parsed.netloc or parsed.path
    if not netloc:
        return ""

    return urlunsplit((scheme, netloc, parsed.path if parsed.netloc else "", parsed.query, ""))


def _sanitize_recent_games(entries: Any) -> List[Dict[str, str]]:
    sanitized: List[Dict[str, str]] = []
    if not isinstance(entries, list):
        return sanitized
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        game_id = str(entry.get("game_id", "") or "").strip()
        normalized_url = _normalize_external_url(entry.get("url", ""))
        if not game_id and not normalized_url:
            continue
        payload: Dict[str, str] = {"game_id": game_id}
        if normalized_url:
            payload["url"] = normalized_url
        sanitized.append(payload)
    return sanitized


def _sanitize_recent_end_screens(entries: Any) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    if not isinstance(entries, list):
        return sanitized
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        game_id = str(entry.get("game_id", "") or "").strip()
        raw_screens = entry.get("end_screens", [])
        screens: List[str] = []
        if isinstance(raw_screens, list):
            for raw_url in raw_screens:
                normalized = _normalize_external_url(raw_url)
                if normalized:
                    screens.append(normalized)
        if not game_id and not screens:
            continue
        sanitized.append({"game_id": game_id, "end_screens": screens})
    return sanitized


def _aggregate_behavior_stats(population: List[Any]) -> Dict[str, Any]:
    totals = {
        "total_decisions": 0,
        "policy_attempts": 0,
        "policy_successes": 0,
        "policy_rejections": 0,
        "policy_sampled_actions": 0,
        "epsilon_random_actions": 0,
        "fallback_decisions": 0,
        "fallback_random_attempts": 0,
        "fallback_random_successes": 0,
        "fallback_passes": 0,
        "no_available_actions": 0,
        "card_play_actions": 0,
        "steel_spent": 0,
        "titanium_spent": 0,
        "draft_decisions_total": 0,
        "draft_decisions_low_hand_ev": 0,
        "hate_draft_picks": 0,
        "hate_draft_picks_low_hand_ev": 0,
    }
    action_counts: Dict[str, int] = {}
    standard_project_counts: Dict[str, int] = {}
    avg_available_actions_samples: List[float] = []
    epsilon_values: List[float] = []
    temperature_values: List[float] = []
    learning_rate_values: List[float] = []

    for agent in population:
        try:
            stats = agent.get_behavior_stats()
        except Exception:
            stats = {}

        for key in totals.keys():
            totals[key] += int(stats.get(key, 0))

        for key, value in dict(stats.get("action_counts", {})).items():
            action_counts[key] = int(action_counts.get(key, 0)) + int(value)

        for key, value in dict(stats.get("standard_project_counts", {})).items():
            standard_project_counts[key] = int(standard_project_counts.get(key, 0)) + int(value)

        if "avg_available_actions" in stats:
            avg_available_actions_samples.append(float(stats.get("avg_available_actions", 0.0)))

        cfg = getattr(agent, "config", None)
        if cfg is not None:
            epsilon_values.append(float(getattr(cfg, "epsilon", 0.0)))
            temperature_values.append(float(getattr(cfg, "temperature", 0.0)))
            learning_rate_values.append(float(getattr(cfg, "learning_rate", 0.0)))

    total_recorded_actions = sum(action_counts.values())
    action_mix = {
        key: _safe_ratio(value, total_recorded_actions)
        for key, value in sorted(action_counts.items(), key=lambda item: item[1], reverse=True)
    }
    total_standard_project_actions = sum(standard_project_counts.values())
    standard_project_mix = {
        key: _safe_ratio(value, total_standard_project_actions)
        for key, value in sorted(standard_project_counts.items(), key=lambda item: item[1], reverse=True)
    }

    return {
        **totals,
        "policy_success_rate": _safe_ratio(totals["policy_successes"], totals["policy_attempts"]),
        "policy_sample_rate": _safe_ratio(totals["policy_sampled_actions"], totals["total_decisions"]),
        "epsilon_random_rate": _safe_ratio(totals["epsilon_random_actions"], totals["total_decisions"]),
        "fallback_decision_rate": _safe_ratio(totals["fallback_decisions"], totals["total_decisions"]),
        "fallback_random_success_rate": _safe_ratio(
            totals["fallback_random_successes"], totals["fallback_random_attempts"]
        ),
        "fallback_pass_rate": _safe_ratio(totals["fallback_passes"], totals["total_decisions"]),
        "hate_draft_rate": _safe_ratio(totals["hate_draft_picks"], totals["draft_decisions_total"]),
        "hate_draft_rate_low_hand_ev": _safe_ratio(
            totals["hate_draft_picks_low_hand_ev"],
            totals["draft_decisions_low_hand_ev"],
        ),
        "avg_available_actions": _safe_mean(avg_available_actions_samples),
        "card_plays_per_game": _safe_ratio(totals["card_play_actions"], sum(max(0, int(getattr(a, "games_played", 0))) for a in population)),
        "standard_project_ratio": _safe_ratio(total_standard_project_actions, totals["card_play_actions"] + total_standard_project_actions),
        "action_counts": action_counts,
        "action_mix": action_mix,
        "standard_project_counts": standard_project_counts,
        "standard_project_mix": standard_project_mix,
        "hyperparameters": {
            "epsilon": _summarize_values(epsilon_values),
            "temperature": _summarize_values(temperature_values),
            "learning_rate": _summarize_values(learning_rate_values),
        },
    }


def _aggregate_transformer_live_stats(population: List[Any]) -> Dict[str, Any]:
    total_agents = int(len(population))
    config_enabled_agents = 0
    reporting_agents = 0
    enabled_reporting_agents = 0
    latest_update_ts = 0.0
    token_count_values: List[float] = []
    active_token_values: List[float] = []
    active_row_values: List[float] = []
    attention_context_norm_values: List[float] = []
    fusion_delta_norm_values: List[float] = []
    fusion_share_values: List[float] = []

    for agent in population:
        cfg = getattr(agent, "config", None)
        if cfg is not None:
            config_enabled_agents += 1

        network = getattr(agent, "network", None)
        raw_stats = getattr(network, "last_transformer_stats", None)
        if not isinstance(raw_stats, dict):
            continue
        reporting_agents += 1

        stats_enabled = bool(raw_stats.get("enabled", False))
        if stats_enabled:
            enabled_reporting_agents += 1

        latest_update_ts = max(latest_update_ts, _to_float(raw_stats.get("timestamp", 0.0), 0.0))
        token_count_values.append(_to_float(raw_stats.get("token_count", 0.0), 0.0))
        active_token_values.append(_to_float(raw_stats.get("active_token_ratio", 0.0), 0.0))
        active_row_values.append(_to_float(raw_stats.get("active_row_ratio", 0.0), 0.0))
        attention_context_norm_values.append(_to_float(raw_stats.get("attention_context_norm", 0.0), 0.0))
        fusion_delta_norm_values.append(_to_float(raw_stats.get("fusion_delta_norm", 0.0), 0.0))
        fusion_share_values.append(_to_float(raw_stats.get("fusion_share", 0.0), 0.0))

    now_ts = float(time.time())
    seconds_since_update = max(0.0, now_ts - latest_update_ts) if latest_update_ts > 0 else None
    return {
        "agents_total": total_agents,
        "agents_config_enabled": int(config_enabled_agents),
        "agents_reporting": int(reporting_agents),
        "agents_reporting_enabled": int(enabled_reporting_agents),
        "token_count": _summarize_values(token_count_values),
        "active_token_ratio": _summarize_values(active_token_values),
        "active_row_ratio": _summarize_values(active_row_values),
        "attention_context_norm": _summarize_values(attention_context_norm_values),
        "fusion_delta_norm": _summarize_values(fusion_delta_norm_values),
        "fusion_share": _summarize_values(fusion_share_values),
        "latest_update_ts": float(latest_update_ts),
        "seconds_since_update": float(seconds_since_update) if seconds_since_update is not None else None,
    }


async def _collect_local_stats_payload() -> Dict[str, Any]:
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    # Get server cluster stats
    server_stats = await coordinator.game_cluster.get_server_stats()

    # Get evolution stats
    evolution_stats = coordinator.evolution_manager.get_evolution_stats()

    # Population stats
    eval_fitness_scores = []
    raw_eval_fitness_scores = []
    for agent in coordinator.population:
        gated_eval_score = coordinator.last_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_eval_fitness') else None
        raw_eval_score = coordinator.last_raw_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_raw_eval_fitness') else None
        eval_fitness_scores.append(float(gated_eval_score) if gated_eval_score is not None else float(agent.get_fitness_score()))
        raw_eval_fitness_scores.append(float(raw_eval_score) if raw_eval_score is not None else float(agent.get_fitness_score()))

    population_stats = {
        "size": len(coordinator.population),
        "current_generation": coordinator.current_generation,
        "total_games_played": sum(agent.games_played for agent in coordinator.population),
        "best_fitness": max(agent.get_fitness_score() for agent in coordinator.population) if coordinator.population else 0,
        "avg_fitness": sum(agent.get_fitness_score() for agent in coordinator.population) / len(coordinator.population) if coordinator.population else 0,
        "best_eval_fitness": max(eval_fitness_scores) if eval_fitness_scores else 0,
        "avg_eval_fitness": _safe_mean(eval_fitness_scores),
        "best_raw_eval_fitness": max(raw_eval_fitness_scores) if raw_eval_fitness_scores else 0,
        "avg_raw_eval_fitness": _safe_mean(raw_eval_fitness_scores),
    }
    behavior_stats = _aggregate_behavior_stats(coordinator.population)
    tournament_progress = coordinator.tournament_manager.get_progress_snapshot()
    generation_metrics = dict(getattr(coordinator, "last_generation_behavior_metrics", {}) or {})
    generation_gate = dict(getattr(coordinator, "last_generation_gate", {}) or {})
    generation_gate_overview = dict(generation_gate)
    generation_gate_overview.pop("per_agent", None)
    transformer_live = _aggregate_transformer_live_stats(coordinator.population)

    recent_games = []
    try:
        for g in getattr(coordinator, 'recent_games', [])[-20:]:
            recent_games.append(g)
    except Exception:
        recent_games = []
    if not recent_games:
        try:
            recent_games = getattr(coordinator.game_cluster, 'recent_games', [])[-20:]
        except Exception:
            recent_games = []
    recent_games = _sanitize_recent_games(recent_games)

    recent_end = []
    try:
        recent_end = getattr(coordinator, 'recent_end_screens', [])[-20:]
    except Exception:
        recent_end = []
    recent_end = _sanitize_recent_end_screens(recent_end)

    return {
        "population": population_stats,
        "behavior": behavior_stats,
        "generation_metrics": generation_metrics,
        "transformer_live": transformer_live,
        "generation_gate": generation_gate_overview,
        "generation_behavior_history": list(getattr(coordinator, "generation_behavior_history", [])[-50:]),
        "servers": server_stats,
        "evolution": evolution_stats,
        "tournaments": {
            "active": len(coordinator.tournament_manager.active_tournaments),
            "planned_games": int(tournament_progress.get("planned_games", 0)),
            "finished_games": int(tournament_progress.get("finished_games", 0)),
            "successful_games": int(tournament_progress.get("successful_games", 0)),
            "failed_games": int(tournament_progress.get("failed_games", 0)),
            "completion_rate": float(tournament_progress.get("completion_rate", 0.0)),
        },
        "links": {
            "end_screens_hint": "See tournament results payloads for per-game end screen URLs"
        },
        "recent_games": recent_games,
        "recent_end_screens": recent_end,
    }


def _collect_local_control_status_payload() -> Dict[str, Any]:
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    return {
        "paused": coordinator.is_paused(),
        "current_generation": coordinator.current_generation,
        "population_size": len(coordinator.population),
        "coordinator_id": str(getattr(coordinator, "coordinator_id", "") or "coord-1"),
        "num_coordinators": int(getattr(coordinator, "num_coordinators", 1) or 1),
    }


def _global_dashboard_sources() -> List[Tuple[str, str]]:
    raw = str(os.getenv("GLOBAL_DASHBOARD_COORDINATOR_URLS", "") or "").strip()
    if raw:
        return _parse_named_urls(raw)
    num_coordinators = max(1, _safe_int(os.getenv("NUM_COORDINATORS", "1"), 1))
    if num_coordinators <= 1:
        return []
    return [
        (f"coord-{idx}", f"http://rl-coordinator-{idx}:5000")
        for idx in range(1, num_coordinators + 1)
    ]


def _global_dashboard_enabled() -> bool:
    if str(os.getenv("GLOBAL_DASHBOARD_ENABLE", "") or "").strip():
        return str(os.getenv("GLOBAL_DASHBOARD_ENABLE", "0")).strip().lower() not in ("0", "false", "no", "off")
    return bool(_global_dashboard_sources())


async def _fetch_remote_json(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return {"error": f"http_{response.status}"}
            payload = await response.json()
            if isinstance(payload, dict):
                return payload
            return {"error": "invalid_json_shape"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _coordinator_summary(
    *,
    coord_id: str,
    base_url: str,
    stats_payload: Dict[str, Any],
    control_payload: Dict[str, Any],
    error: str = "",
) -> Dict[str, Any]:
    population = dict(stats_payload.get("population", {}) or {})
    tournaments = dict(stats_payload.get("tournaments", {}) or {})
    generation_metrics = dict(stats_payload.get("generation_metrics", {}) or {})
    servers = dict(stats_payload.get("servers", {}) or {})
    return {
        "coordinator_id": str(coord_id),
        "base_url": str(base_url).rstrip("/"),
        "dashboard_url": f"{str(base_url).rstrip('/')}/dashboard/local",
        "stats_url": f"{str(base_url).rstrip('/')}/stats",
        "population_url": f"{str(base_url).rstrip('/')}/population",
        "paused": bool(control_payload.get("paused", False)),
        "current_generation": _safe_int(control_payload.get("current_generation", population.get("current_generation", 0)), 0),
        "population_size": _safe_int(control_payload.get("population_size", population.get("size", 0)), 0),
        "best_eval_fitness": float(population.get("best_eval_fitness", 0.0) or 0.0),
        "avg_eval_fitness": float(population.get("avg_eval_fitness", 0.0) or 0.0),
        "rollout_steps": _safe_int(generation_metrics.get("rollout/steps_collected", 0), 0),
        "ppo_agents_optimized": _safe_int(generation_metrics.get("ppo/agents_optimized", 0), 0),
        "tournaments_active": _safe_int(tournaments.get("active", 0), 0),
        "planned_games": _safe_int(tournaments.get("planned_games", 0), 0),
        "finished_games": _safe_int(tournaments.get("finished_games", 0), 0),
        "completion_rate": float(tournaments.get("completion_rate", 0.0) or 0.0),
        "total_servers": _safe_int(servers.get("total_servers", 0), 0),
        "healthy_servers": _safe_int(servers.get("healthy_servers", 0), 0),
        "active_games": _safe_int(servers.get("total_active_games", 0), 0),
        "error": str(error or ""),
        "stats": stats_payload,
        "control": control_payload,
    }


def _load_orchestrator_snapshot() -> Dict[str, Any]:
    output_root = os.path.abspath(
        str(
            os.getenv("GLOBAL_DASHBOARD_ORCH_OUTPUT_ROOT", "")
            or os.getenv("ORCH_OUTPUT_ROOT", "")
            or "/app/rl-models-global"
        ).strip()
    )
    state = _load_json_file(os.path.join(output_root, "orchestrator_state.json"))
    manifest = _load_json_file(os.path.join(output_root, "champion", "current", "champion_manifest.json"))
    winner = dict(manifest.get("winner", {}) or {})
    metrics = dict(winner.get("metrics", {}) or {})
    return {
        "output_root": output_root,
        "state": state,
        "manifest": manifest,
        "summary": {
            "status": str(state.get("status", "unknown") or "unknown"),
            "updated_at": str(state.get("updated_at", "") or ""),
            "last_round_id": str(state.get("last_round_id", "") or ""),
            "last_error": str(state.get("last_error", "") or ""),
            "winner_candidate_id": str(winner.get("candidate_id", "") or ""),
            "winner_coordinator_id": str(winner.get("coordinator_id", "") or ""),
            "winner_generation": _safe_int(winner.get("generation", -1), -1),
            "winner_win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
            "winner_avg_vp": float(metrics.get("avg_vp", 0.0) or 0.0),
            "promotion_applied": bool(dict(manifest.get("promotion", {}) or {}).get("applied", False)),
            "promotion_reason": str(dict(manifest.get("promotion", {}) or {}).get("reason", "") or ""),
            "coord_progress": dict(state.get("coord_progress", {}) or {}),
        },
    }


async def _collect_global_dashboard_payload() -> Dict[str, Any]:
    sources = _global_dashboard_sources()
    current_coord_id = str(getattr(coordinator, "coordinator_id", "") or "coord-1")
    if not sources and coordinator is not None:
        sources = [(current_coord_id, "http://127.0.0.1:5000")]

    local_stats = await _collect_local_stats_payload()
    local_control = _collect_local_control_status_payload()
    coordinators_payload: List[Dict[str, Any]] = []

    timeout = aiohttp.ClientTimeout(total=4.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        remote_tasks = []
        remote_meta: List[Tuple[str, str]] = []
        for coord_id, base_url in sources:
            normalized_base = str(base_url).rstrip("/")
            if str(coord_id) == current_coord_id:
                coordinators_payload.append(
                    _coordinator_summary(
                        coord_id=coord_id,
                        base_url=normalized_base,
                        stats_payload=local_stats,
                        control_payload=local_control,
                    )
                )
                continue
            remote_meta.append((coord_id, normalized_base))
            remote_tasks.append(
                asyncio.gather(
                    _fetch_remote_json(session, f"{normalized_base}/stats"),
                    _fetch_remote_json(session, f"{normalized_base}/control/status"),
                )
            )

        if remote_tasks:
            remote_results = await asyncio.gather(*remote_tasks, return_exceptions=True)
            for idx, result in enumerate(remote_results):
                coord_id, base_url = remote_meta[idx]
                if isinstance(result, Exception):
                    coordinators_payload.append(
                        _coordinator_summary(
                            coord_id=coord_id,
                            base_url=base_url,
                            stats_payload={},
                            control_payload={},
                            error=f"{type(result).__name__}: {result}",
                        )
                    )
                    continue
                stats_payload, control_payload = result
                error_bits = []
                if "error" in stats_payload:
                    error_bits.append(str(stats_payload.get("error")))
                if "error" in control_payload:
                    error_bits.append(str(control_payload.get("error")))
                coordinators_payload.append(
                    _coordinator_summary(
                        coord_id=coord_id,
                        base_url=base_url,
                        stats_payload=stats_payload if "error" not in stats_payload else {},
                        control_payload=control_payload if "error" not in control_payload else {},
                        error=" | ".join(error_bits),
                    )
                )

    coordinators_payload.sort(key=lambda item: str(item.get("coordinator_id", "")))
    generations = [int(item.get("current_generation", 0) or 0) for item in coordinators_payload if not item.get("error")]
    orchestrator = _load_orchestrator_snapshot()

    return {
        "generated_at": _utc_now_iso(),
        "summary": {
            "coordinator_count": len(coordinators_payload),
            "paused_count": sum(1 for item in coordinators_payload if bool(item.get("paused", False))),
            "generation_min": min(generations) if generations else 0,
            "generation_max": max(generations) if generations else 0,
            "generation_spread": (max(generations) - min(generations)) if len(generations) >= 2 else 0,
            "population_total": sum(_safe_int(item.get("population_size", 0), 0) for item in coordinators_payload),
            "active_tournaments_total": sum(_safe_int(item.get("tournaments_active", 0), 0) for item in coordinators_payload),
            "active_games_total": sum(_safe_int(item.get("active_games", 0), 0) for item in coordinators_payload),
            "healthy_servers_total": sum(_safe_int(item.get("healthy_servers", 0), 0) for item in coordinators_payload),
            "servers_total": sum(_safe_int(item.get("total_servers", 0), 0) for item in coordinators_payload),
        },
        "coordinators": coordinators_payload,
        "orchestrator": orchestrator,
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {"message": "Terraforming Mars RL Training Monitor", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return {
        "status": "healthy",
        "coordinator": "running",
        "generation": coordinator.current_generation,
        "population_size": len(coordinator.population),
        "coordinator_id": str(getattr(coordinator, "coordinator_id", "") or "coord-1"),
    }

@app.get("/stats")
async def get_stats():
    """Get current training statistics"""
    try:
        return await _collect_local_stats_payload()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/population")
async def get_population():
    """Get current population details"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    population_data = []
    for agent in coordinator.population:
        agent_data = agent.get_config()
        cfg = agent_data.get("config", {})
        behavior_stats = agent.get_behavior_stats() if hasattr(agent, "get_behavior_stats") else {}
        # Show both lifetime fitness (agent.get_fitness_score) and last eval fitness used for selection
        last_eval = coordinator.last_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_eval_fitness') else None
        raw_eval = coordinator.last_raw_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_raw_eval_fitness') else None
        gate_per_agent = dict((getattr(coordinator, "last_generation_gate", {}) or {}).get("per_agent", {}) or {})
        gate_info = gate_per_agent.get(agent.id, {})
        gateBehavior = gate_info.get("behavior", {})
        agent_data.update({
            "fitness_score": agent.get_fitness_score(),
            "last_eval_fitness": last_eval if last_eval is not None else agent.get_fitness_score(),
            "last_raw_eval_fitness": raw_eval if raw_eval is not None else agent.get_fitness_score(),
            "win_rate": agent.wins / agent.games_played if agent.games_played > 0 else 0,
            "avg_vp": agent.total_victory_points / agent.games_played if agent.games_played > 0 else 0,
            "learning_rate": float(cfg.get("learning_rate", 0.0)),
            "epsilon": float(cfg.get("epsilon", 0.0)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "elo": coordinator.metrics_tracker.get_elo(agent.id),
            "efficiency": float(gateBehavior.get("efficiency_ratio", 0.0)),
            "synergy": float(gateBehavior.get("synergy_score", 0.0)),
            "behavior": behavior_stats,
            "policy_success_rate": float(behavior_stats.get("policy_success_rate", 0.0)),
            "epsilon_random_rate": float(behavior_stats.get("epsilon_random_rate", 0.0)),
            "fallback_pass_rate": float(behavior_stats.get("fallback_pass_rate", 0.0)),
            "total_decisions": int(behavior_stats.get("total_decisions", 0)),
            "promotion_gate": gate_info,
        })
        population_data.append(agent_data)
    
    # Sort by last evaluation fitness (matches selection and logs), fallback to lifetime
    population_data.sort(key=lambda x: x.get('last_eval_fitness', x['fitness_score']), reverse=True)
    
    return {
        "generation": coordinator.current_generation,
        "population": population_data
    }

@app.get("/tournaments")
async def get_tournaments():
    """Get active tournament information"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    active_tournaments = []
    for tournament_id, tournament in coordinator.tournament_manager.active_tournaments.items():
        tournament_info = await coordinator.tournament_manager.get_tournament_status(tournament_id)
        active_tournaments.append(tournament_info)
    progress = coordinator.tournament_manager.get_progress_snapshot()

    # NOTE: Completed tournament results (with game end-screen URLs) are returned by the coordinator during evaluations.
    return {
        "active_tournaments": active_tournaments,
        "total_active": len(active_tournaments),
        "progress": progress,
    }

@app.get("/servers")
async def get_server_status():
    """Get game server status"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    health_status = await coordinator.game_cluster.health_check()
    server_stats = await coordinator.game_cluster.get_server_stats()
    
    # Try to include recent created game URLs
    recent_games = []
    try:
        recent_games = getattr(coordinator.game_cluster, 'recent_games', [])[-20:]
    except Exception:
        recent_games = []
    recent_games = _sanitize_recent_games(recent_games)

    return {
        "health": health_status,
        "stats": server_stats,
        "recent_games": recent_games
    }

@app.post("/control/pause")
async def pause_training():
    """Pause training"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    coordinator.pause_training()
    return {"message": "Training paused", "paused": True}

@app.post("/control/resume")
async def resume_training():
    """Resume training"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    coordinator.resume_training()
    return {"message": "Training resumed", "paused": False}

@app.get("/control/status")
async def get_training_status():
    """Get current training status"""
    return _collect_local_control_status_payload()


@app.get("/global/stats")
async def get_global_stats():
    """Aggregate multi-coordinator and orchestrator state for the global dashboard."""
    try:
        return await _collect_global_dashboard_payload()
    except Exception as e:
        logger.error(f"Error getting global stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/game")
async def run_debug_game(request: DebugGameRequest):
    """Run a single debug game with 4 agents"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    if len(coordinator.population) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 agents in population")
    
    result = await coordinator.run_debug_game(request.agent_ids)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result


@app.post("/debug/decision-snapshots/capture")
async def capture_decision_snapshot(request: DecisionSnapshotCaptureRequest):
    capture_request = create_capture_request(
        agent_id=request.agent_id,
        game_id=request.game_id,
        player_id=request.player_id,
        note=request.note,
        include_state_vector=bool(request.include_state_vector),
    )
    return {
        "message": "Decision snapshot request armed",
        "request": capture_request,
    }


@app.get("/debug/decision-snapshots")
async def get_decision_snapshots():
    return {
        "requests": list_capture_requests(),
        "snapshots": list_saved_snapshots(),
    }


@app.get("/debug/decision-snapshots/{snapshot_id}")
async def get_decision_snapshot(snapshot_id: str):
    try:
        return load_snapshot(snapshot_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/play/human-vs-generation")
async def play_human_vs_generation(request: HumanVsGenerationRequest):
    """Start a game with one human and saved-generation AI opponents."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    if request.bot_count < 1 or request.bot_count > 3:
        raise HTTPException(status_code=400, detail="bot_count must be between 1 and 3")
    if not request.random_generation and request.generation is None:
        raise HTTPException(status_code=400, detail="generation is required when random_generation=false")

    result = await coordinator.run_human_vs_generation_game(
        generation=request.generation,
        random_generation=request.random_generation,
        human_name=request.human_name,
        bot_count=request.bot_count,
        agent_indices=request.agent_indices,
        seed=request.seed,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/play/human-vs-best")
async def play_human_vs_best(request: HumanVsBestRequest):
    """Start a game with one human against the best saved agent checkpoint."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    if request.bot_count < 1 or request.bot_count > 3:
        raise HTTPException(status_code=400, detail="bot_count must be between 1 and 3")

    result = await coordinator.run_human_vs_best_agent_game(
        human_name=request.human_name,
        bot_count=request.bot_count,
        seed=request.seed,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/decision-explainer", response_class=HTMLResponse)
async def decision_explainer(request: Request):
    return templates.TemplateResponse(
        "decision_explainer.html",
        {
            "request": request,
            "snapshot_id": str(request.query_params.get("snapshot_id", "") or "").strip(),
        },
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render global dashboard by default in multi-coordinator setups; local otherwise."""
    try:
        auto_refresh_ms = max(1000, int(os.getenv("DASHBOARD_REFRESH_MS", "10000")))
    except Exception:
        auto_refresh_ms = 10000

    use_global = _global_dashboard_enabled() and str(request.query_params.get("view", "") or "").strip().lower() != "local"
    template_name = "dashboard_global.html" if use_global else "dashboard.html"
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "auto_refresh_ms": auto_refresh_ms,
            "global_dashboard_enabled": bool(_global_dashboard_enabled()),
        },
    )


@app.get("/dashboard/local", response_class=HTMLResponse)
async def dashboard_local(request: Request):
    """Always render the local coordinator dashboard."""
    try:
        auto_refresh_ms = max(1000, int(os.getenv("DASHBOARD_REFRESH_MS", "10000")))
    except Exception:
        auto_refresh_ms = 10000
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "auto_refresh_ms": auto_refresh_ms,
            "global_dashboard_enabled": bool(_global_dashboard_enabled()),
        },
    )


@app.get("/dashboard/global", response_class=HTMLResponse)
async def dashboard_global(request: Request):
    """Always render the aggregate multi-coordinator dashboard."""
    try:
        auto_refresh_ms = max(1000, int(os.getenv("DASHBOARD_REFRESH_MS", "10000")))
    except Exception:
        auto_refresh_ms = 10000
    return templates.TemplateResponse(
        "dashboard_global.html",
        {
            "request": request,
            "auto_refresh_ms": auto_refresh_ms,
            "global_dashboard_enabled": bool(_global_dashboard_enabled()),
        },
    )

async def start_api_server(coordinator_instance):
    """Start the API server with coordinator reference"""
    global coordinator
    coordinator = coordinator_instance
    
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=5000, 
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

