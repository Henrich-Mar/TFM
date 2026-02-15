"""
API Server for monitoring RL training progress
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from typing import Dict, Any, Optional, List
import uvicorn
import os
from urllib.parse import urlsplit, urlunsplit

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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        "population_size": len(coordinator.population)
    }

@app.get("/stats")
async def get_stats():
    """Get current training statistics"""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
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
        
        # Gather recent tournament/game end screens (shallow aggregation)
        recent_end_screens = []
        try:
            # We don't persist tournaments; pull from the most recent evaluation run results if coordinator cached any
            pass
        except Exception:
            recent_end_screens = []

        # Recently created games (if coordinator or cluster recorded any)
        recent_games = []
        try:
            for g in getattr(coordinator, 'recent_games', [])[-20:]:
                recent_games.append(g)
        except Exception:
            recent_games = []
        # Fallback to cluster scratchpad if coordinator list is empty
        if not recent_games:
            try:
                recent_games = getattr(coordinator.game_cluster, 'recent_games', [])[-20:]
            except Exception:
                recent_games = []
        recent_games = _sanitize_recent_games(recent_games)

        # Recently completed end-screen URLs
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
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return {
        "paused": coordinator.is_paused(),
        "current_generation": coordinator.current_generation,
        "population_size": len(coordinator.population)
    }

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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render dashboard from a standalone HTML template."""
    try:
        auto_refresh_ms = max(1000, int(os.getenv("DASHBOARD_REFRESH_MS", "5000")))
    except Exception:
        auto_refresh_ms = 5000

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "auto_refresh_ms": auto_refresh_ms,
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

