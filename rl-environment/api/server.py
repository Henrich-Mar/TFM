"""
API Server for monitoring RL training progress
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from typing import Dict, Any, Optional, List
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(title="Terraforming Mars RL Monitor", version="1.0.0")

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
    }
    action_counts: Dict[str, int] = {}
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
        "action_counts": action_counts,
        "action_mix": action_mix,
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
        for agent in coordinator.population:
            eval_score = coordinator.last_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_eval_fitness') else None
            eval_fitness_scores.append(float(eval_score) if eval_score is not None else float(agent.get_fitness_score()))

        population_stats = {
            "size": len(coordinator.population),
            "current_generation": coordinator.current_generation,
            "total_games_played": sum(agent.games_played for agent in coordinator.population),
            "best_fitness": max(agent.get_fitness_score() for agent in coordinator.population) if coordinator.population else 0,
            "avg_fitness": sum(agent.get_fitness_score() for agent in coordinator.population) / len(coordinator.population) if coordinator.population else 0,
            "best_eval_fitness": max(eval_fitness_scores) if eval_fitness_scores else 0,
            "avg_eval_fitness": _safe_mean(eval_fitness_scores),
        }
        behavior_stats = _aggregate_behavior_stats(coordinator.population)
        
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

        # Recently completed end-screen URLs
        recent_end = []
        try:
            recent_end = getattr(coordinator, 'recent_end_screens', [])[-20:]
        except Exception:
            recent_end = []

        return {
            "population": population_stats,
            "behavior": behavior_stats,
            "servers": server_stats,
            "evolution": evolution_stats,
            "tournaments": {
                "active": len(coordinator.tournament_manager.active_tournaments)
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
        agent_data.update({
            "fitness_score": agent.get_fitness_score(),
            "last_eval_fitness": last_eval if last_eval is not None else agent.get_fitness_score(),
            "win_rate": agent.wins / agent.games_played if agent.games_played > 0 else 0,
            "avg_vp": agent.total_victory_points / agent.games_played if agent.games_played > 0 else 0,
            "learning_rate": float(cfg.get("learning_rate", 0.0)),
            "epsilon": float(cfg.get("epsilon", 0.0)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "behavior": behavior_stats,
            "policy_success_rate": float(behavior_stats.get("policy_success_rate", 0.0)),
            "epsilon_random_rate": float(behavior_stats.get("epsilon_random_rate", 0.0)),
            "fallback_pass_rate": float(behavior_stats.get("fallback_pass_rate", 0.0)),
            "total_decisions": int(behavior_stats.get("total_decisions", 0)),
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

    # NOTE: Completed tournament results (with game end-screen URLs) are returned by the coordinator during evaluations.
    return {
        "active_tournaments": active_tournaments,
        "total_active": len(active_tournaments)
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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """HTML dashboard focused on RL behavior and training diagnostics."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Terraforming Mars RL Training Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {
                --bg: #f3f5f7;
                --panel: #ffffff;
                --ink: #1b2330;
                --muted: #5c697a;
                --accent: #0b6e4f;
                --warn: #ad6a00;
                --danger: #b02a37;
                --border: #dbe2ea;
            }

            * { box-sizing: border-box; }

            body {
                margin: 0;
                background: radial-gradient(1200px 400px at 80% -10%, #e8f5f0 0%, var(--bg) 55%);
                color: var(--ink);
                font-family: "Segoe UI", "Trebuchet MS", Tahoma, sans-serif;
            }

            .shell {
                max-width: 1360px;
                margin: 0 auto;
                padding: 24px 18px 40px;
            }

            .topbar {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 14px;
            }

            h1, h2, h3 {
                margin: 0;
                font-weight: 700;
            }

            .subtitle {
                margin-top: 6px;
                color: var(--muted);
                font-size: 14px;
            }

            .actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            button {
                border: 1px solid #0a4f3a;
                background: var(--accent);
                color: white;
                padding: 10px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            }

            button.secondary {
                background: #0f4c81;
                border-color: #0f4c81;
            }

            button.warn {
                background: #9b4400;
                border-color: #9b4400;
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            .status-line {
                margin-top: 8px;
                color: var(--muted);
                font-size: 13px;
            }

            .grid {
                display: grid;
                gap: 14px;
            }

            .grid.cols-3 {
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            }

            .grid.cols-2 {
                grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            }

            .card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px;
                box-shadow: 0 3px 10px rgba(10, 25, 40, 0.05);
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 8px;
                margin-top: 10px;
            }

            .metric {
                border: 1px solid var(--border);
                border-radius: 9px;
                padding: 8px;
                background: #f8fafc;
            }

            .metric-value {
                font-size: 24px;
                font-weight: 700;
                color: var(--ink);
            }

            .metric-label {
                font-size: 12px;
                color: var(--muted);
                margin-top: 2px;
            }

            .metric-sub {
                margin-top: 3px;
                font-size: 12px;
                color: var(--muted);
            }

            .good {
                color: var(--accent);
                font-weight: 700;
            }

            .bad {
                color: var(--danger);
                font-weight: 700;
            }

            .hint-list {
                margin: 10px 0 0 18px;
                padding: 0;
                color: var(--ink);
                line-height: 1.38;
                font-size: 14px;
            }

            .hint-list li { margin-bottom: 7px; }

            .mono { font-family: Consolas, "Courier New", monospace; }

            .kv-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
                font-size: 14px;
            }

            .kv-table th,
            .kv-table td {
                padding: 7px 8px;
                border-bottom: 1px solid var(--border);
            }

            .kv-table th {
                text-align: left;
                color: var(--muted);
                font-weight: 600;
                white-space: nowrap;
            }

            .kv-table td { text-align: right; }

            .bar-list { margin-top: 8px; }

            .bar-row {
                display: grid;
                grid-template-columns: 110px 1fr 64px;
                align-items: center;
                gap: 8px;
                margin: 6px 0;
            }

            .bar-name {
                color: var(--muted);
                font-size: 12px;
                text-transform: capitalize;
            }

            .bar-track {
                height: 10px;
                border-radius: 8px;
                background: #e4ecf3;
                overflow: hidden;
            }

            .bar-fill {
                height: 100%;
                background: linear-gradient(90deg, #0b6e4f, #18a06f);
            }

            .bar-value {
                text-align: right;
                font-size: 12px;
                color: var(--muted);
                font-variant-numeric: tabular-nums;
            }

            .trend-wrap {
                margin-top: 10px;
                border: 1px solid var(--border);
                border-radius: 10px;
                background: #fcfdff;
                padding: 6px;
                overflow-x: auto;
            }

            .legend {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                font-size: 12px;
                color: var(--muted);
                margin-top: 6px;
            }

            .dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 4px;
                vertical-align: middle;
            }

            .list {
                margin: 8px 0 0 18px;
                padding: 0;
            }

            .list li { margin-bottom: 4px; }

            .table-wrap {
                overflow-x: auto;
                margin-top: 10px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                min-width: 980px;
                font-size: 13px;
            }

            th,
            td {
                border-bottom: 1px solid var(--border);
                padding: 8px;
                text-align: right;
                font-variant-numeric: tabular-nums;
            }

            th {
                background: #f4f8fb;
                color: var(--muted);
                font-weight: 600;
            }

            td.left,
            th.left { text-align: left; }

            .empty {
                color: var(--muted);
                font-size: 14px;
                margin-top: 8px;
            }

            @media (max-width: 760px) {
                .bar-row {
                    grid-template-columns: 90px 1fr 56px;
                }
            }
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="topbar">
                <div>
                    <h1>Terraforming Mars RL Dashboard</h1>
                    <div class="subtitle">Use behavior telemetry to detect when agents are exploring too much, failing policy actions, or defaulting to pass.</div>
                    <div id="control-status" class="status-line">Loading status...</div>
                    <div id="human-link-status" class="status-line"></div>
                </div>
                <div class="actions">
                    <button class="secondary" onclick="refreshAll()">Refresh</button>
                    <button id="pause-btn" class="warn" onclick="pauseTraining()">Pause Training</button>
                    <button id="resume-btn" class="secondary" onclick="resumeTraining()">Resume Training</button>
                    <button id="human-vs-btn" class="secondary" onclick="startHumanVsGeneration()">Play Vs Generation</button>
                    <button id="debug-btn" onclick="runDebugGame()">Run Debug Game</button>
                </div>
            </div>

            <div class="grid cols-3">
                <div class="card">
                    <h3>Training</h3>
                    <div id="training-stats" class="metric-grid">Loading...</div>
                </div>
                <div class="card">
                    <h3>RL Behavior</h3>
                    <div id="behavior-stats" class="metric-grid">Loading...</div>
                </div>
                <div class="card">
                    <h3>Game Servers</h3>
                    <div id="server-stats" class="metric-grid">Loading...</div>
                </div>
            </div>

            <div class="grid cols-2" style="margin-top: 14px;">
                <div class="card">
                    <h3>Fitness and Diversity Trend</h3>
                    <div class="trend-wrap" id="trend-chart">Loading...</div>
                    <div class="legend" id="trend-meta"></div>
                </div>
                <div class="card">
                    <h3>Steering Hints</h3>
                    <div id="steering-hints" class="empty">Loading...</div>
                    <h3 style="margin-top: 14px;">Exploration Spread</h3>
                    <div id="hyperparams" class="empty">Loading...</div>
                    <h3 style="margin-top: 14px;">Action Mix</h3>
                    <div id="action-mix" class="empty">Loading...</div>
                </div>
            </div>

            <div class="grid cols-2" style="margin-top: 14px;">
                <div class="card">
                    <h3>Recent Games</h3>
                    <div id="recent-games" class="empty">Loading...</div>
                </div>
                <div class="card">
                    <h3>Recent End Screens</h3>
                    <div id="recent-ends" class="empty">Loading...</div>
                </div>
            </div>

            <div class="card" style="margin-top: 14px;">
                <h3>Top Agents (Behavior View)</h3>
                <div id="top-agents" class="table-wrap">Loading...</div>
            </div>
        </div>

        <script>
            const AUTO_REFRESH_MS = 30000;

            function fmt(value, digits = 2) {
                const n = Number(value ?? 0);
                return Number.isFinite(n) ? n.toFixed(digits) : (0).toFixed(digits);
            }

            function pct(value, digits = 1) {
                return `${fmt((Number(value ?? 0) * 100), digits)}%`;
            }

            function metricCard(label, value, sub = '') {
                return `
                    <div class="metric">
                        <div class="metric-value">${value}</div>
                        <div class="metric-label">${label}</div>
                        ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
                    </div>
                `;
            }

            function updateTrainingCards(stats) {
                const p = stats.population || {};
                const t = stats.tournaments || {};
                const html = [
                    metricCard('Generation', fmt(p.current_generation || 0, 0)),
                    metricCard('Population', fmt(p.size || 0, 0)),
                    metricCard('Best Eval Fitness', fmt(p.best_eval_fitness || 0, 2)),
                    metricCard('Avg Eval Fitness', fmt(p.avg_eval_fitness || 0, 2)),
                    metricCard('Total Games', fmt(p.total_games_played || 0, 0)),
                    metricCard('Active Tournaments', fmt(t.active || 0, 0)),
                ].join('');
                document.getElementById('training-stats').innerHTML = html;
            }

            function updateBehaviorCards(stats) {
                const b = stats.behavior || {};
                const html = [
                    metricCard('Policy Success', pct(b.policy_success_rate || 0), `${fmt(b.policy_successes || 0, 0)}/${fmt(b.policy_attempts || 0, 0)} accepted`),
                    metricCard('Epsilon Random Rate', pct(b.epsilon_random_rate || 0), `${fmt(b.epsilon_random_actions || 0, 0)} sampled`),
                    metricCard('Fallback Decision Rate', pct(b.fallback_decision_rate || 0), `${fmt(b.fallback_decisions || 0, 0)} turns`),
                    metricCard('Fallback Pass Rate', pct(b.fallback_pass_rate || 0), `${fmt(b.fallback_passes || 0, 0)} forced passes`),
                    metricCard('Avg Available Actions', fmt(b.avg_available_actions || 0, 1), 'signal breadth per decision'),
                    metricCard('Total Decisions', fmt(b.total_decisions || 0, 0)),
                ].join('');
                document.getElementById('behavior-stats').innerHTML = html;
            }

            function updateServerCards(stats) {
                const s = stats.servers || {};
                const healthy = Number(s.healthy_servers || 0);
                const total = Number(s.total_servers || 0);
                const healthyClass = total > 0 && healthy === total ? 'good' : 'bad';
                const html = [
                    metricCard('Healthy Servers', `<span class="${healthyClass}">${healthy}/${total}</span>`),
                    metricCard('Active Games', fmt(s.total_active_games || 0, 0)),
                    metricCard('Queued Games', fmt(s.total_queued_games || 0, 0)),
                ].join('');
                document.getElementById('server-stats').innerHTML = html;
            }

            function renderList(containerId, items, emptyLabel) {
                if (!items.length) {
                    document.getElementById(containerId).innerHTML = `<div class="empty">${emptyLabel}</div>`;
                    return;
                }

                const html = `<ol class="list">${items.map(item => `<li>${item}</li>`).join('')}</ol>`;
                document.getElementById(containerId).innerHTML = html;
            }

            function updateRecentLists(stats) {
                const recentGames = (stats.recent_games || []).slice(-10).reverse().map((g) => {
                    if (!g || !g.url) return `<span class="mono">${(g && g.game_id) ? g.game_id : 'unknown'}</span>`;
                    return `<a href="${g.url}" target="_blank" rel="noopener noreferrer"><span class="mono">${g.game_id}</span></a>`;
                });

                const recentEnds = (stats.recent_end_screens || []).slice(-10).reverse().map((e) => {
                    const url = (e && e.end_screens && e.end_screens.length) ? e.end_screens[0] : null;
                    if (!url) return `<span class="mono">${(e && e.game_id) ? e.game_id : 'unknown'}</span>`;
                    return `<a href="${url}" target="_blank" rel="noopener noreferrer"><span class="mono">${e.game_id}</span></a>`;
                });

                renderList('recent-games', recentGames, 'No recent games');
                renderList('recent-ends', recentEnds, 'No completed games yet');
            }

            function updateHyperparameterTable(stats) {
                const hp = (stats.behavior || {}).hyperparameters || {};
                const rows = [
                    ['Epsilon', hp.epsilon || {}],
                    ['Temperature', hp.temperature || {}],
                    ['Learning Rate', hp.learning_rate || {}],
                ];

                const html = `
                    <table class="kv-table">
                        <thead>
                            <tr>
                                <th>Hyperparameter</th>
                                <th>Min</th>
                                <th>Mean</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${rows.map(([name, values]) => `
                                <tr>
                                    <th>${name}</th>
                                    <td>${fmt(values.min || 0, name === 'Learning Rate' ? 5 : 3)}</td>
                                    <td>${fmt(values.mean || 0, name === 'Learning Rate' ? 5 : 3)}</td>
                                    <td>${fmt(values.max || 0, name === 'Learning Rate' ? 5 : 3)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;

                document.getElementById('hyperparams').innerHTML = html;
            }

            function humanizeActionName(name) {
                return String(name || '')
                    .replace(/_/g, ' ')
                    .replace(/\b\w/g, (c) => c.toUpperCase());
            }

            function updateActionMix(stats) {
                const mix = (stats.behavior || {}).action_mix || {};
                const entries = Object.entries(mix);

                if (!entries.length) {
                    document.getElementById('action-mix').innerHTML = '<div class="empty">No action telemetry yet</div>';
                    return;
                }

                const rows = entries.map(([name, ratio]) => {
                    const width = Math.max(2, Number(ratio || 0) * 100);
                    return `
                        <div class="bar-row">
                            <div class="bar-name">${humanizeActionName(name)}</div>
                            <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
                            <div class="bar-value">${pct(ratio, 1)}</div>
                        </div>
                    `;
                }).join('');

                document.getElementById('action-mix').innerHTML = `<div class="bar-list">${rows}</div>`;
            }

            function buildPath(values, width, height, padding) {
                if (!values.length) return '';
                const min = Math.min(...values);
                const max = Math.max(...values);
                const range = (max - min) || 1;
                const denominator = Math.max(1, values.length - 1);
                return values.map((value, index) => {
                    const x = padding + (index / denominator) * (width - 2 * padding);
                    const y = height - padding - ((value - min) / range) * (height - 2 * padding);
                    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
                }).join(' ');
            }

            function updateTrend(stats) {
                const evolution = stats.evolution || {};
                const fitnessHistory = (evolution.fitness_history || []).slice(-50);
                const diversityHistory = (evolution.diversity_history || []).slice(-50);
                const width = 920;
                const height = 250;
                const padding = 22;

                if (!fitnessHistory.length) {
                    document.getElementById('trend-chart').innerHTML = '<div class="empty">Trend will appear after at least one evaluated generation.</div>';
                    document.getElementById('trend-meta').innerHTML = '';
                    return;
                }

                const meanFitness = fitnessHistory.map((x) => Number(x.mean || 0));
                const bestFitness = fitnessHistory.map((x) => Number(x.max || 0));
                const stdFitness = fitnessHistory.map((x) => Number(x.std || 0));
                const diversity = diversityHistory.map((x) => Number(x || 0));

                const meanPath = buildPath(meanFitness, width, height, padding);
                const bestPath = buildPath(bestFitness, width, height, padding);
                const stdPath = buildPath(stdFitness, width, height, padding);
                const divPath = buildPath(diversity, width, height, padding);

                const gridLines = [0.2, 0.4, 0.6, 0.8].map((ratio) => {
                    const y = (padding + ratio * (height - 2 * padding)).toFixed(1);
                    return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="#e4ebf2" stroke-width="1" />`;
                }).join('');

                const svg = `
                    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Fitness and diversity trend">
                        <rect x="0" y="0" width="${width}" height="${height}" fill="#fcfdff" />
                        ${gridLines}
                        ${meanPath ? `<path d="${meanPath}" fill="none" stroke="#0b6e4f" stroke-width="2.2" />` : ''}
                        ${bestPath ? `<path d="${bestPath}" fill="none" stroke="#1f4e99" stroke-width="2.2" />` : ''}
                        ${stdPath ? `<path d="${stdPath}" fill="none" stroke="#8a4d00" stroke-width="2" stroke-dasharray="4 3" />` : ''}
                        ${divPath ? `<path d="${divPath}" fill="none" stroke="#8b2bb8" stroke-width="2" />` : ''}
                        <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#b8c5d3" stroke-width="1.2" />
                    </svg>
                `;

                document.getElementById('trend-chart').innerHTML = svg;
                document.getElementById('trend-meta').innerHTML = `
                    <span><span class="dot" style="background:#0b6e4f;"></span>Mean fitness (auto-scaled)</span>
                    <span><span class="dot" style="background:#1f4e99;"></span>Best fitness (auto-scaled)</span>
                    <span><span class="dot" style="background:#8a4d00;"></span>Fitness stddev (auto-scaled)</span>
                    <span><span class="dot" style="background:#8b2bb8;"></span>Diversity (auto-scaled)</span>
                `;
            }

            function updateSteeringHints(stats) {
                const hints = [];
                const b = stats.behavior || {};
                const p = stats.population || {};
                const evolution = stats.evolution || {};
                const history = evolution.fitness_history || [];

                if ((b.total_decisions || 0) < 50) {
                    hints.push('Low decision count. Run more games before tuning hyperparameters aggressively.');
                }

                if ((b.epsilon_random_rate || 0) > 0.22) {
                    hints.push(`High epsilon-driven randomness (${pct(b.epsilon_random_rate || 0)}). Consider lowering epsilon range for newly created agents.`);
                }

                if ((b.policy_attempts || 0) >= 20 && (b.policy_success_rate || 0) < 0.35) {
                    hints.push(`Policy actions are frequently rejected (${pct(1 - (b.policy_success_rate || 0))} rejection). Focus on action decoder validity for common waiting states.`);
                }

                if ((b.fallback_pass_rate || 0) > 0.15) {
                    hints.push(`Pass fallback is high (${pct(b.fallback_pass_rate || 0)}). This often means the policy cannot find valid productive actions.`);
                }

                if (history.length >= 12) {
                    const prev = history.slice(-12, -6).map((x) => Number(x.mean || 0));
                    const now = history.slice(-6).map((x) => Number(x.mean || 0));
                    const prevMean = prev.length ? prev.reduce((a, b) => a + b, 0) / prev.length : 0;
                    const nowMean = now.length ? now.reduce((a, b) => a + b, 0) / now.length : 0;
                    if (nowMean <= (prevMean * 1.02)) {
                        hints.push('Average fitness appears flat over the last 12 generations. Consider stronger mutation or more self-play learning signal.');
                    }
                }

                if ((p.best_eval_fitness || 0) > 0 && (b.policy_success_rate || 0) > 0.55 && (b.fallback_pass_rate || 0) < 0.08) {
                    hints.push('Behavior signal is healthy: policy acceptance is strong and forced pass fallback is low.');
                }

                const content = hints.length
                    ? `<ol class="hint-list">${hints.map((h) => `<li>${h}</li>`).join('')}</ol>`
                    : '<div class="empty">No warnings yet. Keep running and watch trend + behavior rates.</div>';

                document.getElementById('steering-hints').innerHTML = content;
            }

            async function loadStats() {
                const response = await fetch('/stats');
                if (!response.ok) throw new Error(`Failed /stats (${response.status})`);
                const stats = await response.json();

                updateTrainingCards(stats);
                updateBehaviorCards(stats);
                updateServerCards(stats);
                updateRecentLists(stats);
                updateHyperparameterTable(stats);
                updateActionMix(stats);
                updateTrend(stats);
                updateSteeringHints(stats);
            }

            async function loadPopulation() {
                const response = await fetch('/population');
                if (!response.ok) throw new Error(`Failed /population (${response.status})`);
                const data = await response.json();
                const topAgents = (data.population || []).slice(0, 12);

                if (!topAgents.length) {
                    document.getElementById('top-agents').innerHTML = '<div class="empty">No agents available</div>';
                    return;
                }

                const rows = topAgents.map((agent) => `
                    <tr>
                        <td class="left mono">${String(agent.id || '').slice(0, 10)}</td>
                        <td>${fmt(agent.last_eval_fitness || 0, 2)}</td>
                        <td>${fmt(agent.fitness_score || 0, 2)}</td>
                        <td>${fmt(agent.games_played || 0, 0)}</td>
                        <td>${pct(agent.win_rate || 0, 1)}</td>
                        <td>${fmt(agent.avg_vp || 0, 1)}</td>
                        <td>${fmt(agent.epsilon || 0, 3)}</td>
                        <td>${fmt(agent.temperature || 0, 3)}</td>
                        <td>${fmt(agent.learning_rate || 0, 5)}</td>
                        <td>${fmt(agent.total_decisions || 0, 0)}</td>
                        <td>${pct(agent.policy_success_rate || 0, 1)}</td>
                        <td>${pct(agent.epsilon_random_rate || 0, 1)}</td>
                        <td>${pct(agent.fallback_pass_rate || 0, 1)}</td>
                    </tr>
                `).join('');

                document.getElementById('top-agents').innerHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th class="left">Agent</th>
                                <th>Eval Fitness</th>
                                <th>Lifetime Fitness</th>
                                <th>Games</th>
                                <th>Win Rate</th>
                                <th>Avg VP</th>
                                <th>Epsilon</th>
                                <th>Temperature</th>
                                <th>Learning Rate</th>
                                <th>Decisions</th>
                                <th>Policy Success</th>
                                <th>Epsilon Random</th>
                                <th>Pass Fallback</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                `;
            }

            async function loadControlStatus() {
                const response = await fetch('/control/status');
                if (!response.ok) throw new Error(`Failed /control/status (${response.status})`);
                const data = await response.json();
                const statusNode = document.getElementById('control-status');
                const pauseBtn = document.getElementById('pause-btn');
                const resumeBtn = document.getElementById('resume-btn');

                if (data.paused) {
                    statusNode.innerHTML = '<span class="bad">Training paused</span>';
                    pauseBtn.disabled = true;
                    resumeBtn.disabled = false;
                } else {
                    statusNode.innerHTML = '<span class="good">Training running</span>';
                    pauseBtn.disabled = false;
                    resumeBtn.disabled = true;
                }
            }

            async function pauseTraining() {
                try {
                    const response = await fetch('/control/pause', { method: 'POST' });
                    if (!response.ok) throw new Error(`Failed /control/pause (${response.status})`);
                    await loadControlStatus();
                } catch (error) {
                    console.error('Pause failed', error);
                    alert('Failed to pause training');
                }
            }

            async function resumeTraining() {
                try {
                    const response = await fetch('/control/resume', { method: 'POST' });
                    if (!response.ok) throw new Error(`Failed /control/resume (${response.status})`);
                    await loadControlStatus();
                } catch (error) {
                    console.error('Resume failed', error);
                    alert('Failed to resume training');
                }
            }

            async function startHumanVsGeneration() {
                const button = document.getElementById('human-vs-btn');
                const statusNode = document.getElementById('human-link-status');
                try {
                    const generationRaw = window.prompt('Generation number (leave blank for random):', '');
                    if (generationRaw === null) return;
                    const generationText = String(generationRaw || '').trim();
                    const randomGeneration = generationText.length === 0;

                    const humanNameRaw = window.prompt('Your player name:', 'You');
                    if (humanNameRaw === null) return;
                    const humanName = String(humanNameRaw || '').trim() || 'You';

                    const payload = {
                        generation: randomGeneration ? null : Number(generationText),
                        random_generation: randomGeneration,
                        human_name: humanName,
                        bot_count: 3,
                    };

                    button.disabled = true;
                    button.textContent = 'Creating...';
                    statusNode.innerHTML = '<span class="mono">Starting human-vs-generation game...</span>';

                    const response = await fetch('/play/human-vs-generation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                    const result = await response.json();
                    if (!response.ok || !result.success) {
                        throw new Error(result.detail || result.error || `HTTP ${response.status}`);
                    }

                    const playerUrl = result.player_url || '';
                    const gameUrl = result.game_url || '';
                    const generation = result.generation;
                    statusNode.innerHTML = playerUrl
                        ? `Play link (generation ${generation}): <a href="${playerUrl}" target="_blank" rel="noopener noreferrer">${playerUrl}</a>`
                        : '<span class="bad">Game created, but no player URL returned.</span>';

                    const summary = [
                        `Game created: ${result.game_id}`,
                        `Generation: ${generation}`,
                        playerUrl ? `Play: ${playerUrl}` : '',
                        gameUrl ? `Spectator: ${gameUrl}` : '',
                    ].filter(Boolean).join('\\n');
                    alert(summary);
                    if (playerUrl) {
                        window.open(playerUrl, '_blank', 'noopener,noreferrer');
                    }
                    await refreshAll();
                } catch (error) {
                    console.error('Human-vs-generation failed', error);
                    statusNode.innerHTML = '<span class="bad">Failed to create human-vs-generation game.</span>';
                    alert('Failed to create human-vs-generation game');
                } finally {
                    button.disabled = false;
                    button.textContent = 'Play Vs Generation';
                }
            }

            async function runDebugGame() {
                const debugBtn = document.getElementById('debug-btn');
                try {
                    debugBtn.disabled = true;
                    debugBtn.textContent = 'Creating...';
                    const response = await fetch('/debug/game', { method: 'POST' });
                    const result = await response.json();
                    if (!response.ok || !result.success) {
                        throw new Error(result.error || `HTTP ${response.status}`);
                    }
                    alert(`Debug game created: ${result.game_id}`);
                    await refreshAll();
                } catch (error) {
                    console.error('Debug game failed', error);
                    alert('Failed to create debug game');
                } finally {
                    debugBtn.disabled = false;
                    debugBtn.textContent = 'Run Debug Game';
                }
            }

            async function refreshAll() {
                try {
                    await Promise.all([loadStats(), loadPopulation(), loadControlStatus()]);
                } catch (error) {
                    console.error('Refresh failed', error);
                }
            }

            refreshAll();
            setInterval(refreshAll, AUTO_REFRESH_MS);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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

