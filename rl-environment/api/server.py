"""
API Server for monitoring RL training progress
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(title="Terraforming Mars RL Monitor", version="1.0.0")

# Global reference to coordinator (set when starting server)
coordinator = None

class DebugGameRequest(BaseModel):
    agent_ids: Optional[List[str]] = None

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
        population_stats = {
            "size": len(coordinator.population),
            "current_generation": coordinator.current_generation,
            "total_games_played": sum(agent.games_played for agent in coordinator.population),
            "best_fitness": max(agent.get_fitness_score() for agent in coordinator.population) if coordinator.population else 0,
            "avg_fitness": sum(agent.get_fitness_score() for agent in coordinator.population) / len(coordinator.population) if coordinator.population else 0
        }
        
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
        # Show both lifetime fitness (agent.get_fitness_score) and last eval fitness used for selection
        last_eval = coordinator.last_eval_fitness.get(agent.id) if hasattr(coordinator, 'last_eval_fitness') else None
        agent_data.update({
            "fitness_score": agent.get_fitness_score(),
            "last_eval_fitness": last_eval if last_eval is not None else agent.get_fitness_score(),
            "win_rate": agent.wins / agent.games_played if agent.games_played > 0 else 0,
            "avg_vp": agent.total_victory_points / agent.games_played if agent.games_played > 0 else 0
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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard for monitoring"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Terraforming Mars RL Training Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 4px; margin: 5px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; font-size: 0.9em; }
            .status-good { color: #28a745; }
            .status-bad { color: #dc3545; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #refresh { position: fixed; top: 20px; right: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Terraforming Mars RL Training Dashboard</h1>
            
            <button id="refresh" onclick="location.reload()">🔄 Refresh</button>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 Training Progress</h3>
                    <div id="training-stats">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>🎮 Game Servers</h3>
                    <div id="server-stats">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>🧬 Population</h3>
                    <div id="population-stats">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>🏆 Tournaments</h3>
                    <div id="tournament-stats">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>⏸️ Training Control</h3>
                    <div id="training-control">
                        <button id="pause-btn" onclick="pauseTraining()">⏸️ Pause</button>
                        <button id="resume-btn" onclick="resumeTraining()">▶️ Resume</button>
                        <button id="debug-btn" onclick="runDebugGame()">🐛 Debug Game</button>
                        <div id="control-status">Loading...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🧭 Recent Games</h3>
                    <div id="recent-games">Loading...</div>
                </div>
                <div class="card">
                    <h3>🏁 Recent End Screens</h3>
                    <div id="recent-ends">Loading...</div>
                </div>
                <div class="card">
                    <h3>🏁 End Summaries</h3>
                    <div id="recent-ends-summary">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Top Performing Agents</h3>
                <div id="top-agents">Loading...</div>
            </div>
        </div>
        
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    // Training stats
                    document.getElementById('training-stats').innerHTML = `
                        <div class="metric">
                            <div class="metric-value">${data.population.current_generation}</div>
                            <div class="metric-label">Generation</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.population.size}</div>
                            <div class="metric-label">Population Size</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.population.best_fitness.toFixed(2)}</div>
                            <div class="metric-label">Best Fitness</div>
                        </div>
                    `;
                    
                    // Server stats
                    document.getElementById('server-stats').innerHTML = `
                        <div class="metric">
                            <div class="metric-value ${data.servers.healthy_servers === data.servers.total_servers ? 'status-good' : 'status-bad'}">
                                ${data.servers.healthy_servers}/${data.servers.total_servers}
                            </div>
                            <div class="metric-label">Healthy Servers</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.servers.total_active_games}</div>
                            <div class="metric-label">Active Games</div>
                        </div>
                    `;

                    // Recent games list
                    const recent = (data.recent_games || []).slice(-10).reverse();
                    document.getElementById('recent-games').innerHTML = recent.length ? `
                        <ul>
                            ${recent.map(g => `<li><a href="${g.url}" target="_blank">${g.game_id}</a></li>`).join('')}
                        </ul>
                    ` : '<div class="metric-label">No recent games</div>';

                    // Recent end screens list (first player link per game)
                    const ends = (data.recent_end_screens || []).slice(-10).reverse();
                    document.getElementById('recent-ends').innerHTML = ends.length ? `
                        <ul>
                            ${ends.map(e => {
                                const url = (e.end_screens && e.end_screens.length) ? e.end_screens[0] : null;
                                return url ? `<li><a href="${url}" target="_blank">${e.game_id}</a></li>` : `<li>${e.game_id}</li>`;
                            }).join('')}
                        </ul>
                    ` : '<div class="metric-label">No completed games yet</div>';

                    // Fetch and render parsed summaries for the most recent completed games
                    const endsToSummarize = ends.slice(0, 5);
                    if (endsToSummarize.length) {
                        const summaries = await Promise.all(endsToSummarize.map(async (e) => {
                            const url = (e.end_screens && e.end_screens.length) ? e.end_screens[0] : null;
                            if (!url) return null;
                            try {
                                const resp = await fetch(`/parse-end-screen?url=${encodeURIComponent(url)}`);
                                if (!resp.ok) return { game_id: e.game_id, url, error: `HTTP ${resp.status}` };
                                const parsed = await resp.json();
                                return { game_id: e.game_id, url, parsed };
                            } catch (err) {
                                return { game_id: e.game_id, url, error: String(err) };
                            }
                        }));
                        const html = summaries.filter(Boolean).map(s => {
                            if (!s.parsed) {
                                return `<div><a href="${s.url}" target="_blank">${s.game_id}</a>: <span class="metric-label">summary unavailable${s.error ? ` (${s.error})` : ''}</span></div>`;
                            }
                            const winner = s.parsed.winner || 'Unknown winner';
                            const players = (s.parsed.players || []).slice().sort((a,b) => (b.total||0) - (a.total||0));
                            const top = players.slice(0, 3).map(p => `${p.name}: ${p.total}`).join(' · ');
                            return `<div>
                                <a href="${s.url}" target="_blank">${s.game_id}</a>: 
                                <strong>${winner}</strong>
                                <span class="metric-label"> | Top: ${top || 'n/a'}</span>
                            </div>`;
                        }).join('');
                        document.getElementById('recent-ends-summary').innerHTML = html || '<div class="metric-label">No summaries</div>';
                    } else {
                        document.getElementById('recent-ends-summary').innerHTML = '<div class="metric-label">No completed games yet</div>';
                    }
                    
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }
            
            async function loadPopulation() {
                try {
                    const response = await fetch('/population');
                    const data = await response.json();
                    
                    const topAgents = data.population.slice(0, 10);
                    
                    document.getElementById('top-agents').innerHTML = `
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background: #f8f9fa;">
                                <th style="padding: 10px; text-align: left;">Agent ID</th>
                                <th style="padding: 10px; text-align: right;">Eval Fitness</th>
                                <th style="padding: 10px; text-align: right;">Lifetime Fitness</th>
                                <th style="padding: 10px; text-align: right;">Games</th>
                                <th style="padding: 10px; text-align: right;">Win Rate</th>
                                <th style="padding: 10px; text-align: right;">Avg VP</th>
                            </tr>
                            ${topAgents.map(agent => `
                                <tr>
                                    <td style=\"padding: 10px;\">${agent.id.substring(0, 8)}...</td>
                                    <td style=\"padding: 10px; text-align: right;\">${(agent.last_eval_fitness ?? agent.fitness_score).toFixed(2)}</td>
                                    <td style=\"padding: 10px; text-align: right;\">${agent.fitness_score.toFixed(2)}</td>
                                    <td style=\"padding: 10px; text-align: right;\">${agent.games_played}</td>
                                    <td style=\"padding: 10px; text-align: right;\">${(agent.win_rate * 100).toFixed(1)}%</td>
                                    <td style=\"padding: 10px; text-align: right;\">${agent.avg_vp.toFixed(1)}</td>
                                </tr>
                            `).join('')}
                        </table>
                    `;
                    
                } catch (error) {
                    console.error('Failed to load population:', error);
                }
            }
            
            async function loadControlStatus() {
                try {
                    const response = await fetch('/control/status');
                    const data = await response.json();
                    
                    const statusDiv = document.getElementById('control-status');
                    const pauseBtn = document.getElementById('pause-btn');
                    const resumeBtn = document.getElementById('resume-btn');
                    
                    if (data.paused) {
                        statusDiv.innerHTML = '<div class="metric-label status-bad">⏸️ Training Paused</div>';
                        pauseBtn.disabled = true;
                        resumeBtn.disabled = false;
                    } else {
                        statusDiv.innerHTML = '<div class="metric-label status-good">▶️ Training Running</div>';
                        pauseBtn.disabled = false;
                        resumeBtn.disabled = true;
                    }
                } catch (error) {
                    console.error('Failed to load control status:', error);
                }
            }
            
            async function pauseTraining() {
                try {
                    const response = await fetch('/control/pause', { method: 'POST' });
                    if (response.ok) {
                        loadControlStatus();
                    } else {
                        alert('Failed to pause training');
                    }
                } catch (error) {
                    console.error('Failed to pause training:', error);
                    alert('Failed to pause training');
                }
            }
            
            async function resumeTraining() {
                try {
                    const response = await fetch('/control/resume', { method: 'POST' });
                    if (response.ok) {
                        loadControlStatus();
                    } else {
                        alert('Failed to resume training');
                    }
                } catch (error) {
                    console.error('Failed to resume training:', error);
                    alert('Failed to resume training');
                }
            }
            
            async function runDebugGame() {
                try {
                    const debugBtn = document.getElementById('debug-btn');
                    debugBtn.disabled = true;
                    debugBtn.textContent = '🐛 Creating...';
                    
                    const response = await fetch('/debug/game', { method: 'POST' });
                    const result = await response.json();
                    
                    if (response.ok && result.success) {
                        alert(`Debug game created! Game ID: ${result.game_id}\\nGame URL: ${result.game_url || 'N/A'}`);
                        // Refresh stats to show the new game
                        loadStats();
                    } else {
                        alert(`Failed to create debug game: ${result.error || 'Unknown error'}`);
                    }
                } catch (error) {
                    console.error('Failed to create debug game:', error);
                    alert('Failed to create debug game');
                } finally {
                    const debugBtn = document.getElementById('debug-btn');
                    debugBtn.disabled = false;
                    debugBtn.textContent = '🐛 Debug Game';
                }
            }
            
            // Load data on page load
            loadStats();
            loadPopulation();
            loadControlStatus(); // Load control status on page load
            
            // Auto-refresh every 30 seconds
            setInterval(() => {
                loadStats();
                loadPopulation();
                loadControlStatus(); // Refresh control status periodically
            }, 30000);
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