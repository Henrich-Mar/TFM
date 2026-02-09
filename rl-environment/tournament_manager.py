"""
Tournament Manager - Handles AI vs AI competitions
"""


import asyncio
import os
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import aiohttp

from models.agent import RLAgent
from game_interface import GameServerCluster, GameInstance

logger = logging.getLogger(__name__)

@dataclass
class TournamentBracket:
    id: str
    agents: List[RLAgent]
    games_per_matchup: int
    created_at: datetime

@dataclass
class GameResult:
    game_id: str
    players: List[Dict[str, Any]]  # player results with agent_id, rank, victory_points, etc.
    duration_seconds: float
    completed: bool
    error_message: Optional[str] = None
    end_screens: List[str] = None

class TournamentManager:
    def __init__(self, game_cluster: GameServerCluster):
        self.game_cluster = game_cluster
        self.active_tournaments: Dict[str, TournamentBracket] = {}
        
    def create_tournaments(self, 
                         population: List[RLAgent], 
                         tournament_size: int,
                         games_per_evaluation: int) -> List[TournamentBracket]:
        """Create tournament brackets from population"""
        tournaments = []
        
        # Shuffle population for random matchups
        shuffled_population = population.copy()
        random.shuffle(shuffled_population)
        
        # Create tournaments
        for i in range(0, len(shuffled_population), tournament_size):
            tournament_agents = shuffled_population[i:i + tournament_size]
            
            # Pad with random agents if needed
            while len(tournament_agents) < tournament_size:
                tournament_agents.append(random.choice(population))
            
            tournament = TournamentBracket(
                id=str(uuid.uuid4()),
                agents=tournament_agents,
                games_per_matchup=games_per_evaluation,
                created_at=datetime.now()
            )
            
            tournaments.append(tournament)
            self.active_tournaments[tournament.id] = tournament
        
        logger.info(f"Created {len(tournaments)} tournaments with {tournament_size} agents each")
        return tournaments
    
    async def run_tournament(self, tournament: TournamentBracket) -> Dict[str, Any]:
        """Run a complete tournament and return results"""
        logger.info(f"Starting tournament {tournament.id} with {len(tournament.agents)} agents")
        
        start_time = datetime.now()
        games = []
        
        try:
            # Create all possible 4-player combinations from tournament agents
            game_combinations = self._create_game_combinations(
                tournament.agents,
                tournament.games_per_matchup,
            )
            
            # Run games with controlled concurrency (configurable via env)
            try:
                concurrency = int(os.getenv('TOURNAMENT_CONCURRENCY', '3'))
            except Exception:
                concurrency = 3
            semaphore = asyncio.Semaphore(max(1, concurrency))
            
            game_tasks = [
                self._run_single_game_with_semaphore(semaphore, agents, tournament.id)
                for agents in game_combinations
            ]
            
            game_results = await asyncio.gather(*game_tasks, return_exceptions=True)
            
            # Filter successful games
            for result in game_results:
                if isinstance(result, GameResult):
                    games.append(result)
                else:
                    logger.error(f"Game failed in tournament {tournament.id}: {result}")
            
        except Exception as e:
            logger.error(f"Tournament {tournament.id} failed: {e}")
        finally:
            # Clean up
            if tournament.id in self.active_tournaments:
                del self.active_tournaments[tournament.id]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        tournament_result = {
            'tournament_id': tournament.id,
            'agents': [agent.id for agent in tournament.agents],
            'games': [self._game_result_to_dict(game) for game in games],
            'duration_seconds': duration,
            'completed_games': len(games),
            'total_planned_games': len(game_combinations)
        }
        
        logger.info(f"Tournament {tournament.id} completed: {len(games)} games in {duration:.1f}s")
        return tournament_result
    
    def _create_game_combinations(self, agents: List[RLAgent], games_per_matchup: int = 1) -> List[List[RLAgent]]:
        """Create 4-player game combinations from tournament agents"""
        import itertools

        repeats = max(1, int(games_per_matchup))
        combinations = []

        # If we have exactly 4 agents, repeat the same lineup.
        if len(agents) == 4:
            combinations = [list(agents) for _ in range(repeats)]

        # If more than 4, create different combinations
        elif len(agents) > 4:
            base_combinations = [list(combo) for combo in itertools.combinations(agents, 4)]
            for combo in base_combinations:
                for _ in range(repeats):
                    combinations.append(list(combo))

        # If less than 4, duplicate agents
        else:
            padded_agents = list(agents)
            while len(padded_agents) < 4:
                padded_agents.append(random.choice(agents))
            combinations = [list(padded_agents) for _ in range(repeats)]

        return combinations
    
    async def _run_single_game_with_semaphore(self, 
                                            semaphore: asyncio.Semaphore,
                                            agents: List[RLAgent],
                                            tournament_id: str) -> GameResult:
        """Run a single game with concurrency control"""
        async with semaphore:
            return await self._run_single_game(agents, tournament_id)
    
    async def _run_single_game(self, agents: List[RLAgent], tournament_id: str) -> GameResult:
        """Run a single 4-player game"""
        provisional_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            fast_mode_env = str(os.getenv('TM_FAST_MODE_OPTION', '1')).strip().lower()
            fast_mode_option = fast_mode_env in ('1', 'true', 'yes', 'on')
            # Get available game server
            game_instance = await self.game_cluster.create_game(
                game_id=provisional_id,
                player_names=[f"Agent_{agent.id[:8]}" for agent in agents],
                game_options={
                    'soloMode': False,
                    'randomMA': 'No randomization',
                    'showTimers': False,
                    'fastModeOption': fast_mode_option,
                    'removeNegativeGlobalEventsOption': True,
                    'undoOption': False
                }
            )
            logger.info("Game %s created with fastModeOption=%s", game_instance.game_id, fast_mode_option)
            # Record a visitable URL for this game id in coordinator.recent_games:
            try:
                public_base = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
                # Spectator URL in UI typically is /game?id={game_id}
                game_url = f"{public_base}/game?id={game_instance.game_id}"
                # Stash in cluster scratchpad
                if hasattr(self.game_cluster, 'recent_games'):
                    self.game_cluster.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
                # Also attempt to store on coordinator if reachable via app global
                from api.server import coordinator as api_coordinator
                if api_coordinator is not None and hasattr(api_coordinator, 'recent_games'):
                    api_coordinator.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
            except Exception:
                pass
            
            # Record game URL for dashboard with correct public base per server
            try:
                # Build mapping from env PUBLIC_TM_MAP if provided, else fallback to PUBLIC_TM_URL
                mapping_str = os.getenv('PUBLIC_TM_MAP', '')
                public_map: Dict[str, str] = {}
                if mapping_str:
                    for pair in mapping_str.split(','):
                        if not pair:
                            continue
                        k, v = pair.split('=')
                        public_map[k.strip()] = v.strip()
                server_key = f"{game_instance.server.host}:{game_instance.server.port}"
                public_base = public_map.get(server_key)
                if not public_base:
                    pub = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
                    public_base = (pub.split(',')[0] if ',' in pub else pub)
                game_url = f"{public_base}/game?id={game_instance.game_id}"
                if hasattr(self.game_cluster, 'recent_games'):
                    self.game_cluster.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
                from api.server import coordinator as api_coordinator
                if api_coordinator is not None and hasattr(api_coordinator, 'recent_games'):
                    api_coordinator.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
            except Exception:
                pass

            # Connect agents to game
            agent_tasks = [
                agent.play_game(game_instance, f"Agent_{agent.id[:8]}")
                for agent in agents
            ]
            
            # Wait for game completion with timeout
            await asyncio.wait_for(
                asyncio.gather(*agent_tasks), 
                timeout=300  # 5 minutes max per game
            )
            
            # Get final game state and results
            game_state = await game_instance.get_final_state()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Extract end-screen URLs and parse authoritative totals/ranks when possible
            players: List[Dict[str, Any]] = []
            name_map: Dict[str, str] = {f"Agent_{agent.id[:8]}": agent.id for agent in agents}
            parsed_scoreboard: Optional[Dict[str, Dict[str, Any]]] = None

            # Build end-screen URLs for convenience
            # Resolve bases per server
            internal_base = game_instance.base_url
            mapping_str = os.getenv('PUBLIC_TM_MAP', '')
            public_map: Dict[str, str] = {}
            if mapping_str:
                try:
                    for pair in mapping_str.split(','):
                        if not pair:
                            continue
                        k, v = pair.split('=')
                        public_map[k.strip()] = v.strip()
                except Exception:
                    logger.warning("Failed to parse PUBLIC_TM_MAP; falling back to PUBLIC_TM_URL")
            server_key = f"{game_instance.server.host}:{game_instance.server.port}"
            public_base = public_map.get(server_key)
            if not public_base:
                pub = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
                public_base = (pub.split(',')[0] if ',' in pub else pub)
            end_screens: List[str] = []  # public links for UI
            end_screens_fetch: List[str] = []  # internal links for container-to-container fetch
            try:
                for p in game_state.get('players', []):
                    pid = p.get('id')
                    if pid:
                        end_screens.append(f"{public_base}/the-end?id={pid}")
                        end_screens_fetch.append(f"{internal_base}/the-end?id={pid}")
                        logger.info(f"End screen: {public_base}/the-end?id={pid}")
                # Prefer internal JSON for reliable totals/ranking
                parsed_scoreboard = None
                try:
                    # Fetch the final game view (players list) to compute scores like the UI
                    async with self.game_cluster.session.get(f"{internal_base}/api/game", params={'id': game_instance.game_id}) as r:
                        if r.status == 200:
                            game_view = await r.json()
                            players_view = game_view.get('players', []) or []
                            # Compute rank: sort by total VP desc, then megacredits desc
                            def vp_total(p):
                                return int(((p.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0))
                            def mc_val(p):
                                return int(p.get('megaCredits', 0) or 0)
                            sorted_players = sorted(players_view, key=lambda p: (vp_total(p), mc_val(p)), reverse=True)
                            parsed_scoreboard = {}
                            for p in sorted_players:
                                parsed_scoreboard[p.get('name')] = {
                                    'name': p.get('name'),
                                    'total': vp_total(p),
                                    'tr': int(((p.get('victoryPointsBreakdown', {}) or {}).get('terraformRating', 0) or 0)),
                                    'mc': mc_val(p),
                                }
                        else:
                            logger.warning(f"Failed to fetch game view for scoring (HTTP {r.status})")
                except Exception:
                    logger.warning("Fetching game view for tournament scoring failed", exc_info=True)
            except Exception:
                logger.warning("Parsing end screen for tournament scoring failed", exc_info=True)
            
            if parsed_scoreboard:
                # Use parsed totals to compute ranks
                # Map our agents to parsed names (should match Agent_<id8>)
                scored_list: List[Dict[str, Any]] = []
                for agent in agents:
                    disp_name = f"Agent_{agent.id[:8]}"
                    pl = parsed_scoreboard.get(disp_name)
                    total = int(pl.get('total', 0)) if pl else 0
                    tr = int(pl.get('tr', 0)) if pl and isinstance(pl.get('tr'), int) else None
                    scored_list.append({'agent_id': agent.id, 'name': disp_name, 'total': total, 'tr': tr})
                # Rank by total desc, tiebreaker TR desc
                scored_list.sort(key=lambda x: (x['total'], x['tr'] if x['tr'] is not None else -1), reverse=True)
                # Assign 1..n ranks
                rank_map: Dict[str, int] = {}
                current_rank = 1
                last_score = None
                for entry in scored_list:
                    score_key = (entry['total'], entry['tr'])
                    if score_key != last_score:
                        rank = current_rank
                        last_score = score_key
                    rank_map[entry['agent_id']] = rank
                    current_rank += 1
                # Emit players list
                for entry in scored_list:
                    players.append({
                        'agent_id': entry['agent_id'],
                        'rank': rank_map[entry['agent_id']],
                        'victory_points': entry['total'],
                        'terraform_rating': entry['tr'] if entry['tr'] is not None else 0,
                        'megacredits': 0,
                        'completed': True,
                    })
            else:
                # Fallback: fetch per-player JSON from internal API to get victoryPointsBreakdown.total
                name_to_pid: Dict[str, str] = {}
                for p in game_state.get('players', []):
                    if p.get('name') and p.get('id'):
                        name_to_pid[p['name']] = p['id']
                internal_base = os.getenv('INTERNAL_TM_URL', os.getenv('PUBLIC_TM_URL', 'http://localhost:8081'))
                totals: Dict[str, Dict[str, Any]] = {}
                try:
                    async with aiohttp.ClientSession() as s:
                        for agent in agents:
                            disp_name = f"Agent_{agent.id[:8]}"
                            pid = name_to_pid.get(disp_name)
                            if not pid:
                                continue
                            api_url = f"{internal_base}/api/player?id={pid}"
                            async with s.get(api_url) as r:
                                if r.status == 200:
                                    data = await r.json()
                                    vp = int((data.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0)
                                    tr = int(data.get('terraformRating', 0) or 0)
                                    totals[agent.id] = { 'name': disp_name, 'total': vp, 'tr': tr }
                                else:
                                    logger.warning(f"Failed to fetch player JSON {api_url} (HTTP {r.status})")
                except Exception:
                    logger.warning("Fetching per-player JSON failed for tournament fallback", exc_info=True)

                if totals:
                    scored_list = []
                    for agent in agents:
                        entry = totals.get(agent.id, { 'name': f"Agent_{agent.id[:8]}", 'total': 0, 'tr': 0 })
                        scored_list.append({ 'agent_id': agent.id, 'name': entry['name'], 'total': entry['total'], 'tr': entry['tr'] })
                    scored_list.sort(key=lambda x: (x['total'], x['tr']), reverse=True)
                    rank_map: Dict[str, int] = {}
                    current_rank = 1
                    last_score = None
                    for e in scored_list:
                        score_key = (e['total'], e['tr'])
                        if score_key != last_score:
                            rank = current_rank
                            last_score = score_key
                        rank_map[e['agent_id']] = rank
                        current_rank += 1
                    for e in scored_list:
                        players.append({
                            'agent_id': e['agent_id'],
                            'rank': rank_map[e['agent_id']],
                            'victory_points': e['total'],
                            'terraform_rating': e['tr'],
                            'megacredits': 0,
                            'completed': True
                        })
                else:
                    # Last-resort fallback to final game state
                    for i, agent in enumerate(agents):
                        player_data = game_state['players'][i]
                        players.append({
                            'agent_id': agent.id,
                            'rank': player_data.get('rank', 4),
                            'victory_points': player_data.get('terraformRating', 20),
                            'terraform_rating': player_data.get('terraformRating', 20),
                            'megacredits': player_data.get('megaCredits', 0),
                            'completed': True
                        })

            actual_game_id = game_instance.game_id
            result = GameResult(
                game_id=actual_game_id,
                players=players,
                duration_seconds=duration,
                completed=True,
                end_screens=end_screens
            )
            # Push into coordinator recent lists if available
            try:
                from api.server import coordinator as api_coordinator
                if api_coordinator is not None:
                    if hasattr(api_coordinator, 'recent_end_screens'):
                        api_coordinator.recent_end_screens.append({"game_id": actual_game_id, "end_screens": end_screens})
                    if hasattr(api_coordinator, 'recent_games'):
                        public_base = os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')
                        api_coordinator.recent_games.append({"game_id": actual_game_id, "url": f"{public_base}/ui/game?id={actual_game_id}"})
            except Exception:
                logger.warning("Failed adding recent game/end screen to coordinator")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Game {provisional_id} timed out after 5 minutes")
            return GameResult(
                game_id=provisional_id,
                players=[{
                    'agent_id': agent.id,
                    'rank': 4,
                    'victory_points': 0,
                    'completed': False
                } for agent in agents],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                completed=False,
                error_message="Game timed out",
                end_screens=[]
            )
            
        except Exception as e:
            logger.error(f"Game {provisional_id} failed: {e}")
            return GameResult(
                game_id=provisional_id,
                players=[{
                    'agent_id': agent.id,
                    'rank': 4,
                    'victory_points': 0,
                    'completed': False
                } for agent in agents],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                completed=False,
                error_message=str(e),
                end_screens=[]
            )
        
        finally:
            # Clean up game instance
            try:
                await game_instance.cleanup()
            except:
                pass
    
    def _game_result_to_dict(self, game_result: GameResult) -> Dict[str, Any]:
        """Convert GameResult to dictionary"""
        return {
            'game_id': game_result.game_id,
            'players': game_result.players,
            'duration_seconds': game_result.duration_seconds,
            'completed': game_result.completed,
            'error_message': game_result.error_message,
            'end_screens': game_result.end_screens or []
        }
    
    async def get_tournament_status(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """Get status of running tournament"""
        if tournament_id not in self.active_tournaments:
            return None
        
        tournament = self.active_tournaments[tournament_id]
        return {
            'tournament_id': tournament_id,
            'agent_count': len(tournament.agents),
            'games_per_matchup': tournament.games_per_matchup,
            'created_at': tournament.created_at.isoformat(),
            'status': 'running'
        }
