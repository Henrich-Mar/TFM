"""
Tournament Manager - Handles AI vs AI competitions
"""


import asyncio
import os
import logging
import random
from contextlib import suppress
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

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
    game_generation: Optional[int] = None  # Final generation when game ended (lower = faster win)

class TournamentManager:
    def __init__(self, game_cluster: GameServerCluster):
        self.game_cluster = game_cluster
        self.active_tournaments: Dict[str, TournamentBracket] = {}
        self.tournament_progress: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _cancel_agent_tasks_with_reason(
        agent_tasks: List[asyncio.Task],
        *,
        reason: str,
        game_id: str,
    ):
        for task in agent_tasks:
            if task.done():
                continue
            try:
                setattr(task, "_cancel_reason", str(reason))
                setattr(task, "_cancel_game_id", str(game_id))
            except Exception:
                pass
            task.cancel()

    @staticmethod
    def _agent_name_token(agent_id: Any, width: int = 8) -> str:
        raw = ''.join(ch for ch in str(agent_id or '') if ch.isalnum())
        if not raw:
            raw = uuid.uuid4().hex
        return raw[: max(4, int(width))].lower()

    @classmethod
    def _build_seat_player_names(cls, agents: List[RLAgent]) -> List[str]:
        # Prefix with seat index so names are unique even when many agents share ID prefixes.
        return [
            f"A{idx + 1}_{cls._agent_name_token(getattr(agent, 'id', ''))}"
            for idx, agent in enumerate(agents)
        ]

    @staticmethod
    def _normalize_agent_game_telemetry(agent_result: Any) -> Dict[str, Any]:
        if not isinstance(agent_result, dict):
            return {
                "draft_decisions": 0,
                "draft_decisions_low_hand_ev": 0,
                "hate_draft_picks": 0,
                "hate_draft_picks_low_hand_ev": 0,
                "hate_draft_rate": 0.0,
                "hate_draft_rate_low_hand_ev": 0.0,
            }
        return {
            "draft_decisions": int(agent_result.get("draft_decisions", 0) or 0),
            "draft_decisions_low_hand_ev": int(agent_result.get("draft_decisions_low_hand_ev", 0) or 0),
            "hate_draft_picks": int(agent_result.get("hate_draft_picks", 0) or 0),
            "hate_draft_picks_low_hand_ev": int(agent_result.get("hate_draft_picks_low_hand_ev", 0) or 0),
            "hate_draft_rate": float(agent_result.get("hate_draft_rate", 0.0) or 0.0),
            "hate_draft_rate_low_hand_ev": float(agent_result.get("hate_draft_rate_low_hand_ev", 0.0) or 0.0),
        }
        
    def create_tournaments(self, 
                         population: List[RLAgent], 
                         tournament_size: int,
                         games_per_evaluation: int,
                         shuffle_population: bool = True) -> List[TournamentBracket]:
        """Create tournament brackets from population"""
        tournaments = []
        
        # Shuffle population for random matchups unless caller provides an ordered list.
        shuffled_population = population.copy()
        if shuffle_population:
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
            planned_games = len(self._create_game_combinations(tournament_agents, games_per_evaluation))
            self.tournament_progress[tournament.id] = {
                "tournament_id": tournament.id,
                "planned_games": int(planned_games),
                "finished_games": 0,
                "successful_games": 0,
                "failed_games": 0,
                "status": "queued",
                "created_at": tournament.created_at.isoformat(),
            }
        
        logger.info(f"Created {len(tournaments)} tournaments with {tournament_size} agents each")
        return tournaments
    
    async def run_tournament(
        self,
        tournament: TournamentBracket,
        global_game_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Run a complete tournament and return results"""
        logger.info(f"Starting tournament {tournament.id} with {len(tournament.agents)} agents")
        
        start_time = datetime.now()
        games = []
        game_combinations: List[List[RLAgent]] = []
        finished_games = 0
        successful_games = 0
        failed_games = 0
        
        try:
            # Create all possible 4-player combinations from tournament agents
            game_combinations = self._create_game_combinations(
                tournament.agents,
                tournament.games_per_matchup,
            )
            self.tournament_progress[tournament.id] = {
                "tournament_id": tournament.id,
                "planned_games": len(game_combinations),
                "finished_games": 0,
                "successful_games": 0,
                "failed_games": 0,
                "status": "running",
                "created_at": tournament.created_at.isoformat(),
            }
            
            # Run games with controlled concurrency (configurable via env)
            try:
                concurrency = int(os.getenv('TOURNAMENT_CONCURRENCY', '3'))
            except Exception:
                concurrency = 3
            semaphore = asyncio.Semaphore(max(1, concurrency))
            
            game_tasks = [
                asyncio.create_task(
                    self._run_single_game_with_semaphore(
                        semaphore,
                        global_game_semaphore,
                        agents,
                        tournament.id,
                    )
                )
                for agents in game_combinations
            ]
            
            for task in asyncio.as_completed(game_tasks):
                try:
                    result = await task
                    games.append(result)
                    if bool(getattr(result, "completed", False)):
                        successful_games += 1
                    else:
                        failed_games += 1
                except asyncio.CancelledError as e:
                    current = asyncio.current_task()
                    if current is not None and getattr(current, "cancelling", lambda: 0)():
                        raise
                    failed_games += 1
                    logger.error(
                        "Game task in tournament %s was cancelled unexpectedly; counting as failed. "
                        "Possible causes: TM_GAME_TIMEOUT_SEC too low, GPU/event-loop blocking, or process signal. %s",
                        tournament.id,
                        e,
                        exc_info=False,
                    )
                except Exception as e:
                    failed_games += 1
                    logger.error(f"Game failed in tournament {tournament.id}: {e}")
                finally:
                    finished_games += 1
                    if tournament.id in self.tournament_progress:
                        self.tournament_progress[tournament.id].update({
                            "finished_games": int(finished_games),
                            "successful_games": int(successful_games),
                            "failed_games": int(failed_games),
                            "status": "running",
                        })
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Tournament {tournament.id} failed: {e}")
        finally:
            # Clean up
            if tournament.id in self.active_tournaments:
                del self.active_tournaments[tournament.id]
            if tournament.id in self.tournament_progress:
                del self.tournament_progress[tournament.id]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        tournament_result = {
            'tournament_id': tournament.id,
            'agents': [agent.id for agent in tournament.agents],
            'games': [self._game_result_to_dict(game) for game in games],
            'duration_seconds': duration,
            'completed_games': int(finished_games),
            'successful_games': int(successful_games),
            'failed_games': int(failed_games),
            'total_planned_games': len(game_combinations),
            'evaluation_source': 'main',
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
    
    async def _run_single_game_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        global_game_semaphore: Optional[asyncio.Semaphore],
        agents: List[RLAgent],
        tournament_id: str,
    ) -> GameResult:
        """Run a single game with concurrency control"""
        async with semaphore:
            if global_game_semaphore is not None:
                async with global_game_semaphore:
                    return await self._run_single_game(agents, tournament_id)
            return await self._run_single_game(agents, tournament_id)
    
    async def _run_single_game(
        self,
        agents: List[RLAgent],
        tournament_id: str,
        game_seed: Optional[int] = None,
        players_beginner: Optional[bool] = None,
    ) -> GameResult:
        """Run a single 4-player game"""
        provisional_id = str(uuid.uuid4())
        actual_game_id = provisional_id
        game_instance: Optional[GameInstance] = None
        start_time = datetime.now()
        seat_player_names = self._build_seat_player_names(agents)
        
        try:
            fast_mode_env = str(os.getenv('TM_FAST_MODE_OPTION', '1')).strip().lower()
            fast_mode_option = fast_mode_env in ('1', 'true', 'yes', 'on')
            # Get available game server
            runtime_options: Dict[str, Any] = {
                'soloMode': False,
                'randomMA': 'No randomization',
                'showTimers': False,
                'fastModeOption': fast_mode_option,
                'removeNegativeGlobalEventsOption': True,
                'undoOption': False,
            }
            if game_seed is not None:
                runtime_options['seed'] = int(game_seed)
            if players_beginner is not None:
                runtime_options['_players_beginner'] = bool(players_beginner)
            game_instance = await self.game_cluster.create_game(
                game_id=provisional_id,
                player_names=seat_player_names,
                game_options=runtime_options,
            )
            actual_game_id = game_instance.game_id
            setattr(game_instance, "rl_seed", game_seed)
            logger.info("Game %s created with fastModeOption=%s", actual_game_id, fast_mode_option)
            # Record game URL for dashboard using canonical public URL resolver.
            try:
                game_url = game_instance.get_public_game_url()
                if hasattr(self.game_cluster, 'recent_games'):
                    self.game_cluster.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
                from api.server import coordinator as api_coordinator
                if api_coordinator is not None and hasattr(api_coordinator, 'recent_games'):
                    api_coordinator.recent_games.append({"game_id": game_instance.game_id, "url": game_url})
            except Exception:
                pass

            # Connect agents to game
            agent_tasks = [
                asyncio.create_task(
                    agent.play_game(game_instance, player_name),
                    name=f"agent-play:{actual_game_id}:{player_name}",
                )
                for agent, player_name in zip(agents, seat_player_names)
            ]
            agent_group = asyncio.gather(*agent_tasks, return_exceptions=True)
             
            # Wait for game completion with configurable timeout.
            try:
                timeout_sec = float(os.getenv('TM_GAME_TIMEOUT_SEC', '300'))
            except Exception:
                timeout_sec = 300.0
            timeout_sec = max(60.0, timeout_sec)
            # Keep the group alive when the supervisory timeout fires.  The timeout
            # handler below then records the reason and cancels every child itself;
            # otherwise wait_for cancels the gather first and its CancelledError can
            # escape the timeout path, shutting down the whole collector.
            agent_results = await asyncio.wait_for(asyncio.shield(agent_group), timeout=timeout_sec)
            task_errors = [r for r in agent_results if isinstance(r, Exception)]
            task_cancellations = [r for r in agent_results if isinstance(r, asyncio.CancelledError)]
            if task_cancellations:
                raise RuntimeError(
                    f"{len(task_cancellations)} agent task(s) were cancelled unexpectedly while game was running"
                )
            if task_errors:
                raise RuntimeError(
                    f"{len(task_errors)} agent task(s) failed; first error: {task_errors[0]!r}"
                )
            agent_telemetry_by_id: Dict[str, Dict[str, Any]] = {}
            for agent, agent_result in zip(agents, agent_results):
                telemetry = self._normalize_agent_game_telemetry(agent_result)
                reported_agent_id = str((agent_result or {}).get("agent_id", "") or "") if isinstance(agent_result, dict) else ""
                key = reported_agent_id if reported_agent_id else str(agent.id)
                agent_telemetry_by_id[key] = telemetry
              
            # Get final game state and results
            game_state = await game_instance.get_final_state()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Extract end-screen URLs and parse authoritative totals/ranks when possible.
            players: List[Dict[str, Any]] = []

            # Build end-screen URLs for convenience.
            internal_base = game_instance.base_url
            public_game_url = game_instance.get_public_game_url()
            public_base = public_game_url.split("/game?id=", 1)[0]
            end_screens: List[str] = []  # public links for UI
            try:
                for p in game_state.get('players', []):
                    pid = p.get('id')
                    if pid:
                        end_screens.append(f"{public_base}/the-end?id={pid}")
                        logger.info(f"End screen: {public_base}/the-end?id={pid}")
            except Exception:
                logger.warning("Parsing end screen for tournament scoring failed", exc_info=True)

            # Prefer per-player JSON (/api/player) and its embedded players[] scoreboard,
            # matching the logic used in RLAgent._record_game_result.
            game_players_by_name: Dict[str, Dict[str, Any]] = {}
            player_ids: List[str] = []
            for p in game_state.get('players', []):
                player_name = p.get('name')
                if player_name:
                    game_players_by_name[str(player_name)] = p
                pid = p.get('id')
                if pid:
                    player_ids.append(str(pid))

            view_players_by_name: Dict[str, Dict[str, Any]] = {}
            if player_ids:
                sample_pid = player_ids[0]
                try:
                    session = game_instance._get_session()
                    async with session.get(
                        f"{internal_base}/api/player",
                        params={'id': sample_pid},
                    ) as r:
                        if r.status == 200:
                            view = await r.json()
                            players_view = view.get('players', []) or []
                            for p in players_view:
                                nm = p.get('name')
                                if nm:
                                    view_players_by_name[str(nm)] = p
                        else:
                            logger.warning(
                                "Failed to fetch shared /api/player view for scoring (HTTP %s)",
                                r.status,
                            )
                except Exception:
                    logger.warning("Fetching /api/player view failed for tournament scoring", exc_info=True)

            scored_list: List[Dict[str, Any]] = []
            for agent, disp_name in zip(agents, seat_player_names):
                source = view_players_by_name.get(disp_name) or game_players_by_name.get(disp_name, {})
                vp_breakdown = dict(source.get('victoryPointsBreakdown', {}) or {})

                vp = int(((source.get('victoryPointsBreakdown', {}) or {}).get('total', source.get('terraformRating', 0)) or 0))
                tr = int(source.get('terraformRating', 0) or 0)
                mc = int(source.get('megaCredits', 0) or 0)

                scored_list.append({
                    'agent_id': agent.id,
                    'name': disp_name,
                    'total': vp,
                    'tr': tr,
                    'mc': mc,
                    'vp_terraforming': int(vp_breakdown.get('terraformRating', tr) or 0),
                    'vp_milestones': int(vp_breakdown.get('milestones', 0) or 0),
                    'vp_awards': int(vp_breakdown.get('awards', 0) or 0),
                    'vp_greenery': int(vp_breakdown.get('greenery', 0) or 0),
                    'vp_city': int(vp_breakdown.get('city', 0) or 0),
                    'vp_cards': int(vp_breakdown.get('victoryPoints', 0) or 0),
                    'town_placements': int(source.get('citiesCount', 0) or 0),
                    'greenery_placements': int(vp_breakdown.get('greenery', 0) or 0),
                })

            scored_list.sort(key=lambda x: (x['total'], x['mc'], x['tr']), reverse=True)

            rank_map: Dict[str, int] = {}
            current_rank = 1
            last_score: Optional[tuple] = None
            for entry in scored_list:
                score_key = (entry['total'], entry['mc'], entry['tr'])
                if score_key != last_score:
                    rank = current_rank
                    last_score = score_key
                rank_map[entry['agent_id']] = rank
                current_rank += 1

            # If the extracted scores are still uniformly zero-like, use final-state rank when available.
            uniform_scores = bool(scored_list) and all(
                (entry['total'], entry['mc'], entry['tr']) == (scored_list[0]['total'], scored_list[0]['mc'], scored_list[0]['tr'])
                for entry in scored_list
            )
            if uniform_scores:
                final_rank_by_name = {
                    str(p.get('name')): int(p.get('rank', 0) or 0)
                    for p in game_state.get('players', [])
                    if p.get('name')
                }
                any_valid_final_rank = any(v > 0 for v in final_rank_by_name.values())
                if any_valid_final_rank:
                    logger.warning(
                        "Uniform extracted tournament scores for game %s; using final-state ranks.",
                        game_instance.game_id,
                    )
                    for entry in scored_list:
                        fr = final_rank_by_name.get(entry['name'], 0)
                        if fr > 0:
                            rank_map[entry['agent_id']] = fr
                else:
                    logger.warning(
                        "Uniform tournament scores for game %s with no valid final ranks; scores=%s",
                        game_instance.game_id,
                        [(e['name'], e['total'], e['mc'], e['tr']) for e in scored_list],
                    )

            for entry in scored_list:
                telemetry = agent_telemetry_by_id.get(entry['agent_id'], self._normalize_agent_game_telemetry(None))
                players.append({
                    'agent_id': entry['agent_id'],
                    'rank': rank_map.get(entry['agent_id'], 4),
                    'victory_points': entry['total'],
                    'terraform_rating': entry['tr'],
                    'megacredits': entry['mc'],
                    'vp_terraforming': entry.get('vp_terraforming', 0),
                    'vp_milestones': entry.get('vp_milestones', 0),
                    'vp_awards': entry.get('vp_awards', 0),
                    'vp_greenery': entry.get('vp_greenery', 0),
                    'vp_city': entry.get('vp_city', 0),
                    'vp_cards': entry.get('vp_cards', 0),
                    'town_placements': entry.get('town_placements', 0),
                    'greenery_placements': entry.get('greenery_placements', 0),
                    'completed': True,
                    'draft_decisions': int(telemetry.get('draft_decisions', 0) or 0),
                    'draft_decisions_low_hand_ev': int(telemetry.get('draft_decisions_low_hand_ev', 0) or 0),
                    'hate_draft_picks': int(telemetry.get('hate_draft_picks', 0) or 0),
                    'hate_draft_picks_low_hand_ev': int(telemetry.get('hate_draft_picks_low_hand_ev', 0) or 0),
                    'hate_draft_rate': float(telemetry.get('hate_draft_rate', 0.0) or 0.0),
                    'hate_draft_rate_low_hand_ev': float(telemetry.get('hate_draft_rate_low_hand_ev', 0.0) or 0.0),
                })

            actual_game_id = game_instance.game_id
            game_generation = None
            try:
                gs_game = game_state.get('game') or {}
                gen_raw = gs_game.get('generation') or game_state.get('generation')
                if gen_raw is not None:
                    game_generation = int(gen_raw)
            except (TypeError, ValueError):
                pass

            result = GameResult(
                game_id=actual_game_id,
                players=players,
                duration_seconds=duration,
                completed=True,
                end_screens=end_screens,
                game_generation=game_generation,
            )
            # Push into coordinator recent lists if available
            try:
                from api.server import coordinator as api_coordinator
                if api_coordinator is not None:
                    if hasattr(api_coordinator, 'recent_end_screens'):
                        api_coordinator.recent_end_screens.append({"game_id": actual_game_id, "end_screens": end_screens})
                    if hasattr(api_coordinator, 'recent_games'):
                        api_coordinator.recent_games.append({"game_id": actual_game_id, "url": game_instance.get_public_game_url()})
            except Exception:
                logger.warning("Failed adding recent game/end screen to coordinator")
            return result
            
        except (asyncio.TimeoutError, TimeoutError):
            # Ensure all in-flight agent tasks are cancelled and awaited so gather
            # does not emit "exception was never retrieved" warnings.
            with suppress(Exception):
                self._cancel_agent_tasks_with_reason(
                    locals().get("agent_tasks", []),
                    reason="game_timeout",
                    game_id=actual_game_id,
                )
            with suppress(asyncio.CancelledError, Exception):
                if "agent_group" in locals():
                    await agent_group
            try:
                timeout_sec = float(os.getenv('TM_GAME_TIMEOUT_SEC', '300'))
            except Exception:
                timeout_sec = 300.0
            timeout_sec = max(60.0, timeout_sec)
            game_url = ""
            server_name = ""
            if game_instance is not None:
                try:
                    game_url = game_instance.get_public_game_url()
                except Exception:
                    game_url = ""
                try:
                    server_name = f"{game_instance.server.host}:{game_instance.server.port}"
                except Exception:
                    server_name = ""
            logger.warning(
                "Game %s timed out after %.0f seconds (server=%s url=%s)",
                actual_game_id,
                timeout_sec,
                server_name or "unknown",
                game_url or "n/a",
            )
            return GameResult(
                game_id=actual_game_id,
                players=[{
                    'agent_id': agent.id,
                    'rank': 4,
                    'victory_points': 0,
                    'completed': False,
                    'draft_decisions': 0,
                    'draft_decisions_low_hand_ev': 0,
                    'hate_draft_picks': 0,
                    'hate_draft_picks_low_hand_ev': 0,
                    'hate_draft_rate': 0.0,
                    'hate_draft_rate_low_hand_ev': 0.0,
                } for agent in agents],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                completed=False,
                error_message="Game timed out",
                end_screens=[]
            )
        except asyncio.CancelledError:
            # Propagate service-level cancellation, but drain child tasks first.
            with suppress(Exception):
                self._cancel_agent_tasks_with_reason(
                    locals().get("agent_tasks", []),
                    reason="tournament_cancelled",
                    game_id=actual_game_id,
                )
            with suppress(asyncio.CancelledError, Exception):
                if "agent_group" in locals():
                    await agent_group
            raise
             
        except Exception as e:
            with suppress(Exception):
                self._cancel_agent_tasks_with_reason(
                    locals().get("agent_tasks", []),
                    reason="game_failed",
                    game_id=actual_game_id,
                )
            with suppress(asyncio.CancelledError, Exception):
                if "agent_group" in locals():
                    await agent_group
            logger.error(f"Game {actual_game_id} failed: {e}")
            return GameResult(
                game_id=actual_game_id,
                players=[{
                    'agent_id': agent.id,
                    'rank': 4,
                    'victory_points': 0,
                    'completed': False,
                    'draft_decisions': 0,
                    'draft_decisions_low_hand_ev': 0,
                    'hate_draft_picks': 0,
                    'hate_draft_picks_low_hand_ev': 0,
                    'hate_draft_rate': 0.0,
                    'hate_draft_rate_low_hand_ev': 0.0,
                } for agent in agents],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                completed=False,
                error_message=str(e),
                end_screens=[]
            )
        
        finally:
            # Clean up game instance
            try:
                if game_instance is not None:
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
            'end_screens': game_result.end_screens or [],
            'game_generation': game_result.game_generation,
        }
    
    async def get_tournament_status(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """Get status of running tournament"""
        if tournament_id not in self.active_tournaments:
            return None
        
        tournament = self.active_tournaments[tournament_id]
        progress = dict(self.tournament_progress.get(tournament_id, {}))
        return {
            'tournament_id': tournament_id,
            'agent_count': len(tournament.agents),
            'games_per_matchup': tournament.games_per_matchup,
            'created_at': tournament.created_at.isoformat(),
            'status': progress.get('status', 'running'),
            'planned_games': int(progress.get('planned_games', 0)),
            'finished_games': int(progress.get('finished_games', 0)),
            'successful_games': int(progress.get('successful_games', 0)),
            'failed_games': int(progress.get('failed_games', 0)),
        }

    def get_progress_snapshot(self) -> Dict[str, Any]:
        """Aggregate in-flight tournament progress for dashboard polling."""
        entries = list(self.tournament_progress.values())
        planned_games = sum(int(e.get("planned_games", 0)) for e in entries)
        finished_games = sum(int(e.get("finished_games", 0)) for e in entries)
        successful_games = sum(int(e.get("successful_games", 0)) for e in entries)
        failed_games = sum(int(e.get("failed_games", 0)) for e in entries)
        completion_rate = (float(finished_games) / float(planned_games)) if planned_games > 0 else 0.0
        return {
            "active_tournaments": len(entries),
            "planned_games": int(planned_games),
            "finished_games": int(finished_games),
            "successful_games": int(successful_games),
            "failed_games": int(failed_games),
            "completion_rate": completion_rate,
        }
