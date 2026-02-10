"""
Game Interface - Handles communication with Terraforming Mars game servers
"""
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class GameServer:
    host: str
    port: int
    active_games: int = 0
    healthy: bool = True
    last_health_check: Optional[datetime] = None

class GameInstance:
    def __init__(self, game_id: str, server: GameServer, session: aiohttp.ClientSession):
        self.game_id = game_id
        self.server = server
        self.session = session
        self.player_ids: Dict[str, str] = {}  # player_name -> player_id
        self.base_url = f"http://{server.host}:{server.port}"
        self.spectator_id: Optional[str] = None

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
        return str(public_base).rstrip('/')

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
            async with self.session.get(f"{self.base_url}/api/game", 
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
                    
        except Exception as e:
            logger.error(f"Failed to join player {player_name}: {e}")
            raise
    
    async def get_player_state(self, player_id: str) -> Dict[str, Any]:
        """Get current state for a specific player"""
        try:
            async with self.session.get(f"{self.base_url}/api/player", 
                                      params={'id': player_id}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise ValueError(f"Failed to get player state: {response.status}")
        except Exception as e:
            logger.error(f"Failed to get player state for {player_id}: {e}")
            raise
    
    async def send_player_input(self, player_id: str, input_data: Dict[str, Any]) -> bool:
        """Send player input to the game"""
        try:
            async with self.session.post(f"{self.base_url}/player/input",
                                       params={'id': player_id},
                                       json=input_data,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    return True
                else:
                    response_text = await response.text()
                    try:
                        payload_preview = json.dumps(input_data, ensure_ascii=True, separators=(',', ':'))
                    except Exception:
                        payload_preview = str(input_data)
                    if len(payload_preview) > 800:
                        payload_preview = payload_preview[:800] + "...(truncated)"
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
                    return False
        except Exception as e:
            logger.error(f"Failed to send input for player {player_id}: {e}")
            return False
    
    async def get_final_state(self) -> Dict[str, Any]:
        """Get final game state after completion"""
        try:
            async with self.session.get(f"{self.base_url}/api/game", 
                                      params={'id': self.game_id}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise ValueError(f"Failed to get final state: {response.status}")
        except Exception as e:
            logger.error(f"Failed to get final state: {e}")
            raise
    
    async def cleanup(self):
        """Clean up game resources"""
        # In a production environment, you might want to delete the game
        # For now, we'll just mark the server as having one less active game
        self.server.active_games -= 1

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
        # Optional cross-component scratchpad for latest game URLs
        self.recent_games: List[Dict[str, str]] = []
        self.base_game_options = self._load_game_options()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close shared HTTP session used for game/server API calls."""
        if self.session:
            await self.session.close()
            self.session = None

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
                'moon': False,
                'pathfinders': False,
                'prelude': True,
                'prelude2': False,
                'promo': False,
                'starwars': False,
                'turmoil': False,
                'underworld': False,
                'venus': False,
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
            'requiresMoonTrackCompletion': False,
            'requiresVenusTrackCompletion': False,
            'seed': "12345",
            'showOtherPlayersVP': False,
            'showTimers': True,
            'shuffleMapOption': True,
            'solarPhaseOption': True,
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
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        
        results = {}
        
        for server in self.servers:
            try:
                async with self.session.get(f"http://{server.host}:{server.port}/") as response:
                    server.healthy = response.status == 200
                    server.last_health_check = datetime.now()
                    results[f"{server.host}:{server.port}"] = server.healthy
            except Exception as e:
                server.healthy = False
                server.last_health_check = datetime.now()
                results[f"{server.host}:{server.port}"] = False
                logger.warning(f"Health check failed for {server.host}:{server.port}: {e}")
        
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
    
    async def create_game(self, 
                         game_id: str,
                         player_names: List[str],
                         game_options: Dict[str, Any]) -> GameInstance:
        """Create a new game on the best available server"""
        server = self._get_best_server()
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        
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
        
        try:
            base_url = f"http://{server.host}:{server.port}"
            
            # Create the game
            async with self.session.post(f"{base_url}/api/creategame", 
                                       json=create_request,
                                       headers={'Content-Type': 'application/json'}) as response:
                if response.status != 200:
                    response_text = await response.text()
                    raise RuntimeError(f"Failed to create game: {response.status} - {response_text}")
                
                # Get the created game data
                game_data = await response.json()
                
                # Extract game ID from response (authoritative)
                actual_game_id = game_data.get('id')
                if not actual_game_id:
                    raise RuntimeError("Game ID not found in response")

            server.active_games += 1
            
            # Create game instance
            game_instance = GameInstance(actual_game_id, server, self.session)
            
            # Initialize player IDs
            for i, player_name in enumerate(player_names):
                try:
                    await game_instance.join_player(player_name)
                except Exception as e:
                    logger.warning(f"Failed to join player {player_name}: {e}")
            
            logger.info(f"Created game {actual_game_id} on {server.host}:{server.port}")
            return game_instance
            
        except Exception as e:
            logger.error(f"Failed to create game on {server.host}:{server.port}: {e}")
            raise
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get statistics for all servers"""
        stats = {
            'total_servers': len(self.servers),
            'healthy_servers': sum(1 for s in self.servers if s.healthy),
            'total_active_games': sum(s.active_games for s in self.servers),
            'servers': []
        }
        
        for server in self.servers:
            server_info = {
                'host': server.host,
                'port': server.port,
                'healthy': server.healthy,
                'active_games': server.active_games,
                'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None
            }
            stats['servers'].append(server_info)
        
        return stats
