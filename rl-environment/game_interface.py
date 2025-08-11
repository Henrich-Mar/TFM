"""
Game Interface - Handles communication with Terraforming Mars game servers
"""
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import random

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
                    logger.error(f"Failed to send input for player {player_id}. Status: {response.status}, Response: {response_text}")
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
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
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
        
        # Prepare game creation request with detailed options
        create_request = {
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
            'players': [
                {
                    'name': name,
                    'color': ['red', 'blue', 'green', 'yellow'][i % 4],
                    'beginner': False,
                    'handicap': 0,
                    'first': False
                }
                for i, name in enumerate(player_names)
            ],
            'politicalAgendasExtension': "Standard",
            'preludeDraftVariant': True,
            'randomFirstPlayer': True,
            'randomMA': "No randomization",
            'removeNegativeGlobalEventsOption': False,
            'requiresMoonTrackCompletion': True,
            'requiresVenusTrackCompletion': True,
            'seed': random.random(),
            'showOtherPlayersVP': False,
            'showTimers': True,
            'shuffleMapOption': True,
            'solarPhaseOption': True,
            'soloTR': False,
            'startingCeos': 3,
            'startingCorporations': 2,
            'startingPreludes': 4,
            'twoCorpsVariant': False,
            'undoOption': False,
            **game_options
        }
        
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