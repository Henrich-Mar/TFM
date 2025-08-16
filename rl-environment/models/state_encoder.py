"""
State Encoder - Converts game state to neural network input
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import json
import logging

logger = logging.getLogger(__name__)

class StateEncoder:
    def __init__(self, state_size: int = 512):
        self.state_size = state_size
        
        # Card mappings (top cards by frequency)
        self.common_cards = self._get_common_cards()
        self.card_to_index = {card: i for i, card in enumerate(self.common_cards)}
        
        # Optional: load authoritative card metadata exported from TS server
        self.card_metadata_by_name: Dict[str, Dict[str, Any]] = self._load_card_metadata()
    
    def _get_common_cards(self) -> List[str]:
        """Get list of most common cards for encoding"""
        # This would normally be loaded from game data
        # For now, using a representative sample
        return [
            'PowerPlant', 'Mine', 'Research', 'Colony', 'City', 'Greenery',
            'SolarPower', 'Livestock', 'Fish', 'Microbes', 'Animals',
            'Trees', 'PowerSupplyConsortium', 'EnergyTapping', 'NuclearPower',
            'Geothermal', 'SolarWindPower', 'WaveEnergy', 'Zeppelins',
            'BuildingIndustries', 'SpaceElevator', 'LunarBeam', 'MassConverter',
            'PhysicsComplex', 'ResearchCoordination', 'TechnologyDemonstration'
        ] + [f"Card_{i}" for i in range(200)]  # Placeholder for remaining cards
    
    def encode(self, player_state: Dict[str, Any]) -> np.ndarray:
        """Encode complete game state into feature vector"""
        features = []
        try:
            # Extract main sections
            game_state = player_state.get('game', {})
            player = player_state.get('thisPlayer', {})
            
            # Global parameters (4 features, normalized)
            features.extend(self._encode_global_parameters(game_state))
            
            # Player resources (8 features, normalized)
            features.extend(self._encode_player_resources(player))
            
            # Player production (8 features, normalized)
            features.extend(self._encode_player_production(player))
            
            # Player tableau/played cards (50 features)
            features.extend(self._encode_tableau(player))
            
            # Cards in hand (50 features)
            features.extend(self._encode_hand_cards(player_state))
            
            # Board state (100 features)
            features.extend(self._encode_board_state(game_state))
            
            # Game phase and generation (10 features)
            features.extend(self._encode_game_phase(game_state))
            
            # Opponents state (simplified, 100 features)
            features.extend(self._encode_opponents_state(player_state.get('players', []), player))
            
            # Current action context (50 features)
            features.extend(self._encode_action_context(player_state))
            
            # Awards and milestones (32 features)
            features.extend(self._encode_awards_milestones(game_state))
            
            # Pad or truncate to exact size
            features = features[:self.state_size]
            while len(features) < self.state_size:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error encoding state: {e}")
            # Return zero vector as fallback
            return np.zeros(self.state_size, dtype=np.float32)
    
    

    def _encode_global_parameters(self, game_state: Dict[str, Any]) -> List[float]:
        """Encode global terraforming parameters and Moon parameters with proper discrete value handling"""
        # Oxygen: discrete values [0, 2, 4, 6, 8, 10, 12, 14] - 8 possible values
        oxygen_level = game_state.get('oxygenLevel', 0)
        # Convert to categorical index (0-7)
        oxygen_index = oxygen_level // 2
        oxygen = float(oxygen_index) / 7.0  # Normalize categorical index

        # Temperature: discrete values from -30 to 8 in 2-step increments
        # [-30, -28, -26, ..., 0, ..., 6, 8] - 20 possible values
        temp_level = game_state.get('temperature', -30)
        # Convert to categorical index (0-19)
        temp_index = (temp_level + 30) // 2
        temperature = float(temp_index) / 19.0  # Normalize categorical index

        # Venus: discrete values from 0 to 30 in 2-step increments
        # [0, 2, 4, ..., 28, 30] - 16 possible values
        venus_level = game_state.get('venusScaleLevel', 0)
        # Convert to categorical index (0-15)
        venus_index = venus_level // 2
        venus = float(venus_index) / 15.0  # Normalize categorical index

        # Generation: typically 1-14
        generation = game_state.get('generation', 1) / 14.0

        # Moon parameters: logisticsRate, miningRate, habitatRate (each 0-8, raised by one)
        moon = game_state.get('moon', {})
        logistics_rate = moon.get('logisticsRate', 0)
        mining_rate = moon.get('miningRate', 0)
        habitat_rate = moon.get('habitatRate', 0)
        # Each is 0-8, so normalize by dividing by 8.0
        logistics_norm = float(logistics_rate) / 8.0
        mining_norm = float(mining_rate) / 8.0
        habitat_norm = float(habitat_rate) / 8.0

        logger.info(
            f"Oxygen: {oxygen} (level {oxygen_level}, index {oxygen_index}), "
            f"Temperature: {temperature} (level {temp_level}, index {temp_index}), "
            f"Venus: {venus} (level {venus_level}, index {venus_index}), "
            f"Moon: logistics {logistics_norm} (raw {logistics_rate}), "
            f"mining {mining_norm} (raw {mining_rate}), "
            f"habitat {habitat_norm} (raw {habitat_rate})"
            f"Generation: {generation}, "
        )

        return [
            oxygen,
            temperature,
            venus,
            generation,
            logistics_norm,
            mining_norm,
            habitat_norm,
        ]

    
    def _encode_player_resources(self, player: Dict[str, Any]) -> List[float]:
        """Encode player resource counts"""
        # Reasonable maximums; TR uses (TR - 20) with max ≈ 43 (from 20 → 63)
        max_resources = [300, 100, 100, 100, 100, 100, 90, 100]
        resources = [
            player.get('megaCredits', 0),
            player.get('steel', 0),
            player.get('titanium', 0),
            player.get('plants', 0),
            player.get('energy', 0),
            player.get('heat', 0),
            # Normalize TR from base 20; clamp at lower bound
            max(player.get('terraformRating', 20) - 20, 0),
            player.get('victoryPointsBreakdown', {}).get('total', 0)
        ]
        logger.info(f"Resources: {resources} for player {player.get('id')}")
        # Normalize
        normalized = [min(res / max_res, 1.0) for res, max_res in zip(resources, max_resources)]
        return normalized
    
    def _encode_player_production(self, player: Dict[str, Any]) -> List[float]:
        """Encode player production values"""
        max_production = [50, 20, 20, 20, 20, 20, 10, 10]  # Reasonable maximums
        
        prod_values = [
            player.get('megaCreditProduction', 0),
            player.get('steelProduction', 0),
            player.get('titaniumProduction', 0),
            player.get('plantProduction', 0),
            player.get('energyProduction', 0),
            player.get('heatProduction', 0),
            0,  # Card draw production (not available)
            0   # Science tags or similar (not available)
        ]
        logger.info(f"Production: {prod_values} for player {player.get('id')}")
        # Normalize and handle negative production
        normalized = [(prod + 10) / (max_prod + 10) for prod, max_prod in zip(prod_values, max_production)]
        return normalized
    
    def _encode_tableau(self, player: Dict[str, Any]) -> List[float]:
        """Encode played cards (tableau)"""
        tableau = player.get('tableau', [])
        encoding = [0.0] * 50
        logger.info(f"Tableau: {tableau} for player {player.get('id')}")
        # Count cards by type/category
        card_counts = {}
        for card in tableau:
            card_name = card.get('name', '')
            if card_name in self.card_to_index:
                idx = self.card_to_index[card_name]
                if idx < 25:  # Only use first 25 slots for specific cards
                    encoding[idx] = 1.0
            
            # Count by all 17 tags
            tags = self._get_card_tags(card_name, fallback=card.get('tags', {}))
            # Building tag (index 25)
            if tags.get('Building', 0) > 0:
                encoding[25] += 0.1
            # Space tag (index 26)
            if tags.get('Space', 0) > 0:
                encoding[26] += 0.1
            # Science tag (index 27)
            if tags.get('Science', 0) > 0:
                encoding[27] += 0.1
            # Power tag (index 28)
            if tags.get('Power', 0) > 0:
                encoding[28] += 0.1
            # Earth tag (index 29)
            if tags.get('Earth', 0) > 0:
                encoding[29] += 0.1
            # Jovian tag (index 30)
            if tags.get('Jovian', 0) > 0:
                encoding[30] += 0.1
            # Venus tag (index 31)
            if tags.get('Venus', 0) > 0:
                encoding[31] += 0.1
            # Plant tag (index 32)
            if tags.get('Plant', 0) > 0:
                encoding[32] += 0.1
            # Microbe tag (index 33)
            if tags.get('Microbe', 0) > 0:
                encoding[33] += 0.1
            # Animal tag (index 34)
            if tags.get('Animal', 0) > 0:
                encoding[34] += 0.1
            # City tag (index 35)
            if tags.get('City', 0) > 0:
                encoding[35] += 0.1
            # Moon tag (index 36)
            if tags.get('Moon', 0) > 0:
                encoding[36] += 0.1
            # Mars tag (index 37)
            if tags.get('Mars', 0) > 0:
                encoding[37] += 0.1
            # Crime tag (index 38)
            if tags.get('Crime', 0) > 0:
                encoding[38] += 0.1
            # Wild tag (index 39)
            if tags.get('Wild', 0) > 0:
                encoding[39] += 0.1
            # Event tag (index 40)
            if tags.get('Event', 0) > 0:
                encoding[40] += 0.1
            # Clone tag (index 41)
            if tags.get('Clone', 0) > 0:
                encoding[41] += 0.1
        
        # Normalize tag counts
        for i in range(25, 42):
            encoding[i] = min(encoding[i], 1.0)
        
        # Card resources (microbes, animals, etc.)
        for card in tableau:
            resources = card.get('resources', 0)
            if resources > 0:
                encoding[42] += min(resources / 10.0, 1.0)  # Normalized resource count
        
        return encoding
    
    def _encode_hand_cards(self, player_state: Dict[str, Any]) -> List[float]:
        """Encode cards in hand"""
        hand = player_state.get('cardsInHand', [])
        encoding = [0.0] * 50
        logger.info(f"Hand: {hand} for player {player_state.get('id')}")
        # Basic hand info
        encoding[0] = min(len(hand) / 10.0, 1.0)  # Hand size
        
        # Card costs
        total_cost = sum(card.get('calculatedCost', 0) for card in hand)
        encoding[1] = min(total_cost / 100.0, 1.0)
        logger.info(f"Total cost: {total_cost} for player {player_state.get('id')}")
        
        # Estimate affordable cards given current resources (rough heuristic)
        player = player_state.get('thisPlayer', {})
        mc = float(player.get('megaCredits', 0) or 0)
        steel = float(player.get('steel', 0) or 0)
        titanium = float(player.get('titanium', 0) or 0)
        # Typical conversion values (base game defaults)
        steel_value = 2.0
        titanium_value = 3.0
        effective_budget = mc + steel * steel_value + titanium * titanium_value
        affordable_count = 0
        for card in hand:
            cost = float(card.get('calculatedCost', 0) or 0)
            if cost <= effective_budget:
                affordable_count += 1
        encoding[2] = min(affordable_count / 10.0, 1.0)
        # Card types in hand - all 17 tags
        for card in hand:
            tags = self._get_card_tags(card.get('name', ''), fallback=card.get('tags', {}))
            # Building tag (index 3)
            if tags.get('Building', 0) > 0:
                encoding[3] += 0.1
            # Space tag (index 4)
            if tags.get('Space', 0) > 0:
                encoding[4] += 0.1
            # Science tag (index 5)
            if tags.get('Science', 0) > 0:
                encoding[5] += 0.1
            # Power tag (index 6)
            if tags.get('Power', 0) > 0:
                encoding[6] += 0.1
            # Earth tag (index 7)
            if tags.get('Earth', 0) > 0:
                encoding[7] += 0.1
            # Jovian tag (index 8)
            if tags.get('Jovian', 0) > 0:
                encoding[8] += 0.1
            # Venus tag (index 9)
            if tags.get('Venus', 0) > 0:
                encoding[9] += 0.1
            # Plant tag (index 10)
            if tags.get('Plant', 0) > 0:
                encoding[10] += 0.1
            # Microbe tag (index 11)
            if tags.get('Microbe', 0) > 0:
                encoding[11] += 0.1
            # Animal tag (index 12)
            if tags.get('Animal', 0) > 0:
                encoding[12] += 0.1
            # City tag (index 13)
            if tags.get('City', 0) > 0:
                encoding[13] += 0.1
            # Moon tag (index 14)
            if tags.get('Moon', 0) > 0:
                encoding[14] += 0.1
            # Mars tag (index 15)
            if tags.get('Mars', 0) > 0:
                encoding[15] += 0.1
            # Crime tag (index 16)
            if tags.get('Crime', 0) > 0:
                encoding[16] += 0.1
            # Wild tag (index 17)
            if tags.get('Wild', 0) > 0:
                encoding[17] += 0.1
            # Event tag (index 18)
            if tags.get('Event', 0) > 0:
                encoding[18] += 0.1
            # Clone tag (index 19)
            if tags.get('Clone', 0) > 0:
                encoding[19] += 0.1
        
        # Normalize
        for i in range(3, 20):
            encoding[i] = min(encoding[i], 1.0)
        logger.info(f"Hand encoding: {encoding} for player {player_state.get('id')}")
        return encoding
    
    def _encode_board_state(self, game_state: Dict[str, Any]) -> List[float]:
        """Encode board state with spatial reasoning"""
        encoding = [0.0] * 100
        
        # Ocean tiles
        ocean_count = game_state.get('oceans', 0)
        encoding[0] = ocean_count / 9.0  # Max 9 oceans
        # Board spaces (simplified)
        spaces = game_state.get('spaces', [])
        
        # Count different tile types
        city_count = 0
        greenery_count = 0
        special_tile_count = 0
        
        for space in spaces:
            tile_type = space.get('tileType')
            if tile_type:
                if tile_type == 'CITY':
                    city_count += 1
                elif tile_type == 'GREENERY':
                    greenery_count += 1
                else:
                    special_tile_count += 1
        
        encoding[1] = min(city_count / 20.0, 1.0)
        encoding[2] = min(greenery_count / 30.0, 1.0)
        encoding[3] = min(special_tile_count / 20.0, 1.0)
        
        # Available spaces by type (simplified)
        available_land = sum(1 for space in spaces if not space.get('tileType') and space.get('spaceType') == 'LAND')
        available_ocean = sum(1 for space in spaces if not space.get('tileType') and space.get('spaceType') == 'OCEAN')
        
        encoding[4] = min(available_land / 50.0, 1.0)
        encoding[5] = min(available_ocean / 9.0, 1.0)
        
        # Add spatial reasoning features
        city_positions = []
        greenery_positions = []
        ocean_positions = []
        
        for space in spaces:
            if space.get('tileType') == 'CITY':
                city_positions.append(self._get_space_coordinates(space))
            elif space.get('tileType') == 'GREENERY':
                greenery_positions.append(self._get_space_coordinates(space))
            elif space.get('tileType') == 'OCEAN':
                ocean_positions.append(self._get_space_coordinates(space))
        
        # Encode spatial patterns
        encoding[50] = self._calculate_city_clustering(city_positions)
        encoding[51] = self._calculate_greenery_spread(greenery_positions)
        encoding[52] = self._calculate_ocean_coverage(ocean_positions)
        
        return encoding

    def _get_space_coordinates(self, space: Dict[str, Any]) -> Tuple[int, int]:
        """Extract coordinates from space ID"""
        space_id = space.get('id', '')
        # Parse coordinates from space ID (e.g., "01" -> (0,1))
        try:
            x = int(space_id[0]) if len(space_id) > 0 else 0
            y = int(space_id[1]) if len(space_id) > 1 else 0
            return (x, y)
        except:
            return (0, 0)
    
    def _calculate_city_clustering(self, city_positions: List[Tuple[int, int]]) -> float:
        """Calculate city clustering score"""
        if len(city_positions) < 2:
            return 0.0
        
        # Calculate average distance between cities
        total_distance = 0
        count = 0
        for i in range(len(city_positions)):
            for j in range(i + 1, len(city_positions)):
                x1, y1 = city_positions[i]
                x2, y2 = city_positions[j]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        # Normalize: closer cities = higher clustering score
        return max(0.0, 1.0 - avg_distance / 10.0)
    
    def _calculate_greenery_spread(self, greenery_positions: List[Tuple[int, int]]) -> float:
        """Calculate greenery spread score"""
        if len(greenery_positions) < 2:
            return 0.0
        
        # Calculate how well greenery is spread across the board
        x_coords = [pos[0] for pos in greenery_positions]
        y_coords = [pos[1] for pos in greenery_positions]
        
        x_range = max(x_coords) - min(x_coords) if x_coords else 0
        y_range = max(y_coords) - min(y_coords) if y_coords else 0
        
        # Normalize spread (wider spread = higher score)
        return min(1.0, (x_range + y_range) / 20.0)
    
    def _calculate_ocean_coverage(self, ocean_positions: List[Tuple[int, int]]) -> float:
        """Calculate ocean coverage score"""
        if not ocean_positions:
            return 0.0
        
        # Calculate how well oceans are distributed
        # This is a simplified version - you might want to implement
        # more sophisticated ocean placement analysis
        return min(1.0, len(ocean_positions) / 9.0)
    
    def _encode_game_phase(self, game_state: Dict[str, Any]) -> List[float]:
        """Encode current game phase and timing"""
        encoding = [0.0] * 10
        
        phase = game_state.get('phase', '')
        generation = game_state.get('generation', 1)
        
        # Phase encoding (one-hot)
        phase_map = {'research': 0, 'drafting': 1, 'action': 2, 'production': 3, 'solar': 4}
        if phase.lower() in phase_map:
            encoding[phase_map[phase.lower()]] = 1.0
        
        # Generation progress
        encoding[5] = min(generation / 14.0, 1.0)
        
        # Turn/round info
        active_player = game_state.get('activePlayer', '')
        encoding[6] = 1.0 if active_player else 0.0
        
        return encoding
    
    def _encode_opponents_state(self, players: List[Dict[str, Any]], current_player: Dict[str, Any]) -> List[float]:
        """Encode simplified opponent information"""
        encoding = [0.0] * 100
        
        current_player_id = current_player.get('id', '')
        
        opponent_idx = 0
        for player in players:
            if player.get('id') != current_player_id and opponent_idx < 3:  # Max 3 opponents
                base_idx = opponent_idx * 20
                
                # Basic opponent info
                encoding[base_idx] = min(player.get('megaCredits', 0) / 300, 1.0)
                # TR normalized similar to self: (TR - 20) / 43, clamped to [0, 1]
                opp_tr = max(player.get('terraformRating', 20) - 20, 0)
                encoding[base_idx + 1] = min(opp_tr / 80.0, 1.0)
                encoding[base_idx + 2] = min(len(player.get('tableau', [])) / 20.0, 1.0)
                encoding[base_idx + 3] = min(player.get('victoryPointsBreakdown', {}).get('total', 0) / 50.0, 1.0)
                
                # Production
                encoding[base_idx + 4] = min(player.get('megaCreditProduction', 0) / 30.0, 1.0)
                encoding[base_idx + 5] = min(player.get('steelProduction', 0) / 20.0, 1.0)
                encoding[base_idx + 6] = min(player.get('titaniumProduction', 0) / 20.0, 1.0)
                encoding[base_idx + 7] = min(player.get('plantProduction', 0) / 20.0, 1.0)
                encoding[base_idx + 8] = min(player.get('energyProduction', 0) / 20.0, 1.0)
                encoding[base_idx + 9] = min(player.get('heatProduction', 0) / 20.0, 1.0)
                
                opponent_idx += 1
        
        return encoding
    
    def _encode_action_context(self, player_state: Dict[str, Any]) -> List[float]:
        """Encode current action context and waiting input"""
        encoding = [0.0] * 50
        
        waiting_for = player_state.get('waitingFor')
        if waiting_for:
            input_type = waiting_for.get('type', '')
            
            # Input type encoding
            type_map = {
                'card': 0, 'or': 1, 'selectSpace': 2, 'selectPayment': 3,
                'selectOption': 4, 'selectAmount': 5, 'selectPlayer': 6
            }
            
            if input_type in type_map:
                encoding[type_map[input_type]] = 1.0
            
            # Available options count
            if input_type == 'or':
                options = waiting_for.get('options', [])
                encoding[10] = min(len(options) / 10.0, 1.0)
                # Encode specific known 'or' options to help agent decide
                for i, option in enumerate(options):
                    title_value = option.get('title', '')
                    if isinstance(title_value, dict):
                        title_value = title_value.get('message', '')
                    title = str(title_value).lower()
                    if 'temperature' in title:
                        encoding[12] = 1.0
                        encoding[13] = (i + 1) / 5.0 # Encode position
                    elif 'oxygen' in title:
                        encoding[14] = 1.0
                        encoding[15] = (i + 1) / 5.0
                    elif 'venus' in title:
                        encoding[16] = 1.0
                        encoding[17] = (i + 1) / 5.0
                    # Additional common options
                    if 'standard project' in title:
                        encoding[20] = 1.0
                        encoding[21] = (i + 1) / 10.0
                    if 'pass' in title:
                        encoding[22] = 1.0
                        encoding[23] = (i + 1) / 10.0
                    if 'sell patent' in title:
                        encoding[24] = 1.0
                        encoding[25] = (i + 1) / 10.0
                    if 'fund an award' in title or 'fund award' in title:
                        encoding[26] = 1.0
                        encoding[27] = (i + 1) / 10.0
            elif input_type == 'card':
                cards = waiting_for.get('cards', [])
                encoding[11] = min(len(cards) / 20.0, 1.0)
        logger.info(f"Action context: {encoding} for player {player_state.get('id')}")
        return encoding
    
    def _encode_awards_milestones(self, game_state: Dict[str, Any]) -> List[float]:
        """Encode awards and milestones state"""
        encoding = [0.0] * 32
        
        # Milestones (16 slots)
        milestones = game_state.get('milestones', [])
        for i, milestone in enumerate(milestones[:16]):
            if milestone.get('playerName'):
                encoding[i] = 1.0
        
        # Awards (16 slots)
        awards = game_state.get('awards', [])
        for i, award in enumerate(awards[:16]):
            if award.get('playerName'):
                encoding[16 + i] = 1.0
        logger.info(f"Awards and milestones: {encoding} for player {game_state.get('id')}")
        return encoding

    def _infer_tags_from_name(self, card_name: str) -> Dict[str, int]:
        """Heuristic tag inference when structured tags are missing.
        Returns a dict like {TagName: 1} for detected tags.
        """
        name = (card_name or '').lower()
        tags: Dict[str, int] = {}
        def set_tag(key: str):
            tags[key] = 1

        # Very rough heuristics to avoid all-zero encodings when tags are missing
        if 'city' in name or 'noctis' in name:
            set_tag('City')
        if 'greenery' in name:
            set_tag('Plant')
        if 'venus' in name:
            set_tag('Venus')
        if 'jovian' in name or 'jupiter' in name or 'ganymede' in name:
            set_tag('Jovian')
        if 'microbe' in name or 'bacteria' in name or 'extremophile' in name:
            set_tag('Microbe')
        if 'animal' in name or 'livestock' in name or 'pets' in name:
            set_tag('Animal')
        if 'power' in name or 'energy' in name:
            set_tag('Power')
        if 'science' in name or 'lab' in name or 'research' in name or 'complex' in name:
            set_tag('Science')
        if 'earth' in name:
            set_tag('Earth')
        if 'space' in name or 'asteroid' in name or 'orbital' in name or 'satellite' in name:
            set_tag('Space')
        if 'moon' in name or 'lunar' in name:
            set_tag('Moon')
        if 'mars' in name or 'martian' in name:
            set_tag('Mars')
        if 'event' in name:
            set_tag('Event')

        return tags

    def _load_card_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load card metadata from JSON if available. The JSON should be a
        mapping of card display name to an object including at least {tags, type, cost}.
        Looks for TM_CARD_METADATA_PATH env var, else tries a default location.
        """
        path = os.getenv('TM_CARD_METADATA_PATH')
        if not path:
            # Try a repo-relative default
            candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     'terraforming-mars', 'card_metadata.json')
            path = candidate
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Normalize tag casing to our expected TitleCase keys used above
                normalized: Dict[str, Dict[str, Any]] = {}
                for name, meta in (data or {}).items():
                    tags = meta.get('tags') or []
                    # Accept both enums and strings; convert to TitleCase keys used in encoder
                    norm_tags = []
                    for t in tags:
                        ts = str(t)
                        # If enum-like 'plant' → 'Plant'
                        norm_tags.append(ts.strip().capitalize())
                    meta['tags'] = norm_tags
                    normalized[name] = meta
                logger.info(f"Loaded card metadata for {len(normalized)} cards from {path}")
                return normalized
        except Exception as e:
            logger.warning(f"Failed to load card metadata from {path}: {e}")
        return {}

    def _get_card_tags(self, card_name: str, fallback: Any = None) -> Dict[str, int]:
        """Return tag presence mapping for a given card name using loaded metadata when available.
        Fallbacks to provided 'fallback' tags or name-based heuristics.
        """
        if card_name and self.card_metadata_by_name:
            meta = self.card_metadata_by_name.get(card_name)
            if meta and meta.get('tags'):
                tag_map: Dict[str, int] = {}
                for tag in meta['tags']:
                    # meta tags expected in TitleCase: e.g., 'Plant', 'Building'
                    tag_map[tag] = 1
                return tag_map
        # Fallback to provided structured tags if any
        if isinstance(fallback, dict) and fallback:
            return fallback
        # Final fallback to heuristics
        return self._infer_tags_from_name(card_name)
    

    