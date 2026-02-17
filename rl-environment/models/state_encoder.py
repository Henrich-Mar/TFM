"""
State Encoder - Converts game state to neural network input
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import logging

logger = logging.getLogger(__name__)

class StateEncoder:
    # Process-local cache keyed by metadata file path.
    _CARD_METADATA_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
    # Mirrors key tile groupings from terraforming-mars/src/common/TileType.ts.
    _CITY_TILE_TYPES = {2, 3, 20, 37, 43}
    _GREENERY_TILE_TYPES = {0, 36}
    _OCEAN_TILE_TYPES = {1, 20, 21, 22, 36, 43}
    
    # Fixed order of all possible milestones (sorted alphabetically for consistency)
    # This ensures each milestone always appears at the same index for PPO auxiliary head
    _ALL_MILESTONES = [
        'Agronomist', 'Architect', 'Briber', 'Builder', 'Builder7',
        'C. Forester', 'Capitalist', 'Coastguard', 'Colonizer', 'Diversifier',
        'Ecologist', 'Economizer', 'Energizer', 'Engineer', 'Farmer',
        'Firestarter', 'Forester', 'Fundraiser', 'Gambler', 'Gardener',
        'Generalist', 'Geologist', 'Hoverlord', 'Hydrologist', 'Irrigator',
        'Land Specialist', 'Landshaper', 'Legend', 'Legend4', 'Lobbyist',
        'Lunarchitect', 'Martian', 'Mayor', 'Metallurgist', 'Minimalist',
        'Networker', 'One Giant Step', 'Philantropist', 'Pioneer', 'Pioneer4',
        'Planetologist', 'Planner', 'Polar Explorer', 'Producer', 'Purifier',
        'Researcher', 'Rim Settler', 'Risktaker', 'Smith', 'Spacefarer',
        'Spacefarer4', 'Specialist', 'Sponsor', 'T. Collector', 'Tactician',
        'Tactician4', 'Terra Pioneer', 'Terraformer', 'Terraformer29', 'Terran',
        'Terran5', 'Thawer', 'Trader', 'Tradesman', 'Tropicalist',
        'Tunneler', 'Tycoon', 'Tycoon10', 'V. Electrician', 'V. Spacefarer',
    ]
    
    # Fixed order of all possible awards (sorted alphabetically for consistency)
    _ALL_AWARDS = [
        'A. Engineer', 'A. Manufacturer', 'A. Zoologist', 'Administrator', 'Banker',
        'Benefactor', 'Biologist', 'Blacksmith', 'Botanist', 'Celebrity',
        'Collector', 'Constructor', 'Contractor', 'Cosmic Settler', 'Cultivator',
        'Curator', 'Desert Settler', 'Edgedancer', 'Electrician', 'Entrepreneur',
        'Estate Dealer', 'Excavator', 'Excentric', 'Forecaster', 'Founder',
        'Full Moon', 'Highlander', 'Incorporator', 'Industrialist', 'Investor',
        'Kingpin', 'Landlord', 'Landscaper', 'Lunar Magnate', 'Magnate',
        'Manufacturer', 'Metropolist', 'Miner', 'Mogul', 'Naturalist',
        'Politician', 'Promoter', 'Rugged', 'Scientist', 'Space Baron',
        'Suburbian', 'T. Politician', 'Thermalist', 'Tourist', 'Traveller',
        'Urbanist', 'Venuphile', 'Visionary', 'Voyager', 'Warmonger',
        'Zoologist',
    ]

    def __init__(self, state_size: int = 1024):
        self.state_size = state_size
        
        # Load authoritative card metadata first; derive common_cards from it when available
        self.card_metadata_by_name: Dict[str, Dict[str, Any]] = self._load_card_metadata()
        self.common_cards = self._get_common_cards(self.card_metadata_by_name)
        self.card_to_index = {card: i for i, card in enumerate(self.common_cards)}

    def _space_type_lower(self, value: Any) -> str:
        return str(value or '').strip().lower()

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if isinstance(value, bool):
                return None
            return int(value)
        except Exception:
            return None

    def _tile_flags(self, space: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Return (is_city, is_greenery, is_ocean) for a board space tile."""
        tile_raw = space.get('tileType')
        tile_num = self._safe_int(tile_raw)
        tile_name = str(tile_raw or '').strip().lower()
        tile_name = tile_name.replace('-', '_').replace(' ', '_')

        is_city = False
        is_greenery = False
        is_ocean = False

        if tile_num is not None:
            is_city = tile_num in self._CITY_TILE_TYPES
            is_greenery = tile_num in self._GREENERY_TILE_TYPES
            is_ocean = tile_num in self._OCEAN_TILE_TYPES

        if tile_name:
            if 'city' in tile_name or tile_name in ('capital', 'new_holland'):
                is_city = True
            if 'greenery' in tile_name or 'wetland' in tile_name:
                is_greenery = True
            if 'ocean' in tile_name or 'wetland' in tile_name:
                is_ocean = True

        return is_city, is_greenery, is_ocean
    
    def _get_common_cards(
        self, card_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """Get list of cards for encoding. Uses card names from card_metadata when
        available; otherwise falls back to a hardcoded sample + placeholders.
        """
        if card_metadata:
            return sorted(card_metadata.keys())
        return [
            'PowerPlant', 'Mine', 'Research', 'Colony', 'City', 'Greenery',
            'SolarPower', 'Livestock', 'Fish', 'Microbes', 'Animals',
            'Trees', 'PowerSupplyConsortium', 'EnergyTapping', 'NuclearPower',
            'Geothermal', 'SolarWindPower', 'WaveEnergy', 'Zeppelins',
            'BuildingIndustries', 'SpaceElevator', 'LunarBeam', 'MassConverter',
            'PhysicsComplex', 'ResearchCoordination', 'TechnologyDemonstration'
        ] + [f"Card_{i}" for i in range(200)]  # Fallback when card_metadata not loaded
    
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
            features.extend(self._encode_board_state(game_state, player))
            
            # Game phase and generation (10 features)
            features.extend(self._encode_game_phase(game_state))
            
            # Opponents state (simplified, 100 features)
            features.extend(self._encode_opponents_state(player_state.get('players', []), player))
            
            # Current action context (50 features)
            features.extend(self._encode_action_context(player_state))
            
            # Awards and milestones (504 features: 70 milestones * 4 + 56 awards * 4)
            # For each milestone/award: [exists_in_game, I_have_it, opponent_has_it, my_progress_normalized]
            features.extend(self._encode_awards_milestones(game_state, player))
            
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

        logger.debug(
            "Global params: oxygen=%.3f(level=%s,index=%s), temp=%.3f(level=%s,index=%s), "
            "venus=%.3f(level=%s,index=%s), moon(logistics=%.3f raw=%s, mining=%.3f raw=%s, "
            "habitat=%.3f raw=%s), generation=%.3f",
            oxygen, oxygen_level, oxygen_index,
            temperature, temp_level, temp_index,
            venus, venus_level, venus_index,
            logistics_norm, logistics_rate,
            mining_norm, mining_rate,
            habitat_norm, habitat_rate,
            generation,
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
        logger.debug("Resources: %s for player %s", resources, player.get('id'))
        # Normalize
        normalized = [min(res / max_res, 1.0) for res, max_res in zip(resources, max_resources)]
        return normalized
    
    def _encode_player_production(self, player: Dict[str, Any]) -> List[float]:
        """Encode player production values"""
        max_production = [100, 30, 30, 30, 30, 30]  # Reasonable maximums
        
        prod_values = [
            player.get('megaCreditProduction', 0),
            player.get('steelProduction', 0),
            player.get('titaniumProduction', 0),
            player.get('plantProduction', 0),
            player.get('energyProduction', 0),
            player.get('heatProduction', 0)
        ]
        logger.debug("Production: %s for player %s", prod_values, player.get('id'))
        # Normalize and handle negative production
        normalized = [(prod + 10) / (max_prod + 10) for prod, max_prod in zip(prod_values, max_production)]
        return normalized
    
    def _encode_tableau(self, player: Dict[str, Any]) -> List[float]:
        """Encode played cards (tableau)"""
        tableau = player.get('tableau', [])
        encoding = [0.0] * 50
        logger.debug("Tableau: %s for player %s", tableau, player.get('id'))
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
        """Encode cards in hand with aggregate and top-K card-level features."""
        encoding = [0.0] * 50

        hand_raw = player_state.get('cardsInHand', [])
        hand_cards = [card for card in hand_raw if isinstance(card, dict)]
        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '')
        prompt_cards_raw = waiting_for.get('cards', []) if isinstance(waiting_for, dict) else []
        prompt_cards = [card for card in prompt_cards_raw if isinstance(card, dict)]

        # Candidate cards are the immediately actionable cards (buy/play prompt),
        # falling back to hand cards when no card prompt is active.
        if waiting_type in ['card', 'projectCard', 'selectProjectCardToPlay'] and prompt_cards:
            candidate_cards = prompt_cards
        else:
            candidate_cards = hand_cards

        player = player_state.get('thisPlayer', {}) or {}
        mc = float(player.get('megaCredits', 0) or 0)
        steel = float(player.get('steel', 0) or 0)
        titanium = float(player.get('titanium', 0) or 0)
        steel_prod = float(player.get('steelProduction', 0) or 0)
        titanium_prod = float(player.get('titaniumProduction', 0) or 0)
        steel_value = float(player.get('steelValue', 2) or 2)
        titanium_value = float(player.get('titaniumValue', 3) or 3)

        tag_order = [
            'Building', 'Space', 'Science', 'Power', 'Earth', 'Jovian',
            'Venus', 'Plant', 'Microbe', 'Animal', 'City', 'Moon',
            'Mars', 'Crime', 'Wild', 'Event', 'Clone',
        ]

        def _affordability(cost: float, tags: Dict[str, int]) -> float:
            purchasing_power = mc
            if tags.get('Building', 0) > 0:
                purchasing_power += steel * steel_value
            if tags.get('Space', 0) > 0:
                purchasing_power += titanium * titanium_value
            if cost <= 0:
                return 1.0
            if purchasing_power >= cost:
                return 1.0
            return max(0.0, 1.0 - ((cost - purchasing_power) / 20.0))

        logger.debug(
            "Hand cards=%s candidate_cards=%s waiting_type=%s for player %s",
            len(hand_cards),
            len(candidate_cards),
            waiting_type,
            player_state.get('id'),
        )

        # Aggregate hand features
        encoding[0] = min(len(hand_cards) / 10.0, 1.0)
        total_hand_cost = sum(self._get_card_cost(card) for card in hand_cards)
        encoding[1] = min(total_hand_cost / 100.0, 1.0)

        affordable_count = 0
        for card in hand_cards:
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost = self._get_card_cost(card)
            if _affordability(cost, tags) >= 0.99:
                affordable_count += 1
            for tag_idx, tag_name in enumerate(tag_order):
                if tags.get(tag_name, 0) > 0:
                    encoding[3 + tag_idx] += 0.1

        encoding[2] = min(affordable_count / 10.0, 1.0)
        for i in range(3, 20):
            encoding[i] = min(encoding[i], 1.0)

        tableau = player.get('tableau', []) or []
        tableau_tag_profile: Dict[str, int] = {}
        for played in tableau:
            if not isinstance(played, dict):
                continue
            played_name = str(played.get('name', '') or '')
            played_tags = self._get_card_tags(played_name, fallback=played.get('tags', {}))
            for tag_name, present in played_tags.items():
                if present:
                    tableau_tag_profile[tag_name] = int(tableau_tag_profile.get(tag_name, 0)) + 1
        max_tag_count = max(tableau_tag_profile.values(), default=1)

        card_eval: List[Dict[str, Any]] = []
        type_counts = {'blue': 0, 'green': 0, 'red': 0}
        for card in candidate_cards:
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost = self._get_card_cost(card)
            cost_norm = min(max(cost, 0.0) / 40.0, 1.0)
            affordable = _affordability(cost, tags)
            vp_proxy = self._get_card_vp_proxy(card, tags)

            key_tags = ['Science', 'Building', 'Space', 'Earth', 'Jovian']
            key_tag_density = sum(1 for tag in key_tags if tags.get(tag, 0) > 0) / float(len(key_tags))

            tableau_synergy_raw = 0.0
            for tag_name, present in tags.items():
                if present:
                    tableau_synergy_raw += float(tableau_tag_profile.get(tag_name, 0)) / float(max_tag_count)
            tableau_synergy = min(tableau_synergy_raw / 4.0, 1.0)

            denom_cost = max(cost, 1.0)
            steel_cover = min((steel * steel_value) / denom_cost, 1.0) if tags.get('Building', 0) > 0 else 0.0
            titanium_cover = min((titanium * titanium_value) / denom_cost, 1.0) if tags.get('Space', 0) > 0 else 0.0
            steel_prod_pull = min(max(steel_prod, 0.0) / 8.0, 1.0) if tags.get('Building', 0) > 0 else 0.0
            titanium_prod_pull = min(max(titanium_prod, 0.0) / 8.0, 1.0) if tags.get('Space', 0) > 0 else 0.0
            resource_synergy = min(
                (0.30 * tableau_synergy)
                + (0.30 * steel_cover)
                + (0.30 * titanium_cover)
                + (0.05 * steel_prod_pull)
                + (0.05 * titanium_prod_pull),
                1.0,
            )

            utility = min(
                (
                    (1.20 * affordable)
                    + (0.95 * vp_proxy)
                    + (0.55 * key_tag_density)
                    + (0.85 * resource_synergy)
                    + (0.35 * (1.0 - cost_norm))
                ) / 3.0,
                1.0,
            )

            card_type = self._get_card_type(name, fallback=card.get('type'))
            if card_type == 'active':
                type_counts['blue'] += 1
            elif card_type == 'automated':
                type_counts['green'] += 1
            elif card_type == 'event':
                type_counts['red'] += 1

            card_eval.append({
                'cost_norm': cost_norm,
                'affordable': affordable,
                'vp_proxy': vp_proxy,
                'key_tag_density': key_tag_density,
                'resource_synergy': resource_synergy,
                'utility': utility,
                'tags': tags,
            })

        if card_eval:
            ranked = sorted(
                card_eval,
                key=lambda item: (
                    item['utility'],
                    item['affordable'],
                    item['vp_proxy'],
                    -item['cost_norm'],
                ),
                reverse=True,
            )
            utilities = [float(item['utility']) for item in ranked]
            encoding[20] = min(len(ranked) / 10.0, 1.0)
            encoding[21] = min(sum(utilities) / float(len(utilities)), 1.0)
            encoding[22] = utilities[0]

            encoding[23] = min(float(type_counts['blue']) / float(len(ranked)), 1.0)
            encoding[24] = min(float(type_counts['green']) / float(len(ranked)), 1.0)
            encoding[25] = min(float(type_counts['red']) / float(len(ranked)), 1.0)

            top_k = 3
            slot_width = 5
            for slot in range(top_k):
                base = 26 + (slot * slot_width)
                if slot >= len(ranked):
                    continue
                item = ranked[slot]
                encoding[base] = float(item['cost_norm'])
                encoding[base + 1] = float(item['affordable'])
                encoding[base + 2] = float(item['vp_proxy'])
                encoding[base + 3] = float(item['key_tag_density'])
                encoding[base + 4] = float(item['resource_synergy'])
        else:
            ranked = []

        # Spendability pressure for steel/titanium using stock + production +
        # currently playable tag opportunities.
        spend_pool = hand_cards if hand_cards else candidate_cards
        pool_count = max(1.0, float(len(spend_pool)))

        building_opp = 0
        building_playable = 0
        space_opp = 0
        space_playable = 0
        for card in spend_pool:
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost = self._get_card_cost(card)
            affordability = _affordability(cost, tags)
            if tags.get('Building', 0) > 0:
                building_opp += 1
                if affordability >= 0.99:
                    building_playable += 1
            if tags.get('Space', 0) > 0:
                space_opp += 1
                if affordability >= 0.99:
                    space_playable += 1

        steel_stock_norm = min(max(steel, 0.0) / 25.0, 1.0)
        steel_prod_norm = min(max(steel_prod, 0.0) / 12.0, 1.0)
        titanium_stock_norm = min(max(titanium, 0.0) / 20.0, 1.0)
        titanium_prod_norm = min(max(titanium_prod, 0.0) / 10.0, 1.0)

        building_opportunity_ratio = min(float(building_opp) / pool_count, 1.0)
        building_playable_ratio = float(building_playable) / float(max(1, building_opp))
        building_playable_now_ratio = min(float(building_playable) / pool_count, 1.0)

        space_opportunity_ratio = min(float(space_opp) / pool_count, 1.0)
        space_playable_ratio = float(space_playable) / float(max(1, space_opp))

        steel_liquidity = min(((steel * steel_value) + (max(steel_prod, 0.0) * steel_value * 2.0)) / 80.0, 1.0)
        steel_pressure = min(
            steel_liquidity
            * (0.20 + (0.80 * building_opportunity_ratio))
            * (1.0 - building_playable_ratio),
            1.0,
        )

        titanium_liquidity = min(
            ((titanium * titanium_value) + (max(titanium_prod, 0.0) * titanium_value * 2.0)) / 90.0,
            1.0,
        )
        titanium_pressure = min(
            titanium_liquidity
            * (0.20 + (0.80 * space_opportunity_ratio))
            * (1.0 - space_playable_ratio),
            1.0,
        )

        encoding[41] = steel_stock_norm
        encoding[42] = steel_prod_norm
        encoding[43] = building_opportunity_ratio
        encoding[44] = building_playable_now_ratio
        encoding[45] = steel_pressure
        encoding[46] = titanium_stock_norm
        encoding[47] = titanium_prod_norm
        encoding[48] = space_opportunity_ratio
        encoding[49] = titanium_pressure

        logger.debug("Hand encoding: %s for player %s", encoding, player_state.get('id'))
        return encoding
    
    def _encode_board_state(
        self,
        game_state: Dict[str, Any],
        current_player: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Encode board state with robust tile parsing and city/greenery combo signals."""
        encoding = [0.0] * 100

        ocean_count = game_state.get('oceans', 0)
        encoding[0] = min(float(ocean_count) / 9.0, 1.0)  # Max 9 oceans

        spaces_raw = game_state.get('spaces', [])
        spaces = [space for space in spaces_raw if isinstance(space, dict)]

        city_count = 0
        greenery_count = 0
        special_tile_count = 0
        available_land = 0
        available_ocean = 0

        own_city_tiles = 0
        own_greenery_tiles = 0
        own_special_tiles = 0
        own_color = str((current_player or {}).get('color', '') or '').strip().lower()
        vpb = (current_player or {}).get('victoryPointsBreakdown', {}) or {}

        city_positions: List[Tuple[int, int]] = []
        greenery_positions: List[Tuple[int, int]] = []
        ocean_positions: List[Tuple[int, int]] = []

        for space in spaces:
            space_type = self._space_type_lower(space.get('spaceType'))
            if space_type == 'colony':
                # Mars-board interaction features should ignore colony spaces.
                continue

            tile_raw = space.get('tileType')
            has_tile = tile_raw is not None
            if not has_tile:
                if space_type in ('land', 'restricted'):
                    available_land += 1
                elif space_type == 'ocean':
                    available_ocean += 1
                continue

            is_city, is_greenery, is_ocean = self._tile_flags(space)
            if is_city:
                city_count += 1
                city_positions.append(self._get_space_coordinates(space))
            if is_greenery:
                greenery_count += 1
                greenery_positions.append(self._get_space_coordinates(space))
            if is_ocean:
                ocean_positions.append(self._get_space_coordinates(space))
            if not (is_city or is_greenery or is_ocean):
                special_tile_count += 1

            if own_color and str(space.get('color', '') or '').strip().lower() == own_color:
                if is_city:
                    own_city_tiles += 1
                if is_greenery:
                    own_greenery_tiles += 1
                if not (is_city or is_greenery or is_ocean):
                    own_special_tiles += 1

        encoding[1] = min(city_count / 20.0, 1.0)
        encoding[2] = min(greenery_count / 30.0, 1.0)
        encoding[3] = min(special_tile_count / 20.0, 1.0)
        encoding[4] = min(available_land / 50.0, 1.0)
        encoding[5] = min(available_ocean / 9.0, 1.0)

        # Map-level ownership footprint.
        encoding[6] = min(own_city_tiles / 8.0, 1.0)
        encoding[7] = min(own_greenery_tiles / 20.0, 1.0)
        encoding[8] = min(own_special_tiles / 12.0, 1.0)
        encoding[9] = min((own_city_tiles + own_greenery_tiles) / 24.0, 1.0)

        # Encode spatial patterns
        encoding[50] = self._calculate_city_clustering(city_positions)
        encoding[51] = self._calculate_greenery_spread(greenery_positions)
        encoding[52] = self._calculate_ocean_coverage(ocean_positions)

        # Direct city-greenery scoring signal (combo utilization).
        own_city_vp = float(vpb.get('city', 0) or 0)
        own_greenery_vp = float(vpb.get('greenery', 0) or 0)
        own_city_count_stat = float((current_player or {}).get('citiesCount', 0) or 0)
        own_greenery_count_stat = float(max(own_greenery_tiles, own_greenery_vp))
        max_city_combo = max(1.0, own_city_count_stat * 6.0)

        encoding[53] = min(own_city_vp / 20.0, 1.0)
        encoding[54] = min(own_greenery_vp / 20.0, 1.0)
        encoding[55] = min(own_city_count_stat / 8.0, 1.0)
        encoding[56] = min(own_greenery_count_stat / 20.0, 1.0)
        encoding[57] = min(own_city_vp / max_city_combo, 1.0)
        encoding[58] = min(own_city_vp / max(1.0, own_greenery_count_stat), 1.0)
        encoding[59] = min(
            ((0.70 * own_city_vp) + (0.30 * own_greenery_vp)) / max(1.0, (5.0 * own_city_count_stat)),
            1.0,
        )
        encoding[60] = min((own_city_vp + own_greenery_vp) / 30.0, 1.0)

        return encoding

    def _get_space_coordinates(self, space: Dict[str, Any]) -> Tuple[int, int]:
        """Extract coordinates from explicit x/y fields, fallback to space ID parsing."""
        x_raw = self._safe_int(space.get('x'))
        y_raw = self._safe_int(space.get('y'))
        if x_raw is not None and y_raw is not None:
            return (x_raw, y_raw)

        space_id = str(space.get('id', '') or '')
        try:
            x = int(space_id[0]) if len(space_id) > 0 else 0
            y = int(space_id[1]) if len(space_id) > 1 else 0
            return (x, y)
        except Exception:
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
                'selectOption': 4, 'selectAmount': 5, 'selectPlayer': 6,
                'selectCard': 7, 'projectCard': 8, 'selectProjectCardToPlay': 9,
                'initialCards': 10, 'payment': 11, 'space': 12
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
                    if 'play project card' in title:
                        encoding[28] = 1.0
                        encoding[29] = (i + 1) / 10.0
                    if 'convert plants' in title:
                        encoding[30] = 1.0
                        encoding[31] = (i + 1) / 10.0
                    if 'convert heat' in title:
                        encoding[32] = 1.0
                        encoding[33] = (i + 1) / 10.0
                    if 'perform an action' in title or 'take action' in title:
                        encoding[34] = 1.0
                        encoding[35] = (i + 1) / 10.0
            elif input_type == 'card':
                cards = waiting_for.get('cards', [])
                encoding[11] = min(len(cards) / 20.0, 1.0)
            elif input_type in ['selectCard', 'projectCard', 'selectProjectCardToPlay']:
                cards = waiting_for.get('cards', [])
                encoding[36] = min(len(cards) / 20.0, 1.0)
            elif input_type in ['selectSpace', 'space']:
                spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
                encoding[37] = min(len(spaces) / 40.0, 1.0)
            elif input_type in ['selectAmount', 'amount']:
                min_amount = float(waiting_for.get('min', 0) or 0)
                max_amount = float(waiting_for.get('max', 0) or 0)
                encoding[38] = min(min_amount / 20.0, 1.0)
                encoding[39] = min(max_amount / 20.0, 1.0)
        logger.debug("Action context: %s for player %s", encoding, player_state.get('id'))
        return encoding
    
    def _project_award_points_for_color(
        self,
        scores: List[Dict[str, Any]],
        own_color: str,
    ) -> float:
        if not scores or not own_color:
            return 0.0

        own_color_l = own_color.strip().lower()
        normalized: List[Tuple[str, float]] = []
        for row in scores:
            if not isinstance(row, dict):
                continue
            color = str(row.get('playerColor', '') or '').strip().lower()
            if not color:
                continue
            try:
                score = float(row.get('playerScore', 0) or 0)
            except Exception:
                score = 0.0
            normalized.append((color, score))

        if not normalized:
            return 0.0

        # Replicate game logic: 1st = 5 VP, 2nd = 2 VP, ties receive full VP for that rank.
        normalized.sort(key=lambda pair: pair[1], reverse=True)
        top_score = normalized[0][1]
        top_colors = [color for color, score in normalized if score == top_score]
        if own_color_l in top_colors:
            return 5.0
        if len(top_colors) > 1:
            return 0.0

        remaining = normalized[len(top_colors):]
        if not remaining:
            return 0.0

        second_score = remaining[0][1]
        second_colors = [color for color, score in remaining if score == second_score]
        if own_color_l in second_colors:
            return 2.0
        return 0.0

    def _encode_awards_milestones(
        self,
        game_state: Dict[str, Any],
        current_player: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Encode milestones/awards with per-milestone/per-award encoding.
        
        For each milestone/award in the fixed list, encodes 4 values:
        1. exists_in_game (1.0 / 0.0) - whether this milestone/award exists in the current game
        2. I_have_it (1.0 / 0.0) - whether the current player owns/claimed it
        3. opponent_has_it (1.0 / 0.0) - whether any opponent owns/claimed it
        4. my_progress_normalized (0.0 - 1.0+) - normalized progress toward claiming/winning it
        
        Progress is calculated from scores array:
        - For milestones: my_score / max(threshold, best_score_among_all_players)
        - For awards: my_score / max(threshold, best_score_among_all_players)
        
        This ensures consistent indexing for PPO auxiliary heads (e.g., aux_pred_milestone_claimability),
        where Gardener will always be at the same index regardless of which map is being played.
        The progress value helps the model understand how close it is to claiming milestones/awards.
        """
        milestones_raw = [m for m in (game_state.get('milestones', []) or []) if isinstance(m, dict)]
        awards_raw = [a for a in (game_state.get('awards', []) or []) if isinstance(a, dict)]

        own_name = str((current_player or {}).get('name', '') or '').strip()
        own_color = str((current_player or {}).get('color', '') or '').strip().lower()
        
        # Build lookup maps for quick access
        milestone_by_name: Dict[str, Dict[str, Any]] = {}
        for m in milestones_raw:
            name = str(m.get('name', '') or '').strip()
            if name:
                milestone_by_name[name] = m
        
        award_by_name: Dict[str, Dict[str, Any]] = {}
        for a in awards_raw:
            name = str(a.get('name', '') or '').strip()
            if name:
                award_by_name[name] = a
        
        # Calculate encoding size: (milestones * 4) + (awards * 4)
        num_milestones = len(self._ALL_MILESTONES)
        num_awards = len(self._ALL_AWARDS)
        encoding_size = (num_milestones * 4) + (num_awards * 4)
        encoding = [0.0] * encoding_size
        
        # Encode milestones: for each milestone in fixed order, encode 4 values
        for idx, milestone_name in enumerate(self._ALL_MILESTONES):
            base_idx = idx * 4
            
            # Check if milestone exists in this game
            milestone = milestone_by_name.get(milestone_name)
            if milestone:
                encoding[base_idx] = 1.0  # exists_in_game
                
                # Check ownership
                owner_name = str(milestone.get('playerName', '') or '').strip()
                owner_color = str(milestone.get('playerColor', '') or '').strip().lower()
                
                if owner_name or owner_color:
                    # Someone owns it
                    is_mine = False
                    is_opponent = False
                    
                    if own_name and owner_name == own_name:
                        is_mine = True
                    elif own_color and owner_color == own_color:
                        is_mine = True
                    else:
                        is_opponent = True
                    
                    encoding[base_idx + 1] = 1.0 if is_mine else 0.0  # I_have_it
                    encoding[base_idx + 2] = 1.0 if is_opponent else 0.0  # opponent_has_it
                    # If claimed, progress is 1.0 (achieved) or 0.0 (opponent has it)
                    encoding[base_idx + 3] = 1.0 if is_mine else 0.0  # my_progress_normalized
                else:
                    # Unclaimed - calculate progress from scores
                    encoding[base_idx + 1] = 0.0  # I_have_it
                    encoding[base_idx + 2] = 0.0  # opponent_has_it
                    
                    # Extract progress from scores array
                    scores = [row for row in (milestone.get('scores', []) or []) if isinstance(row, dict)]
                    my_score = 0.0
                    max_score = 0.0
                    
                    for row in scores:
                        try:
                            score = float(row.get('playerScore', 0) or 0)
                        except Exception:
                            score = 0.0
                        max_score = max(max_score, score)
                        
                        row_color = str(row.get('playerColor', '') or '').strip().lower()
                        if own_color and row_color == own_color:
                            my_score = score
                    
                    # Normalize progress: my_score / max(threshold, max_score)
                    # Threshold prevents division by zero and represents typical milestone requirement
                    # Most milestones require 3-8 of something, so threshold of 3 is reasonable
                    threshold = 3.0
                    denominator = max(threshold, max_score, 1.0)
                    progress_normalized = min(my_score / denominator, 2.0)  # Cap at 2.0 (200% of threshold)
                    encoding[base_idx + 3] = float(progress_normalized)  # my_progress_normalized
            else:
                # Milestone doesn't exist in this game
                encoding[base_idx] = 0.0  # exists_in_game
                encoding[base_idx + 1] = 0.0  # I_have_it
                encoding[base_idx + 2] = 0.0  # opponent_has_it
                encoding[base_idx + 3] = 0.0  # my_progress_normalized
        
        # Encode awards: for each award in fixed order, encode 4 values
        awards_base_idx = num_milestones * 4
        for idx, award_name in enumerate(self._ALL_AWARDS):
            base_idx = awards_base_idx + (idx * 4)
            
            # Check if award exists in this game
            award = award_by_name.get(award_name)
            if award:
                encoding[base_idx] = 1.0  # exists_in_game
                
                # Check funding (awards are "funded" not "claimed", but same concept)
                funder_name = str(award.get('playerName', '') or '').strip()
                funder_color = str(award.get('playerColor', '') or '').strip().lower()
                
                if funder_name or funder_color:
                    # Someone funded it
                    is_mine = False
                    is_opponent = False
                    
                    if own_name and funder_name == own_name:
                        is_mine = True
                    elif own_color and funder_color == own_color:
                        is_mine = True
                    else:
                        is_opponent = True
                    
                    encoding[base_idx + 1] = 1.0 if is_mine else 0.0  # I_have_it (funded)
                    encoding[base_idx + 2] = 1.0 if is_opponent else 0.0  # opponent_has_it (funded)
                    
                    # For funded awards, calculate progress toward winning (1st/2nd place)
                    scores = [row for row in (award.get('scores', []) or []) if isinstance(row, dict)]
                    my_score = 0.0
                    max_score = 0.0
                    
                    for row in scores:
                        try:
                            score = float(row.get('playerScore', 0) or 0)
                        except Exception:
                            score = 0.0
                        max_score = max(max_score, score)
                        
                        row_color = str(row.get('playerColor', '') or '').strip().lower()
                        if own_color and row_color == own_color:
                            my_score = score
                    
                    # Normalize progress: my_score / max(threshold, max_score)
                    # Awards typically have scores in range 1-20+, threshold of 5 is reasonable
                    threshold = 5.0
                    denominator = max(threshold, max_score, 1.0)
                    progress_normalized = min(my_score / denominator, 2.0)  # Cap at 2.0 (200% of threshold)
                    encoding[base_idx + 3] = float(progress_normalized)  # my_progress_normalized
                else:
                    # Unfunded - calculate progress from scores if available
                    encoding[base_idx + 1] = 0.0  # I_have_it
                    encoding[base_idx + 2] = 0.0  # opponent_has_it
                    
                    # Extract progress from scores array
                    scores = [row for row in (award.get('scores', []) or []) if isinstance(row, dict)]
                    my_score = 0.0
                    max_score = 0.0
                    
                    for row in scores:
                        try:
                            score = float(row.get('playerScore', 0) or 0)
                        except Exception:
                            score = 0.0
                        max_score = max(max_score, score)
                        
                        row_color = str(row.get('playerColor', '') or '').strip().lower()
                        if own_color and row_color == own_color:
                            my_score = score
                    
                    # Normalize progress
                    threshold = 5.0
                    denominator = max(threshold, max_score, 1.0)
                    progress_normalized = min(my_score / denominator, 2.0)
                    encoding[base_idx + 3] = float(progress_normalized)  # my_progress_normalized
            else:
                # Award doesn't exist in this game
                encoding[base_idx] = 0.0  # exists_in_game
                encoding[base_idx + 1] = 0.0  # I_have_it
                encoding[base_idx + 2] = 0.0  # opponent_has_it
                encoding[base_idx + 3] = 0.0  # my_progress_normalized
        
        logger.debug(
            "Awards/milestones encoding: %d features (%d milestones * 4 + %d awards * 4) for game/player %s",
            encoding_size, num_milestones, num_awards, game_state.get('id')
        )
        return encoding

    def _normalize_tag_name(self, value: Any) -> str:
        raw = str(value or '').strip()
        if not raw:
            return ''
        if raw[0].islower():
            return raw.capitalize()
        return raw

    def _normalize_card_type(self, value: Any) -> str:
        raw = str(value or '').strip().lower()
        if not raw:
            return ''
        mapping = {
            'blue': 'active',
            'green': 'automated',
            'red': 'event',
        }
        return mapping.get(raw, raw)

    def _get_card_cost(self, card: Dict[str, Any]) -> float:
        name = str(card.get('name', '') or '')
        meta = self.card_metadata_by_name.get(name, {}) if name else {}
        raw = card.get('calculatedCost', card.get('cost', meta.get('cost', 0)))
        try:
            return max(float(raw or 0), 0.0)
        except Exception:
            return 0.0

    def _get_card_type(self, card_name: str, fallback: Any = None) -> str:
        fallback_type = self._normalize_card_type(fallback)
        if fallback_type:
            return fallback_type
        if card_name and self.card_metadata_by_name:
            meta = self.card_metadata_by_name.get(card_name, {})
            meta_type = self._normalize_card_type(meta.get('type', ''))
            if meta_type:
                return meta_type
        return ''

    def _get_card_vp_proxy(self, card: Dict[str, Any], tags: Optional[Dict[str, int]] = None) -> float:
        name = str(card.get('name', '') or '')
        meta = self.card_metadata_by_name.get(name, {}) if name else {}
        raw_vp = card.get('victoryPoints', meta.get('victoryPoints', 0))
        try:
            numeric_vp = float(raw_vp or 0)
            if numeric_vp > 0:
                return min(numeric_vp / 5.0, 1.0)
        except Exception:
            pass

        resolved_tags = tags or self._get_card_tags(name, fallback=card.get('tags', {}))
        proxy = 0.0
        if resolved_tags.get('Jovian', 0) > 0:
            proxy += 0.20
        if resolved_tags.get('Science', 0) > 0:
            proxy += 0.12
        if resolved_tags.get('City', 0) > 0:
            proxy += 0.12
        if resolved_tags.get('Animal', 0) > 0:
            proxy += 0.10
        if resolved_tags.get('Microbe', 0) > 0:
            proxy += 0.08
        if resolved_tags.get('Earth', 0) > 0:
            proxy += 0.06
        if resolved_tags.get('Plant', 0) > 0:
            proxy += 0.05
        card_type = self._get_card_type(name, fallback=card.get('type', ''))
        if card_type == 'active':
            proxy += 0.05
        elif card_type == 'event':
            proxy += 0.03
        proxy += 0.10 * min(self._get_card_cost(card) / 40.0, 1.0)
        return min(max(proxy, 0.0), 1.0)

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
        module_dir = os.path.abspath(os.path.dirname(__file__))
        one_up = os.path.abspath(os.path.join(module_dir, '..'))
        two_up = os.path.abspath(os.path.join(module_dir, '..', '..'))

        candidate_paths: List[str] = []
        env_path = os.getenv('TM_CARD_METADATA_PATH')
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.extend([
            os.path.join(one_up, 'card_metadata.json'),
            os.path.join(two_up, 'card_metadata.json'),
            os.path.join(two_up, 'terraforming-mars', 'card_metadata.json'),
            os.path.join(one_up, 'terraforming-mars', 'card_metadata.json'),
        ])

        seen: set = set()
        for candidate in candidate_paths:
            if not candidate:
                continue
            path = os.path.abspath(candidate)
            if path in seen:
                continue
            seen.add(path)

            cached = StateEncoder._CARD_METADATA_CACHE.get(path)
            if cached is not None:
                if cached:
                    return cached
                continue
            if not os.path.exists(path):
                continue
            if os.path.getsize(path) <= 0:
                StateEncoder._CARD_METADATA_CACHE[path] = {}
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    StateEncoder._CARD_METADATA_CACHE[path] = {}
                    continue
                normalized: Dict[str, Dict[str, Any]] = {}
                for name, meta in (data or {}).items():
                    meta_copy = dict(meta or {})
                    tags = meta_copy.get('tags') or []
                    norm_tags: List[str] = []
                    for tag in tags:
                        normalized_tag = self._normalize_tag_name(tag)
                        if normalized_tag:
                            norm_tags.append(normalized_tag)
                    meta_copy['tags'] = norm_tags
                    meta_copy['type'] = self._normalize_card_type(meta_copy.get('type', ''))
                    normalized[str(name)] = meta_copy
                logger.info(f"Loaded card metadata for {len(normalized)} cards from {path}")
                StateEncoder._CARD_METADATA_CACHE[path] = normalized
                return normalized
            except Exception as e:
                logger.warning(f"Failed to load card metadata from {path}: {e}")
                StateEncoder._CARD_METADATA_CACHE[path] = {}

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
                    normalized_tag = self._normalize_tag_name(tag)
                    if normalized_tag:
                        tag_map[normalized_tag] = 1
                return tag_map
        # Fallback to provided structured tags if any
        if isinstance(fallback, dict) and fallback:
            normalized: Dict[str, int] = {}
            for key, value in fallback.items():
                if not value:
                    continue
                normalized_key = self._normalize_tag_name(key)
                if normalized_key:
                    normalized[normalized_key] = 1
            if normalized:
                return normalized
        if isinstance(fallback, list) and fallback:
            normalized = {}
            for tag in fallback:
                normalized_tag = self._normalize_tag_name(tag)
                if normalized_tag:
                    normalized[normalized_tag] = 1
            if normalized:
                return normalized
        # Final fallback to heuristics
        return self._infer_tags_from_name(card_name)
    

    

