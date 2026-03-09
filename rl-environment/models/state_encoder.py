"""
State Encoder - Converts game state to neural network input
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import logging

from .rust_backend import get_rust_module
from .requirement_planning import RequirementPlanner

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

    # Transformer card-token layout encoded inside the tail of the fixed-size state vector.
    _DEFAULT_CARD_TOKEN_DIM = 8
    _DEFAULT_TABLEAU_TOKEN_COUNT = 8
    _DEFAULT_HAND_TOKEN_COUNT = 64
    _DEFAULT_OPPONENT_TOKEN_COUNT = 6
    _SPACE_FEATURE_BASE = 64
    _SPACE_CANDIDATE_SLOT_COUNT = 80
    _SPACE_FEATURE_PLANES = 4
    _SPACE_SUMMARY_FEATURE_COUNT = 16

    def __init__(
        self,
        state_size: int = 1024,
        card_token_dim: int = _DEFAULT_CARD_TOKEN_DIM,
        tableau_token_count: int = _DEFAULT_TABLEAU_TOKEN_COUNT,
        hand_token_count: int = _DEFAULT_HAND_TOKEN_COUNT,
        opponent_token_count: int = _DEFAULT_OPPONENT_TOKEN_COUNT,
    ):
        self.state_size = state_size
        self.card_token_dim = max(1, int(card_token_dim))
        self.tableau_token_count = max(0, int(tableau_token_count))
        self.hand_token_count = max(0, int(hand_token_count))
        self.opponent_token_count = max(0, int(opponent_token_count))
        self.card_token_count = self.tableau_token_count + self.hand_token_count + self.opponent_token_count
        self.card_token_vector_size = self.card_token_count * self.card_token_dim
        if self.card_token_vector_size > int(self.state_size):
            raise ValueError(
                f"Card token vector size {self.card_token_vector_size} exceeds state_size {self.state_size}"
            )
        
        # Load authoritative card metadata first; derive common_cards from it when available
        self.card_metadata_by_name: Dict[str, Dict[str, Any]] = self._load_card_metadata()
        self.requirement_planner = RequirementPlanner(self.card_metadata_by_name)
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

    def _space_id(self, space: Dict[str, Any]) -> str:
        return str(space.get('id', space.get('spaceId', '')) or '').strip()

    def _space_owner_color(self, space: Dict[str, Any]) -> str:
        return str(
            space.get('color', space.get('playerColor', space.get('owner', ''))) or ''
        ).strip().lower()

    def _infer_space_prompt_intent(self, waiting_for: Dict[str, Any]) -> str:
        title_raw = waiting_for.get('title', '')
        if isinstance(title_raw, dict):
            title_raw = title_raw.get('message', '')
        title = str(title_raw or '').lower()
        button = str(waiting_for.get('buttonLabel', '') or '').lower()
        context = f"{title} {button}".strip()

        if 'greenery' in context:
            return 'greenery'
        if 'ocean' in context or 'aquifer' in context:
            return 'ocean'
        if 'for city tile' in context or 'place a city tile' in context or 'place city tile' in context:
            return 'city'
        if 'select space for city tile' in context:
            return 'city'
        if 'special tile' in context or 'hazard tile' in context:
            return 'special'
        if ' for ' in context and ' tile' in context:
            return 'special'
        if 'adjacent to a city tile' in context or 'adjacent to city tile' in context:
            return 'special'
        return 'unknown'

    def _all_board_spaces(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        spaces: List[Dict[str, Any]] = []
        for space in (game_state.get('spaces', []) or []):
            if isinstance(space, dict):
                spaces.append(space)
        moon = game_state.get('moon', {}) or {}
        for space in (moon.get('spaces', []) or []):
            if isinstance(space, dict):
                spaces.append(space)
        return spaces

    def _neighbor_coordinates(self, coords: Tuple[int, int], middle_row: float) -> List[Tuple[int, int]]:
        x, y = coords
        left_space = (x - 1, y)
        right_space = (x + 1, y)
        top_left_space = [x, y - 1]
        top_right_space = [x, y - 1]
        bottom_left_space = [x, y + 1]
        bottom_right_space = [x, y + 1]

        if y < middle_row:
            bottom_left_space[0] -= 1
            top_right_space[0] += 1
        elif y == middle_row:
            bottom_right_space[0] += 1
            top_right_space[0] += 1
        else:
            bottom_right_space[0] += 1
            top_left_space[0] -= 1

        return [
            tuple(top_left_space),
            tuple(top_right_space),
            right_space,
            tuple(bottom_right_space),
            tuple(bottom_left_space),
            left_space,
        ]

    def _neighbor_ids_from_coordinates(
        self,
        coords: Tuple[int, int],
        coord_to_id: Dict[Tuple[int, int], str],
        middle_row: float,
        self_id: str = '',
    ) -> List[str]:
        neighbors: List[str] = []
        for neighbor_coords in self._neighbor_coordinates(coords, middle_row):
            neighbor_id = coord_to_id.get(neighbor_coords)
            if neighbor_id and neighbor_id != self_id:
                neighbors.append(neighbor_id)
        return neighbors

    def _build_board_adjacency_index(self, spaces: List[Dict[str, Any]]) -> Tuple[Dict[str, set], Dict[Tuple[int, int], str], float]:
        coord_to_id: Dict[Tuple[int, int], str] = {}
        parsed_spaces: List[Tuple[str, int, int]] = []
        max_y = 0

        for space in spaces:
            if not isinstance(space, dict):
                continue
            sid = self._space_id(space)
            if not sid:
                continue
            if self._space_type_lower(space.get('spaceType')) == 'colony':
                continue
            x = self._safe_int(space.get('x'))
            y = self._safe_int(space.get('y'))
            if x is None or y is None or x < 0 or y < 0:
                continue
            coord_to_id[(x, y)] = sid
            parsed_spaces.append((sid, x, y))
            if y > max_y:
                max_y = y

        if not parsed_spaces:
            return {}, coord_to_id, 0.0

        middle_row = max_y / 2.0
        adjacency: Dict[str, set] = {}
        for sid, x, y in parsed_spaces:
            adjacency[sid] = set(self._neighbor_ids_from_coordinates((x, y), coord_to_id, middle_row, sid))
        return adjacency, coord_to_id, middle_row

    def _estimate_space_bonus_value(self, space: Dict[str, Any]) -> float:
        total = 0.0
        for item in (space.get('bonus', []) or []):
            label = ''
            if isinstance(item, dict):
                label = str(item.get('type', item.get('name', item.get('bonus', item.get('resource', '')))) or '')
            else:
                label = str(item or '')
            token = label.strip().lower().replace('_', ' ')
            amount = 1.0
            for part in token.split():
                try:
                    amount = float(part)
                    break
                except Exception:
                    continue
            if 'card' in token:
                total += 1.8 * amount
            elif 'titanium' in token:
                total += 1.6 * amount
            elif 'steel' in token:
                total += 1.3 * amount
            elif 'plant' in token:
                total += 1.1 * amount
            elif 'heat' in token or 'energy' in token:
                total += 0.8 * amount
            elif 'megacredit' in token or token == 'mc':
                total += 0.6 * amount
            else:
                total += 0.9 * amount
        return min(total, 6.0)

    def _space_feature_layout(self) -> Tuple[int, int, int, int]:
        start = max(0, min(int(self._SPACE_FEATURE_BASE), int(self.state_size)))
        prefix_limit = int(self.state_size)
        card_start, _ = self._card_token_segment_bounds()
        if card_start > 0:
            prefix_limit = min(prefix_limit, int(card_start))
        if prefix_limit <= start:
            return start, 0, start, start

        usable = max(0, prefix_limit - start)
        if usable < self._SPACE_SUMMARY_FEATURE_COUNT:
            return start, 0, start, start

        slot_count = min(
            int(self._SPACE_CANDIDATE_SLOT_COUNT),
            max(0, (usable - int(self._SPACE_SUMMARY_FEATURE_COUNT)) // int(self._SPACE_FEATURE_PLANES)),
        )
        if slot_count <= 0:
            return start, 0, start, start

        summary_start = start + (slot_count * int(self._SPACE_FEATURE_PLANES))
        end = min(prefix_limit, summary_start + int(self._SPACE_SUMMARY_FEATURE_COUNT))
        return start, slot_count, summary_start, end

    def _space_prompt_candidate_scores(
        self,
        candidate: Dict[str, Any],
        board_by_id: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, set],
        coord_to_id: Dict[Tuple[int, int], str],
        middle_row: float,
        own_color: str,
        intent: str,
    ) -> Tuple[float, float, float, float]:
        sid = self._space_id(candidate)
        neighbor_ids = list(adjacency.get(sid, set()))
        if not neighbor_ids:
            neighbor_ids = self._neighbor_ids_from_coordinates(
                self._get_space_coordinates(candidate),
                coord_to_id,
                middle_row,
                sid,
            )

        own_city_adj = 0
        enemy_city_adj = 0
        own_greenery_adj = 0
        enemy_greenery_adj = 0
        ocean_adj = 0
        empty_land_adj = 0

        for neighbor_id in neighbor_ids:
            neighbor = board_by_id.get(neighbor_id)
            if not isinstance(neighbor, dict):
                continue
            is_city, is_greenery, is_ocean = self._tile_flags(neighbor)
            owner = self._space_owner_color(neighbor)
            if is_city:
                if owner and owner == own_color:
                    own_city_adj += 1
                elif owner:
                    enemy_city_adj += 1
            if is_greenery:
                if owner and owner == own_color:
                    own_greenery_adj += 1
                elif owner:
                    enemy_greenery_adj += 1
            if is_ocean:
                ocean_adj += 1
            if neighbor.get('tileType') is None and self._space_type_lower(neighbor.get('spaceType')) in ('land', 'restricted', 'ocean'):
                empty_land_adj += 1

        all_greenery_adj = own_greenery_adj + enemy_greenery_adj
        bonus_norm = min(self._estimate_space_bonus_value(candidate) / 6.0, 1.0)
        own_city_norm = min(own_city_adj / 3.0, 1.0)
        enemy_city_norm = min(enemy_city_adj / 3.0, 1.0)
        own_greenery_norm = min(own_greenery_adj / 4.0, 1.0)
        enemy_greenery_norm = min(enemy_greenery_adj / 4.0, 1.0)
        all_greenery_norm = min(all_greenery_adj / 6.0, 1.0)
        ocean_norm = min(ocean_adj / 3.0, 1.0)
        empty_norm = min(empty_land_adj / 6.0, 1.0)

        if intent == 'city':
            self_score = min(1.0, (0.55 * all_greenery_norm) + (0.25 * empty_norm) + (0.15 * ocean_norm) + (0.20 * bonus_norm))
            deny_score = enemy_city_norm
            risk_score = own_city_norm
            total_score = max(0.0, min(1.0, self_score + (0.45 * deny_score) - (0.25 * risk_score)))
        elif intent == 'greenery':
            self_score = min(1.0, (0.70 * own_city_norm) + (0.20 * ocean_norm) + (0.20 * bonus_norm))
            deny_score = min(1.0, 0.15 * enemy_greenery_norm)
            risk_score = enemy_city_norm
            total_score = max(0.0, min(1.0, self_score + (0.10 * deny_score) - (0.80 * risk_score)))
        elif intent == 'ocean':
            self_score = min(1.0, (0.45 * bonus_norm) + (0.30 * empty_norm) + (0.15 * own_city_norm) + (0.15 * own_greenery_norm))
            deny_score = min(1.0, 0.70 * enemy_city_norm)
            risk_score = min(1.0, 0.50 * own_city_norm)
            total_score = max(0.0, min(1.0, self_score + (0.35 * deny_score) - (0.25 * risk_score)))
        elif intent == 'special':
            self_score = min(1.0, (0.40 * bonus_norm) + (0.25 * own_city_norm) + (0.15 * ocean_norm) + (0.15 * own_greenery_norm))
            deny_score = min(1.0, (0.80 * enemy_city_norm) + (0.20 * enemy_greenery_norm))
            risk_score = min(1.0, (0.70 * own_city_norm) + (0.20 * own_greenery_norm))
            total_score = max(0.0, min(1.0, self_score + (0.55 * deny_score) - (0.35 * risk_score)))
        else:
            self_score = min(1.0, (0.35 * bonus_norm) + (0.25 * own_city_norm) + (0.25 * ocean_norm) + (0.15 * all_greenery_norm))
            deny_score = enemy_city_norm
            risk_score = own_city_norm
            total_score = max(0.0, min(1.0, self_score + (0.35 * deny_score) - (0.35 * risk_score)))

        return total_score, self_score, deny_score, risk_score

    def _inject_space_prompt_features(self, state_vector: np.ndarray, player_state: Dict[str, Any]) -> None:
        if not isinstance(state_vector, np.ndarray) or state_vector.ndim != 1:
            return

        start, slot_count, summary_start, end = self._space_feature_layout()
        if slot_count <= 0 or end <= start:
            return

        state_vector[start:end] = 0.0

        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '').strip().lower()
        if waiting_type not in ('space', 'selectspace'):
            return

        raw_spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', [])) or []
        if not isinstance(raw_spaces, list) or not raw_spaces:
            return

        game_state = player_state.get('game', {}) or {}
        board_spaces = self._all_board_spaces(game_state)
        board_by_id = {
            self._space_id(space): space
            for space in board_spaces
            if isinstance(space, dict) and self._space_id(space)
        }
        adjacency, coord_to_id, middle_row = self._build_board_adjacency_index(board_spaces)
        own_color = str((player_state.get('thisPlayer', {}) or {}).get('color', '') or '').strip().lower()
        intent = self._infer_space_prompt_intent(waiting_for)

        totals: List[float] = []
        self_scores: List[float] = []
        deny_scores: List[float] = []
        risk_scores: List[float] = []

        for raw_space in raw_spaces[:slot_count]:
            if isinstance(raw_space, dict):
                sid = self._space_id(raw_space)
                resolved = dict(board_by_id.get(sid, {}))
                resolved.update(raw_space)
            else:
                sid = str(raw_space or '').strip()
                resolved = dict(board_by_id.get(sid, {}))
                if sid and 'id' not in resolved:
                    resolved['id'] = sid
            total_score, self_score, deny_score, risk_score = self._space_prompt_candidate_scores(
                resolved,
                board_by_id,
                adjacency,
                coord_to_id,
                middle_row,
                own_color,
                intent,
            )
            totals.append(float(total_score))
            self_scores.append(float(self_score))
            deny_scores.append(float(deny_score))
            risk_scores.append(float(risk_score))

        total_plane = start
        self_plane = start + slot_count
        deny_plane = start + (2 * slot_count)
        risk_plane = start + (3 * slot_count)

        for idx, value in enumerate(totals):
            state_vector[total_plane + idx] = value
        for idx, value in enumerate(self_scores):
            state_vector[self_plane + idx] = value
        for idx, value in enumerate(deny_scores):
            state_vector[deny_plane + idx] = value
        for idx, value in enumerate(risk_scores):
            state_vector[risk_plane + idx] = value

        summary = np.zeros((int(self._SPACE_SUMMARY_FEATURE_COUNT),), dtype=np.float32)
        if totals:
            sorted_totals = sorted(totals, reverse=True)
            risk_threshold = 0.35 if intent == 'greenery' else 0.45
            summary[0] = 1.0
            summary[1] = 1.0 if intent == 'city' else 0.0
            summary[2] = 1.0 if intent == 'greenery' else 0.0
            summary[3] = 1.0 if intent == 'ocean' else 0.0
            summary[4] = 1.0 if intent == 'special' else 0.0
            summary[5] = 1.0 if intent == 'unknown' else 0.0
            summary[6] = min(len(totals) / max(1.0, float(slot_count)), 1.0)
            summary[7] = float(max(totals))
            summary[8] = float(sum(totals) / len(totals))
            summary[9] = float(max(self_scores))
            summary[10] = float(max(deny_scores))
            summary[11] = float(max(risk_scores))
            summary[12] = float(np.argmax(np.asarray(totals, dtype=np.float32)) / max(1.0, float(len(totals) - 1)))
            summary[13] = float(sum(1 for value in risk_scores if value >= risk_threshold) / len(risk_scores))
            summary[14] = float(sum(deny_scores) / len(deny_scores))
            summary[15] = (
                float(max(0.0, sorted_totals[0] - sorted_totals[1]))
                if len(sorted_totals) > 1
                else float(sorted_totals[0])
            )

        write_len = min(len(summary), max(0, end - summary_start))
        if write_len > 0:
            state_vector[summary_start:summary_start + write_len] = summary[:write_len]
    
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
    
    def encode(self, player_state: Dict[str, Any], turn_action_count: int = 0) -> np.ndarray:
        """Encode complete game state into feature vector.

        Args:
            player_state: Full player-state dict from the game server.
            turn_action_count: Number of actions already taken this turn by this
                player (0 = first action slot, 1 = second action slot, …).
                Supplied by the agent so the network can plan sequences.
        """
        rust_tfm_rl = get_rust_module(required=True)

        # The JSON payload is passed as a raw string to bypass Python object churn.
        rust_encoded = rust_tfm_rl.encode_state(
            json.dumps(player_state),
            int(turn_action_count),
            int(self.state_size),
        )
        encoded = np.array(rust_encoded, dtype=np.float32)
        if encoded.shape != (int(self.state_size),):
            raise RuntimeError(
                f"Rust encoder returned invalid shape {encoded.shape}, expected ({int(self.state_size)},)"
            )
        self._inject_space_prompt_features(encoded, player_state)
        self._inject_card_token_features(encoded, player_state)
        return encoded

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
        vp_resource_cards = 0.0
        conversion_cards = 0.0
        stealing_cards = 0.0
        adding_cards = 0.0
        conversion_ready = 0.0
        vp_resource_value = 0.0
        for card in tableau:
            resources = self._get_numeric_resource_count(card)
            behavior = self._classify_card_resource_behavior(card)
            vp_per_resource = self._extract_vp_per_resource(card)
            if resources > 0:
                encoding[42] += min(resources / 10.0, 1.0)  # Normalized resource count
            if behavior == 'vp_accumulation':
                vp_resource_cards += 1.0
                if vp_per_resource > 0.0 and resources > 0.0:
                    vp_resource_value += min((resources * vp_per_resource) / 10.0, 1.0)
            elif behavior == 'conversion':
                conversion_cards += 1.0
                threshold = self._get_conversion_threshold(card)
                if threshold > 0.0 and resources >= threshold:
                    conversion_ready += 1.0
            elif behavior == 'stealing':
                stealing_cards += 1.0
            elif behavior == 'adding':
                adding_cards += 1.0

        total_cards = float(max(len(tableau), 1))
        encoding[43] = min(vp_resource_cards / 6.0, 1.0)
        encoding[44] = min(conversion_cards / 6.0, 1.0)
        encoding[45] = min(stealing_cards / 6.0, 1.0)
        encoding[46] = min(adding_cards / 6.0, 1.0)
        encoding[47] = min(vp_resource_value / max(vp_resource_cards, 1.0), 1.0) if vp_resource_cards > 0 else 0.0
        encoding[48] = min(conversion_ready / max(conversion_cards, 1.0), 1.0) if conversion_cards > 0 else 0.0
        encoding[49] = min((vp_resource_cards + conversion_cards + stealing_cards + adding_cards) / total_cards, 1.0)
        
        return encoding
    
    def _encode_hand_cards(self, player_state: Dict[str, Any]) -> List[float]:
        """Encode cards in hand with aggregate and top-K card-level features."""
        encoding = [0.0] * 50

        hand_cards = self._get_owned_hand_cards(player_state)
        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '')
        candidate_cards = self._get_candidate_hand_cards(player_state)

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
        rust_tfm_rl = get_rust_module(required=True)
        player_json = json.dumps(player)

        def _affordability(cost: float, tags: Dict[str, int]) -> float:
            return float(
                rust_tfm_rl.estimate_affordability(
                    player_json,
                    json.dumps({'cost': cost, 'tags': tags}),
                )
            )

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

        hand_affordable: Dict[int, bool] = {}
        if hand_cards:
            affordability_cards: List[Dict[str, Any]] = []
            for card in hand_cards:
                name = str(card.get('name', '') or '')
                tags = self._get_card_tags(name, fallback=card.get('tags', {}))
                affordability_cards.append(
                    {
                        'name': name,
                        'cost': self._get_card_cost(card),
                        'tags': tags,
                    }
                )
            try:
                flags = rust_tfm_rl.can_afford_cards(player_json, json.dumps(affordability_cards))
                for idx, raw in enumerate(flags):
                    hand_affordable[idx] = bool(raw)
            except Exception:
                hand_affordable = {}

        affordable_count = 0
        for idx, card in enumerate(hand_cards):
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost = self._get_card_cost(card)
            if hand_affordable:
                is_affordable = bool(hand_affordable.get(idx, False))
            else:
                is_affordable = _affordability(cost, tags) >= 0.99
            if is_affordable:
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
            requirement_plan = self._evaluate_card_requirement_plan(card, player_state)
            readiness_score, reachability_score, unmet_requirement_penalty = self._requirement_penalty_details(requirement_plan)

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
                    + (0.85 * readiness_score)
                    + (0.35 * reachability_score)
                    - (0.60 * unmet_requirement_penalty)
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
                'requirement_plan': requirement_plan,
                'readiness_score': readiness_score,
                'reachability_score': reachability_score,
                'unmet_requirement_penalty': unmet_requirement_penalty,
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

    def _card_token_segment_bounds(self) -> Tuple[int, int]:
        if self.state_size <= 0 or self.card_token_vector_size <= 0 or self.card_token_dim <= 0:
            return (0, 0)
        start = max(0, int(self.state_size) - int(self.card_token_vector_size))
        end = min(int(self.state_size), start + int(self.card_token_vector_size))
        return (start, end)

    def _build_tag_profile_from_cards(self, cards: List[Dict[str, Any]]) -> Dict[str, int]:
        profile: Dict[str, int] = {}
        for card in cards:
            if not isinstance(card, dict):
                continue
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            for tag_name, present in tags.items():
                if present:
                    profile[tag_name] = int(profile.get(tag_name, 0)) + 1
        return profile

    def _merge_tag_profiles(self, *profiles: Dict[str, int]) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            for tag_name, count in profile.items():
                merged[tag_name] = int(merged.get(tag_name, 0)) + int(count or 0)
        return merged

    def _card_dedupe_key(self, card: Dict[str, Any]) -> str:
        name = str(card.get('name', '') or '').strip()
        if name:
            return name
        try:
            return json.dumps(card, sort_keys=True)
        except Exception:
            return repr(sorted(card.items()))

    def _dedupe_cards(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            if not isinstance(card, dict):
                continue
            key = self._card_dedupe_key(card)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(card)
        return deduped

    def _get_owned_hand_cards(self, player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        hand_cards_raw = player_state.get('cardsInHand', [])
        hand_cards = [card for card in hand_cards_raw if isinstance(card, dict)]
        if hand_cards:
            return self._dedupe_cards(hand_cards)

        player_hand = (player_state.get('thisPlayer', {}) or {}).get('cardsInHand', [])
        hand_cards = [card for card in player_hand if isinstance(card, dict)]
        return self._dedupe_cards(hand_cards)

    def _get_prompt_hand_cards(self, player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '')
        if waiting_type == 'or':
            return self._get_or_project_card_candidates(waiting_for)
        if waiting_type in ['card', 'projectCard', 'selectCard', 'selectProjectCardToPlay']:
            prompt_cards_raw = waiting_for.get('cards', []) if isinstance(waiting_for, dict) else []
            return self._dedupe_cards([card for card in prompt_cards_raw if isinstance(card, dict)])
        return []

    def _get_candidate_hand_cards(self, player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '')
        prompt_cards = self._get_prompt_hand_cards(player_state)
        hand_cards = self._get_owned_hand_cards(player_state)

        if prompt_cards:
            return self._dedupe_cards(prompt_cards + hand_cards)
        if waiting_type in ['initialCards', 'selectInitialCards']:
            startup_cards: List[Dict[str, Any]] = []
            for option in waiting_for.get('options', []) or []:
                if not isinstance(option, dict):
                    continue
                for card in option.get('cards', []) or []:
                    if isinstance(card, dict):
                        startup_cards.append(card)
            if startup_cards:
                return self._dedupe_cards(startup_cards)
        return hand_cards

    def _get_or_project_card_candidates(self, waiting_for: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(waiting_for, dict):
            return []
        if str(waiting_for.get('type', '') or '') != 'or':
            return []

        collected: List[Dict[str, Any]] = []
        seen_names: set[str] = set()
        for option in waiting_for.get('options', []) or []:
            if not isinstance(option, dict):
                continue
            option_type = str(option.get('type', '') or '')
            if option_type not in ['projectCard', 'selectProjectCardToPlay']:
                continue
            for card in option.get('cards', []) or []:
                if not isinstance(card, dict):
                    continue
                name = str(card.get('name', '') or '').strip()
                dedupe_key = name or repr(sorted(card.items()))
                if dedupe_key in seen_names:
                    continue
                seen_names.add(dedupe_key)
                collected.append(card)
        return collected

    def _estimate_affordability_for_card(
        self,
        player: Dict[str, Any],
        card: Dict[str, Any],
        tags: Optional[Dict[str, int]] = None,
    ) -> float:
        resolved_tags = tags or self._get_card_tags(str(card.get('name', '') or ''), fallback=card.get('tags', {}))
        cost = self._get_card_cost(card)
        rust_tfm_rl = get_rust_module(required=True)
        return float(
            rust_tfm_rl.estimate_affordability(
                json.dumps(player),
                json.dumps({'cost': cost, 'tags': resolved_tags}),
            )
        )

    def _get_card_metadata(self, card: Dict[str, Any]) -> Dict[str, Any]:
        name = str(card.get('name', '') or '')
        if not name:
            return {}
        return self.card_metadata_by_name.get(name, {}) or {}

    def _evaluate_card_requirement_plan(
        self,
        card: Dict[str, Any],
        player_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return self.requirement_planner.evaluate_card(card, player_state)
        except Exception:
            return {
                'requirements': [],
                'requirement_plan': [],
                'all_satisfied': True,
                'blocking_count': 0,
                'primary_gap_label': '',
                'primary_gap_axis': '',
                'reachability_score': 1.0,
                'readiness_score': 1.0,
                'plan_summary': 'Requirement planning unavailable.',
                'server_override': False,
                'masked_by_server': False,
            }

    def _requirement_penalty_details(self, requirement_plan: Dict[str, Any]) -> Tuple[float, float, float]:
        readiness = float(requirement_plan.get('readiness_score', 1.0) or 1.0)
        reachability = float(requirement_plan.get('reachability_score', 1.0) or 1.0)
        blocking_count = int(requirement_plan.get('blocking_count', 0) or 0)
        unmet_penalty = min(1.0, float(blocking_count) / 3.0)
        for row in requirement_plan.get('requirement_plan', []) or []:
            if not isinstance(row, dict) or bool(row.get('satisfied', False)):
                continue
            axis = str(row.get('type', '') or '')
            if axis == 'temperature' and int(row.get('remaining_steps', 0) or 0) > 4:
                unmet_penalty += 0.15
            elif axis in ('oxygen', 'oceans', 'venus') and int(row.get('remaining', 0) or 0) > 3:
                unmet_penalty += 0.10
        return readiness, reachability, min(1.0, unmet_penalty)

    def _get_numeric_resource_count(self, card: Dict[str, Any]) -> float:
        try:
            value = float(card.get('resources', 0) or 0)
            return max(value, 0.0)
        except Exception:
            return 0.0

    def _normalize_resource_type(self, value: Any) -> str:
        raw = str(value or '').strip().lower().replace('-', '').replace('_', '').replace(' ', '')
        if not raw:
            return ''
        alias = {
            'microbe': 'microbe',
            'microbes': 'microbe',
            'animal': 'animal',
            'animals': 'animal',
            'floater': 'floater',
            'floaters': 'floater',
            'science': 'science',
            'fighter': 'fighter',
            'fighters': 'fighter',
            'asteroid': 'asteroid',
            'astro': 'asteroid',
        }
        return alias.get(raw, raw)

    def _get_card_resource_type(self, card: Dict[str, Any]) -> str:
        card_type = self._normalize_resource_type(card.get('resourceType', ''))
        if card_type:
            return card_type
        meta = self._get_card_metadata(card)
        return self._normalize_resource_type(meta.get('resourceType', ''))

    def _extract_vp_per_resource(self, card: Dict[str, Any]) -> float:
        meta = self._get_card_metadata(card)
        raw_meta = meta.get('vpPerResource')
        raw_card = card.get('vpPerResource')
        for value in (raw_card, raw_meta):
            try:
                vp_per_resource = float(value)
                if vp_per_resource > 0.0:
                    return vp_per_resource
            except Exception:
                continue

        dynamic_meta = meta.get('dynamicVictoryPoints') if isinstance(meta.get('dynamicVictoryPoints'), dict) else {}
        dynamic_card = card.get('dynamicVictoryPoints') if isinstance(card.get('dynamicVictoryPoints'), dict) else {}
        for dyn in (dynamic_card, dynamic_meta):
            try:
                points = float(dyn.get('points', 0) or 0)
                target = float(dyn.get('target', 0) or 0)
                if points > 0.0 and target > 0.0:
                    return points / target
            except Exception:
                continue

        for raw_vp in (card.get('victoryPoints', None), meta.get('victoryPoints', None)):
            if isinstance(raw_vp, str) and '/' in raw_vp:
                parts = raw_vp.split('/')
                if len(parts) >= 2:
                    try:
                        points = float(parts[0])
                        target = float(parts[1])
                        if points > 0.0 and target > 0.0:
                            return points / target
                    except Exception:
                        pass
        return 0.0

    def _get_conversion_threshold(self, card: Dict[str, Any]) -> float:
        meta = self._get_card_metadata(card)
        for value in (card.get('resourceConversionThreshold', None), meta.get('resourceConversionThreshold', None)):
            try:
                threshold = float(value)
                if threshold > 0.0:
                    return threshold
            except Exception:
                continue
        return 0.0

    def _classify_card_resource_behavior(self, card: Dict[str, Any]) -> str:
        meta = self._get_card_metadata(card)
        behavior_raw = str(card.get('resourceBehavior', meta.get('resourceBehavior', 'none')) or 'none').strip().lower()
        known_behaviors = {'vp_accumulation', 'conversion', 'stealing', 'adding', 'none'}
        behavior = behavior_raw if behavior_raw in known_behaviors else 'none'

        adds_to_any = bool(card.get('resourceActionAddsToAnyCard', meta.get('resourceActionAddsToAnyCard', False)))
        targets_opponent = bool(card.get('resourceActionTargetsOpponent', meta.get('resourceActionTargetsOpponent', False)))
        removes_resources = bool(card.get('resourceActionRemovesResources', meta.get('resourceActionRemovesResources', False)))
        vp_per_resource = self._extract_vp_per_resource(card)
        has_resource_type = bool(self._get_card_resource_type(card))

        # Prefer explicit action semantics first when available.
        if targets_opponent and removes_resources:
            return 'stealing'
        if adds_to_any:
            return 'adding'
        if removes_resources:
            return 'conversion'
        if vp_per_resource > 0.0 and has_resource_type:
            return 'vp_accumulation'
        if behavior in known_behaviors:
            return behavior
        return 'none'

    def _estimate_final_vp_from_resources(self, card: Dict[str, Any]) -> float:
        vp_per_resource = self._extract_vp_per_resource(card)
        if vp_per_resource <= 0.0:
            return 0.0
        return self._get_numeric_resource_count(card) * vp_per_resource

    def _aggregate_resource_totals_by_type(self, cards: List[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = {
            'microbe': 0.0,
            'animal': 0.0,
            'floater': 0.0,
            'science': 0.0,
            'fighter': 0.0,
            'asteroid': 0.0,
        }
        for card in cards or []:
            if not isinstance(card, dict):
                continue
            resource_type = self._get_card_resource_type(card)
            if resource_type not in totals:
                continue
            totals[resource_type] += self._get_numeric_resource_count(card)
        return totals

    def _aggregate_opponent_resource_totals(
        self,
        players: List[Dict[str, Any]],
        own_id: str,
        own_color: str,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {
            'microbe': 0.0,
            'animal': 0.0,
            'floater': 0.0,
            'science': 0.0,
            'fighter': 0.0,
            'asteroid': 0.0,
        }
        for rival in players or []:
            if not isinstance(rival, dict):
                continue
            rival_id = str(rival.get('id', '') or '')
            rival_color = str(rival.get('color', '') or '').strip().lower()
            if own_id and rival_id == own_id:
                continue
            if own_color and rival_color and rival_color == own_color:
                continue
            rival_totals = self._aggregate_resource_totals_by_type(
                [card for card in (rival.get('tableau', []) or []) if isinstance(card, dict)]
            )
            for key, value in rival_totals.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def _build_card_token_features(
        self,
        card: Dict[str, Any],
        player: Dict[str, Any],
        own_tag_profile: Dict[str, int],
        opponent_tag_profile: Dict[str, int],
        card_group: str,
        opponent_resource_totals: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        name = str(card.get('name', '') or '')
        tags = self._get_card_tags(name, fallback=card.get('tags', {}))
        cost_norm = min(self._get_card_cost(card) / 50.0, 1.0)
        vp_proxy = self._get_card_vp_proxy(card, tags)
        affordability = self._estimate_affordability_for_card(player, card, tags)

        key_tags = ['Science', 'Building', 'Space', 'Earth', 'Jovian']
        key_tag_density = sum(1 for tag_name in key_tags if tags.get(tag_name, 0) > 0) / float(len(key_tags))
        own_overlap = sum(float(own_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
        opp_overlap = sum(float(opponent_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
        own_overlap_norm = min(own_overlap / 8.0, 1.0)
        opp_overlap_norm = min(opp_overlap / 8.0, 1.0)

        # Keep token width fixed; opponent cards use hate-draft signal in the last slot.
        final_slot = max(0.0, opp_overlap_norm - own_overlap_norm)
        if card_group == 'tableau':
            final_slot = own_overlap_norm
        elif card_group == 'opponent':
            affordability = 0.5
            final_slot = opp_overlap_norm

        base_features = [
            float(cost_norm),
            float(vp_proxy),
            float(affordability),
            float(key_tag_density),
            1.0 if tags.get('Building', 0) > 0 else 0.0,
            1.0 if tags.get('Space', 0) > 0 else 0.0,
            1.0 if tags.get('Science', 0) > 0 else 0.0,
            float(final_slot),
        ]

        resource_type = self._get_card_resource_type(card)
        behavior = self._classify_card_resource_behavior(card)
        resources = self._get_numeric_resource_count(card)
        vp_per_resource = self._extract_vp_per_resource(card)
        estimated_vp = self._estimate_final_vp_from_resources(card)
        threshold = self._get_conversion_threshold(card)
        readiness = (resources / threshold) if threshold > 0.0 else 0.0
        opponent_totals = opponent_resource_totals or {}
        opponent_pressure = 0.0
        if resource_type:
            opponent_pressure = min(float(opponent_totals.get(resource_type, 0.0)) / 12.0, 1.0)

        resource_features = [
            min(resources / 20.0, 1.0),                                 # [8] resource count on card
            1.0 if resource_type else 0.0,                               # [9] has resource type
            1.0 if resource_type == 'microbe' else 0.0,                  # [10] resource type microbe
            1.0 if resource_type == 'animal' else 0.0,                   # [11] resource type animal
            1.0 if resource_type == 'floater' else 0.0,                  # [12] resource type floater
            1.0 if behavior == 'vp_accumulation' else 0.0,               # [13] VP accumulation behavior
            1.0 if behavior == 'conversion' else 0.0,                    # [14] conversion behavior
            1.0 if behavior == 'stealing' else 0.0,                      # [15] stealing behavior
            1.0 if behavior == 'adding' else 0.0,                        # [16] adding behavior
            min(vp_per_resource / 2.0, 1.0),                             # [17] VP ratio signal
            min(estimated_vp / 15.0, 1.0),                               # [18] expected VP at game end
            min(readiness / 3.0, 1.0),                                   # [19] conversion readiness
        ]
        if behavior == 'stealing':
            # Local indices are [0..11] for resource_features; slot 11 is readiness/pressure.
            resource_features[11] = max(resource_features[11], opponent_pressure)
        elif behavior == 'adding':
            resource_features[11] = max(resource_features[11], opponent_pressure)

        all_features = base_features + resource_features
        if len(all_features) >= self.card_token_dim:
            return all_features[:self.card_token_dim]

        padded = list(all_features)
        while len(padded) < self.card_token_dim:
            padded.append(0.0)
        return padded

    def _select_top_cards(
        self,
        cards: List[Dict[str, Any]],
        count: int,
        player: Dict[str, Any],
        player_state: Optional[Dict[str, Any]],
        own_tag_profile: Dict[str, int],
        opponent_tag_profile: Dict[str, int],
        card_group: str,
        opponent_resource_totals: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost_norm = min(self._get_card_cost(card) / 50.0, 1.0)
            vp_proxy = self._get_card_vp_proxy(card, tags)
            affordability = self._estimate_affordability_for_card(player, card, tags)
            key_tag_density = sum(
                1 for tag_name in ['Science', 'Building', 'Space', 'Earth', 'Jovian'] if tags.get(tag_name, 0) > 0
            ) / 5.0
            own_overlap = sum(float(own_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
            opp_overlap = sum(float(opponent_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
            own_overlap_norm = min(own_overlap / 8.0, 1.0)
            opp_overlap_norm = min(opp_overlap / 8.0, 1.0)
            hate_signal = max(0.0, opp_overlap_norm - own_overlap_norm)

            behavior = self._classify_card_resource_behavior(card)
            resource_type = self._get_card_resource_type(card)
            resources_norm = min(self._get_numeric_resource_count(card) / 12.0, 1.0)
            vp_per_resource = self._extract_vp_per_resource(card)
            vp_resource_value = min((self._get_numeric_resource_count(card) * max(vp_per_resource, 0.0)) / 12.0, 1.0)
            threshold = self._get_conversion_threshold(card)
            conversion_ready = min((self._get_numeric_resource_count(card) / threshold), 2.0) / 2.0 if threshold > 0.0 else 0.0
            opponent_pressure = 0.0
            if resource_type and isinstance(opponent_resource_totals, dict):
                opponent_pressure = min(float(opponent_resource_totals.get(resource_type, 0.0)) / 12.0, 1.0)
            readiness_score = 1.0
            reachability_score = 1.0
            unmet_requirement_penalty = 0.0
            if card_group not in ('tableau', 'opponent') and isinstance(player_state, dict):
                requirement_plan = self._evaluate_card_requirement_plan(card, player_state)
                readiness_score, reachability_score, unmet_requirement_penalty = self._requirement_penalty_details(requirement_plan)

            resource_score = 0.0
            if behavior == 'vp_accumulation':
                resource_score = (0.65 * vp_resource_value) + (0.35 * resources_norm)
            elif behavior == 'conversion':
                threshold_norm = min(threshold / 8.0, 1.0) if threshold > 0.0 else 0.0
                resource_score = (0.55 * conversion_ready) + (0.25 * resources_norm) + (0.20 * (1.0 - threshold_norm))
            elif behavior == 'stealing':
                resource_score = (0.50 * opponent_pressure) + (0.25 * resources_norm) + (0.25 * hate_signal)
            elif behavior == 'adding':
                resource_score = (0.45 * resources_norm) + (0.35 * opponent_pressure)

            if card_group == 'tableau':
                score = (
                    (1.10 * vp_proxy)
                    + (0.80 * key_tag_density)
                    + (0.75 * own_overlap_norm)
                    + (0.80 * resource_score)
                    - (0.20 * cost_norm)
                )
            elif card_group == 'opponent':
                score = (
                    (1.10 * vp_proxy)
                    + (0.90 * key_tag_density)
                    + (0.80 * opp_overlap_norm)
                    + (0.45 * resources_norm)
                    + (0.15 * cost_norm)
                )
            else:
                score = (
                    (1.15 * affordability)
                    + (0.90 * vp_proxy)
                    + (0.60 * key_tag_density)
                    + (0.55 * own_overlap_norm)
                    + (0.40 * hate_signal)
                    + (0.55 * resource_score)
                    + (0.85 * readiness_score)
                    + (0.35 * reachability_score)
                    - (0.60 * unmet_requirement_penalty)
                    + (0.20 * (1.0 - cost_norm))
                )

            tie_break = (sum(ord(ch) for ch in name) % 100) * 1e-6
            ranked.append((float(score + tie_break), card))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:count]]

    def _inject_card_token_features(self, state_vector: np.ndarray, player_state: Dict[str, Any]) -> None:
        if not isinstance(state_vector, np.ndarray) or state_vector.ndim != 1:
            return
        if self.card_token_count <= 0 or self.card_token_dim <= 0:
            return

        start, end = self._card_token_segment_bounds()
        if end <= start:
            return

        usable_length = int(end - start)
        token_slots = int(usable_length // self.card_token_dim)
        if token_slots <= 0:
            return

        player = player_state.get('thisPlayer', {}) or {}
        tableau_cards = [card for card in (player.get('tableau', []) or []) if isinstance(card, dict)]
        owned_hand_cards = self._get_owned_hand_cards(player_state)
        prompt_hand_cards = self._get_prompt_hand_cards(player_state)

        players = player_state.get('players', []) or []
        own_id = str(player.get('id', '') or '')
        own_color = str(player.get('color', '') or '').strip().lower()
        opponent_tableau_cards: List[Dict[str, Any]] = []
        for rival in players:
            if not isinstance(rival, dict):
                continue
            rival_id = str(rival.get('id', '') or '')
            rival_color = str(rival.get('color', '') or '').strip().lower()
            if own_id and rival_id == own_id:
                continue
            if own_color and rival_color and rival_color == own_color:
                continue
            opponent_tableau_cards.extend(
                [card for card in (rival.get('tableau', []) or []) if isinstance(card, dict)]
            )
        opponent_resource_totals = self._aggregate_opponent_resource_totals(players, own_id, own_color)

        own_tag_profile = self._merge_tag_profiles(
            self._build_tag_profile_from_cards(tableau_cards),
            self._build_tag_profile_from_cards(owned_hand_cards),
        )
        opponent_tag_profile = self._build_tag_profile_from_cards(opponent_tableau_cards)

        selected_tableau = self._select_top_cards(
            tableau_cards,
            min(self.tableau_token_count, token_slots),
            player,
            player_state,
            own_tag_profile,
            opponent_tag_profile,
            card_group='tableau',
            opponent_resource_totals=opponent_resource_totals,
        )
        remaining_slots = max(0, token_slots - len(selected_tableau))
        hand_slot_cap = min(self.hand_token_count, remaining_slots)
        selected_hand: List[Dict[str, Any]] = []
        if hand_slot_cap > 0 and prompt_hand_cards:
            selected_prompt = self._select_top_cards(
                prompt_hand_cards,
                hand_slot_cap,
                player,
                player_state,
                own_tag_profile,
                opponent_tag_profile,
                card_group='hand',
                opponent_resource_totals=opponent_resource_totals,
            )
            selected_hand.extend(selected_prompt)
            prompt_keys = {self._card_dedupe_key(card) for card in selected_prompt}
            supplemental_hand = [
                card for card in owned_hand_cards
                if self._card_dedupe_key(card) not in prompt_keys
            ]
            remaining_hand_slots = max(0, hand_slot_cap - len(selected_hand))
            if remaining_hand_slots > 0:
                selected_hand.extend(
                    self._select_top_cards(
                        supplemental_hand,
                        remaining_hand_slots,
                        player,
                        player_state,
                        own_tag_profile,
                        opponent_tag_profile,
                        card_group='hand',
                        opponent_resource_totals=opponent_resource_totals,
                    )
                )
        else:
            selected_hand = self._select_top_cards(
                owned_hand_cards,
                hand_slot_cap,
                player,
                player_state,
                own_tag_profile,
                opponent_tag_profile,
                card_group='hand',
                opponent_resource_totals=opponent_resource_totals,
            )
        remaining_slots = max(0, remaining_slots - len(selected_hand))
        selected_opponent = self._select_top_cards(
            opponent_tableau_cards,
            min(self.opponent_token_count, remaining_slots),
            player,
            player_state,
            own_tag_profile,
            opponent_tag_profile,
            card_group='opponent',
            opponent_resource_totals=opponent_resource_totals,
        )

        token_cards: List[Tuple[Dict[str, Any], str]] = []
        token_cards.extend((card, 'tableau') for card in selected_tableau)
        token_cards.extend((card, 'hand') for card in selected_hand)
        token_cards.extend((card, 'opponent') for card in selected_opponent)

        flat_tokens = np.zeros((token_slots * self.card_token_dim,), dtype=np.float32)
        for idx, (card, card_group) in enumerate(token_cards[:token_slots]):
            token_values = self._build_card_token_features(
                card=card,
                player=player,
                own_tag_profile=own_tag_profile,
                opponent_tag_profile=opponent_tag_profile,
                card_group=card_group,
                opponent_resource_totals=opponent_resource_totals,
            )
            token_vec = np.asarray(token_values[:self.card_token_dim], dtype=np.float32)
            base = idx * self.card_token_dim
            flat_tokens[base:base + self.card_token_dim] = token_vec

        write_len = min(usable_length, flat_tokens.size)
        state_vector[start:start + write_len] = flat_tokens[:write_len]

    def build_prompt_card_rankings(
        self,
        player_state: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        waiting_for = player_state.get('waitingFor', {}) or {}
        waiting_type = str(waiting_for.get('type', '') or '')
        if waiting_type not in ['card', 'projectCard', 'selectCard', 'selectProjectCardToPlay', 'or']:
            return []

        player = player_state.get('thisPlayer', {}) or {}
        if waiting_type == 'or':
            prompt_cards = self._get_or_project_card_candidates(waiting_for)
        else:
            prompt_cards = [card for card in (waiting_for.get('cards', []) or []) if isinstance(card, dict)]
        if not prompt_cards:
            return []

        players = player_state.get('players', []) or []
        own_id = str(player.get('id', '') or '')
        own_color = str(player.get('color', '') or '').strip().lower()
        tableau_cards = [card for card in (player.get('tableau', []) or []) if isinstance(card, dict)]
        owned_hand_cards = self._get_owned_hand_cards(player_state)
        opponent_tableau_cards: List[Dict[str, Any]] = []
        for rival in players:
            if not isinstance(rival, dict):
                continue
            rival_id = str(rival.get('id', '') or '')
            rival_color = str(rival.get('color', '') or '').strip().lower()
            if own_id and rival_id == own_id:
                continue
            if own_color and rival_color and rival_color == own_color:
                continue
            opponent_tableau_cards.extend([card for card in (rival.get('tableau', []) or []) if isinstance(card, dict)])

        own_tag_profile = self._merge_tag_profiles(
            self._build_tag_profile_from_cards(tableau_cards),
            self._build_tag_profile_from_cards(owned_hand_cards),
        )
        opponent_tag_profile = self._build_tag_profile_from_cards(opponent_tableau_cards)
        opponent_resource_totals = self._aggregate_opponent_resource_totals(players, own_id, own_color)

        ranked: List[Dict[str, Any]] = []
        for card in prompt_cards:
            name = str(card.get('name', '') or '')
            tags = self._get_card_tags(name, fallback=card.get('tags', {}))
            cost_norm = min(self._get_card_cost(card) / 50.0, 1.0)
            vp_proxy = self._get_card_vp_proxy(card, tags)
            affordability = self._estimate_affordability_for_card(player, card, tags)
            key_tag_density = sum(
                1 for tag_name in ['Science', 'Building', 'Space', 'Earth', 'Jovian'] if tags.get(tag_name, 0) > 0
            ) / 5.0
            own_overlap = sum(float(own_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
            opp_overlap = sum(float(opponent_tag_profile.get(tag_name, 0)) for tag_name, present in tags.items() if present)
            own_overlap_norm = min(own_overlap / 8.0, 1.0)
            opp_overlap_norm = min(opp_overlap / 8.0, 1.0)
            hate_signal = max(0.0, opp_overlap_norm - own_overlap_norm)

            behavior = self._classify_card_resource_behavior(card)
            resource_type = self._get_card_resource_type(card)
            resources_norm = min(self._get_numeric_resource_count(card) / 12.0, 1.0)
            vp_per_resource = self._extract_vp_per_resource(card)
            vp_resource_value = min((self._get_numeric_resource_count(card) * max(vp_per_resource, 0.0)) / 12.0, 1.0)
            threshold = self._get_conversion_threshold(card)
            conversion_ready = min((self._get_numeric_resource_count(card) / threshold), 2.0) / 2.0 if threshold > 0.0 else 0.0
            opponent_pressure = min(float(opponent_resource_totals.get(resource_type, 0.0)) / 12.0, 1.0) if resource_type else 0.0

            resource_score = 0.0
            if behavior == 'vp_accumulation':
                resource_score = (0.65 * vp_resource_value) + (0.35 * resources_norm)
            elif behavior == 'conversion':
                threshold_norm = min(threshold / 8.0, 1.0) if threshold > 0.0 else 0.0
                resource_score = (0.55 * conversion_ready) + (0.25 * resources_norm) + (0.20 * (1.0 - threshold_norm))
            elif behavior == 'stealing':
                resource_score = (0.50 * opponent_pressure) + (0.25 * resources_norm) + (0.25 * hate_signal)
            elif behavior == 'adding':
                resource_score = (0.45 * resources_norm) + (0.35 * opponent_pressure)

            requirement_plan = self._evaluate_card_requirement_plan(card, player_state)
            readiness_score, reachability_score, unmet_requirement_penalty = self._requirement_penalty_details(requirement_plan)
            selection_score = (
                (1.15 * affordability)
                + (0.90 * vp_proxy)
                + (0.60 * key_tag_density)
                + (0.55 * own_overlap_norm)
                + (0.40 * hate_signal)
                + (0.55 * resource_score)
                + (0.85 * readiness_score)
                + (0.35 * reachability_score)
                - (0.60 * unmet_requirement_penalty)
                + (0.20 * (1.0 - cost_norm))
            )

            ranked.append({
                'name': name,
                'cost': self._get_card_cost(card),
                'tags': sorted([tag_name for tag_name, present in tags.items() if present]),
                'disabled': bool(card.get('isDisabled', False)),
                'warnings': list(card.get('warnings', []) or []),
                'selection_score': float(selection_score),
                'affordability': float(affordability),
                'vp_proxy': float(vp_proxy),
                'requirements': list(requirement_plan.get('requirements', []) or []),
                'requirement_plan': list(requirement_plan.get('requirement_plan', []) or []),
                'plan_summary': str(requirement_plan.get('plan_summary', '') or ''),
                'reachability_score': float(requirement_plan.get('reachability_score', 1.0) or 1.0),
                'readiness_score': float(requirement_plan.get('readiness_score', 1.0) or 1.0),
                'all_satisfied': bool(requirement_plan.get('all_satisfied', True)),
                'blocking_count': int(requirement_plan.get('blocking_count', 0) or 0),
                'server_override': bool(requirement_plan.get('server_override', False)),
                'masked_by_server': bool(requirement_plan.get('masked_by_server', False)),
            })

        ranked.sort(
            key=lambda item: (
                float(item.get('selection_score', 0.0)),
                float(item.get('readiness_score', 0.0)),
                float(item.get('reachability_score', 0.0)),
                -float(item.get('cost', 0.0)),
            ),
            reverse=True,
        )
        if limit is not None:
            return ranked[:max(0, int(limit))]
        return ranked
    
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

        # Moon board state (slots 10-29): only populated when moon expansion is present
        moon_enc = self._encode_moon_board_state(game_state, current_player)
        for i, val in enumerate(moon_enc):
            if 10 + i < 100:
                encoding[10 + i] = val

        return encoding

    def _encode_moon_board_state(
        self,
        game_state: Dict[str, Any],
        current_player: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Encode Moon board state. Returns 20 values. TileType: 29=mine, 30=habitat, 31=road."""
        encoding = [0.0] * 20
        moon = game_state.get('moon', {})
        spaces_raw = moon.get('spaces', [])
        if not spaces_raw:
            return encoding

        spaces = [s for s in spaces_raw if isinstance(s, dict)]
        own_color = str((current_player or {}).get('color', '') or '').strip().lower()
        vpb = (current_player or {}).get('victoryPointsBreakdown', {}) or {}

        road_count = 0
        mine_count = 0
        habitat_count = 0
        available_moon_land = 0
        own_roads = 0
        own_mines = 0
        own_habitats = 0

        for space in spaces:
            st = self._space_type_lower(space.get('spaceType', ''))
            if st == 'colony':
                continue
            tile_raw = space.get('tileType')
            has_tile = tile_raw is not None
            tid = self._safe_int(tile_raw)
            color = str(space.get('color', '') or '').strip().lower()

            if not has_tile:
                if st in ('land', 'lunar_mine'):
                    available_moon_land += 1
                continue

            if tid == 31:
                road_count += 1
                if color == own_color:
                    own_roads += 1
            elif tid == 29:
                mine_count += 1
                if color == own_color:
                    own_mines += 1
            elif tid == 30:
                habitat_count += 1
                if color == own_color:
                    own_habitats += 1

        logistics = moon.get('logisticsRate', 0)
        mining = moon.get('miningRate', 0)
        habitat = moon.get('habitatRate', 0)

        encoding[0] = float(logistics) / 8.0
        encoding[1] = float(mining) / 8.0
        encoding[2] = float(habitat) / 8.0
        encoding[3] = min(road_count / 24.0, 1.0)
        encoding[4] = min(mine_count / 24.0, 1.0)
        encoding[5] = min(habitat_count / 24.0, 1.0)
        encoding[6] = min(available_moon_land / 35.0, 1.0)
        encoding[7] = min(own_roads / 12.0, 1.0)
        encoding[8] = min(own_mines / 12.0, 1.0)
        encoding[9] = min(own_habitats / 12.0, 1.0)
        encoding[10] = min(float(vpb.get('moonRoads', 0) or 0) / 12.0, 1.0)
        encoding[11] = min(float(vpb.get('moonMines', 0) or 0) / 12.0, 1.0)
        encoding[12] = min(float(vpb.get('moonHabitats', 0) or 0) / 12.0, 1.0)
        encoding[13] = min((own_roads + own_mines + own_habitats) / 24.0, 1.0)

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
    
    def _encode_game_phase(self, game_state: Dict[str, Any], turn_action_count: int = 0) -> List[float]:
        """Encode current game phase and timing.

        Slot layout (10 values total):
          [0-4]  Phase one-hot: research / drafting / action / production / solar
          [5]    Generation progress (0→1 over 14 generations)
          [6]    Is-active-player flag
          [7]    First-action slot indicator  (1.0 if turn_action_count == 0)
          [8]    Second-action slot indicator (1.0 if turn_action_count == 1)
          [9]    Further-action indicator     (clamped count / 4 for 3+)

        Slots 7-9 let the policy network distinguish "I still have both actions"
        from "I only have my second action left", enabling it to plan sequences
        such as: play engine card first → claim milestone second.
        """
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

        # Turn-action slots (only meaningful during action phase)
        count = max(0, int(turn_action_count))
        encoding[7] = 1.0 if count == 0 else 0.0   # first action
        encoding[8] = 1.0 if count == 1 else 0.0   # second action
        encoding[9] = min(float(max(0, count - 1)) / 4.0, 1.0)  # overflow (blue cards etc.)
        
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

                # Opponent card-resource detail (slots 10-19 of each opponent block).
                tableau_cards = [card for card in (player.get('tableau', []) or []) if isinstance(card, dict)]
                resource_totals = self._aggregate_resource_totals_by_type(tableau_cards)
                total_resource_tokens = 0.0
                resource_holders = 0.0
                highest_stack = 0.0
                for card in tableau_cards:
                    resources = self._get_numeric_resource_count(card)
                    if resources <= 0.0:
                        continue
                    resource_type = self._get_card_resource_type(card)
                    if resource_type:
                        resource_holders += 1.0
                    total_resource_tokens += resources
                    highest_stack = max(highest_stack, resources)

                stealable_total = (
                    resource_totals.get('microbe', 0.0)
                    + resource_totals.get('animal', 0.0)
                    + resource_totals.get('floater', 0.0)
                    + resource_totals.get('science', 0.0)
                    + resource_totals.get('fighter', 0.0)
                )

                encoding[base_idx + 10] = min(total_resource_tokens / 20.0, 1.0)
                encoding[base_idx + 11] = min(resource_holders / 8.0, 1.0)
                encoding[base_idx + 12] = min(resource_totals.get('microbe', 0.0) / 10.0, 1.0)
                encoding[base_idx + 13] = min(resource_totals.get('animal', 0.0) / 10.0, 1.0)
                encoding[base_idx + 14] = min(resource_totals.get('floater', 0.0) / 10.0, 1.0)
                encoding[base_idx + 15] = min(resource_totals.get('science', 0.0) / 10.0, 1.0)
                encoding[base_idx + 16] = min(resource_totals.get('fighter', 0.0) / 10.0, 1.0)
                encoding[base_idx + 17] = min(resource_totals.get('asteroid', 0.0) / 10.0, 1.0)
                encoding[base_idx + 18] = min(highest_stack / 8.0, 1.0)
                encoding[base_idx + 19] = min(stealable_total / 15.0, 1.0)
                 
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
            elif input_type in ['initialCards', 'selectInitialCards']:
                options = [opt for opt in (waiting_for.get('options', []) or []) if isinstance(opt, dict)]
                corp_option = None
                prelude_option = None
                project_option = None
                for idx, option in enumerate(options):
                    title_value = option.get('title', '')
                    if isinstance(title_value, dict):
                        title_value = title_value.get('message', '')
                    title = str(title_value or '').lower()
                    min_cards = int(option.get('min', 0) or 0)
                    max_cards = int(option.get('max', 0) or 0)
                    if corp_option is None and ('corporation' in title or (idx == 0 and min_cards == 1 and max_cards == 1)):
                        corp_option = option
                    elif prelude_option is None and ('prelude' in title or (min_cards == 2 and max_cards == 2)):
                        prelude_option = option
                    elif project_option is None and ('initial cards' in title or 'cards to buy' in title or 'project' in title):
                        project_option = option

                corp_cards = [c for c in (corp_option.get('cards', []) if corp_option else []) if isinstance(c, dict)]
                prelude_cards = [c for c in (prelude_option.get('cards', []) if prelude_option else []) if isinstance(c, dict)]
                project_cards = [c for c in (project_option.get('cards', []) if project_option else []) if isinstance(c, dict)]

                encoding[42] = min(len(corp_cards) / 4.0, 1.0)
                best_starting_mc = 0.0
                best_card_cost = max(float((player_state.get('thisPlayer', {}) or {}).get('cardCost', 3) or 3), 1.0)
                best_keep_cap = 0.0
                project_max = len(project_cards)
                if project_option is not None:
                    try:
                        project_max = int(project_option.get('max', len(project_cards)) or len(project_cards))
                    except Exception:
                        project_max = len(project_cards)
                for corp in corp_cards:
                    name = str(corp.get('name', '') or '')
                    meta = self.card_metadata_by_name.get(name, {}) if name else {}
                    start_mc = float(meta.get('startingMegaCredits', meta.get('startingMegacredits', 0)) or 0)
                    card_cost = float(meta.get('cardCost', (player_state.get('thisPlayer', {}) or {}).get('cardCost', 3)) or 3)
                    card_cost = max(card_cost, 1.0)
                    keep_cap = min(float(project_max), float(max(0, int(start_mc // card_cost))))
                    if keep_cap > best_keep_cap or (keep_cap == best_keep_cap and start_mc > best_starting_mc):
                        best_keep_cap = keep_cap
                        best_starting_mc = start_mc
                        best_card_cost = card_cost

                encoding[43] = min(best_starting_mc / 80.0, 1.0)
                encoding[44] = min(best_card_cost / 8.0, 1.0)
                encoding[45] = min(best_keep_cap / 10.0, 1.0)
                encoding[46] = min(len(prelude_cards) / 8.0, 1.0)

                prelude_utility = 0.0
                if prelude_cards:
                    quality = []
                    for card in prelude_cards:
                        cost_norm = min(self._get_card_cost(card) / 40.0, 1.0)
                        quality.append(float(self._get_card_vp_proxy(card) + (0.25 * (1.0 - cost_norm))))
                    prelude_utility = float(sum(quality) / max(1, len(quality)))
                encoding[47] = min(prelude_utility, 1.0)

                project_qualities: List[float] = []
                for card in project_cards:
                    cost_norm = min(self._get_card_cost(card) / 40.0, 1.0)
                    project_qualities.append(float(self._get_card_vp_proxy(card) + (0.35 * (1.0 - cost_norm))))
                if project_qualities:
                    encoding[48] = min(float(sum(project_qualities) / len(project_qualities)), 1.0)
                    encoding[49] = min(float(max(project_qualities)), 1.0)
            elif input_type in ['selectCard', 'projectCard', 'selectProjectCardToPlay']:
                cards = waiting_for.get('cards', [])
                encoding[36] = min(len(cards) / 20.0, 1.0)
            elif input_type in ['selectSpace', 'space']:
                spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
                encoding[37] = min(len(spaces) / 40.0, 1.0)
                # Flag moon space selection (space ids like m01, m34)
                if spaces:
                    first_space = spaces[0]
                    sid = str(
                        first_space.get('id', first_space.get('spaceId', first_space))
                        if isinstance(first_space, dict)
                        else first_space
                    ).strip().lower()
                    if sid.startswith('m') and len(sid) >= 2 and sid[1:].isdigit():
                        encoding[41] = 1.0
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
            os.path.join(two_up, 'card_metadata.json'),
            os.path.join(two_up, 'terraforming-mars', 'card_metadata.json'),
            os.path.join(one_up, 'terraforming-mars', 'card_metadata.json'),
            os.path.join(one_up, 'card_metadata.json'),
        ])

        seen: set = set()
        fallback_data: Optional[Dict[str, Dict[str, Any]]] = None
        fallback_path: str = ''
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
                has_corp_economics = any(
                    isinstance(meta, dict) and (
                        meta.get('startingMegaCredits', None) is not None
                        or meta.get('cardCost', None) is not None
                    )
                    for meta in normalized.values()
                )
                StateEncoder._CARD_METADATA_CACHE[path] = normalized
                if has_corp_economics:
                    logger.info(f"Loaded card metadata for {len(normalized)} cards from {path}")
                    return normalized
                if normalized and fallback_data is None:
                    fallback_data = normalized
                    fallback_path = path
                    logger.info(
                        "Loaded metadata from %s without corporation economics; continuing search",
                        path,
                    )
            except Exception as e:
                logger.warning(f"Failed to load card metadata from {path}: {e}")
                StateEncoder._CARD_METADATA_CACHE[path] = {}

        if fallback_data:
            logger.info(
                "Using fallback card metadata for %d cards from %s",
                len(fallback_data),
                fallback_path or "unknown path",
            )
            return fallback_data
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
    

    

