"""
Shared scoring utilities used by both tournament fitness and policy training.
"""
from typing import Any, Dict, List, Optional
import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


SELECTION_RANK_POINTS = {
    1: _env_float("SELECTION_RANK_1_POINTS", 150.0),
    2: _env_float("SELECTION_RANK_2_POINTS", 50.0),
    3: _env_float("SELECTION_RANK_3_POINTS", 25.0),
    4: _env_float("SELECTION_RANK_4_POINTS", 5.0),
}
SELECTION_VP_WEIGHT = _env_float("SELECTION_VP_WEIGHT", 0.5)
SELECTION_COMPLETION_BONUS = _env_float("SELECTION_COMPLETION_BONUS", 10.0)
SELECTION_INCOMPLETE_PENALTY = _env_float("SELECTION_INCOMPLETE_PENALTY", -50.0)


def calculate_selection_score(rank: Any, victory_points: Any, completed: Any) -> float:
    """Raw game score used for evolutionary selection."""
    try:
        rank_int = int(rank)
    except Exception:
        rank_int = 4

    try:
        vp = float(victory_points)
    except Exception:
        vp = 0.0

    ranking_points = SELECTION_RANK_POINTS.get(rank_int, 0.0)

    vp_bonus = vp * SELECTION_VP_WEIGHT
    completion_bonus = SELECTION_COMPLETION_BONUS if bool(completed) else SELECTION_INCOMPLETE_PENALTY
    return ranking_points + vp_bonus + completion_bonus


def calculate_terminal_reward(rank: Any, victory_points: Any, completed: Any) -> float:
    """
    Normalized policy-learning reward derived from the same objective as selection.
    Keeps optimization stable while preserving score ordering.
    """
    raw = calculate_selection_score(rank, victory_points, completed)
    normalized = (raw - 100.0) / 50.0
    return max(-2.0, min(2.0, float(normalized)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        # Apply Hard Mode scaling
        return float(value)
    except Exception:
        return float(default)


def _extract_player(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    player = state.get('thisPlayer', {})
    return player if isinstance(player, dict) else {}


def _extract_game(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    game = state.get('game', {})
    return game if isinstance(game, dict) else {}


def _extract_hand(state: Optional[Dict[str, Any]], player: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(state, dict):
        cards = state.get('cardsInHand')
        if isinstance(cards, list):
            return cards
    cards_from_player = player.get('cardsInHand')
    if isinstance(cards_from_player, list):
        return cards_from_player
    return []


def _extract_tags(card: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    tags = card.get('tags')
    if isinstance(tags, dict):
        for key, value in tags.items():
            if value:
                out[str(key)] = 1
        return out
    if isinstance(tags, list):
        for item in tags:
            out[str(item)] = 1
    return out


def _card_quality(card: Dict[str, Any], player: Dict[str, Any]) -> float:
    cost = _safe_float(card.get('calculatedCost', card.get('cost', 0)), 0.0)
    tags = _extract_tags(card)
    vp_raw = card.get('victoryPoints', 0)
    vp = max(0.0, _safe_float(vp_raw, 0.0))

    mc = _safe_float(player.get('megaCredits', 0), 0.0)
    steel = _safe_float(player.get('steel', 0), 0.0)
    titanium = _safe_float(player.get('titanium', 0), 0.0)
    steel_value = _safe_float(player.get('steelValue', 2), 2.0)
    titanium_value = _safe_float(player.get('titaniumValue', 3), 3.0)

    purchasing_power = mc
    if tags.get('Building'):
        purchasing_power += steel * steel_value
    if tags.get('Space'):
        purchasing_power += titanium * titanium_value

    affordable = 1.0 if purchasing_power >= cost else max(0.0, 1.0 - ((cost - purchasing_power) / 20.0))
    cheapness = max(0.0, 1.0 - min(cost / 40.0, 1.0))

    tag_score = 0.0
    if tags.get('Science'):
        tag_score += 0.45
    if tags.get('Building'):
        tag_score += 0.35
    if tags.get('Space'):
        tag_score += 0.35
    if tags.get('Earth'):
        tag_score += 0.20
    if tags.get('Plant'):
        tag_score += 0.20

    quality = (affordable * 1.2) + (cheapness * 0.7) + (vp * 0.35) + tag_score
    return max(0.0, min(quality / 3.0, 1.0))


def _hand_quality(cards: List[Dict[str, Any]], player: Dict[str, Any]) -> float:
    if not cards:
        return 0.0
    scored = sorted((_card_quality(card, player) for card in cards), reverse=True)
    top_k = scored[:min(3, len(scored))]
    return float(sum(top_k) / max(1, len(top_k)))


def _can_afford_card_now(card: Dict[str, Any], player: Dict[str, Any]) -> bool:
    cost = _safe_float(card.get('calculatedCost', card.get('cost', 0)), 0.0)
    if cost <= 0.0:
        return True

    tags = _extract_tags(card)
    mc = _safe_float(player.get('megaCredits', 0), 0.0)
    steel = _safe_float(player.get('steel', 0), 0.0)
    titanium = _safe_float(player.get('titanium', 0), 0.0)
    steel_value = _safe_float(player.get('steelValue', 2), 2.0)
    titanium_value = _safe_float(player.get('titaniumValue', 3), 3.0)

    purchasing_power = mc
    if tags.get('Building'):
        purchasing_power += steel * steel_value
    if tags.get('Space'):
        purchasing_power += titanium * titanium_value
    return purchasing_power >= cost


def _vp_component(player: Dict[str, Any], key: str) -> float:
    breakdown = player.get('victoryPointsBreakdown', {})
    if not isinstance(breakdown, dict):
        return 0.0
    return _safe_float(breakdown.get(key, 0), 0.0)


def _normalize_token(value: Any) -> str:
    return str(value or '').strip().lower()


def _matches_player_payload(payload: Dict[str, Any], player: Dict[str, Any]) -> bool:
    own_name = _normalize_token(player.get('name'))
    own_color = _normalize_token(player.get('color'))
    payload_name = _normalize_token(payload.get('playerName'))
    payload_color = _normalize_token(payload.get('playerColor'))
    if own_name and payload_name and own_name == payload_name:
        return True
    if own_color and payload_color and own_color == payload_color:
        return True
    return False


def _owned_milestone_count(game_state: Dict[str, Any], player: Dict[str, Any]) -> int:
    count = 0
    milestones = game_state.get('milestones', []) or []
    for milestone in milestones:
        if isinstance(milestone, dict) and _matches_player_payload(milestone, player):
            count += 1
    return int(count)


def _owned_funded_awards(game_state: Dict[str, Any], player: Dict[str, Any]) -> List[Dict[str, Any]]:
    owned: List[Dict[str, Any]] = []
    awards = game_state.get('awards', []) or []
    for award in awards:
        if not isinstance(award, dict):
            continue
        if _matches_player_payload(award, player):
            owned.append(award)
    return owned


def _funded_award_count(game_state: Dict[str, Any]) -> int:
    count = 0
    awards = game_state.get('awards', []) or []
    for award in awards:
        if not isinstance(award, dict):
            continue
        if award.get('playerName') or award.get('playerColor'):
            count += 1
    return int(count)


def _award_identity(award: Dict[str, Any]) -> str:
    name = _normalize_token(award.get('name') or award.get('title'))
    color = _normalize_token(award.get('playerColor'))
    player_name = _normalize_token(award.get('playerName'))
    return f"{name}|{player_name}|{color}"


def _award_expected_points_for_player(scores: List[Dict[str, Any]], player: Dict[str, Any]) -> float:
    own_name = _normalize_token(player.get('name'))
    own_color = _normalize_token(player.get('color'))
    rows: List[Dict[str, Any]] = []
    for row in scores:
        if not isinstance(row, dict):
            continue
        row_name = _normalize_token(row.get('playerName'))
        row_color = _normalize_token(row.get('playerColor'))
        if not row_name and not row_color:
            continue
        rows.append(
            {
                "name": row_name,
                "color": row_color,
                "score": _safe_float(row.get('score', 0), 0.0),
            }
        )
    if not rows:
        return 0.0

    rows.sort(key=lambda item: item["score"], reverse=True)
    top_score = rows[0]["score"]
    top_rows = [row for row in rows if row["score"] == top_score]

    def _is_own(row: Dict[str, Any]) -> bool:
        if own_color and row["color"] and own_color == row["color"]:
            return True
        if own_name and row["name"] and own_name == row["name"]:
            return True
        return False

    if top_score > 0.0 and any(_is_own(row) for row in top_rows):
        return 5.0

    remaining_rows = [row for row in rows if row["score"] < top_score]
    if not remaining_rows:
        return 0.0
    second_score = remaining_rows[0]["score"]
    second_rows = [row for row in remaining_rows if row["score"] == second_score]
    if second_score > 0.0 and any(_is_own(row) for row in second_rows):
        return 2.0
    return 0.0


def _estimate_award_funding_cost(prior_funded_count: int) -> float:
    # Terraforming Mars award costs: 8, 14, 20 MC.
    return float(8 + (6 * max(0, int(prior_funded_count))))


def _extract_selected_card_name(action_input: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(action_input, dict):
        return None
    stack: List[Dict[str, Any]] = [action_input]
    while stack:
        payload = stack.pop()
        if not isinstance(payload, dict):
            continue
        action_type = _normalize_token(payload.get('type'))
        if action_type == 'projectcard':
            card_name = str(payload.get('card', '') or '').strip()
            if card_name:
                return card_name
        if action_type == 'card':
            selected = payload.get('cards', []) or []
            if selected:
                first = selected[0]
                if isinstance(first, dict):
                    card_name = str(first.get('name', '') or '').strip()
                else:
                    card_name = str(first or '').strip()
                if card_name:
                    return card_name
            card_name = str(payload.get('card', '') or '').strip()
            if card_name:
                return card_name
        nested_response = payload.get('response')
        if isinstance(nested_response, dict):
            stack.append(nested_response)
        nested_responses = payload.get('responses')
        if isinstance(nested_responses, list):
            for item in nested_responses:
                if isinstance(item, dict):
                    stack.append(item)
    return None


def _find_card_by_name(cards: List[Dict[str, Any]], card_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not card_name:
        return None
    needle = _normalize_token(card_name)
    for card in cards:
        if not isinstance(card, dict):
            continue
        if _normalize_token(card.get('name')) == needle:
            return card
    return None


def _card_nominal_vp(card: Optional[Dict[str, Any]]) -> float:
    if not isinstance(card, dict):
        return 0.0
    return max(0.0, _safe_float(card.get('victoryPoints', 0), 0.0))


REWARD_NON_ACTION = -0.01  # Small penalty for doing nothing

# Scoring Mode
SCORING_MODE = os.getenv("SCORING_MODE", "NORMAL").upper()
IS_HARD_MODE = SCORING_MODE == "HARD"

# Reward Scaling
STEP_REWARD_SCALE = 0.3 if IS_HARD_MODE else 1.0
TERMINAL_REWARD_SCALE = 2.0 if IS_HARD_MODE else 1.0

def calculate_step_reward(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    action_input: Dict[str, Any]
) -> float:
    """
    Calculate reward for a single step/action.
    
    Args:
        before_state: Player state before action
        after_state: Player state after action
        action_input: Action that was taken
        
    Returns:
        float: Calculated reward
    """
    if not before_state or not after_state:
        return 0.0

    before_player = _extract_player(before_state)
    after_player = _extract_player(after_state)
    before_game = _extract_game(before_state)
    after_game = _extract_game(after_state)
    if not before_player or not after_player:
        return 0.0

    reward = 0.0

    # Reward tangible engine growth.
    before_tableau = before_player.get('tableau', []) or []
    after_tableau = after_player.get('tableau', []) or []
    tableau_delta = len(after_tableau) - len(before_tableau)
    reward += max(-0.08, min(0.18, 0.05 * float(tableau_delta)))

    production_keys = [
        'megaCreditProduction',
        'steelProduction',
        'titaniumProduction',
        'plantProduction',
        'energyProduction',
        'heatProduction',
    ]
    production_delta = 0.0
    for key in production_keys:
        production_delta += _safe_float(after_player.get(key, 0), 0.0) - _safe_float(before_player.get(key, 0), 0.0)
    reward += max(-0.10, min(0.14, 0.02 * production_delta))

    # Reward direct VP improvements from milestones and awards.
    milestone_delta = _vp_component(after_player, 'milestones') - _vp_component(before_player, 'milestones')
    reward += max(-0.06, min(0.20, 0.035 * milestone_delta))

    award_delta = _vp_component(after_player, 'awards') - _vp_component(before_player, 'awards')
    reward += max(-0.08, min(0.18, 0.030 * award_delta))

    # Reward greener-city map synergy (city VP component is adjacency-driven).
    city_combo_delta = _vp_component(after_player, 'city') - _vp_component(before_player, 'city')
    greenery_delta = _vp_component(after_player, 'greenery') - _vp_component(before_player, 'greenery')
    combo_delta = city_combo_delta + (0.25 * greenery_delta)
    reward += max(-0.10, min(0.15, 0.05 * combo_delta))

    # Reward card quality improvements in hand.
    before_hand = _extract_hand(before_state, before_player)
    after_hand = _extract_hand(after_state, after_player)
    hand_delta = _hand_quality(after_hand, after_player) - _hand_quality(before_hand, before_player)
    reward += max(-0.12, min(0.12, 0.20 * hand_delta))

    # Reward productive resource utilization pressure conversion.
    before_steel = _safe_float(before_player.get('steel', 0), 0.0)
    after_steel = _safe_float(after_player.get('steel', 0), 0.0)
    before_titanium = _safe_float(before_player.get('titanium', 0), 0.0)
    after_titanium = _safe_float(after_player.get('titanium', 0), 0.0)
    before_mc = _safe_float(before_player.get('megaCredits', 0), 0.0)
    after_mc = _safe_float(after_player.get('megaCredits', 0), 0.0)

    steel_spent = max(0.0, before_steel - after_steel)
    titanium_spent = max(0.0, before_titanium - after_titanium)
    mc_spent = max(0.0, before_mc - after_mc)
    utilization_reward = (0.015 * steel_spent) + (0.02 * titanium_spent) + (0.002 * min(mc_spent, 25.0))
    reward += max(0.0, min(0.14, utilization_reward))

    # Bias learning signal toward card-engine development.
    action_type = ''
    if isinstance(action_input, dict):
        action_type = str(action_input.get('type', '') or '').lower()
    if action_type == 'projectcard' or (action_type == 'card' and 'card' in action_input):
        reward += 0.07
    elif action_type == 'standardproject':
        reward -= 0.04
        affordable_cards = sum(1 for card in before_hand if _can_afford_card_now(card, before_player))
        if affordable_cards > 0:
            reward -= min(0.08, 0.02 * float(affordable_cards))

    # Light penalty for pass to discourage low-value inactivity.
    if action_type == 'pass':
        reward -= 0.02

    # Penalize routine sell-patents behavior, especially when playable cards were already affordable.
    before_waiting = before_state.get('waitingFor', {}) if isinstance(before_state, dict) else {}
    before_title = before_waiting.get('title', '')
    if isinstance(before_title, dict):
        before_title = before_title.get('message', '')
    before_title_l = str(before_title or '').lower()
    sold_cards = action_input.get('cards', []) if isinstance(action_input, dict) else []
    is_sell_patents_action = (
        action_type == 'card'
        and isinstance(sold_cards, list)
        and len(sold_cards) > 0
        and 'sell patent' in before_title_l
    )
    if is_sell_patents_action:
        reward -= min(0.10, 0.04 * float(len(sold_cards)))
        affordable_cards = sum(1 for card in before_hand if _can_afford_card_now(card, before_player))
        if affordable_cards > 0:
            reward -= min(0.10, 0.02 * float(affordable_cards))

    generation_raw = _safe_float(before_game.get('generation', 1), 1.0)
    generation_progress = max(0.0, min(generation_raw / 14.0, 1.0))
    endgame_pressure = max(0.0, min((generation_progress - 0.60) / 0.40, 1.0))

    # Milestone closing pressure: reward milestone claims more when they happen earlier.
    before_owned_milestones = _owned_milestone_count(before_game, before_player)
    after_owned_milestones = _owned_milestone_count(after_game, after_player)
    milestone_claim_delta = max(0, after_owned_milestones - before_owned_milestones)
    if milestone_claim_delta > 0:
        early_factor = max(0.10, 1.0 - generation_progress)
        milestone_claim_reward = 0.09 * float(milestone_claim_delta) * early_factor
        if before_mc >= 8.0:
            milestone_claim_reward += 0.04 * float(milestone_claim_delta) * early_factor
        reward += min(0.18, milestone_claim_reward)

    # Awards closing pressure: reinforce positive EV funding and discourage poor-value funding.
    before_owned_awards = _owned_funded_awards(before_game, before_player)
    after_owned_awards = _owned_funded_awards(after_game, after_player)
    if len(after_owned_awards) > len(before_owned_awards):
        before_keys = {_award_identity(award) for award in before_owned_awards}
        newly_funded = [
            award for award in after_owned_awards
            if _award_identity(award) not in before_keys
        ]
        if not newly_funded:
            newly_funded = after_owned_awards[len(before_owned_awards):]
        prior_funded_total = _funded_award_count(before_game)
        for idx, award in enumerate(newly_funded):
            scores = [row for row in (award.get('scores', []) or []) if isinstance(row, dict)]
            expected_vp = _award_expected_points_for_player(scores, after_player)
            estimated_cost = _estimate_award_funding_cost(prior_funded_total + idx)
            expected_net_vp = expected_vp - (estimated_cost / 5.0)
            timing_factor = 0.5 + (0.5 * generation_progress)
            if expected_net_vp > 0.0:
                reward += min(0.18, (0.04 + (0.035 * expected_net_vp)) * timing_factor)
            else:
                reward -= min(0.14, (0.03 + (0.04 * abs(expected_net_vp))) * timing_factor)

    # Final-generation card VP pressure: prefer affordable VP cards over low-ceiling alternatives.
    selected_card_name = _extract_selected_card_name(action_input)
    selected_card = _find_card_by_name(before_hand, selected_card_name)
    selected_card_vp = _card_nominal_vp(selected_card)
    affordable_vp_cards = [
        card for card in before_hand
        if _card_nominal_vp(card) > 0.0 and _can_afford_card_now(card, before_player)
    ]
    best_affordable_vp = max((_card_nominal_vp(card) for card in affordable_vp_cards), default=0.0)
    if endgame_pressure > 0.0:
        if selected_card_vp > 0.0:
            reward += min(0.16, endgame_pressure * (0.05 + (0.03 * min(selected_card_vp, 4.0))))
        elif action_type == 'standardproject' and best_affordable_vp > 0.0:
            reward -= min(0.16, endgame_pressure * (0.05 + (0.02 * min(best_affordable_vp, 5.0))))
        elif action_type == 'pass' and best_affordable_vp > 0.0:
            reward -= min(0.12, endgame_pressure * (0.03 + (0.02 * min(best_affordable_vp, 5.0))))

    # Apply Hard Mode scaling
    return float(max(-0.35, min(0.35, float(reward))) * STEP_REWARD_SCALE)
