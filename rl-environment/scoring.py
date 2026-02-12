"""
Shared scoring utilities used by both tournament fitness and policy training.
"""
from typing import Any, Dict, List, Optional


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

    ranking_points = {
        1: 100.0,
        2: 75.0,
        3: 50.0,
        4: 25.0,
    }.get(rank_int, 0.0)

    vp_bonus = vp * 0.5
    completion_bonus = 10.0 if bool(completed) else -50.0
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
        return float(value)
    except Exception:
        return float(default)


def _extract_player(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    player = state.get('thisPlayer', {})
    return player if isinstance(player, dict) else {}


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


def calculate_step_reward(
    before_state: Optional[Dict[str, Any]],
    after_state: Optional[Dict[str, Any]],
    action_input: Optional[Dict[str, Any]] = None,
) -> float:
    """Dense shaping reward used during self-play training."""
    if not before_state or not after_state:
        return 0.0

    before_player = _extract_player(before_state)
    after_player = _extract_player(after_state)
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

    return max(-0.35, min(0.35, float(reward)))
