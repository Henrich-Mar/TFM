"""
Shared scoring utilities used by both tournament fitness and policy training.
"""
from typing import Any, Dict, List, Optional, Tuple
import json
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
# Bonus per generation under 14 when winning; e.g. 3.0 means gen 11 wins get +9 vs gen 14
SELECTION_FAST_WIN_BONUS_PER_GEN = _env_float("SELECTION_FAST_WIN_BONUS_PER_GEN", 3.0)
SELECTION_FAST_WIN_MAX_BONUS = _env_float("SELECTION_FAST_WIN_MAX_BONUS", 15.0)


def calculate_selection_score(
    rank: Any,
    victory_points: Any,
    completed: Any,
    game_generation: Optional[int] = None,
) -> float:
    """Raw game score used for evolutionary selection.
    Favors agents who win in fewer generations when game_generation is provided.
    """
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
    base_score = ranking_points + vp_bonus + completion_bonus

    # Fast-win bonus: lower generation = faster terraforming = higher score
    fast_win_bonus = 0.0
    if game_generation is not None and bool(completed) and 1 <= int(game_generation) <= 14:
        gens_early = max(0, 14 - int(game_generation))
        if rank_int == 1:
            fast_win_bonus = min(
                float(SELECTION_FAST_WIN_MAX_BONUS),
                float(gens_early) * float(SELECTION_FAST_WIN_BONUS_PER_GEN),
            )
        elif rank_int == 2:
            fast_win_bonus = 0.5 * min(
                float(SELECTION_FAST_WIN_MAX_BONUS),
                float(gens_early) * float(SELECTION_FAST_WIN_BONUS_PER_GEN),
            )
        elif rank_int == 3:
            fast_win_bonus = 0.25 * min(
                float(SELECTION_FAST_WIN_MAX_BONUS),
                float(gens_early) * float(SELECTION_FAST_WIN_BONUS_PER_GEN),
            )

    return base_score + fast_win_bonus


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


_CARD_METADATA_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CARD_METADATA_LOWER_INDEX: Optional[Dict[str, str]] = None  # lower_name -> canonical key


def _load_card_metadata() -> Dict[str, Dict[str, Any]]:
    global _CARD_METADATA_CACHE, _CARD_METADATA_LOWER_INDEX
    if _CARD_METADATA_CACHE is not None:
        return _CARD_METADATA_CACHE

    module_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(module_dir, '..'))
    two_up = os.path.abspath(os.path.join(module_dir, '..', '..'))
    cwd = os.getcwd()
    candidate_paths = [
        os.getenv('TM_CARD_METADATA_PATH', ''),
        os.path.join(repo_root, 'card_metadata.json'),
        os.path.join(repo_root, 'terraforming-mars', 'card_metadata.json'),
        os.path.join(module_dir, 'card_metadata.json'),
        os.path.join(two_up, 'card_metadata.json'),
        os.path.join(cwd, 'card_metadata.json'),
        '/app/card_metadata.json',
    ]
    for path in candidate_paths:
        if not path:
            continue
        full = os.path.abspath(path)
        if not os.path.exists(full) or os.path.getsize(full) <= 0:
            continue
        try:
            with open(full, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                normalized: Dict[str, Dict[str, Any]] = {}
                lower_index: Dict[str, str] = {}
                for name, meta in data.items():
                    key = str(name or '').strip()
                    if key:
                        normalized[key] = dict(meta or {})
                        lower_index[key.lower()] = key
                _CARD_METADATA_CACHE = normalized
                _CARD_METADATA_LOWER_INDEX = lower_index
                return _CARD_METADATA_CACHE
        except Exception:
            continue
    _CARD_METADATA_CACHE = {}
    _CARD_METADATA_LOWER_INDEX = {}
    return _CARD_METADATA_CACHE


def _card_metadata_by_name(card_name: Any) -> Dict[str, Any]:
    """Look up card metadata by name. Uses case-insensitive fallback for game API variants."""
    name = str(card_name or '').strip()
    if not name:
        return {}
    metadata = _load_card_metadata()
    meta = metadata.get(name) or {}
    if meta:
        return meta
    # Case-insensitive fallback: game API may send different casing
    if _CARD_METADATA_LOWER_INDEX is not None:
        canonical = _CARD_METADATA_LOWER_INDEX.get(name.lower())
        if canonical:
            return metadata.get(canonical, {}) or {}
    return {}


def _normalize_resource_type(value: Any) -> str:
    raw = str(value or '').strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    if not raw:
        return ''
    aliases = {
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
    return aliases.get(raw, raw)


def _card_resource_type(card: Dict[str, Any]) -> str:
    direct = _normalize_resource_type(card.get('resourceType', ''))
    if direct:
        return direct
    meta = _card_metadata_by_name(card.get('name', ''))
    return _normalize_resource_type(meta.get('resourceType', ''))


def _resource_count(card: Dict[str, Any]) -> float:
    return max(0.0, _safe_float(card.get('resources', 0), 0.0))


def _vp_per_resource(card: Dict[str, Any]) -> float:
    meta = _card_metadata_by_name(card.get('name', ''))
    for raw in (card.get('vpPerResource', None), meta.get('vpPerResource', None)):
        val = _safe_float(raw, 0.0)
        if val > 0.0:
            return val

    dyn_card = card.get('dynamicVictoryPoints') if isinstance(card.get('dynamicVictoryPoints'), dict) else {}
    dyn_meta = meta.get('dynamicVictoryPoints') if isinstance(meta.get('dynamicVictoryPoints'), dict) else {}
    for dyn in (dyn_card, dyn_meta):
        points = _safe_float(dyn.get('points', 0), 0.0)
        target = _safe_float(dyn.get('target', 0), 0.0)
        if points > 0.0 and target > 0.0:
            return points / target
    return 0.0


def _resource_conversion_threshold(card: Dict[str, Any]) -> float:
    meta = _card_metadata_by_name(card.get('name', ''))
    return max(
        _safe_float(card.get('resourceConversionThreshold', 0), 0.0),
        _safe_float(meta.get('resourceConversionThreshold', 0), 0.0),
    )


def _resource_behavior(card: Dict[str, Any]) -> str:
    meta = _card_metadata_by_name(card.get('name', ''))
    behavior = str(card.get('resourceBehavior', meta.get('resourceBehavior', 'none')) or 'none').strip().lower()
    valid = {'vp_accumulation', 'conversion', 'stealing', 'adding', 'none'}

    adds_to_any = bool(card.get('resourceActionAddsToAnyCard', meta.get('resourceActionAddsToAnyCard', False)))
    targets_opponent = bool(card.get('resourceActionTargetsOpponent', meta.get('resourceActionTargetsOpponent', False)))
    removes_resources = bool(card.get('resourceActionRemovesResources', meta.get('resourceActionRemovesResources', False)))
    vp_ratio = _vp_per_resource(card)
    has_resource_type = bool(_card_resource_type(card))

    if targets_opponent and removes_resources:
        return 'stealing'
    if adds_to_any:
        return 'adding'
    if removes_resources:
        return 'conversion'
    if vp_ratio > 0.0 and has_resource_type:
        return 'vp_accumulation'
    return behavior if behavior in valid else 'none'


def _resource_management_adjustment(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    action_input: Dict[str, Any],
) -> float:
    before_player = _extract_player(before_state)
    after_player = _extract_player(after_state)
    before_tableau = [card for card in (before_player.get('tableau', []) or []) if isinstance(card, dict)]
    after_tableau = [card for card in (after_player.get('tableau', []) or []) if isinstance(card, dict)]
    if not before_tableau or not after_tableau:
        return 0.0

    before_by_name: Dict[str, Dict[str, Any]] = {}
    after_by_name: Dict[str, Dict[str, Any]] = {}
    for card in before_tableau:
        key = _normalize_token(card.get('name'))
        if key and key not in before_by_name:
            before_by_name[key] = card
    for card in after_tableau:
        key = _normalize_token(card.get('name'))
        if key and key not in after_by_name:
            after_by_name[key] = card

    adjustment = 0.0
    for key, before_card in before_by_name.items():
        after_card = after_by_name.get(key)
        if not isinstance(after_card, dict):
            continue
        before_resources = _resource_count(before_card)
        after_resources = _resource_count(after_card)
        if abs(after_resources - before_resources) <= 1e-9:
            continue

        behavior = _resource_behavior(after_card)
        vp_ratio = _vp_per_resource(after_card)
        threshold = _resource_conversion_threshold(after_card)
        delta = after_resources - before_resources

        if delta < 0.0:
            spent = abs(delta)
            if behavior == 'vp_accumulation' and vp_ratio > 0.0:
                vp_lost = spent * vp_ratio
                adjustment -= min(0.18, 0.03 + (0.035 * vp_lost))
            elif behavior == 'conversion':
                if threshold > 0.0 and before_resources >= threshold:
                    adjustment += min(0.11, 0.02 + (0.015 * spent))
                else:
                    adjustment -= min(0.06, 0.015 + (0.01 * spent))
        else:
            gained = delta
            if behavior == 'vp_accumulation' and vp_ratio > 0.0:
                adjustment += min(0.10, 0.01 * gained * max(vp_ratio, 0.25))
            elif behavior == 'stealing':
                adjustment += min(0.07, 0.01 * gained)

    # Mild penalty for over-hoarding conversion resources or passing while conversions are ready.
    conversion_ready_cards = 0
    for card in after_tableau:
        behavior = _resource_behavior(card)
        if behavior != 'conversion':
            continue
        resources = _resource_count(card)
        threshold = _resource_conversion_threshold(card)
        if threshold > 0.0 and resources >= threshold:
            conversion_ready_cards += 1
        if threshold > 0.0 and resources >= max(8.0, threshold * 3.0):
            adjustment -= 0.02

    action_type = str((action_input or {}).get('type', '') or '').lower() if isinstance(action_input, dict) else ''
    if action_type == 'pass' and conversion_ready_cards > 0:
        adjustment -= min(0.06, 0.02 * float(conversion_ready_cards))

    return float(max(-0.20, min(0.20, adjustment)))


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
    """Extract tags from card with proper list/dict handling and TitleCase normalization."""
    out: Dict[str, int] = {}
    
    if not isinstance(card, dict):
        return out
    
    tags = card.get('tags')
    
    # Handle list format (most common from card_metadata.json)
    if isinstance(tags, list):
        for item in tags:
            if item:
                # Normalize to TitleCase for consistency with hate-draft profiles
                tag_name = str(item).strip()
                if tag_name and tag_name[0].islower():
                    tag_name = tag_name.capitalize()
                out[tag_name] = 1
        if out:
            return out
    
    # Handle dict format (legacy)
    if isinstance(tags, dict):
        for key, value in tags.items():
            if value:
                tag_name = str(key).strip()
                if tag_name and tag_name[0].islower():
                    tag_name = tag_name.capitalize()
                out[tag_name] = 1
        if out:
            return out
    
    # Fallback: try metadata lookup if card has name
    card_name = str(card.get('name', '') or '').strip()
    if card_name:
        metadata = _card_metadata_by_name(card_name)
        meta_tags = metadata.get('tags', [])
        if isinstance(meta_tags, list):
            for item in meta_tags:
                if item:
                    tag_name = str(item).strip()
                    if tag_name and tag_name[0].islower():
                        tag_name = tag_name.capitalize()
                    out[tag_name] = 1
    
    return out


def _build_tag_profile_from_player_tags(player: Dict[str, Any]) -> Dict[str, int]:
    """Build tag profile from game's player.tags dict (e.g. {'building': 2, 'space': 0, 'jovian': 1})."""
    profile: Dict[str, int] = {}
    tags_raw = player.get('tags') if isinstance(player, dict) else None
    if not isinstance(tags_raw, dict):
        return profile
    for key, value in tags_raw.items():
        if not key or value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        tag_name = str(key).strip()
        if tag_name and tag_name[0].islower():
            tag_name = tag_name.capitalize()
        profile[tag_name] = profile.get(tag_name, 0) + count
    return profile


def _build_tag_profile_from_cards(cards: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate tag counts across cards for hate-draft overlap computation."""
    profile: Dict[str, int] = {}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        tags = _extract_tags(card)
        for tag_name, count in tags.items():
            if count:
                # Ensure TitleCase consistency
                if tag_name and tag_name[0].islower():
                    tag_name = tag_name.capitalize()
                profile[tag_name] = profile.get(tag_name, 0) + count
    return profile


def _merge_tag_profiles(primary: Dict[str, int], fallback: Dict[str, int]) -> Dict[str, int]:
    """Merge two profiles; primary wins for each tag, fallback fills gaps."""
    out: Dict[str, int] = dict(primary)
    for tag_name, count in (fallback or {}).items():
        if count and tag_name not in out:
            out[tag_name] = count
    return out


def _tag_overlap_for_card(card: Dict[str, Any], profile: Dict[str, int]) -> float:
    """Sum profile counts for tags present on the card. Normalize by max 8."""
    tags = _extract_tags(card)
    overlap = sum(float(profile.get(tag_name, 0)) for tag_name in tags)
    return min(overlap / 8.0, 1.0)


def _hate_draft_adjustment(
    before_state: Dict[str, Any],
    action_input: Dict[str, Any],
) -> Tuple[float, bool]:
    """
    Bonus when the agent drafts/keeps cards that hurt opponents (high opp overlap, low own overlap).
    Returns (adjustment, applied).
    """
    if not isinstance(action_input, dict):
        return 0.0, False
    action_type = str(action_input.get('type', '') or '').lower()
    before_player = _extract_player(before_state)
    own_color = _normalize_token(before_player.get('color'))
    own_id = str(before_player.get('id', '') or '')
    own_name = _normalize_token(before_player.get('name'))
    players = before_state.get('players', []) or []
    if not isinstance(players, list) or not players:
        game = before_state.get('game') or {}
        players = (game.get('players', []) or []) if isinstance(game, dict) else []
    if not isinstance(players, list):
        return 0.0, False

    opponent_tableau: List[Dict[str, Any]] = []
    opponent_tags_aggregate: Dict[str, int] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        if (own_id and str(p.get('id', '')) == own_id) or (own_color and _normalize_token(p.get('color')) == own_color) or (own_name and _normalize_token(p.get('name')) == own_name):
            continue
        opponent_tableau.extend([c for c in (p.get('tableau', []) or []) if isinstance(c, dict)])
        ptags = _build_tag_profile_from_player_tags(p)
        for k, v in ptags.items():
            opponent_tags_aggregate[k] = opponent_tags_aggregate.get(k, 0) + v
    own_tableau = [c for c in (before_player.get('tableau', []) or []) if isinstance(c, dict)]
    own_profile_from_tags = _build_tag_profile_from_player_tags(before_player)
    own_profile_from_tableau = _build_tag_profile_from_cards(own_tableau)
    own_profile = _merge_tag_profiles(own_profile_from_tags, own_profile_from_tableau)
    opponent_profile_from_tableau = _build_tag_profile_from_cards(opponent_tableau)
    opponent_profile = _merge_tag_profiles(opponent_tags_aggregate, opponent_profile_from_tableau)

    selected_cards: List[Dict[str, Any]] = []
    if action_type == 'initialcards':
        for resp in (action_input.get('responses', []) or []):
            if not isinstance(resp, dict):
                continue
            cards_raw = resp.get('cards', [])
            if isinstance(cards_raw, list):
                for c in cards_raw:
                    if isinstance(c, dict):
                        selected_cards.append(c)
                    elif isinstance(c, str):
                        selected_cards.append({'name': c})
            elif isinstance(cards_raw, str):
                selected_cards.append({'name': cards_raw})
    elif action_type == 'card':
        card_names = action_input.get('cards', []) or []
        if not isinstance(card_names, list):
            card_names = [card_names] if card_names else []
        waiting = (before_state.get('waitingFor', {}) or {})
        if not isinstance(waiting, dict):
            waiting = {}
        wait_type = _normalize_token(waiting.get('type'))
        prompt_cards = waiting.get('cards', []) or []
        title = _title_text(waiting.get('title', '')).lower()
        is_draft = (
            wait_type in ('card', 'selectcard', 'draft', 'draftcards')
            or 'draft' in title
            or ('keep' in title and 'card' in title)
        )
        if is_draft and prompt_cards and card_names:
            for name in card_names:
                n = str(name).strip()
                for pc in prompt_cards:
                    if isinstance(pc, dict) and str(pc.get('name', '')).strip() == n:
                        selected_cards.append(pc)
                        break
                else:
                    selected_cards.append({'name': n})

    if not selected_cards:
        return 0.0, False

    hate_signals: List[float] = []
    for card in selected_cards:
        opp_overlap = _tag_overlap_for_card(card, opponent_profile)
        own_overlap = _tag_overlap_for_card(card, own_profile)
        hate_signals.append(max(0.0, opp_overlap - own_overlap))

    avg_hate = sum(hate_signals) / len(hate_signals) if hate_signals else 0.0
    # Lower threshold AND add gradual bonus even below threshold
    if avg_hate < 0.12:  # 0.2 → 0.12 (much lower)
        # Still give small bonus for any hate signal
        bonus = min(0.04, 0.02 * avg_hate + 0.01)
        applied = avg_hate > 0.05  # Only flag as applied if there's some signal
    else:
        bonus = min(0.25, 0.10 * avg_hate + 0.05)  # Much stronger bonus
        applied = True

    return float(bonus), applied


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
    requested = str(key or '').strip()
    # TR is tracked as a live top-level field in player state; VP breakdown can lag.
    if requested in ('terraforming', 'terraformRating'):
        top_level_tr = _safe_float(player.get('terraformRating', 0), 0.0)
        if abs(top_level_tr) > 1e-9:
            return top_level_tr

    breakdown = player.get('victoryPointsBreakdown', {})
    if not isinstance(breakdown, dict):
        return 0.0
    # Support both legacy and current Terraforming Mars VP breakdown keys.
    aliases = {
        'terraforming': ('terraforming', 'terraformRating'),
        'terraformRating': ('terraformRating', 'terraforming'),
        'cards': ('cards', 'victoryPoints'),
        'victoryPoints': ('victoryPoints', 'cards'),
    }
    for candidate in aliases.get(requested, (requested,)):
        if candidate in breakdown:
            return _safe_float(breakdown.get(candidate, 0), 0.0)
    return 0.0


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


def _get_claimable_milestone_names(game_state: Dict[str, Any], player: Dict[str, Any]) -> List[str]:
    """
    Return names of milestones that are currently reachable (progress >= 1.0)
    by *this* player but not yet claimed by anyone.

    A milestone is considered claimable when the player's own score entry in the
    milestone's scores list is >= the published minimum threshold.  Because the
    server does not always expose the hard threshold we rely on the convention
    that progress >= 1.0 in the normalised representation (own_score /
    max(threshold, max_score) >= 1.0) means the player already qualifies.
    """
    claimable: List[str] = []
    own_color = _normalize_token(player.get('color'))
    own_name = _normalize_token(player.get('name'))
    if not own_color and not own_name:
        return claimable

    milestones = game_state.get('milestones', []) or []
    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        # Skip already-claimed milestones
        if milestone.get('playerName') or milestone.get('playerColor'):
            continue
        scores = [row for row in (milestone.get('scores', []) or []) if isinstance(row, dict)]
        if not scores:
            continue

        own_score = 0.0
        max_score = 0.0
        for row in scores:
            try:
                score = float(row.get('playerScore', 0) or 0)
            except Exception:
                score = 0.0
            max_score = max(max_score, score)
            row_color = _normalize_token(row.get('playerColor'))
            row_name = _normalize_token(row.get('playerName'))
            if (own_color and row_color == own_color) or (own_name and row_name == own_name):
                own_score = score

        # Normalised progress >= 1.0 means player already meets the threshold
        threshold = 3.0
        denominator = max(threshold, max_score, 1.0)
        if (own_score / denominator) >= 1.0:
            name = str(milestone.get('name', '') or '').strip()
            if name:
                claimable.append(name)

    return claimable


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


def _award_projection_for_player(scores: List[Dict[str, Any]], player: Dict[str, Any]) -> Tuple[float, float]:
    """
    Return (projected_points, confidence) for the player's current award standing.

    projected_points is 5 for current first place, 2 for current second place, 0 otherwise.
    confidence is a conservative [0.0, 1.0] estimate based on score magnitude, lead gap,
    and ties (ties reduce confidence because standings are volatile).
    """
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
                # Server payloads can use either "score" or "playerScore".
                "score": _safe_float(row.get('score', row.get('playerScore', 0)), 0.0),
            }
        )
    if not rows:
        return 0.0, 0.0

    rows.sort(key=lambda item: item["score"], reverse=True)
    top_score = rows[0]["score"]
    top_rows = [row for row in rows if row["score"] == top_score]

    def _is_own(row: Dict[str, Any]) -> bool:
        if own_color and row["color"] and own_color == row["color"]:
            return True
        if own_name and row["name"] and own_name == row["name"]:
            return True
        return False

    if top_score <= 0.0:
        return 0.0, 0.0

    remaining_rows = [row for row in rows if row["score"] < top_score]
    second_score = remaining_rows[0]["score"] if remaining_rows else 0.0
    second_rows = [row for row in remaining_rows if row["score"] == second_score]

    projected_points = 0.0
    rank_tied = False
    gap_to_fall = 0.0

    own_in_top = any(_is_own(row) for row in top_rows)
    own_in_second = bool(second_rows) and any(_is_own(row) for row in second_rows)

    if own_in_top:
        projected_points = 5.0
        rank_tied = len(top_rows) > 1
        gap_to_fall = 0.0 if rank_tied else max(0.0, top_score - second_score)
    elif second_score > 0.0 and own_in_second:
        projected_points = 2.0
        rank_tied = len(second_rows) > 1
        third_rows = [row for row in rows if row["score"] < second_score]
        third_score = third_rows[0]["score"] if third_rows else 0.0
        gap_to_fall = 0.0 if rank_tied else max(0.0, second_score - third_score)
    else:
        return 0.0, 0.0

    # Confidence increases with absolute score progression and clear lead margin.
    score_signal = max(0.0, min(top_score / 10.0, 1.0))
    gap_signal = max(0.0, min(gap_to_fall / max(2.0, top_score), 1.0))
    tie_multiplier = 0.55 if rank_tied else 1.0
    confidence = (0.18 + (0.42 * score_signal) + (0.40 * gap_signal)) * tie_multiplier
    confidence = max(0.08, min(confidence, 1.0))
    return float(projected_points), float(confidence)


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

_CITY_TILE_TYPES = {2, 3, 20, 37, 43}
_GREENERY_TILE_TYPES = {0, 36}
_OCEAN_TILE_TYPES = {1, 20, 21, 22, 36, 43}

def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None

def _title_text(value: Any) -> str:
    if isinstance(value, dict):
        message = str(value.get('message', '') or '')
        data = value.get('data', [])
        if not message:
            return ''
        if isinstance(data, list):
            rendered = message
            for idx, token in enumerate(data):
                if isinstance(token, dict):
                    replacement = str(token.get('value', '') or '')
                else:
                    replacement = str(token or '')
                rendered = rendered.replace(f"${{{idx}}}", replacement)
            return rendered
        return message
    if isinstance(value, str):
        return value
    return ''

def _space_id(space: Any) -> str:
    if isinstance(space, dict):
        return str(space.get('id') or space.get('spaceId') or '').strip()
    return str(space or '').strip()

def _space_owner_color(space: Dict[str, Any]) -> str:
    return str(space.get('color', '') or '').strip().lower()

def _space_type_lower(value: Any) -> str:
    return str(value or '').strip().lower()

def _space_tile_flags(space: Dict[str, Any]) -> tuple[bool, bool, bool]:
    tile_raw = space.get('tileType')
    tile_num = _optional_int(tile_raw)
    tile_name = str(tile_raw or '').strip().lower().replace('-', '_').replace(' ', '_')

    is_city = False
    is_greenery = False
    is_ocean = False

    if tile_num is not None:
        is_city = tile_num in _CITY_TILE_TYPES
        is_greenery = tile_num in _GREENERY_TILE_TYPES
        is_ocean = tile_num in _OCEAN_TILE_TYPES

    if tile_name:
        if 'city' in tile_name or tile_name in ('capital', 'new_holland'):
            is_city = True
        if 'greenery' in tile_name or 'wetland' in tile_name:
            is_greenery = True
        if 'ocean' in tile_name or 'wetland' in tile_name:
            is_ocean = True

    return is_city, is_greenery, is_ocean

def _build_board_adjacency_index(spaces: List[Dict[str, Any]]) -> Dict[str, set]:
    coord_to_id: Dict[tuple[int, int], str] = {}
    mars_spaces: List[tuple[str, int, int]] = []
    max_y = 0

    for space in spaces:
        if not isinstance(space, dict):
            continue
        sid = _space_id(space)
        if not sid:
            continue
        if _space_type_lower(space.get('spaceType')) == 'colony':
            continue
        x = _optional_int(space.get('x'))
        y = _optional_int(space.get('y'))
        if x is None or y is None or x < 0 or y < 0:
            continue
        coord_to_id[(x, y)] = sid
        mars_spaces.append((sid, x, y))
        if y > max_y:
            max_y = y

    if not mars_spaces:
        return {}

    middle_row = max_y / 2.0
    adjacency: Dict[str, set] = {}
    for sid, x, y in mars_spaces:
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

        neighbor_coords = [
            tuple(top_left_space),
            tuple(top_right_space),
            right_space,
            tuple(bottom_right_space),
            tuple(bottom_left_space),
            left_space,
        ]
        neighbors = set()
        for coord in neighbor_coords:
            neighbor_id = coord_to_id.get(coord)
            if neighbor_id and neighbor_id != sid:
                neighbors.add(neighbor_id)
        adjacency[sid] = neighbors

    return adjacency

def _extract_space_action_payload(action_input: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(action_input, dict):
        return None
    stack: List[Dict[str, Any]] = [action_input]
    while stack:
        payload = stack.pop()
        if not isinstance(payload, dict):
            continue
        if _normalize_token(payload.get('type')) == 'space':
            if payload.get('spaceId') is not None:
                return payload
        nested_response = payload.get('response')
        if isinstance(nested_response, dict):
            stack.append(nested_response)
        nested_responses = payload.get('responses')
        if isinstance(nested_responses, list):
            for item in nested_responses:
                if isinstance(item, dict):
                    stack.append(item)
    return None

def _infer_space_prompt_tile_intent(waiting_for: Dict[str, Any]) -> str:
    title = _title_text(waiting_for.get('title', '')).lower()
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

def _space_waiting_prompt(before_waiting: Dict[str, Any], action_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    waiting_type = _normalize_token(before_waiting.get('type'))
    if waiting_type in ('space', 'selectspace'):
        return before_waiting
    if waiting_type == 'or':
        options = before_waiting.get('options', []) or []
        selected_idx = _optional_int(action_input.get('index'))
        if selected_idx is not None and 0 <= selected_idx < len(options):
            selected = options[selected_idx]
            if isinstance(selected, dict) and _normalize_token(selected.get('type')) in ('space', 'selectspace'):
                return selected
        for option in options:
            if isinstance(option, dict) and _normalize_token(option.get('type')) in ('space', 'selectspace'):
                return option
    return None

def _waiting_space_ids(waiting_prompt: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(waiting_prompt, dict):
        return []
    raw_spaces = waiting_prompt.get('availableSpaces', waiting_prompt.get('spaces', []))
    if not isinstance(raw_spaces, list):
        return []
    out: List[str] = []
    for space in raw_spaces:
        sid = _space_id(space)
        if sid:
            out.append(sid)
    return out

def _newly_placed_space(before_game: Dict[str, Any], after_game: Dict[str, Any], own_color: str) -> Optional[Dict[str, Any]]:
    before_spaces = [space for space in (before_game.get('spaces', []) or []) if isinstance(space, dict)]
    after_spaces = [space for space in (after_game.get('spaces', []) or []) if isinstance(space, dict)]
    before_by_id = {_space_id(space): space for space in before_spaces if _space_id(space)}
    candidates: List[Dict[str, Any]] = []

    for space in after_spaces:
        sid = _space_id(space)
        if not sid:
            continue
        before_space = before_by_id.get(sid, {})
        before_has_tile = before_space.get('tileType') is not None
        after_has_tile = space.get('tileType') is not None
        if not before_has_tile and after_has_tile:
            candidates.append(space)

    if not candidates:
        return None
    for space in candidates:
        if _space_owner_color(space) == own_color:
            return space
    return candidates[0]

def _space_placement_adjustment(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    action_input: Dict[str, Any],
) -> float:
    space_action = _extract_space_action_payload(action_input)
    if not isinstance(space_action, dict):
        return 0.0

    before_player = _extract_player(before_state)
    own_color = _normalize_token(before_player.get('color'))
    if not own_color:
        return 0.0

    before_game = _extract_game(before_state)
    after_game = _extract_game(after_state)
    if not before_game or not after_game:
        return 0.0

    placed_space = _newly_placed_space(before_game, after_game, own_color)
    if not isinstance(placed_space, dict):
        return 0.0
    placed_space_id = _space_id(placed_space)
    if not placed_space_id:
        return 0.0

    board_spaces = [space for space in (before_game.get('spaces', []) or []) if isinstance(space, dict)]
    adjacency = _build_board_adjacency_index(board_spaces)
    neighbors = adjacency.get(placed_space_id, set())

    enemy_city_ids = set()
    own_city_ids = set()
    for space in board_spaces:
        sid = _space_id(space)
        if not sid:
            continue
        is_city, _, _ = _space_tile_flags(space)
        owner = _space_owner_color(space)
        if not is_city or not owner:
            continue
        if owner == own_color:
            own_city_ids.add(sid)
        else:
            enemy_city_ids.add(sid)

    enemy_city_adj = sum(1 for sid in neighbors if sid in enemy_city_ids)
    own_city_adj = sum(1 for sid in neighbors if sid in own_city_ids)

    before_waiting = before_state.get('waitingFor', {}) if isinstance(before_state, dict) else {}
    waiting_prompt = _space_waiting_prompt(before_waiting, action_input)
    intent = _infer_space_prompt_tile_intent(waiting_prompt or before_waiting or {})

    adjustment = 0.0
    if intent == 'greenery':
        waiting_ids = _waiting_space_ids(waiting_prompt)
        risky_total = 0
        if waiting_ids:
            for sid in waiting_ids:
                option_neighbors = adjacency.get(sid, set())
                if any(nid in enemy_city_ids for nid in option_neighbors):
                    risky_total += 1
        all_options_risky = bool(waiting_ids) and risky_total >= len(waiting_ids)

        if enemy_city_adj > 0:
            base_penalty = min(0.16, 0.08 * float(enemy_city_adj))
            adjustment -= base_penalty * (0.25 if all_options_risky else 1.0)
        elif own_city_adj > 0:
            adjustment += min(0.06, 0.03 * float(own_city_adj))
    elif intent == 'special' and enemy_city_adj > 0:
        # Strategic blocking around enemy cities is a valid positive tactic.
        adjustment += min(0.05, 0.02 * float(enemy_city_adj))
    elif intent == 'city' and enemy_city_adj > 0:
        # Placing a city adjacent to enemy cities blocks their greenery expansion.
        adjustment += min(0.05, 0.02 * float(enemy_city_adj))

    return float(adjustment)


REWARD_NON_ACTION = -0.01  # Small penalty for doing nothing

# Scoring Mode
SCORING_MODE = os.getenv("SCORING_MODE", "NORMAL").upper()
IS_HARD_MODE = SCORING_MODE == "HARD"

# Reward Scaling
STEP_REWARD_SCALE = 0.3 if IS_HARD_MODE else 1.0
TERMINAL_REWARD_SCALE = 2.0 if IS_HARD_MODE else 1.0

def calculate_step_reward_decomposition(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    action_input: Dict[str, Any],
) -> Dict[str, float]:
    """
    Return decomposed dense reward components used for value-shaping diagnostics.
    """
    if not before_state or not after_state:
        return {
            "tr_component": 0.0,
            "cards_vp_component": 0.0,
            "city_greenery_component": 0.0,
            "milestones_awards_component": 0.0,
            "other_component": 0.0,
            "raw_total": 0.0,
            "scaled_total": 0.0,
            "hate_draft_bonus_applied": False,
            "sniping_milestone_applied": False,
            "sniping_award_applied": False,
        }

    before_player = _extract_player(before_state)
    after_player = _extract_player(after_state)
    before_game = _extract_game(before_state)
    after_game = _extract_game(after_state)
    if not before_player or not after_player:
        return {
            "tr_component": 0.0,
            "cards_vp_component": 0.0,
            "city_greenery_component": 0.0,
            "milestones_awards_component": 0.0,
            "other_component": 0.0,
            "raw_total": 0.0,
            "scaled_total": 0.0,
            "hate_draft_bonus_applied": False,
            "sniping_milestone_applied": False,
            "sniping_award_applied": False,
        }

    tr_component = 0.0
    cards_vp_component = 0.0
    city_greenery_component = 0.0
    milestones_awards_component = 0.0
    other_component = 0.0
    hate_draft_bonus_applied = False
    sniping_milestone_applied = False
    sniping_award_applied = False

    # Reward tangible engine growth.
    before_tableau = before_player.get('tableau', []) or []
    after_tableau = after_player.get('tableau', []) or []
    tableau_delta = len(after_tableau) - len(before_tableau)
    other_component += max(-0.08, min(0.18, 0.05 * float(tableau_delta)))

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
    other_component += max(-0.10, min(0.14, 0.02 * production_delta))

    # Decomposed VP-driven shaping buckets.
    terraforming_delta = _vp_component(after_player, 'terraforming') - _vp_component(before_player, 'terraforming')
    tr_component += max(-0.15, min(0.30, 0.15 * terraforming_delta))      # 0.045 → 0.15 (3.3x)

    milestone_delta = _vp_component(after_player, 'milestones') - _vp_component(before_player, 'milestones')
    milestones_awards_component += max(-0.12, min(0.35, 0.08 * milestone_delta))    # 0.035 → 0.08 (2.3x)

    award_delta = _vp_component(after_player, 'awards') - _vp_component(before_player, 'awards')
    milestones_awards_component += max(-0.15, min(0.40, 0.06 * award_delta))        # 0.030 → 0.06 (2x)

    city_combo_delta = _vp_component(after_player, 'city') - _vp_component(before_player, 'city')
    greenery_delta = _vp_component(after_player, 'greenery') - _vp_component(before_player, 'greenery')
    combo_delta = city_combo_delta + (0.25 * greenery_delta)
    city_greenery_component += max(-0.20, min(0.30, 0.12 * combo_delta))   # 0.05 → 0.12 (2.4x)

    cards_vp_delta = _vp_component(after_player, 'cards') - _vp_component(before_player, 'cards')
    cards_vp_component += max(-0.20, min(0.35, 0.12 * cards_vp_delta))     # 0.045 → 0.12 (2.7x)
    # Reward card quality improvements in hand.
    before_hand = _extract_hand(before_state, before_player)
    after_hand = _extract_hand(after_state, after_player)
    hand_delta = _hand_quality(after_hand, after_player) - _hand_quality(before_hand, before_player)
    other_component += max(-0.12, min(0.12, 0.20 * hand_delta))

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
    other_component += max(0.0, min(0.14, utilization_reward))

    action_type = ''
    if isinstance(action_input, dict):
        action_type = str(action_input.get('type', '') or '').lower()
    if action_type == 'projectcard' or (action_type == 'card' and 'card' in action_input):
        cards_vp_component += 0.07
    elif action_type == 'standardproject':
        other_component -= 0.04
        affordable_cards = sum(1 for card in before_hand if _can_afford_card_now(card, before_player))
        if affordable_cards > 0:
            other_component -= min(0.08, 0.02 * float(affordable_cards))

    # Light penalty for pass to discourage low-value inactivity.
    if action_type == 'pass':
        other_component -= 0.02

    # Tile-placement tactical shaping:
    # - discourage greenery placements adjacent to enemy cities (unless forced)
    # - allow/encourage special-tile blocking around enemy cities
    # - reward city placement adjacent to enemy cities (blocking)
    other_component += _space_placement_adjustment(before_state, after_state, action_input)

    # Hate-draft bonus: reward keeping/drafting cards that hurt opponents (high opp tag overlap, low own).
    hate_draft_adj, hate_draft_bonus_applied = _hate_draft_adjustment(before_state, action_input)
    other_component += hate_draft_adj

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
        sold_count = float(len(sold_cards))
        before_affordable = sum(1 for card in before_hand if _can_afford_card_now(card, before_player))
        after_affordable = sum(1 for card in after_hand if _can_afford_card_now(card, after_player))
        affordable_delta = float(after_affordable - before_affordable)

        # Base penalty: discourage routine liquidity dumps.
        other_component -= min(0.18, 0.06 * sold_count)

        # Additional pressure when selling does not unlock immediate playable cards.
        if affordable_delta <= 0.0:
            other_component -= min(0.12, 0.03 * sold_count)
        else:
            other_component += min(0.04, 0.015 * affordable_delta)

        # Extra penalty for large single-action dumps.
        if sold_count > 3.0:
            other_component -= min(0.10, 0.02 * (sold_count - 3.0))

        # Selling persistent engine cards is usually long-term value destruction.
        sold_names = {
            str(name or "").strip()
            for name in sold_cards
            if isinstance(name, str) and str(name or "").strip()
        }
        persistent_sold = 0
        for card in before_hand:
            if not isinstance(card, dict):
                continue
            name = str(card.get("name", "") or "").strip()
            if not name or name not in sold_names:
                continue
            meta = _card_metadata_by_name(name)
            has_action = bool(card.get("hasAction", meta.get("hasAction", False)))
            vp = _safe_float(meta.get("victoryPoints", card.get("victoryPoints", 0)), 0.0)
            desc = str(card.get("description", meta.get("description", "")) or "").lower()
            is_persistent = bool(
                has_action
                or vp > 0.0
                or ("production" in desc)
                or ("ongoing" in desc)
                or ("effect" in desc)
            )
            if is_persistent:
                persistent_sold += 1
        if persistent_sold > 0:
            other_component -= min(0.20, 0.07 * float(persistent_sold))

    generation_raw = _safe_float(before_game.get('generation', 1), 1.0)
    generation_progress = max(0.0, min(generation_raw / 14.0, 1.0))
    endgame_pressure = max(0.0, min((generation_progress - 0.60) / 0.40, 1.0))

    # Milestone closing pressure: reward milestone claims more when they happen earlier.
    # Base reward raised from 0.09 → 0.20 so a claim clearly outweighs a card-play bonus.
    before_owned_milestones = _owned_milestone_count(before_game, before_player)
    after_owned_milestones = _owned_milestone_count(after_game, after_player)
    milestone_claim_delta = max(0, after_owned_milestones - before_owned_milestones)
    if milestone_claim_delta > 0:
        early_factor = max(0.10, 1.0 - generation_progress)
        milestone_claim_reward = 0.20 * float(milestone_claim_delta) * early_factor
        if before_mc >= 8.0:
            # Small extra bonus for claiming with comfortable cash (shows timing awareness)
            milestone_claim_reward += 0.04 * float(milestone_claim_delta) * early_factor
        milestones_awards_component += min(0.30, milestone_claim_reward)
        # Sniping bonus: reward claiming in late game (denying opponents).
        if generation_progress > 0.55:
            milestones_awards_component += min(0.12, 0.06 * float(milestone_claim_delta))
            sniping_milestone_applied = True

    # Milestone Regret Penalty: if this player could have claimed a milestone
    # before this action (progress >= 1.0, unclaimed) but an opponent claimed it
    # in the after_state, apply a regret penalty to train the agent to prioritise
    # such opportunities.
    claimable_before = set(_get_claimable_milestone_names(before_game, before_player))
    if claimable_before:
        after_milestones = after_game.get('milestones', []) or []
        after_player_color = _normalize_token(after_player.get('color'))
        after_player_name = _normalize_token(after_player.get('name'))
        for milestone in after_milestones:
            if not isinstance(milestone, dict):
                continue
            name = str(milestone.get('name', '') or '').strip()
            if name not in claimable_before:
                continue
            # Check if someone else claimed it now
            claimer_color = _normalize_token(milestone.get('playerColor'))
            claimer_name = _normalize_token(milestone.get('playerName'))
            if not claimer_color and not claimer_name:
                continue  # Still unclaimed
            own_claimed = (
                (after_player_color and claimer_color == after_player_color)
                or (after_player_name and claimer_name == after_player_name)
            )
            if not own_claimed:
                # An opponent stole a milestone we could have claimed → regret penalty
                milestones_awards_component -= 0.20

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
            projected_points, projection_confidence = _award_projection_for_player(scores, after_player)
            expected_vp = projected_points * projection_confidence
            estimated_cost = _estimate_award_funding_cost(prior_funded_total + idx)
            estimated_cost_vp = estimated_cost / 5.0
            # Early funding has high uncertainty and high opportunity cost.
            # Apply a generation-weighted commitment tax so gen-1/2 funding requires
            # genuinely strong projected standings before receiving positive shaping.
            commitment_window = max(0.0, 1.0 - (generation_progress / 0.75))
            commitment_tax_vp = estimated_cost_vp * commitment_window * 0.65
            expected_net_vp = expected_vp - estimated_cost_vp - commitment_tax_vp
            positive_timing_factor = 0.20 + (0.80 * generation_progress)
            negative_timing_factor = 1.15 - (0.55 * generation_progress)
            if expected_net_vp > 0.0:
                milestones_awards_component += min(0.14, (0.015 + (0.028 * expected_net_vp)) * positive_timing_factor)
                # Sniping bonus: late-game award funding when we're ahead (positive EV).
                if generation_progress > 0.55 and projected_points >= 3.0 and projection_confidence >= 0.55:
                    milestones_awards_component += min(0.12, 0.06 * expected_net_vp) # 0.04→0.10, 0.02→0.05
                    sniping_award_applied = True
            else:
                milestones_awards_component -= min(0.24, (0.04 + (0.045 * abs(expected_net_vp))) * negative_timing_factor)

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
            cards_vp_component += min(0.16, endgame_pressure * (0.05 + (0.03 * min(selected_card_vp, 4.0))))
        elif action_type == 'standardproject' and best_affordable_vp > 0.0:
            cards_vp_component -= min(0.16, endgame_pressure * (0.05 + (0.02 * min(best_affordable_vp, 5.0))))
        elif action_type == 'pass' and best_affordable_vp > 0.0:
            cards_vp_component -= min(0.12, endgame_pressure * (0.03 + (0.02 * min(best_affordable_vp, 5.0))))

    # Resource-pattern shaping: penalize burning VP-hoard resources and reward efficient conversions.
    cards_vp_component += _resource_management_adjustment(before_state, after_state, action_input)

    raw_total = (
        tr_component
        + cards_vp_component
        + city_greenery_component
        + milestones_awards_component
        + other_component
    )
    clamped_total = float(max(-0.35, min(0.35, float(raw_total))))
    scaled_total = float(clamped_total * STEP_REWARD_SCALE)
    return {
        "tr_component": float(tr_component),
        "cards_vp_component": float(cards_vp_component),
        "city_greenery_component": float(city_greenery_component),
        "milestones_awards_component": float(milestones_awards_component),
        "other_component": float(other_component),
        "raw_total": float(raw_total),
        "clamped_total": float(clamped_total),
        "step_reward_scale": float(STEP_REWARD_SCALE),
        "scaled_total": float(scaled_total),
        "hate_draft_bonus_applied": bool(hate_draft_bonus_applied),
        "sniping_milestone_applied": bool(sniping_milestone_applied),
        "sniping_award_applied": bool(sniping_award_applied),
    }


def calculate_step_reward(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    action_input: Dict[str, Any],
) -> float:
    """
    Backward-compatible scalar step reward wrapper.
    """
    return float(calculate_step_reward_decomposition(before_state, after_state, action_input).get("scaled_total", 0.0))
