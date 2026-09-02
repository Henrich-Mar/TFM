"""
Action Decoder - Converts neural network output to game actions
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import random
import os
import json
import itertools

from .rust_backend import get_rust_module
from .planner_common import PlannerConfig, token_from_features

logger = logging.getLogger(__name__)

_CARD_SELECTION_MASK_BASE = 520
_CARD_SELECTION_MASK_LIMIT = 80
_CARD_SELECTION_CANDIDATE_LIMIT = 12
_STARTUP_PLAN_BASE = 850
_STARTUP_PLAN_LIMIT = 32
_PAYMENT_ACTION_BASE = 400
_PAYMENT_ACTION_VARIANTS = 8
_PAYMENT_ALL_KEYS = [
    'megaCredits', 'steel', 'titanium', 'heat', 'plants',
    'microbes', 'floaters', 'lunaArchivesScience', 'spireScience',
    'seeds', 'auroraiData', 'graphene', 'kuiperAsteroids'
]
_PAYMENT_DEFAULT_VALUES = {
    'megaCredits': 1,
    'steel': 2,
    'titanium': 3,
    'heat': 1,
    'plants': 3,
    'microbes': 2,
    'floaters': 3,
    'lunaArchivesScience': 1,
    'spireScience': 2,
    'seeds': 5,
    'auroraiData': 3,
    'graphene': 4,
    'kuiperAsteroids': 1,
}


def _canonical_payment_resource(resource: Any) -> Optional[str]:
    token = ''.join(ch for ch in str(resource or '') if ch.isalnum()).lower()
    if not token:
        return None
    aliases = {
        'megacredits': 'megaCredits',
        'megacredit': 'megaCredits',
        'mc': 'megaCredits',
        'steel': 'steel',
        'titanium': 'titanium',
        'heat': 'heat',
        'plants': 'plants',
        'plant': 'plants',
        'microbes': 'microbes',
        'microbe': 'microbes',
        'floaters': 'floaters',
        'floater': 'floaters',
        'lunaarchivesscience': 'lunaArchivesScience',
        'spirescience': 'spireScience',
        'seeds': 'seeds',
        'auroraidata': 'auroraiData',
        'graphene': 'graphene',
        'kuiperasteroids': 'kuiperAsteroids',
    }
    if token in aliases:
        return aliases[token]

    for key in _PAYMENT_ALL_KEYS:
        normalized_key = ''.join(ch for ch in key if ch.isalnum()).lower()
        if token == normalized_key:
            return key
    return None

def _title_text(value: Any) -> str:
    """Normalize title payloads from server into plain text for matching."""
    if isinstance(value, dict):
        return str(value.get('message', '') or '')
    if isinstance(value, str):
        return value
    return ''

def _card_cost(card: Dict[str, Any]) -> int:
    try:
        return int(card.get('calculatedCost', card.get('cost', 0)) or 0)
    except Exception:
        return 0

def _card_vp(card: Dict[str, Any]) -> int:
    # Prefer metadata VP (card payload often omits it).
    name = str(card.get('name', '') or '')
    meta = _CARD_META_CACHE.get(name, {}) if isinstance(_CARD_META_CACHE, dict) else {}
    try:
        return int(meta.get('victoryPoints', card.get('victoryPoints', 0)) or 0)
    except Exception:
        return 0

def _metadata_for_card(card_or_name: Any) -> Dict[str, Any]:
    if isinstance(card_or_name, dict):
        card_name = str(card_or_name.get('name', '') or '')
    else:
        card_name = str(card_or_name or '')
    if not card_name or not isinstance(_CARD_META_CACHE, dict):
        return {}
    meta = _CARD_META_CACHE.get(card_name, {})
    return meta if isinstance(meta, dict) else {}

def _card_starting_megacredits(card: Dict[str, Any], default: int = 0) -> int:
    raw_direct = card.get('startingMegaCredits', card.get('startingMegacredits', None))
    if raw_direct is not None:
        direct = _safe_int(raw_direct, default)
        if direct > 0:
            return int(direct)
    meta = _metadata_for_card(card)
    raw_meta = meta.get('startingMegaCredits', meta.get('startingMegacredits', None))
    if raw_meta is not None:
        return _safe_int(raw_meta, default)
    return int(default)

def _card_keep_cost(card: Dict[str, Any], default: int = 3) -> int:
    raw_direct = card.get('cardCost', None)
    if raw_direct is not None:
        return max(1, _safe_int(raw_direct, default))
    meta = _metadata_for_card(card)
    if meta.get('cardCost', None) is not None:
        return max(1, _safe_int(meta.get('cardCost', default), default))
    return max(1, int(default))

def _with_metadata_tags(card: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(card or {})
    if not out.get('tags'):
        tags = _metadata_tags(str(out.get('name', '') or ''))
        if tags:
            out['tags'] = tags
    return out

def _extra_mc_needed_to_afford(player: Dict[str, Any], card: Dict[str, Any], max_extra: int) -> Optional[int]:
    safe_player = dict(player or {})
    if _can_afford_card(safe_player, card):
        return 0

    try:
        base_mc = int(safe_player.get('megaCredits', 0) or 0)
    except Exception:
        base_mc = 0
    limit = max(0, int(max_extra))
    for extra in range(1, limit + 1):
        probe = dict(safe_player)
        probe['megaCredits'] = base_mc + extra
        if _can_afford_card(probe, card):
            return extra
    return None


def _sell_patents_cap_for_generation(generation: int) -> int:
    if generation <= 6:
        return 1
    if generation <= 10:
        return 2
    return 2


def _is_engine_or_persistent_card(card: Dict[str, Any]) -> bool:
    meta = _metadata_for_card(card)
    if bool(card.get('hasAction', meta.get('hasAction', False))):
        return True
    vp = float(_card_vp(card))
    if vp > 0.0:
        return True
    desc = str(card.get('description', meta.get('description', '')) or '').lower()
    if 'production' in desc or 'ongoing' in desc or 'effect' in desc:
        return True
    tags = _card_tags(card)
    if tags.get('Science') or tags.get('Earth'):
        return True
    return False


def _should_offer_sell_patents(
    cards: List[Dict[str, Any]],
    waiting_for: Dict[str, Any],
    player_state: Optional[Dict[str, Any]],
    allow_mandatory_fallback: bool = True,
) -> bool:
    if not cards:
        return False

    min_cards = max(0, _safe_int(waiting_for.get('min', 1), 1))
    can_pass = bool(waiting_for.get('canPass', False))
    if allow_mandatory_fallback and min_cards > 0 and not can_pass:
        # Mandatory sell flow: must keep action available.
        return True

    if not isinstance(player_state, dict):
        return bool(min_cards > 0)
    player = player_state.get('thisPlayer', {}) or {}
    if not isinstance(player, dict):
        return bool(min_cards > 0)
    hand_cards = player_state.get('cardsInHand', []) or []
    if not isinstance(hand_cards, list) or not hand_cards:
        hand_cards = cards

    game = player_state.get('game', {}) if isinstance(player_state.get('game', {}), dict) else {}
    generation = _safe_int(game.get('generation', 1), 1)
    sell_cap = min(len(cards), max(1, _sell_patents_cap_for_generation(generation)))
    max_cards = max(min_cards, _safe_int(waiting_for.get('max', len(cards)), len(cards)))
    sell_cap = min(sell_cap, max_cards)
    if sell_cap <= 0:
        return False

    # Only offer selling when it unlocks at least one meaningful immediate play.
    for hand_card in hand_cards:
        if not isinstance(hand_card, dict):
            continue
        needed = _extra_mc_needed_to_afford(player, _with_metadata_tags(hand_card), sell_cap)
        if needed is None or needed <= 0:
            continue
        if needed > sell_cap:
            continue
        if _is_engine_or_persistent_card(hand_card) or _card_cost(hand_card) >= 15:
            return True

    return bool(allow_mandatory_fallback and min_cards > 0 and not can_pass)


def _select_patents_to_sell(
    cards: List[Dict[str, Any]],
    waiting_for: Dict[str, Any],
    player_state: Optional[Dict[str, Any]],
    action_index: Optional[int] = None,
) -> List[str]:
    if not cards:
        return []

    min_cards = int(waiting_for.get('min', 1) or 1)
    max_cards = int(waiting_for.get('max', len(cards)) or len(cards))
    min_cards = max(0, min(min_cards, len(cards)))
    max_cards = max(min_cards, min(max_cards, len(cards)))

    # If caller encoded a bitmask, honor it first.
    normalized = 0
    try:
        normalized = int(action_index) if action_index is not None else 0
    except Exception:
        normalized = 0
    if normalized >= 702:
        normalized -= 702
    if normalized > 0:
        selected_by_mask: List[str] = []
        for i, card in enumerate(cards):
            if (normalized >> i) & 1:
                name = str(card.get('name', '') or '')
                if name:
                    selected_by_mask.append(name)
        if len(selected_by_mask) >= min_cards:
            return selected_by_mask[:max_cards]

    player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
    game = (player_state or {}).get('game', {}) if isinstance(player_state, dict) else {}
    generation = _safe_int(game.get('generation', 1), 1)
    strategic_sell_cap = _sell_patents_cap_for_generation(generation)
    can_pass = bool(waiting_for.get('canPass', False))

    def _sell_priority(card: Dict[str, Any]) -> Tuple[float, int, str]:
        """Lower tuple means more disposable in a sell-patents action."""
        name = str(card.get('name', '') or '')
        meta = _metadata_for_card(card)
        tags = _card_tags(card)
        has_action = bool(card.get('hasAction', meta.get('hasAction', False)))
        vp = float(_card_vp(card))
        keep_signal = 0.0
        if has_action:
            keep_signal += 2.8
        if vp > 0:
            keep_signal += min(vp, 3.0) * 1.4
        if tags.get('Science'):
            keep_signal += 0.8
        if tags.get('Building') or tags.get('Space'):
            keep_signal += 0.35
        if player and _can_afford_card(player, card):
            keep_signal += 1.1
        return (keep_signal, _card_cost(card), name)

    ranked_all = sorted(cards, key=_sell_priority)

    # Strategic sell: only sell up to a small cap to unlock a high-value card.
    if player and len(cards) > 1:
        hand_cards = (player_state or {}).get('cardsInHand', []) if isinstance(player_state, dict) else []
        if not isinstance(hand_cards, list) or not hand_cards:
            hand_cards = cards

        max_sell_for_target = min(max_cards, len(cards) - 1)
        max_sell_for_target = min(max_sell_for_target, strategic_sell_cap)
        best_target_name: Optional[str] = None
        best_needed: Optional[int] = None
        best_rank: Optional[Tuple[int, int, int]] = None
        best_target_persistent = False

        for hand_card in hand_cards:
            candidate = _with_metadata_tags(hand_card)
            name = str(candidate.get('name', '') or '')
            if not name:
                continue
            needed = _extra_mc_needed_to_afford(player, candidate, max_sell_for_target)
            if needed is None or needed <= 0:
                continue
            if needed > max_sell_for_target:
                continue
            target_persistent = _is_engine_or_persistent_card(candidate)

            # Higher VP first, then fewer sells, then higher impact/cost card.
            rank = (1 if target_persistent else 0, _card_vp(candidate), -int(needed), _card_cost(candidate))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_target_name = name
                best_needed = int(needed)
                best_target_persistent = bool(target_persistent)

        # If pass is legal, skip selling unless it unlocks a persistent/engine target.
        if can_pass and (best_target_name is None or not best_target_persistent):
            return []

        if best_target_name is not None and best_needed is not None:
            sell_count = max(min_cards, min(max_cards, best_needed, strategic_sell_cap))
            sell_pool = [c for c in ranked_all if str(c.get('name', '') or '') != best_target_name]
            if len(sell_pool) < sell_count:
                sell_pool = ranked_all
            selected_names = []
            for card in sell_pool:
                name = str(card.get('name', '') or '')
                if not name or name in selected_names:
                    continue
                selected_names.append(name)
                if len(selected_names) >= sell_count:
                    break
            if len(selected_names) >= min_cards:
                return selected_names[:max_cards]

    # Conservative default: sell the cheapest cards first.
    if can_pass:
        return []
    default_count = max(min_cards, 1)
    default_count = min(default_count, strategic_sell_cap)
    default_count = max(min_cards, min(default_count, max_cards))
    selected_default = []
    for card in ranked_all:
        name = str(card.get('name', '') or '')
        if not name or name in selected_default:
            continue
        selected_default.append(name)
        if len(selected_default) >= default_count:
            break
    return selected_default[:max_cards]

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return int(default)


def _rust_backend():
    return get_rust_module(required=True)


def _can_afford_cards_batch(player: Dict[str, Any], cards: List[Dict[str, Any]]) -> List[bool]:
    backend = _rust_backend()
    player_json = json.dumps(player or {})
    payload_cards: List[Dict[str, Any]] = []
    for card in cards:
        payload_cards.append(
            {
                'name': str(card.get('name', '') or ''),
                'calculatedCost': float(_card_cost(card)),
                'cost': float(_card_cost(card)),
                'tags': dict(_card_tags(card)),
                'victoryPoints': float(_card_vp(card)),
            }
        )
    out = backend.can_afford_cards(player_json, json.dumps(payload_cards))
    return [bool(v) for v in out]

def _is_special_card_prompt_title(title_l: str) -> bool:
    return (
        'standard project' in title_l
        or 'convert plants' in title_l
        or 'convert heat' in title_l
        or 'sell patents' in title_l
    )

def _is_card_selection_prompt(waiting_for: Dict[str, Any]) -> bool:
    title = _title_text(waiting_for.get('title', '')).lower()
    button_label = str(waiting_for.get('buttonLabel', '') or '').lower()
    return (
        'prelude' in title
        or 'select' in title
        or button_label in ['keep', 'buy', 'select', 'choose', 'take action', 'discard', 'confirm', 'ok', 'save']
        or waiting_for.get('showOnlyInLearnerMode', False)
        or waiting_for.get('selectBlueCardAction', False)
        or ('min' in waiting_for and 'max' in waiting_for)
    )

def _score_card_for_selection(card: Dict[str, Any], player_state: Optional[Dict[str, Any]]) -> float:
    cost = float(_card_cost(card))
    vp = float(_card_vp(card))
    tags = _card_tags(card)

    player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
    mc = float(player.get('megaCredits', 0) or 0)
    steel = float(player.get('steel', 0) or 0)
    titanium = float(player.get('titanium', 0) or 0)
    steel_value = float(player.get('steelValue', 2) or 2)
    titanium_value = float(player.get('titaniumValue', 3) or 3)

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
        tag_score += 0.2
    if tags.get('Plant'):
        tag_score += 0.2

    deterministic_tiebreak = (sum(ord(ch) for ch in str(card.get('name', '') or '')) % 97) * 1e-5
    return (vp * 0.45) + (affordable * 1.4) + (cheapness * 0.8) + tag_score + deterministic_tiebreak

def _enumerate_card_selection_masks(
    cards: List[Dict[str, Any]],
    min_cards: int,
    max_cards: int,
    player_state: Optional[Dict[str, Any]],
    limit: int = _CARD_SELECTION_MASK_LIMIT,
) -> List[int]:
    if not cards:
        return []
    payload_cards: List[Dict[str, Any]] = []
    for card in cards:
        payload_cards.append(
            {
                'name': str(card.get('name', '') or ''),
                'calculatedCost': float(_card_cost(card)),
                'cost': float(_card_cost(card)),
                'tags': dict(_card_tags(card)),
                'victoryPoints': float(_card_vp(card)),
                'isDisabled': bool(card.get('isDisabled', False)),
            }
        )
    payload = {
        'cards': payload_cards,
        'minCards': _safe_int(min_cards, 0),
        'maxCards': _safe_int(max_cards, len(cards)),
        'playerState': player_state or {},
    }
    combos = _rust_backend().enumerate_card_selection_combos(json.dumps(payload), max(1, int(limit)))
    masks: List[int] = []
    for combo in combos:
        mask = 0
        for idx in combo:
            try:
                mask |= (1 << int(idx))
            except Exception:
                continue
        masks.append(mask)
    return masks

def _decode_card_selection_mask_action(
    waiting_for: Dict[str, Any],
    action_index: Optional[int],
    player_state: Optional[Dict[str, Any]],
) -> Optional[int]:
    if action_index is None:
        return None
    normalized = _safe_int(action_index, -1)
    offset = normalized - _CARD_SELECTION_MASK_BASE
    if offset < 0 or offset >= _CARD_SELECTION_MASK_LIMIT:
        return None

    cards = waiting_for.get('cards', []) or []
    min_cards = _safe_int(waiting_for.get('min', 1), 1)
    max_cards = _safe_int(waiting_for.get('max', len(cards)), len(cards))
    masks = _enumerate_card_selection_masks(cards, min_cards, max_cards, player_state, _CARD_SELECTION_MASK_LIMIT)
    if 0 <= offset < len(masks):
        return int(masks[offset])
    return None

def _normalize_tag_name(tag: Any) -> str:
    raw = str(tag or '').strip()
    if not raw:
        return ''
    if raw and raw[0].islower():
        raw = raw.capitalize()
    return raw

def _card_tags(card: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    tags = card.get('tags')
    if isinstance(tags, dict):
        for k, v in tags.items():
            if v:
                name = _normalize_tag_name(k)
                if name:
                    out[name] = 1
    elif isinstance(tags, list):
        for tag in tags:
            name = _normalize_tag_name(tag)
            if name:
                out[name] = 1
    if out:
        return out
    return _metadata_tags(str(card.get('name', '') or ''))

def _initial_option_role(option: Dict[str, Any], index: int) -> str:
    title_l = _title_text(option.get('title', '')).lower()
    if 'corporation' in title_l:
        return 'corporation'
    if 'prelude' in title_l:
        return 'prelude'
    if 'ceo' in title_l:
        return 'ceo'
    if 'initial cards' in title_l or 'cards to buy' in title_l or 'project' in title_l:
        return 'project'
    min_cards = _safe_int(option.get('min', 0), 0)
    max_cards = _safe_int(option.get('max', 0), 0)
    if min_cards == 2 and max_cards == 2:
        return 'prelude'
    if min_cards == 1 and max_cards == 1 and index == 0:
        return 'corporation'
    return 'project' if index >= 2 else 'unknown'

def _score_initial_card(
    card: Dict[str, Any],
    role: str,
    project_tag_counts: Dict[str, int],
) -> float:
    name = str(card.get('name', '') or '')
    cost = float(_card_cost(card))
    vp = float(_card_vp(card))
    tags = _card_tags(card)
    starting_mc = float(_card_starting_megacredits(card, default=0))
    card_cost_override = float(_card_keep_cost(card, default=3))

    score = 0.0
    if role == 'project':
        score += vp * 6.0
        score += max(0.0, 20.0 - cost) * 0.18
    else:
        score += starting_mc * 0.45
        score += vp * 4.0
        if role == 'corporation':
            score -= max(0.0, card_cost_override - 3.0) * 14.0
            score += max(0.0, 3.0 - card_cost_override) * 6.0

    # Lightweight tag heuristics. For corporation/prelude, boost tags that match offered projects.
    tag_weights = {
        'Science': 2.3,
        'Building': 1.9,
        'Space': 1.8,
        'Plant': 1.6,
        'Earth': 1.3,
        'Microbe': 1.0,
        'Animal': 1.0,
        'Jovian': 1.0,
    }
    for tag_name, weight in tag_weights.items():
        if not tags.get(tag_name):
            continue
        if role == 'project':
            score += weight
        else:
            score += weight * (1.0 + 0.15 * float(project_tag_counts.get(tag_name, 0)))

    # Deterministic tiny tie-breaker by card name.
    score += (sum(ord(c) for c in name) % 100) * 1e-5
    return score

def _select_initial_card_names(
    option: Dict[str, Any],
    role: str,
    project_tag_counts: Dict[str, int],
    cap: Optional[int] = None,
    force_minimum: Optional[int] = None,
) -> List[str]:
    # Prefer currently enabled cards; disabled entries can produce invalid
    # SelectCardResponse payloads during startup flows.
    cards = _enabled_cards(option)
    if not cards:
        return []

    min_cards = max(0, min(_safe_int(option.get('min', 0), 0), len(cards)))
    max_cards = max(min_cards, min(_safe_int(option.get('max', len(cards)), len(cards)), len(cards)))
    if force_minimum is not None:
        min_cards = max(min_cards, int(force_minimum))
        min_cards = min(min_cards, max_cards)

    target = max_cards
    if cap is not None:
        target = min(target, max(0, int(cap)))
    target = max(min_cards, min(target, max_cards))

    ranked_cards = sorted(
        cards,
        key=lambda c: _score_initial_card(c, role, project_tag_counts),
        reverse=True,
    )

    if target <= 0:
        return []

    selected_names: List[str] = []
    for card in ranked_cards:
        name = str(card.get('name', '') or '')
        if not name or name in selected_names:
            continue
        selected_names.append(name)
        if len(selected_names) >= target:
            break

    if len(selected_names) < min_cards:
        for card in cards:
            name = str(card.get('name', '') or '')
            if not name or name in selected_names:
                continue
            selected_names.append(name)
            if len(selected_names) >= min_cards:
                break

    return selected_names[:max_cards]

def _enabled_cards(option: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards = [c for c in (option.get('cards', []) or []) if isinstance(c, dict)]
    enabled = [card for card in cards if not card.get('isDisabled', False)]
    return enabled if enabled else cards

def _count_tags(cards: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for card in cards:
        for tag_name in _card_tags(card).keys():
            counts[tag_name] = int(counts.get(tag_name, 0)) + 1
    return counts

def _startup_project_subset_scores(
    project_cards: List[Dict[str, Any]],
    min_cards: int,
    max_cards: int,
    project_tag_counts: Dict[str, int],
    corp_tags: Dict[str, int],
    prelude_tags: Dict[str, int],
    limit: int,
) -> List[Tuple[float, List[str]]]:
    if not project_cards:
        return [(0.0, [])] if min_cards == 0 else []

    ranked_cards = sorted(
        project_cards,
        key=lambda c: _score_initial_card(c, 'project', project_tag_counts),
        reverse=True,
    )
    candidate_cards = ranked_cards[:min(len(ranked_cards), 12)]

    top_scored: List[Tuple[float, List[str], str]] = []
    if min_cards == 0:
        top_scored.append((0.0, [], ''))

    for pick_count in range(max(1, min_cards), max_cards + 1):
        if pick_count > len(candidate_cards):
            break
        for combo in itertools.combinations(candidate_cards, pick_count):
            score_base = 0.0
            combo_tag_counts: Dict[str, int] = {}
            combo_cost = 0.0
            names: List[str] = []
            for card in combo:
                score_base += _score_initial_card(card, 'project', project_tag_counts)
                combo_cost += float(_card_cost(card))
                names.append(str(card.get('name', '') or ''))
                for tag_name in _card_tags(card).keys():
                    combo_tag_counts[tag_name] = int(combo_tag_counts.get(tag_name, 0)) + 1

            cheap_bonus = 0.0
            if pick_count > 0:
                cheap_bonus = max(0.0, (20.0 * pick_count) - combo_cost) / max(1.0, 20.0 * pick_count)

            synergy_bonus = 0.0
            for tag_name, count in combo_tag_counts.items():
                corp_pull = float(corp_tags.get(tag_name, 0))
                prelude_pull = float(prelude_tags.get(tag_name, 0))
                synergy_bonus += float(count) * (0.08 + (0.04 * corp_pull) + (0.02 * prelude_pull))

            diversity_bonus = 0.04 * float(len(combo_tag_counts))
            keep_count_bonus = 0.03 * float(pick_count)
            total = float(score_base + (0.70 * cheap_bonus) + synergy_bonus + diversity_bonus + keep_count_bonus)
            signature = '|'.join(sorted(names))
            top_scored.append((total, names, signature))

    top_scored.sort(key=lambda item: (item[0], len(item[1]), item[2]), reverse=True)
    dedup_seen: Set[str] = set()
    out: List[Tuple[float, List[str]]] = []
    for score, names, signature in top_scored:
        if signature in dedup_seen:
            continue
        dedup_seen.add(signature)
        out.append((float(score), list(names)))
        if len(out) >= max(1, int(limit)):
            break
    return out

def _enumerate_startup_plan_payloads(
    waiting_for: Dict[str, Any],
    player_state: Optional[Dict[str, Any]],
    max_plans: int = _STARTUP_PLAN_LIMIT,
) -> List[Dict[str, Any]]:
    options = waiting_for.get('options', []) or []
    if not options:
        return []

    roles = [_initial_option_role(opt, idx) for idx, opt in enumerate(options)]
    corp_idx = next((i for i, role in enumerate(roles) if role == 'corporation'), -1)
    project_idx = next((i for i, role in enumerate(roles) if role == 'project'), -1)
    prelude_idx = next((i for i, role in enumerate(roles) if role == 'prelude'), -1)
    ceo_idx = next((i for i, role in enumerate(roles) if role == 'ceo'), -1)
    if corp_idx < 0 or project_idx < 0:
        return []

    corp_option = options[corp_idx]
    project_option = options[project_idx]
    prelude_option = options[prelude_idx] if prelude_idx >= 0 else None
    ceo_option = options[ceo_idx] if ceo_idx >= 0 else None
    corp_cards = _enabled_cards(corp_option)
    project_cards = _enabled_cards(project_option)
    prelude_cards = _enabled_cards(prelude_option) if prelude_option else []
    ceo_cards = _enabled_cards(ceo_option) if ceo_option else []

    if not corp_cards:
        return []

    player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
    project_tag_counts: Dict[str, int] = {}
    for card in project_cards:
        for tag_name in _card_tags(card).keys():
            project_tag_counts[tag_name] = int(project_tag_counts.get(tag_name, 0)) + 1

    prelude_choices: List[Tuple[float, List[str], Dict[str, int]]] = [(0.0, [], {})]
    if prelude_option is not None:
        prelude_min = _safe_int(prelude_option.get('min', 2), 2)
        prelude_max = _safe_int(prelude_option.get('max', prelude_min), prelude_min)
        prelude_max = min(prelude_max, len(prelude_cards))
        prelude_min = min(prelude_min, prelude_max)
        prelude_choices = []
        if prelude_max == 0 and prelude_min == 0:
            prelude_choices.append((0.0, [], {}))
        else:
            for count in range(prelude_min, prelude_max + 1):
                for combo in itertools.combinations(prelude_cards, count):
                    names = [str(card.get('name', '') or '') for card in combo]
                    combo_score = sum(_score_initial_card(card, 'prelude', project_tag_counts) for card in combo)
                    combo_tags = _count_tags(list(combo))
                    signature = '|'.join(sorted(names))
                    prelude_choices.append((float(combo_score), names, combo_tags))
            prelude_choices.sort(key=lambda item: (item[0], len(item[1]), '|'.join(sorted(item[1]))), reverse=True)
            prelude_choices = prelude_choices[:32]

    ceo_choices: List[Tuple[float, List[str], Dict[str, int]]] = [(0.0, [], {})]
    if ceo_option is not None:
        ceo_min = _safe_int(ceo_option.get('min', 1), 1)
        ceo_max = _safe_int(ceo_option.get('max', ceo_min), ceo_min)
        ceo_max = min(ceo_max, len(ceo_cards))
        ceo_min = min(ceo_min, ceo_max)
        ceo_choices = []
        for count in range(ceo_min, ceo_max + 1):
            for combo in itertools.combinations(ceo_cards, count):
                names = [str(card.get('name', '') or '') for card in combo]
                combo_score = sum(_score_initial_card(card, 'ceo', project_tag_counts) for card in combo)
                combo_tags = _count_tags(list(combo))
                ceo_choices.append((float(combo_score), names, combo_tags))
        if not ceo_choices and ceo_min == 0:
            ceo_choices.append((0.0, [], {}))
        ceo_choices.sort(key=lambda item: (item[0], len(item[1]), '|'.join(sorted(item[1]))), reverse=True)
        ceo_choices = ceo_choices[:8]

    project_min = _safe_int(project_option.get('min', 0), 0)
    project_max = _safe_int(project_option.get('max', len(project_cards)), len(project_cards))
    project_max = min(project_max, len(project_cards))
    project_min = min(project_min, project_max)

    top_candidates: List[Tuple[float, Dict[str, Any], Tuple[Any, ...], Dict[str, Any]]] = []
    for corp_card in corp_cards:
        corp_name = str(corp_card.get('name', '') or '')
        corp_score = _score_initial_card(corp_card, 'corporation', project_tag_counts)
        corp_tags = _count_tags([corp_card])
        corp_start_mc = _card_starting_megacredits(
            corp_card,
            default=_safe_int(player.get('megaCredits', 40), 40),
        )
        corp_card_cost = _card_keep_cost(
            corp_card,
            default=_safe_int(player.get('cardCost', 3), 3),
        )
        affordability_cap = max(0, int(corp_start_mc // max(1, corp_card_cost)))
        legal_project_max = min(project_max, affordability_cap)
        if legal_project_max < project_min:
            if project_min == 0:
                legal_project_max = 0
            else:
                continue

        for prelude_score, prelude_names, prelude_tags in prelude_choices:
            project_choices = _startup_project_subset_scores(
                project_cards=project_cards,
                min_cards=project_min,
                max_cards=legal_project_max,
                project_tag_counts=project_tag_counts,
                corp_tags=corp_tags,
                prelude_tags=prelude_tags,
                limit=64,
            )
            if not project_choices:
                continue
            for ceo_score, ceo_names, ceo_tags in ceo_choices:
                ceo_tag_bonus = 0.0
                for tag_name, count in ceo_tags.items():
                    ceo_tag_bonus += 0.05 * float(count) * (1.0 + 0.1 * float(project_tag_counts.get(tag_name, 0)))
                for project_score, project_names in project_choices:
                    keep_count = len(project_names)
                    budget_ratio = float(keep_count) / float(max(1, legal_project_max))
                    total_score = (
                        float(corp_score)
                        + float(prelude_score)
                        + float(ceo_score)
                        + float(project_score)
                        + (0.18 * budget_ratio)
                    )
                    responses: List[Dict[str, Any]] = []
                    for idx, option in enumerate(options):
                        if idx == corp_idx:
                            responses.append({'type': 'card', 'cards': [corp_name]})
                        elif idx == prelude_idx:
                            responses.append({'type': 'card', 'cards': list(prelude_names)})
                        elif idx == ceo_idx:
                            responses.append({'type': 'card', 'cards': list(ceo_names)})
                        elif idx == project_idx:
                            responses.append({'type': 'card', 'cards': list(project_names)})
                        else:
                            responses.append(build_response_for_input(option, None, player_state))
                    payload = {'type': 'initialCards', 'responses': responses}
                    
                    # Tuple signature for O(1) deduplication (much faster than json.dumps)
                    sig_prelude = tuple(sorted(prelude_names))
                    sig_ceo = tuple(sorted(ceo_names))
                    sig_project = tuple(sorted(project_names))
                    signature = (corp_name, sig_prelude, sig_ceo, sig_project)
                    
                    top_candidates.append(
                        (
                            float(total_score + ceo_tag_bonus),
                            payload,
                            signature,
                            {
                                'index': len(top_candidates),
                                'score': float(total_score + ceo_tag_bonus),
                                'corp': corp_name,
                                'prelude': list(prelude_names),
                                'ceo': list(ceo_names),
                                'project': list(project_names),
                            },
                        )
                    )

    if not top_candidates:
        return []

    rank_payload = {'candidates': [item[3] for item in top_candidates]}
    backend = get_rust_module(required=False)
    if backend is not None:
        try:
            ranked_json = backend.rank_startup_plans(
                json.dumps(rank_payload),
                max(1, int(max_plans)),
            )
            ranked = json.loads(str(ranked_json or "[]"))
            ranked_payloads: List[Dict[str, Any]] = []
            for row in ranked:
                idx = _safe_int((row or {}).get('index', -1), -1)
                if 0 <= idx < len(top_candidates):
                    ranked_payloads.append(top_candidates[idx][1])
                if len(ranked_payloads) >= max(1, int(max_plans)):
                    break
            if ranked_payloads:
                return ranked_payloads
        except Exception as exc:
            logger.warning(
                "Rust startup-plan ranking failed; falling back to Python ranking: %s",
                exc,
            )

    # Strict Rust cutover should always return ranked payloads, but keep this guard
    # in case malformed ranking data is produced.
    top_candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
    dedup: Set[Tuple[Any, ...]] = set()
    selected: List[Dict[str, Any]] = []
    for _, payload, signature, _ in top_candidates:
        if signature in dedup:
            continue
        dedup.add(signature)
        selected.append(payload)
        if len(selected) >= max(1, int(max_plans)):
            break
    return selected

def _build_initial_setup_response_legacy(waiting_for: Dict[str, Any], player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    options = waiting_for.get('options', []) or []
    if not options:
        return {'type': 'initialCards', 'responses': []}

    roles = [_initial_option_role(opt, idx) for idx, opt in enumerate(options)]

    # Gather project-tag distribution for simple corp/prelude synergy.
    project_tag_counts: Dict[str, int] = {}
    for idx, option in enumerate(options):
        if roles[idx] != 'project':
            continue
        for card in option.get('cards', []) or []:
            for tag_name in _card_tags(card).keys():
                project_tag_counts[tag_name] = int(project_tag_counts.get(tag_name, 0)) + 1
        break

    responses: List[Optional[Dict[str, Any]]] = [None] * len(options)
    selected_corp_card: Optional[Dict[str, Any]] = None

    # Pick corporation first (needed for card-buy budget).
    for idx, option in enumerate(options):
        if roles[idx] != 'corporation':
            continue
        chosen = _select_initial_card_names(option, 'corporation', project_tag_counts, cap=1, force_minimum=1)
        responses[idx] = {'type': 'card', 'cards': chosen}
        if chosen:
            for candidate in option.get('cards', []) or []:
                if str(candidate.get('name', '') or '') == chosen[0]:
                    selected_corp_card = candidate
                    break
        break

    for idx, option in enumerate(options):
        if responses[idx] is not None:
            continue
        role = roles[idx]

        if role == 'prelude':
            min_cards = _safe_int(option.get('min', 2), 2)
            max_cards = _safe_int(option.get('max', min_cards), min_cards)
            chosen = _select_initial_card_names(option, 'prelude', project_tag_counts, cap=max_cards, force_minimum=min_cards)
            responses[idx] = {'type': 'card', 'cards': chosen}
            continue

        if role == 'ceo':
            chosen = _select_initial_card_names(option, 'ceo', project_tag_counts, cap=1, force_minimum=1)
            responses[idx] = {'type': 'card', 'cards': chosen}
            continue

        if role == 'project':
            # Keep project cards under the hard affordability check:
            # selected_count * card_cost <= selected_corporation.startingMegaCredits
            player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
            corp_start_mc = _safe_int(
                (selected_corp_card or {}).get('startingMegaCredits', player.get('megaCredits', 40)),
                40,
            )
            card_cost = _safe_int(
                (selected_corp_card or {}).get('cardCost', player.get('cardCost', 3)),
                3,
            )
            card_cost = max(1, card_cost)

            min_cards = _safe_int(option.get('min', 0), 0)
            max_cards = _safe_int(option.get('max', len(option.get('cards', []) or [])), len(option.get('cards', []) or []))
            affordability_cap = max(0, corp_start_mc // card_cost)
            keep_cap = min(max_cards, affordability_cap, 6)
            if keep_cap >= 2:
                keep_cap = max(2, keep_cap)

            scored = sorted(
                option.get('cards', []) or [],
                key=lambda c: _score_initial_card(c, 'project', project_tag_counts),
                reverse=True,
            )
            strong_cards = [c for c in scored if _score_initial_card(c, 'project', project_tag_counts) >= 1.0]
            desired = min(keep_cap, len(strong_cards)) if keep_cap > 0 else 0
            if keep_cap >= 2:
                desired = max(desired, 2)
            desired = max(min_cards, min(keep_cap, desired))
            chosen = _select_initial_card_names(option, 'project', project_tag_counts, cap=desired, force_minimum=min_cards)
            responses[idx] = {'type': 'card', 'cards': chosen}
            continue

        # Unknown initial option type; use safe generic behavior.
        responses[idx] = build_response_for_input(option, None, player_state)

    final_responses: List[Dict[str, Any]] = [
        response if response is not None else {'type': 'pass'}
        for response in responses
    ]
    return {'type': 'initialCards', 'responses': final_responses}

def _build_initial_setup_response(waiting_for: Dict[str, Any], player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    selection_mode = str(os.getenv("AGENT_STARTUP_PLAN_SELECTION", "best")).strip().lower()
    if selection_mode in ("legacy", "original", "legacy_only"):
        return _build_initial_setup_response_legacy(waiting_for, player_state)

    plans = _enumerate_startup_plan_payloads(
        waiting_for=waiting_for,
        player_state=player_state,
        max_plans=_STARTUP_PLAN_LIMIT,
    )
    if plans:
        if selection_mode in ("random", "sample", "uniform"):
            try:
                top_k = int(os.getenv("AGENT_STARTUP_PLAN_RANDOM_TOP_K", "4"))
            except Exception:
                top_k = 4
            top_k = max(1, min(int(top_k), len(plans)))
            return random.choice(plans[:top_k])
        return plans[0]
    return _build_initial_setup_response_legacy(waiting_for, player_state)

def _payment_empty() -> Dict[str, int]:
    return {k: 0 for k in _PAYMENT_ALL_KEYS}

def _extract_reserve_units(source: Optional[Dict[str, Any]]) -> Dict[str, int]:
    reserve = _payment_empty()
    if not isinstance(source, dict):
        return reserve
    raw_reserve = source.get('reserveUnits', {})
    if not isinstance(raw_reserve, dict):
        return reserve
    for raw_key, raw_value in raw_reserve.items():
        canonical = _canonical_payment_resource(raw_key)
        if not canonical:
            continue
        reserve[canonical] = max(reserve[canonical], max(0, _safe_int(raw_value, 0)))
    return reserve

def _merge_reserve_units(
    waiting_for: Optional[Dict[str, Any]],
    card: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    merged = _payment_empty()
    waiting_reserve = _extract_reserve_units(waiting_for)
    card_reserve = _extract_reserve_units(card)
    for key in _PAYMENT_ALL_KEYS:
        merged[key] = max(
            int(waiting_reserve.get(key, 0) or 0),
            int(card_reserve.get(key, 0) or 0),
        )
    return merged

def _payment_signature(payment: Dict[str, int]) -> Tuple[int, ...]:
    return tuple(int(payment.get(k, 0) or 0) for k in _PAYMENT_ALL_KEYS)

def _payment_action_offset(action_index: Optional[int]) -> int:
    idx = _safe_int(action_index, 0)
    if _PAYMENT_ACTION_BASE <= idx < (_PAYMENT_ACTION_BASE + 100):
        return int(idx - _PAYMENT_ACTION_BASE)
    return 0

def _extract_payment_project_name(waiting_for: Dict[str, Any]) -> str:
    title = waiting_for.get('title')
    if isinstance(title, dict):
        try:
            for item in title.get('data', []) or []:
                value = item.get('value')
                if isinstance(value, str) and value:
                    return value
        except Exception:
            return ''
        return ''
    if isinstance(title, str):
        return title
    return ''

def _payment_allowed(waiting_for: Dict[str, Any], resource: str) -> bool:
    payment_options = waiting_for.get('paymentOptions', {}) or {}
    if resource == 'megaCredits':
        return True
    if resource == 'titanium' and payment_options.get('lunaTradeFederationTitanium', False):
        return True
    return bool(payment_options.get(resource, False))

def _payment_available_units(
    waiting_for: Dict[str, Any],
    player: Dict[str, Any],
    resource: str,
    reserve_units: Optional[Dict[str, int]] = None,
) -> int:
    if resource in ['megaCredits', 'steel', 'titanium', 'heat', 'plants']:
        raw_units = max(0, int(player.get(resource, 0) or 0))
    elif resource in waiting_for:
        raw_units = max(0, int(waiting_for.get(resource, 0) or 0))
    else:
        raw_units = max(0, int(player.get(resource, 0) or 0))
    reserved = max(0, int((reserve_units or {}).get(resource, 0) or 0))
    return max(0, int(raw_units - reserved))

def _payment_values(waiting_for: Dict[str, Any], player: Dict[str, Any]) -> Dict[str, int]:
    values = dict(_PAYMENT_DEFAULT_VALUES)
    steel_value = max(1, int(player.get('steelValue', 2) or 2))
    titanium_value = max(1, int(player.get('titaniumValue', 3) or 3))
    values['steel'] = steel_value
    values['titanium'] = titanium_value

    # Luna Trade Federation: titanium spent as MC loses one value point.
    payment_options = waiting_for.get('paymentOptions', {}) or {}
    if payment_options.get('lunaTradeFederationTitanium', False) and not payment_options.get('titanium', False):
        values['titanium'] = max(1, titanium_value - 1)
    return values

def _apply_project_specific_payment_constraints(
    payment: Dict[str, int],
    waiting_for: Dict[str, Any],
    player: Dict[str, Any],
    reserve_units: Optional[Dict[str, int]] = None,
) -> None:
    project_name_l = _extract_payment_project_name(waiting_for).lower()
    if not project_name_l:
        return

    if ('road' in project_name_l) and ('infrastructure' in project_name_l or 'moon' in project_name_l):
        if _payment_allowed(waiting_for, 'steel'):
            have = _payment_available_units(waiting_for, player, 'steel', reserve_units=reserve_units)
            if have > 0 and int(payment.get('steel', 0) or 0) < 1:
                payment['steel'] = 1
    if ('lunar mine' in project_name_l) or ('moon mine' in project_name_l):
        if _payment_allowed(waiting_for, 'titanium'):
            have = _payment_available_units(waiting_for, player, 'titanium', reserve_units=reserve_units)
            if have > 0 and int(payment.get('titanium', 0) or 0) < 1:
                payment['titanium'] = 1
    if ('lunar habitat' in project_name_l) or ('moon habitat' in project_name_l):
        if _payment_allowed(waiting_for, 'titanium'):
            have = _payment_available_units(waiting_for, player, 'titanium', reserve_units=reserve_units)
            if have > 0 and int(payment.get('titanium', 0) or 0) < 1:
                payment['titanium'] = 1

    # Stratospheric Birds (and similar floater-retain prompts) cannot spend every floater.
    floater_reserve = max(0, int((reserve_units or {}).get('floaters', 0) or 0))
    if 'stratospheric birds' in project_name_l:
        floater_reserve = max(int(floater_reserve), 1)
    if floater_reserve > 0:
        total_floaters = _payment_available_units(waiting_for, player, 'floaters', reserve_units=None)
        max_spend_floaters = max(0, int(total_floaters - floater_reserve))
        spent_floaters = max(0, int(payment.get('floaters', 0) or 0))
        if spent_floaters > max_spend_floaters:
            payment['floaters'] = int(max_spend_floaters)

def _finalize_payment(payment: Dict[str, int]) -> Dict[str, int]:
    out = _payment_empty()
    for key in _PAYMENT_ALL_KEYS:
        out[key] = max(0, int(payment.get(key, 0) or 0))
    return out

def _payment_total_value(payment: Dict[str, int], values: Dict[str, int]) -> int:
    total = 0
    for key in _PAYMENT_ALL_KEYS:
        units = max(0, int(payment.get(key, 0) or 0))
        unit_value = max(0, int(values.get(key, 0) or 0))
        total += int(units * unit_value)
    return int(total)

def _is_valid_payment_candidate(
    payment: Dict[str, int],
    waiting_for: Dict[str, Any],
    player: Dict[str, Any],
    available_by_resource: Dict[str, int],
    values: Dict[str, int],
    amount: int,
    reserve_units: Optional[Dict[str, int]] = None,
) -> bool:
    normalized = _finalize_payment(payment)

    # Enforce project-specific constraints and validate resulting usage.
    _apply_project_specific_payment_constraints(
        normalized,
        waiting_for,
        player,
        reserve_units=reserve_units,
    )
    normalized = _finalize_payment(normalized)

    for key in _PAYMENT_ALL_KEYS:
        units = int(normalized.get(key, 0) or 0)
        if units <= 0:
            continue
        if not _payment_allowed(waiting_for, key):
            return False
        available_units = max(0, int(available_by_resource.get(key, 0) or 0))
        if units > available_units:
            return False

    paid_value = _payment_total_value(normalized, values)
    if int(amount) > 0 and int(paid_value) < int(amount):
        return False
    return True

def _enumerate_payment_candidates(
    waiting_for: Dict[str, Any],
    player_state: Optional[Dict[str, Any]],
    max_candidates: int = _PAYMENT_ACTION_VARIANTS,
) -> List[Dict[str, int]]:
    player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
    amount = max(0, int(waiting_for.get('amount', 0) or 0))
    values = _payment_values(waiting_for, player)
    reserve_units = _merge_reserve_units(waiting_for)

    available_by_resource: Dict[str, int] = {}
    for res in _PAYMENT_ALL_KEYS:
        if not _payment_allowed(waiting_for, res):
            available_by_resource[res] = 0
            continue
        available_by_resource[res] = _payment_available_units(
            waiting_for,
            player,
            res,
            reserve_units=reserve_units,
        )

    mc_available = int(available_by_resource.get('megaCredits', 0) or 0)
    non_mc_resources: List[Tuple[str, int, int]] = []
    for res in _PAYMENT_ALL_KEYS:
        if res == 'megaCredits':
            continue
        units = int(available_by_resource.get(res, 0) or 0)
        value = int(values.get(res, 0) or 0)
        if units > 0 and value > 0:
            non_mc_resources.append((res, units, value))

    max_unit_value = 1
    if non_mc_resources:
        max_unit_value = max(int(v) for _, _, v in non_mc_resources)
    # Keep search bounded: enough to find minimal-overpay mixes.
    cap = int(amount + (max_unit_value * 3))

    # dp[payment_value] -> (units_used, {resource -> units})
    dp: Dict[int, Tuple[int, Dict[str, int]]] = {0: (0, {})}
    for res, units_avail, unit_value in non_mc_resources:
        existing_items = list(dp.items())
        for paid_value, (units_used, combo) in existing_items:
            for used in range(1, units_avail + 1):
                new_value = paid_value + (used * unit_value)
                if new_value > cap:
                    break
                next_combo = dict(combo)
                next_combo[res] = int(next_combo.get(res, 0) + used)
                next_units = int(units_used + used)
                existing = dp.get(new_value)
                if existing is None or next_units < int(existing[0]):
                    dp[new_value] = (next_units, next_combo)

    scored_candidates: List[Tuple[Tuple[int, int, int, int], Dict[str, int]]] = []
    for paid_non_mc, (units_used, combo) in dp.items():
        needed_mc = max(0, int(amount - paid_non_mc))
        if needed_mc > mc_available:
            continue
        total_paid = int(paid_non_mc + needed_mc)
        overpay = max(0, int(total_paid - amount))

        payment = _payment_empty()
        for res, units in combo.items():
            payment[res] = int(units)
        payment['megaCredits'] = int(needed_mc)
        _apply_project_specific_payment_constraints(
            payment,
            waiting_for,
            player,
            reserve_units=reserve_units,
        )
        payment = _finalize_payment(payment)

        # Prefer exact/low-overpay, then lower MC usage, then higher non-MC usage.
        score = (overpay, int(payment['megaCredits']), -int(paid_non_mc), int(units_used))
        scored_candidates.append((score, payment))

    scored_candidates.sort(key=lambda item: item[0])

    candidates: List[Dict[str, int]] = []
    seen = set()

    def add_candidate(payment: Dict[str, int]) -> None:
        nonlocal candidates
        if len(candidates) >= max(1, int(max_candidates)):
            return
        normalized = _finalize_payment(payment)
        if not _is_valid_payment_candidate(
            normalized,
            waiting_for,
            player,
            available_by_resource,
            values,
            amount,
            reserve_units=reserve_units,
        ):
            return
        _apply_project_specific_payment_constraints(
            normalized,
            waiting_for,
            player,
            reserve_units=reserve_units,
        )
        normalized = _finalize_payment(normalized)
        signature = _payment_signature(normalized)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(normalized)

    for _, payment in scored_candidates:
        add_candidate(payment)

    # Add constrained fallbacks (must still pass legality checks).
    if amount > 0:
        if mc_available >= amount:
            all_mc = _payment_empty()
            all_mc['megaCredits'] = int(amount)
            add_candidate(all_mc)

        max_local = _payment_empty()
        max_local['megaCredits'] = max(0, int(mc_available))
        for res, units, _ in non_mc_resources:
            max_local[res] = int(max(0, units))
        add_candidate(max_local)

        by_value_desc = sorted(non_mc_resources, key=lambda item: item[2], reverse=True)
        for res, units_avail, unit_value in by_value_desc:
            single_resource = _payment_empty()
            needed_units = int((amount + unit_value - 1) // unit_value)
            if needed_units <= int(units_avail):
                single_resource[res] = int(needed_units)
                add_candidate(single_resource)

            hybrid = _payment_empty()
            use_units = max(1, min(int(units_avail), int(amount // unit_value) if unit_value > 0 else 0))
            hybrid[res] = int(use_units)
            hybrid['megaCredits'] = max(0, int(amount - (use_units * unit_value)))
            add_candidate(hybrid)

    if not candidates:
        if amount <= 0:
            add_candidate(_payment_empty())
        else:
            return []
    return candidates[:max(1, int(max_candidates))]

# --- Action Response Builder for Terraforming Mars RL Agent ---
def build_response_for_input(waiting_for, action_index=None, player_state=None):
    """
    Given a waiting_for dict (and optionally an action_index and player_state),
    returns the correct response dictionary for all possible input types,
    including recursive handling for 'or' and 'and'.
    """
    input_type = waiting_for.get('type', '')
    alias_input_types = {
        'selectAmount': 'amount',
        'selectOption': 'option',
        'selectPlayer': 'player',
        'selectPayment': 'payment',
        'selectResources': 'resources',
        'selectProductionToLose': 'productionToLose',
        'selectColony': 'colony',
        'selectParty': 'party',
        'selectDelegate': 'delegate',
        'selectGlobalEvent': 'globalEvent',
        'selectClaimedUndergroundToken': 'claimedUndergroundToken',
        'selectInitialCards': 'initialCards',
    }
    input_type = alias_input_types.get(input_type, input_type)
    # Helper to get a default action index if not provided
    def get_default_index(options):
        return 0 if not options else min(len(options) - 1, 0)
    def normalize_index(raw_index, base=None):
        try:
            idx = int(raw_index)
        except Exception:
            return 0
        if base is not None and idx >= int(base):
            idx -= int(base)
        return idx

    # --- Recursive types ---
    if input_type == 'or':
        options = waiting_for.get('options', [])
        # logger.info(f"Building OR response with action_index={action_index}, options={[opt.get('type', 'unknown') for opt in options]}")
        
        # Map action_index to the correct option based on action ranges
        selected_idx = 0
        selected_via_option_range = False
        if action_index is not None:
            # Check if this is a SELECT_OPTION range action (200+)
            if 200 <= action_index < 300:
                # Map SELECT_OPTION range back to option index
                selected_idx = action_index - 200
                selected_via_option_range = True
                # This action chooses the OR option itself, not an inner sub-index.
                action_index = None
            elif _CARD_SELECTION_MASK_BASE <= action_index < (_CARD_SELECTION_MASK_BASE + _CARD_SELECTION_MASK_LIMIT):
                matched = False
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title_l = _title_text(option.get('title', '')).lower()
                    if option_type in ['selectCard', 'card'] and not _is_special_card_prompt_title(option_title_l):
                        if _is_card_selection_prompt(option):
                            selected_idx = i
                            matched = True
                            break
                if not matched:
                    selected_idx = 0
            elif action_index >= 600 and action_index < 700:
                # This is a direct award selection (600+ range)
                # Find which option contains the award selection
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type == 'or' and 'fund an award' in option_title.lower():
                        selected_idx = i
                        # Adjust action_index for the award selection
                        action_index = action_index - 600
                        break
                else:
                    # If no award option found, default to first option
                    selected_idx = 0
            elif action_index >= 100 and action_index < 200:
                # This is a direct standard project selection (100-199 range)
                # Find which option contains the standard projects
                logger.info(f"Standard project selection action_index: {action_index}")
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type in ['selectCard', 'card'] and 'standard project' in option_title.lower():
                        selected_idx = i
                        # Adjust action_index for the standard project selection
                        # Instead of just subtracting 100, we need to map to the actual project index
                        project_index = action_index - 100
                        action_index = project_index
                        logger.info(f"Standard project selection action_index: {action_index}")
                        break
                else:
                    # If no standard projects option found, default to first option
                    selected_idx = 0
            elif action_index == 700:
                # This is a direct convert plants action
                # Find which option contains the convert plants action
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type in ['selectCard', 'card'] and 'convert plants' in option_title.lower():
                        selected_idx = i
                        action_index = 0  # Convert plants doesn't need sub-index
                        break
                else:
                    # If no convert plants option found, default to first option
                    selected_idx = 0
            elif action_index == 701:
                # This is a direct convert heat action
                # Find which option contains the convert heat action
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type in ['selectCard', 'card'] and 'convert heat' in option_title.lower():
                        selected_idx = i
                        action_index = 0  # Convert heat doesn't need sub-index
                        break
                else:
                    # If no convert heat option found, default to first option
                    selected_idx = 0
            elif action_index == 702:
                # This is a direct sell patents action
                # Find which option contains the sell patents action
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type in ['selectCard', 'card'] and 'sell patents' in option_title.lower():
                        selected_idx = i
                        action_index = 0  # Sell patents doesn't need sub-index
                        break
                else:
                    # If no sell patents option found, default to first option
                    selected_idx = 0
            elif action_index >= 0 and action_index < 100:
                logger.info(f"Project card selection action_index: {action_index}")
                # This is a direct project card selection (0-99 range)
                # Find which option contains the project cards
                matched = False
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title = _title_text(option.get('title', ''))
                    
                    if option_type in ['selectProjectCardToPlay', 'projectCard']:
                        selected_idx = i
                        # action_index is already correct for card selection
                        matched = True
                        break
                    elif option_type in ['selectCard', 'card'] and 'standard project' in option_title.lower():
                        # Check if this is a standard project card
                        cards = option.get('cards', [])
                        if action_index < len(cards):
                            selected_idx = i
                            # Adjust action_index for the standard project selection
                            action_index = action_index
                            matched = True
                            break
                    elif option_type in ['selectCard', 'card'] and not _is_special_card_prompt_title(option_title.lower()):
                        if _is_card_selection_prompt(option):
                            cards = option.get('cards', []) or []
                            enabled_indices = [idx for idx, c in enumerate(cards) if not c.get('isDisabled', False)]
                            if not enabled_indices:
                                enabled_indices = list(range(len(cards)))
                            normalized_action = int(action_index)
                            if normalized_action in enabled_indices or (0 <= normalized_action < len(cards)):
                                selected_idx = i
                                matched = True
                                break
                if not matched:
                    # No card-like option matched; treat low index as plain OR option index.
                    if 0 <= action_index < len(options):
                        selected_idx = int(action_index)
                    else:
                        selected_idx = 0
            else:
                # Determine which option this action_index corresponds to
                current_action_count = 0
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    if option_type in ['selectProjectCardToPlay', 'projectCard']:
                        cards = option.get('cards', [])
                        if current_action_count <= action_index < current_action_count + len(cards):
                            selected_idx = i
                            # Adjust action_index for the sub-option
                            action_index = action_index - current_action_count
                            logger.info(f"Standard project selection action_index: {action_index}")
                            break
                        current_action_count += len(cards)
                    elif option_type in ['selectCard', 'card'] and 'standard project' in _title_text(option.get('title', '')).lower():
                        cards = option.get('cards', [])
                        if current_action_count <= action_index < current_action_count + len(cards):
                            selected_idx = i
                            action_index = action_index - current_action_count
                            logger.info(f"Standard project selection action_index: {action_index}")
                            break
                        current_action_count += len(cards)
                    elif option_type == 'or' and 'fund an award' in _title_text(option.get('title', '')).lower():
                        award_options = option.get('options', [])
                        if current_action_count <= action_index < current_action_count + len(award_options):
                            selected_idx = i
                            action_index = action_index - current_action_count
                            logger.info(f"Award selection action_index: {action_index}")
                            break
                        current_action_count += len(award_options)
                    elif option_type in ['selectCard', 'card'] and 'convert plants' in _title_text(option.get('title', '')).lower():
                        if action_index == current_action_count:
                            selected_idx = i
                            logger.info(f"Convert plants action_index: {action_index}")
                            break
                        current_action_count += 1
                    elif option_type in ['selectCard', 'card'] and 'convert heat' in _title_text(option.get('title', '')).lower():
                        if action_index == current_action_count:
                            selected_idx = i
                            logger.info(f"Convert heat action_index: {action_index}")
                            break
                        current_action_count += 1
                    elif option_type in ['selectCard', 'card'] and 'sell patents' in _title_text(option.get('title', '')).lower():
                        if action_index == current_action_count:
                            selected_idx = i
                            logger.info(f"Sell patents action_index: {action_index}")
                            break
                        current_action_count += 1
                    else:
                        if current_action_count == action_index:
                            selected_idx = i
                            logger.info(f"Other action_index: {action_index}")
                            break
                        current_action_count += 1
        
        if len(options) == 0:
            return {'type': 'option'}
        if selected_idx < 0 or selected_idx >= len(options):
            selected_idx = 0
        selected_option = options[selected_idx] if options else {}
        
        # logger.info(f"Selected option {selected_idx}: {selected_option.get('type', 'unknown')} with title: {selected_option.get('title', 'unknown')}")
        
        # If this selection came from SELECT_OPTION range, reset sub-index to a sane default
        # so that sub-responses like projectCard/selectCard don't see an out-of-range index
        sub_action_index = action_index
        try:
            option_title = selected_option.get('title', '')
            if isinstance(option_title, dict):
                option_title = option_title.get('message', '')
            option_title_l = str(option_title).lower()
        except Exception:
            option_title_l = ''

        # For explicit SELECT_OPTION picks, avoid leaking outer action index into nested input parsing.
        if selected_via_option_range:
            sub_action_index = None
        # For standard projects, we need to pass the project index to the sub-option
        elif selected_option.get('type', '') in ['selectCard', 'card'] and 'standard project' in option_title_l:
            # action_index already contains the project index, so we don't need to change sub_action_index
            pass
        elif action_index is not None and 200 <= action_index < 300:
            opt_type = selected_option.get('type', '')
            if opt_type in ['projectCard', 'selectProjectCardToPlay']:
                sub_action_index = 0
            elif opt_type == 'card' and (
                selected_option.get('selectBlueCardAction', False)
                or 'standard project' in option_title_l
                or 'sell patents' in option_title_l
            ):
                # For standard projects selected via SELECT_OPTION, extract the project index
                if 'standard project' in option_title_l and action_index >= 200:
                    # Map the SELECT_OPTION index back to the project index
                    # SELECT_OPTION_STANDARD_PROJECTS is at base 200, so action_index 200+ maps to project 0+
                    # But we need to be more careful about the mapping
                    # If action_index is exactly 203 (SELECT_OPTION_STANDARD_PROJECTS), we should select a random project
                    # or use some other mechanism to select a specific project
                    if action_index == 203:  # SELECT_OPTION_STANDARD_PROJECTS
                        sub_action_index = 0
                        logger.info(f"Standard project selection via SELECT_OPTION: action_index={action_index}, defaulting to project 0")
                    else:
                        # For other action indices, try to map to a project index
                        # This is a fallback for cases where we have more specific action indices
                        sub_action_index = action_index - 200
                        logger.info(f"Standard project selection via SELECT_OPTION: action_index={action_index}, project_index={sub_action_index}")
                else:
                    sub_action_index = 0
                logger.info(f"Standard project selection action_index: {action_index}")

        # Recursively build the response for the selected option
        sub_response = build_response_for_input(selected_option, sub_action_index, player_state)
        # logger.info(f"Sub-response for option {selected_idx}: {sub_response}")
        return {'type': 'or', 'index': int(selected_idx), 'response': sub_response}
    elif input_type == 'and':
        options = waiting_for.get('options', [])
        responses = [build_response_for_input(opt, None, player_state) for opt in options]
        return {'type': 'and', 'responses': responses}
    elif input_type == 'initialCards':
        normalized_initial_index = _safe_int(action_index, -1)
        if action_index is None or normalized_initial_index == 800:
            return _build_initial_setup_response(waiting_for, player_state)
        if _STARTUP_PLAN_BASE <= normalized_initial_index < (_STARTUP_PLAN_BASE + _STARTUP_PLAN_LIMIT):
            # Using the ActionDecoder instance for caching if available (via dirty hack or refactoring)
            # However, build_response_for_input is static. 
            # We'll regenerate here, but _enumerate_startup_plan_payloads is now faster.
            startup_plans = _enumerate_startup_plan_payloads(
                waiting_for=waiting_for,
                player_state=player_state,
                max_plans=_STARTUP_PLAN_LIMIT,
            )
            startup_offset = normalized_initial_index - _STARTUP_PLAN_BASE
            if 0 <= startup_offset < len(startup_plans):
                return startup_plans[startup_offset]
            return _build_initial_setup_response(waiting_for, player_state)
        options = waiting_for.get('options', [])
        # Use action_index to determine selections if available
        if action_index is not None:
            # For initialCards, action_index should be a bitmask for all selections
            # We'll break it into parts: [corp_index, prelude_bitmask, project_bitmask]
            corp_index = (action_index >> 0) & 0xFF
            prelude_bitmask = (action_index >> 8) & 0xFFFF
            project_bitmask = (action_index >> 24) & 0xFFFFFFFF
            
            responses = []
            for i, opt in enumerate(options):
                # For corporation selection (single choice)
                if i == 0:
                    response = build_response_for_input(opt, corp_index, player_state)
                # For prelude selection (choose 2)
                elif i == 1:
                    response = build_response_for_input(opt, prelude_bitmask, player_state)
                # For project cards selection (multiple choices)
                else:
                    # Get player's starting money
                    player_money = player_state.get('thisPlayer', {}).get('megaCredits', 0) if player_state else 0
                    
                    # If action_index is 800, select affordable cards
                    if action_index == 800:
                        # Calculate how many cards the player can afford (3-8 is reasonable)
                        max_affordable = min(8, max(3, player_money // 8))
                        n_cards = len(opt.get('cards', []))
                        
                        # Create a bitmask with max_affordable cards selected
                        project_bitmask = (1 << max_affordable) - 1
                    response = build_response_for_input(opt, project_bitmask, player_state)
                responses.append(response)
            return {'type': 'initialCards', 'responses': responses}
        else:
            # Fallback to original behavior if no action_index
            responses = [build_response_for_input(opt, None, player_state) for opt in options]
            return {'type': 'initialCards', 'responses': responses}

    # --- Simple types ---
    elif input_type == 'amount':
        min_amount = int(waiting_for.get('min', 0))
        max_amount = int(waiting_for.get('max', 10))
        if action_index is not None:
            amount = normalize_index(action_index, 500)
        else:
            amount = min_amount if min_amount <= max_amount else 0
        amount = max(min_amount, min(max_amount, amount))
        return {'type': 'amount', 'amount': amount}
    elif input_type == 'card':
        cards = waiting_for.get('cards', [])
        min_cards = _safe_int(waiting_for.get('min', 1), 1)
        max_cards = _safe_int(waiting_for.get('max', len(cards)), len(cards))
        n = len(cards)
        title = _title_text(waiting_for.get('title', '')).lower()
        button_label = waiting_for.get('buttonLabel', '').lower()
        show_only_learner = waiting_for.get('showOnlyInLearnerMode', False)
        select_blue = waiting_for.get('selectBlueCardAction', False)
        if 'convert plants' in title:
            return {'type': 'convertPlants'}
        if 'convert heat' in title:
            return {'type': 'convertHeat'}
        # Treat Play flows as action (not selection). Selection covers keep/buy/choose/select screens.
        is_selection = (
            'prelude' in title
            or 'select' in title
            or button_label in ['keep', 'buy', 'select', 'choose', 'take action', 'discard', 'confirm', 'ok']
            or show_only_learner
            or select_blue
            or ('min' in waiting_for and 'max' in waiting_for)
        )
        # Special-case: Standard projects shown as 'card' type requires a selection list of names
        if 'standard project' in title:
            enabled_cards = [c for c in cards if not c.get('isDisabled', False)]
            if not enabled_cards:
                return {'type': 'pass'}
            idx = normalize_index(action_index, 100) if action_index is not None else 0
            if idx < 0 or idx >= len(enabled_cards):
                idx = 0
            chosen = [enabled_cards[idx]['name']]
            return {'type': 'card', 'cards': chosen}
        # Special-case: Sell patents sometimes appears as 'card' type
        if 'sell patent' in title:
            selected_names = _select_patents_to_sell(cards, waiting_for, player_state, action_index)
            if not selected_names:
                return {'type': 'pass'}
            return {'type': 'card', 'cards': selected_names}
        if is_selection:
            # Use bitmask logic for card selection (buy/keep/prelude phase)
            if min_cards == 1 and max_cards == 1:
                if action_index is not None and 0 <= action_index < n:
                    card_names = [cards[action_index]['name']]
                else:
                    # Heuristic: for prompts like "remove X from any card", pick the card with the most resources
                    keywords = ['remove', 'from any card', 'floater', 'floaters', 'microbe', 'microbes', 'animal', 'animals']
                    if any(k in title for k in keywords):
                        best_idx = 0
                        best_val = -1
                        for i, c in enumerate(cards):
                            # SerializedCard may use resourceCount; some UIs expose 'resources'
                            val = int(c.get('resourceCount', c.get('resources', 0)) or 0)
                            if val > best_val:
                                best_val = val
                                best_idx = i
                        card_names = [cards[best_idx]['name']] if cards else []
                    else:
                        card_names = [cards[0]['name']] if cards else []
            elif n > 0 and max_cards > 1:
                if action_index is not None:
                    normalized_action = _safe_int(action_index, 0)
                    decoded_mask = _decode_card_selection_mask_action(waiting_for, normalized_action, player_state)
                    if decoded_mask is None:
                        # Backward-compatible fallback:
                        # - index in [0, n): treat as selecting that single card
                        # - otherwise interpret as legacy raw bitmask
                        if 0 <= normalized_action < n:
                            decoded_mask = (1 << normalized_action)
                        else:
                            decoded_mask = max(0, normalized_action)

                    selected = []
                    for i in range(n):
                        if ((decoded_mask >> i) & 1) and not cards[i].get('isDisabled', False):
                            selected.append(cards[i]['name'])
                    if len(selected) < min_cards:
                        for i in range(n):
                            if cards[i].get('isDisabled', False):
                                continue
                            if cards[i]['name'] not in selected:
                                selected.append(cards[i]['name'])
                                if len(selected) >= min_cards:
                                    break
                    if len(selected) > max_cards:
                        selected = selected[:max_cards]
                    card_names = selected
                else:
                    fallback_count = max(0, min(min_cards, n))
                    card_names = [c['name'] for c in cards[:fallback_count] if not c.get('isDisabled', False)]
            else:
                fallback_count = max(0, min(min_cards, n))
                card_names = [c['name'] for c in cards[:fallback_count]] if cards else []
            return {'type': 'card', 'cards': card_names}
        else:
            # Play-card action (main phase)
            card_idx = action_index if action_index is not None else 0
            if 0 <= card_idx < n:
                card = cards[card_idx]
                # Calculate proper payment based on player resources
                if player_state:
                    payment = _calculate_card_payment(player_state, card)
                    if payment is None:
                        return {'type': 'pass'}
                else:
                    payment = {k: 0 for k in ['megaCredits', 'steel', 'titanium', 'heat', 'plants']}
                return {'type': 'card', 'card': card['name'], 'payment': payment}
            else:
                return {'type': 'pass'}
    elif input_type == 'colony':
        colonies = waiting_for.get('colonies', [])
        idx = normalize_index(action_index, 720) if action_index is not None else 0
        colony_name = colonies[idx] if colonies and 0 <= idx < len(colonies) else (colonies[0] if colonies else '')
        return {'type': 'colony', 'colonyName': colony_name}
    elif input_type == 'delegate':
        players = waiting_for.get('players', [])
        idx = normalize_index(action_index, 740) if action_index is not None else 0
        player = players[idx] if players and 0 <= idx < len(players) else (players[0] if players else '')
        return {'type': 'delegate', 'player': player}
    elif input_type == 'option':
        options = waiting_for.get('options', [])
        if options:
            idx = normalize_index(action_index, 200) if action_index is not None else get_default_index(options)
            idx = max(0, min(idx, len(options) - 1))
            return {'type': 'option', 'index': int(idx)}
        return {'type': 'option'}
    elif input_type == 'party':
        parties = waiting_for.get('parties', [])
        idx = normalize_index(action_index, 730) if action_index is not None else 0
        party_name = parties[idx] if parties and 0 <= idx < len(parties) else (parties[0] if parties else '')
        return {'type': 'party', 'partyName': party_name}
    elif input_type == 'payment':
        payment_candidates = _enumerate_payment_candidates(
            waiting_for,
            player_state,
            max_candidates=_PAYMENT_ACTION_VARIANTS,
        )
        offset = _payment_action_offset(action_index)
        idx = max(0, min(offset, len(payment_candidates) - 1))
        payment = dict(payment_candidates[idx]) if payment_candidates else _payment_empty()

        player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
        _apply_project_specific_payment_constraints(
            payment,
            waiting_for,
            player,
            reserve_units=_merge_reserve_units(waiting_for),
        )
        return {'type': 'payment', 'payment': _finalize_payment(payment)}
    elif input_type == 'player':
        players = waiting_for.get('players', [])
        idx = normalize_index(action_index, 600) if action_index is not None else 0
        player = players[idx] if players and 0 <= idx < len(players) else (players[0] if players else '')
        return {'type': 'player', 'player': player}
    elif input_type == 'productionToLose':
        units = {k: 0 for k in ['megaCreditProduction', 'steelProduction', 'titaniumProduction', 'plantProduction', 'energyProduction', 'heatProduction']}
        return {'type': 'productionToLose', 'units': units}
    elif input_type == 'projectCard':
        # Terraforming Mars SelectProjectCardToPlayResponse does NOT accept {"type":"pass"}
        # - it always requires {type: 'projectCard', card, payment}. Never return pass.
        cards = waiting_for.get('cards', [])
        if not cards:
            return {'type': 'option'}

        # For projectCard, action_index should be the card index directly
        card_idx = normalize_index(action_index, 0) if action_index is not None else 0
        payment_options = waiting_for.get('paymentOptions', {})
        candidate_indices: List[int] = []

        if player_state:
            affordable_indices: List[int] = []
            for i, candidate in enumerate(cards):
                reserve_units = _merge_reserve_units(waiting_for, candidate)
                if _can_afford_card_with_payment_options(
                    player_state,
                    candidate,
                    payment_options,
                    reserve_units=reserve_units,
                    waiting_for=waiting_for,
                ):
                    affordable_indices.append(i)

            if affordable_indices:
                if card_idx in affordable_indices:
                    candidate_indices.append(card_idx)
                elif 0 <= card_idx < len(affordable_indices):
                    candidate_indices.append(int(affordable_indices[card_idx]))
                for i in affordable_indices:
                    if i not in candidate_indices:
                        candidate_indices.append(i)
            elif 0 <= card_idx < len(cards):
                candidate_indices.append(card_idx)
        else:
            if 0 <= card_idx < len(cards):
                candidate_indices.append(card_idx)
            for i in range(len(cards)):
                if i not in candidate_indices:
                    candidate_indices.append(i)

        for idx in candidate_indices:
            card = cards[idx]
            payment = _build_payment_with_options(
                player_state,
                card,
                payment_options,
                reserve_units=_merge_reserve_units(waiting_for, card),
                waiting_for=waiting_for,
            )
            if payment is not None:
                return {'type': 'projectCard', 'card': card['name'], 'payment': payment}

        # No valid payment candidate for this prompt/card state.
        return {'type': 'option'}
    elif input_type in ['space', 'selectSpace']:
        # Prefer explicit availableSpaces with IDs; fallback to 'spaces'
        spaces = waiting_for.get('availableSpaces') or waiting_for.get('spaces', [])
        # Accept both list of space dicts or list of ids/strings
        if not spaces:
            return {'type': 'pass'}
        idx = normalize_index(action_index, 300) if action_index is not None else 0
        if isinstance(spaces[0], dict):
            # Use chosen index when valid, otherwise fallback to first non-disabled.
            if 0 <= idx < len(spaces) and not spaces[idx].get('isDisabled', False):
                chosen = spaces[idx]
            else:
                chosen = next((s for s in spaces if not s.get('isDisabled', False)), spaces[0])
            return {'type': 'space', 'spaceId': str(chosen.get('id') or chosen.get('spaceId') or chosen)}
        else:
            # Already ids/strings
            if idx < 0 or idx >= len(spaces):
                idx = 0
            return {'type': 'space', 'spaceId': str(spaces[idx])}
    elif input_type == 'aresGlobalParameters':
        return {'type': 'aresGlobalParameters', 'response': {'lowOceanDelta': 0, 'highOceanDelta': 0, 'temperatureDelta': 0, 'oxygenDelta': 0}}
    elif input_type == 'globalEvent':
        events = waiting_for.get('globalEventNames', waiting_for.get('events', []))
        idx = normalize_index(action_index, 750) if action_index is not None else 0
        event_name = events[idx] if events and 0 <= idx < len(events) else (events[0] if events else '')
        return {'type': 'globalEvent', 'globalEventName': event_name}
    elif input_type == 'policy':
        policies = waiting_for.get('policies', [])
        idx = normalize_index(action_index, 840) if action_index is not None else 0
        policy_id = policies[idx] if policies and 0 <= idx < len(policies) else (policies[0] if policies else '')
        return {'type': 'policy', 'policyId': policy_id}
    elif input_type == 'resource':
        resources = waiting_for.get('include', [])
        idx = normalize_index(action_index, 820) if action_index is not None else 0
        resource = resources[idx] if resources and 0 <= idx < len(resources) else (resources[0] if resources else '')
        return {'type': 'resource', 'resource': resource}
    elif input_type == 'resources':
        units = {k: 0 for k in ['megaCredits', 'steel', 'titanium', 'plants', 'energy', 'heat']}
        return {'type': 'resources', 'units': units}
    elif input_type == 'claimedUndergroundToken':
        tokens = waiting_for.get('tokens', [])
        idx = normalize_index(action_index, 760) if action_index is not None else 0
        selected = [tokens[idx]] if tokens and 0 <= idx < len(tokens) else ([tokens[0]] if tokens else [])
        return {'type': 'claimedUndergroundToken', 'selected': selected}
    elif input_type == 'selectProjectCardToPlay':
        # Handle project card selection for playing.
        # Terraforming Mars SelectProjectCardToPlayResponse does NOT accept {"type":"pass"}
        # - it always requires {type: 'projectCard', card, payment}. Never return pass.
        cards = waiting_for.get('cards', [])
        if not cards:
            return {'type': 'option'}  # No cards - cannot produce valid response
             
        card_idx = normalize_index(action_index, 0) if action_index is not None else 0
        payment_options = waiting_for.get('paymentOptions', {}) if isinstance(waiting_for, dict) else {}
        candidate_indices: List[int] = []
        if player_state:
            affordable_indices: List[int] = []
            for i, candidate in enumerate(cards):
                reserve_units = _merge_reserve_units(waiting_for, candidate)
                if _can_afford_card_with_payment_options(
                    player_state,
                    candidate,
                    payment_options,
                    reserve_units=reserve_units,
                    waiting_for=waiting_for,
                ):
                    affordable_indices.append(i)

            if affordable_indices:
                if card_idx in affordable_indices:
                    candidate_indices.append(card_idx)
                elif 0 <= card_idx < len(affordable_indices):
                    candidate_indices.append(int(affordable_indices[card_idx]))
                for i in affordable_indices:
                    if i not in candidate_indices:
                        candidate_indices.append(i)
            elif 0 <= card_idx < len(cards):
                candidate_indices.append(card_idx)
        else:
            if 0 <= card_idx < len(cards):
                candidate_indices.append(card_idx)
            for i in range(len(cards)):
                if i not in candidate_indices:
                    candidate_indices.append(i)

        for idx in candidate_indices:
            card = cards[idx]
            payment = _build_payment_with_options(
                player_state,
                card,
                payment_options,
                reserve_units=_merge_reserve_units(waiting_for, card),
                waiting_for=waiting_for,
            )
            if payment is not None:
                return {'type': 'projectCard', 'card': card['name'], 'payment': payment}

        # No valid payment candidate for this prompt/card state.
        return {'type': 'option'}
    elif input_type == 'selectCard' and 'standard project' in _title_text(waiting_for.get('title', '')).lower():
        # Handle standard project selection
        cards = waiting_for.get('cards', [])
        if not cards:
            return {'type': 'pass'}
            
        card_idx = normalize_index(action_index, 100) if action_index is not None else 0
        enabled_cards = [(i, card) for i, card in enumerate(cards) if not card.get('isDisabled', False)]
        if not enabled_cards:
            return {'type': 'pass'}
            
        # Select the appropriate card based on action_index
        enabled_index_to_card = {i: card for i, card in enabled_cards}
        if card_idx in enabled_index_to_card:
            card = enabled_index_to_card[card_idx]
        elif 0 <= card_idx < len(enabled_cards):
            _, card = enabled_cards[card_idx]
        else:
            # Default to first available project
            _, card = enabled_cards[0]
        
        name = (card.get('name') or '')

        # Some servers accept selectCard form; others may want a projectCard with inline payment.
        # Prefer selectCard by default; fallback to projectCard if paymentOptions provided.
        if waiting_for.get('paymentOptions'):
            payment = _build_payment_with_options(
                player_state,
                card,
                waiting_for.get('paymentOptions', {}),
                reserve_units=_merge_reserve_units(waiting_for, card),
                waiting_for=waiting_for,
            )
            if payment is None:
                return {'type': 'pass'}
            return {'type': 'projectCard', 'card': name, 'payment': payment}
        return {'type': 'card', 'cards': [name]}
    elif input_type == 'selectCard' and 'convert plants' in _title_text(waiting_for.get('title', '')).lower():
        # Handle convert plants action
        return {'type': 'convertPlants'}
    elif input_type == 'selectCard' and 'convert heat' in _title_text(waiting_for.get('title', '')).lower():
        # Handle convert heat action
        return {'type': 'convertHeat'}
    elif input_type == 'selectCard' and 'sell patents' in _title_text(waiting_for.get('title', '')).lower():
        # Handle sell patents action
        cards = waiting_for.get('cards', [])
        if not cards:
            return {'type': 'pass'}
        selected_names = _select_patents_to_sell(cards, waiting_for, player_state, action_index)
        if not selected_names:
            return {'type': 'pass'}
        return {'type': 'card', 'cards': selected_names}
    else:
        # Fallback: return a pass/option action
        return {'type': 'option'}

# --- End Action Response Builder ---

def _calculate_card_payment(player_state: Dict[str, Any], card: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Calculate optimal payment for a card based on player resources"""
    player = player_state.get('thisPlayer', {})
    player_mc = player.get('megaCredits', 0)
    player_steel = player.get('steel', 0)
    player_titanium = player.get('titanium', 0)
    player_heat = player.get('heat', 0)
    player_plants = player.get('plants', 0)
    steel_value = player.get('steelValue', 2)
    titanium_value = player.get('titaniumValue', 3)
    corporation = player.get('corporation', {})
    corp_name = corporation.get('name', '') if isinstance(corporation, dict) else str(corporation)

    # Get card cost
    cost = card.get('calculatedCost', card.get('cost', 0))
    tags = card.get('tags', {}) or _metadata_tags(card.get('name', ''))
    if isinstance(tags, list):
        tags = {_normalize_tag_name(t): 1 for t in tags if _normalize_tag_name(t)}
    elif not isinstance(tags, dict):
        tags = {}
    card_name_l = str(card.get('name', '') or '').lower()
    card_type_l = str(card.get('type', '') or '').lower()

    payment = {
        'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0
    }
    
    cost_remaining = cost

    # Use steel for Building tags (floor units first; top-up handled later if needed)
    if tags.get('Building') and player_steel > 0:
        usable_steel_value = player_steel * steel_value
        steel_to_pay = min(cost_remaining, usable_steel_value)
        # Use ceiling division to spend MORE steel (not less)
        steel_units = min(player_steel, (steel_to_pay + steel_value - 1) // steel_value)
        payment['steel'] = steel_units
        cost_remaining -= steel_units * steel_value

    # Use titanium for Space tags (floor units first; top-up handled later if needed)
    if tags.get('Space') and player_titanium > 0:
        usable_titanium_value = player_titanium * titanium_value
        titanium_to_pay = min(cost_remaining, usable_titanium_value)
        # Use ceiling division to spend MORE titanium (not less)
        titanium_units = min(player_titanium, (titanium_to_pay + titanium_value - 1) // titanium_value)
        payment['titanium'] = titanium_units
        cost_remaining -= titanium_units * titanium_value

    # Helion: allow using heat as money
    if 'helion' in corp_name.lower() and player_heat > 0 and cost_remaining > 0:
        heat_to_pay = min(cost_remaining, player_heat)
        payment['heat'] = int(heat_to_pay)
        cost_remaining -= heat_to_pay

    # Pay remainder with MegaCredits first
    if cost_remaining > 0:
        pay_mc = min(cost_remaining, player_mc)
        payment['megaCredits'] = int(pay_mc)
        cost_remaining -= pay_mc

    # If still short, top-up with extra steel/titanium units (legal overpay).
    if cost_remaining > 0:
        remaining_steel = max(0, int(player_steel - payment.get('steel', 0)))
        remaining_titanium = max(0, int(player_titanium - payment.get('titanium', 0)))
        best = None
        for add_steel in range(remaining_steel + 1):
            steel_value_paid = add_steel * max(1, int(steel_value))
            for add_titanium in range(remaining_titanium + 1):
                value_paid = steel_value_paid + (add_titanium * max(1, int(titanium_value)))
                if value_paid < cost_remaining:
                    continue
                units = add_steel + add_titanium
                overpay = value_paid - cost_remaining
                score = (overpay, units)
                if best is None or score < best[0]:
                    best = (score, add_steel, add_titanium, value_paid)
        if best is not None:
            _, add_steel, add_titanium, value_paid = best
            payment['steel'] = int(payment.get('steel', 0) + add_steel)
            payment['titanium'] = int(payment.get('titanium', 0) + add_titanium)
            cost_remaining -= int(value_paid)

    # Still short means unaffordable with legal resources.
    if cost_remaining > 0:
        return None
    
    # Ensure all values are integers
    for k in payment:
        payment[k] = int(payment[k])
        
    return payment

def _build_payment_with_options(
    player_state: Dict[str, Any],
    card: Dict[str, Any],
    payment_options: Dict[str, Any],
    reserve_units: Optional[Dict[str, int]] = None,
    waiting_for: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, int]]:
    """Build a payment dict honoring paymentOptions from waiting_for."""
    if player_state is None:
        player_state = {}
    player = player_state.get('thisPlayer', {})
    reserve = dict(_payment_empty())
    if reserve_units:
        for key in _PAYMENT_ALL_KEYS:
            reserve[key] = max(0, int(reserve_units.get(key, 0) or 0))

    player_mc = max(0, int(player.get('megaCredits', 0) or 0) - int(reserve.get('megaCredits', 0) or 0))
    player_steel = max(0, int(player.get('steel', 0) or 0) - int(reserve.get('steel', 0) or 0))
    player_titanium = max(0, int(player.get('titanium', 0) or 0) - int(reserve.get('titanium', 0) or 0))
    player_heat = max(0, int(player.get('heat', 0) or 0) - int(reserve.get('heat', 0) or 0))
    player_plants = max(0, int(player.get('plants', 0) or 0) - int(reserve.get('plants', 0) or 0))
    steel_value = max(1, int(player.get('steelValue', 2) or 2))
    titanium_value = max(1, int(player.get('titaniumValue', 3) or 3))

    cost = int(card.get('calculatedCost', card.get('cost', 0)) or 0)
    tags = card.get('tags', {}) or _metadata_tags(card.get('name', ''))
    if isinstance(tags, list):
        tags = {_normalize_tag_name(t): 1 for t in tags if _normalize_tag_name(t)}
    elif not isinstance(tags, dict):
        tags = {}
    card_name_l = str(card.get('name', '') or '').lower()
    card_type_l = str(card.get('type', '') or '').lower()

    # Stratospheric Birds requires spending a floater from a card as an extra
    # condition; keep one floater reserved even when reserveUnits is absent.
    if 'stratospheric birds' in card_name_l:
        reserve['floaters'] = max(int(reserve.get('floaters', 0) or 0), 1)

    # Default zeroes for all known keys the server may accept
    payment = {
        'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0,
        'microbes': 0, 'floaters': 0, 'lunaArchivesScience': 0, 'spireScience': 0,
        'seeds': 0, 'auroraiData': 0, 'graphene': 0, 'kuiperAsteroids': 0
    }

    def default_allowed(resource: str) -> bool:
        if resource == 'steel':
            return bool(tags.get('Building'))
        if resource == 'titanium':
            return bool(tags.get('Space'))
        if resource in ['microbes', 'seeds']:
            return bool(tags.get('Plant'))
        if resource == 'floaters':
            return bool(tags.get('Venus'))
        if resource == 'lunaArchivesScience':
            return bool(tags.get('Moon'))
        if resource == 'graphene':
            return bool(tags.get('City') or tags.get('Space'))
        if resource in ['spireScience', 'auroraiData']:
            return ('standard' in card_type_l and 'project' in card_type_l) or card_name_l.endswith(':sp')
        if resource == 'kuiperAsteroids':
            return ('aquifer' in card_name_l) or ('asteroid' in card_name_l)
        return False

    def allowed(resource: str) -> bool:
        if resource == 'megaCredits':
            return True
        if resource == 'titanium' and payment_options.get('lunaTradeFederationTitanium', False):
            return True
        if resource in payment_options:
            return bool(payment_options.get(resource, False))
        return default_allowed(resource)

    # Luna Trade Federation: titanium spent as MC loses one value point.
    if payment_options.get('lunaTradeFederationTitanium', False) and not payment_options.get('titanium', False):
        titanium_value = max(1, int(titanium_value) - 1)

    cost_remaining = cost

    can_use_steel = allowed('steel')
    can_use_titanium = allowed('titanium')

    # Use steel if allowed (floor units first; top-up handled later)
    if can_use_steel and player_steel > 0:
        usable_steel_value = player_steel * steel_value
        steel_to_pay = min(cost_remaining, usable_steel_value)
        # Use ceiling division to spend MORE steel
        steel_units = min(player_steel, (steel_to_pay + steel_value - 1) // steel_value)
        payment['steel'] = int(steel_units)
        cost_remaining -= steel_units * steel_value

    # Use titanium if allowed (LTF may allow this even for non-space cards).
    if can_use_titanium and player_titanium > 0:
        usable_titanium_value = player_titanium * titanium_value
        titanium_to_pay = min(cost_remaining, usable_titanium_value)
        # Use ceiling division to spend MORE titanium
        titanium_units = min(player_titanium, (titanium_to_pay + titanium_value - 1) // titanium_value)
        payment['titanium'] = int(titanium_units)
        cost_remaining -= titanium_units * titanium_value

    # Use heat if allowed (e.g., Helion or general payment allowed)
    if allowed('heat') and player_heat > 0 and cost_remaining > 0:
        heat_to_pay = min(cost_remaining, player_heat)
        payment['heat'] = int(heat_to_pay)
        cost_remaining -= heat_to_pay

    # Use plants if allowed
    if allowed('plants') and player_plants > 0 and cost_remaining > 0:
        plants_to_pay = min(cost_remaining, player_plants)
        payment['plants'] = int(plants_to_pay)
        cost_remaining -= plants_to_pay

    # Pay remainder with MC if allowed
    if allowed('megaCredits') and cost_remaining > 0:
        payment['megaCredits'] = int(min(cost_remaining, player_mc))
        cost_remaining -= payment['megaCredits']

    # If still short, top-up with extra steel/titanium units (legal overpay).
    if cost_remaining > 0:
        topup_resources: List[Tuple[str, int, int]] = []
        if can_use_steel:
            remaining_steel = max(0, int(player_steel - payment.get('steel', 0)))
            if remaining_steel > 0:
                topup_resources.append(('steel', remaining_steel, max(1, int(steel_value))))
        if can_use_titanium:
            remaining_titanium = max(0, int(player_titanium - payment.get('titanium', 0)))
            if remaining_titanium > 0:
                topup_resources.append(('titanium', remaining_titanium, max(1, int(titanium_value))))

        if topup_resources:
            best = None
            if len(topup_resources) == 1:
                name, max_units, unit_value = topup_resources[0]
                for units in range(0, max_units + 1):
                    value_paid = units * unit_value
                    if value_paid < cost_remaining:
                        continue
                    overpay = value_paid - cost_remaining
                    score = (overpay, units)
                    if best is None or score < best[0]:
                        best = (score, {name: units}, value_paid)
            else:
                (name_a, max_a, value_a), (name_b, max_b, value_b) = topup_resources[:2]
                for units_a in range(0, max_a + 1):
                    value_paid_a = units_a * value_a
                    for units_b in range(0, max_b + 1):
                        value_paid = value_paid_a + (units_b * value_b)
                        if value_paid < cost_remaining:
                            continue
                        overpay = value_paid - cost_remaining
                        score = (overpay, units_a + units_b)
                        if best is None or score < best[0]:
                            best = (score, {name_a: units_a, name_b: units_b}, value_paid)

            if best is not None:
                _, units_map, value_paid = best
                for key, add_units in units_map.items():
                    payment[key] = int(payment.get(key, 0) + int(add_units))
                cost_remaining -= int(value_paid)

    if cost_remaining > 0:
        # Fallback: reuse the payment candidate enumerator with inferred per-card options.
        waiting_payload: Dict[str, Any] = {}
        if isinstance(waiting_for, dict):
            waiting_payload = dict(waiting_for)
        waiting_payload['amount'] = int(cost)

        effective_options = {}
        if isinstance(waiting_payload.get('paymentOptions'), dict):
            effective_options.update(waiting_payload.get('paymentOptions', {}))
        if isinstance(payment_options, dict):
            effective_options.update(payment_options)
        luna_titanium_mode = bool(payment_options.get('lunaTradeFederationTitanium', False)) if isinstance(payment_options, dict) else False

        for resource in _PAYMENT_ALL_KEYS:
            if resource == 'megaCredits':
                continue
            if (
                resource == 'titanium'
                and luna_titanium_mode
                and 'titanium' not in effective_options
            ):
                # Keep titanium unset in LTF mode so payment valuation applies -1 correctly.
                continue
            if resource not in effective_options:
                effective_options[resource] = bool(allowed(resource))
        if isinstance(payment_options, dict) and 'lunaTradeFederationTitanium' in payment_options:
            effective_options['lunaTradeFederationTitanium'] = bool(payment_options.get('lunaTradeFederationTitanium', False))

        waiting_payload['paymentOptions'] = effective_options

        reserve_payload = dict(waiting_payload.get('reserveUnits', {})) if isinstance(waiting_payload.get('reserveUnits', {}), dict) else {}
        for key in _PAYMENT_ALL_KEYS:
            reserve_payload[key] = max(0, int(reserve.get(key, 0) or 0))
        waiting_payload['reserveUnits'] = reserve_payload

        candidate = _enumerate_payment_candidates(
            waiting_payload,
            player_state,
            max_candidates=1,
        )
        if candidate:
            return _finalize_payment(candidate[0])
        return None

    return payment

def _can_afford_card_with_payment_options(
    player_state: Optional[Dict[str, Any]],
    card: Dict[str, Any],
    payment_options: Optional[Dict[str, Any]],
    reserve_units: Optional[Dict[str, int]] = None,
    waiting_for: Optional[Dict[str, Any]] = None,
) -> bool:
    payment = _build_payment_with_options(
        player_state or {},
        card,
        payment_options or {},
        reserve_units=reserve_units,
        waiting_for=waiting_for,
    )
    return payment is not None

# --- Metadata helpers for tags when missing ---
def _metadata_candidate_paths() -> List[str]:
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

    resolved: List[str] = []
    seen = set()
    for candidate in candidate_paths:
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved

def _metadata_loader() -> Dict[str, Dict[str, Any]]:
    fallback_data: Optional[Dict[str, Dict[str, Any]]] = None
    fallback_path: Optional[str] = None
    for path in _metadata_candidate_paths():
        if not os.path.exists(path):
            continue
        if os.path.getsize(path) <= 0:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                has_corp_economics = any(
                    isinstance(meta, dict) and (
                        meta.get('startingMegaCredits', None) is not None
                        or meta.get('cardCost', None) is not None
                    )
                    for meta in data.values()
                )
                if has_corp_economics:
                    logger.info("Loaded card metadata for %d cards from %s", len(data), path)
                    return data
                if fallback_data is None:
                    fallback_data = data
                    fallback_path = path
                    logger.info(
                        "Loaded metadata from %s without corporation economics; continuing search",
                        path,
                    )
        except Exception as e:
            logger.warning("Failed to load card metadata from %s: %s", path, e)
    if fallback_data:
        logger.info(
            "Using fallback card metadata for %d cards from %s",
            len(fallback_data),
            fallback_path or "unknown path",
        )
        return fallback_data
    return {}

_CARD_META_CACHE: Dict[str, Dict[str, Any]] = _metadata_loader()

def _metadata_tags(card_name: str) -> Dict[str, int]:
    if not card_name or not _CARD_META_CACHE:
        return {}
    meta = _CARD_META_CACHE.get(card_name)
    if not meta:
        return {}
    out: Dict[str, int] = {}
    tags = meta.get('tags', []) or []
    if isinstance(tags, dict):
        for key, present in tags.items():
            if not present:
                continue
            ts = str(key).strip()
            if ts and ts[0].islower():
                ts = ts.capitalize()
            if ts:
                out[ts] = 1
        return out
    for t in tags:
        ts = str(t).strip()
        if ts and ts[0].islower():
            ts = ts.capitalize()
        if ts:
            out[ts] = 1
    return out
def _can_afford_card(player: Dict[str, Any], card: Dict[str, Any]) -> bool:
    payload_card = {
        'name': str(card.get('name', '') or ''),
        'calculatedCost': float(_card_cost(card)),
        'cost': float(_card_cost(card)),
        'tags': dict(_card_tags(card)),
        'victoryPoints': float(_card_vp(card)),
    }
    return bool(_can_afford_cards_batch(player, [payload_card])[0])

class ActionDecoder:
    def __init__(self, planner_config: Optional[PlannerConfig] = None):
        self.planner_config = planner_config or PlannerConfig()
        # Action space mapping
        self.action_types = {
            'PLAY_CARD': 0,
            'STANDARD_PROJECT': 100,
            'SELECT_OPTION': 200,
            'SELECT_CARD_MASK': _CARD_SELECTION_MASK_BASE,
            'STARTUP_PLAN': _STARTUP_PLAN_BASE,
            'SELECT_SPACE': 300,
            'SELECT_PAYMENT': _PAYMENT_ACTION_BASE,
            'SELECT_AMOUNT': 500,
            'PASS': 900,
            'END_TURN': 950
        }
        
        # Performance caches
        self._startup_plan_cache = []
        self._startup_plan_cache_state_id = None
        
        # Standard projects mapping - use actual game API names (card_metadata)
        self.standard_projects = [
            'Power Plant:SP', 'Asteroid:SP', 'Aquifer', 'Greenery', 'City', 'Colony',
            'Air Scrapping', 'Buffer Gas',
            'Road Infrastructure', 'Lunar Mine', 'Lunar Habitat',  # Moon expansion
            'Road Infrastructure (var. 1)', 'Lunar Mine (var. 1)', 'Lunar Habitat (var. 1)',
            'Road Infrastructure (var. 2)', 'Lunar Mine (var. 2)', 'Lunar Habitat (var. 2)',
        ]
        self.card_selection_mask_limit = _CARD_SELECTION_MASK_LIMIT
        self.startup_plan_limit = _STARTUP_PLAN_LIMIT

    def _option_payload_for_action(self, action_index: int, waiting_for: Dict[str, Any]) -> Dict[str, Any]:
        input_type = str(waiting_for.get('type', '') or '').lower()
        options = waiting_for.get('options', []) or []
        if input_type == 'or':
            option_idx = int(action_index) - int(self.action_types['SELECT_OPTION'])
            if 0 <= option_idx < len(options):
                option = options[option_idx]
                if isinstance(option, dict):
                    return option
        if input_type in ('option', 'selectoption'):
            option_idx = int(action_index) - int(self.action_types['SELECT_OPTION'])
            if 0 <= option_idx < len(options):
                option = options[option_idx]
                if isinstance(option, dict):
                    return option
        if input_type == 'or':
            for option in options:
                if not isinstance(option, dict):
                    continue
                option_type = str(option.get('type', '') or '')
                option_title_l = _title_text(option.get('title', '')).lower()
                if option_type == 'or' and 'fund an award' in option_title_l and 600 <= int(action_index) < 700:
                    award_idx = int(action_index) - 600
                    award_options = option.get('options', []) or []
                    if 0 <= award_idx < len(award_options):
                        award_option = award_options[award_idx]
                        if isinstance(award_option, dict):
                            return award_option
        return {}

    def _semantic_family(
        self,
        action_index: int,
        waiting_for: Dict[str, Any],
        decoded_action: Optional[Dict[str, Any]],
    ) -> str:
        input_type = str(waiting_for.get('type', '') or '').lower()
        title_l = _title_text(waiting_for.get('title', '')).lower()
        button_l = str(waiting_for.get('buttonLabel', '') or '').lower()
        combined = f"{title_l} {button_l}".strip()
        option_payload = self._option_payload_for_action(action_index, waiting_for)
        option_title_l = _title_text(option_payload.get('title', '')).lower()
        nested_option_titles = [
            _title_text(option.get('title', '')).lower()
            for option in (waiting_for.get('options', []) or [])
            if isinstance(option, dict)
        ]
        has_award_branch = any('fund an award' in title for title in nested_option_titles)
        has_convert_plants_branch = any('convert plants' in title for title in nested_option_titles)
        has_convert_heat_branch = any('convert heat' in title for title in nested_option_titles)
        has_sell_patents_branch = any('sell patents' in title for title in nested_option_titles)
        if int(action_index) == 700 and ('convert plants' in combined or has_convert_plants_branch):
            return 'convert_plants'
        if int(action_index) == 701 and ('convert heat' in combined or has_convert_heat_branch):
            return 'convert_heat'
        if int(action_index) == 702 and ('sell patents' in combined or has_sell_patents_branch):
            return 'sell_patents'
        if int(action_index) >= int(self.action_types['PASS']):
            return 'pass'
        if 0 <= int(action_index) < int(self.action_types['STANDARD_PROJECT']):
            return 'play_card'
        if int(self.action_types['STANDARD_PROJECT']) <= int(action_index) < int(self.action_types['SELECT_OPTION']):
            return 'standard_project'
        if int(self.action_types['SELECT_SPACE']) <= int(action_index) < int(self.action_types['SELECT_PAYMENT']):
            return 'select_space'
        if int(self.action_types['SELECT_PAYMENT']) <= int(action_index) < int(self.action_types['SELECT_AMOUNT']):
            return 'select_payment'
        if int(self.action_types['SELECT_AMOUNT']) <= int(action_index) < int(_CARD_SELECTION_MASK_BASE):
            return 'select_amount'
        # Award actions use the 600+ range, which overlaps the generic card-mask
        # range. Classify semantic prompts before applying that numeric fallback.
        if ('award' in combined and ('fund' in combined or 'award' in input_type)) or (
            600 <= int(action_index) < 700 and has_award_branch
        ):
            return 'fund_award'
        if 'milestone' in combined and ('claim' in combined or 'fund' in combined):
            return 'claim_milestone'
        if 'award' in option_title_l:
            return 'fund_award'
        if 'milestone' in option_title_l:
            return 'claim_milestone'
        if int(_CARD_SELECTION_MASK_BASE) <= int(action_index) < int(_CARD_SELECTION_MASK_BASE + _CARD_SELECTION_MASK_LIMIT):
            return 'card_subset'
        if int(_STARTUP_PLAN_BASE) <= int(action_index) < int(_STARTUP_PLAN_BASE + _STARTUP_PLAN_LIMIT):
            return 'startup_plan'
        if input_type in ('space', 'selectspace'):
            return 'select_space'
        if input_type in ('projectcard', 'selectprojectcardtoplay'):
            return 'play_card'
        if input_type in ('card', 'selectcard'):
            return 'card_prompt'
        if input_type in ('or', 'option', 'selectoption'):
            return 'select_option'
        if input_type in (
            'selectplayer', 'player', 'selectresources', 'resources',
            'selectproductiontolose', 'productiontolose', 'selectcolony', 'colony',
            'selectparty', 'party', 'selectdelegate', 'delegate',
        ):
            return 'select_option'
        if isinstance(decoded_action, dict):
            decoded_type = str(decoded_action.get('type', '') or '').lower()
            if decoded_type == 'space':
                return 'select_space'
            if decoded_type in ('card', 'projectcard'):
                return 'play_card'
        return 'other'

    def _descriptor_labels(
        self,
        action_index: int,
        waiting_for: Dict[str, Any],
        family: str,
    ) -> Dict[str, str]:
        title_l = _title_text(waiting_for.get('title', ''))
        payload = self._option_payload_for_action(action_index, waiting_for)
        label = _title_text(payload.get('title', '')) or title_l
        award_name = ''
        milestone_name = ''
        card_name = ''
        project_name = ''
        if family == 'play_card':
            cards = waiting_for.get('cards', []) or []
            card_idx = int(action_index)
            if str(waiting_for.get('type', '') or '').lower() in ('projectcard', 'selectprojectcardtoplay'):
                card_idx = int(action_index)
            if 0 <= card_idx < 100:
                if 0 <= card_idx < len(cards):
                    card_name = str((cards[card_idx] or {}).get('name', '') or '')
            label = card_name or label or 'Play project card'
        elif family == 'standard_project':
            project_idx = int(action_index) - int(self.action_types['STANDARD_PROJECT'])
            if 0 <= project_idx < len(self.standard_projects):
                project_name = str(self.standard_projects[project_idx] or '')
            label = project_name or label or 'Standard project'
        elif family == 'fund_award':
            award_name = _title_text(payload.get('title', '')) or _title_text(payload.get('name', ''))
            label = award_name or label or 'Fund award'
        elif family == 'claim_milestone':
            milestone_name = _title_text(payload.get('title', '')) or _title_text(payload.get('name', ''))
            label = milestone_name or label or 'Claim milestone'
        elif family == 'select_space':
            label = title_l or 'Select space'
        return {
            "label": str(label or family).strip(),
            "award_name": str(award_name or '').strip(),
            "milestone_name": str(milestone_name or '').strip(),
            "card_name": str(card_name or '').strip(),
            "project_name": str(project_name or '').strip(),
        }

    def _build_action_token(
        self,
        player_state: Dict[str, Any],
        action_index: int,
        family: str,
        label_info: Dict[str, str],
        decoded_action: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        player = player_state.get('thisPlayer', {}) or {}
        game = player_state.get('game', {}) or {}
        title_l = _title_text((player_state.get('waitingFor', {}) or {}).get('title', '')).lower()
        generation = max(1.0, float(game.get('generation', 1) or 1))
        plants = max(0.0, float(player.get('plants', 0) or 0))
        heat = max(0.0, float(player.get('heat', 0) or 0))
        plant_prod = max(0.0, float(player.get('plantProduction', 0) or 0))
        heat_prod = max(0.0, float(player.get('heatProduction', 0) or 0))
        mc = max(0.0, float(player.get('megaCredits', 0) or 0))
        vp = max(0.0, float(((player.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0)))

        family_order = [
            'play_card', 'standard_project', 'select_option', 'select_space',
            'fund_award', 'claim_milestone', 'convert_plants', 'convert_heat',
            'select_payment', 'select_amount', 'card_subset', 'startup_plan',
            'sell_patents', 'pass', 'other',
        ]
        features: List[float] = [1.0 if family == name else 0.0 for name in family_order]
        features.extend([
            min(generation / 14.0, 1.0),
            min(mc / 80.0, 1.0),
            min(plants / 16.0, 1.0),
            min(heat / 16.0, 1.0),
            min(plant_prod / 8.0, 1.0),
            min(heat_prod / 8.0, 1.0),
            min(vp / 60.0, 1.0),
        ])

        card_name = label_info.get("card_name", "")
        project_name = label_info.get("project_name", "")
        award_name = label_info.get("award_name", "")
        milestone_name = label_info.get("milestone_name", "")

        cost_norm = 0.0
        vp_norm = 0.0
        building_tag = 0.0
        space_tag = 0.0
        science_tag = 0.0
        plant_tag = 0.0
        city_project = 0.0
        greenery_project = 0.0
        moon_project = 0.0
        if card_name:
            waiting_for = player_state.get('waitingFor', {}) or {}
            cards = waiting_for.get('cards', []) or []
            for card in cards:
                if not isinstance(card, dict) or str(card.get('name', '') or '') != card_name:
                    continue
                cost_norm = min(float(_card_cost(card)) / 40.0, 1.0)
                vp_norm = min(float(_card_vp(card)) / 5.0, 1.0)
                tags = _card_tags(card)
                building_tag = 1.0 if tags.get('Building', 0) > 0 else 0.0
                space_tag = 1.0 if tags.get('Space', 0) > 0 else 0.0
                science_tag = 1.0 if tags.get('Science', 0) > 0 else 0.0
                plant_tag = 1.0 if tags.get('Plant', 0) > 0 else 0.0
                moon_project = 1.0 if tags.get('Moon', 0) > 0 else 0.0
                break
        if project_name:
            name_l = project_name.lower()
            cost_lookup = {
                'greenery': 23.0,
                'city': 25.0,
                'aquifer': 18.0,
                'power plant': 11.0,
                'asteroid': 14.0,
                'road infrastructure': 18.0,
                'lunar mine': 18.0,
                'lunar habitat': 18.0,
                'colony': 17.0,
            }
            for key, cost in cost_lookup.items():
                if key in name_l:
                    cost_norm = min(float(cost) / 40.0, 1.0)
                    break
            greenery_project = 1.0 if 'greenery' in name_l else 0.0
            city_project = 1.0 if 'city' in name_l or 'habitat' in name_l else 0.0
            moon_project = 1.0 if 'lunar' in name_l or 'road infrastructure' in name_l else moon_project

        spend_threshold_resource = 1.0 if family in ('convert_plants', 'convert_heat', 'standard_project') else 0.0
        carry_plants = 1.0 if family in ('pass', 'play_card', 'claim_milestone', 'fund_award') and 7.0 <= plants < 8.0 and plant_prod > 0.0 else 0.0
        carry_heat = 1.0 if family in ('pass', 'play_card', 'claim_milestone', 'fund_award') and 7.0 <= heat < 8.0 and heat_prod > 0.0 else 0.0
        if family == 'convert_plants' and 7.0 <= plants < 16.0 and plant_prod > 0.0:
            carry_plants = -1.0
        if family == 'convert_heat' and 7.0 <= heat < 16.0 and heat_prod > 0.0:
            carry_heat = -1.0
        combo_city_anchor = 1.0 if city_project > 0.0 or 'city' in title_l else 0.0
        combo_greenery_followup = 1.0 if greenery_project > 0.0 or family == 'convert_plants' else 0.0
        raises_claimability = 1.0 if family == 'claim_milestone' else 0.0
        locks_award_lead = 1.0 if family == 'fund_award' else 0.0
        immediate_board = 1.0 if family in ('select_space', 'standard_project', 'convert_plants') else 0.0

        features.extend([
            cost_norm,
            vp_norm,
            building_tag,
            space_tag,
            science_tag,
            plant_tag,
            city_project,
            greenery_project,
            moon_project,
            1.0 if award_name else 0.0,
            1.0 if milestone_name else 0.0,
            spend_threshold_resource,
            carry_plants,
            carry_heat,
            combo_city_anchor,
            combo_greenery_followup,
            raises_claimability,
            locks_award_lead,
            immediate_board,
        ])

        label_l = str(label_info.get("label", '') or '').lower()
        features.extend([
            1.0 if 'greenery' in label_l else 0.0,
            1.0 if 'city' in label_l else 0.0,
            1.0 if 'road' in label_l else 0.0,
            1.0 if 'mine' in label_l else 0.0,
            1.0 if 'habitat' in label_l else 0.0,
            1.0 if 'award' in label_l else 0.0,
            1.0 if 'milestone' in label_l else 0.0,
            1.0 if 'pass' in label_l else 0.0,
        ])
        return token_from_features(type_id=8, features=features, planner_config=self.planner_config)

    def _build_action_descriptor(
        self,
        action_index: int,
        action_position: int,
        player_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        waiting_for = player_state.get('waitingFor', {}) or {}
        decoded_action = self.decode_action(action_index, player_state)
        family = self._semantic_family(action_index, waiting_for, decoded_action)
        label_info = self._descriptor_labels(action_index, waiting_for, family)
        return {
            "action_index": int(action_index),
            "action_position": int(action_position),
            "family": family,
            "label": label_info.get("label", family),
            "decoded_action": decoded_action,
            "card_name": label_info.get("card_name", ""),
            "project_name": label_info.get("project_name", ""),
            "award_name": label_info.get("award_name", ""),
            "milestone_name": label_info.get("milestone_name", ""),
            "token_features": self._build_action_token(
                player_state=player_state,
                action_index=action_index,
                family=family,
                label_info=label_info,
                decoded_action=decoded_action,
            ).astype(np.float32),
        }

    def get_legal_action_descriptors(self, player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        descriptors: List[Dict[str, Any]] = []
        for action_position, action_index in enumerate(self.get_available_actions(player_state)):
            descriptors.append(self._build_action_descriptor(int(action_index), int(action_position), player_state))
        return descriptors

    def _is_convert_heat_wasteful(self, player_state: Dict[str, Any]) -> bool:
        """Check if converting heat to temperature is wasteful"""
        if not player_state:
            return False
        game = player_state.get('game', {})
        temp = game.get('temperature', -30)
        return temp >= 8

    def _is_standalone_global_option(
        self,
        title_l: str,
        prefixes: Tuple[str, ...],
    ) -> bool:
        """Detect simple option titles that only advance a single global parameter."""
        normalized = ' '.join(str(title_l or '').strip().lower().split())
        if not normalized:
            return False
        for prefix in prefixes:
            if not normalized.startswith(prefix):
                continue
            remainder = normalized[len(prefix):].strip()
            if not remainder:
                return True
            tokens = remainder.replace('+', ' ').split()
            if tokens and all(token in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'step', 'steps'} for token in tokens):
                return True
        return False

    def _is_option_wasteful(self, option_payload: Dict[str, Any], player_state: Dict[str, Any]) -> bool:
        """Mask obviously dead endgame branches across option-like prompts."""
        if not player_state or not isinstance(option_payload, dict):
            return False

        game = player_state.get('game', {}) or {}
        title_l = _title_text(option_payload.get('title', '')).strip().lower()
        if not title_l:
            return False

        if self._is_convert_heat_wasteful(player_state):
            if 'convert heat' in title_l or ('8 heat' in title_l and 'temperature' in title_l):
                return True
            if self._is_standalone_global_option(title_l, ('increase temperature', 'raise temperature', 'raise the temperature')):
                return True

        oxygen_level = _safe_int(game.get('oxygenLevel', game.get('oxygen', 0)), 0)
        if oxygen_level >= 14 and self._is_standalone_global_option(
            title_l,
            ('increase oxygen', 'raise oxygen', 'raise the oxygen'),
        ):
            return True

        if _safe_int(game.get('oceans', 0), 0) >= 9 and self._is_standalone_global_option(
            title_l,
            ('add an ocean', 'place an ocean', 'place ocean'),
        ):
            return True

        if _safe_int(game.get('venusScaleLevel', 0), 0) >= 30 and self._is_standalone_global_option(
            title_l,
            ('increase venus', 'raise venus', 'raise the venus'),
        ):
            return True

        return False

    def build_initial_setup_response(self, player_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        input_type = waiting_for.get('type', '')
        if input_type not in ['initialCards', 'selectInitialCards']:
            return None
        return _build_initial_setup_response(waiting_for, player_state)

    def _get_cached_startup_plans(self, waiting_for: Dict[str, Any], player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Manage caching of startup plans to avoid redundant O(N^3) calculations."""
        # Use player ID and step/gameAge as a rough state identifier
        player = player_state.get('thisPlayer', {})
        game = player_state.get('game', {})
        state_id = (
            game.get('id'),
            player.get('id'),
            game.get('step'),
            game.get('gameAge'),
        )
        
        if self._startup_plan_cache_state_id == state_id:
            return self._startup_plan_cache
        
        plans = _enumerate_startup_plan_payloads(
            waiting_for=waiting_for,
            player_state=player_state,
            max_plans=_STARTUP_PLAN_LIMIT,
        )
        
        self._startup_plan_cache = plans
        self._startup_plan_cache_state_id = state_id
        return plans
    
    def get_available_actions(self, player_state: Dict[str, Any]) -> List[int]:
        """Get list of available action indices for current game state"""
        available_actions = []
        try:
            waiting_for = player_state.get('waitingFor')
            if not waiting_for:
                return []
            input_type = waiting_for.get('type', '')
            
            # Handle all known input types
            if input_type == 'or':
                options = waiting_for.get('options', [])
                player = player_state.get('thisPlayer', {}) if player_state else {}
                for i, option in enumerate(options):
                    option_type = option.get('type', '')
                    option_title_l = _title_text(option.get('title', '')).lower()
                    allow_select_option = True
                    added_concrete_action = False
                    option_is_wasteful = self._is_option_wasteful(option, player_state)

                    if option_type in ['selectProjectCardToPlay', 'projectCard']:
                        # Prefer concrete card actions instead of generic OR selection.
                        cards = option.get('cards', [])
                        payment_options = option.get('paymentOptions', {})
                        affordable_indices = []
                        for j, card in enumerate(cards):
                            reserve_units = _merge_reserve_units(option, card)
                            if player_state and _can_afford_card_with_payment_options(
                                player_state,
                                card,
                                payment_options,
                                reserve_units=reserve_units,
                                waiting_for=option,
                            ):
                                affordable_indices.append(j)
                        candidate_indices = affordable_indices
                        if candidate_indices:
                            for j in candidate_indices:
                                available_actions.append(self.action_types['PLAY_CARD'] + j)
                            added_concrete_action = True
                        else:
                            # Avoid surfacing known-unaffordable project-card branches.
                            allow_select_option = False
                    elif option_type in ['selectCard', 'card'] and 'standard project' in option_title_l:
                        cards = option.get('cards', [])
                        enabled_indices = [
                            j for j, card in enumerate(cards) 
                            if not card.get('isDisabled', False)
                        ]
                        if enabled_indices:
                            for j in enabled_indices:
                                available_actions.append(self.action_types['STANDARD_PROJECT'] + j)
                            added_concrete_action = True
                        else:
                            # Option shown but no enabled projects: avoid selecting it.
                            allow_select_option = False
                    elif option_type == 'or' and 'fund an award' in option_title_l:
                        award_options = option.get('options', [])
                        for j, _ in enumerate(award_options):
                            available_actions.append(600 + j)
                        added_concrete_action = len(award_options) > 0
                    elif option_type in ['selectCard', 'card'] and 'convert plants' in option_title_l:
                        available_actions.append(700)
                        added_concrete_action = True
                    elif option_type in ['selectCard', 'card'] and 'convert heat' in option_title_l:
                        if not self._is_convert_heat_wasteful(player_state):
                            available_actions.append(701)
                            added_concrete_action = True
                        else:
                            allow_select_option = False
                    elif option_type in ['selectCard', 'card'] and 'sell patents' in option_title_l:
                        if _should_offer_sell_patents(
                            option.get('cards', []),
                            option,
                            player_state,
                            allow_mandatory_fallback=False,
                        ):
                            available_actions.append(702)
                            added_concrete_action = True
                        else:
                            allow_select_option = False
                    elif option_type in ['selectCard', 'card'] and not _is_special_card_prompt_title(option_title_l):
                        option_cards = option.get('cards', []) or []
                        min_cards = _safe_int(option.get('min', 1), 1)
                        max_cards = _safe_int(option.get('max', len(option_cards)), len(option_cards))
                        if _is_card_selection_prompt(option) and max_cards > 1:
                            masks = _enumerate_card_selection_masks(
                                option_cards,
                                min_cards,
                                max_cards,
                                player_state,
                                _CARD_SELECTION_MASK_LIMIT,
                            )
                            for j, _ in enumerate(masks):
                                available_actions.append(self.action_types['SELECT_CARD_MASK'] + j)
                            added_concrete_action = len(masks) > 0
                            if not added_concrete_action:
                                allow_select_option = False
                        elif _is_card_selection_prompt(option):
                            enabled_indices = [j for j, card in enumerate(option_cards) if not card.get('isDisabled', False)]
                            if not enabled_indices:
                                enabled_indices = list(range(len(option_cards)))
                            for j in enabled_indices:
                                available_actions.append(self.action_types['PLAY_CARD'] + j)
                            added_concrete_action = len(enabled_indices) > 0
                            if not added_concrete_action:
                                allow_select_option = False

                    # Keep SELECT_OPTION only when we do not have a concrete safer index,
                    # or when this option is simple and does not require sub-selection.
                    if not added_concrete_action and allow_select_option and not option_is_wasteful:
                        available_actions.append(self.action_types['SELECT_OPTION'] + i)
            elif input_type == 'option':
                options = waiting_for.get('options', [])
                for i, option in enumerate(options):
                    if self._is_option_wasteful(option, player_state or {}):
                        continue
                    available_actions.append(self.action_types['SELECT_OPTION'] + i)
            elif input_type == 'card':
                cards = waiting_for.get('cards', [])
                can_pass = waiting_for.get('canPass', False)
                min_cards = _safe_int(waiting_for.get('min', 1), 1)
                max_cards = _safe_int(waiting_for.get('max', len(cards)), len(cards))
                
                # Filter out unaffordable cards in play-card scenarios
                title = _title_text(waiting_for.get('title', '')).lower()
                button_label = waiting_for.get('buttonLabel', '').lower()
                select_blue = waiting_for.get('selectBlueCardAction', False)
                if 'convert plants' in title:
                    available_actions.append(700)
                elif 'convert heat' in title:
                    if not self._is_convert_heat_wasteful(player_state or {}):
                        available_actions.append(701)
                elif 'sell patents' in title:
                    if _should_offer_sell_patents(cards, waiting_for, player_state):
                        available_actions.append(702)
                elif 'standard project' in title:
                    enabled_cards = [
                        (i, card) for i, card in enumerate(cards)
                        if not card.get('isDisabled', False)
                    ]
                    for i, _ in enabled_cards:
                        available_actions.append(self.action_types['STANDARD_PROJECT'] + i)
                
                # Check if this is a card purchase/selection scenario vs playing cards
                is_selection = _is_card_selection_prompt(waiting_for)
                
                if not available_actions and not is_selection and player_state:
                    # For playing cards, check affordability
                    player = player_state.get('thisPlayer', {})
                    affordability_flags = _can_afford_cards_batch(player, cards)
                    affordable_cards = [
                        i for i, is_affordable in enumerate(affordability_flags)
                        if bool(is_affordable)
                    ]
                    
                    # If we have affordable cards, only include those
                    if affordable_cards:
                        for i in affordable_cards:
                            available_actions.append(self.action_types['PLAY_CARD'] + i)
                    else:
                        # No affordable cards: avoid guaranteed payment rejection.
                        if can_pass:
                            available_actions.append(self.action_types['PASS'])
                elif not available_actions:
                    # For selection scenarios, support multi-card subset actions.
                    if is_selection and max_cards > 1:
                        masks = _enumerate_card_selection_masks(
                            cards,
                            min_cards,
                            max_cards,
                            player_state,
                            _CARD_SELECTION_MASK_LIMIT,
                        )
                        for i, _ in enumerate(masks):
                            available_actions.append(self.action_types['SELECT_CARD_MASK'] + i)
                    if not available_actions:
                        enabled_indices = [i for i, card in enumerate(cards) if not card.get('isDisabled', False)]
                        if not enabled_indices:
                            enabled_indices = list(range(len(cards)))
                        for i in enabled_indices:
                            available_actions.append(self.action_types['PLAY_CARD'] + i)
                        
                if can_pass:
                    available_actions.append(self.action_types['PASS'])
            elif input_type == 'selectCard':
                cards = waiting_for.get('cards', [])
                can_pass = waiting_for.get('canPass', False)
                title = _title_text(waiting_for.get('title', '')).lower()
                min_cards = _safe_int(waiting_for.get('min', 1), 1)
                max_cards = _safe_int(waiting_for.get('max', len(cards)), len(cards))
                if 'standard project' in title:
                    enabled_cards = [
                        (i, card) for i, card in enumerate(cards)
                        if not card.get('isDisabled', False)
                    ]
                    for i, _ in enabled_cards:
                        available_actions.append(self.action_types['STANDARD_PROJECT'] + i)
                elif 'convert plants' in title:
                    available_actions.append(700)
                elif 'convert heat' in title:
                    if not self._is_convert_heat_wasteful(player_state or {}):
                        available_actions.append(701)
                elif 'sell patents' in title:
                    if _should_offer_sell_patents(cards, waiting_for, player_state):
                        available_actions.append(702)
                else:
                    if _is_card_selection_prompt(waiting_for) and max_cards > 1:
                        masks = _enumerate_card_selection_masks(
                            cards,
                            min_cards,
                            max_cards,
                            player_state,
                            _CARD_SELECTION_MASK_LIMIT,
                        )
                        for i, _ in enumerate(masks):
                            available_actions.append(self.action_types['SELECT_CARD_MASK'] + i)
                    if not available_actions:
                        enabled_indices = [i for i, card in enumerate(cards) if not card.get('isDisabled', False)]
                        if not enabled_indices:
                            enabled_indices = list(range(len(cards)))
                        for i in enabled_indices:
                            available_actions.append(self.action_types['PLAY_CARD'] + i)
                if can_pass:
                    available_actions.append(self.action_types['PASS'])
            elif input_type in ['projectCard', 'selectProjectCardToPlay']:
                cards = waiting_for.get('cards', [])
                # NOTE: Terraforming Mars SelectProjectCardToPlayResponse does NOT accept
                # {"type":"pass"} - it always requires {type, card, payment}. Never add PASS.
                payment_options = waiting_for.get('paymentOptions', {})
                
                # Filter out unaffordable cards
                affordable_cards = []
                for i, card in enumerate(cards):
                    if player_state:
                        reserve_units = _merge_reserve_units(waiting_for, card)
                        if _can_afford_card_with_payment_options(
                            player_state,
                            card,
                            payment_options,
                            reserve_units=reserve_units,
                            waiting_for=waiting_for,
                        ):
                            affordable_cards.append(i)
                    else:
                        # If no player state, assume all cards are affordable
                        affordable_cards.append(i)
                
                if affordable_cards:
                    for i in affordable_cards:
                        available_actions.append(self.action_types['PLAY_CARD'] + i)
                elif not player_state:
                    for i in range(len(cards)):
                        available_actions.append(self.action_types['PLAY_CARD'] + i)
            elif input_type == 'selectSpace' or input_type == 'space':
                spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
                for i, space in enumerate(spaces):
                    if isinstance(space, dict) and space.get('isDisabled', False):
                        continue
                    available_actions.append(self.action_types['SELECT_SPACE'] + i)
            elif input_type == 'selectPayment' or input_type == 'payment':
                payment_candidates = _enumerate_payment_candidates(
                    waiting_for,
                    player_state,
                    max_candidates=_PAYMENT_ACTION_VARIANTS,
                )
                variant_count = min(_PAYMENT_ACTION_VARIANTS, len(payment_candidates))
                for i in range(variant_count):
                    available_actions.append(self.action_types['SELECT_PAYMENT'] + i)
            elif input_type == 'selectAmount' or input_type == 'amount':
                min_amount = int(waiting_for.get('min', 0))
                max_amount = int(waiting_for.get('max', 10))
                for amount in range(min_amount, min(max_amount + 1, 20)):
                    available_actions.append(self.action_types['SELECT_AMOUNT'] + amount)
            elif input_type == 'selectOption' or input_type == 'option':
                options = waiting_for.get('options', [])
                for i, option in enumerate(options):
                    if self._is_option_wasteful(option, player_state or {}):
                        continue
                    available_actions.append(self.action_types['SELECT_OPTION'] + i)
            elif input_type == 'selectPlayer' or input_type == 'player':
                players = waiting_for.get('players', [])
                for i, _ in enumerate(players):
                    available_actions.append(600 + i)
            elif input_type == 'selectResources' or input_type == 'resources':
                available_actions.append(700)
            elif input_type == 'selectProductionToLose' or input_type == 'productionToLose':
                available_actions.append(710)
            elif input_type == 'selectColony' or input_type == 'colony':
                colonies = waiting_for.get('colonies', [])
                for i, _ in enumerate(colonies):
                    available_actions.append(720 + i)
            elif input_type == 'selectParty' or input_type == 'party':
                parties = waiting_for.get('parties', [])
                for i, _ in enumerate(parties):
                    available_actions.append(730 + i)
            elif input_type == 'selectDelegate' or input_type == 'delegate':
                players = waiting_for.get('players', [])
                for i, _ in enumerate(players):
                    available_actions.append(740 + i)
            elif input_type == 'selectGlobalEvent' or input_type == 'globalEvent':
                events = waiting_for.get('globalEventNames', waiting_for.get('events', []))
                for i, _ in enumerate(events):
                    available_actions.append(750 + i)
            elif input_type == 'selectClaimedUndergroundToken' or input_type == 'claimedUndergroundToken':
                tokens = waiting_for.get('tokens', [])
                for i, _ in enumerate(tokens):
                    available_actions.append(760 + i)
            elif input_type == 'selectInitialCards' or input_type == 'initialCards':
                startup_plans = self._get_cached_startup_plans(waiting_for, player_state)
                for i, _ in enumerate(startup_plans):
                    available_actions.append(_STARTUP_PLAN_BASE + i)
                # Keep deterministic fallback always available.
                available_actions.append(800)
            elif input_type == 'aresGlobalParameters':
                available_actions.append(810)
            elif input_type == 'resource':
                resources = waiting_for.get('include', [])
                for i, _ in enumerate(resources):
                    available_actions.append(820 + i)
            elif input_type == 'and':
                available_actions.append(830)
            elif input_type == 'policy':
                policies = waiting_for.get('policies', [])
                for i, _ in enumerate(policies):
                    available_actions.append(840 + i)
            else:
                # logger.warning(f"Unknown input_type '{input_type}' in get_available_actions. Defaulting to PASS.")
                available_actions.append(self.action_types['PASS'])
            if available_actions:
                # Deduplicate while preserving order to reduce repeated retries.
                available_actions = list(dict.fromkeys(available_actions))
            if not available_actions:
                # Avoid generating guaranteed-invalid pass payloads. SelectProjectCardToPlay
                # does not accept {"type":"pass"} - it requires {type, card, payment}.
                if str(input_type or '') not in [
                    'payment', 'selectPayment', 'projectCard', 'selectProjectCardToPlay'
                ]:
                    available_actions.append(self.action_types['PASS'])
        except Exception as e:
            # logger.error(f"Error getting available actions: {e}")
            available_actions = [self.action_types['PASS']]
        return available_actions

    def decode_action(self, action_index: int, player_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert action index to game input format using the generic builder.
        """
        try:
            waiting_for = player_state.get('waitingFor')
            if not waiting_for:
                # logger.error(f"decode_action: No waitingFor in player_state. Returning pass.")
                return self._create_pass_action()

            input_type = waiting_for.get('type', '')
            # logger.info(f"Decoding action: input_type={input_type}, action_index={action_index}, waiting_for={waiting_for}")

            # Use the new builder for all input types
            response = build_response_for_input(waiting_for, action_index, player_state)

            # logger.info(f"agent response: {response}")
            return response

        except Exception as e:
            # logger.error(f"Error decoding action {action_index} for input_type {waiting_for.get('type', '') if waiting_for else 'None'}: {e}")
            return self._create_pass_action()
    
    def _create_card_action(self, card_idx: int, waiting_for: Dict[str, Any], player_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create card play action"""
        cards = waiting_for.get('cards', [])
        
        if card_idx < len(cards):
            card = cards[card_idx]
            return {
                'type': 'card',
                'card': card.get('name', ''),
                'payment': self._create_default_payment(player_state, card)
            }
        else:
            return self._create_pass_action()
    
    def _create_option_action(self, option_idx: int, waiting_for: Dict[str, Any]) -> Dict[str, Any]:
        """Create option selection action"""
        if waiting_for.get('type') == 'or':
            options = waiting_for.get('options', [])
            if option_idx < len(options):
                return {
                    'type': 'or',
                    'index': option_idx
                }
        
        return {
            'type': 'option',
            'index': option_idx
        }
    
    def _create_space_action(self, space_idx: int, waiting_for: Dict[str, Any]) -> Dict[str, Any]:
        """Create space selection action"""
        spaces = waiting_for.get('availableSpaces', [])
        
        if space_idx < len(spaces):
            space = spaces[space_idx]
            return {
                'type': 'space',
                'spaceId': space.get('id', ''),
                'spaceType': space.get('spaceType', '')
            }
        else:
            # Fallback: select first available space
            if spaces:
                return {
                    'type': 'space',
                    'spaceId': spaces[0].get('id', ''),
                    'spaceType': spaces[0].get('spaceType', '')
                }
            return self._create_pass_action()
    
    def _create_payment_action(self, payment_idx: int, waiting_for: Dict[str, Any]) -> Dict[str, Any]:
        """Create payment selection action"""
        # Simplified payment logic
        # In a real implementation, this would analyze available resources
        
        payment_options = [
            {'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0},
            {'megaCredits': 1, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0},
            {'megaCredits': 0, 'steel': 1, 'titanium': 0, 'heat': 0, 'plants': 0},
            {'megaCredits': 0, 'steel': 0, 'titanium': 1, 'heat': 0, 'plants': 0},
            {'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 1, 'plants': 0},
            {'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 1},
        ]
        
        if payment_idx < len(payment_options):
            payment = payment_options[payment_idx]
        else:
            payment = payment_options[0]
        
        return {
            'type': 'payment',
            'payment': payment
        }
    
    def _create_amount_action(self, amount: int, waiting_for: Dict[str, Any]) -> Dict[str, Any]:
        """Create amount selection action"""
        max_amount = waiting_for.get('max', 10)
        min_amount = waiting_for.get('min', 0)
        
        # Clamp amount to valid range
        amount = max(min_amount, min(amount, max_amount))
        
        return {
            'type': 'amount',
            'amount': amount
        }
    def _should_build_greenery(self, player_state: Dict[str, Any]) -> bool:
        """Strategic greenery placement logic"""
        player = player_state.get('thisPlayer', {})
        game = player_state.get('game', {})
        
        # Build greenery if:
        # 1. Oxygen is low and we have plants
        # 2. We have plant production
        # 3. It's early game
        oxygen = game.get('oxygenLevel', 0)
        plants = player.get('plants', 0)
        plant_prod = player.get('plantProduction', 0)
        generation = game.get('generation', 1)
    
        return (oxygen < 8 and plants >= 8) or (plant_prod > 0 and generation < 8)
    
    
    def _create_standard_project_action(self, project_idx: int, player_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create standard project action with strategic selection"""
        player = player_state.get('thisPlayer', {})
    
        # Strategic project selection based on game state
        if self._should_build_greenery(player_state):
            project = 'Greenery'
        elif self._should_build_city(player_state):
            project = 'City'
        elif self._should_build_power_plant(player_state):
            project = 'Power Plant:SP'
        else:
            # Default to first available
            project = self.standard_projects[project_idx % len(self.standard_projects)]
        
        return {
            'type': 'standardProject',
            'project': project,
            'payment': self._calculate_optimal_payment(player_state, project)
        }
    
    def _create_pass_action(self) -> Dict[str, Any]:
        """Create pass/skip action"""
        return {
            'type': 'pass'
        }
    
    def _create_default_payment(self, player_state: Dict[str, Any] = None, card: Dict[str, Any] = None, cost_reduction: int = 0, payment_options: dict = None) -> Dict[str, Any]:
        # Extract player resources
        if not player_state or not card:
            return {'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0}
            
        player = player_state.get('thisPlayer', {})
        player_mc = player.get('megaCredits', 0)
        player_steel = player.get('steel', 0)
        player_titanium = player.get('titanium', 0)
        player_heat = player.get('heat', 0)
        player_plants = player.get('plants', 0)
        steel_value = player.get('steelValue', 2)
        titanium_value = player.get('titaniumValue', 3)

        # Calculate final cost
        total_discount = cost_reduction
        if 'discount' in card and isinstance(card['discount'], list):
            for d in card['discount']:
                if isinstance(d, dict) and 'amount' in d and ('tag' not in d or d.get('per') is None):
                    total_discount += d['amount']
        
        cost = max(card.get('calculatedCost', 0) - total_discount, 0)
        
        tags = card.get('tags', {})
        payment = {
            'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0,
            'microbes': 0, 'floaters': 0
        }
        
        cost_remaining = cost

        def allowed(resource):
            if payment_options is None:
                return True
            return payment_options.get(resource, True)

        # Use steel for Building tags
        if tags.get('Building') and allowed('steel'):
            usable_steel_value = player_steel * steel_value
            steel_to_pay = min(cost_remaining, usable_steel_value)
            steel_units = steel_to_pay // steel_value
            payment['steel'] = steel_units
            cost_remaining -= steel_units * steel_value

        # Use titanium for Space tags
        if tags.get('Space') and allowed('titanium'):
            usable_titanium_value = player_titanium * titanium_value
            titanium_to_pay = min(cost_remaining, usable_titanium_value)
            titanium_units = titanium_to_pay // titanium_value
            payment['titanium'] = titanium_units
            cost_remaining -= titanium_units * titanium_value

        # Use heat if allowed (e.g., Helion)
        if allowed('heat'):
            heat_to_pay = min(cost_remaining, player_heat)
            payment['heat'] = heat_to_pay
            cost_remaining -= heat_to_pay

        # Use plants if allowed
        if allowed('plants'):
            plants_to_pay = min(cost_remaining, player_plants)
            payment['plants'] = plants_to_pay
            cost_remaining -= plants_to_pay

        # Pay remainder with MegaCredits
        if allowed('megaCredits'):
            payment['megaCredits'] = cost_remaining
        
        # Ensure all values are integers
        for k in payment:
            payment[k] = int(payment[k])
            
        return payment
