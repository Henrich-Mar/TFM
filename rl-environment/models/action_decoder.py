"""
Action Decoder - Converts neural network output to game actions
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import random
import os
import json

logger = logging.getLogger(__name__)

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

    ranked_all = sorted(
        cards,
        key=lambda c: (_card_cost(c), str(c.get('name', '') or ''))
    )

    # Strategic sell: if possible, sell exactly enough cheap cards to afford
    # the best reachable card in hand (prioritize VP gain).
    player = (player_state or {}).get('thisPlayer', {}) if isinstance(player_state, dict) else {}
    if player and len(cards) > 1:
        hand_cards = (player_state or {}).get('cardsInHand', []) if isinstance(player_state, dict) else []
        if not isinstance(hand_cards, list) or not hand_cards:
            hand_cards = cards

        max_sell_for_target = min(max_cards, len(cards) - 1)
        best_target_name: Optional[str] = None
        best_needed: Optional[int] = None
        best_rank: Optional[Tuple[int, int, int]] = None

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

            # Higher VP first, then fewer sells, then higher impact/cost card.
            rank = (_card_vp(candidate), -int(needed), _card_cost(candidate))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_target_name = name
                best_needed = int(needed)

        if best_target_name is not None and best_needed is not None:
            sell_count = max(min_cards, min(max_cards, best_needed))
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
    default_count = max(min_cards, 1)
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
    starting_mc = float(_safe_int(card.get('startingMegaCredits', card.get('startingMegacredits', 0)), 0))
    card_cost_override = float(_safe_int(card.get('cardCost', 3), 3))

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
    cards = option.get('cards', []) or []
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

def _build_initial_setup_response(waiting_for: Dict[str, Any], player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
        if action_index is None or _safe_int(action_index, -1) == 800:
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
        min_cards = waiting_for.get('min', 1)
        max_cards = waiting_for.get('max', len(cards))
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
                    selected = []
                    for i in range(n):
                        if (action_index >> i) & 1:
                            selected.append(cards[i]['name'])
                    if len(selected) < min_cards:
                        for i in range(n):
                            if cards[i]['name'] not in selected:
                                selected.append(cards[i]['name'])
                                if len(selected) >= min_cards:
                                    break
                    if len(selected) > max_cards:
                        selected = selected[:max_cards]
                    card_names = selected
                else:
                    card_names = [c['name'] for c in cards[:min_cards]]
            else:
                card_names = [c['name'] for c in cards[:min_cards]] if cards else []
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
        # Build a valid payment using per-resource unit values and allowed payment options.
        # Server expects counts of resources; value conversion is done server-side.
        all_keys = [
            'megaCredits', 'steel', 'titanium', 'heat', 'plants',
            'microbes', 'floaters', 'lunaArchivesScience', 'spireScience',
            'seeds', 'auroraiData', 'graphene', 'kuiperAsteroids'
        ]
        payment: Dict[str, int] = {k: 0 for k in all_keys}
        amount = int(waiting_for.get('amount', 0) or 0)
        payment_options = waiting_for.get('paymentOptions', {}) or {}
        player = (player_state or {}).get('thisPlayer', {}) if player_state else {}

        # Try to detect which project/card this payment is for (e.g., Road Infrastructure / Lunar Mine / Lunar Habitat)
        title = waiting_for.get('title')
        project_name = ''
        if isinstance(title, dict):
            # title: { message: 'Select how to pay for the ${0} standard project', data: [{type:..., value: 'Road Infrastructure'}] }
            try:
                data_items = title.get('data', []) or []
                for item in data_items:
                    val = item.get('value')
                    if isinstance(val, str) and len(val) > 0:
                        project_name = val
                        break
            except Exception:
                project_name = ''
        elif isinstance(title, str):
            project_name = title
        project_name_l = project_name.lower() if isinstance(project_name, str) else ''

        # Unit values aligning with Payment.DEFAULT_PAYMENT_VALUES and player-specific steel/titanium values
        steel_value = int(player.get('steelValue', 2) or 2)
        titanium_value = int(player.get('titaniumValue', 3) or 3)
        # Special case: Luna Trade Federation titanium pays 2 M€ each via a special path
        if payment_options.get('lunaTradeFederationTitanium', False) and not payment_options.get('titanium', False):
            titanium_value = 2
        value_by_resource: Dict[str, int] = {
            'megaCredits': 1,
            'steel': steel_value,
            'titanium': titanium_value,
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

        # Determine allowed resources. Keys missing from paymentOptions are disallowed, except megaCredits.
        def is_allowed(res: str) -> bool:
            if res == 'megaCredits':
                return True
            # Luna Trade Federation grants a separate titanium path; expose it as 'titanium'
            if res == 'titanium' and payment_options.get('lunaTradeFederationTitanium', False):
                return True
            return bool(payment_options.get(res, False))

        # Determine available units. For card-resources, waiting_for often includes numeric counts.
        def available_units(res: str) -> int:
            if res in ['megaCredits', 'steel', 'titanium', 'heat', 'plants']:
                return int(player.get(res, 0) or 0)
            # card resources: prefer waiting_for value; fallback to 0
            return int(waiting_for.get(res, 0) or 0)

        # Spend non-MC resources first, sorted by unit value descending, then finish with MC
        amount_remaining = amount
        spend_order = [r for r in all_keys if r != 'megaCredits' and is_allowed(r)]
        spend_order.sort(key=lambda r: value_by_resource.get(r, 0), reverse=True)

        for res in spend_order:
            units_avail = available_units(res)
            unit_value = value_by_resource.get(res, 0)
            if units_avail <= 0 or unit_value <= 0 or amount_remaining <= 0:
                continue
            # Use as many units as needed without overshooting; MC will cover the remainder
            max_units_to_use = amount_remaining // unit_value
            if max_units_to_use <= 0:
                continue
            use_units = min(units_avail, max_units_to_use)
            payment[res] = int(use_units)
            amount_remaining -= use_units * unit_value

        # Finish with MC (always allowed)
        if amount_remaining > 0:
            payment['megaCredits'] = int(min(amount_remaining, available_units('megaCredits')))
            amount_remaining -= payment['megaCredits']

        # Enforce mandatory extra resource for certain standard projects (not included in 'amount')
        try:
            if project_name_l:
                # Road Infrastructure requires at least 1 steel
                if ('road' in project_name_l) and ('infrastructure' in project_name_l or 'moon' in project_name_l):
                    if is_allowed('steel'):
                        have = available_units('steel')
                        if have > 0 and payment.get('steel', 0) < 1:
                            payment['steel'] = 1
                # Lunar Mine / Moon Mine requires at least 1 titanium
                if ('lunar mine' in project_name_l) or ('moon mine' in project_name_l):
                    if is_allowed('titanium') or payment_options.get('lunaTradeFederationTitanium', False):
                        have = available_units('titanium')
                        if have > 0 and payment.get('titanium', 0) < 1:
                            payment['titanium'] = 1
                # Lunar Habitat / Moon Habitat requires at least 1 titanium
                if ('lunar habitat' in project_name_l) or ('moon habitat' in project_name_l):
                    if is_allowed('titanium') or payment_options.get('lunaTradeFederationTitanium', False):
                        have = available_units('titanium')
                        if have > 0 and payment.get('titanium', 0) < 1:
                            payment['titanium'] = 1
        except Exception:
            pass

        # Ensure well-formed response with integer, non-negative counts
        for k in payment:
            v = int(payment[k] or 0)
            payment[k] = 0 if v < 0 else v

        return {'type': 'payment', 'payment': payment}
    elif input_type == 'player':
        players = waiting_for.get('players', [])
        idx = normalize_index(action_index, 600) if action_index is not None else 0
        player = players[idx] if players and 0 <= idx < len(players) else (players[0] if players else '')
        return {'type': 'player', 'player': player}
    elif input_type == 'productionToLose':
        units = {k: 0 for k in ['megaCreditProduction', 'steelProduction', 'titaniumProduction', 'plantProduction', 'energyProduction', 'heatProduction']}
        return {'type': 'productionToLose', 'units': units}
    elif input_type == 'projectCard':
        cards = waiting_for.get('cards', [])
        can_pass = bool(waiting_for.get('canPass', False))
        if not cards:
            return {'type': 'pass'}

        # For projectCard, action_index should be the card index directly
        card_idx = normalize_index(action_index, 0) if action_index is not None else 0
        affordable_indices: List[int] = []
        payment_options = waiting_for.get('paymentOptions', {})
        if player_state:
            for i, candidate in enumerate(cards):
                if _can_afford_card_with_payment_options(player_state, candidate, payment_options):
                    affordable_indices.append(i)

        if affordable_indices:
            if card_idx in affordable_indices:
                chosen_idx = card_idx
            elif 0 <= card_idx < len(affordable_indices):
                chosen_idx = affordable_indices[card_idx]
            else:
                chosen_idx = affordable_indices[0]
            card = cards[chosen_idx]
        elif can_pass:
            return {'type': 'pass'}
        elif 0 <= card_idx < len(cards):
            card = cards[card_idx]
        else:
            card = cards[0]

        # Build inline payment respecting paymentOptions
        payment = _build_payment_with_options(player_state, card, payment_options)
        if payment is None:
            return {'type': 'pass'}
        return {'type': 'projectCard', 'card': card['name'], 'payment': payment}
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
        # Handle project card selection for playing
        cards = waiting_for.get('cards', [])
        can_pass = bool(waiting_for.get('canPass', False))
        if not cards:
            return {'type': 'pass'}
             
        card_idx = normalize_index(action_index, 0) if action_index is not None else 0
        affordable_indices: List[int] = []
        payment_options = waiting_for.get('paymentOptions', {}) if isinstance(waiting_for, dict) else {}
        if player_state:
            for i, candidate in enumerate(cards):
                if _can_afford_card_with_payment_options(player_state, candidate, payment_options):
                    affordable_indices.append(i)
        if affordable_indices:
            if card_idx in affordable_indices:
                chosen_idx = card_idx
            elif 0 <= card_idx < len(affordable_indices):
                chosen_idx = affordable_indices[card_idx]
            else:
                chosen_idx = affordable_indices[0]
            card = cards[chosen_idx]
            payment = _build_payment_with_options(player_state, card, payment_options)
            if payment is None:
                return {'type': 'pass'}
            return {'type': 'projectCard', 'card': card['name'], 'payment': payment}
        if can_pass:
            return {'type': 'pass'}
        if 0 <= card_idx < len(cards):
            card = cards[card_idx]
            payment = _build_payment_with_options(player_state, card, payment_options)
            if payment is None:
                return {'type': 'pass'}
            return {'type': 'projectCard', 'card': card['name'], 'payment': payment}
        return {'type': 'pass'}
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
            payment = _build_payment_with_options(player_state, card, waiting_for.get('paymentOptions', {}))
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
    
    payment = {
        'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0
    }
    
    cost_remaining = cost

    # Use steel for Building tags (floor units first; top-up handled later if needed)
    if tags.get('Building') and player_steel > 0:
        usable_steel_value = player_steel * steel_value
        steel_to_pay = min(cost_remaining, usable_steel_value)
        steel_units = steel_to_pay // steel_value
        payment['steel'] = steel_units
        cost_remaining -= steel_units * steel_value

    # Use titanium for Space tags (floor units first; top-up handled later if needed)
    if tags.get('Space') and player_titanium > 0:
        usable_titanium_value = player_titanium * titanium_value
        titanium_to_pay = min(cost_remaining, usable_titanium_value)
        titanium_units = titanium_to_pay // titanium_value
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

def _build_payment_with_options(player_state: Dict[str, Any], card: Dict[str, Any], payment_options: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Build a payment dict honoring paymentOptions from waiting_for."""
    if player_state is None:
        player_state = {}
    player = player_state.get('thisPlayer', {})
    player_mc = int(player.get('megaCredits', 0) or 0)
    player_steel = int(player.get('steel', 0) or 0)
    player_titanium = int(player.get('titanium', 0) or 0)
    player_heat = int(player.get('heat', 0) or 0)
    player_plants = int(player.get('plants', 0) or 0)
    steel_value = int(player.get('steelValue', 2) or 2)
    titanium_value = int(player.get('titaniumValue', 3) or 3)

    cost = int(card.get('calculatedCost', card.get('cost', 0)) or 0)
    tags = card.get('tags', {}) or _metadata_tags(card.get('name', ''))

    # Default zeroes for all known keys the server may accept
    payment = {
        'megaCredits': 0, 'steel': 0, 'titanium': 0, 'heat': 0, 'plants': 0,
        'microbes': 0, 'floaters': 0, 'lunaArchivesScience': 0, 'spireScience': 0,
        'seeds': 0, 'auroraiData': 0, 'graphene': 0, 'kuiperAsteroids': 0
    }

    def allowed(resource: str) -> bool:
        return payment_options.get(resource, True)

    cost_remaining = cost

    # Use steel for Building tags if allowed (floor units first; top-up handled later)
    if tags.get('Building') and allowed('steel') and player_steel > 0:
        usable_steel_value = player_steel * steel_value
        steel_to_pay = min(cost_remaining, usable_steel_value)
        steel_units = steel_to_pay // steel_value
        payment['steel'] = int(steel_units)
        cost_remaining -= steel_units * steel_value

    # Use titanium for Space tags if allowed (floor units first; top-up handled later)
    if tags.get('Space') and allowed('titanium') and player_titanium > 0:
        usable_titanium_value = player_titanium * titanium_value
        titanium_to_pay = min(cost_remaining, usable_titanium_value)
        titanium_units = titanium_to_pay // titanium_value
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
        if tags.get('Building') and allowed('steel'):
            remaining_steel = max(0, int(player_steel - payment.get('steel', 0)))
            if remaining_steel > 0:
                topup_resources.append(('steel', remaining_steel, max(1, int(steel_value))))
        if tags.get('Space') and allowed('titanium'):
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

    # Unaffordable under current payment options.
    if cost_remaining > 0:
        return None

    return payment

def _can_afford_card_with_payment_options(
    player_state: Optional[Dict[str, Any]],
    card: Dict[str, Any],
    payment_options: Optional[Dict[str, Any]],
) -> bool:
    payment = _build_payment_with_options(player_state or {}, card, payment_options or {})
    return payment is not None

# --- Metadata helpers for tags when missing ---
def _metadata_loader() -> Dict[str, Dict[str, Any]]:
    path = os.getenv('TM_CARD_METADATA_PATH')
    if not path:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

_CARD_META_CACHE: Dict[str, Dict[str, Any]] = _metadata_loader()

def _metadata_tags(card_name: str) -> Dict[str, int]:
    if not card_name or not _CARD_META_CACHE:
        return {}
    meta = _CARD_META_CACHE.get(card_name)
    if not meta:
        return {}
    out: Dict[str, int] = {}
    for t in meta.get('tags', []) or []:
        ts = str(t).strip()
        if ts and ts[0].islower():
            ts = ts.capitalize()
        out[ts] = 1
    return out
def _can_afford_card(player: Dict[str, Any], card: Dict[str, Any]) -> bool:
    """Check if player can afford to play a card"""
    cost = card.get('calculatedCost', card.get('cost', 0))
    if cost <= 0:
        return True
        
    player_mc = player.get('megaCredits', 0)
    player_steel = player.get('steel', 0)
    player_titanium = player.get('titanium', 0)
    player_heat = player.get('heat', 0)
    player_plants = player.get('plants', 0)
    steel_value = player.get('steelValue', 2)
    titanium_value = player.get('titaniumValue', 3)
    
    total_purchasing_power = player_mc
    
    # Add steel value for Building cards
    tags = card.get('tags', {}) or _metadata_tags(card.get('name', ''))
    if tags.get('Building'):
        total_purchasing_power += player_steel * steel_value
        
    # Add titanium value for Space cards  
    if tags.get('Space'):
        total_purchasing_power += player_titanium * titanium_value
    
    # Check corporation-specific abilities (simplified)
    corporation = player.get('corporation', {})
    corp_name = corporation.get('name', '') if isinstance(corporation, dict) else str(corporation)
    
    # Helion can use heat as money
    if 'helion' in corp_name.lower():
        total_purchasing_power += player_heat
        
    # Check if they can afford it
    return total_purchasing_power >= cost

class ActionDecoder:
    def __init__(self):
        # Action space mapping
        self.action_types = {
            'PLAY_CARD': 0,
            'STANDARD_PROJECT': 100,
            'SELECT_OPTION': 200,
            'SELECT_SPACE': 300,
            'SELECT_PAYMENT': 400,
            'SELECT_AMOUNT': 500,
            'PASS': 900,
            'END_TURN': 950
        }
        
        # Standard projects mapping - updated with all available standard projects
        self.standard_projects = [
            'Power Plant:SP', 'Asteroid:SP', 'Aquifer', 'Greenery', 'City', 'Colony',
            'Air Scrapping', 'Buffer Gas', 'Moon Habitat', 'Moon Mine', 'Moon Road'
        ]

    def build_initial_setup_response(self, player_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        input_type = waiting_for.get('type', '')
        if input_type not in ['initialCards', 'selectInitialCards']:
            return None
        return _build_initial_setup_response(waiting_for, player_state)
    
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

                    if option_type in ['selectProjectCardToPlay', 'projectCard']:
                        # Prefer concrete card actions instead of generic OR selection.
                        cards = option.get('cards', [])
                        payment_options = option.get('paymentOptions', {})
                        affordable_indices = []
                        for j, card in enumerate(cards):
                            if player_state and _can_afford_card_with_payment_options(player_state, card, payment_options):
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
                        enabled_indices = [j for j, card in enumerate(cards) if not card.get('isDisabled', False)]
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
                        available_actions.append(701)
                        added_concrete_action = True
                    elif option_type in ['selectCard', 'card'] and 'sell patents' in option_title_l:
                        if option.get('cards', []):
                            available_actions.append(702)
                            added_concrete_action = True
                        else:
                            allow_select_option = False

                    # Keep SELECT_OPTION only when we do not have a concrete safer index,
                    # or when this option is simple and does not require sub-selection.
                    if not added_concrete_action and allow_select_option:
                        available_actions.append(self.action_types['SELECT_OPTION'] + i)
            elif input_type == 'option':
                options = waiting_for.get('options', [])
                for i, _ in enumerate(options):
                    available_actions.append(self.action_types['SELECT_OPTION'] + i)
            elif input_type == 'card':
                cards = waiting_for.get('cards', [])
                can_pass = waiting_for.get('canPass', False)
                
                # Filter out unaffordable cards in play-card scenarios
                title = _title_text(waiting_for.get('title', '')).lower()
                button_label = waiting_for.get('buttonLabel', '').lower()
                select_blue = waiting_for.get('selectBlueCardAction', False)
                if 'convert plants' in title:
                    available_actions.append(700)
                elif 'convert heat' in title:
                    available_actions.append(701)
                elif 'sell patents' in title:
                    if cards:
                        available_actions.append(702)
                elif 'standard project' in title:
                    enabled_cards = [(i, card) for i, card in enumerate(cards) if not card.get('isDisabled', False)]
                    for i, _ in enabled_cards:
                        available_actions.append(self.action_types['STANDARD_PROJECT'] + i)
                
                # Check if this is a card purchase/selection scenario vs playing cards
                is_selection = (
                    'prelude' in title
                    or 'select' in title
                    or button_label in ['keep', 'buy', 'select', 'choose', 'take action', 'discard', 'confirm', 'ok']
                    or waiting_for.get('showOnlyInLearnerMode', False)
                    or select_blue
                )
                
                if not available_actions and not is_selection and player_state:
                    # For playing cards, check affordability
                    player = player_state.get('thisPlayer', {})
                    affordable_cards = []
                    for i, card in enumerate(cards):
                        if _can_afford_card(player, card):
                            affordable_cards.append(i)
                    
                    # If we have affordable cards, only include those
                    if affordable_cards:
                        for i in affordable_cards:
                            available_actions.append(self.action_types['PLAY_CARD'] + i)
                    else:
                        # No affordable cards: avoid guaranteed payment rejection.
                        if can_pass:
                            available_actions.append(self.action_types['PASS'])
                elif not available_actions:
                    # For selection scenarios, include all cards
                    for i, _ in enumerate(cards):
                        available_actions.append(self.action_types['PLAY_CARD'] + i)
                        
                if can_pass:
                    available_actions.append(self.action_types['PASS'])
            elif input_type == 'selectCard':
                cards = waiting_for.get('cards', [])
                can_pass = waiting_for.get('canPass', False)
                title = _title_text(waiting_for.get('title', '')).lower()
                if 'standard project' in title:
                    enabled_cards = [(i, card) for i, card in enumerate(cards) if not card.get('isDisabled', False)]
                    for i, _ in enabled_cards:
                        available_actions.append(self.action_types['STANDARD_PROJECT'] + i)
                elif 'convert plants' in title:
                    available_actions.append(700)
                elif 'convert heat' in title:
                    available_actions.append(701)
                elif 'sell patents' in title:
                    if cards:
                        available_actions.append(702)
                else:
                    enabled_indices = [i for i, card in enumerate(cards) if not card.get('isDisabled', False)]
                    if not enabled_indices:
                        enabled_indices = list(range(len(cards)))
                    for i in enabled_indices:
                        available_actions.append(self.action_types['PLAY_CARD'] + i)
                if can_pass:
                    available_actions.append(self.action_types['PASS'])
            elif input_type in ['projectCard', 'selectProjectCardToPlay']:
                cards = waiting_for.get('cards', [])
                can_pass = waiting_for.get('canPass', False)
                payment_options = waiting_for.get('paymentOptions', {})
                
                # Filter out unaffordable cards
                affordable_cards = []
                for i, card in enumerate(cards):
                    if player_state:
                        if _can_afford_card_with_payment_options(player_state, card, payment_options):
                            affordable_cards.append(i)
                    else:
                        # If no player state, assume all cards are affordable
                        affordable_cards.append(i)
                
                # If we have affordable cards, only include those
                if affordable_cards:
                    for i in affordable_cards:
                        available_actions.append(self.action_types['PLAY_CARD'] + i)
                else:
                    # No affordable cards: avoid guaranteed payment rejection.
                    if can_pass:
                        available_actions.append(self.action_types['PASS'])
                         
                if can_pass:
                    available_actions.append(self.action_types['PASS'])
            elif input_type == 'selectSpace' or input_type == 'space':
                spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
                for i, space in enumerate(spaces):
                    if isinstance(space, dict) and space.get('isDisabled', False):
                        continue
                    available_actions.append(self.action_types['SELECT_SPACE'] + i)
            elif input_type == 'selectPayment' or input_type == 'payment':
                for i in range(10):
                    available_actions.append(self.action_types['SELECT_PAYMENT'] + i)
            elif input_type == 'selectAmount' or input_type == 'amount':
                min_amount = int(waiting_for.get('min', 0))
                max_amount = int(waiting_for.get('max', 10))
                for amount in range(min_amount, min(max_amount + 1, 20)):
                    available_actions.append(self.action_types['SELECT_AMOUNT'] + amount)
            elif input_type == 'selectOption' or input_type == 'option':
                options = waiting_for.get('options', [])
                for i, _ in enumerate(options):
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

            # Optionally, add runId if present
            run_id = waiting_for.get('runId')
            if run_id:
                response['runId'] = run_id
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
