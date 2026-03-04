import itertools
import json
import sys
from typing import Any, Dict, List

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")


rust_tfm_rl = pytest.importorskip("rust_tfm_rl")


def _card_cost(card: Dict[str, Any]) -> float:
    try:
        return float(card.get("calculatedCost", card.get("cost", 0)) or 0)
    except Exception:
        return 0.0


def _card_tags(card: Dict[str, Any]) -> Dict[str, int]:
    tags = card.get("tags", {}) or {}
    out: Dict[str, int] = {}
    if isinstance(tags, dict):
        for key, value in tags.items():
            if value:
                out[str(key)] = 1
    return out


def _affordability_ref(player: Dict[str, Any], card: Dict[str, Any]) -> bool:
    cost = _card_cost(card)
    if cost <= 0:
        return True
    tags = _card_tags(card)
    purchasing_power = float(player.get("megaCredits", 0) or 0)
    if tags.get("Building"):
        purchasing_power += float(player.get("steel", 0) or 0) * float(player.get("steelValue", 2) or 2)
    if tags.get("Space"):
        purchasing_power += float(player.get("titanium", 0) or 0) * float(player.get("titaniumValue", 3) or 3)
    corp = player.get("corporation", {}) or {}
    corp_name = str(corp.get("name", "") or "").lower() if isinstance(corp, dict) else str(corp).lower()
    if "helion" in corp_name:
        purchasing_power += float(player.get("heat", 0) or 0)
    return purchasing_power >= cost


def _score_ref(card: Dict[str, Any], player: Dict[str, Any]) -> float:
    cost = _card_cost(card)
    tags = _card_tags(card)
    purchasing_power = float(player.get("megaCredits", 0) or 0)
    if tags.get("Building"):
        purchasing_power += float(player.get("steel", 0) or 0) * float(player.get("steelValue", 2) or 2)
    if tags.get("Space"):
        purchasing_power += float(player.get("titanium", 0) or 0) * float(player.get("titaniumValue", 3) or 3)
    affordable = 1.0 if purchasing_power >= cost else max(0.0, 1.0 - ((cost - purchasing_power) / 20.0))
    cheapness = max(0.0, 1.0 - min(cost / 40.0, 1.0))
    vp = float(card.get("victoryPoints", 0) or 0)
    tag_score = 0.0
    if tags.get("Science"):
        tag_score += 0.45
    if tags.get("Building"):
        tag_score += 0.35
    if tags.get("Space"):
        tag_score += 0.35
    if tags.get("Earth"):
        tag_score += 0.2
    if tags.get("Plant"):
        tag_score += 0.2
    name = str(card.get("name", "") or "")
    tiebreak = (sum(ord(ch) for ch in name) % 97) * 1e-5
    return (vp * 0.45) + (affordable * 1.4) + (cheapness * 0.8) + tag_score + tiebreak


def _enumerate_masks_ref(cards: List[Dict[str, Any]], min_cards: int, max_cards: int, player: Dict[str, Any], limit: int) -> List[int]:
    enabled = [idx for idx, card in enumerate(cards) if not card.get("isDisabled", False)]
    if not enabled:
        return []
    min_pick = max(0, min(int(min_cards), len(enabled)))
    max_pick = max(min_pick, min(int(max_cards), len(enabled)))
    scored = sorted(enabled, key=lambda idx: _score_ref(cards[idx], player), reverse=True)
    candidate_budget = min(len(scored), max(12, min(max_pick, 14)))
    candidate_indices = sorted(scored[:candidate_budget])
    score_map = {idx: _score_ref(cards[idx], player) for idx in candidate_indices}

    ranked = []
    if min_pick == 0:
        ranked.append((0.0, 0, 0))
    for pick_count in range(max(1, min_pick), max_pick + 1):
        for combo in itertools.combinations(candidate_indices, pick_count):
            mask = 0
            combo_score = 0.0
            for idx in combo:
                mask |= (1 << idx)
                combo_score += score_map[idx]
            combo_score += 0.03 * float(pick_count)
            ranked.append((combo_score, pick_count, mask))
    ranked.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    out = []
    seen = set()
    for _, _, mask in ranked:
        if mask in seen:
            continue
        seen.add(mask)
        out.append(mask)
        if len(out) >= max(1, int(limit)):
            break
    return out


def test_can_afford_cards_parity() -> None:
    player = {
        "megaCredits": 15,
        "steel": 4,
        "titanium": 2,
        "steelValue": 2,
        "titaniumValue": 3,
        "heat": 5,
        "corporation": {"name": "Helion"},
    }
    cards = [
        {"name": "A", "calculatedCost": 8, "tags": {"Building": True}},
        {"name": "B", "calculatedCost": 25, "tags": {"Space": True}},
        {"name": "C", "calculatedCost": 19, "tags": {"Science": True}},
    ]
    expected = [_affordability_ref(player, card) for card in cards]
    got = rust_tfm_rl.can_afford_cards(json.dumps(player), json.dumps(cards))
    assert [bool(v) for v in got] == expected


def test_card_selection_combo_parity_top_masks() -> None:
    player = {"megaCredits": 20, "steel": 3, "titanium": 2, "steelValue": 2, "titaniumValue": 3}
    cards: List[Dict[str, Any]] = []
    for idx in range(10):
        cards.append(
            {
                "name": f"C{idx}",
                "calculatedCost": 4 + idx,
                "victoryPoints": idx % 3,
                "tags": {"Building": bool(idx % 2), "Science": bool(idx % 4 == 0)},
                "isDisabled": bool(idx == 9),
            }
        )

    payload = {"cards": cards, "minCards": 1, "maxCards": 3, "playerState": {"thisPlayer": player}}
    combos = rust_tfm_rl.enumerate_card_selection_combos(json.dumps(payload), 20)
    rust_masks = []
    for combo in combos:
        mask = 0
        for idx in combo:
            mask |= (1 << int(idx))
        rust_masks.append(mask)

    py_masks = _enumerate_masks_ref(cards, 1, 3, player, 20)
    assert rust_masks[:5] == py_masks[:5]


def test_rank_startup_plans_deduplicates_and_sorts() -> None:
    payload = {
        "candidates": [
            {"index": 0, "score": 12.0, "corp": "A", "prelude": ["P1"], "ceo": [], "project": ["X", "Y"]},
            {"index": 1, "score": 14.0, "corp": "A", "prelude": ["P1"], "ceo": [], "project": ["Y", "X"]},
            {"index": 2, "score": 10.0, "corp": "B", "prelude": [], "ceo": [], "project": ["K"]},
        ]
    }
    ranked_json = rust_tfm_rl.rank_startup_plans(json.dumps(payload), 8)
    ranked = json.loads(ranked_json)
    assert ranked[0]["index"] == 1
    assert [row["index"] for row in ranked] == [1, 2]
