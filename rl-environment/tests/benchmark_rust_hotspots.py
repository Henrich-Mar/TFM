#!/usr/bin/env python3
"""
Micro-benchmarks for Python hotspot baseline and Rust kernel speed.

Run from repository root:
    python rl-environment/tests/benchmark_rust_hotspots.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _card_tags(card: Dict[str, Any]) -> Dict[str, int]:
    tags = card.get("tags", {})
    out: Dict[str, int] = {}
    if isinstance(tags, dict):
        for k, v in tags.items():
            if v:
                kk = str(k)
                if kk and kk[0].islower():
                    kk = kk.capitalize()
                out[kk] = 1
    elif isinstance(tags, list):
        for item in tags:
            kk = str(item)
            if kk and kk[0].islower():
                kk = kk.capitalize()
            out[kk] = 1
    return out


def _card_cost(card: Dict[str, Any]) -> float:
    try:
        return float(card.get("calculatedCost", card.get("cost", 0)) or 0)
    except Exception:
        return 0.0


def _affordability_python(player: Dict[str, Any], card: Dict[str, Any]) -> float:
    tags = _card_tags(card)
    cost = _card_cost(card)
    mc = float(player.get("megaCredits", 0) or 0)
    steel = float(player.get("steel", 0) or 0)
    titanium = float(player.get("titanium", 0) or 0)
    steel_value = float(player.get("steelValue", 2) or 2)
    titanium_value = float(player.get("titaniumValue", 3) or 3)

    purchasing_power = mc
    if tags.get("Building"):
        purchasing_power += steel * steel_value
    if tags.get("Space"):
        purchasing_power += titanium * titanium_value
    if cost <= 0:
        return 1.0
    if purchasing_power >= cost:
        return 1.0
    return max(0.0, 1.0 - ((cost - purchasing_power) / 20.0))


def _score_card(card: Dict[str, Any], player: Dict[str, Any]) -> float:
    cost = _card_cost(card)
    vp = float(card.get("victoryPoints", 0) or 0)
    tags = _card_tags(card)
    mc = float(player.get("megaCredits", 0) or 0)
    steel = float(player.get("steel", 0) or 0)
    titanium = float(player.get("titanium", 0) or 0)
    steel_value = float(player.get("steelValue", 2) or 2)
    titanium_value = float(player.get("titaniumValue", 3) or 3)
    purchasing_power = mc
    if tags.get("Building"):
        purchasing_power += steel * steel_value
    if tags.get("Space"):
        purchasing_power += titanium * titanium_value
    affordable = 1.0 if purchasing_power >= cost else max(0.0, 1.0 - ((cost - purchasing_power) / 20.0))
    cheapness = max(0.0, 1.0 - min(cost / 40.0, 1.0))
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
    tiebreak = (sum(ord(ch) for ch in str(card.get("name", ""))) % 97) * 1e-5
    return (vp * 0.45) + (affordable * 1.4) + (cheapness * 0.8) + tag_score + tiebreak


def _enumerate_masks_python(cards: List[Dict[str, Any]], min_cards: int, max_cards: int, player: Dict[str, Any], limit: int) -> List[int]:
    import itertools

    enabled = [idx for idx, card in enumerate(cards) if not card.get("isDisabled", False)]
    if not enabled:
        return []
    min_pick = max(0, min(int(min_cards), len(enabled)))
    max_pick = max(min_pick, min(int(max_cards), len(enabled)))
    if max_pick <= 0:
        return [0] if min_pick == 0 else []

    scored = sorted(enabled, key=lambda idx: _score_card(cards[idx], player), reverse=True)
    candidate_budget = min(len(scored), max(12, min(max_pick, 14)))
    candidate_indices = sorted(scored[:candidate_budget])
    score_map = {idx: _score_card(cards[idx], player) for idx in candidate_indices}

    ranked: List[Tuple[float, int, int]] = []
    if min_pick == 0:
        ranked.append((0.0, 0, 0))
    for pick_count in range(max(1, min_pick), max_pick + 1):
        if pick_count > len(candidate_indices):
            break
        for combo in itertools.combinations(candidate_indices, pick_count):
            mask = 0
            combo_score = 0.0
            for idx in combo:
                mask |= (1 << idx)
                combo_score += score_map[idx]
            combo_score += 0.03 * float(pick_count)
            ranked.append((combo_score, pick_count, mask))
    ranked.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    out: List[int] = []
    seen = set()
    for _, _, mask in ranked:
        if mask in seen:
            continue
        seen.add(mask)
        out.append(mask)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_selection_payload(player: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for idx in range(16):
        cards.append(
            {
                "name": f"Card_{idx}",
                "calculatedCost": 4 + (idx % 11),
                "victoryPoints": idx % 3,
                "tags": {
                    "Building": bool(idx % 2 == 0),
                    "Space": bool(idx % 3 == 0),
                    "Science": bool(idx % 4 == 0),
                    "Earth": bool(idx % 5 == 0),
                    "Plant": bool(idx % 6 == 0),
                },
                "isDisabled": bool(idx in (13, 15)),
            }
        )
    return cards


def _time_it(fn, iterations: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed / float(iterations)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    player_path = root / "player_state" / "player.json"
    with player_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    player = payload.get("thisPlayer", {}) or {}
    cards = _build_selection_payload(player)

    affordability_cards = cards * 12

    py_aff_ms = _time_it(
        lambda: [_affordability_python(player, c) for c in affordability_cards],
        iterations=60,
    ) * 1000.0
    py_combo_ms = _time_it(
        lambda: _enumerate_masks_python(cards, 1, 4, player, 80),
        iterations=120,
    ) * 1000.0

    rust_aff_ms = None
    rust_combo_ms = None
    speedup_aff = None
    speedup_combo = None

    try:
        import rust_tfm_rl  # type: ignore

        player_json = json.dumps(player)
        cards_json = json.dumps(affordability_cards)
        selection_payload = {
            "cards": cards,
            "minCards": 1,
            "maxCards": 4,
            "playerState": {"thisPlayer": player},
        }

        rust_aff_ms = _time_it(
            lambda: rust_tfm_rl.can_afford_cards(player_json, cards_json),
            iterations=60,
        ) * 1000.0
        rust_combo_ms = _time_it(
            lambda: rust_tfm_rl.enumerate_card_selection_combos(json.dumps(selection_payload), 80),
            iterations=120,
        ) * 1000.0
        if rust_aff_ms > 0:
            speedup_aff = py_aff_ms / rust_aff_ms
        if rust_combo_ms > 0:
            speedup_combo = py_combo_ms / rust_combo_ms
    except Exception as exc:
        print(f"Rust module unavailable for compare run: {exc}")

    print("Hotspot benchmark results (ms per call)")
    print(f"- affordability_python_batch: {py_aff_ms:.3f}")
    print(f"- card_selection_python:      {py_combo_ms:.3f}")
    if rust_aff_ms is not None and rust_combo_ms is not None:
        print(f"- affordability_rust_batch:   {rust_aff_ms:.3f}")
        print(f"- card_selection_rust:        {rust_combo_ms:.3f}")
        print(f"- speedup_affordability:      {speedup_aff:.2f}x")
        print(f"- speedup_card_selection:     {speedup_combo:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
