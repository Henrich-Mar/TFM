"""Card-derivable award and milestone tracks, including random-MA pools.

Each track is a weighted sum of per-card features plus the amount of that sum
that "owns" the track (a milestone threshold, or a typical winning level for an
award). Features come from card metadata: tags, immediate production / stock /
tiles / global steps, card type, cost band, requirements and resource hosting.

Tracks that depend on board geometry, colonies, delegates, hand size or other
state a card does not carry are left out on purpose; they read as "no focus".
Terraformer, Benefactor, Planner and Tycoon are left out too: every plan
advances them, so they say nothing about commitment.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .card_catalog import resolve_behavior

_RESOURCES = ("megacredits", "steel", "titanium", "plants", "energy", "heat")
_PROD = {f"prod:{r}" for r in _RESOURCES}


def _t(*tags: str) -> Dict[str, float]:
    return {f"tag:{tag}": 1.0 for tag in tags}


def _w(**weights: float) -> Dict[str, float]:
    return {key.replace("__", ":"): float(value) for key, value in weights.items()}


#: normalized name -> (feature weights, owning amount)
TRACKS: Dict[str, Tuple[Dict[str, float], float]] = {
    # ---- milestones ----
    "agronomist": (_t("plant"), 4.0),
    "architect": (_t("city"), 3.0),
    "builder": (_t("building"), 7.5),
    "cforester": (_w(prod__plants=1), 3.0),
    "forester": (_w(prod__plants=1), 4.0),
    "diversifier": (_w(distinct_tags=1), 8.0),
    "ecologist": (_t("plant", "microbe", "animal"), 4.0),
    "economizer": (_w(prod__heat=1), 5.0),
    "energizer": (_w(prod__energy=1), 6.0),
    "engineer": (_w(prod__energy=1, prod__heat=1), 10.0),
    "farmer": (_w(host__animal=2, host__microbe=2), 5.0),
    "firestarter": (_w(stock__heat=1, prod__heat=3, prod__energy=2), 20.0),
    "fundraiser": (_w(prod__megacredits=1), 12.0),
    "gardener": (_w(greenery=1), 3.0),
    "hydrologist": (_w(oceans=1), 4.0),
    "landspecialist": (_w(special=1), 3.0),
    "legend": (_w(event=1), 4.5),
    "martian": (_t("mars"), 4.0),
    "mayor": (_w(city=1), 3.0),
    "metallurgist": (_w(prod__steel=1, prod__titanium=1), 6.0),
    "smith": (_w(prod__steel=1, prod__titanium=1), 6.0),
    "onegiantstep": (_t("moon"), 6.0),
    "philantropist": (_w(vp_card=1), 5.0),
    "planetologist": (_t("earth", "venus", "jovian"), 6.0),
    "producer": (_w(prod_total=1), 16.0),
    "researcher": (_t("science"), 4.0),
    "rimsettler": (_t("jovian"), 3.0),
    "spacefarer": (_t("space"), 5.0),
    "vspacefarer": (_t("space"), 4.0),
    "sponsor": (_w(cost20=1), 3.0),
    "tactician": (_w(has_req=1), 4.5),
    "terran": (_t("earth"), 5.5),
    "thawer": (_w(temperature=1), 5.0),
    "velectrician": (_t("power"), 4.0),
    "hoverlord": (_w(host__floater=2), 7.0),
    # ---- awards ----
    "aengineer": (_w(prod_alter=1), 6.0),
    "amanufacturer": (_w(active=1), 6.0),
    "azoologist": (_w(host__animal=2, host__microbe=2), 8.0),
    "administrator": (_w(no_tags=1), 4.0),
    "banker": (_w(prod__megacredits=1), 8.0),
    "biologist": (_t("plant", "animal", "microbe"), 6.0),
    "blacksmith": (_w(prod__steel=1, prod__titanium=1), 6.0),
    "botanist": (_w(prod__plants=1), 6.0),
    "celebrity": (_w(cost20=1), 4.0),
    "constructor": (_w(city=1), 4.0),
    "contractor": (_t("building"), 8.0),
    "cultivator": (_w(greenery=1), 4.0),
    "curator": (_w(max_tag=1), 6.0),
    "electrician": (_t("power"), 5.0),
    "excentric": (_w(host__animal=2, host__microbe=2, host__floater=2, host__other=2), 8.0),
    "forecaster": (_w(has_req=1), 6.0),
    "incorporator": (_w(cost10=1), 8.0),
    "industrialist": (_w(stock__steel=1, stock__energy=1, prod__steel=2, prod__energy=2), 15.0),
    "investor": (_t("earth"), 6.0),
    "kingpin": (_t("crime"), 4.0),
    "fullmoon": (_t("moon"), 5.0),
    "landlord": (_w(tiles=1), 6.0),
    "magnate": (_w(automated=1), 10.0),
    "manufacturer": (_w(prod__steel=1, prod__heat=1), 8.0),
    "metropolist": (_w(city=1), 4.0),
    "miner": (_w(stock__steel=1, stock__titanium=1, prod__steel=2, prod__titanium=2), 10.0),
    "mogul": (_w(prod_nonmc=1), 12.0),
    "naturalist": (_w(prod__plants=1, prod__heat=1), 8.0),
    "promoter": (_w(event=1), 6.0),
    "scientist": (_t("science"), 5.0),
    "spacebaron": (_t("space"), 6.0),
    "thermalist": (_w(stock__heat=1, prod__heat=3, prod__energy=2), 15.0),
    "traveller": (_t("earth", "jovian"), 6.0),
    "venuphile": (_t("venus"), 5.0),
    "voyager": (_t("jovian"), 4.0),
    "warmonger": (_w(attack=1), 4.0),
    "zoologist": (_w(host__animal=2), 6.0),
}

_CORP_PRODUCTION = re.compile(
    r"(\d+)\s+(m€|megacredit\w*|steel|titanium|plant\w*|energy|heat)\s+production", re.IGNORECASE
)


def normalize_name(name: Any) -> str:
    return re.sub(r"[^a-z]", "", str(name or "").lower())


def track_for(name: Any) -> Optional[Tuple[Dict[str, float], float]]:
    return TRACKS.get(normalize_name(name))


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def card_features(meta: Mapping[str, Any]) -> Dict[str, float]:
    """Per-card feature counts used by :data:`TRACKS`."""
    features: Dict[str, float] = {}

    def add(key: str, value: float) -> None:
        if value:
            features[key] = features.get(key, 0.0) + float(value)

    card_type = str(meta.get("type") or meta.get("cardType") or "").lower()
    is_event = "event" in card_type
    is_project = card_type in {"automated", "active", "event"}
    tags = [str(tag).lower() for tag in (meta.get("tags") or []) if str(tag).strip()]
    if not is_event:
        # Event tags are discarded once played; only the event pile counts.
        for tag in tags:
            add(f"tag:{tag}", 1.0)
    try:
        behavior = resolve_behavior(meta)
    except Exception:
        behavior = {}
    immediate = behavior.get("immediate") or {}
    production = {r: _num((immediate.get("production") or {}).get(r)) for r in _RESOURCES}
    if card_type == "corporation" and not any(production.values()):
        description = str(meta.get("description") or "")
        for amount, resource in _CORP_PRODUCTION.findall(description):
            key = resource.lower()
            key = "megacredits" if key.startswith(("m€", "megacredit")) else key
            key = "plants" if key.startswith("plant") else key
            if key in production:
                production[key] += _num(amount)
    for resource, value in production.items():
        add(f"prod:{resource}", value)
        add(f"stock:{resource}", _num((immediate.get("stock") or {}).get(resource)))
    if any(production.values()):
        add("prod_alter", 1.0)
    add("prod_total", sum(max(0.0, v) for v in production.values()))
    add("prod_nonmc", sum(max(0.0, v) for k, v in production.items() if k != "megacredits"))
    global_steps = immediate.get("global") or {}
    city = _num(immediate.get("place_city"))
    greenery = _num(immediate.get("place_greenery"))
    special = _num(immediate.get("place_special"))
    oceans = _num(global_steps.get("oceans"))
    add("city", city)
    add("greenery", greenery)
    add("special", special)
    add("oceans", oceans)
    add("tiles", city + greenery + special + oceans)
    add("temperature", _num(global_steps.get("temperature")))
    add("attack", 1.0 if _num(immediate.get("attack")) > 0 or _num(immediate.get("resource_remove")) > 0 else 0.0)
    if is_project:
        add("event", 1.0 if is_event else 0.0)
        add("active", 1.0 if card_type == "active" else 0.0)
        add("automated", 1.0 if card_type == "automated" else 0.0)
        cost = _num(meta.get("cost"))
        if not is_event:
            add("cost20", 1.0 if cost >= 20 else 0.0)
            add("cost10", 1.0 if cost <= 10 else 0.0)
            add("no_tags", 0.0 if tags else 1.0)
        add("has_req", 1.0 if (meta.get("requirements") or []) else 0.0)
        add("vp_card", 1.0 if _num(meta.get("vpPoints")) > 0 else 0.0)
    resource_type = str(meta.get("resourceType") or "").lower()
    if resource_type:
        host = resource_type if resource_type in {"animal", "microbe", "floater"} else "other"
        add(f"host:{host}", 1.0)
    return features


def plan_totals(features: Dict[str, float]) -> Dict[str, float]:
    """Add plan-level features that are not sums (distinct / max tag counts)."""
    totals = dict(features)
    tag_counts = [value for key, value in features.items() if key.startswith("tag:") and value > 0]
    totals["distinct_tags"] = float(len(tag_counts))
    totals["max_tag"] = float(max(tag_counts, default=0.0))
    return totals


def track_value(weights: Dict[str, float], totals: Dict[str, float]) -> float:
    return sum(weight * totals.get(feature, 0.0) for feature, weight in weights.items())
