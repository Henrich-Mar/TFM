"""Stable card catalog and 128-d static metadata for the V4 policy."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .v4_flags import v4_enabled

CARD_CAPACITY = 2048
METADATA_DIM = 128
UNKNOWN_ID = 0
PLANNER_SCHEMA_VERSION = "planner.card_aware.v1"
CATALOG_SCHEMA_VERSION = "card_catalog.v1"

TAG_ORDER: tuple[str, ...] = (
    "Building", "Space", "Science", "Power", "Earth", "Jovian",
    "Venus", "Plant", "Microbe", "Animal", "City", "Moon",
    "Mars", "Crime", "Wild", "Event",
)
TYPE_ORDER: tuple[str, ...] = (
    "automated", "active", "event", "prelude", "corporation",
    "ceo", "standard_project", "standard_action",
)
RESOURCE_TYPE_ORDER: tuple[str, ...] = (
    "Microbe", "Animal", "Science", "Floater", "Asteroid", "Data",
)
RESOURCE_BEHAVIOR_ORDER: tuple[str, ...] = (
    "none", "vp_accumulation", "conversion", "stealing", "adding",
)
PRODUCTION_ORDER: tuple[str, ...] = (
    "megacredits", "steel", "titanium", "plants", "energy", "heat", "microbes", "animals",
)

TAGS = slice(0, 16)
TYPES = slice(16, 24)
SCALARS = slice(24, 32)
REQUIREMENTS = slice(32, 48)
RESOURCES = slice(48, 60)
IMMEDIATE = slice(60, 88)
ACTION_EFFECTS = slice(88, 116)
FLAGS = slice(116, 128)

_CARD_FAMILIES = {"play_card", "card_subset", "card_prompt"}
_EFFECT_KEYS = (
    "production", "stock", "global", "draw", "place_city", "place_greenery",
    "place_special", "attack", "discount", "resource_add", "resource_remove",
)
_RESOURCE_WORDS = {
    "mega credit": "megacredits",
    "megacredit": "megacredits",
    "megacredits": "megacredits",
    "m€": "megacredits",
    "mc": "megacredits",
    "steel": "steel",
    "titanium": "titanium",
    "plant": "plants",
    "plants": "plants",
    "energy": "energy",
    "heat": "heat",
    "microbe": "microbes",
    "microbes": "microbes",
    "animal": "animals",
    "animals": "animals",
}
_CATALOG: Optional["CardCatalog"] = None


class CardEncodingError(RuntimeError):
    """A legal action referenced a card the candidate window cannot represent."""


class CardCatalogError(RuntimeError):
    """The card catalog cannot be built or does not match a checkpoint."""


def _clamp(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(float(value), limit))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _round_vector(values: Sequence[float]) -> List[float]:
    return [round(float(item), 6) for item in values]


def metadata_path() -> Path:
    env = str(os.getenv("TM_CARD_METADATA_PATH", "") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "card_metadata.json",
        here.parents[1] / "card_metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def catalog_path() -> Path:
    env = str(os.getenv("TM_CARD_CATALOG_PATH", "") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return metadata_path().with_name("card_catalog.v1.json")


def _empty_effect() -> Dict[str, Any]:
    return {
        "production": {name: 0.0 for name in PRODUCTION_ORDER},
        "stock": {name: 0.0 for name in PRODUCTION_ORDER},
        "global": {"temperature": 0.0, "oxygen": 0.0, "oceans": 0.0, "venus": 0.0},
        "draw": 0.0,
        "place_city": 0.0,
        "place_greenery": 0.0,
        "place_special": 0.0,
        "attack": 0.0,
        "discount": 0.0,
        "resource_add": 0.0,
        "resource_remove": 0.0,
    }


def _merge_effect(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _empty_effect()
    for key, value in base.items():
        merged[key] = dict(value) if isinstance(value, dict) else value
    for key in _EFFECT_KEYS:
        if key not in update:
            continue
        incoming = update[key]
        if isinstance(merged.get(key), dict) and isinstance(incoming, dict):
            for name, amount in incoming.items():
                canonical = _RESOURCE_WORDS.get(str(name).strip().lower(), str(name).strip().lower())
                if canonical in merged[key]:
                    merged[key][canonical] += _num(amount)
                elif key == "global" and str(name) in merged[key]:
                    merged[key][str(name)] += _num(amount)
        elif not isinstance(merged.get(key), dict):
            merged[key] = _num(incoming)
    return merged


def _resource_name(text: str) -> Optional[str]:
    cleaned = " ".join(str(text or "").strip().lower().replace("€", " megacredits ").split())
    cleaned = cleaned.replace("megacredits", "megacredits")
    for word, canonical in sorted(_RESOURCE_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if word in cleaned:
            return canonical
    return None


def _add_production(effect: Dict[str, Any], resource: Optional[str], amount: float) -> None:
    if resource in effect["production"]:
        effect["production"][resource] += float(amount)


def _add_stock(effect: Dict[str, Any], resource: Optional[str], amount: float) -> None:
    if resource in effect["stock"]:
        effect["stock"][resource] += float(amount)


def _parse_effect_text(text: str) -> Dict[str, Any]:
    effect = _empty_effect()
    body = " ".join(str(text or "").replace("M€", " megacredits ").split())
    if not body:
        return effect
    for match in re.finditer(
        r"(increase|decrease)\s+(?:your|any|an opponent's|opponents')\s+([a-z ]+?)\s+production\s+(?:by\s+)?(\d+)\s+steps?",
        body,
        flags=re.IGNORECASE,
    ):
        sign = -1.0 if match.group(1).lower() == "decrease" else 1.0
        opponent = "opponent" in match.group(0).lower()
        amount = sign * float(match.group(3))
        if opponent:
            effect["attack"] += abs(amount)
        else:
            _add_production(effect, _resource_name(match.group(2)), amount)
    for match in re.finditer(
        r"(?:gain|lose)\s+(\d+)\s+([a-z]+)",
        body,
        flags=re.IGNORECASE,
    ):
        sign = -1.0 if "lose" in match.group(0).lower() else 1.0
        _add_stock(effect, _resource_name(match.group(2)), sign * float(match.group(1)))
    for match in re.finditer(
        r"raise\s+(?:the\s+)?temperature\s+(\d+)\s+steps?",
        body,
        flags=re.IGNORECASE,
    ):
        effect["global"]["temperature"] += float(match.group(1))
    for match in re.finditer(
        r"(?:raise|increase)\s+(?:the\s+)?oxygen(?:\s+level)?\s+(\d+)\s+steps?",
        body,
        flags=re.IGNORECASE,
    ):
        effect["global"]["oxygen"] += float(match.group(1))
    for match in re.finditer(r"place\s+(?:an?\s+)?ocean", body, flags=re.IGNORECASE):
        effect["global"]["oceans"] += 1.0
    for match in re.finditer(r"place\s+(?:a\s+)?greenery", body, flags=re.IGNORECASE):
        effect["place_greenery"] += 1.0
    for match in re.finditer(r"place\s+(?:a\s+)?city", body, flags=re.IGNORECASE):
        effect["place_city"] += 1.0
    for match in re.finditer(r"place\s+(?:a\s+)?(colony|special tile|ocean tile)", body, flags=re.IGNORECASE):
        effect["place_special"] += 1.0
    for match in re.finditer(r"draw\s+(\d+)\s+cards?", body, flags=re.IGNORECASE):
        effect["draw"] += float(match.group(1))
    if re.search(r"draw\s+a\s+card", body, flags=re.IGNORECASE):
        effect["draw"] += 1.0
    for match in re.finditer(r"(\d+)\s+M?C\s+discount|discount\s+of\s+(\d+)", body, flags=re.IGNORECASE):
        effect["discount"] += float(match.group(1) or match.group(2) or 0)
    for match in re.finditer(
        r"add\s+(\d+)\s+(microbes?|animals?|floaters?|science|data|asteroids?)",
        body,
        flags=re.IGNORECASE,
    ):
        effect["resource_add"] += float(match.group(1))
    for match in re.finditer(r"remove\s+(?:up\s+to\s+)?(\d+)", body, flags=re.IGNORECASE):
        effect["resource_remove"] += float(match.group(1))
        if "opponent" in body.lower() or "any player" in body.lower():
            effect["attack"] += float(match.group(1))
    return effect


def parse_description(description: str) -> Dict[str, Any]:
    text = str(description or "")
    parts = re.split(r"\bAction:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    return {
        "immediate": _parse_effect_text(parts[0]),
        "action": _parse_effect_text(parts[1] if len(parts) > 1 else ""),
        "source": "description",
    }


def _structured_behavior(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        return None
    if isinstance(raw.get("immediate"), dict) or isinstance(raw.get("action"), dict):
        return {
            "immediate": _merge_effect(_empty_effect(), raw.get("immediate") or {}),
            "action": _merge_effect(_empty_effect(), raw.get("action") or {}),
            "source": "structured",
        }
    if any(key in raw for key in ("production", "stock", "global", "draw", "discount", "resource_add")):
        return {
            "immediate": _merge_effect(_empty_effect(), raw),
            "action": _empty_effect(),
            "source": "structured",
        }
    return None


def resolve_behavior(meta: Mapping[str, Any]) -> Dict[str, Any]:
    structured = _structured_behavior(meta.get("behavior"))
    if structured is not None:
        return structured
    parsed = parse_description(str(meta.get("description", "") or ""))
    parsed["source"] = "description"
    return parsed


def _tag_flags(meta: Mapping[str, Any]) -> List[float]:
    tags = meta.get("tags") or []
    present = set()
    if isinstance(tags, list):
        present = {str(item).strip().lower() for item in tags if str(item).strip()}
    elif isinstance(tags, dict):
        present = {str(key).strip().lower() for key, value in tags.items() if value}
    return [1.0 if name.lower() in present else 0.0 for name in TAG_ORDER]


def _type_flags(meta: Mapping[str, Any]) -> List[float]:
    card_type = str(meta.get("type", "") or "").strip().lower()
    return [1.0 if card_type == name else 0.0 for name in TYPE_ORDER]


def _requirement_vector(meta: Mapping[str, Any]) -> List[float]:
    values = [0.0] * 16
    requirements = [item for item in (meta.get("requirements") or []) if isinstance(item, dict)]
    tag_count = 0.0
    for requirement in requirements:
        if "oceans" in requirement:
            values[0] = max(values[0], _num(requirement.get("oceans")) / 9.0)
        if "oxygen" in requirement:
            values[1] = max(values[1], _num(requirement.get("oxygen")) / 14.0)
        if "temperature" in requirement:
            values[2] = max(values[2], (_num(requirement.get("temperature"), -30.0) + 30.0) / 50.0)
        if "venus" in requirement:
            values[3] = max(values[3], _num(requirement.get("venus")) / 30.0)
        if "tr" in requirement:
            values[4] = max(values[4], _num(requirement.get("tr")) / 40.0)
        if "cities" in requirement:
            values[5] = max(values[5], _num(requirement.get("cities")) / 8.0)
        if "greeneries" in requirement:
            values[6] = max(values[6], _num(requirement.get("greeneries")) / 6.0)
        if "colonies" in requirement:
            values[7] = max(values[7], _num(requirement.get("colonies")) / 8.0)
        if "floaters" in requirement:
            values[8] = max(values[8], _num(requirement.get("floaters")) / 8.0)
        if "production" in requirement:
            values[9] = 1.0
        if "corruption" in requirement:
            values[10] = max(values[10], _num(requirement.get("corruption")) / 10.0)
        if "party" in requirement or requirement.get("partyLeader") or requirement.get("chairman"):
            values[11] = 1.0
        if requirement.get("chairman"):
            values[12] = 1.0
        if requirement.get("max"):
            values[13] = 1.0
        if "tag" in requirement:
            tag_count = max(tag_count, _num(requirement.get("count"), 1.0) or 1.0)
    values[14] = min(tag_count / 5.0, 1.0)
    values[15] = min(len(requirements) / 6.0, 1.0)
    return [max(0.0, min(item, 1.0)) for item in values]


def _resource_vector(meta: Mapping[str, Any]) -> List[float]:
    resource_type = str(meta.get("resourceType", "") or "").strip().lower()
    flags = []
    matched = False
    for name in RESOURCE_TYPE_ORDER:
        present = resource_type == name.lower()
        matched = matched or present
        flags.append(1.0 if present else 0.0)
    flags.append(1.0 if resource_type and not matched else 0.0)
    behavior = str(meta.get("resourceBehavior", "none") or "none").strip().lower()
    if behavior not in RESOURCE_BEHAVIOR_ORDER:
        behavior = "none"
    flags.extend(1.0 if behavior == name else 0.0 for name in RESOURCE_BEHAVIOR_ORDER)
    return flags


def _effect_vector(effect: Mapping[str, Any]) -> List[float]:
    values: List[float] = []
    production = effect.get("production") or {}
    stock = effect.get("stock") or {}
    global_params = effect.get("global") or {}
    values.extend(_clamp(_num(production.get(name)) / 8.0) for name in PRODUCTION_ORDER)
    values.extend(_clamp(_num(stock.get(name)) / 8.0) for name in PRODUCTION_ORDER)
    values.extend([
        _clamp(_num(global_params.get("temperature")) / 8.0),
        _clamp(_num(global_params.get("oxygen")) / 8.0),
        _clamp(_num(global_params.get("oceans")) / 4.0),
        _clamp(_num(global_params.get("venus")) / 8.0),
        _clamp(_num(effect.get("draw")) / 4.0),
        _clamp(_num(effect.get("place_city")) / 2.0),
        _clamp(_num(effect.get("place_greenery")) / 2.0),
        _clamp(_num(effect.get("place_special")) / 2.0),
        _clamp(_num(effect.get("attack")) / 4.0),
        _clamp(_num(effect.get("discount")) / 10.0),
        _clamp(_num(effect.get("resource_add")) / 4.0),
        _clamp(_num(effect.get("resource_remove")) / 4.0),
    ])
    if len(values) != 28:
        raise CardCatalogError(f"effect vector must contain 28 fields, found {len(values)}")
    return values


def _has_production(effect: Mapping[str, Any]) -> bool:
    return any(abs(_num((effect.get("production") or {}).get(name))) > 0.0 for name in PRODUCTION_ORDER)


def _has_placement(effect: Mapping[str, Any]) -> bool:
    global_params = effect.get("global") or {}
    return any(
        abs(_num(item)) > 0.0
        for item in (
            effect.get("place_city"),
            effect.get("place_greenery"),
            effect.get("place_special"),
            global_params.get("oceans"),
        )
    )


def metadata_vector(meta: Mapping[str, Any]) -> List[float]:
    behavior = resolve_behavior(meta)
    immediate = behavior["immediate"]
    action = behavior["action"]
    numeric_vp = meta.get("victoryPoints", 0)
    vp = _num(numeric_vp) if not isinstance(numeric_vp, (dict, list, str)) else 0.0
    cost = _num(meta.get("cost", 0))
    starting_mc = _num(meta.get("startingMegaCredits", 0))
    card_cost = _num(meta.get("cardCost", 0))
    tr = _num(meta.get("tr", meta.get("startingTR", meta.get("terraformRating", 0))))
    production_mass = sum(abs(_num((immediate.get("production") or {}).get(name))) for name in PRODUCTION_ORDER)
    scalars = [
        min(cost / 50.0, 1.0),
        _clamp(vp / 10.0),
        1.0 if str(meta.get("resourceType", "") or "").strip() else 0.0,
        min(starting_mc / 80.0, 1.0),
        min(card_cost / 15.0, 1.0),
        min(tr / 30.0, 1.0),
        min(_num(immediate.get("discount")) / 20.0, 1.0),
        min(production_mass / 10.0, 1.0),
    ]
    resource_behavior = str(meta.get("resourceBehavior", "none") or "none").strip().lower()
    description = str(meta.get("description", "") or "")
    flags = [
        1.0 if meta.get("hasAction") or re.search(r"\bAction:", description, flags=re.IGNORECASE) else 0.0,
        1.0 if meta.get("requirements") else 0.0,
        1.0 if meta.get("resourceActionAddsToAnyCard") else 0.0,
        1.0 if meta.get("resourceActionTargetsOpponent") else 0.0,
        1.0 if meta.get("resourceActionRemovesResources") else 0.0,
        1.0 if behavior.get("source") == "description" else 0.0,
        1.0 if behavior.get("source") == "structured" else 0.0,
        1.0 if abs(vp) > 0.0 else 0.0,
        1.0 if _has_production(immediate) or _has_production(action) else 0.0,
        1.0 if _has_placement(immediate) or _has_placement(action) else 0.0,
        1.0 if resource_behavior not in {"", "none"} else 0.0,
        1.0 if str(meta.get("name", "") or "").strip() and str(meta.get("type", "") or "").strip() else 0.0,
    ]
    vector = (
        _tag_flags(meta)
        + _type_flags(meta)
        + scalars
        + _requirement_vector(meta)
        + _resource_vector(meta)
        + _effect_vector(immediate)
        + _effect_vector(action)
        + flags
    )
    if len(vector) != METADATA_DIM:
        raise CardCatalogError(f"metadata vector must contain {METADATA_DIM} values, found {len(vector)}")
    return _round_vector(vector)


def _card_name(card: Mapping[str, Any]) -> str:
    return str(card.get("name", "") or "").strip()


def referenced_card_names(descriptor: Mapping[str, Any]) -> List[str]:
    family = str(descriptor.get("family", "") or "")
    if family not in _CARD_FAMILIES:
        return []
    names: List[str] = []
    decoded = descriptor.get("decoded_action")
    if isinstance(decoded, dict):
        raw_card = decoded.get("card")
        if isinstance(raw_card, str) and raw_card.strip():
            names.append(raw_card.strip())
        elif isinstance(raw_card, dict) and _card_name(raw_card):
            names.append(_card_name(raw_card))
        for card in decoded.get("cards", []) or []:
            if isinstance(card, str) and card.strip():
                names.append(card.strip())
            elif isinstance(card, dict) and _card_name(card):
                names.append(_card_name(card))
    label_name = str(descriptor.get("card_name", "") or "").strip()
    if label_name:
        names.append(label_name)
    unique: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


class CardCatalog:
    def __init__(self, entries: Sequence[Mapping[str, Any]], vectors: Mapping[int, Sequence[float]], sha256: str):
        self.entries = list(entries)
        self.sha256 = str(sha256)
        self.id_by_name = {str(item["name"]): int(item["id"]) for item in self.entries}
        self._vectors = {int(card_id): list(vector) for card_id, vector in vectors.items()}
        if UNKNOWN_ID in self.id_by_name.values():
            raise CardCatalogError("catalog id 0 is reserved for unknown cards")
        if any(card_id <= 0 or card_id >= CARD_CAPACITY for card_id in self.id_by_name.values()):
            raise CardCatalogError(f"catalog ids must stay inside 1..{CARD_CAPACITY - 1}")

    def id_for_name(self, name: str) -> int:
        return int(self.id_by_name.get(str(name or "").strip(), UNKNOWN_ID))

    def metadata_vector_for_id(self, card_id: int) -> List[float]:
        if int(card_id) <= 0:
            return [0.0] * METADATA_DIM
        vector = self._vectors.get(int(card_id))
        if vector is None:
            raise CardCatalogError(f"catalog id {card_id} has no metadata vector")
        return list(vector)

    def metadata_matrix(self) -> np.ndarray:
        matrix = np.zeros((CARD_CAPACITY, METADATA_DIM), dtype=np.float32)
        for card_id, vector in self._vectors.items():
            matrix[int(card_id)] = np.asarray(vector, dtype=np.float32)
        return matrix

    def behavior_for_name(self, name: str, metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        meta = metadata.get(str(name), {})
        return resolve_behavior(meta if isinstance(meta, Mapping) else {})


def _hash_catalog(entries: Sequence[Mapping[str, Any]], vectors: Mapping[int, Sequence[float]]) -> str:
    payload = {
        "capacity": CARD_CAPACITY,
        "unknown_id": UNKNOWN_ID,
        "cards": [
            {
                "id": int(item["id"]),
                "name": str(item["name"]),
                "metadata": list(vectors[int(item["id"])]),
            }
            for item in entries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_catalog(metadata: Mapping[str, Mapping[str, Any]]) -> CardCatalog:
    if not isinstance(metadata, Mapping):
        raise CardCatalogError("card metadata must be an object keyed by card name")
    names = sorted(str(name) for name in metadata.keys())
    if len(names) > CARD_CAPACITY - 1:
        raise CardCatalogError(
            f"card catalog capacity is {CARD_CAPACITY - 1} named cards; metadata contains {len(names)}"
        )
    entries = []
    vectors: Dict[int, List[float]] = {}
    for offset, name in enumerate(names, start=1):
        meta = metadata.get(name) or {}
        if not isinstance(meta, Mapping):
            raise CardCatalogError(f"metadata for {name!r} must be an object")
        vector = metadata_vector(meta)
        entries.append({"id": offset, "name": name})
        vectors[offset] = vector
    digest = _hash_catalog(entries, vectors)
    return CardCatalog(entries, vectors, digest)


def load_metadata(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    source = Path(path) if path is not None else metadata_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CardCatalogError(f"card metadata must be a JSON object: {source}")
    return payload


def get_catalog(force_reload: bool = False) -> CardCatalog:
    global _CATALOG
    if _CATALOG is None or force_reload:
        _CATALOG = build_catalog(load_metadata())
    return _CATALOG


def write_catalog(path: Optional[Path] = None) -> Path:
    catalog = get_catalog(force_reload=True)
    destination = Path(path) if path is not None else catalog_path()
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "capacity": CARD_CAPACITY,
        "unknown_id": UNKNOWN_ID,
        "sha256": catalog.sha256,
        "cards": list(catalog.entries),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def validate_checkpoint_catalog(checkpoint: Mapping[str, Any]) -> None:
    if not v4_enabled():
        return
    version = checkpoint.get("experiment_version")
    if version != "tfm-rl-v4":
        raise CardCatalogError(f"TFM RL v4 refuses incompatible checkpoint marker {version!r}")
    expected = get_catalog().sha256
    actual = str(checkpoint.get("card_catalog_sha256") or "")
    if actual != expected:
        raise CardCatalogError(
            f"TFM RL v4 refuses catalog hash {actual or '<missing>'} against current catalog {expected}"
        )


def bind_action_card_mask(
    hand_cards: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    *,
    hand_limit: int,
    catalog: CardCatalog,
) -> tuple[np.ndarray, np.ndarray]:
    """Map play/subset actions onto prompt-first candidate tokens.

    Prompt cards are already expected at the front of ``hand_cards``. Cards
    past ``hand_limit`` are not encoded; a legal reference to one fails.
    """
    limit = max(0, int(hand_limit))
    kept = [card for card in hand_cards if isinstance(card, Mapping)][:limit]
    omitted = [card for card in hand_cards if isinstance(card, Mapping)][limit:]
    omitted_names = {_card_name(card) for card in omitted if _card_name(card)}
    index: Dict[str, int] = {}
    for position, card in enumerate(kept):
        name = _card_name(card)
        if name and name not in index:
            index[name] = position
    rows: List[List[bool]] = []
    for descriptor in descriptors:
        family = str(descriptor.get("family", "") or "")
        names = referenced_card_names(descriptor)
        mask = [False] * len(kept)
        if family == "play_card" and len(names) != 1:
            raise CardEncodingError("play_card must reference exactly one card")
        if family in _CARD_FAMILIES:
            for name in names:
                if name in omitted_names or name not in index:
                    reason = (
                        f"omitted by the {limit}-card candidate limit"
                        if name in omitted_names
                        else "not present in the encoded candidate tokens"
                    )
                    raise CardEncodingError(f"legal action references card {name!r} which is {reason}")
                mask[index[name]] = True
            if family == "play_card" and sum(1 for item in mask if item) != 1:
                raise CardEncodingError("play_card mask must select exactly one candidate token")
        rows.append(mask)
    if rows:
        action_mask = np.asarray(rows, dtype=np.bool_)
    else:
        action_mask = np.zeros((0, len(kept)), dtype=np.bool_)
    hand_ids = np.asarray([catalog.id_for_name(_card_name(card)) for card in kept], dtype=np.int64)
    return hand_ids, action_mask


def iter_subset_masks(card_count: int) -> Iterable[List[bool]]:
    total = 1 << max(0, int(card_count))
    for pattern in range(total):
        yield [bool(pattern & (1 << bit)) for bit in range(int(card_count))]


def main() -> None:
    destination = write_catalog()
    catalog = get_catalog()
    print(json.dumps({
        "path": str(destination),
        "cards": len(catalog.entries),
        "capacity": CARD_CAPACITY,
        "sha256": catalog.sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
