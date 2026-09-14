import json
import hashlib
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from models.planner_common import PLANNER_OPPORTUNITY_LIMIT, planner_aux_layout
    from models.state_encoder import StateEncoder
except Exception:
    PLANNER_OPPORTUNITY_LIMIT = 12
    planner_aux_layout = None
    StateEncoder = None


_REQUEST_LOCK = threading.Lock()
_PENDING_CAPTURE_REQUESTS: Dict[str, Dict[str, Any]] = {}

_CARD_TAG_RESOLVER: Any = None


def legal_action_descriptor_fingerprint(descriptors: List[Dict[str, Any]]) -> str:
    """Hash the canonical, non-model-specific part of a prompt action catalog."""
    canonical = [
        {
            "action_index": _safe_int(row.get("action_index", -1), -1),
            "family": str(row.get("family", "") or ""),
            "label": str(row.get("label", "") or ""),
            "decoded_action": row.get("decoded_action", {}),
        }
        for row in (descriptors or [])
    ]
    encoded = json.dumps(
        _json_safe(canonical),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_action_payload(payload: Any) -> str:
    """Order-invariant identity for replaying a recorded choice against a new catalog."""
    if not isinstance(payload, dict):
        return ""
    normalized = dict(payload)
    cards = normalized.get("cards")
    if isinstance(cards, list):
        normalized["cards"] = sorted(str(item) for item in cards)
    return json.dumps(_json_safe(normalized), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def remap_action_index_by_payload(
    descriptors: List[Dict[str, Any]],
    payload: Any,
) -> Optional[int]:
    target = canonical_action_payload(payload)
    if not target:
        return None
    matches = [
        _safe_int(row.get("action_index", -1), -1)
        for row in (descriptors or [])
        if canonical_action_payload(row.get("decoded_action", {})) == target
    ]
    matches = [item for item in matches if item >= 0]
    if len(matches) != 1:
        return None
    return int(matches[0])


def _resolve_card_tags(card: Any) -> List[str]:
    raw = card.get("tags", []) or [] if isinstance(card, dict) else []
    if isinstance(raw, dict):
        resolved = [str(tag).strip() for tag, present in raw.items() if present]
    elif isinstance(raw, list):
        resolved = [str(tag).strip() for tag in raw if str(tag or "").strip()]
    else:
        resolved = []
    if resolved:
        return resolved
    name = str(card.get("name", "") or "").strip() if isinstance(card, dict) else ""
    if not name or StateEncoder is None:
        return []
    global _CARD_TAG_RESOLVER
    if _CARD_TAG_RESOLVER is None:
        try:
            _CARD_TAG_RESOLVER = StateEncoder()
        except Exception:
            _CARD_TAG_RESOLVER = object()
    if not _card_tag_resolver_ready():
        return []
    try:
        tag_map = _CARD_TAG_RESOLVER._get_card_tags(name, fallback={}) or {}
    except Exception:
        return []
    return sorted(str(tag) for tag, present in tag_map.items() if present)

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SPACE_BONUS_LABELS = {
    0: "Titanium", 1: "Steel", 2: "Plant", 3: "Card", 4: "Heat", 5: "Ocean",
    6: "M€", 7: "Animal", 8: "Microbe", 9: "Energy", 10: "Data", 11: "Science",
    12: "Energy production", 13: "Temperature", 15: "Asteroid", 16: "Delegate",
    17: "Colony", 18: "Temperature (pay 4 M€)",
}
_TILE_TYPE_LABELS = {
    0: "Greenery", 1: "Ocean", 2: "City", 3: "Capital", 4: "Commercial District",
    5: "Ecological Zone", 6: "Industrial Center", 7: "Lava Flows", 8: "Mining Area",
    9: "Mining Rights", 10: "Mohole Area", 11: "Natural Preserve", 12: "Nuclear Zone",
    13: "Restricted Area", 14: "Deimos Down", 15: "Great Dam", 16: "Magnetic Field Generators",
    17: "Biofertilizer Facility", 18: "Metallic Asteroid", 19: "Solar Farm", 20: "Ocean City",
    21: "Ocean Farm", 22: "Ocean Sanctuary", 23: "Mild Dust Storm", 24: "Severe Dust Storm",
    25: "Mild Erosion", 26: "Severe Erosion", 27: "Mining (Steel)", 28: "Mining (Titanium)",
    29: "Moon Mine", 30: "Moon Habitat", 31: "Moon Road", 32: "Luna Trade Station",
    33: "Luna Mining Hub", 34: "Luna Train Station", 35: "Lunar Mine Urbanization",
    36: "Wetlands", 37: "Red City", 38: "Martian Nature Wonders", 39: "Crashlanding",
    40: "Mars Nomads", 41: "Rey Skywalker", 42: "Man-made Volcano", 43: "New Holland",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _message_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("message", "")
    return str(value or "").strip()


def _space_bonus_label(bonus: Any) -> str:
    try:
        numeric = int(bonus)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return _SPACE_BONUS_LABELS.get(numeric, f"Unknown bonus ({numeric})")
    if isinstance(bonus, dict):
        return _message_text(bonus.get("name", bonus.get("type", bonus.get("bonus", "Unknown bonus"))))
    return _message_text(bonus).replace("_", " ").title() or "Unknown bonus"


def _tile_type_label(tile_type: Any) -> str:
    if tile_type is None or _message_text(tile_type) == "":
        return "Empty"
    try:
        numeric = int(tile_type)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return _TILE_TYPE_LABELS.get(numeric, f"Unknown tile ({numeric})")
    return _message_text(tile_type).replace("_", " ").title()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)
    return str(value)


def _sanitize_id(value: Any) -> str:
    text = _SAFE_ID_RE.sub("-", str(value or "").strip()).strip("-._")
    return text or "snapshot"


def get_snapshot_root() -> str:
    env_path = str(os.getenv("DECISION_SNAPSHOT_DIR", "") or "").strip()
    if env_path:
        root = Path(env_path)
    else:
        root = Path(__file__).resolve().parent / "debug_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def get_annotation_root() -> str:
    env_path = str(os.getenv("V2_TEACHER_ANNOTATION_DIR", "") or "").strip()
    root = Path(env_path) if env_path else Path(get_snapshot_root()) / "annotations"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def create_capture_request(
    agent_id: Optional[str] = None,
    game_id: Optional[str] = None,
    player_id: Optional[str] = None,
    note: str = "",
    include_state_vector: bool = False,
) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "created_at": _utc_now_iso(),
        "status": "pending",
        "agent_id": str(agent_id or "").strip() or None,
        "game_id": str(game_id or "").strip() or None,
        "player_id": str(player_id or "").strip() or None,
        "note": str(note or "").strip(),
        "include_state_vector": bool(include_state_vector),
    }
    with _REQUEST_LOCK:
        _PENDING_CAPTURE_REQUESTS[request_id] = dict(request)
    return dict(request)


def has_pending_capture_request(agent_id: Optional[str] = None) -> bool:
    agent_id_norm = str(agent_id or "").strip()
    with _REQUEST_LOCK:
        for request in _PENDING_CAPTURE_REQUESTS.values():
            if str(request.get("status", "")) != "pending":
                continue
            request_agent = str(request.get("agent_id", "") or "").strip()
            if request_agent and agent_id_norm and request_agent != agent_id_norm:
                continue
            return True
    return False


def reserve_pending_capture_request(
    agent_id: Optional[str],
    game_id: Optional[str],
    player_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    agent_id_norm = str(agent_id or "").strip()
    game_id_norm = str(game_id or "").strip()
    player_id_norm = str(player_id or "").strip()

    with _REQUEST_LOCK:
        for request_id, request in list(_PENDING_CAPTURE_REQUESTS.items()):
            if str(request.get("status", "")) != "pending":
                continue

            req_agent = str(request.get("agent_id", "") or "").strip()
            req_game = str(request.get("game_id", "") or "").strip()
            req_player = str(request.get("player_id", "") or "").strip()

            if req_agent and req_agent != agent_id_norm:
                continue
            if req_game and req_game != game_id_norm:
                continue
            if req_player and req_player != player_id_norm:
                continue

            request["status"] = "capturing"
            request["reserved_at"] = _utc_now_iso()
            request["matched_agent_id"] = agent_id_norm or None
            request["matched_game_id"] = game_id_norm or None
            request["matched_player_id"] = player_id_norm or None
            _PENDING_CAPTURE_REQUESTS[request_id] = dict(request)
            return dict(request)
    return None


def complete_capture_request(request_id: str, snapshot_id: str, snapshot_path: str) -> Optional[Dict[str, Any]]:
    with _REQUEST_LOCK:
        request = _PENDING_CAPTURE_REQUESTS.get(str(request_id))
        if request is None:
            return None
        request["status"] = "captured"
        request["completed_at"] = _utc_now_iso()
        request["snapshot_id"] = str(snapshot_id)
        request["snapshot_path"] = str(snapshot_path)
        _PENDING_CAPTURE_REQUESTS[request_id] = dict(request)
        return dict(request)


def fail_capture_request(request_id: str, error: str) -> Optional[Dict[str, Any]]:
    with _REQUEST_LOCK:
        request = _PENDING_CAPTURE_REQUESTS.get(str(request_id))
        if request is None:
            return None
        request["status"] = "failed"
        request["failed_at"] = _utc_now_iso()
        request["error"] = str(error or "").strip()
        _PENDING_CAPTURE_REQUESTS[request_id] = dict(request)
        return dict(request)


def list_capture_requests() -> List[Dict[str, Any]]:
    with _REQUEST_LOCK:
        requests = [dict(item) for item in _PENDING_CAPTURE_REQUESTS.values()]
    requests.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return requests


def reset_capture_state() -> None:
    with _REQUEST_LOCK:
        _PENDING_CAPTURE_REQUESTS.clear()


def _card_tag_resolver_ready() -> bool:
    return StateEncoder is not None and isinstance(_CARD_TAG_RESOLVER, StateEncoder)


def _resolve_card_victory_points(card: Dict[str, Any]) -> float:
    direct = _safe_float(card.get("victoryPoints", 0.0))
    if direct:
        return direct
    name = str(card.get("name", "") or "").strip()
    if not name or not _card_tag_resolver_ready():
        return 0.0
    meta = (_CARD_TAG_RESOLVER.card_metadata_by_name or {}).get(name, {}) or {}
    try:
        return float(meta.get("victoryPoints", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_card_type(card: Dict[str, Any]) -> str:
    direct = str(card.get("type", "") or "").strip()
    if direct:
        return direct
    name = str(card.get("name", "") or "").strip()
    if not name or not _card_tag_resolver_ready():
        return ""
    meta = (_CARD_TAG_RESOLVER.card_metadata_by_name or {}).get(name, {}) or {}
    return str(meta.get("type", "") or "").strip()


def _card_summary(card: Any) -> Dict[str, Any]:
    if not isinstance(card, dict):
        return {"label": _message_text(card)}
    tags = _resolve_card_tags(card)
    calculated_cost = _safe_float(card.get("calculatedCost", card.get("cost", 0.0)))
    base_cost = _safe_float(card.get("cost", calculated_cost))
    summary = {
        "name": str(card.get("name", "") or "").strip(),
        "cost": calculated_cost,
        "calculated_cost": calculated_cost,
        "base_cost": base_cost,
        "tags": [str(tag or "").strip() for tag in tags if str(tag or "").strip()],
        "victory_points": _resolve_card_victory_points(card),
        "type": _resolve_card_type(card),
        "has_action": bool(card.get("hasAction", False)),
        "disabled": bool(card.get("isDisabled", False)),
        "label": str(card.get("name", "") or card.get("type", "") or "card").strip(),
    }
    for key in (
        "requirements",
        "requirement_plan",
        "plan_summary",
        "reachability_score",
        "readiness_score",
        "all_satisfied",
        "blocking_count",
        "server_override",
        "masked_by_server",
    ):
        if key in card:
            summary[key] = card.get(key)
    description = _message_text(card.get("description", ""))
    if description:
        summary["description"] = description
    return summary


def _resolved_player_hand_cards(player_state: Optional[Dict[str, Any]], player: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not isinstance(player_state, dict):
        player_state = {}
    if not isinstance(player, dict):
        player = (player_state.get("thisPlayer", {}) or {}) if isinstance(player_state, dict) else {}

    for raw_cards in (
        player_state.get("cardsInHand", []) if isinstance(player_state, dict) else [],
        player.get("cardsInHand", []) if isinstance(player, dict) else [],
    ):
        if not isinstance(raw_cards, list):
            continue
        cards = [card for card in raw_cards if isinstance(card, dict)]
        if cards:
            return cards
    return []


def _resolved_player_hand_count(player_state: Optional[Dict[str, Any]], player: Optional[Dict[str, Any]] = None) -> int:
    cards = _resolved_player_hand_cards(player_state, player)
    if cards:
        return int(len(cards))
    if isinstance(player, dict):
        count = player.get("cardsInHandNbr", None)
        if count is not None:
            return _safe_int(count)
    return 0


def _player_summary(player: Any, hand_count_override: Optional[int] = None) -> Dict[str, Any]:
    if not isinstance(player, dict):
        return {"label": _message_text(player)}
    productions = {
        "mc": _safe_float(player.get("megaCreditProduction", 0.0)),
        "steel": _safe_float(player.get("steelProduction", 0.0)),
        "titanium": _safe_float(player.get("titaniumProduction", 0.0)),
        "plants": _safe_float(player.get("plantProduction", 0.0)),
        "energy": _safe_float(player.get("energyProduction", 0.0)),
        "heat": _safe_float(player.get("heatProduction", 0.0)),
    }
    resources = {
        "mc": _safe_float(player.get("megaCredits", 0.0)),
        "steel": _safe_float(player.get("steel", 0.0)),
        "titanium": _safe_float(player.get("titanium", 0.0)),
        "plants": _safe_float(player.get("plants", 0.0)),
        "energy": _safe_float(player.get("energy", 0.0)),
        "heat": _safe_float(player.get("heat", 0.0)),
    }
    return {
        "id": str(player.get("id", "") or "").strip(),
        "name": str(player.get("name", "") or "").strip(),
        "color": str(player.get("color", "") or "").strip(),
        "terraform_rating": _safe_float(player.get("terraformRating", 0.0)),
        "resources": resources,
        "production": productions,
        "hand_count": _safe_int(hand_count_override if hand_count_override is not None else len(player.get("cardsInHand", []) or [])),
        "card_cost": _safe_float(player.get("cardCost", 3.0), 3.0),
        "tableau_count": _safe_int(len(player.get("tableau", []) or [])),
        "cities_count": _safe_int(len(player.get("cityCards", []) or [])),
        "greeneries_count": _safe_int(len(player.get("greeneryCards", []) or [])),
    }


def _is_same_player(candidate: Any, this_player: Any, player_id: Optional[str] = None) -> bool:
    """Identify thisPlayer in the public players list when ids are redacted."""
    if not isinstance(candidate, dict) or not isinstance(this_player, dict):
        return False

    candidate_id = str(candidate.get("id", "") or "").strip()
    this_id = str(this_player.get("id", "") or "").strip()
    requested_id = str(player_id or "").strip()
    if candidate_id and (candidate_id == this_id or candidate_id == requested_id):
        return True

    candidate_color = str(candidate.get("color", "") or "").strip().lower()
    this_color = str(this_player.get("color", "") or "").strip().lower()
    if candidate_color and this_color:
        return candidate_color == this_color

    candidate_name = str(candidate.get("name", "") or "").strip()
    this_name = str(this_player.get("name", "") or "").strip()
    return bool(candidate_name and this_name and candidate_name == this_name)


def _option_summary(option: Any, index: int) -> Dict[str, Any]:
    if not isinstance(option, dict):
        return {"index": int(index), "label": _message_text(option)}
    summary = {
        "index": int(index),
        "type": str(option.get("type", "") or "").strip(),
        "title": _message_text(option.get("title", "")),
        "label": _message_text(option.get("title", "")) or str(option.get("type", "") or f"option-{index}"),
        "cards_count": _safe_int(len(option.get("cards", []) or [])),
        "options_count": _safe_int(len(option.get("options", []) or [])),
    }
    if option.get("cards"):
        summary["cards_preview"] = [_card_summary(card) for card in list(option.get("cards", [])[:8])]
    if option.get("options"):
        summary["options_preview"] = [
            {
                "index": int(i),
                "label": _message_text(child.get("title", "")) or str(child.get("type", "") or f"option-{i}"),
                "type": str(child.get("type", "") or "").strip(),
            }
            for i, child in enumerate(list(option.get("options", [])[:8]))
            if isinstance(child, dict)
        ]
    return summary


def _space_summary(space: Any, index: int) -> Dict[str, Any]:
    if not isinstance(space, dict):
        return {"index": int(index), "label": _message_text(space)}
    bonuses = [_space_bonus_label(item) for item in (space.get("bonus", []) or [])]
    return {
        "index": int(index),
        "id": str(space.get("id", "") or space.get("spaceId", "") or "").strip(),
        "name": str(space.get("name", "") or space.get("tileType", "") or f"space-{index}").strip(),
        "space_type": str(space.get("spaceType", "") or space.get("tileType", "") or "").strip(),
        "x": _safe_int(space.get("x", 0)),
        "y": _safe_int(space.get("y", 0)),
        "tile_type": space.get("tileType"),
        "tile_label": _tile_type_label(space.get("tileType")),
        "bonus": list(space.get("bonus", []) or []),
        "bonus_labels": bonuses,
        "bonus_summary": ", ".join(bonuses) if bonuses else "None",
        "disabled": bool(space.get("isDisabled", False)),
        "owner": str(space.get("color", space.get("playerColor", space.get("owner", ""))) or "").strip(),
    }


def _candidate_space_id(space: Any) -> str:
    if isinstance(space, dict):
        return str(space.get("id", space.get("spaceId", "")) or "").strip()
    return _message_text(space).strip()


def _prompt_space_candidates(waiting_for: Dict[str, Any]) -> List[Any]:
    """Return space candidates from the current prompt and any nested action branch."""
    candidates: List[Any] = []
    seen: set[str] = set()

    def append_spaces(spaces: Any) -> None:
        for space in spaces or []:
            identity = _candidate_space_id(space)
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            candidates.append(space)

    def walk(prompt: Any) -> None:
        if not isinstance(prompt, dict):
            return
        append_spaces(prompt.get("availableSpaces", prompt.get("spaces", [])) or [])
        for option in prompt.get("options", []) or []:
            walk(option)

    walk(waiting_for)
    return candidates


def _space_ids_from_action(action: Any) -> set[str]:
    ids: set[str] = set()
    stack = [action] if isinstance(action, dict) else []
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if str(current.get("type", "") or "").lower() in ("space", "selectspace"):
            space_id = str(current.get("spaceId", current.get("id", "")) or "").strip()
            if space_id:
                ids.add(space_id)
        response = current.get("response")
        if isinstance(response, dict):
            stack.append(response)
        for response in current.get("responses", []) or []:
            if isinstance(response, dict):
                stack.append(response)
    return ids


_PAYMENT_RESOURCE_KEYS = {
    "megaCredits": "mc",
    "steel": "steel",
    "titanium": "titanium",
    "plants": "plants",
    "energy": "energy",
    "heat": "heat",
}


def _action_payment_projection(player: Dict[str, Any], action: Any) -> Dict[str, Any]:
    """Summarize the known payment while retaining the pre-action decision state."""
    payment = {target: 0.0 for target in _PAYMENT_RESOURCE_KEYS.values()}
    stack = [action] if isinstance(action, dict) else []
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        raw_payment = current.get("payment", {}) or {}
        if isinstance(raw_payment, dict):
            for source, target in _PAYMENT_RESOURCE_KEYS.items():
                payment[target] += _safe_float(raw_payment.get(source, 0.0))
        response = current.get("response")
        if isinstance(response, dict):
            stack.append(response)
        for response in current.get("responses", []) or []:
            if isinstance(response, dict):
                stack.append(response)

    resources_before = dict(_player_summary(player).get("resources", {}) or {})
    return {
        "state_timing": "before_action",
        "payment": payment,
        "resources_after_payment": {
            resource: float(resources_before.get(resource, 0.0)) - float(payment.get(resource, 0.0))
            for resource in _PAYMENT_RESOURCE_KEYS.values()
        },
    }


def _board_surface_space(
    space: Any,
    legal_ids: set[str],
    chosen_ids: set[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(space, dict):
        return None
    try:
        x, y = int(space.get("x")), int(space.get("y"))
    except (TypeError, ValueError):
        return None
    if x < 0 or y < 0:
        return None
    space_id = str(space.get("id", space.get("spaceId", "")) or "").strip()
    if not space_id:
        return None
    bonuses = list(space.get("bonus", []) or [])
    bonus_labels = [_space_bonus_label(item) for item in bonuses]
    return {
        "id": space_id,
        "x": x,
        "y": y,
        "space_type": str(space.get("spaceType", "") or "").strip(),
        "tile_type": space.get("tileType"),
        "tile_label": _tile_type_label(space.get("tileType")),
        "owner": str(space.get("color", space.get("playerColor", space.get("owner", ""))) or "").strip(),
        "bonus": bonuses,
        "bonus_labels": bonus_labels,
        "bonus_summary": ", ".join(bonus_labels) if bonus_labels else "None",
        "legal_candidate": space_id in legal_ids,
        "chosen": space_id in chosen_ids,
    }


def _snapshot_board_surface(
    game: Dict[str, Any],
    waiting_for: Dict[str, Any],
    action_input: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    candidate_spaces = _prompt_space_candidates(waiting_for)
    legal_ids = {_candidate_space_id(item) for item in candidate_spaces}
    legal_ids.discard("")
    chosen_ids = _space_ids_from_action(action_input)

    def summarize(spaces: Any) -> List[Dict[str, Any]]:
        result = [_board_surface_space(space, legal_ids, chosen_ids) for space in (spaces or [])]
        return [item for item in result if item is not None]

    moon = game.get("moon", {}) or {}
    mars_spaces = summarize(game.get("spaces", []))
    moon_spaces = summarize(moon.get("spaces", []))
    known_ids = {str(item["id"]) for item in mars_spaces + moon_spaces}
    moon_ids = {str(item["id"]) for item in moon_spaces}
    # Some response variants expose a legal location before it appears in the
    # public board list. Preserve it as a virtual board cell so the reviewer
    # can still see every legal choice.
    for candidate in candidate_spaces:
        compact = _board_surface_space(candidate, legal_ids, chosen_ids)
        if compact is None or compact["id"] in known_ids:
            continue
        candidate_type = str(candidate.get("spaceType", "") or "").lower() if isinstance(candidate, dict) else ""
        is_moon = compact["id"] in moon_ids or compact["id"].lower().startswith("m") or "moon" in candidate_type or "lunar" in candidate_type
        (moon_spaces if is_moon else mars_spaces).append(compact)
        known_ids.add(compact["id"])
    return {"spaces": mars_spaces, "moon_spaces": moon_spaces}


def _award_summary(award: Any, index: int) -> Dict[str, Any]:
    if not isinstance(award, dict):
        return {"index": int(index), "label": _message_text(award)}
    scores = []
    for row in award.get("scores", []) or []:
        if not isinstance(row, dict):
            continue
        scores.append(
            {
                "player_name": str(row.get("playerName", row.get("name", "")) or "").strip(),
                "player_color": str(row.get("playerColor", row.get("color", "")) or "").strip(),
                "score": _safe_float(row.get("playerScore", row.get("score", 0.0))),
            }
        )
    return {
        "index": int(index),
        "name": str(award.get("name", "") or award.get("title", "") or "").strip(),
        "funded_by": str(
            award.get("playerName", "")
            or award.get("playerColor", "")
            or award.get("color", "")
            or ""
        ).strip(),
        "scores": scores,
    }


def _milestone_summary(milestone: Any, index: int) -> Dict[str, Any]:
    if not isinstance(milestone, dict):
        return {"index": int(index), "label": _message_text(milestone)}
    scores = []
    for row in milestone.get("scores", []) or []:
        if not isinstance(row, dict):
            continue
        scores.append(
            {
                "player_name": str(row.get("playerName", row.get("name", "")) or "").strip(),
                "player_color": str(row.get("playerColor", row.get("color", "")) or "").strip(),
                "score": _safe_float(row.get("playerScore", row.get("score", 0.0))),
            }
        )
    return {
        "index": int(index),
        "name": str(milestone.get("name", "") or "").strip(),
        "claimed_by": str(milestone.get("playerName", "") or milestone.get("playerColor", "") or "").strip(),
        "minimum": _safe_float(milestone.get("minimum", 0.0)),
        "scores": scores,
    }


def _top_aux_predictions(aux_predictions: List[float]) -> List[Dict[str, Any]]:
    if StateEncoder is not None:
        milestone_count = len(getattr(StateEncoder, "_ALL_MILESTONES", []))
        if len(aux_predictions) >= milestone_count and milestone_count > 0:
            entries = []
            for idx, value in enumerate(list(aux_predictions[:milestone_count])):
                entries.append({"index": int(idx), "value": _safe_float(value)})
            entries.sort(key=lambda item: item["value"], reverse=True)
            return entries[:10]
    entries: List[Dict[str, Any]] = []
    for idx, value in enumerate(list(aux_predictions[:70])):
        entries.append({"index": int(idx), "value": _safe_float(value)})
    entries.sort(key=lambda item: item["value"], reverse=True)
    return entries[:10]


def _legacy_aux_prediction_summary(aux_predictions: List[float]) -> Dict[str, Any]:
    return {
        "award_ev": _safe_float(aux_predictions[70] if len(aux_predictions) > 70 else 0.0),
        "playable_cards": _safe_float(aux_predictions[71] if len(aux_predictions) > 71 else 0.0),
        "steel_target": _safe_float(aux_predictions[72] if len(aux_predictions) > 72 else 0.0),
        "titanium_target": _safe_float(aux_predictions[73] if len(aux_predictions) > 73 else 0.0),
        "top_milestone_predictions": _top_aux_predictions(aux_predictions),
    }


def _named_prediction_rows(values: List[float], names: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "index": int(idx),
                "name": str(names[idx] if idx < len(names) else f"item-{idx}"),
                "value": _safe_float(value),
            }
        )
    rows.sort(key=lambda item: item["value"], reverse=True)
    return rows[:limit]


def _planner_aux_context() -> Dict[str, Any]:
    milestone_names = list(getattr(StateEncoder, "_ALL_MILESTONES", [])) if StateEncoder is not None else []
    award_names = list(getattr(StateEncoder, "_ALL_AWARDS", [])) if StateEncoder is not None else []
    if planner_aux_layout is None or not milestone_names:
        return {}
    layout = planner_aux_layout(len(milestone_names), len(award_names), PLANNER_OPPORTUNITY_LIMIT)
    return {
        "milestone_names": milestone_names,
        "award_names": award_names,
        "layout": layout,
    }


def _aux_prediction_summary(aux_predictions: List[float]) -> Dict[str, Any]:
    context = _planner_aux_context()
    layout = dict(context.get("layout", {}) or {})
    if not layout:
        return _legacy_aux_prediction_summary(aux_predictions)
    planner_dim = max(int(value.stop or 0) for value in layout.values())
    if len(aux_predictions) < planner_dim:
        return _legacy_aux_prediction_summary(aux_predictions)

    milestone_names = list(context.get("milestone_names", []) or [])
    award_names = list(context.get("award_names", []) or [])
    milestone_claim = aux_predictions[layout["milestone_claim_now"]]
    award_ev = aux_predictions[layout["award_fund_now_ev"]]
    award_rank = aux_predictions[layout["award_rank_class"]]
    opportunity_values = aux_predictions[layout["board_opportunity_value"]]
    deny_risk_values = aux_predictions[layout["deny_risk"]]

    opportunity_entries = []
    for idx, value in enumerate(opportunity_values):
        opportunity_entries.append(
            {
                "index": int(idx),
                "value": _safe_float(value),
                "deny_risk": _safe_float(deny_risk_values[idx] if idx < len(deny_risk_values) else 0.0),
            }
        )
    opportunity_entries.sort(key=lambda item: (item["value"], item["deny_risk"]), reverse=True)

    return {
        "milestone_claim_now": _named_prediction_rows(list(milestone_claim), milestone_names),
        "award_fund_now_ev": _named_prediction_rows(list(award_ev), award_names),
        "award_rank_class": _named_prediction_rows(list(award_rank), award_names),
        "carry_save_plants_value": _safe_float(aux_predictions[layout["carry_save_plants_value"].start]),
        "carry_save_heat_value": _safe_float(aux_predictions[layout["carry_save_heat_value"].start]),
        "next_turn_combo_value": _safe_float(aux_predictions[layout["next_turn_combo_value"].start]),
        "next_generation_combo_value": _safe_float(aux_predictions[layout["next_generation_combo_value"].start]),
        "top_milestone_predictions": _named_prediction_rows(list(milestone_claim), milestone_names),
        "top_award_ev_predictions": _named_prediction_rows(list(award_ev), award_names),
        "top_board_opportunity_predictions": opportunity_entries[:10],
    }


def _descriptor_lookup(action_descriptors: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for descriptor in action_descriptors:
        if not isinstance(descriptor, dict):
            continue
        idx = _safe_int(descriptor.get("action_index", -1), -1)
        if idx < 0:
            continue
        out[idx] = {
            "action_index": idx,
            "action_position": _safe_int(descriptor.get("action_position", -1), -1),
            "family": str(descriptor.get("family", "") or "").strip(),
            "label": str(descriptor.get("label", "") or "").strip(),
            "card_name": str(descriptor.get("card_name", "") or "").strip(),
            "project_name": str(descriptor.get("project_name", "") or "").strip(),
            "award_name": str(descriptor.get("award_name", "") or "").strip(),
            "milestone_name": str(descriptor.get("milestone_name", "") or "").strip(),
            "decoded_action": descriptor.get("decoded_action", {}),
            "space_features": dict(descriptor.get("space_features", {}) or {}),
        }
    return out


def _merge_policy_rows_with_descriptors(rows: List[Dict[str, Any]], action_descriptors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    descriptor_by_index = _descriptor_lookup(action_descriptors)
    merged: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row or {})
        descriptor = descriptor_by_index.get(_safe_int(entry.get("action_index", -1), -1), {})
        if descriptor:
            for key, value in descriptor.items():
                if key == "action_index":
                    continue
                if entry.get(key) in (None, "", {}, []):
                    entry[key] = value
        merged.append(entry)
    return merged


def _fallback_chosen_descriptor(
    chosen_action_index: int,
    action_input: Optional[Dict[str, Any]],
    action_meta: Dict[str, Any],
) -> Dict[str, Any]:
    label = str(action_meta.get("chosen_action_label", "") or "").strip()
    family = ""
    if "(" in label:
        family = label.split("(", 1)[0].strip().lower()
    if not family and isinstance(action_input, dict):
        family = str(action_input.get("type", "") or "").strip().lower()
    return {
        "action_index": int(chosen_action_index),
        "action_position": _safe_int(action_meta.get("chosen_action_position", -1), -1),
        "family": family,
        "label": label,
        "decoded_action": action_input or {},
    }


def _summarize_prompt_candidates(waiting_for: Dict[str, Any], game: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options = waiting_for.get("options", []) or []
    cards = waiting_for.get("cards", []) or []
    if not cards:
        cards = _or_project_card_candidates(waiting_for)
    spaces = _prompt_space_candidates(waiting_for)
    players = waiting_for.get("players", []) or []
    colonies = waiting_for.get("colonies", []) or []
    parties = waiting_for.get("parties", []) or []
    events = waiting_for.get("globalEventNames", waiting_for.get("events", [])) or []
    tokens = waiting_for.get("tokens", []) or []
    policies = waiting_for.get("policies", []) or []
    include = waiting_for.get("include", []) or []

    board_spaces = list((game or {}).get("spaces", []) or [])
    moon = (game or {}).get("moon", {}) or {}
    board_spaces.extend(list(moon.get("spaces", []) or []))
    board_by_id = {
        _candidate_space_id(space): space
        for space in board_spaces
        if isinstance(space, dict) and _candidate_space_id(space)
    }
    map_candidates = []
    for idx, space in enumerate(spaces):
        if isinstance(space, dict):
            map_candidates.append(_space_summary(space, idx))
            continue
        space_id = _candidate_space_id(space)
        board_space = board_by_id.get(space_id)
        if isinstance(board_space, dict):
            map_candidates.append(_space_summary(board_space, idx))
        else:
            map_candidates.append({"index": int(idx), "id": space_id, "label": space_id})
    moon_candidates = [
        dict(item)
        for item in map_candidates
        if "moon" in str(item.get("name", "")).lower()
        or "lunar" in str(item.get("name", "")).lower()
        or "moon" in str(item.get("space_type", "")).lower()
        or "lunar" in str(item.get("space_type", "")).lower()
    ]

    payment_context = {
        "amount": _safe_float(waiting_for.get("amount", 0.0)),
        "min": _safe_float(waiting_for.get("min", 0.0)),
        "max": _safe_float(waiting_for.get("max", 0.0)),
        "title": _message_text(waiting_for.get("title", "")),
    }

    return {
        "cards": [_card_summary(card) for card in cards],
        "options": [_option_summary(option, idx) for idx, option in enumerate(options)],
        "map_candidates": map_candidates,
        "moon_candidates": moon_candidates,
        "players": [
            {
                "index": int(idx),
                "name": str(player.get("name", "") or player.get("color", "") or f"player-{idx}").strip()
                if isinstance(player, dict)
                else _message_text(player),
            }
            for idx, player in enumerate(players)
        ],
        "colonies": [{"index": int(idx), "label": _message_text(colony)} for idx, colony in enumerate(colonies)],
        "parties": [{"index": int(idx), "label": _message_text(party)} for idx, party in enumerate(parties)],
        "global_events": [{"index": int(idx), "label": _message_text(event)} for idx, event in enumerate(events)],
        "tokens": [{"index": int(idx), "label": _message_text(token)} for idx, token in enumerate(tokens)],
        "policies": [{"index": int(idx), "label": _message_text(policy)} for idx, policy in enumerate(policies)],
        "resource_types": [{"index": int(idx), "label": _message_text(item)} for idx, item in enumerate(include)],
        "payment_context": payment_context,
    }


def _or_project_card_candidates(waiting_for: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(waiting_for, dict):
        return []
    if str(waiting_for.get("type", "") or "") != "or":
        return []

    candidates: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for option in waiting_for.get("options", []) or []:
        if not isinstance(option, dict):
            continue
        option_type = str(option.get("type", "") or "")
        if option_type not in ["projectCard", "selectProjectCardToPlay"]:
            continue
        for card in option.get("cards", []) or []:
            if not isinstance(card, dict):
                continue
            name = str(card.get("name", "") or "").strip()
            dedupe_key = name or repr(sorted(card.items()))
            if dedupe_key in seen_names:
                continue
            seen_names.add(dedupe_key)
            candidates.append(card)
    return candidates


def _snapshot_hand_cards(player_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [_card_summary(card) for card in _resolved_player_hand_cards(player_state)]


def _attach_requirement_rankings(cards: List[Dict[str, Any]], rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards or not rankings:
        return cards
    ranking_by_name = {
        str(item.get("name", "") or "").strip(): item
        for item in rankings
        if isinstance(item, dict) and str(item.get("name", "") or "").strip()
    }
    out: List[Dict[str, Any]] = []
    for card in cards:
        merged = dict(card)
        name = str(card.get("name", "") or "").strip()
        ranked = ranking_by_name.get(name)
        if ranked:
            for key in (
                "requirements",
                "requirement_plan",
                "plan_summary",
                "reachability_score",
                "readiness_score",
                "all_satisfied",
                "blocking_count",
                "server_override",
                "masked_by_server",
                "selection_score",
            ):
                if key in ranked:
                    merged[key] = ranked.get(key)
        out.append(merged)
    return out


def build_decision_snapshot(
    request: Dict[str, Any],
    agent_id: str,
    game_id: Optional[str],
    game_url: Optional[str],
    player_id: Optional[str],
    player_state: Dict[str, Any],
    action_input: Optional[Dict[str, Any]],
    action_index: Optional[int],
    action_meta: Dict[str, Any],
    sampled_from_policy: bool,
    send_outcome: str,
    turn_action_count: int,
    state_vector: Optional[Any] = None,
) -> Dict[str, Any]:
    game = player_state.get("game", {}) or {}
    waiting_for = player_state.get("waitingFor", {}) or {}
    this_player = player_state.get("thisPlayer", {}) or {}
    other_players = [
        _player_summary(player)
        for player in (player_state.get("players", []) or [])
        if isinstance(player, dict) and not _is_same_player(player, this_player, player_id)
    ]
    prompt_candidates = _summarize_prompt_candidates(waiting_for, game)
    prompt_card_rankings = list(action_meta.get("prompt_card_rankings", []) or [])
    if prompt_card_rankings:
        prompt_candidates["cards"] = _attach_requirement_rankings(list(prompt_candidates.get("cards", []) or []), prompt_card_rankings)
    aux_predictions = list(action_meta.get("aux_predictions", []) or [])
    aux_targets = dict(action_meta.get("aux_targets", {}) or {})
    action_descriptors = list(action_meta.get("action_descriptors", []) or [])
    chosen_action_index = _safe_int(action_index, -1)
    merged_policy_ranking = _merge_policy_rows_with_descriptors(list(action_meta.get("policy_ranking", []) or []), action_descriptors)
    merged_top_actions = _merge_policy_rows_with_descriptors(list(action_meta.get("policy_top_actions", []) or []), action_descriptors)
    chosen_descriptor = _descriptor_lookup(action_descriptors).get(chosen_action_index, {})
    if not chosen_descriptor:
        chosen_descriptor = _fallback_chosen_descriptor(chosen_action_index, action_input, action_meta)
    resolved_hand_count = _resolved_player_hand_count(player_state, this_player)
    hand_cards = _snapshot_hand_cards(player_state)
    if prompt_card_rankings:
        hand_cards = _attach_requirement_rankings(hand_cards, prompt_card_rankings)
    board_surface = _snapshot_board_surface(game, waiting_for, action_input)

    snapshot = {
        "schema_version": "decision_snapshot.v1",
        "captured_at": _utc_now_iso(),
        "request": {
            "request_id": str(request.get("request_id", "") or "").strip(),
            "note": str(request.get("note", "") or "").strip(),
            "include_state_vector": bool(request.get("include_state_vector", False)),
        },
        "agent": {
            "id": str(agent_id or "").strip(),
        },
        "prompt": {
            "game_id": str(game_id or "").strip(),
            "seed": action_meta.get("game_seed"),
            "game_url": str(game_url or "").strip(),
            "player_id": str(player_id or "").strip(),
            "player_name": str(this_player.get("name", "") or "").strip(),
            "phase": str(game.get("phase", "") or "").strip(),
            "phase_index": _safe_int(action_meta.get("phase_index", 0)),
            "generation": _safe_int(game.get("generation", 0)),
            "prompt_type": str(waiting_for.get("type", "") or "").strip(),
            "prompt_title": _message_text(waiting_for.get("title", "")),
            "turn_action_count": int(turn_action_count),
            "decision_sequence": _safe_int(action_meta.get("decision_sequence", 0)),
            "sampled_from_policy": bool(sampled_from_policy),
            "send_outcome": str(send_outcome or "").strip(),
        },
        "raw": {
            # Guided captures are training/debug artifacts and deliberately
            # retain the exact private player view needed to reproduce the
            # legal-action catalog.  _json_safe handles numpy values below.
            "player_state": player_state,
            "waiting_for": waiting_for,
        },
        "state": {
            "this_player": _player_summary(this_player, hand_count_override=resolved_hand_count),
            "hand": hand_cards,
            "tableau": [_card_summary(card) for card in (this_player.get("tableau", []) or [])],
            "opponents": other_players,
            "board": {
                "temperature": _safe_float(game.get("temperature", 0.0)),
                "oxygen": _safe_float(game.get("oxygenLevel", game.get("oxygen", 0.0))),
                "oceans": _safe_float(game.get("oceans", 0.0)),
                "venus": _safe_float(game.get("venusScaleLevel", 0.0)),
                "map_name": str(game.get("boardName", "") or game.get("mapName", "") or "").strip(),
                "spaces": board_surface["spaces"],
                "moon_spaces": board_surface["moon_spaces"],
            },
            "awards": [_award_summary(award, idx) for idx, award in enumerate(game.get("awards", []) or [])],
            "milestones": [_milestone_summary(milestone, idx) for idx, milestone in enumerate(game.get("milestones", []) or [])],
            "prompt_candidates": prompt_candidates,
            "prompt_card_rankings": prompt_card_rankings,
            "planner": dict(action_meta.get("bundle_summary", {}) or {}),
            "planner_bundle": action_meta.get("planner_bundle", {}),
            "action_projection": _action_payment_projection(this_player, action_input),
        },
        "policy": {
            "chosen_action_index": chosen_action_index,
            "chosen_action_label": str(action_meta.get("chosen_action_label", "") or "").strip(),
            "chosen_action_payload": action_input or {},
            "chosen_action_descriptor": chosen_descriptor,
            "policy_temperature": _safe_float(action_meta.get("policy_temperature", 0.0)),
            "value_estimate": _safe_float(action_meta.get("value_old", 0.0)),
            "raw_available_actions": [int(item) for item in (action_meta.get("available_actions_raw", []) or [])],
            "filtered_available_actions": [int(item) for item in (action_meta.get("available_actions_filtered", []) or [])],
            "legal_actions": [int(item) for item in (action_meta.get("legal_actions", []) or [])],
            "action_descriptors": action_descriptors,
            "action_rankings": merged_policy_ranking,
            "top_actions": merged_top_actions,
        },
        "aux": {
            "targets": aux_targets,
            "predictions": _aux_prediction_summary(aux_predictions),
        },
        "diagnostics": {
            "transformer": dict(action_meta.get("transformer_stats", {}) or {}),
            "review_priority": dict(action_meta.get("review_priority", {}) or {}),
            "external_policy": dict(action_meta.get("external_policy", {}) or {}),
            "rare_state": {
                "weight": _safe_float(action_meta.get("rare_state_weight", 1.0)),
                "award_funding": _safe_float(action_meta.get("rare_award_funding", 0.0)),
                "milestone_timing": _safe_float(action_meta.get("rare_milestone_timing", 0.0)),
                "draft_keep_buy": _safe_float(action_meta.get("rare_draft_keep_buy", 0.0)),
                "high_cost_payment": _safe_float(action_meta.get("rare_high_cost_payment", 0.0)),
                "payment_value_estimate": _safe_float(action_meta.get("payment_value_estimate", 0.0)),
            },
        },
        "execution": {
            "finalized": False,
            "finalized_at": "",
            "action_source": "",
            "sent_action_index": None,
            "sent_action_payload": {},
            "server_accepted": None,
            "error": "",
            "training_eligible": False,
        },
    }
    if _safe_bool(request.get("include_state_vector", False)) and state_vector is not None:
        try:
            snapshot["state_vector"] = [float(item) for item in list(state_vector)]
        except Exception:
            snapshot["state_vector"] = []
    return snapshot


def save_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(get_snapshot_root())
    prompt = dict(snapshot.get("prompt", {}) or {})
    agent = dict(snapshot.get("agent", {}) or {})
    request = dict(snapshot.get("request", {}) or {})

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = "_".join(
        [
            timestamp,
            _sanitize_id(agent.get("id", "agent"))[:12],
            _sanitize_id(prompt.get("prompt_type", "prompt"))[:24],
            _sanitize_id(prompt.get("game_id", "game"))[:12],
            _sanitize_id(request.get("request_id", "req"))[:12],
        ]
    )
    path = root / f"{snapshot_id}.json"

    payload = _json_safe(dict(snapshot))
    payload["snapshot_id"] = snapshot_id
    payload["snapshot_path"] = str(path)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return {
        "snapshot_id": snapshot_id,
        "snapshot_path": str(path),
        "snapshot": payload,
    }


def save_action_pipeline_quarantine(
    *,
    agent_id: str,
    game_id: str,
    player_id: str,
    player_state: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """Persist an invalid/overflow prompt without making it training eligible."""
    root = Path(get_snapshot_root()) / "quarantine"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_id = "_".join(
        [
            timestamp,
            _sanitize_id(agent_id)[:12],
            _sanitize_id(game_id)[:12],
            "invalid",
        ]
    )
    path = root / f"{snapshot_id}.json"
    payload = _json_safe({
        "schema_version": "action_pipeline_quarantine.v1",
        "snapshot_id": snapshot_id,
        "captured_at": _utc_now_iso(),
        "agent": {"id": str(agent_id or "")},
        "prompt": {
            "game_id": str(game_id or ""),
            "player_id": str(player_id or ""),
            "prompt_type": str(((player_state.get("waitingFor", {}) or {}).get("type", "")) or ""),
        },
        "raw": {
            "player_state": player_state,
            "waiting_for": player_state.get("waitingFor", {}) or {},
        },
        "state": {"planner_bundle": {}},
        "execution": {
            "finalized": True,
            "action_source": "invalid",
            "server_accepted": False,
            "fallback_used": False,
            "training_eligible": False,
            "validation_errors": [str(reason or "invalid_action_pipeline")],
        },
    })
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return payload


def finalize_snapshot_execution(
    snapshot_id: str,
    action_index: Optional[int],
    action_payload: Optional[Dict[str, Any]],
    action_source: str,
    server_accepted: bool,
    error: str = "",
) -> Dict[str, Any]:
    """Atomically attach the action actually sent and its server outcome."""
    safe_id = _sanitize_id(snapshot_id)
    path = Path(get_snapshot_root()) / f"{safe_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid snapshot payload: {snapshot_id}")
    accepted = bool(server_accepted)
    source = str(action_source or "").strip()
    execution = {
        "finalized": True,
        "finalized_at": _utc_now_iso(),
        "action_source": source,
        "sent_action_index": int(action_index) if action_index is not None else None,
        "sent_action_payload": dict(action_payload or {}),
        "server_accepted": accepted,
        "error": str(error or "").strip(),
        "training_eligible": bool(accepted and source in {"policy", "teacher", "human_annotation"}),
    }
    payload["execution"] = execution
    prompt = dict(payload.get("prompt", {}) or {})
    prompt["send_outcome"] = "accepted" if accepted else "rejected"
    payload["prompt"] = prompt
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return payload


def list_saved_snapshots() -> List[Dict[str, Any]]:
    root = Path(get_snapshot_root())
    items: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            saved_at_ms = int(path.stat().st_mtime_ns // 1_000_000)
        except OSError:
            saved_at_ms = 0
        prompt = dict(payload.get("prompt", {}) or {})
        agent = dict(payload.get("agent", {}) or {})
        execution = dict(payload.get("execution", {}) or {})
        review = dict((payload.get("diagnostics", {}) or {}).get("review_priority", {}) or {})
        items.append(
            {
                "snapshot_id": str(payload.get("snapshot_id", path.stem)),
                "captured_at": str(payload.get("captured_at", "") or ""),
                "agent_id": str(agent.get("id", "") or ""),
                "game_id": str(prompt.get("game_id", "") or ""),
                "player_id": str(prompt.get("player_id", "") or ""),
                "player_name": str(prompt.get("player_name", "") or ""),
                "phase": str(prompt.get("phase", "") or ""),
                "prompt_type": str(prompt.get("prompt_type", "") or ""),
                "send_outcome": str(prompt.get("send_outcome", "") or ""),
                "decision_sequence": _safe_int(prompt.get("decision_sequence", 0)),
                "server_accepted": execution.get("server_accepted"),
                "saved_at_ms": saved_at_ms,
                "review_priority": _safe_float(review.get("priority_score", 0.0)),
                "path": str(path),
            }
        )
    items.sort(key=lambda item: (float(item.get("review_priority", 0.0)), str(item.get("captured_at", ""))), reverse=True)
    return items


def load_snapshot(snapshot_id: str) -> Dict[str, Any]:
    safe_id = _sanitize_id(snapshot_id)
    path = Path(get_snapshot_root()) / f"{safe_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid snapshot payload: {snapshot_id}")
    return payload


def save_snapshot_annotation(
    snapshot_id: str,
    accepted_action_indices: List[int],
    note: str = "",
    skip: bool = False,
) -> Dict[str, Any]:
    snapshot = load_snapshot(snapshot_id)
    legal = {int(item) for item in ((snapshot.get("policy", {}) or {}).get("legal_actions", []) or [])}
    accepted = sorted({int(item) for item in (accepted_action_indices or [])})
    if not skip and not accepted:
        raise ValueError("select at least one acceptable action or mark the snapshot skipped")
    invalid = [item for item in accepted if item not in legal]
    if invalid:
        raise ValueError(f"annotation contains non-legal actions: {invalid}")
    payload = {
        "schema_version": "teacher_annotation.v1",
        "snapshot_id": str(snapshot.get("snapshot_id", snapshot_id)),
        "annotated_at": _utc_now_iso(),
        "decision_sequence": _safe_int((snapshot.get("prompt", {}) or {}).get("decision_sequence", 0)),
        "accepted_action_indices": accepted,
        "note": str(note or "").strip(),
        "skip": bool(skip),
    }
    path = Path(get_annotation_root()) / f"{_sanitize_id(snapshot_id)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return payload


def load_snapshot_annotation(snapshot_id: str) -> Dict[str, Any]:
    path = Path(get_annotation_root()) / f"{_sanitize_id(snapshot_id)}.json"
    if not path.is_file():
        raise FileNotFoundError(snapshot_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid teacher annotation")
    return payload


def load_annotation_replay(source_game_id: str, agent_id: str) -> List[Dict[str, Any]]:
    """Load saved guided labels for one agent in their original decision order.

    Snapshot IDs include a timestamp and are necessarily different when a game is
    recreated.  This joins the annotation files back to their source snapshots by
    game and agent instead, so a deterministic game can reuse the recorded
    choices.
    """
    source_game_id = str(source_game_id or "").strip()
    agent_id = str(agent_id or "").strip()
    if not source_game_id or not agent_id:
        return []

    replay: List[Dict[str, Any]] = []
    annotation_root = Path(get_annotation_root())
    snapshot_root = Path(get_snapshot_root())
    for annotation_path in annotation_root.glob("*.json"):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(annotation, dict) or bool(annotation.get("skip", False)):
                continue
            accepted = sorted({int(item) for item in (annotation.get("accepted_action_indices", []) or [])})
            if not accepted:
                continue
            snapshot_id = str(annotation.get("snapshot_id", "") or "").strip()
            snapshot_path = snapshot_root / f"{_sanitize_id(snapshot_id)}.json"
            if not snapshot_id or not snapshot_path.is_file():
                continue
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(snapshot, dict):
                continue
            prompt = snapshot.get("prompt", {}) or {}
            snapshot_agent_id = str((snapshot.get("agent", {}) or {}).get("id", "") or "").strip()
            if str(prompt.get("game_id", "") or "").strip() != source_game_id or snapshot_agent_id != agent_id:
                continue
            proposed_action_index = _safe_int((snapshot.get("policy", {}) or {}).get("chosen_action_index", -1), -1)
            source_descriptors = list((snapshot.get("policy", {}) or {}).get("action_descriptors", []) or [])
            selected_action_index = proposed_action_index if proposed_action_index in accepted else accepted[0]
            selected_payload = next(
                (
                    dict(row.get("decoded_action", {}) or {})
                    for row in source_descriptors
                    if _safe_int(row.get("action_index", -1), -1) == int(selected_action_index)
                ),
                {},
            )
            selected_label = next(
                (
                    str(row.get("label", "") or "")
                    for row in source_descriptors
                    if _safe_int(row.get("action_index", -1), -1) == int(selected_action_index)
                ),
                "",
            )
            replay.append(
                {
                    "source_snapshot_id": snapshot_id,
                    "captured_at": str(snapshot.get("captured_at", "") or ""),
                    "annotated_at": str(annotation.get("annotated_at", "") or ""),
                    "decision_sequence": _safe_int(
                        annotation.get("decision_sequence", prompt.get("decision_sequence", 0))
                    ),
                    "accepted_action_indices": accepted,
                    "selected_action_index": selected_action_index,
                    "selected_action_payload": selected_payload,
                    "selected_action_label": selected_label,
                    "legal_action_fingerprint": legal_action_descriptor_fingerprint(source_descriptors),
                    "prompt": {
                        "player_name": str(prompt.get("player_name", "") or ""),
                        "phase": str(prompt.get("phase", "") or ""),
                        "generation": _safe_int(prompt.get("generation", 0)),
                        "prompt_type": str(prompt.get("prompt_type", "") or ""),
                        "prompt_title": str(prompt.get("prompt_title", "") or ""),
                        "turn_action_count": _safe_int(prompt.get("turn_action_count", 0)),
                    },
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed or partial debug file must not make unrelated replays
            # unavailable. The caller will still fail clearly if no usable steps
            # remain for the requested source game.
            continue

    # New captures have an explicit per-game sequence.  Legacy guided captures
    # are ordered by the microsecond annotation timestamp, not their
    # second-resolution snapshot name.
    if replay and all(_safe_int(step.get("decision_sequence", 0)) > 0 for step in replay):
        replay.sort(key=lambda step: (_safe_int(step.get("decision_sequence", 0)), str(step.get("source_snapshot_id", ""))))
    else:
        replay.sort(
            key=lambda step: (
                str(step.get("annotated_at", "") or step.get("captured_at", "")),
                str(step.get("source_snapshot_id", "")),
            )
        )
    return replay


def _teacher_row_to_replay_step(row: Dict[str, Any]) -> Dict[str, Any]:
    descriptors = list(row.get("action_descriptors", []) or [])
    indices = [int(item) for item in (row.get("action_indices", []) or [])]
    position = int(row.get("chosen_action_position", -1))
    selected = indices[position] if 0 <= position < len(indices) else -1
    selected_payload = {}
    selected_label = ""
    if 0 <= position < len(descriptors):
        selected_payload = dict(descriptors[position].get("decoded_action", {}) or {})
        selected_label = str(descriptors[position].get("label", "") or "")
    return {
        "source_snapshot_id": str(row.get("sample_id", row.get("episode_id", "teacher-shard")) or "teacher-shard"),
        "selected_action_index": int(selected),
        "selected_action_payload": selected_payload,
        "selected_action_label": selected_label,
        "legal_action_fingerprint": legal_action_descriptor_fingerprint(descriptors),
        "action_descriptors": descriptors,
    }


def load_teacher_shard_episodes(teacher_source: str, game_id: str) -> List[List[Dict[str, Any]]]:
    """Load every teacher episode for one recovered game (typically one per seat)."""
    import gzip
    import pickle

    root = Path(str(teacher_source or "")).expanduser()
    game_id = str(game_id or "").strip()
    if not root.is_dir() or not game_id:
        return []
    episodes: List[List[Dict[str, Any]]] = []
    for path in sorted(root.rglob("episode_*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as handle:
                rows = list(pickle.load(handle) or [])
        except Exception:
            continue
        if not rows:
            continue
        if {str(row.get("game_id", "") or "") for row in rows} != {game_id}:
            continue
        episodes.append([_teacher_row_to_replay_step(row) for row in rows])
    return episodes


_TEACHER_SHARD_REPLAY_LOCK = threading.Lock()
_TEACHER_SHARD_REPLAY_STATE: Dict[str, Any] = {
    "game_id": "",
    "unbound": [],
    "bound": {},
    "pending": {},
}


def reset_teacher_shard_replay() -> None:
    with _TEACHER_SHARD_REPLAY_LOCK:
        _TEACHER_SHARD_REPLAY_STATE["game_id"] = ""
        _TEACHER_SHARD_REPLAY_STATE["unbound"] = []
        _TEACHER_SHARD_REPLAY_STATE["bound"] = {}
        _TEACHER_SHARD_REPLAY_STATE["pending"] = {}


def configure_teacher_shard_replay(
    teacher_source: str,
    game_id: str,
    *,
    exclude_fingerprint_sequence: Optional[List[str]] = None,
) -> int:
    """Prepare unbound seat trajectories for a multi-seat deterministic replay."""
    episodes = load_teacher_shard_episodes(teacher_source, game_id)
    exclude = [str(item) for item in (exclude_fingerprint_sequence or []) if str(item)]
    unbound: List[List[Dict[str, Any]]] = []
    for episode in episodes:
        fingerprints = [str(step.get("legal_action_fingerprint", "") or "") for step in episode]
        if exclude and len(fingerprints) == len(exclude) and fingerprints == exclude:
            continue
        unbound.append(episode)
    with _TEACHER_SHARD_REPLAY_LOCK:
        _TEACHER_SHARD_REPLAY_STATE["game_id"] = str(game_id)
        _TEACHER_SHARD_REPLAY_STATE["unbound"] = unbound
        _TEACHER_SHARD_REPLAY_STATE["bound"] = {}
        _TEACHER_SHARD_REPLAY_STATE["pending"] = {}
    return len(unbound)


def _match_teacher_step(step: Dict[str, Any], legal_descriptors: List[Dict[str, Any]]) -> Optional[int]:
    expected_fp = str(step.get("legal_action_fingerprint", "") or "")
    actual_fp = legal_action_descriptor_fingerprint(legal_descriptors)
    selected = int(step.get("selected_action_index", -1))
    legal = {int(row.get("action_index", -1)) for row in (legal_descriptors or [])}
    if expected_fp and expected_fp == actual_fp and selected in legal:
        return selected

    payload = step.get("selected_action_payload", {})
    if _is_discriminative_payload(payload):
        remapped = remap_action_index_by_payload(legal_descriptors, payload)
        if remapped is not None and int(remapped) in legal:
            return int(remapped)

    label = str(step.get("selected_action_label", "") or "").strip()
    if label:
        label_matches = [
            int(row.get("action_index", -1))
            for row in (legal_descriptors or [])
            if str(row.get("label", "") or "").strip() == label
        ]
        label_matches = [item for item in label_matches if item in legal]
        if len(label_matches) == 1:
            return int(label_matches[0])
    return None


def _is_discriminative_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    if payload.get("cards") or payload.get("spaceId") or payload.get("card"):
        return True
    response = payload.get("response")
    if isinstance(response, dict):
        if response.get("cards") or response.get("spaceId") or response.get("card"):
            return True
        if response.get("type") and response.get("type") not in {"option", "pass"}:
            return True
    if payload.get("type") in {"projectCard", "card", "space", "payment", "player"}:
        return True
    return False


def take_teacher_shard_replay_action(
    agent_id: str,
    legal_descriptors: List[Dict[str, Any]],
) -> Optional[int]:
    """Consume the next matching teacher-shard action for a non-annotated seat."""
    agent_id = str(agent_id or "").strip()
    if not agent_id:
        return None
    with _TEACHER_SHARD_REPLAY_LOCK:
        bound: Dict[str, List[Dict[str, Any]]] = _TEACHER_SHARD_REPLAY_STATE["bound"]
        unbound: List[List[Dict[str, Any]]] = _TEACHER_SHARD_REPLAY_STATE["unbound"]
        steps = bound.get(agent_id)
        if steps is None:
            match_index = None
            selected = None
            for idx, episode in enumerate(unbound):
                if not episode:
                    continue
                selected = _match_teacher_step(episode[0], legal_descriptors)
                if selected is None:
                    continue
                match_index = idx
                break
            if match_index is None or selected is None:
                return None
            steps = unbound.pop(match_index)
            bound[agent_id] = steps
        if not steps:
            raise RuntimeError(f"teacher shard replay exhausted for agent {agent_id}")
        selected = _match_teacher_step(steps[0], legal_descriptors)
        if selected is None:
            raise RuntimeError(
                "teacher shard replay diverged for agent "
                f"{agent_id} at source={steps[0].get('source_snapshot_id')}"
            )
        _TEACHER_SHARD_REPLAY_STATE["pending"][agent_id] = {
            "action_index": int(selected),
            "source_snapshot_id": steps[0].get("source_snapshot_id"),
        }
        return int(selected)


def confirm_teacher_shard_replay_action(agent_id: str, action_index: Optional[int]) -> None:
    agent_id = str(agent_id or "").strip()
    with _TEACHER_SHARD_REPLAY_LOCK:
        pending = _TEACHER_SHARD_REPLAY_STATE["pending"].pop(agent_id, None)
        steps = _TEACHER_SHARD_REPLAY_STATE["bound"].get(agent_id)
        if not isinstance(pending, dict) or not isinstance(steps, list) or not steps:
            return
        if int(pending.get("action_index", -1)) != int(action_index if action_index is not None else -1):
            return
        steps.pop(0)


def assert_teacher_shard_replay_complete() -> None:
    with _TEACHER_SHARD_REPLAY_LOCK:
        leftover_unbound = sum(len(episode) for episode in _TEACHER_SHARD_REPLAY_STATE["unbound"])
        leftover_bound = {
            agent_id: len(steps)
            for agent_id, steps in _TEACHER_SHARD_REPLAY_STATE["bound"].items()
            if steps
        }
    if leftover_unbound or leftover_bound:
        raise RuntimeError(
            "teacher shard replay left unused steps: "
            f"unbound={leftover_unbound} bound={leftover_bound}"
        )
