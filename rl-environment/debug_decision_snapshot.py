import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_REQUEST_LOCK = threading.Lock()
_PENDING_CAPTURE_REQUESTS: Dict[str, Dict[str, Any]] = {}

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _card_summary(card: Any) -> Dict[str, Any]:
    if not isinstance(card, dict):
        return {"label": _message_text(card)}
    tags = card.get("tags", []) or []
    summary = {
        "name": str(card.get("name", "") or "").strip(),
        "cost": _safe_float(card.get("cost", 0.0)),
        "tags": [str(tag or "").strip() for tag in tags if str(tag or "").strip()],
        "victory_points": _safe_float(card.get("victoryPoints", 0.0)),
        "type": str(card.get("type", "") or "").strip(),
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


def _player_summary(player: Any) -> Dict[str, Any]:
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
        "hand_count": _safe_int(len(player.get("cardsInHand", []) or [])),
        "tableau_count": _safe_int(len(player.get("tableau", []) or [])),
        "cities_count": _safe_int(len(player.get("cityCards", []) or [])),
        "greeneries_count": _safe_int(len(player.get("greeneryCards", []) or [])),
    }


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
    return {
        "index": int(index),
        "id": str(space.get("id", "") or space.get("spaceId", "") or "").strip(),
        "name": str(space.get("name", "") or space.get("tileType", "") or f"space-{index}").strip(),
        "space_type": str(space.get("spaceType", "") or space.get("tileType", "") or "").strip(),
        "x": _safe_int(space.get("x", 0)),
        "y": _safe_int(space.get("y", 0)),
        "bonus": [_message_text(item) for item in (space.get("bonus", []) or []) if _message_text(item)],
        "disabled": bool(space.get("isDisabled", False)),
        "owner": str(space.get("playerColor", "") or space.get("owner", "") or "").strip(),
    }


def _award_summary(award: Any, index: int) -> Dict[str, Any]:
    if not isinstance(award, dict):
        return {"index": int(index), "label": _message_text(award)}
    scores = []
    for row in award.get("scores", []) or []:
        if not isinstance(row, dict):
            continue
        scores.append(
            {
                "player_name": str(row.get("playerName", "") or "").strip(),
                "player_color": str(row.get("playerColor", "") or "").strip(),
                "score": _safe_float(row.get("playerScore", row.get("score", 0.0))),
            }
        )
    return {
        "index": int(index),
        "name": str(award.get("name", "") or award.get("title", "") or "").strip(),
        "funded_by": str(award.get("playerName", "") or award.get("playerColor", "") or "").strip(),
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
                "player_name": str(row.get("playerName", "") or "").strip(),
                "player_color": str(row.get("playerColor", "") or "").strip(),
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
    entries: List[Dict[str, Any]] = []
    for idx, value in enumerate(list(aux_predictions[:70])):
        entries.append({"index": int(idx), "value": _safe_float(value)})
    entries.sort(key=lambda item: item["value"], reverse=True)
    return entries[:10]


def _summarize_prompt_candidates(waiting_for: Dict[str, Any]) -> Dict[str, Any]:
    options = waiting_for.get("options", []) or []
    cards = waiting_for.get("cards", []) or []
    spaces = waiting_for.get("availableSpaces", waiting_for.get("spaces", [])) or []
    players = waiting_for.get("players", []) or []
    colonies = waiting_for.get("colonies", []) or []
    parties = waiting_for.get("parties", []) or []
    events = waiting_for.get("globalEventNames", waiting_for.get("events", [])) or []
    tokens = waiting_for.get("tokens", []) or []
    policies = waiting_for.get("policies", []) or []
    include = waiting_for.get("include", []) or []

    map_candidates = [_space_summary(space, idx) for idx, space in enumerate(spaces)]
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
        if isinstance(player, dict) and str(player.get("id", "") or "") != str(player_id or "")
    ]
    prompt_candidates = _summarize_prompt_candidates(waiting_for)
    prompt_card_rankings = list(action_meta.get("prompt_card_rankings", []) or [])
    if prompt_card_rankings:
        prompt_candidates["cards"] = _attach_requirement_rankings(list(prompt_candidates.get("cards", []) or []), prompt_card_rankings)
    aux_predictions = list(action_meta.get("aux_predictions", []) or [])
    aux_targets = dict(action_meta.get("aux_targets", {}) or {})
    hand_cards = [_card_summary(card) for card in (this_player.get("cardsInHand", waiting_for.get("cards", [])) or [])]
    if prompt_card_rankings:
        hand_cards = _attach_requirement_rankings(hand_cards, prompt_card_rankings)

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
            "game_url": str(game_url or "").strip(),
            "player_id": str(player_id or "").strip(),
            "player_name": str(this_player.get("name", "") or "").strip(),
            "phase": str(game.get("phase", "") or "").strip(),
            "phase_index": _safe_int(action_meta.get("phase_index", 0)),
            "generation": _safe_int(game.get("generation", 0)),
            "prompt_type": str(waiting_for.get("type", "") or "").strip(),
            "prompt_title": _message_text(waiting_for.get("title", "")),
            "turn_action_count": int(turn_action_count),
            "sampled_from_policy": bool(sampled_from_policy),
            "send_outcome": str(send_outcome or "").strip(),
        },
        "state": {
            "this_player": _player_summary(this_player),
            "hand": hand_cards,
            "tableau": [_card_summary(card) for card in (this_player.get("tableau", []) or [])],
            "opponents": other_players,
            "board": {
                "temperature": _safe_float(game.get("temperature", 0.0)),
                "oxygen": _safe_float(game.get("oxygenLevel", game.get("oxygen", 0.0))),
                "oceans": _safe_float(game.get("oceans", 0.0)),
                "venus": _safe_float(game.get("venusScaleLevel", 0.0)),
                "map_name": str(game.get("boardName", "") or game.get("mapName", "") or "").strip(),
            },
            "awards": [_award_summary(award, idx) for idx, award in enumerate(game.get("awards", []) or [])],
            "milestones": [_milestone_summary(milestone, idx) for idx, milestone in enumerate(game.get("milestones", []) or [])],
            "prompt_candidates": prompt_candidates,
            "prompt_card_rankings": prompt_card_rankings,
        },
        "policy": {
            "chosen_action_index": _safe_int(action_index, -1),
            "chosen_action_label": str(action_meta.get("chosen_action_label", "") or "").strip(),
            "chosen_action_payload": action_input or {},
            "policy_temperature": _safe_float(action_meta.get("policy_temperature", 0.0)),
            "value_estimate": _safe_float(action_meta.get("value_old", 0.0)),
            "raw_available_actions": [int(item) for item in (action_meta.get("available_actions_raw", []) or [])],
            "filtered_available_actions": [int(item) for item in (action_meta.get("available_actions_filtered", []) or [])],
            "legal_actions": [int(item) for item in (action_meta.get("legal_actions", []) or [])],
            "action_rankings": list(action_meta.get("policy_ranking", []) or []),
            "top_actions": list(action_meta.get("policy_top_actions", []) or []),
        },
        "aux": {
            "targets": aux_targets,
            "predictions": {
                "award_ev": _safe_float(aux_predictions[70] if len(aux_predictions) > 70 else 0.0),
                "playable_cards": _safe_float(aux_predictions[71] if len(aux_predictions) > 71 else 0.0),
                "steel_target": _safe_float(aux_predictions[72] if len(aux_predictions) > 72 else 0.0),
                "titanium_target": _safe_float(aux_predictions[73] if len(aux_predictions) > 73 else 0.0),
                "top_milestone_predictions": _top_aux_predictions(aux_predictions),
            },
        },
        "diagnostics": {
            "transformer": dict(action_meta.get("transformer_stats", {}) or {}),
            "rare_state": {
                "weight": _safe_float(action_meta.get("rare_state_weight", 1.0)),
                "award_funding": _safe_float(action_meta.get("rare_award_funding", 0.0)),
                "milestone_timing": _safe_float(action_meta.get("rare_milestone_timing", 0.0)),
                "draft_keep_buy": _safe_float(action_meta.get("rare_draft_keep_buy", 0.0)),
                "high_cost_payment": _safe_float(action_meta.get("rare_high_cost_payment", 0.0)),
                "payment_value_estimate": _safe_float(action_meta.get("payment_value_estimate", 0.0)),
            },
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

    payload = dict(snapshot)
    payload["snapshot_id"] = snapshot_id
    payload["snapshot_path"] = str(path)

    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "snapshot_id": snapshot_id,
        "snapshot_path": str(path),
        "snapshot": payload,
    }


def list_saved_snapshots() -> List[Dict[str, Any]]:
    root = Path(get_snapshot_root())
    items: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt = dict(payload.get("prompt", {}) or {})
        agent = dict(payload.get("agent", {}) or {})
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
                "path": str(path),
            }
        )
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
