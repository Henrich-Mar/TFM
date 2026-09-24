"""Canonical executable payloads for V4 legal-action catalogs."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from .action_contract import Action


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    raise TypeError(f"payload value is not JSON-serializable: {type(value)!r}")


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-normalized payload with sorted keys at every object."""
    return json.loads(json.dumps(payload, sort_keys=True, default=_json_default))


def canonical_payload_text(payload: Dict[str, Any]) -> str:
    return json.dumps(normalize_payload(payload), sort_keys=True, separators=(",", ":"), default=_json_default)


def canonicalize_legal_actions(actions: Sequence[Action]) -> Tuple[List[Action], List[Dict[str, int]]]:
    """Keep the first action for each exact payload and record dropped aliases.

    Remaining callers must still reject a catalog that contains two different
    payloads that canonicalize to the same text. This function removes exact
    duplicates before that validation runs.
    """
    kept: List[Action] = []
    aliases: List[Dict[str, int]] = []
    seen: Dict[str, Action] = {}
    for action in actions:
        key = canonical_payload_text(dict(action.payload))
        prior = seen.get(key)
        if prior is not None:
            aliases.append({"dropped_action_id": int(action.action_id), "kept_action_id": int(prior.action_id)})
            continue
        normalized = normalize_payload(dict(action.payload))
        replacement = Action(
            action_id=int(action.action_id),
            family=action.family,
            payload=normalized,
            description=str(action.description),
        )
        seen[key] = replacement
        kept.append(replacement)
    return kept, aliases


def merge_equivalent_action_mass(
    descriptors: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, int]]]:
    """Sum probability mass of equivalent payloads before any later normalization."""
    if len(descriptors) != len(probabilities):
        raise ValueError("descriptor and probability lengths must match")
    kept: List[Dict[str, Any]] = []
    merged: List[float] = []
    aliases: List[Dict[str, int]] = []
    index_of: Dict[str, int] = {}
    for position, (descriptor, probability) in enumerate(zip(descriptors, probabilities)):
        payload = descriptor.get("decoded_action")
        if not isinstance(payload, dict):
            raise ValueError("descriptor is missing decoded_action")
        key = canonical_payload_text(payload)
        probability_value = float(probability)
        if probability_value < 0.0 or not _finite(probability_value):
            raise ValueError("action probability must be finite and non-negative")
        existing = index_of.get(key)
        if existing is None:
            index_of[key] = len(kept)
            row = dict(descriptor)
            row["decoded_action"] = normalize_payload(payload)
            kept.append(row)
            merged.append(probability_value)
            continue
        merged[existing] += probability_value
        aliases.append({
            "dropped_position": int(position),
            "kept_position": int(existing),
        })
    return kept, merged, aliases


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
