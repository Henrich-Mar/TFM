"""Re-encode raw human decisions onto the V4 card-aware legal-action catalog."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from models.action_canonical import canonical_payload_text, merge_equivalent_action_mass
from models.action_decoder import ActionDecoder
from models.state_encoder import StateEncoder
from training.teacher_dataset import TeacherDatasetStore, split_for_episode


def select_held_out_human_games(game_ids: Sequence[str], count: int = 2) -> List[str]:
    unique = sorted({str(value).strip() for value in game_ids if str(value).strip()})
    if len(unique) < count:
        raise ValueError(f"need at least {count} human source games; found {len(unique)}")
    return sorted(
        unique,
        key=lambda value: hashlib.sha256(f"v4-human-holdout:{value}".encode("utf-8")).hexdigest(),
    )[:count]


def _split_key_for_human_game(game_id: str, holdout: bool) -> str:
    desired = "test" if holdout else "train"
    for nonce in range(10_000):
        key = f"v4-human:{'holdout' if holdout else 'train'}:{game_id}:{nonce}"
        if split_for_episode(key) == desired:
            return key
    raise RuntimeError(f"could not construct a stable {desired} split key for human game {game_id}")


def reencode_human_events(
    events: Sequence[Mapping[str, Any]],
    completed_episodes: Sequence[Mapping[str, Any]],
    *,
    encoder: Optional[StateEncoder] = None,
    decoder: Optional[ActionDecoder] = None,
    encode: Optional[Callable[..., Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Match each human choice by canonical payload and copy terminal outcomes.

    Equivalent legacy actions have their probability mass merged before the
    choice is aligned to the canonical V4 catalog. Decisions whose episode has
    no completed outcome keep an invalid value target.
    """
    outcomes = {
        str(episode.get("episode_id", "") or ""): episode
        for episode in completed_episodes
        if episode.get("completed") and episode.get("value_target") is not None
    }
    action_decoder = decoder or ActionDecoder()
    state_encoder = encoder or StateEncoder()
    samples: List[Dict[str, Any]] = []
    for event in events:
        state = event.get("player_state")
        if not isinstance(state, dict):
            raise ValueError("human event is missing player_state")
        chosen_payload = event.get("response")
        if not isinstance(chosen_payload, dict):
            raise ValueError("human event is missing its response payload")
        legacy_descriptors = list(event.get("action_descriptors") or [])
        legacy_probabilities = list(event.get("probabilities") or [])
        if legacy_descriptors and legacy_probabilities:
            legacy_descriptors, legacy_probabilities, _aliases = merge_equivalent_action_mass(
                legacy_descriptors,
                legacy_probabilities,
            )
        legal = action_decoder.enumerate_legal_actions(state)
        if legal.status != "active":
            raise ValueError(f"human event has no active legal actions: {legal.reason}")
        descriptors = [
            action_decoder._build_action_descriptor_from_action(action, position, state)
            for position, action in enumerate(legal.actions)
        ]
        chosen_key = canonical_payload_text(chosen_payload)
        matches = [
            index for index, descriptor in enumerate(descriptors)
            if canonical_payload_text(descriptor.get("decoded_action") or {}) == chosen_key
        ]
        if len(matches) != 1:
            raise ValueError(f"human response did not match exactly one canonical action: {chosen_key}")
        chosen = matches[0]
        probabilities = [0.0] * len(descriptors)
        probabilities[chosen] = 1.0
        if legacy_probabilities:
            # Preserve merged mass when the legacy distribution was richer than a one-hot.
            by_key = {
                canonical_payload_text(descriptor.get("decoded_action") or {}): probability
                for descriptor, probability in zip(legacy_descriptors, legacy_probabilities)
            }
            if any(canonical_payload_text(descriptor.get("decoded_action") or {}) in by_key for descriptor in descriptors):
                probabilities = [
                    float(by_key.get(canonical_payload_text(descriptor.get("decoded_action") or {}), 0.0))
                    for descriptor in descriptors
                ]
                total = sum(probabilities)
                if total > 0.0:
                    probabilities = [item / total for item in probabilities]
        episode_id = str(event.get("episode_id", "") or "")
        game_id = str(event.get("game_id", "") or episode_id).strip()
        outcome = outcomes.get(episode_id)
        value_valid = outcome is not None
        bundle = encode(state, descriptors) if encode is not None else state_encoder.encode(state, 0, descriptors)
        target = max(range(len(probabilities)), key=probabilities.__getitem__)
        samples.append({
            "schema_version": "teacher_sample.v5",
            "episode_id": episode_id,
            "game_id": game_id,
            "planner_bundle": bundle,
            "action_descriptors": descriptors,
            "action_indices": [int(row.get("action_index", -1)) for row in descriptors],
            "teacher_probabilities": probabilities,
            "chosen_action_position": chosen,
            "target_action_position": target,
            "target_family": str(descriptors[target].get("family", "other") or "other"),
            "selected_action_payload": dict(descriptors[chosen].get("decoded_action") or {}),
            "value_target": float(outcome.get("value_target", 0.0)) if value_valid else 0.0,
            "value_target_valid": value_valid,
            "source": "human.reencode.v5",
            "action_source": "human_annotation",
            "confidence": 1.0,
            "is_forced": len(descriptors) <= 1,
            "policy_target_valid": len(descriptors) > 1,
            "sample_weight": 1.0,
            "seed": int(event.get("seed", -1) or -1),
            "server_accepted": True,
            "fallback_used": False,
            "training_eligible": True,
            "validation_errors": [],
            "canonical_aliases": list(getattr(action_decoder, "last_canonical_aliases", []) or []),
        })
    return samples


def partition_human_samples(
    samples: Sequence[Mapping[str, Any]],
    held_out_game_count: int = 2,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    game_ids = [str(item.get("game_id", item.get("episode_id", "")) or "") for item in samples]
    held_out = set(select_held_out_human_games(game_ids, held_out_game_count))
    partitioned: List[Dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        game_id = str(item.get("game_id", item.get("episode_id", "")) or "")
        is_holdout = game_id in held_out
        item["human_source_game_id"] = game_id
        item["human_holdout"] = is_holdout
        item["human_split"] = "test" if is_holdout else "train"
        item["sample_weight"] = 1.0
        partitioned.append(item)
    provenance = {
        "strategy": "whole-game-deterministic-v1",
        "held_out_games": sorted(held_out),
        "training_games": sorted(set(game_ids) - held_out),
    }
    return partitioned, provenance


def write_reencoded_human_samples(
    dataset_dir: str,
    samples: Sequence[Mapping[str, Any]],
    held_out_game_count: int = 2,
) -> Dict[str, Any]:
    partitioned, provenance = partition_human_samples(samples, held_out_game_count)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in partitioned:
        grouped[str(item["human_source_game_id"])].append(item)
    store = TeacherDatasetStore(dataset_dir)
    written = 0
    for game_id, rows in sorted(grouped.items()):
        holdout = bool(rows[0].get("human_holdout", False))
        store.append_episode(
            f"human-v5-{hashlib.sha256(game_id.encode('utf-8')).hexdigest()[:16]}",
            rows,
            split_key=_split_key_for_human_game(game_id, holdout),
        )
        written += len(rows)
    return {**provenance, "samples": written, "dataset_dir": str(store.root)}
