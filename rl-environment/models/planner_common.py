from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


def stable_identity_features(value: Any, width: int = 6) -> List[float]:
    """Return deterministic, process-stable identity features in [-1, 1].

    Python's built-in hash is intentionally randomized per process.  These
    compact digest features let a fixed-width token distinguish named board
    concepts (awards, milestones, player colours) across training workers.
    """
    count = max(0, int(width))
    if count == 0:
        return []
    normalized = str(value or "").strip().lower().encode("utf-8")
    if not normalized:
        return [0.0] * count
    digest = hashlib.sha256(normalized).digest()
    return [((float(digest[index % len(digest)]) / 127.5) - 1.0) for index in range(count)]


def _safe_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class PlannerConfig:
    token_dim: int = 64
    global_dim: int = 16
    type_vocab_size: int = 16
    opportunity_limit: int = 12
    tableau_limit: int = 24
    hand_limit: int = 24
    opponent_limit: int = 4

    @classmethod
    def from_env(cls, base: Optional["PlannerConfig"] = None) -> "PlannerConfig":
        seed = base or cls()
        return cls(
            token_dim=max(16, _safe_env_int("AGENT_PLANNER_TOKEN_DIM", int(seed.token_dim))),
            global_dim=max(16, _safe_env_int("AGENT_PLANNER_GLOBAL_DIM", int(seed.global_dim))),
            type_vocab_size=max(10, _safe_env_int("AGENT_PLANNER_TYPE_VOCAB_SIZE", int(seed.type_vocab_size))),
            opportunity_limit=max(1, _safe_env_int("AGENT_PLANNER_OPPORTUNITY_LIMIT", int(seed.opportunity_limit))),
            tableau_limit=max(0, _safe_env_int("AGENT_PLANNER_TABLEAU_LIMIT", int(seed.tableau_limit))),
            hand_limit=max(0, _safe_env_int("AGENT_PLANNER_HAND_LIMIT", int(seed.hand_limit))),
            opponent_limit=max(0, _safe_env_int("AGENT_PLANNER_OPPONENT_LIMIT", int(seed.opponent_limit))),
        )


DEFAULT_PLANNER_CONFIG = PlannerConfig()

# Backward-compatible aliases for default planner dimensions. Live code should
# prefer an explicit PlannerConfig instead of relying on these module defaults.
PLANNER_TOKEN_DIM = DEFAULT_PLANNER_CONFIG.token_dim
PLANNER_GLOBAL_DIM = DEFAULT_PLANNER_CONFIG.global_dim
PLANNER_OPPORTUNITY_LIMIT = DEFAULT_PLANNER_CONFIG.opportunity_limit


@dataclass
class PlannerStateBundle:
    world_tokens: np.ndarray
    world_token_types: np.ndarray
    world_mask: np.ndarray
    hand_tokens: np.ndarray
    hand_mask: np.ndarray
    action_tokens: np.ndarray
    action_mask: np.ndarray
    action_indices: np.ndarray
    action_positions: np.ndarray
    global_scalars: np.ndarray
    terminal: bool = False
    world_card_ids: Optional[np.ndarray] = None
    hand_card_ids: Optional[np.ndarray] = None
    action_card_mask: Optional[np.ndarray] = None
    planner_schema_version: Optional[str] = None
    card_catalog_sha256: Optional[str] = None
    unresolved_known_card_references: int = 0

    def to_serializable(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "world_tokens": np.asarray(self.world_tokens, dtype=np.float32),
            "world_token_types": np.asarray(self.world_token_types, dtype=np.int64),
            "world_mask": np.asarray(self.world_mask, dtype=np.bool_),
            "hand_tokens": np.asarray(self.hand_tokens, dtype=np.float32),
            "hand_mask": np.asarray(self.hand_mask, dtype=np.bool_),
            "action_tokens": np.asarray(self.action_tokens, dtype=np.float32),
            "action_mask": np.asarray(self.action_mask, dtype=np.bool_),
            "action_indices": np.asarray(self.action_indices, dtype=np.int64),
            "action_positions": np.asarray(self.action_positions, dtype=np.int64),
            "global_scalars": np.asarray(self.global_scalars, dtype=np.float32),
            "terminal": bool(self.terminal),
        }
        if self.planner_schema_version:
            payload.update({
                "planner_schema_version": str(self.planner_schema_version),
                "card_catalog_sha256": str(self.card_catalog_sha256 or ""),
                "world_card_ids": np.asarray(self.world_card_ids if self.world_card_ids is not None else [], dtype=np.int64),
                "hand_card_ids": np.asarray(self.hand_card_ids if self.hand_card_ids is not None else [], dtype=np.int64),
                "action_card_mask": np.asarray(
                    self.action_card_mask if self.action_card_mask is not None else np.zeros((0, 0), dtype=np.bool_),
                    dtype=np.bool_,
                ),
                "unresolved_known_card_references": int(self.unresolved_known_card_references),
            })
        return payload


def _resolve_config(planner_config: Optional[PlannerConfig] = None) -> PlannerConfig:
    return planner_config or DEFAULT_PLANNER_CONFIG


def empty_token_matrix(planner_config: Optional[PlannerConfig] = None) -> np.ndarray:
    config = _resolve_config(planner_config)
    return np.zeros((0, int(config.token_dim)), dtype=np.float32)


def empty_int_vector() -> np.ndarray:
    return np.zeros((0,), dtype=np.int64)


def empty_bool_vector() -> np.ndarray:
    return np.zeros((0,), dtype=np.bool_)


def ensure_bundle(
    raw_bundle: Any,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, np.ndarray]:
    config = _resolve_config(planner_config)
    if isinstance(raw_bundle, PlannerStateBundle):
        raw_bundle = raw_bundle.to_serializable()
    if not isinstance(raw_bundle, dict):
        raise TypeError("Planner state bundle must be a dict or PlannerStateBundle")
    action_tokens = np.asarray(
        raw_bundle.get("action_tokens", empty_token_matrix(config)),
        dtype=np.float32,
    )
    action_mask = np.asarray(raw_bundle.get("action_mask", empty_bool_vector()), dtype=np.bool_)
    action_indices = np.asarray(raw_bundle.get("action_indices", empty_int_vector()), dtype=np.int64)
    action_positions = np.asarray(raw_bundle.get("action_positions", empty_int_vector()), dtype=np.int64)
    action_count = int(action_tokens.shape[0])
    if any(int(values.shape[0]) != action_count for values in (action_mask, action_indices, action_positions)):
        raise ValueError("planner action token, mask, index, and position lengths must match")
    result = {
        "world_tokens": np.asarray(
            raw_bundle.get("world_tokens", empty_token_matrix(config)),
            dtype=np.float32,
        ),
        "world_token_types": np.asarray(raw_bundle.get("world_token_types", empty_int_vector()), dtype=np.int64),
        "world_mask": np.asarray(raw_bundle.get("world_mask", empty_bool_vector()), dtype=np.bool_),
        "hand_tokens": np.asarray(
            raw_bundle.get("hand_tokens", empty_token_matrix(config)),
            dtype=np.float32,
        ),
        "hand_mask": np.asarray(raw_bundle.get("hand_mask", empty_bool_vector()), dtype=np.bool_),
        "action_tokens": action_tokens,
        "action_mask": action_mask,
        "action_indices": action_indices,
        "action_positions": action_positions,
        "global_scalars": np.asarray(
            raw_bundle.get("global_scalars", np.zeros((int(config.global_dim),), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
        "terminal": bool(raw_bundle.get("terminal", False)),
    }
    card_fields = _card_aware_fields(raw_bundle, action_count, result["world_tokens"], result["hand_tokens"])
    result.update(card_fields)
    return result


def _card_aware_fields(
    raw_bundle: Dict[str, Any],
    action_count: int,
    world_tokens: np.ndarray,
    hand_tokens: np.ndarray,
) -> Dict[str, Any]:
    from .card_catalog import PLANNER_SCHEMA_VERSION, get_catalog
    from .v4_flags import v4_enabled

    present = any(
        key in raw_bundle
        for key in ("planner_schema_version", "world_card_ids", "hand_card_ids", "action_card_mask")
    )
    if not present and not v4_enabled():
        return {}
    schema = str(raw_bundle.get("planner_schema_version", "") or "")
    if schema != PLANNER_SCHEMA_VERSION:
        raise ValueError("V4 refuses a planner bundle that is not planner.card_aware.v1")
    expected_hash = get_catalog().sha256
    actual_hash = str(raw_bundle.get("card_catalog_sha256", "") or "")
    if actual_hash != expected_hash:
        raise ValueError("V4 planner bundle catalog hash does not match the current card catalog")
    world_ids = np.asarray(raw_bundle.get("world_card_ids", []), dtype=np.int64).reshape(-1)
    hand_ids = np.asarray(raw_bundle.get("hand_card_ids", []), dtype=np.int64).reshape(-1)
    action_card_mask = np.asarray(raw_bundle.get("action_card_mask", []), dtype=np.bool_)
    world_count = int(world_tokens.shape[0])
    hand_count = int(hand_tokens.shape[0])
    if int(world_ids.shape[0]) != world_count:
        raise ValueError("world_card_ids must align with world tokens")
    if int(hand_ids.shape[0]) != hand_count:
        raise ValueError("hand_card_ids must align with hand tokens")
    if action_card_mask.ndim == 1:
        action_card_mask = action_card_mask.reshape(action_count, hand_count)
    if action_card_mask.shape != (action_count, hand_count):
        raise ValueError("action_card_mask must have shape [action, hand]")
    unresolved = int(raw_bundle.get("unresolved_known_card_references", 0) or 0)
    if unresolved != 0:
        raise ValueError("V4 planner bundle contains unresolved known-card references")
    return {
        "planner_schema_version": schema,
        "card_catalog_sha256": actual_hash,
        "world_card_ids": world_ids,
        "hand_card_ids": hand_ids,
        "action_card_mask": action_card_mask,
        "unresolved_known_card_references": unresolved,
    }


def bundle_to_torch(
    raw_bundle: Any,
    device: torch.device,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, torch.Tensor]:
    bundle = ensure_bundle(raw_bundle, planner_config=planner_config)
    tensors = {
        "world_tokens": torch.tensor(bundle["world_tokens"], dtype=torch.float32, device=device).unsqueeze(0),
        "world_token_types": torch.tensor(bundle["world_token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "world_mask": torch.tensor(bundle["world_mask"], dtype=torch.bool, device=device).unsqueeze(0),
        "hand_tokens": torch.tensor(bundle["hand_tokens"], dtype=torch.float32, device=device).unsqueeze(0),
        "hand_mask": torch.tensor(bundle["hand_mask"], dtype=torch.bool, device=device).unsqueeze(0),
        "action_tokens": torch.tensor(bundle["action_tokens"], dtype=torch.float32, device=device).unsqueeze(0),
        "action_mask": torch.tensor(bundle["action_mask"], dtype=torch.bool, device=device).unsqueeze(0),
        "action_indices": torch.tensor(bundle["action_indices"], dtype=torch.long, device=device).unsqueeze(0),
        "action_positions": torch.tensor(bundle["action_positions"], dtype=torch.long, device=device).unsqueeze(0),
        "global_scalars": torch.tensor(bundle["global_scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "terminal_mask": torch.tensor([bool(bundle["terminal"])], dtype=torch.bool, device=device),
    }
    if "world_card_ids" in bundle:
        tensors["world_card_ids"] = torch.tensor(bundle["world_card_ids"], dtype=torch.long, device=device).unsqueeze(0)
        tensors["hand_card_ids"] = torch.tensor(bundle["hand_card_ids"], dtype=torch.long, device=device).unsqueeze(0)
        tensors["action_card_mask"] = torch.tensor(bundle["action_card_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    return tensors


def pad_bundle_batch(
    raw_bundles: Sequence[Any],
    device: torch.device,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, torch.Tensor]:
    config = _resolve_config(planner_config)
    bundles = [ensure_bundle(item, planner_config=config) for item in raw_bundles]
    if not bundles:
        raise ValueError("Cannot pad empty planner bundle batch")

    max_world = max(int(item["world_tokens"].shape[0]) for item in bundles)
    max_hand = max(int(item["hand_tokens"].shape[0]) for item in bundles)
    max_action = max(max(1, int(item["action_tokens"].shape[0])) for item in bundles)
    batch = len(bundles)
    token_dim = int(config.token_dim)
    global_dim = int(config.global_dim)

    world_tokens = torch.zeros((batch, max_world, token_dim), dtype=torch.float32, device=device)
    world_types = torch.zeros((batch, max_world), dtype=torch.long, device=device)
    world_mask = torch.zeros((batch, max_world), dtype=torch.bool, device=device)
    hand_tokens = torch.zeros((batch, max_hand, token_dim), dtype=torch.float32, device=device)
    hand_mask = torch.zeros((batch, max_hand), dtype=torch.bool, device=device)
    action_tokens = torch.zeros((batch, max_action, token_dim), dtype=torch.float32, device=device)
    action_mask = torch.zeros((batch, max_action), dtype=torch.bool, device=device)
    action_indices = torch.full((batch, max_action), -1, dtype=torch.long, device=device)
    action_positions = torch.zeros((batch, max_action), dtype=torch.long, device=device)
    global_scalars = torch.zeros((batch, global_dim), dtype=torch.float32, device=device)
    terminal_mask = torch.zeros((batch,), dtype=torch.bool, device=device)
    card_aware = any("world_card_ids" in item for item in bundles)
    if card_aware and not all("world_card_ids" in item for item in bundles):
        raise ValueError("cannot mix card-aware and legacy planner bundles in one batch")
    world_card_ids = torch.zeros((batch, max_world), dtype=torch.long, device=device) if card_aware else None
    hand_card_ids = torch.zeros((batch, max_hand), dtype=torch.long, device=device) if card_aware else None
    action_card_mask = torch.zeros((batch, max_action, max_hand), dtype=torch.bool, device=device) if card_aware else None

    for row, item in enumerate(bundles):
        world_count = int(item["world_tokens"].shape[0])
        if world_count > 0:
            world_tokens[row, :world_count] = torch.tensor(item["world_tokens"], dtype=torch.float32, device=device)
            world_types[row, :world_count] = torch.tensor(item["world_token_types"], dtype=torch.long, device=device)
            world_mask[row, :world_count] = torch.tensor(item["world_mask"], dtype=torch.bool, device=device)
            if world_card_ids is not None:
                world_card_ids[row, :world_count] = torch.tensor(item["world_card_ids"], dtype=torch.long, device=device)

        hand_count = int(item["hand_tokens"].shape[0])
        if hand_count > 0:
            hand_tokens[row, :hand_count] = torch.tensor(item["hand_tokens"], dtype=torch.float32, device=device)
            hand_mask[row, :hand_count] = torch.tensor(item["hand_mask"], dtype=torch.bool, device=device)
            if hand_card_ids is not None:
                hand_card_ids[row, :hand_count] = torch.tensor(item["hand_card_ids"], dtype=torch.long, device=device)

        action_count = int(item["action_tokens"].shape[0])
        terminal = bool(item.get("terminal", False))
        terminal_mask[row] = terminal
        if terminal and action_count > 0:
            raise ValueError("terminal planner bundle must not contain action rows")
        if action_count > 0:
            if not terminal and not bool(np.asarray(item["action_mask"], dtype=np.bool_).any()):
                raise ValueError("active planner bundle has an empty legal-action mask")
            action_tokens[row, :action_count] = torch.tensor(item["action_tokens"], dtype=torch.float32, device=device)
            action_mask[row, :action_count] = torch.tensor(item["action_mask"], dtype=torch.bool, device=device)
            action_indices[row, :action_count] = torch.tensor(item["action_indices"], dtype=torch.long, device=device)
            action_positions[row, :action_count] = torch.tensor(item["action_positions"], dtype=torch.long, device=device)
            if action_card_mask is not None and int(item["hand_tokens"].shape[0]) > 0:
                action_card_mask[row, :action_count, :int(item["hand_tokens"].shape[0])] = torch.tensor(
                    item["action_card_mask"],
                    dtype=torch.bool,
                    device=device,
                )
        else:
            if not terminal:
                raise ValueError("planner bundle has zero actions without an explicit terminal flag")
            action_positions[row, 0] = 0

        g = np.asarray(item["global_scalars"], dtype=np.float32).reshape(-1)
        take = min(int(g.size), global_dim)
        if take > 0:
            global_scalars[row, :take] = torch.tensor(g[:take], dtype=torch.float32, device=device)

    padded = {
        "world_tokens": world_tokens,
        "world_token_types": world_types,
        "world_mask": world_mask,
        "hand_tokens": hand_tokens,
        "hand_mask": hand_mask,
        "action_tokens": action_tokens,
        "action_mask": action_mask,
        "action_indices": action_indices,
        "action_positions": action_positions,
        "global_scalars": global_scalars,
        "terminal_mask": terminal_mask,
    }
    if card_aware:
        padded["world_card_ids"] = world_card_ids
        padded["hand_card_ids"] = hand_card_ids
        padded["action_card_mask"] = action_card_mask
    return padded


def token_from_features(
    type_id: int,
    features: Sequence[float],
    feature_dim: Optional[int] = None,
    type_vocab_size: Optional[int] = None,
    planner_config: Optional[PlannerConfig] = None,
) -> np.ndarray:
    config = _resolve_config(planner_config)
    feature_dim = int(feature_dim if feature_dim is not None else config.token_dim)
    type_vocab_size = int(type_vocab_size if type_vocab_size is not None else config.type_vocab_size)
    vec = np.zeros((feature_dim,), dtype=np.float32)
    vec[0] = float(type_id) / float(max(1, type_vocab_size))
    flat = [float(item) for item in list(features)]
    take = min(len(flat), feature_dim - 1)
    if take > 0:
        vec[1:1 + take] = np.asarray(flat[:take], dtype=np.float32)
    return vec


def planner_aux_layout(
    num_milestones: int,
    num_awards: int,
    opportunity_limit: Optional[int] = None,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, slice]:
    config = _resolve_config(planner_config)
    opportunity_limit = int(opportunity_limit if opportunity_limit is not None else config.opportunity_limit)
    milestone_claim_start = 0
    milestone_turns_start = milestone_claim_start + int(num_milestones)
    award_ev_start = milestone_turns_start + int(num_milestones)
    award_rank_start = award_ev_start + int(num_awards)
    scalar_start = award_rank_start + int(num_awards)
    board_opportunity_start = scalar_start + 4
    deny_risk_start = board_opportunity_start + int(opportunity_limit)
    return {
        "milestone_claim_now": slice(milestone_claim_start, milestone_turns_start),
        "milestone_turns_to_claim_bucket": slice(milestone_turns_start, award_ev_start),
        "award_fund_now_ev": slice(award_ev_start, award_rank_start),
        "award_rank_class": slice(award_rank_start, scalar_start),
        "carry_save_plants_value": slice(scalar_start, scalar_start + 1),
        "carry_save_heat_value": slice(scalar_start + 1, scalar_start + 2),
        "next_turn_combo_value": slice(scalar_start + 2, scalar_start + 3),
        "next_generation_combo_value": slice(scalar_start + 3, scalar_start + 4),
        "board_opportunity_value": slice(board_opportunity_start, deny_risk_start),
        "deny_risk": slice(deny_risk_start, deny_risk_start + int(opportunity_limit)),
    }


def planner_aux_dim(
    num_milestones: int,
    num_awards: int,
    opportunity_limit: Optional[int] = None,
    planner_config: Optional[PlannerConfig] = None,
) -> int:
    layout = planner_aux_layout(
        num_milestones=num_milestones,
        num_awards=num_awards,
        opportunity_limit=opportunity_limit,
        planner_config=planner_config,
    )
    return int(layout["deny_risk"].stop)
