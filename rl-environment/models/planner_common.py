from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


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

    def to_serializable(self) -> Dict[str, Any]:
        return {
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
        }


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
    return {
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
        "action_tokens": np.asarray(
            raw_bundle.get("action_tokens", empty_token_matrix(config)),
            dtype=np.float32,
        ),
        "action_mask": np.asarray(raw_bundle.get("action_mask", empty_bool_vector()), dtype=np.bool_),
        "action_indices": np.asarray(raw_bundle.get("action_indices", empty_int_vector()), dtype=np.int64),
        "action_positions": np.asarray(raw_bundle.get("action_positions", empty_int_vector()), dtype=np.int64),
        "global_scalars": np.asarray(
            raw_bundle.get("global_scalars", np.zeros((int(config.global_dim),), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
    }


def bundle_to_torch(
    raw_bundle: Any,
    device: torch.device,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, torch.Tensor]:
    bundle = ensure_bundle(raw_bundle, planner_config=planner_config)
    return {
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
    }


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

    for row, item in enumerate(bundles):
        world_count = int(item["world_tokens"].shape[0])
        if world_count > 0:
            world_tokens[row, :world_count] = torch.tensor(item["world_tokens"], dtype=torch.float32, device=device)
            world_types[row, :world_count] = torch.tensor(item["world_token_types"], dtype=torch.long, device=device)
            world_mask[row, :world_count] = torch.tensor(item["world_mask"], dtype=torch.bool, device=device)

        hand_count = int(item["hand_tokens"].shape[0])
        if hand_count > 0:
            hand_tokens[row, :hand_count] = torch.tensor(item["hand_tokens"], dtype=torch.float32, device=device)
            hand_mask[row, :hand_count] = torch.tensor(item["hand_mask"], dtype=torch.bool, device=device)

        action_count = int(item["action_tokens"].shape[0])
        if action_count > 0:
            action_tokens[row, :action_count] = torch.tensor(item["action_tokens"], dtype=torch.float32, device=device)
            action_mask[row, :action_count] = torch.tensor(item["action_mask"], dtype=torch.bool, device=device)
            action_indices[row, :action_count] = torch.tensor(item["action_indices"], dtype=torch.long, device=device)
            action_positions[row, :action_count] = torch.tensor(item["action_positions"], dtype=torch.long, device=device)
        else:
            action_mask[row, 0] = True
            action_positions[row, 0] = 0

        g = np.asarray(item["global_scalars"], dtype=np.float32).reshape(-1)
        take = min(int(g.size), global_dim)
        if take > 0:
            global_scalars[row, :take] = torch.tensor(g[:take], dtype=torch.float32, device=device)

    return {
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
    }


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
