"""
PPO utilities shared by RL agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import os
import numpy as np
import torch
import torch.nn.functional as F

import logging

_ppo_logger = logging.getLogger(__name__)

AUX_HEAD_NAMES: List[str] = [
    "milestone_claimability",
    "award_ev",
    "playable_cards",
    "steel_target",
    "titanium_target",
]


def _get_training_device() -> torch.device:
    """Return the best available device for PPO training.

    Checks ``PPO_DEVICE`` env var first (e.g. ``cuda``, ``cpu``).
    Falls back to CUDA if available, otherwise CPU.
    """
    env_device = str(os.getenv("PPO_DEVICE", "")).strip().lower()
    if env_device:
        try:
            return torch.device(env_device)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).strip().lower()


def _move_network_and_optimizer_to(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    network.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _select_ppo_device(
    network: torch.nn.Module,
    requested_device: Optional[torch.device] = None,
) -> torch.device:
    device = requested_device or _get_training_device()
    if device.type != "cuda":
        return device

    try:
        param_bytes = sum(int(param.numel()) * int(param.element_size()) for param in network.parameters())
        min_free_mb = max(128, int(os.getenv("PPO_CUDA_MIN_FREE_MB", "512")))
        required_bytes = max(int(min_free_mb) * 1024 * 1024, int(param_bytes * 4))
        free_mem, _ = torch.cuda.mem_get_info(device.index or 0)
        if int(free_mem) < int(required_bytes):
            _ppo_logger.warning(
                "Skipping CUDA PPO due to low free VRAM: free=%.1f MiB required>=%.1f MiB device=%s",
                float(free_mem) / (1024.0 * 1024.0),
                float(required_bytes) / (1024.0 * 1024.0),
                device,
            )
            return torch.device("cpu")
    except Exception:
        pass
    return device


@dataclass
class PPORolloutStep:
    state: np.ndarray
    action: int
    logp_old: float
    value_old: float
    reward: float
    done: bool
    legal_actions: List[int]
    phase_index: int = 0
    recurrent_state: Optional[np.ndarray] = None
    aux_milestone_claimability: Optional[np.ndarray] = None  # Vector of 70 milestone claimability scores
    aux_award_ev: float = 0.0
    aux_playable_cards: float = 0.0
    aux_steel_target: float = 0.0
    aux_titanium_target: float = 0.0
    aux_pred_milestone_claimability: Optional[np.ndarray] = None  # Vector of 70 milestone predictions
    aux_pred_award_ev: float = 0.0
    aux_pred_playable_cards: float = 0.0
    aux_pred_steel_target: float = 0.0
    aux_pred_titanium_target: float = 0.0
    rare_state_weight: float = 1.0
    rare_award_funding: float = 0.0
    rare_milestone_timing: float = 0.0
    rare_draft_keep_buy: float = 0.0
    rare_high_cost_payment: float = 0.0
    reward_tr_component: float = 0.0
    reward_cards_vp_component: float = 0.0
    reward_city_greenery_component: float = 0.0
    reward_milestones_awards_component: float = 0.0
    reward_other_component: float = 0.0
    reward_shaping_coef: float = 0.0
    state_schema_version: str = "v1"


@dataclass
class PPOHyperParameters:
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch_size: int = 1024
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    target_kl: float = 0.02


def _normalize_network_output(raw_output: Any) -> Dict[str, Optional[torch.Tensor]]:
    if isinstance(raw_output, dict):
        policy_logits = raw_output.get("policy_logits")
        value = raw_output.get("value")
        recurrent_state = raw_output.get("recurrent_state")
        aux_predictions = raw_output.get("aux_predictions")
        aux_milestone_logits = raw_output.get("aux_milestone_logits")
    elif isinstance(raw_output, (tuple, list)) and len(raw_output) >= 2:
        policy_logits = raw_output[0]
        value = raw_output[1]
        recurrent_state = raw_output[2] if len(raw_output) > 2 else None
        aux_predictions = raw_output[3] if len(raw_output) > 3 else None
        aux_milestone_logits = raw_output[4] if len(raw_output) > 4 else None
    else:
        raise ValueError("Unsupported network output format")

    if policy_logits is None or value is None:
        raise ValueError("Network output missing policy_logits/value tensors")
    return {
        "policy_logits": policy_logits,
        "value": value,
        "recurrent_state": recurrent_state,
        "aux_predictions": aux_predictions,
        "aux_milestone_logits": aux_milestone_logits,
    }


def _forward_network(
    network: torch.nn.Module,
    states: torch.Tensor,
    phase_indices: Optional[torch.Tensor] = None,
    recurrent_state: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    try:
        raw = network(states, phase_indices=phase_indices, recurrent_state=recurrent_state)
    except TypeError:
        raw = network(states)
    return _normalize_network_output(raw)


def _build_recurrent_state_batch(
    steps: Sequence[PPORolloutStep],
    default_dim: int,
) -> Optional[torch.Tensor]:
    recurrent_dim = max(0, int(default_dim))
    if recurrent_dim <= 0:
        for step in steps:
            raw_state = getattr(step, "recurrent_state", None)
            if raw_state is None:
                continue
            vec = np.asarray(raw_state, dtype=np.float32).reshape(-1)
            if vec.size > 0:
                recurrent_dim = int(vec.size)
                break
    if recurrent_dim <= 0:
        return None

    out = torch.zeros((len(steps), recurrent_dim), dtype=torch.float32)
    for row_idx, step in enumerate(steps):
        raw_state = getattr(step, "recurrent_state", None)
        if raw_state is None:
            continue
        vec = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if vec.size <= 0:
            continue
        use = min(recurrent_dim, int(vec.size))
        out[row_idx, :use] = torch.from_numpy(vec[:use])
    return out


def _build_aux_target_batch(steps: Sequence[PPORolloutStep]) -> torch.Tensor:
    # 70 milestones + 4 other aux targets = 74 total
    out = torch.zeros((len(steps), 74), dtype=torch.float32)
    for row_idx, step in enumerate(steps):
        # Milestone claimability: vector of 70 values
        milestone_target = getattr(step, "aux_milestone_claimability", None)
        if milestone_target is not None:
            if isinstance(milestone_target, np.ndarray):
                milestone_vec = milestone_target.flatten()[:70]  # Ensure exactly 70 values
                out[row_idx, :70] = torch.from_numpy(milestone_vec.astype(np.float32))
            elif isinstance(milestone_target, (list, tuple)):
                milestone_vec = np.asarray(milestone_target[:70], dtype=np.float32)
                out[row_idx, :min(70, len(milestone_vec))] = torch.from_numpy(milestone_vec[:70])
        # Fallback: if milestone_target is still a scalar (backward compatibility)
        elif hasattr(step, "aux_milestone_claimability") and isinstance(getattr(step, "aux_milestone_claimability"), (int, float)):
            # Old format: single float, put it in first position
            out[row_idx, 0] = float(getattr(step, "aux_milestone_claimability", 0.0))
        
        # Other aux targets (indices 70-73)
        out[row_idx, 70] = float(getattr(step, "aux_award_ev", 0.0))
        out[row_idx, 71] = float(getattr(step, "aux_playable_cards", 0.0))
        out[row_idx, 72] = float(getattr(step, "aux_steel_target", 0.0))
        out[row_idx, 73] = float(getattr(step, "aux_titanium_target", 0.0))
    return out


def _build_rare_state_payload(steps: Sequence[PPORolloutStep]) -> Dict[str, torch.Tensor]:
    weights = torch.ones((len(steps),), dtype=torch.float32)
    tags = torch.zeros((len(steps), 4), dtype=torch.float32)
    for row_idx, step in enumerate(steps):
        raw_weight = float(getattr(step, "rare_state_weight", 1.0) or 1.0)
        weights[row_idx] = max(1.0, raw_weight)
        tags[row_idx, 0] = 1.0 if float(getattr(step, "rare_award_funding", 0.0) or 0.0) > 0.0 else 0.0
        tags[row_idx, 1] = 1.0 if float(getattr(step, "rare_milestone_timing", 0.0) or 0.0) > 0.0 else 0.0
        tags[row_idx, 2] = 1.0 if float(getattr(step, "rare_draft_keep_buy", 0.0) or 0.0) > 0.0 else 0.0
        tags[row_idx, 3] = 1.0 if float(getattr(step, "rare_high_cost_payment", 0.0) or 0.0) > 0.0 else 0.0
    rare_any = (tags.sum(dim=1) > 0.0).float()
    return {
        "weights": weights,
        "rare_any": rare_any,
        "award_funding": tags[:, 0],
        "milestone_timing": tags[:, 1],
        "draft_keep_buy": tags[:, 2],
        "high_cost_payment": tags[:, 3],
    }


def _build_reward_component_batch(steps: Sequence[PPORolloutStep]) -> Dict[str, torch.Tensor]:
    out = {
        "tr": torch.zeros((len(steps),), dtype=torch.float32),
        "cards_vp": torch.zeros((len(steps),), dtype=torch.float32),
        "city_greenery": torch.zeros((len(steps),), dtype=torch.float32),
        "milestones_awards": torch.zeros((len(steps),), dtype=torch.float32),
        "other": torch.zeros((len(steps),), dtype=torch.float32),
        "shaping_coef": torch.zeros((len(steps),), dtype=torch.float32),
    }
    for row_idx, step in enumerate(steps):
        out["tr"][row_idx] = float(getattr(step, "reward_tr_component", 0.0) or 0.0)
        out["cards_vp"][row_idx] = float(getattr(step, "reward_cards_vp_component", 0.0) or 0.0)
        out["city_greenery"][row_idx] = float(getattr(step, "reward_city_greenery_component", 0.0) or 0.0)
        out["milestones_awards"][row_idx] = float(getattr(step, "reward_milestones_awards_component", 0.0) or 0.0)
        out["other"][row_idx] = float(getattr(step, "reward_other_component", 0.0) or 0.0)
        out["shaping_coef"][row_idx] = float(getattr(step, "reward_shaping_coef", 0.0) or 0.0)
    return out


def _build_legal_mask_batch(steps: Sequence[PPORolloutStep], action_dim: int) -> torch.Tensor:
    mask = torch.zeros((len(steps), action_dim), dtype=torch.bool)
    for row_idx, step in enumerate(steps):
        has_valid = False
        for action_idx in step.legal_actions:
            idx = int(action_idx)
            if 0 <= idx < action_dim:
                mask[row_idx, idx] = True
                has_valid = True
        chosen = int(step.action)
        if 0 <= chosen < action_dim:
            mask[row_idx, chosen] = True
            has_valid = True
        if not has_valid:
            mask[row_idx, :] = True
    return mask


def _compute_gae_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, torch.Tensor]:
    # Bootstrap with old values from the next observed state.
    next_values = torch.zeros_like(values)
    if values.numel() > 1:
        next_values[:-1] = values[1:]

    deltas = rewards + (gamma * next_values * (1.0 - dones)) - values
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    for idx in reversed(range(rewards.numel())):
        mask = 1.0 - dones[idx]
        gae = deltas[idx] + (gamma * gae_lambda * mask * gae)
        advantages[idx] = gae

    returns = advantages + values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return {"advantages": advantages, "returns": returns}


def optimize_ppo_policy(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: Sequence[PPORolloutStep],
    ppo: PPOHyperParameters,
    policy_temperature: float = 1.0,
    ppo_device_override: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if not steps:
        return {}

    # --- Device selection: move network + data to GPU for training, back to CPU after ---
    original_device = next(network.parameters()).device
    device = _select_ppo_device(network, requested_device=ppo_device_override)

    def _move_network_to(dev: torch.device) -> None:
        network.to(dev)
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(dev)

    def _prepare_data(dev: torch.device) -> tuple:
        """Build all training tensors on *dev*. Returns (sample action_dim, data dict)."""
        _s_state = torch.from_numpy(np.asarray(steps[0].state, dtype=np.float32)).unsqueeze(0).to(dev)
        _s_phase = torch.tensor([int(getattr(steps[0], "phase_index", 0))], dtype=torch.long).to(dev)
        with torch.no_grad():
            _s_out = _forward_network(network, _s_state, phase_indices=_s_phase, recurrent_state=None)
        _action_dim = int(_s_out["policy_logits"].shape[-1])

        _states = torch.from_numpy(
            np.stack([np.asarray(step.state, dtype=np.float32) for step in steps], axis=0)
        ).float().to(dev)
        _actions = torch.tensor([int(step.action) for step in steps], dtype=torch.long).to(dev)
        _old_log_probs = torch.tensor([float(step.logp_old) for step in steps], dtype=torch.float32).to(dev)
        _old_values = torch.tensor([float(step.value_old) for step in steps], dtype=torch.float32).to(dev)
        _rewards = torch.tensor([float(step.reward) for step in steps], dtype=torch.float32).to(dev)
        _dones = torch.tensor([1.0 if bool(step.done) else 0.0 for step in steps], dtype=torch.float32).to(dev)
        _phase_indices = torch.tensor([int(getattr(step, "phase_index", 0)) for step in steps], dtype=torch.long).to(dev)
        _recurrent_states = _build_recurrent_state_batch(steps, default_dim=int(getattr(network, "recurrent_size", 0)))
        if _recurrent_states is not None:
            _recurrent_states = _recurrent_states.to(dev)
        _aux_targets = _build_aux_target_batch(steps).to(dev)
        _legal_masks = _build_legal_mask_batch(steps, action_dim=_action_dim).to(dev)
        _rare_payload = _build_rare_state_payload(steps)
        _rare_payload = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in _rare_payload.items()}
        return _action_dim, {
            "states": _states, "actions": _actions, "old_log_probs": _old_log_probs,
            "old_values": _old_values, "rewards": _rewards, "dones": _dones,
            "phase_indices": _phase_indices, "recurrent_states": _recurrent_states,
            "aux_targets": _aux_targets, "legal_masks": _legal_masks,
            "rare_payload": _rare_payload,
        }

    if next(network.parameters()).device != device:
        _move_network_to(device)

    if device.type != "cpu":
        try:
            _ppo_logger.info("PPO training on device=%s (%d steps)", device, len(steps))
            action_dim, _d = _prepare_data(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            _ppo_logger.warning(
                "CUDA OOM during PPO data preparation (%s). Clearing cache and falling back to CPU.",
                e,
            )
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            _move_network_to(device)
            action_dim, _d = _prepare_data(device)
        except Exception as e:
            _ppo_logger.warning("Failed to move network/data to %s, falling back to CPU: %s", device, e)
            device = torch.device("cpu")
            _move_network_to(device)
            action_dim, _d = _prepare_data(device)
    else:
        action_dim, _d = _prepare_data(device)

    states = _d["states"]
    actions = _d["actions"]
    old_log_probs = _d["old_log_probs"]
    old_values = _d["old_values"]
    rewards = _d["rewards"]
    dones = _d["dones"]
    phase_indices = _d["phase_indices"]
    recurrent_states = _d["recurrent_states"]
    aux_targets = _d["aux_targets"]
    legal_masks = _d["legal_masks"]
    rare_payload = _d["rare_payload"]
    reward_components = _build_reward_component_batch(steps)

    rare_priority_enabled = str(os.getenv("PPO_RARE_STATE_PRIORITY_ENABLED", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    try:
        rare_priority_alpha = max(0.0, float(os.getenv("PPO_RARE_STATE_PRIORITY_ALPHA", "1.5")))
    except Exception:
        rare_priority_alpha = 1.5
    # rare_sample_weights must stay on CPU for torch.multinomial
    rare_sample_weights_cpu = rare_payload["weights"].cpu() if rare_payload["weights"].device.type != "cpu" else rare_payload["weights"]
    rare_sample_weights = 1.0 + ((rare_sample_weights_cpu - 1.0) * float(rare_priority_alpha))
    rare_sample_weights = torch.clamp(rare_sample_weights, min=1e-6)

    gae_payload = _compute_gae_returns(
        rewards=rewards,
        dones=dones,
        values=old_values,
        gamma=float(ppo.gamma),
        gae_lambda=float(ppo.gae_lambda),
    )
    advantages = gae_payload["advantages"]
    returns = gae_payload["returns"]

    minibatch_size = max(1, min(int(ppo.minibatch_size), len(steps)))
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    total_grad_norm = 0.0
    total_aux_loss = 0.0
    total_aux_mse = torch.zeros((74,), dtype=torch.float32)  # 70 milestones + 4 other (always CPU for metrics)
    total_aux_mae = torch.zeros((74,), dtype=torch.float32)
    aux_metric_updates = 0
    sampled_total_count = 0.0
    sampled_rare_count = 0.0
    sampled_award_count = 0.0
    sampled_milestone_count = 0.0
    sampled_draft_count = 0.0
    sampled_high_cost_count = 0.0
    num_updates = 0
    early_stopped = False

    network.train()
    for _epoch in range(max(1, int(ppo.epochs))):
        if rare_priority_enabled and len(steps) > 1:
            indices = torch.multinomial(rare_sample_weights, num_samples=len(steps), replacement=True)
        else:
            indices = torch.randperm(len(steps))
        for start in range(0, len(steps), minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_old_values = old_values[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            batch_legal_masks = legal_masks[batch_idx]
            batch_phase_indices = phase_indices[batch_idx]
            batch_recurrent_states = recurrent_states[batch_idx] if recurrent_states is not None else None
            batch_aux_targets = aux_targets[batch_idx]
            batch_rare_any = rare_payload["rare_any"][batch_idx]
            batch_rare_award = rare_payload["award_funding"][batch_idx]
            batch_rare_milestone = rare_payload["milestone_timing"][batch_idx]
            batch_rare_draft = rare_payload["draft_keep_buy"][batch_idx]
            batch_rare_high_cost = rare_payload["high_cost_payment"][batch_idx]
            sampled_total_count += float(batch_idx.numel())
            sampled_rare_count += float(batch_rare_any.sum().item())
            sampled_award_count += float(batch_rare_award.sum().item())
            sampled_milestone_count += float(batch_rare_milestone.sum().item())
            sampled_draft_count += float(batch_rare_draft.sum().item())
            sampled_high_cost_count += float(batch_rare_high_cost.sum().item())

            out = _forward_network(
                network,
                batch_states,
                phase_indices=batch_phase_indices,
                recurrent_state=batch_recurrent_states,
            )
            logits = out["policy_logits"]
            value_preds = out["value"]
            aux_predictions = out.get("aux_predictions")
            aux_milestone_logits = out.get("aux_milestone_logits")  # Raw logits for BCE loss
            logits = logits / max(float(policy_temperature), 1e-3)
            masked_logits = logits.masked_fill(~batch_legal_masks, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            ratios = torch.exp(selected_log_probs - batch_old_log_probs)
            clipped_ratios = torch.clamp(
                ratios,
                1.0 - float(ppo.clip_eps),
                1.0 + float(ppo.clip_eps),
            )
            policy_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).mean()

            value_preds = value_preds.squeeze(-1)
            if float(ppo.value_clip_eps) > 0.0:
                value_delta = value_preds - batch_old_values
                clipped_values = batch_old_values + torch.clamp(
                    value_delta,
                    -float(ppo.value_clip_eps),
                    float(ppo.value_clip_eps),
                )
                value_loss = 0.5 * torch.max(
                    (value_preds - batch_returns).pow(2),
                    (clipped_values - batch_returns).pow(2),
                ).mean()
            else:
                value_loss = 0.5 * (value_preds - batch_returns).pow(2).mean()

            if aux_predictions is not None and batch_aux_targets.shape[1] >= 74:
                # Split aux targets: milestones (0:70) and other (70:74)
                milestone_targets = batch_aux_targets[:, :70]
                other_targets = batch_aux_targets[:, 70:74]
                
                # Use BCEWithLogitsLoss for milestones (multi-label binary classification)
                if aux_milestone_logits is not None:
                    milestone_loss = F.binary_cross_entropy_with_logits(
                        aux_milestone_logits, milestone_targets, reduction='mean'
                    )
                else:
                    # Fallback: use MSE if logits not available
                    milestone_predictions = aux_predictions[:, :70]
                    milestone_loss = F.mse_loss(milestone_predictions, milestone_targets)
                
                # Use MSE for other aux targets (keeping original behavior)
                other_predictions = aux_predictions[:, 70:74]
                other_loss = F.mse_loss(other_predictions, other_targets)
                
                # Combined aux loss
                aux_loss = milestone_loss + other_loss
                
                # Update metrics
                if aux_predictions.dim() == 2:
                    # Milestone metrics (first 70)
                    milestone_diff = aux_predictions[:, :70] - milestone_targets
                    total_aux_mse[:70] += milestone_diff.pow(2).mean(dim=0).detach().cpu()
                    total_aux_mae[:70] += milestone_diff.abs().mean(dim=0).detach().cpu()
                    
                    # Other aux metrics (last 4)
                    other_diff = other_predictions - other_targets
                    total_aux_mse[70:74] += other_diff.pow(2).mean(dim=0).detach().cpu()
                    total_aux_mae[70:74] += other_diff.abs().mean(dim=0).detach().cpu()
                    aux_metric_updates += 1
            elif aux_predictions is not None:
                # Fallback for old format (backward compatibility)
                aux_loss = F.mse_loss(aux_predictions, batch_aux_targets)
                if aux_predictions.dim() == 2 and int(aux_predictions.shape[1]) >= 5:
                    diff = aux_predictions[:, :5] - batch_aux_targets[:, :5]
                    total_aux_mse[:5] += diff.pow(2).mean(dim=0).detach().cpu()
                    total_aux_mae[:5] += diff.abs().mean(dim=0).detach().cpu()
                    aux_metric_updates += 1
            else:
                aux_loss = torch.tensor(0.0, dtype=torch.float32, device=value_loss.device)

            total_loss = (
                policy_loss
                + (float(ppo.value_coef) * value_loss)
                + (float(ppo.aux_coef) * aux_loss)
                - (float(ppo.entropy_coef) * entropy)
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), float(ppo.max_grad_norm))
            optimizer.step()

            approx_kl = (batch_old_log_probs - selected_log_probs).mean().item()
            clip_fraction = ((ratios - 1.0).abs() > float(ppo.clip_eps)).float().mean().item()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_approx_kl += float(approx_kl)
            total_clip_fraction += float(clip_fraction)
            total_grad_norm += float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            total_aux_loss += float(aux_loss.item())
            num_updates += 1

            if float(ppo.target_kl) > 0.0 and approx_kl > (1.5 * float(ppo.target_kl)):
                early_stopped = True
                break
        if early_stopped:
            break

    network.eval()

    with torch.no_grad():
        latest_out = _forward_network(
            network,
            states,
            phase_indices=phase_indices,
            recurrent_state=recurrent_states,
        )
        value_preds = latest_out["value"]
        value_preds = value_preds.squeeze(-1)
        variance_returns = torch.var(returns)
        if float(variance_returns.item()) > 1e-8:
            explained_variance = 1.0 - (torch.var(returns - value_preds) / variance_returns)
            explained_variance_value = float(explained_variance.item())
        else:
            explained_variance_value = 0.0

    # Restore network to its pre-training device.
    current_device = next(network.parameters()).device
    if current_device != original_device:
        try:
            network.to(original_device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(original_device)
        except Exception as e:
            _ppo_logger.warning("Failed to restore network to %s after PPO: %s", original_device, e)
    if device.type == "cuda" or current_device.type == "cuda":
        torch.cuda.empty_cache()

    updates = max(1, num_updates)
    aux_updates = max(1, int(aux_metric_updates))
    metrics: Dict[str, Any] = {
        "ppo/policy_loss": total_policy_loss / updates,
        "ppo/value_loss": total_value_loss / updates,
        "ppo/entropy": total_entropy / updates,
        "ppo/approx_kl": total_approx_kl / updates,
        "ppo/clip_fraction": total_clip_fraction / updates,
        "ppo/aux_loss": total_aux_loss / updates,
        "ppo/explained_variance": explained_variance_value,
        "ppo/grad_norm": total_grad_norm / updates,
        "ppo/learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
        "ppo/update_steps": int(num_updates),
        "ppo/early_stop_kl_ratio": 1.0 if early_stopped else 0.0,
        "ppo/target_kl": float(ppo.target_kl),
        "ppo/rare_state_priority_enabled": 1.0 if rare_priority_enabled else 0.0,
        "ppo/rare_state_priority_alpha": float(rare_priority_alpha),
        "rollout/rare_state_base_ratio": float(rare_payload["rare_any"].mean().item()),
        "rollout/rare_state_base_ratio_award_funding": float(rare_payload["award_funding"].mean().item()),
        "rollout/rare_state_base_ratio_milestone_timing": float(rare_payload["milestone_timing"].mean().item()),
        "rollout/rare_state_base_ratio_draft_keep_buy": float(rare_payload["draft_keep_buy"].mean().item()),
        "rollout/rare_state_base_ratio_high_cost_payment": float(rare_payload["high_cost_payment"].mean().item()),
        "rollout/rare_state_sampled_ratio": (sampled_rare_count / sampled_total_count) if sampled_total_count > 0.0 else 0.0,
        "rollout/rare_state_sampled_ratio_award_funding": (sampled_award_count / sampled_total_count) if sampled_total_count > 0.0 else 0.0,
        "rollout/rare_state_sampled_ratio_milestone_timing": (sampled_milestone_count / sampled_total_count) if sampled_total_count > 0.0 else 0.0,
        "rollout/rare_state_sampled_ratio_draft_keep_buy": (sampled_draft_count / sampled_total_count) if sampled_total_count > 0.0 else 0.0,
        "rollout/rare_state_sampled_ratio_high_cost_payment": (sampled_high_cost_count / sampled_total_count) if sampled_total_count > 0.0 else 0.0,
        "rollout/rare_state_weight_mean": float(rare_payload["weights"].mean().item()),
        "rollout/reward_tr_component_mean": float(reward_components["tr"].mean().item()),
        "rollout/reward_cards_vp_component_mean": float(reward_components["cards_vp"].mean().item()),
        "rollout/reward_city_greenery_component_mean": float(reward_components["city_greenery"].mean().item()),
        "rollout/reward_milestones_awards_component_mean": float(reward_components["milestones_awards"].mean().item()),
        "rollout/reward_other_component_mean": float(reward_components["other"].mean().item()),
        "rollout/reward_shaping_coef_mean": float(reward_components["shaping_coef"].mean().item()),
        "rollout/steps": int(len(steps)),
    }
    # Metrics for milestone claimability (first 70)
    milestone_mse_mean = float(total_aux_mse[:70].mean().item()) / float(aux_updates) if aux_updates > 0 else 0.0
    milestone_mae_mean = float(total_aux_mae[:70].mean().item()) / float(aux_updates) if aux_updates > 0 else 0.0
    metrics["ppo/aux_mse_milestone_claimability"] = milestone_mse_mean
    metrics["ppo/aux_mae_milestone_claimability"] = milestone_mae_mean
    
    # Metrics for other aux heads (indices 70-73)
    for idx, head_name in enumerate(AUX_HEAD_NAMES[1:], start=70):  # Skip milestone_claimability, start at index 70
        if idx < 74:
            metrics[f"ppo/aux_mse_{head_name}"] = float(total_aux_mse[idx].item()) / float(aux_updates) if aux_updates > 0 else 0.0
            metrics[f"ppo/aux_mae_{head_name}"] = float(total_aux_mae[idx].item()) / float(aux_updates) if aux_updates > 0 else 0.0
    return metrics
