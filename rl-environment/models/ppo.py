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

AUX_HEAD_NAMES: List[str] = [
    "milestone_claimability",
    "award_ev",
    "playable_cards",
    "steel_target",
    "titanium_target",
]


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
    aux_milestone_claimability: float = 0.0
    aux_award_ev: float = 0.0
    aux_playable_cards: float = 0.0
    aux_steel_target: float = 0.0
    aux_titanium_target: float = 0.0
    aux_pred_milestone_claimability: float = 0.0
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
    elif isinstance(raw_output, (tuple, list)) and len(raw_output) >= 2:
        policy_logits = raw_output[0]
        value = raw_output[1]
        recurrent_state = raw_output[2] if len(raw_output) > 2 else None
        aux_predictions = raw_output[3] if len(raw_output) > 3 else None
    else:
        raise ValueError("Unsupported network output format")

    if policy_logits is None or value is None:
        raise ValueError("Network output missing policy_logits/value tensors")
    return {
        "policy_logits": policy_logits,
        "value": value,
        "recurrent_state": recurrent_state,
        "aux_predictions": aux_predictions,
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
    out = torch.zeros((len(steps), 5), dtype=torch.float32)
    for row_idx, step in enumerate(steps):
        out[row_idx, 0] = float(getattr(step, "aux_milestone_claimability", 0.0))
        out[row_idx, 1] = float(getattr(step, "aux_award_ev", 0.0))
        out[row_idx, 2] = float(getattr(step, "aux_playable_cards", 0.0))
        out[row_idx, 3] = float(getattr(step, "aux_steel_target", 0.0))
        out[row_idx, 4] = float(getattr(step, "aux_titanium_target", 0.0))
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
    gae = torch.tensor(0.0, dtype=torch.float32)
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
) -> Dict[str, Any]:
    if not steps:
        return {}

    sample_state = torch.from_numpy(np.asarray(steps[0].state, dtype=np.float32)).unsqueeze(0)
    sample_phase = torch.tensor([int(getattr(steps[0], "phase_index", 0))], dtype=torch.long)
    with torch.no_grad():
        sample_out = _forward_network(network, sample_state, phase_indices=sample_phase, recurrent_state=None)
        sample_logits = sample_out["policy_logits"]
    action_dim = int(sample_logits.shape[-1])

    states = torch.from_numpy(np.stack([np.asarray(step.state, dtype=np.float32) for step in steps], axis=0)).float()
    actions = torch.tensor([int(step.action) for step in steps], dtype=torch.long)
    old_log_probs = torch.tensor([float(step.logp_old) for step in steps], dtype=torch.float32)
    old_values = torch.tensor([float(step.value_old) for step in steps], dtype=torch.float32)
    rewards = torch.tensor([float(step.reward) for step in steps], dtype=torch.float32)
    dones = torch.tensor([1.0 if bool(step.done) else 0.0 for step in steps], dtype=torch.float32)
    phase_indices = torch.tensor([int(getattr(step, "phase_index", 0)) for step in steps], dtype=torch.long)
    recurrent_states = _build_recurrent_state_batch(steps, default_dim=int(getattr(network, "recurrent_size", 0)))
    aux_targets = _build_aux_target_batch(steps)
    legal_masks = _build_legal_mask_batch(steps, action_dim=action_dim)
    rare_payload = _build_rare_state_payload(steps)
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
    rare_sample_weights = 1.0 + ((rare_payload["weights"] - 1.0) * float(rare_priority_alpha))
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
    total_aux_mse = torch.zeros((5,), dtype=torch.float32)
    total_aux_mae = torch.zeros((5,), dtype=torch.float32)
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

            if aux_predictions is not None:
                aux_loss = F.mse_loss(aux_predictions, batch_aux_targets)
                if aux_predictions.dim() == 2 and int(aux_predictions.shape[1]) >= 5:
                    diff = aux_predictions[:, :5] - batch_aux_targets[:, :5]
                    total_aux_mse += diff.pow(2).mean(dim=0).detach().cpu()
                    total_aux_mae += diff.abs().mean(dim=0).detach().cpu()
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
    for idx, head_name in enumerate(AUX_HEAD_NAMES):
        metrics[f"ppo/aux_mse_{head_name}"] = float(total_aux_mse[idx].item()) / float(aux_updates)
        metrics[f"ppo/aux_mae_{head_name}"] = float(total_aux_mae[idx].item()) / float(aux_updates)
    return metrics
