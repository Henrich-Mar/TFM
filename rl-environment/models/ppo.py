"""
PPO utilities shared by RL agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPORolloutStep:
    state: np.ndarray
    action: int
    logp_old: float
    value_old: float
    reward: float
    done: bool
    legal_actions: List[int]
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
    max_grad_norm: float = 1.0
    target_kl: float = 0.02


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
    with torch.no_grad():
        sample_logits, _ = network(sample_state)
    action_dim = int(sample_logits.shape[-1])

    states = torch.from_numpy(np.stack([np.asarray(step.state, dtype=np.float32) for step in steps], axis=0)).float()
    actions = torch.tensor([int(step.action) for step in steps], dtype=torch.long)
    old_log_probs = torch.tensor([float(step.logp_old) for step in steps], dtype=torch.float32)
    old_values = torch.tensor([float(step.value_old) for step in steps], dtype=torch.float32)
    rewards = torch.tensor([float(step.reward) for step in steps], dtype=torch.float32)
    dones = torch.tensor([1.0 if bool(step.done) else 0.0 for step in steps], dtype=torch.float32)
    legal_masks = _build_legal_mask_batch(steps, action_dim=action_dim)

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
    num_updates = 0
    early_stopped = False

    network.train()
    for _epoch in range(max(1, int(ppo.epochs))):
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

            logits, value_preds = network(batch_states)
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

            total_loss = (
                policy_loss
                + (float(ppo.value_coef) * value_loss)
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
            num_updates += 1

            if float(ppo.target_kl) > 0.0 and approx_kl > (1.5 * float(ppo.target_kl)):
                early_stopped = True
                break
        if early_stopped:
            break

    network.eval()

    with torch.no_grad():
        _latest_logits, value_preds = network(states)
        value_preds = value_preds.squeeze(-1)
        variance_returns = torch.var(returns)
        if float(variance_returns.item()) > 1e-8:
            explained_variance = 1.0 - (torch.var(returns - value_preds) / variance_returns)
            explained_variance_value = float(explained_variance.item())
        else:
            explained_variance_value = 0.0

    updates = max(1, num_updates)
    return {
        "ppo/policy_loss": total_policy_loss / updates,
        "ppo/value_loss": total_value_loss / updates,
        "ppo/entropy": total_entropy / updates,
        "ppo/approx_kl": total_approx_kl / updates,
        "ppo/clip_fraction": total_clip_fraction / updates,
        "ppo/explained_variance": explained_variance_value,
        "ppo/grad_norm": total_grad_norm / updates,
        "ppo/learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
        "ppo/update_steps": int(num_updates),
        "ppo/early_stop_kl_ratio": 1.0 if early_stopped else 0.0,
        "ppo/target_kl": float(ppo.target_kl),
        "rollout/steps": int(len(steps)),
    }
