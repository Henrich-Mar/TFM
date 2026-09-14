import sys
from pathlib import Path

import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.planner_common import PLANNER_GLOBAL_DIM, PLANNER_TOKEN_DIM, planner_aux_dim
from models.ppo import (
    PPOHyperParameters,
    PPORolloutStep,
    _is_cuda_oom,
    _select_restore_device,
    optimize_ppo_policy,
    validate_rollout_step,
)
from models.state_encoder import StateEncoder


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_size = 0
        self.fc = torch.nn.Linear(4, 3)
        self.value = torch.nn.Linear(4, 1)

    def forward(self, states, phase_indices=None, recurrent_state=None):
        summary = states["global_scalars"][:, :4]
        return {
            "policy_logits": self.fc(summary),
            "value": self.value(summary),
            "recurrent_state": None,
            "aux_predictions": torch.zeros((int(summary.shape[0]), 1), dtype=summary.dtype, device=summary.device),
            "aux_milestone_logits": None,
        }


class _WideAuxNet(_TinyNet):
    def __init__(self, aux_dim: int) -> None:
        super().__init__()
        self.aux_dim = int(aux_dim)

    def forward(self, states, phase_indices=None, recurrent_state=None):
        base = super().forward(states, phase_indices=phase_indices, recurrent_state=recurrent_state)
        summary = states["global_scalars"][:, :4]
        base["aux_predictions"] = torch.zeros(
            (int(summary.shape[0]), self.aux_dim),
            dtype=summary.dtype,
            device=summary.device,
        )
        return base


def test_cuda_driver_oom_message_is_recognized() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA driver error: out of memory"))
    assert _is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not _is_cuda_oom(RuntimeError("ordinary CPU failure"))


def _planner_bundle() -> dict:
    return {
        "world_tokens": torch.zeros((2, PLANNER_TOKEN_DIM), dtype=torch.float32).numpy(),
        "world_token_types": torch.tensor([1, 2], dtype=torch.long).numpy(),
        "world_mask": torch.tensor([True, True], dtype=torch.bool).numpy(),
        "hand_tokens": torch.zeros((1, PLANNER_TOKEN_DIM), dtype=torch.float32).numpy(),
        "hand_mask": torch.tensor([True], dtype=torch.bool).numpy(),
        "action_tokens": torch.zeros((3, PLANNER_TOKEN_DIM), dtype=torch.float32).numpy(),
        "action_mask": torch.tensor([True, True, True], dtype=torch.bool).numpy(),
        "action_indices": torch.tensor([0, 1, 2], dtype=torch.long).numpy(),
        "action_positions": torch.tensor([0, 1, 2], dtype=torch.long).numpy(),
        "global_scalars": torch.zeros((PLANNER_GLOBAL_DIM,), dtype=torch.float32).numpy(),
    }


def test_rollout_validation_rejects_untrusted_policy_transitions() -> None:
    base = dict(
        state_bundle=_planner_bundle(),
        action=0,
        action_index=0,
        logp_old=0.0,
        value_old=0.0,
        reward=0.0,
        done=False,
        legal_actions=[0, 1, 2],
    )
    validate_rollout_step(PPORolloutStep(**base))

    empty_mask = _planner_bundle()
    empty_mask["action_mask"][:] = False
    with pytest.raises(ValueError, match="empty legal-action mask"):
        validate_rollout_step(PPORolloutStep(**{**base, "state_bundle": empty_mask}))
    with pytest.raises(ValueError, match="does not match planner bundle"):
        validate_rollout_step(PPORolloutStep(**{**base, "action_index": 2}))
    with pytest.raises(ValueError, match="action_source must be policy"):
        validate_rollout_step(PPORolloutStep(**{**base, "action_source": "teacher"}))
    with pytest.raises(ValueError, match="not accepted"):
        validate_rollout_step(PPORolloutStep(**{**base, "server_accepted": False}))
    with pytest.raises(ValueError, match="fallback"):
        validate_rollout_step(PPORolloutStep(**{**base, "fallback_used": True}))
    with pytest.raises(ValueError, match="finite"):
        validate_rollout_step(PPORolloutStep(**{**base, "logp_old": float("nan")}))


def test_ppo_value_loss_ignores_invalid_value_targets() -> None:
    class _ValueNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent_size = 0
            self.fc = torch.nn.Linear(4, 3)
            # Force a large prediction so an unmasked invalid target would dominate loss.
            self.value = torch.nn.Linear(4, 1)
            with torch.no_grad():
                self.value.weight.zero_()
                self.value.bias.fill_(10.0)

        def forward(self, states, phase_indices=None, recurrent_state=None):
            summary = states["global_scalars"][:, :4]
            return {
                "policy_logits": self.fc(summary),
                "value": self.value(summary),
                "recurrent_state": None,
                "aux_predictions": None,
                "aux_milestone_logits": None,
            }

    network = _ValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    valid = PPORolloutStep(
        state_bundle=_planner_bundle(),
        action=0,
        logp_old=0.0,
        value_old=0.0,
        reward=0.0,
        done=True,
        legal_actions=[0, 1, 2],
        value_target_valid=True,
    )
    invalid = PPORolloutStep(
        state_bundle=_planner_bundle(),
        action=0,
        logp_old=0.0,
        value_old=0.0,
        reward=1000.0,
        done=True,
        legal_actions=[0, 1, 2],
        value_target_valid=False,
    )
    ppo = PPOHyperParameters(epochs=1, minibatch_size=2, value_clip_eps=0.0, entropy_coef=0.0, aux_coef=0.0)
    metrics = optimize_ppo_policy(
        network=network,
        optimizer=optimizer,
        steps=[valid, invalid],
        ppo=ppo,
        ppo_device_override=torch.device("cpu"),
    )
    # Invalid row would contribute ~0.5*(10-1000)^2 if included; masked loss stays near 0.5*10^2.
    assert float(metrics["ppo/value_loss"]) < 100.0


def test_ppo_device_override_moves_network_to_cpu() -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    steps = [
        PPORolloutStep(
            state_bundle=_planner_bundle(),
            action=0,
            logp_old=0.0,
            value_old=0.0,
            reward=0.0,
            done=False,
            legal_actions=[0, 1, 2],
            phase_index=0,
            aux_targets=torch.zeros(1, dtype=torch.float32).numpy(),
        )
        for _ in range(2)
    ]
    ppo = PPOHyperParameters(epochs=1, minibatch_size=1)

    metrics = optimize_ppo_policy(
        network=network,
        optimizer=optimizer,
        steps=steps,
        ppo=ppo,
        ppo_device_override=torch.device("cpu"),
    )

    assert next(network.parameters()).device.type == "cpu"
    assert int(metrics["rollout/steps"]) == 2


def test_select_restore_device_skips_cuda_restore_when_vram_unavailable(monkeypatch) -> None:
    network = _TinyNet()

    monkeypatch.setattr(
        "models.ppo._select_ppo_device",
        lambda network, requested_device=None: torch.device("cpu"),
    )

    restore = _select_restore_device(network, torch.device("cuda:0"))

    assert restore is None


def test_ppo_reports_grouped_planner_aux_metrics() -> None:
    aux_dim = planner_aux_dim(
        num_milestones=len(StateEncoder._ALL_MILESTONES),
        num_awards=len(StateEncoder._ALL_AWARDS),
    )
    network = _WideAuxNet(aux_dim=aux_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    steps = [
        PPORolloutStep(
            state_bundle=_planner_bundle(),
            action=0,
            logp_old=0.0,
            value_old=0.0,
            reward=0.0,
            done=False,
            legal_actions=[0, 1, 2],
            phase_index=0,
            aux_targets=torch.zeros(aux_dim, dtype=torch.float32).numpy(),
        )
        for _ in range(2)
    ]
    ppo = PPOHyperParameters(epochs=1, minibatch_size=1)

    metrics = optimize_ppo_policy(
        network=network,
        optimizer=optimizer,
        steps=steps,
        ppo=ppo,
        ppo_device_override=torch.device("cpu"),
    )

    assert "ppo/aux_mse_milestone_claim_now" in metrics
    assert "ppo/aux_mse_award_fund_now_ev" in metrics
    assert "ppo/aux_mse_board_opportunity_value" in metrics
    assert "ppo/aux_mae_deny_risk" in metrics


def test_ppo_pads_planner_bundles_per_minibatch(monkeypatch) -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    call_sizes: list[int] = []

    from models import ppo as ppo_module
    original_pad_bundle_batch = ppo_module.pad_bundle_batch

    def _recording_pad_bundle_batch(raw_bundles, device):
        call_sizes.append(len(raw_bundles))
        return original_pad_bundle_batch(raw_bundles, device)

    monkeypatch.setattr(ppo_module, "pad_bundle_batch", _recording_pad_bundle_batch)

    steps = [
        PPORolloutStep(
            state_bundle=_planner_bundle(),
            action=0,
            logp_old=0.0,
            value_old=0.0,
            reward=0.0,
            done=False,
            legal_actions=[0, 1, 2],
            phase_index=0,
            aux_targets=torch.zeros(1, dtype=torch.float32).numpy(),
        )
        for _ in range(5)
    ]
    ppo = PPOHyperParameters(epochs=1, minibatch_size=2)

    optimize_ppo_policy(
        network=network,
        optimizer=optimizer,
        steps=steps,
        ppo=ppo,
        ppo_device_override=torch.device("cpu"),
    )

    assert call_sizes
    assert max(call_sizes) <= 2
