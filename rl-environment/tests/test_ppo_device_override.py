import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ppo import PPOHyperParameters, PPORolloutStep, optimize_ppo_policy


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_size = 0
        self.fc = torch.nn.Linear(4, 3)
        self.value = torch.nn.Linear(4, 1)

    def forward(self, states, phase_indices=None, recurrent_state=None):
        return {
            "policy_logits": self.fc(states),
            "value": self.value(states),
            "recurrent_state": None,
            "aux_predictions": None,
            "aux_milestone_logits": None,
        }


def test_ppo_device_override_moves_network_to_cpu() -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    steps = [
        PPORolloutStep(
            state=torch.zeros(4, dtype=torch.float32).numpy(),
            action=0,
            logp_old=0.0,
            value_old=0.0,
            reward=0.0,
            done=False,
            legal_actions=[0, 1, 2],
            phase_index=0,
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
