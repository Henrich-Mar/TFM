import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.agent as agent_module


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, states, phase_indices=None, recurrent_state=None):
        logits = self.fc(states)
        value = logits[:, :1]
        return {
            "policy_logits": logits,
            "value": value,
            "recurrent_state": None,
            "aux_predictions": None,
            "aux_milestone_logits": None,
        }


def test_run_ppo_update_sync_holds_model_device_lock(monkeypatch) -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    agent = SimpleNamespace(
        id="agent-test",
        network=network,
        optimizer=optimizer,
        ppo_hparams=SimpleNamespace(entropy_coef=0.0, target_kl=0.02),
        _model_device_lock=threading.RLock(),
        _inference_device=torch.device("cpu"),
        _adapt_ppo_learning_rate=lambda approx_kl: {},
        _try_reclaim_cuda=lambda: None,
    )

    def _fake_optimize(**kwargs):
        assert agent._model_device_lock._is_owned()
        return {"ppo/approx_kl": 0.0}

    monkeypatch.setattr(agent_module, "optimize_ppo_policy", _fake_optimize)
    monkeypatch.setattr(agent_module, "_select_ppo_device", lambda network: torch.device("cpu"))

    metrics = agent_module._run_ppo_update_sync(
        agent,
        steps=[],
        current_entropy_coef=0.01,
        policy_temp=1.0,
    )

    assert metrics["ppo/learning_rate"] == optimizer.param_groups[0]["lr"]
    assert metrics["ppo/target_kl"] == 0.02


def test_cuda_oom_cpu_fallback_caps_minibatch(monkeypatch) -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    hparams = SimpleNamespace(
        entropy_coef=0.0,
        target_kl=0.02,
        minibatch_size=512,
    )
    agent = SimpleNamespace(
        id="agent-test",
        network=network,
        optimizer=optimizer,
        ppo_hparams=hparams,
        _model_device_lock=threading.RLock(),
        _inference_device=torch.device("cpu"),
        _adapt_ppo_learning_rate=lambda approx_kl: {},
        _try_reclaim_cuda=lambda: None,
    )
    calls = []

    def _fake_optimize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"ppo/approx_kl": 0.0}

    monkeypatch.setenv("PPO_CPU_FALLBACK_MINIBATCH_SIZE", "16")
    monkeypatch.setattr(agent_module, "optimize_ppo_policy", _fake_optimize)
    monkeypatch.setattr(agent_module, "_select_ppo_device", lambda network: torch.device("cuda:0"))
    monkeypatch.setattr(agent_module, "_move_network_and_optimizer_to", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    agent_module._run_ppo_update_sync(
        agent,
        steps=[],
        current_entropy_coef=0.01,
        policy_temp=1.0,
    )

    assert len(calls) == 2
    assert calls[0]["ppo_device_override"] == torch.device("cuda:0")
    assert calls[1]["ppo_device_override"] == torch.device("cpu")
    assert calls[1]["ppo"].minibatch_size == 16
    assert agent.ppo_hparams.minibatch_size == 512
