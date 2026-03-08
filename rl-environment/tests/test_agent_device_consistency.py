import sys
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


def test_devices_match_treats_cuda_alias_and_index_zero_as_same() -> None:
    assert agent_module._devices_match(torch.device("cuda"), torch.device("cuda:0"))
    assert agent_module._devices_match(torch.device("cuda:0"), torch.device("cuda"))


def test_ensure_network_device_consistency_repairs_mixed_devices(monkeypatch) -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    moves: list[torch.device] = []

    fake_agent = SimpleNamespace(
        id="agent-test",
        network=network,
        optimizer=optimizer,
        _inference_device=torch.device("cuda:0"),
        _get_network_devices=lambda: [torch.device("cuda:0"), torch.device("cpu")],
    )

    def _fake_move(net, opt, device):
        moves.append(device)

    monkeypatch.setattr(agent_module, "_move_network_and_optimizer_to", _fake_move)

    repaired = agent_module.RLAgent._ensure_network_device_consistency(
        fake_agent,
        torch.device("cuda:0"),
    )

    assert repaired == torch.device("cuda:0")
    assert fake_agent._inference_device == torch.device("cuda:0")
    assert moves == [torch.device("cuda:0")]


def test_ensure_network_device_consistency_ignores_cuda_alias_drift(monkeypatch) -> None:
    network = _TinyNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    moves: list[torch.device] = []

    fake_agent = SimpleNamespace(
        id="agent-test",
        network=network,
        optimizer=optimizer,
        _inference_device=torch.device("cuda"),
        _get_network_devices=lambda: [torch.device("cuda:0")],
    )

    def _fake_move(net, opt, device):
        moves.append(device)

    monkeypatch.setattr(agent_module, "_move_network_and_optimizer_to", _fake_move)

    repaired = agent_module.RLAgent._ensure_network_device_consistency(
        fake_agent,
        torch.device("cuda"),
    )

    assert repaired == torch.device("cuda")
    assert moves == []
