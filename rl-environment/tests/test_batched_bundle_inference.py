import asyncio
import sys
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.agent import AgentConfig, InferenceBatcher, TerraformingMarsNetwork
from models.planner_common import bundle_to_torch


def _config() -> AgentConfig:
    return AgentConfig(
        hidden_size=64,
        recurrent_size=16,
        planner_tableau_limit=8,
        planner_hand_limit=4,
        transformer_heads=4,
        transformer_layers=2,
    )


def _bundle(rng: np.random.Generator, config: AgentConfig, world: int, hand: int, actions: int) -> dict:
    dim = config.planner_token_dim
    return {
        "world_tokens": rng.normal(size=(world, dim)).astype(np.float32),
        "world_token_types": rng.integers(0, 8, size=(world,)).astype(np.int64),
        "world_mask": np.ones((world,), dtype=np.bool_),
        "hand_tokens": rng.normal(size=(hand, dim)).astype(np.float32),
        "hand_mask": np.ones((hand,), dtype=np.bool_),
        "action_tokens": rng.normal(size=(actions, dim)).astype(np.float32),
        "action_mask": np.ones((actions,), dtype=np.bool_),
        "action_indices": np.arange(actions, dtype=np.int64),
        "action_positions": np.arange(actions, dtype=np.int64),
        "global_scalars": rng.normal(size=(config.planner_global_dim,)).astype(np.float32),
    }


class _FakeAgent:
    """The slice of RLAgent that InferenceBatcher touches."""

    def __init__(self, config: AgentConfig):
        self.id = "batch-test"
        self.config = config
        self.network = TerraformingMarsNetwork(config).eval()
        self.optimizer = None
        self._model_device_lock = threading.RLock()
        self._inference_device = torch.device("cpu")

    def _ensure_network_device_consistency(self, device):
        return device

    def _forward_network(self, state_tensor, phase_indices=None, recurrent_state=None):
        return self.network(state_tensor, phase_indices=phase_indices, recurrent_state=recurrent_state)

    def _effective_policy_temperature(self) -> float:
        return 1.0


def test_batched_bundles_match_single_forwards():
    torch.manual_seed(0)
    config = _config()
    agent = _FakeAgent(config)
    rng = np.random.default_rng(0)
    shapes = [(6, 2, 3), (9, 0, 1), (4, 3, 7)]
    bundles = [_bundle(rng, config, *shape) for shape in shapes]
    recurrent = [torch.randn(config.recurrent_size) for _ in bundles]

    expected = []
    with torch.no_grad():
        for bundle, rec in zip(bundles, recurrent):
            out = agent.network(
                bundle_to_torch(bundle, torch.device("cpu"), planner_config=config.planner_config()),
                phase_indices=torch.tensor([1]),
                recurrent_state=rec.unsqueeze(0),
            )
            expected.append((torch.softmax(out["policy_logits"], dim=-1), out["value"]))

    batcher = InferenceBatcher(agent, max_batch=len(bundles), deadline_ms=50.0)
    try:
        async def run_all():
            return await asyncio.gather(*(batcher.infer(b, 1, r) for b, r in zip(bundles, recurrent)))

        results = asyncio.run(run_all())
    finally:
        batcher.shutdown()

    for (probs, value), (_, got_value, _, _, got_probs), (_, _, actions) in zip(expected, results, shapes):
        assert tuple(got_probs.shape) == (1, actions)
        assert torch.allclose(got_probs, probs, atol=1e-5)
        assert torch.allclose(got_value, value, atol=1e-5)


def test_batched_requests_keep_their_own_temperature():
    torch.manual_seed(0)
    config = _config()
    agent = _FakeAgent(config)
    rng = np.random.default_rng(1)
    bundle = _bundle(rng, config, 5, 1, 4)
    rec = torch.zeros(config.recurrent_size)
    batcher = InferenceBatcher(agent, max_batch=2, deadline_ms=50.0)
    try:
        async def run_both():
            return await asyncio.gather(batcher.infer(bundle, 0, rec, 1.0), batcher.infer(bundle, 0, rec, 0.5))

        (logits_t1, *_), (logits_t05, *_) = asyncio.run(run_both())
    finally:
        batcher.shutdown()

    assert torch.allclose(logits_t05, logits_t1 * 2.0, atol=1e-5)


def test_bound_seat_resolves_the_leaders_current_batcher():
    from models.agent import RLAgent

    leader = RLAgent(agent_id="leader-batcher-test")
    seat = RLAgent(agent_id="seat-batcher-test", config=leader.config)
    seat.bind_shared_learner(leader)
    sentinel = object()
    leader._inference_batcher = sentinel  # a reload replaces the leader's batcher
    assert seat._inference_batcher is None
    assert seat._active_inference_batcher() is sentinel
