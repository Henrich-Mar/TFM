"""
RL Agent - Neural network model for playing Terraforming Mars
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import os
import time
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict
from collections import deque

from game_interface import GameInstance, ServerTransportError
from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
from .planner_common import (
    PlannerConfig,
    bundle_to_torch,
    planner_aux_layout,
    planner_aux_dim,
    pad_bundle_batch,
)
from .rust_backend import require_backend_info
from scoring import calculate_terminal_reward, calculate_step_reward_decomposition
import random
import aiohttp

from .ppo import (
    PPORolloutStep,
    PPOHyperParameters,
    _is_cuda_oom,
    _move_network_and_optimizer_to,
    _select_ppo_device,
    optimize_ppo_policy,
)
from debug_decision_snapshot import (
    build_decision_snapshot,
    complete_capture_request,
    fail_capture_request,
    has_pending_capture_request,
    reserve_pending_capture_request,
    save_snapshot,
)

logger = logging.getLogger(__name__)

_AGENT_ARCHITECTURE_FIELDS: Tuple[str, ...] = (
    "hidden_size",
    "recurrent_size",
    "phase_head_count",
    "planner_token_dim",
    "planner_global_dim",
    "planner_type_vocab_size",
    "planner_opportunity_limit",
    "planner_tableau_limit",
    "planner_hand_limit",
    "planner_opponent_limit",
    "transformer_heads",
    "transformer_layers",
    "transformer_dropout",
    "planner_aux_output_dim",
)


def _safe_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _safe_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _safe_env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Inference device resolution (GPU when available, controlled by env var)
# ---------------------------------------------------------------------------
_inference_device: Optional[torch.device] = None
_inference_device_lock = threading.Lock()


def _devices_match(left: torch.device, right: torch.device) -> bool:
    left = torch.device(left)
    right = torch.device(right)
    if left.type != right.type:
        return False
    if left.type == "cuda":
        return (
            left.index == right.index
            or left.index is None
            or right.index is None
        )
    return left == right


def _resolve_inference_device() -> torch.device:
    """Return the device used for inference, determined once from
    ``AGENT_INFERENCE_DEVICE`` (``auto`` | ``cpu`` | ``cuda`` | ``cuda:N``).
    """
    global _inference_device
    with _inference_device_lock:
        if _inference_device is not None:
            return _inference_device
        env_val = os.getenv("AGENT_INFERENCE_DEVICE", "auto").strip().lower()
        if env_val == "auto":
            _inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif env_val.startswith("cuda"):
            _inference_device = torch.device(env_val)
        else:
            _inference_device = torch.device("cpu")
        logger.info("Inference device resolved to %s", _inference_device)
        return _inference_device


# Thread pool for offloading inference off the asyncio event loop.
# Lazily initialized; size controlled by AGENT_INFERENCE_THREADS (0 = auto).
_inference_executor: Optional[ThreadPoolExecutor] = None
_inference_executor_lock = threading.Lock()


def _get_inference_executor() -> ThreadPoolExecutor:
    """Return the shared inference thread pool, creating it on first use."""
    global _inference_executor
    with _inference_executor_lock:
        if _inference_executor is not None:
            return _inference_executor
        try:
            n = int(os.getenv("AGENT_INFERENCE_THREADS", "0"))
        except Exception:
            n = 0
        if n <= 0:
            device = _resolve_inference_device()
            if device.type == "cuda":
                # GPU forward passes serialize on the default CUDA stream;
                # a small pool keeps CPU prep concurrent without excessive
                # GPU context-switching.
                n = 4
            else:
                n = min(32, (os.cpu_count() or 8))
        _inference_executor = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="agent_inference")
        return _inference_executor


# Thread pool for offloading PPO optimization off the asyncio event loop.
# Keeps the event loop responsive for HTTP/game handling during training.
_ppo_executor: Optional[ThreadPoolExecutor] = None
_ppo_executor_lock = threading.Lock()


def _get_ppo_executor() -> ThreadPoolExecutor:
    """Return the shared PPO thread pool, creating it on first use."""
    global _ppo_executor
    with _ppo_executor_lock:
        if _ppo_executor is not None:
            return _ppo_executor
        try:
            n = int(os.getenv("PPO_EXECUTOR_WORKERS", "4"))
        except Exception:
            n = 4
        n = max(1, min(n, 16))
        _ppo_executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="agent_ppo")
        return _ppo_executor


def _run_ppo_update_sync(
    agent: "RLAgent",
    steps: List[PPORolloutStep],
    current_entropy_coef: float,
    policy_temp: float,
) -> Dict[str, Any]:
    """Run PPO optimization synchronously (for executor). Must not call async code."""
    with agent._model_device_lock:
        agent.ppo_hparams.entropy_coef = float(current_entropy_coef)
        requested_device = _select_ppo_device(agent.network)
        try:
            metrics = optimize_ppo_policy(
                network=agent.network,
                optimizer=agent.optimizer,
                steps=steps,
                ppo=agent.ppo_hparams,
                policy_temperature=policy_temp,
                ppo_device_override=requested_device,
            )
        except Exception as exc:
            if requested_device.type != "cpu" and _is_cuda_oom(exc):
                logger.warning(
                    "CUDA OOM during PPO update for agent %s; retrying on CPU.",
                    agent.id[:8],
                )
                try:
                    agent.optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                torch.cuda.empty_cache()
                _move_network_and_optimizer_to(agent.network, agent.optimizer, torch.device("cpu"))
                metrics = optimize_ppo_policy(
                    network=agent.network,
                    optimizer=agent.optimizer,
                    steps=steps,
                    ppo=agent.ppo_hparams,
                    policy_temperature=policy_temp,
                    ppo_device_override=torch.device("cpu"),
                )
            else:
                raise
        if metrics:
            metrics.update(agent._adapt_ppo_learning_rate(float(metrics.get("ppo/approx_kl", 0.0))))
            metrics["ppo/learning_rate"] = float(agent.optimizer.param_groups[0].get("lr", 0.0))
            metrics["ppo/target_kl"] = float(agent.ppo_hparams.target_kl)
            metrics["ppo/entropy_coef"] = float(current_entropy_coef)
        agent.network.eval()

        # PPO training may strand the network on CPU after a CUDA OOM fallback.
        # Re-sync so _inference_device matches where the network actually lives,
        # and opportunistically try to reclaim CUDA if enough VRAM is now free.
        try:
            actual_device = next(agent.network.parameters()).device
            if not _devices_match(actual_device, agent._inference_device):
                logger.warning(
                    "Network device (%s) differs from inference device (%s) after PPO; re-syncing.",
                    actual_device, agent._inference_device,
                )
                agent._inference_device = actual_device
        except StopIteration:
            pass
        agent._try_reclaim_cuda()

        return metrics or {}


# ---------------------------------------------------------------------------
# Batched GPU inference: collects per-agent inference requests from
# concurrent games and runs one batched forward pass on the GPU.
# ---------------------------------------------------------------------------

_InferenceRequest = Tuple[np.ndarray, int, torch.Tensor, "asyncio.Future[Any]", asyncio.AbstractEventLoop]


class InferenceBatcher:
    """Collects inference requests for a single agent's network and runs
    them in one batched GPU forward pass.

    Each agent owns one batcher instance (created lazily).  Callers await
    ``batcher.infer(...)`` which returns the same 5-tuple as
    ``_sync_forward_and_probs``.
    """

    def __init__(
        self,
        agent: "RLAgent",
        max_batch: int = 32,
        deadline_ms: float = 3.0,
    ):
        self._agent = agent
        self._max_batch = max(1, max_batch)
        self._deadline_sec = max(0.0005, deadline_ms / 1000.0)
        self._pending: List[_InferenceRequest] = []
        self._lock = threading.Lock()
        self._has_items = threading.Event()
        self._batch_ready = threading.Event()
        self._alive = True
        self._worker = threading.Thread(
            target=self._run, daemon=True,
            name=f"infer_batch_{agent.id[:8]}",
        )
        self._worker.start()

    # -- public async API ----------------------------------------------------

    async def infer(
        self,
        state_vector: np.ndarray,
        phase_index: int,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any, torch.Tensor]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        with self._lock:
            self._pending.append((state_vector, phase_index, recurrent_state, fut, loop))
            if len(self._pending) >= self._max_batch:
                self._batch_ready.set()
        self._has_items.set()
        return await fut

    # -- batch forward with OOM fallback ------------------------------------

    def _run_batch_forward(
        self, batch: list, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any, torch.Tensor]:
        with self._agent._model_device_lock:
            device = self._agent._ensure_network_device_consistency(device)

            states = torch.tensor(
                np.stack([r[0] for r in batch]),
                dtype=torch.float32, device=device,
            )
            phases = torch.tensor(
                [r[1] for r in batch],
                dtype=torch.long, device=device,
            )
            recurrent = torch.stack([r[2] for r in batch]).to(device)

            try:
                with torch.no_grad():
                    use_amp = device.type == "cuda"
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = self._agent._forward_network(
                            states, phase_indices=phases, recurrent_state=recurrent,
                        )

                    policy_logits = out["policy_logits"].float()
                    value = out["value"].float()
                    recurrent_out = out.get("recurrent_state")
                    if recurrent_out is not None:
                        recurrent_out = recurrent_out.float()
                    aux_preds = out.get("aux_predictions")

                    temperature = self._agent._effective_policy_temperature()
                    policy_logits = policy_logits / max(temperature, 1e-3)
                    policy_probs = F.softmax(policy_logits, dim=-1)

                if device.type != "cpu":
                    policy_logits = policy_logits.cpu()
                    value = value.cpu()
                    policy_probs = policy_probs.cpu()
                    if aux_preds is not None:
                        aux_preds = aux_preds.cpu()

                return policy_logits, value, recurrent_out, aux_preds, policy_probs

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if device.type == "cpu":
                    raise
                err_msg = str(exc)
                if "CUDA" not in err_msg and "out of memory" not in err_msg:
                    raise
                logger.warning("CUDA OOM in batched inference; clearing cache and falling back to CPU.")
                torch.cuda.empty_cache()
                cpu = torch.device("cpu")
                _move_network_and_optimizer_to(self._agent.network, self._agent.optimizer, cpu)
                self._agent._inference_device = cpu
                return self._run_batch_forward(batch, cpu)

    # -- background worker ---------------------------------------------------

    def _run(self) -> None:
        while self._alive:
            self._has_items.wait()
            self._has_items.clear()
            if not self._alive:
                break
            # Brief accumulation window (or early-out when batch is full).
            self._batch_ready.wait(timeout=self._deadline_sec)
            self._batch_ready.clear()
            self._process_batch()

    def _process_batch(self) -> None:
        with self._lock:
            batch = self._pending[: self._max_batch]
            self._pending = self._pending[self._max_batch :]
            if self._pending:
                self._has_items.set()

        if not batch:
            return

        try:
            device = self._agent._inference_device
            policy_logits, value, recurrent_out, aux_preds, policy_probs = (
                self._run_batch_forward(batch, device)
            )

            for i, (_, _, _, fut, loop) in enumerate(batch):
                r_out = recurrent_out[i : i + 1] if recurrent_out is not None else None
                a_out = aux_preds[i : i + 1] if aux_preds is not None else None
                result = (
                    policy_logits[i : i + 1],
                    value[i : i + 1],
                    r_out,
                    a_out,
                    policy_probs[i : i + 1],
                )
                loop.call_soon_threadsafe(fut.set_result, result)
        except Exception as exc:
            for _, _, _, fut, loop in batch:
                if not fut.done():
                    loop.call_soon_threadsafe(fut.set_exception, exc)

    def shutdown(self) -> None:
        self._alive = False
        self._has_items.set()
        self._batch_ready.set()
        self._worker.join(timeout=5.0)


@dataclass
class AgentConfig:
    hidden_size: int = 256
    recurrent_size: int = 128
    phase_head_count: int = 6
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    epsilon: float = 0.05  # Reduced for more policy-driven behavior
    temperature: float = 1.2  # Slightly increased for more exploration
    max_thinking_time: float = 5.0  # Max seconds to think per move
    train_from_self_play: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    max_episode_steps: int = 512
    planner_token_dim: int = 64
    planner_global_dim: int = 16
    planner_type_vocab_size: int = 16
    planner_opportunity_limit: int = 12
    planner_tableau_limit: int = 24
    planner_hand_limit: int = 24
    planner_opponent_limit: int = 4
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    planner_aux_output_dim: int = 280

    @classmethod
    def from_env(cls, base: Optional["AgentConfig"] = None) -> "AgentConfig":
        seed = base or cls()
        payload = asdict(seed)
        payload.update({
            "hidden_size": _safe_env_int("AGENT_HIDDEN_SIZE", int(payload["hidden_size"])),
            "recurrent_size": _safe_env_int("AGENT_RECURRENT_SIZE", int(payload["recurrent_size"])),
            "phase_head_count": _safe_env_int("AGENT_PHASE_HEAD_COUNT", int(payload["phase_head_count"])),
            "planner_token_dim": _safe_env_int("AGENT_PLANNER_TOKEN_DIM", int(payload["planner_token_dim"])),
            "planner_global_dim": _safe_env_int("AGENT_PLANNER_GLOBAL_DIM", int(payload["planner_global_dim"])),
            "planner_type_vocab_size": _safe_env_int("AGENT_PLANNER_TYPE_VOCAB_SIZE", int(payload["planner_type_vocab_size"])),
            "planner_opportunity_limit": _safe_env_int("AGENT_PLANNER_OPPORTUNITY_LIMIT", int(payload["planner_opportunity_limit"])),
            "planner_tableau_limit": _safe_env_int("AGENT_PLANNER_TABLEAU_LIMIT", int(payload["planner_tableau_limit"])),
            "planner_hand_limit": _safe_env_int("AGENT_PLANNER_HAND_LIMIT", int(payload["planner_hand_limit"])),
            "planner_opponent_limit": _safe_env_int("AGENT_PLANNER_OPPONENT_LIMIT", int(payload["planner_opponent_limit"])),
            "transformer_heads": _safe_env_int("AGENT_TRANSFORMER_HEADS", int(payload["transformer_heads"])),
            "transformer_layers": _safe_env_int("AGENT_TRANSFORMER_LAYERS", int(payload["transformer_layers"])),
            "transformer_dropout": _safe_env_float("AGENT_TRANSFORMER_DROPOUT", float(payload["transformer_dropout"])),
            "planner_aux_output_dim": _safe_env_int("AGENT_PLANNER_AUX_OUTPUT_DIM", int(payload["planner_aux_output_dim"])),
        })
        return cls(**payload)

    def planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            token_dim=max(16, int(self.planner_token_dim)),
            global_dim=max(16, int(self.planner_global_dim)),
            type_vocab_size=max(10, int(self.planner_type_vocab_size)),
            opportunity_limit=max(1, int(self.planner_opportunity_limit)),
            tableau_limit=max(0, int(self.planner_tableau_limit)),
            hand_limit=max(0, int(self.planner_hand_limit)),
            opponent_limit=max(0, int(self.planner_opponent_limit)),
        )

    def architecture_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            key: payload[key]
            for key in _AGENT_ARCHITECTURE_FIELDS
            if key in payload
        }

    def architecture_signature(self) -> str:
        return json.dumps(self.architecture_dict(), sort_keys=True, separators=(",", ":"))

class TerraformingMarsNetwork(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        planner = config.planner_config()
        self.planner_config = planner
        self.recurrent_size = max(16, int(config.recurrent_size))
        self.phase_head_count = max(2, int(config.phase_head_count))
        self.hidden_size = max(64, int(config.hidden_size))
        transformer_heads = max(1, int(getattr(config, "transformer_heads", 4)))
        transformer_layers = max(1, int(getattr(config, "transformer_layers", 2)))
        transformer_dropout = max(0.0, float(getattr(config, "transformer_dropout", 0.1)))
        self.planner_aux_output_dim = max(32, int(getattr(config, "planner_aux_output_dim", 280)))
        self.action_dim = 0

        self.world_projection = nn.Linear(int(planner.token_dim), self.hidden_size)
        self.hand_projection = nn.Linear(int(planner.token_dim), self.hidden_size)
        self.action_projection = nn.Linear(int(planner.token_dim), self.hidden_size)
        self.global_projection = nn.Linear(int(planner.global_dim), self.hidden_size)
        self.world_type_embedding = nn.Embedding(int(planner.type_vocab_size), self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=transformer_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=transformer_dropout,
            batch_first=True,
            activation='gelu',
        )
        try:
            self.world_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.world_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.hand_to_world = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.action_to_world = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(self.hidden_size)
        self.recurrent_cell = nn.GRUCell(self.hidden_size, self.recurrent_size)
        self.recurrent_to_hidden = nn.Linear(self.recurrent_size, self.hidden_size)
        self.phase_embedding = nn.Embedding(self.phase_head_count, self.hidden_size)
        self.action_context_fuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.action_logit_head = nn.Linear(self.hidden_size, 1)
        self.last_transformer_stats: Dict[str, Any] = {
            "enabled": True,
            "active_token_ratio": 0.0,
            "active_row_ratio": 0.0,
            "attention_context_norm": 0.0,
            "fusion_delta_norm": 0.0,
            "fusion_share": 0.0,
            "shared_pre_norm": 0.0,
            "shared_post_norm": 0.0,
            "token_count": 0,
            "token_dim": int(planner.token_dim),
            "timestamp": 0.0,
        }
        value_hidden_1 = max(64, int(self.hidden_size))
        value_hidden_2 = max(32, int(self.hidden_size // 2))
        self.value_trunk = nn.Sequential(
            nn.Linear(self.hidden_size, value_hidden_1),
            nn.ReLU(),
            nn.Linear(value_hidden_1, value_hidden_2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(value_hidden_2, 1)
        self.aux_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.planner_aux_output_dim),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((int(values.shape[0]), int(values.shape[-1])), dtype=values.dtype, device=values.device)
        mask_f = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)
        return (values * mask_f).sum(dim=1) / denom

    def _prepare_recurrent(self, batch_size: int, device: torch.device, dtype: torch.dtype, recurrent_state: Optional[torch.Tensor]) -> torch.Tensor:
        if recurrent_state is None:
            return torch.zeros((batch_size, self.recurrent_size), dtype=dtype, device=device)
        recurrent_state = recurrent_state.to(device=device, dtype=dtype)
        if recurrent_state.dim() == 1:
            recurrent_state = recurrent_state.unsqueeze(0)
        if int(recurrent_state.shape[0]) != batch_size:
            recurrent_state = recurrent_state[:1].repeat(batch_size, 1)
        if int(recurrent_state.shape[1]) < self.recurrent_size:
            padded = torch.zeros((batch_size, self.recurrent_size), dtype=dtype, device=device)
            padded[:, :int(recurrent_state.shape[1])] = recurrent_state
            recurrent_state = padded
        elif int(recurrent_state.shape[1]) > self.recurrent_size:
            recurrent_state = recurrent_state[:, :self.recurrent_size]
        return recurrent_state

    def forward(
        self,
        state: Any,
        phase_indices: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            raise TypeError("Planner policy expects a structured planner state bundle")

        world_tokens = state["world_tokens"].float()
        world_token_types = state["world_token_types"].long()
        world_mask = state["world_mask"].bool()
        hand_tokens = state["hand_tokens"].float()
        hand_mask = state["hand_mask"].bool()
        action_tokens = state["action_tokens"].float()
        action_mask = state["action_mask"].bool()
        global_scalars = state["global_scalars"].float()

        batch_size = int(world_tokens.shape[0])
        device = world_tokens.device

        world_embed = self.world_projection(world_tokens) + self.world_type_embedding(
            torch.clamp(world_token_types, 0, int(self.planner_config.type_vocab_size) - 1)
        )
        global_embed = self.global_projection(global_scalars).unsqueeze(1)
        world_embed = world_embed + global_embed
        safe_world_mask = world_mask.clone()
        if safe_world_mask.numel() > 0:
            empty_rows = ~safe_world_mask.any(dim=1)
            if bool(empty_rows.any()):
                safe_world_mask[empty_rows, 0] = True
        else:
            empty_rows = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        encoded_world = self.world_encoder(world_embed, src_key_padding_mask=~safe_world_mask)
        world_summary = self._masked_mean(encoded_world, safe_world_mask)

        if hand_tokens.shape[1] > 0 and bool(hand_mask.any()):
            projected_hand = self.hand_projection(hand_tokens)
            hand_attended, _ = self.hand_to_world(
                query=projected_hand,
                key=encoded_world,
                value=encoded_world,
                key_padding_mask=~safe_world_mask,
                need_weights=False,
            )
            hand_summary = self._masked_mean(hand_attended, hand_mask)
        else:
            hand_summary = torch.zeros_like(world_summary)

        if phase_indices is None:
            phase_indices = torch.zeros((batch_size,), dtype=torch.long, device=device)
        else:
            phase_indices = torch.clamp(phase_indices.to(device=device, dtype=torch.long).reshape(-1), 0, self.phase_head_count - 1)
            if int(phase_indices.shape[0]) != batch_size:
                phase_indices = phase_indices[:1].repeat(batch_size)
        phase_context = self.phase_embedding(phase_indices)

        summary = self.summary_norm(world_summary + hand_summary + phase_context + global_embed.squeeze(1))
        recurrent_in = self._prepare_recurrent(batch_size, device, summary.dtype, recurrent_state)
        recurrent_out = self.recurrent_cell(summary, recurrent_in)
        recurrent_context = torch.tanh(self.recurrent_to_hidden(recurrent_out))
        fused_summary = summary + recurrent_context

        projected_actions = self.action_projection(action_tokens)
        if projected_actions.shape[1] > 0 and bool(action_mask.any()):
            action_attended, _ = self.action_to_world(
                query=projected_actions,
                key=encoded_world,
                value=encoded_world,
                key_padding_mask=~safe_world_mask,
                need_weights=False,
            )
        else:
            action_attended = projected_actions

        summary_expanded = fused_summary.unsqueeze(1).expand(-1, int(projected_actions.shape[1]), -1)
        fused_actions = self.action_context_fuser(torch.cat([projected_actions, action_attended, summary_expanded], dim=-1))
        policy_logits = self.action_logit_head(fused_actions).squeeze(-1)
        policy_logits = policy_logits.masked_fill(~action_mask, -1e9)

        value_features = self.value_trunk(fused_summary)
        value = self.value_head(value_features)
        aux_raw = self.aux_head(fused_summary)
        aux_predictions = torch.sigmoid(aux_raw)

        with torch.no_grad():
            self.last_transformer_stats = {
                "enabled": True,
                "active_token_ratio": float(world_mask.float().mean().item()) if world_mask.numel() > 0 else 0.0,
                "active_row_ratio": float(world_mask.any(dim=1).float().mean().item()) if world_mask.numel() > 0 else 0.0,
                "attention_context_norm": float(encoded_world.detach().norm(dim=-1).mean().item()) if encoded_world.numel() > 0 else 0.0,
                "fusion_delta_norm": float(fused_summary.detach().norm(dim=-1).mean().item()) if fused_summary.numel() > 0 else 0.0,
                "fusion_share": 0.0,
                "shared_pre_norm": float(world_summary.detach().norm(dim=-1).mean().item()) if world_summary.numel() > 0 else 0.0,
                "shared_post_norm": float(fused_summary.detach().norm(dim=-1).mean().item()) if fused_summary.numel() > 0 else 0.0,
                "token_count": int(world_tokens.shape[1]),
                "token_dim": int(world_tokens.shape[2]) if world_tokens.dim() == 3 else int(self.planner_config.token_dim),
                "timestamp": float(time.time()),
            }

        return {
            "policy_logits": policy_logits,
            "value": value,
            "recurrent_state": recurrent_out,
            "aux_predictions": aux_predictions,
            "aux_milestone_logits": None,
        }


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
        raise ValueError("Network output missing policy/value tensors")

    return {
        "policy_logits": policy_logits,
        "value": value,
        "recurrent_state": recurrent_state,
        "aux_predictions": aux_predictions,
        "aux_milestone_logits": aux_milestone_logits,
    }
# Removed conflicting Agent class - using RLAgent instead
        
class RLAgent:
    def __init__(self, config: AgentConfig = None, agent_id: str = None):
        self.id = agent_id or str(uuid.uuid4())
        if config is None:
            config = self.build_env_config()
        self.config = config
        # PPO optimizer LR is decoupled from evolutionary config learning_rate by default.
        self.ppo_learning_rate = self._safe_env_float("PPO_LEARNING_RATE", 3e-4)
        self.ppo_lr_min = self._safe_env_float(
            "PPO_LR_MIN",
            max(1e-6, float(self.ppo_learning_rate) * 0.2),
        )
        self.ppo_lr_max = self._safe_env_float(
            "PPO_LR_MAX",
            max(float(self.ppo_lr_min), float(self.ppo_learning_rate) * 5.0),
        )
        self.ppo_lr_adapt_up = self._safe_env_float("PPO_LR_ADAPT_UP", 1.08)
        self.ppo_lr_adapt_down = self._safe_env_float("PPO_LR_ADAPT_DOWN", 0.75)
        
        # Neural network
        self.network = TerraformingMarsNetwork(self.config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.ppo_learning_rate)
        self.network.eval()
        self._model_device_lock = threading.RLock()

        self._inference_device = _resolve_inference_device()
        self._move_network_to_inference_device()
        self._inference_batcher: Optional[InferenceBatcher] = None
        self._init_inference_batcher()

        rust_info = require_backend_info()
        logger.info(
            "Rust backend ready: module=%s api=%s crate=%s",
            rust_info.get("module"),
            rust_info.get("api_version"),
            rust_info.get("crate_version"),
        )

        # Game interaction components
        planner_config = self.config.planner_config()
        self.state_encoder = StateEncoder(planner_config=planner_config)
        self.action_decoder = ActionDecoder(planner_config=planner_config)
        
        # Self-play learning configuration (env overrides)
        self.train_from_self_play = (
            self.config.train_from_self_play and os.getenv("SELF_PLAY_LEARNING", "1") != "0"
        )
        self.self_play_reward_scale = float(os.getenv("SELF_PLAY_REWARD_SCALE", "1.0"))
        self.training_lock = asyncio.Lock()
        self.ppo_enable = str(os.getenv("PPO_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.ppo_rollout_steps = self._safe_env_int("PPO_ROLLOUT_STEPS", 8192)
        self.ppo_buffer_max_steps = self._safe_env_int("PPO_BUFFER_MAX_STEPS", 65536)
        self.state_schema_version = str(os.getenv("STATE_SCHEMA_VERSION", "v1")).strip() or "v1"
        self.strict_on_policy_sampling = str(os.getenv("PPO_STRICT_ON_POLICY", "1")).strip().lower() not in ("0", "false", "no", "off")
        self.exploration_decay_games = max(1, self._safe_env_int("EXPLORATION_DECAY_GAMES", 200))
        self.policy_epsilon_cap = max(0.0, self._safe_env_float("POLICY_EPSILON_CAP", 0.02))
        self.policy_epsilon_floor = max(0.0, self._safe_env_float("POLICY_EPSILON_FLOOR", 0.001))
        self.policy_temperature_cap = max(1e-3, self._safe_env_float("POLICY_TEMPERATURE_CAP", 1.0))
        self.policy_temperature_floor = max(1e-3, self._safe_env_float("POLICY_TEMPERATURE_FLOOR", 0.75))
        self.project_card_priority_weight = max(0.1, self._safe_env_float("PLAY_CARD_PRIORITY_WEIGHT", 1.6))
        self.max_fallback_attempts = max(1, self._safe_env_int("MAX_FALLBACK_ACTION_ATTEMPTS", 6))
        self.max_fallback_random_retries_per_prompt = max(1, self._safe_env_int("MAX_FALLBACK_RANDOM_RETRIES_PER_PROMPT", 10))
        self.rejected_action_memory_size = max(64, self._safe_env_int("REJECTED_ACTION_MEMORY_SIZE", 2048))
        self.ppo_hparams = PPOHyperParameters(
            clip_eps=self._safe_env_float("PPO_CLIP_EPS", 0.2),
            value_clip_eps=self._safe_env_float("PPO_VALUE_CLIP_EPS", 0.2),
            gamma=self._safe_env_float("PPO_GAMMA", 0.99),
            gae_lambda=self._safe_env_float("PPO_GAE_LAMBDA", 0.95),
            epochs=self._safe_env_int("PPO_EPOCHS", 4),
            minibatch_size=self._safe_env_int("PPO_MINIBATCH_SIZE", 1024),
            entropy_coef=self._safe_env_float("PPO_ENTROPY_COEF", 0.01),
            value_coef=self._safe_env_float("PPO_VALUE_COEF", 0.7),
            aux_coef=self._safe_env_float("PPO_AUX_COEF", 0.1),
            max_grad_norm=self._safe_env_float("PPO_MAX_GRAD_NORM", 1.0),
            target_kl=self._safe_env_float("PPO_TARGET_KL", 0.01),
        )
        self.ppo_entropy_coef_start = max(
            0.0,
            self._safe_env_float("PPO_ENTROPY_COEF_START", float(self.ppo_hparams.entropy_coef)),
        )
        self.ppo_entropy_coef_end = max(
            0.0,
            self._safe_env_float("PPO_ENTROPY_COEF_END", float(self.ppo_hparams.entropy_coef)),
        )
        self.ppo_entropy_coef_anneal_games = max(
            1,
            self._safe_env_int("PPO_ENTROPY_COEF_ANNEAL_GAMES", 1500),
        )
        self.reward_shaping_initial_coef = max(0.0, self._safe_env_float("PPO_SHAPING_INITIAL_COEF", 1.0))
        self.reward_shaping_final_coef = max(0.0, self._safe_env_float("PPO_SHAPING_FINAL_COEF", 0.0))
        self.reward_shaping_anneal_games = max(1, self._safe_env_int("PPO_SHAPING_ANNEAL_GAMES", 2500))
        self.rare_state_priority_bonus = max(0.0, self._safe_env_float("PPO_RARE_STATE_PRIORITY_BONUS", 0.9))
        self.rare_high_cost_payment_threshold = max(1.0, self._safe_env_float("PPO_RARE_HIGH_COST_PAYMENT_THRESHOLD", 20.0))
        self.reward_tr_weight = self._safe_env_float("PPO_SHAPING_TR_WEIGHT", 3.3)
        self.reward_cards_vp_weight = self._safe_env_float("PPO_SHAPING_CARDS_VP_WEIGHT", 2.7)
        self.reward_city_greenery_weight = self._safe_env_float("PPO_SHAPING_CITY_GREENERY_WEIGHT", 2.4)
        self.reward_milestones_awards_weight = self._safe_env_float("PPO_SHAPING_MILESTONES_AWARDS_WEIGHT", 2.3)
        self.reward_other_weight = self._safe_env_float("PPO_SHAPING_OTHER_WEIGHT", 0.5)
        self.reward_debug_enabled = str(os.getenv("PPO_REWARD_DEBUG_ENABLED", "0")).strip().lower() not in ("0", "false", "no", "off")
        self.reward_debug_threshold = max(0.0, self._safe_env_float("PPO_REWARD_DEBUG_THRESHOLD", 0.001))
        self.reward_debug_log_every = max(1, self._safe_env_int("PPO_REWARD_DEBUG_LOG_EVERY", 200))
        self._reward_debug_counter = 0
        self.rollout_buffer: deque[PPORolloutStep] = deque(maxlen=max(1, int(self.ppo_buffer_max_steps)))
        self.active_poll_interval_sec = max(
            0.0,
            self._safe_env_float(
                "AGENT_ACTIVE_POLL_INTERVAL_SEC",
                self._safe_env_float("AGENT_POLL_INTERVAL_SEC", 0.2),
            ),
        )
        # Backward compatibility for existing call-sites that still reference poll_interval_sec.
        self.poll_interval_sec = float(self.active_poll_interval_sec)
        self.idle_poll_interval_sec = max(
            0.0,
            self._safe_env_float("AGENT_IDLE_POLL_INTERVAL_SEC", 0.12),
        )
        self.post_move_sleep_sec = float(os.getenv("AGENT_POST_MOVE_SLEEP_SEC", "0.0"))
        self.failure_pause_sec = float(os.getenv("AGENT_FAILURE_PAUSE_SEC", "0.0"))
        self.timing_log_every_n_decisions = max(
            1,
            self._safe_env_int("AGENT_TIMING_LOG_EVERY_N_DECISIONS", 200),
        )
        self.initial_cards_fallback_max_attempts = max(
            0,
            self._safe_env_int("AGENT_INITIAL_CARDS_FALLBACK_MAX_ATTEMPTS", 1),
        )
        self.initial_cards_reject_pause_sec = max(
            0.0,
            self._safe_env_float(
                "AGENT_INITIAL_CARDS_REJECT_PAUSE_SEC",
                max(self.failure_pause_sec, self.poll_interval_sec),
            ),
        )
        startup_autosubmit_raw = str(os.getenv("AGENT_STARTUP_AUTOSUBMIT", "")).strip().lower()
        startup_selection_mode = str(os.getenv("AGENT_STARTUP_PLAN_SELECTION", "best")).strip().lower()
        if startup_autosubmit_raw:
            self.startup_autosubmit = startup_autosubmit_raw in ("1", "true", "yes", "on")
        else:
            # Legacy startup selection expects deterministic heuristic submission,
            # not policy-sampled startup-plan actions.
            self.startup_autosubmit = startup_selection_mode in ("legacy", "original", "legacy_only")
        self.stuck_log_cooldown_sec = float(os.getenv("AGENT_STUCK_LOG_COOLDOWN_SEC", "5.0"))
        self._last_stuck_log_by_player: Dict[str, float] = {}
        self._rejected_actions_by_prompt: Dict[str, set[int]] = {}
        self._rejected_action_prompt_order: deque[str] = deque()
        self._fallback_random_retries_by_prompt: Dict[str, int] = {}
        self._fallback_retry_prompt_order: deque[str] = deque()
        self._recurrent_hidden_by_player: Dict[str, torch.Tensor] = {}
        # Track how many actions each player has taken in the current action-phase
        # turn.  Reset when phase transitions away from 'action'.
        # This counter is passed to the state encoder so the network can
        # differentiate first-action from second-action decisions.
        self._turn_action_count_by_player: Dict[str, int] = {}
        self._last_phase_by_player: Dict[str, str] = {}
        
        # Performance tracking
        self.games_played = 0
        self.total_victory_points = 0
        self.wins = 0
        self.decision_stats: Dict[str, Any] = {
            'total_decisions': 0,
            'policy_attempts': 0,
            'policy_successes': 0,
            'policy_rejections': 0,
            'policy_sampled_actions': 0,
            'epsilon_random_actions': 0,
            'fallback_decisions': 0,
            'fallback_random_attempts': 0,
            'fallback_random_successes': 0,
            'fallback_passes': 0,
            'no_available_actions': 0,
            'sum_available_actions': 0,
            'available_action_observations': 0,
            'action_type_counts': {},
            'standard_project_counts': {},
            'card_play_actions': 0,
            'steel_spent': 0,
            'titanium_spent': 0,
            'project_payment_value_total': 0.0,
            'metal_payment_value_total': 0.0,
            'steel_payment_value_total': 0.0,
            'titanium_payment_value_total': 0.0,
            'action_mask_observations': 0,
            'action_legal_count_total': 0,
            'action_rejected_by_server': 0,
            'policy_actions_blocked_by_reject_cache': 0,
            'rare_state_samples': 0,
            'rare_award_funding': 0,
            'rare_milestone_timing': 0,
            'rare_draft_keep_buy': 0,
            'rare_high_cost_payment': 0,
            'hate_draft_picks': 0,
            'draft_decisions_total': 0,
            'draft_decisions_low_hand_ev': 0,
            'hate_draft_picks_low_hand_ev': 0,
            'milestone_snipes': 0,
            'award_snipes': 0,
            'timing_totals_sec': {},
            'timing_counts': {},
            'timing_log_events': 0,
        }

    @staticmethod
    def build_env_config(base: Optional[AgentConfig] = None) -> AgentConfig:
        return AgentConfig.from_env(base=base)

    @staticmethod
    def peek_checkpoint_config(path: str) -> Optional[AgentConfig]:
        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            return None
        config_payload = checkpoint.get("config", {})
        if not isinstance(config_payload, dict):
            return None
        defaults = asdict(AgentConfig())
        merged = defaults.copy()
        for key, value in config_payload.items():
            if key in defaults:
                merged[key] = value
        # Compatibility-only mapping for older checkpoint/config payloads.
        legacy_config_map = {
            "tableau_token_count": "planner_tableau_limit",
            "hand_token_count": "planner_hand_limit",
            "opponent_token_count": "planner_opponent_limit",
            "card_token_dim": "planner_token_dim",
        }
        for legacy_key, new_key in legacy_config_map.items():
            if legacy_key in config_payload and new_key in defaults:
                merged[new_key] = config_payload[legacy_key]
        try:
            return AgentConfig(**merged)
        except Exception:
            return None

    @staticmethod
    def _safe_env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    @staticmethod
    def _safe_env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return int(default)

    def _move_network_to_inference_device(self) -> None:
        """Move the network (and optimizer states) to ``self._inference_device``."""
        with self._model_device_lock:
            target = self._inference_device
            try:
                current = next(self.network.parameters()).device
            except StopIteration:
                return
            if _devices_match(current, target):
                return
            try:
                _move_network_and_optimizer_to(self.network, self.optimizer, target)
            except Exception as exc:
                logger.warning(
                    "Failed to move network to %s (%s); falling back to CPU",
                    target, exc,
                )
                self._inference_device = torch.device("cpu")
                _move_network_and_optimizer_to(self.network, self.optimizer, self._inference_device)

    def _get_network_devices(self) -> List[torch.device]:
        devices: List[torch.device] = []
        seen: set[str] = set()
        for tensor in list(self.network.parameters()) + list(self.network.buffers()):
            device = tensor.device
            key = str(device)
            if key in seen:
                continue
            seen.add(key)
            devices.append(device)
        return devices

    def _ensure_network_device_consistency(
        self,
        preferred_device: Optional[torch.device] = None,
    ) -> torch.device:
        """Repair mixed or stale model-device placement before inference."""
        target = preferred_device or self._inference_device
        devices = self._get_network_devices()
        if not devices:
            return target
        if len(devices) == 1 and _devices_match(devices[0], target):
            return target

        device_list = ", ".join(str(device) for device in devices)
        if len(devices) == 1:
            logger.warning(
                "Network device drift for agent %s: model=%s inference=%s. Re-syncing.",
                self.id[:8],
                devices[0],
                target,
            )
        else:
            logger.warning(
                "Mixed network devices for agent %s: %s. Re-syncing to %s.",
                self.id[:8],
                device_list,
                target,
            )

        try:
            _move_network_and_optimizer_to(self.network, self.optimizer, target)
            self._inference_device = target
            return target
        except Exception as exc:
            if target.type == "cpu":
                raise
            cpu = torch.device("cpu")
            logger.warning(
                "Failed to re-sync network to %s for agent %s (%s); falling back to CPU.",
                target,
                self.id[:8],
                exc,
            )
            _move_network_and_optimizer_to(self.network, self.optimizer, cpu)
            self._inference_device = cpu
            return cpu

    def _try_reclaim_cuda(self) -> None:
        """If currently on CPU after an OOM fallback, attempt to move back to CUDA."""
        with self._model_device_lock:
            if self._inference_device.type != "cpu":
                return
            desired = _resolve_inference_device()
            if desired.type == "cpu":
                return
            torch.cuda.empty_cache()
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(desired.index or 0)
            except Exception:
                return
            param_bytes = sum(p.numel() * p.element_size() for p in self.network.parameters())
            if free_mem < param_bytes * 3:
                return
            logger.info("Reclaiming CUDA: %.0f MiB free, model ~%.0f MiB", free_mem / 1e6, param_bytes / 1e6)
            self._inference_device = desired
            self._move_network_to_inference_device()

    def _init_inference_batcher(self) -> None:
        """Create an ``InferenceBatcher`` when CUDA inference is active and
        the ``AGENT_INFERENCE_BATCH`` env var is not explicitly disabled."""
        env_val = os.getenv("AGENT_INFERENCE_BATCH", "auto").strip().lower()
        if env_val in ("0", "false", "no", "off"):
            return
        if env_val == "auto" and self._inference_device.type != "cuda":
            return
        max_batch = max(1, self._safe_env_int("AGENT_INFERENCE_BATCH_SIZE", 32))
        deadline_ms = max(0.5, self._safe_env_float("AGENT_INFERENCE_BATCH_DEADLINE_MS", 1.0))
        self._inference_batcher = InferenceBatcher(
            self, max_batch=max_batch, deadline_ms=deadline_ms,
        )

    def _forward_network(
        self,
        state_tensor: torch.Tensor,
        phase_indices: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        try:
            raw = self.network(state_tensor, phase_indices=phase_indices, recurrent_state=recurrent_state)
        except TypeError:
            raw = self.network(state_tensor)
        return _normalize_network_output(raw)

    def _zero_recurrent_state(self) -> torch.Tensor:
        recurrent_size = max(1, int(getattr(self.network, "recurrent_size", max(16, self.config.hidden_size // 2))))
        return torch.zeros((recurrent_size,), dtype=torch.float32, device=self._inference_device)

    def _get_recurrent_state_for_player(self, player_id: Optional[str]) -> torch.Tensor:
        if not player_id:
            return self._zero_recurrent_state()
        cached = self._recurrent_hidden_by_player.get(str(player_id))
        if isinstance(cached, torch.Tensor):
            return cached.detach().clone().float().reshape(-1)
        return self._zero_recurrent_state()

    def _set_recurrent_state_for_player(self, player_id: Optional[str], recurrent_state: Any) -> None:
        if not player_id:
            return
        if recurrent_state is None:
            self._recurrent_hidden_by_player.pop(str(player_id), None)
            return
        device = self._inference_device
        if isinstance(recurrent_state, torch.Tensor):
            vec = recurrent_state.detach().float().reshape(-1)
            if not _devices_match(vec.device, device):
                vec = vec.to(device)
        else:
            try:
                vec = torch.tensor(
                    np.asarray(recurrent_state, dtype=np.float32).reshape(-1),
                    dtype=torch.float32, device=device,
                )
            except Exception:
                return
        self._recurrent_hidden_by_player[str(player_id)] = vec

    def _clear_recurrent_state_for_player(self, player_id: Optional[str]) -> None:
        if player_id:
            self._recurrent_hidden_by_player.pop(str(player_id), None)

    # ------------------------------------------------------------------
    # Turn-action count helpers
    # ------------------------------------------------------------------

    def _get_turn_action_count(self, player_id: Optional[str]) -> int:
        """Return the number of actions taken so far this turn (0 = first action)."""
        if not player_id:
            return 0
        return int(self._turn_action_count_by_player.get(str(player_id), 0))

    def _increment_turn_action_count(self, player_id: Optional[str]) -> None:
        """Increment the per-player action counter for the current turn."""
        if not player_id:
            return
        key = str(player_id)
        self._turn_action_count_by_player[key] = int(self._turn_action_count_by_player.get(key, 0)) + 1

    def _maybe_reset_turn_action_count(self, player_id: Optional[str], player_state: Optional[Dict[str, Any]]) -> None:
        """Reset the turn action counter when the phase transitions to/from 'action'.

        TM gives each player exactly 2 actions per generation-round during the
        action phase.  We detect a new round by watching for a phase change or
        by seeing the game move into a non-action phase and back.
        """
        if not player_id:
            return
        key = str(player_id)
        phase = ""
        if isinstance(player_state, dict):
            game = player_state.get("game", {}) or {}
            phase = str(game.get("phase", "") or "").strip().lower()

        last_phase = self._last_phase_by_player.get(key, "")
        if phase != last_phase:
            # Phase changed: if we're entering action phase (fresh round) reset counter
            if phase == "action":
                self._turn_action_count_by_player[key] = 0
            self._last_phase_by_player[key] = phase

    def _extract_phase_index(self, player_state: Optional[Dict[str, Any]]) -> int:
        phase = ""
        if isinstance(player_state, dict):
            game = player_state.get("game", {}) or {}
            phase = str(game.get("phase", "") or "").strip().lower()
        phase_map = {
            "research": 0,
            "drafting": 1,
            "action": 2,
            "production": 3,
            "solar": 4,
        }
        return int(phase_map.get(phase, 5))

    def _project_award_points_for_color(self, scores: List[Dict[str, Any]], own_color: str) -> float:
        if not scores or not own_color:
            return 0.0
        own_color_l = str(own_color).strip().lower()
        normalized: List[Tuple[str, float]] = []
        for row in scores:
            if not isinstance(row, dict):
                continue
            color = str(row.get("playerColor", "") or "").strip().lower()
            if not color:
                continue
            try:
                score = float(row.get("playerScore", 0) or 0)
            except Exception:
                score = 0.0
            normalized.append((color, score))
        if not normalized:
            return 0.0
        normalized.sort(key=lambda pair: pair[1], reverse=True)
        top_score = normalized[0][1]
        top_colors = [color for color, score in normalized if score == top_score]
        if own_color_l in top_colors:
            return 5.0
        if len(top_colors) > 1:
            return 0.0
        remaining = normalized[len(top_colors):]
        if not remaining:
            return 0.0
        second_score = remaining[0][1]
        second_colors = [color for color, score in remaining if score == second_score]
        if own_color_l in second_colors:
            return 2.0
        return 0.0

    def _compute_aux_targets_legacy(self, player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        targets = {
            "milestone_claimability": np.zeros(70, dtype=np.float32),  # Vector of 70 milestone claimability scores
            "award_ev": 0.0,
            "playable_cards": 0.0,
            "steel_target": 0.0,
            "titanium_target": 0.0,
        }
        if not isinstance(player_state, dict):
            return targets

        game = player_state.get("game", {}) or {}
        player = player_state.get("thisPlayer", {}) or {}
        own_color = str(player.get("color", "") or "").strip().lower()

        # Build milestone lookup map
        milestones_raw = [m for m in (game.get("milestones", []) or []) if isinstance(m, dict)]
        milestone_by_name: Dict[str, Dict[str, Any]] = {}
        for m in milestones_raw:
            name = str(m.get("name", "") or "").strip()
            if name:
                milestone_by_name[name] = m

        # Compute claimability for each milestone in fixed order
        milestone_claimability = np.zeros(70, dtype=np.float32)
        for idx, milestone_name in enumerate(StateEncoder._ALL_MILESTONES):
            milestone = milestone_by_name.get(milestone_name)
            if not milestone:
                # Milestone doesn't exist in this game
                milestone_claimability[idx] = 0.0
                continue
            
            # Check if already claimed
            if milestone.get("playerName"):
                owner_color = str(milestone.get("playerColor", "") or "").strip().lower()
                if own_color and owner_color == own_color:
                    milestone_claimability[idx] = 1.0  # I have it
                else:
                    milestone_claimability[idx] = 0.0  # Opponent has it
                continue
            
            # Unclaimed - calculate progress from scores
            scores = [row for row in (milestone.get("scores", []) or []) if isinstance(row, dict)]
            if not scores:
                milestone_claimability[idx] = 0.0
                continue
            
            own_score = 0.0
            max_score = 0.0
            for row in scores:
                try:
                    score = float(row.get("playerScore", 0) or 0)
                except Exception:
                    score = 0.0
                max_score = max(max_score, score)
                row_color = str(row.get("playerColor", "") or "").strip().lower()
                if own_color and row_color == own_color:
                    own_score = score
            
            # Normalize progress: own_score / max(threshold, max_score)
            threshold = 3.0
            denominator = max(threshold, max_score, 1.0)
            progress_normalized = min(own_score / denominator, 1.0)
            milestone_claimability[idx] = float(progress_normalized)
        
        targets["milestone_claimability"] = milestone_claimability

        # Award EV proxy from currently funded awards.
        award_points = 0.0
        for award in game.get("awards", []) or []:
            if not isinstance(award, dict):
                continue
            if not award.get("playerName"):
                continue
            scores = [row for row in (award.get("scores", []) or []) if isinstance(row, dict)]
            award_points += self._project_award_points_for_color(scores, own_color)
        targets["award_ev"] = float(max(0.0, min(award_points / 15.0, 1.0)))

        cards_source = list(self.state_encoder._get_owned_hand_cards(player_state) or [])

        mc = float(player.get("megaCredits", 0) or 0)
        steel = float(player.get("steel", 0) or 0)
        titanium = float(player.get("titanium", 0) or 0)
        steel_prod = float(player.get("steelProduction", 0) or 0)
        titanium_prod = float(player.get("titaniumProduction", 0) or 0)
        steel_value = float(player.get("steelValue", 2) or 2)
        titanium_value = float(player.get("titaniumValue", 3) or 3)

        playable_now = 0
        building_opps = 0
        space_opps = 0
        for card in cards_source:
            card_name = str(card.get("name", "") or "")
            tags = {}
            try:
                tags = self.state_encoder._get_card_tags(card_name, fallback=card.get("tags", {}))
            except Exception:
                tags = {}
            try:
                card_cost = float(card.get("calculatedCost", card.get("cost", 0)) or 0)
            except Exception:
                card_cost = 0.0

            purchasing_power = mc
            if tags.get("Building", 0):
                purchasing_power += steel * steel_value
                building_opps += 1
            if tags.get("Space", 0):
                purchasing_power += titanium * titanium_value
                space_opps += 1
            if card_cost <= 0.0 or purchasing_power >= card_cost:
                playable_now += 1

        targets["playable_cards"] = float(max(0.0, min(float(playable_now) / 10.0, 1.0)))

        pool_count = float(max(1, len(cards_source)))
        building_ratio = float(building_opps) / pool_count
        space_ratio = float(space_opps) / pool_count
        steel_liquidity = min(((steel * steel_value) + (max(steel_prod, 0.0) * steel_value * 2.0)) / 80.0, 1.0)
        titanium_liquidity = min(((titanium * titanium_value) + (max(titanium_prod, 0.0) * titanium_value * 2.0)) / 90.0, 1.0)
        targets["steel_target"] = float(max(0.0, min(steel_liquidity * (0.25 + 0.75 * building_ratio), 1.0)))
        targets["titanium_target"] = float(max(0.0, min(titanium_liquidity * (0.25 + 0.75 * space_ratio), 1.0)))
        return targets

    def _compute_aux_targets(self, player_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        num_milestones = len(StateEncoder._ALL_MILESTONES)
        num_awards = len(StateEncoder._ALL_AWARDS)
        layout = planner_aux_layout(
            num_milestones=num_milestones,
            num_awards=num_awards,
            opportunity_limit=int(self.config.planner_opportunity_limit),
        )
        total_size = planner_aux_dim(
            num_milestones=num_milestones,
            num_awards=num_awards,
            opportunity_limit=int(self.config.planner_opportunity_limit),
        )
        targets = {"planner_vector": np.zeros(total_size, dtype=np.float32)}
        if not isinstance(player_state, dict):
            return targets

        game = player_state.get("game", {}) or {}
        player = player_state.get("thisPlayer", {}) or {}
        vector = targets["planner_vector"]
        milestone_claim_slice = layout["milestone_claim_now"]
        milestone_turns_slice = layout["milestone_turns_to_claim_bucket"]
        award_ev_slice = layout["award_fund_now_ev"]
        award_rank_slice = layout["award_rank_class"]
        carry_plants_slice = layout["carry_save_plants_value"]
        carry_heat_slice = layout["carry_save_heat_value"]
        next_turn_slice = layout["next_turn_combo_value"]
        next_gen_slice = layout["next_generation_combo_value"]
        opp_value_slice = layout["board_opportunity_value"]
        deny_risk_slice = layout["deny_risk"]

        for idx, milestone_name in enumerate(StateEncoder._ALL_MILESTONES):
            features = self.state_encoder._milestone_token_features(game, player, milestone_name)
            vector[milestone_claim_slice.start + idx] = float(features[6] if len(features) > 6 else 0.0)
            vector[milestone_turns_slice.start + idx] = float(1.0 - (features[5] if len(features) > 5 else 1.0))

        for idx, award_name in enumerate(StateEncoder._ALL_AWARDS):
            features = self.state_encoder._award_token_features(game, player, award_name)
            vector[award_ev_slice.start + idx] = float(features[6] if len(features) > 6 else 0.0)
            vector[award_rank_slice.start + idx] = float(features[5] if len(features) > 5 else 0.0)

        generation = max(1.0, float(game.get("generation", 1) or 1))
        plants = max(0.0, float(player.get("plants", 0) or 0))
        heat = max(0.0, float(player.get("heat", 0) or 0))
        plant_prod = max(0.0, float(player.get("plantProduction", 0) or 0))
        heat_prod = max(0.0, float(player.get("heatProduction", 0) or 0))
        vector[carry_plants_slice.start] = max(0.0, min(((plants + plant_prod) - 8.0 + 2.0) / 6.0, 1.0))
        vector[carry_heat_slice.start] = max(0.0, min(((heat + heat_prod) - 8.0 + 2.0) / 6.0, 1.0))
        vector[next_turn_slice.start] = max(0.0, min((generation - 8.0) / 6.0, 1.0))
        vector[next_gen_slice.start] = max(0.0, min((generation - 6.0) / 8.0, 1.0))

        opportunities = self.state_encoder._collect_board_opportunity_rows(player_state)
        for idx, row in enumerate(opportunities[:int(self.config.planner_opportunity_limit)]):
            vector[opp_value_slice.start + idx] = float(row[6] if len(row) > 6 else 0.0)
            vector[deny_risk_slice.start + idx] = float(row[8] if len(row) > 8 else 0.0)
        return targets

    def _exploration_progress(self) -> float:
        games = max(0, int(self.games_played))
        decay_games = max(1, int(self.exploration_decay_games))
        return max(0.0, min(1.0, float(games) / float(decay_games)))

    def _effective_policy_epsilon(self, force_random: bool = False) -> float:
        if force_random:
            return 1.0
        raw = max(0.0, float(self.config.epsilon))
        capped = min(raw, float(self.policy_epsilon_cap))
        floor = min(capped, max(0.0, float(self.policy_epsilon_floor)))
        progress = self._exploration_progress()
        decayed = floor + ((capped - floor) * (1.0 - progress))
        if self.strict_on_policy_sampling and self.ppo_enable:
            return 0.0
        return max(0.0, float(decayed))

    def _effective_policy_temperature(self) -> float:
        raw = max(1e-3, float(self.config.temperature))
        capped = min(raw, max(1e-3, float(self.policy_temperature_cap)))
        floor = min(capped, max(1e-3, float(self.policy_temperature_floor)))
        progress = self._exploration_progress()
        decayed = floor + ((capped - floor) * (1.0 - progress))
        # When PPO is active with strict on-policy sampling, always use
        # temperature=1.0 so the log-probs stored at collection time match
        # the temperature used during PPO optimization.  This mirrors the
        # epsilon=0.0 override in _effective_policy_epsilon().
        if self.strict_on_policy_sampling and self.ppo_enable:
            return 1.0
        return max(1e-3, float(decayed))

    def _build_prompt_signature(self, player_state: Dict[str, Any]) -> str:
        waiting_for = player_state.get('waitingFor', {}) if isinstance(player_state, dict) else {}
        if not isinstance(waiting_for, dict):
            return ''

        cards = waiting_for.get('cards', []) or []
        card_signature = []
        for card in cards[:16]:
            if not isinstance(card, dict):
                continue
            card_signature.append(
                {
                    "name": str(card.get("name", "") or ""),
                    "cost": int(card.get("calculatedCost", card.get("cost", 0)) or 0),
                    "reserveUnits": card.get("reserveUnits", {}) if isinstance(card.get("reserveUnits", {}), dict) else {},
                }
            )

        options = waiting_for.get('options', []) or []
        option_signature = []
        for option in options[:16]:
            if not isinstance(option, dict):
                continue
            title = option.get("title", "")
            if isinstance(title, dict):
                title = title.get("message", "")
            option_signature.append(
                {
                    "type": str(option.get("type", "") or ""),
                    "title": str(title or ""),
                    "buttonLabel": str(option.get("buttonLabel", "") or ""),
                }
            )

        player = player_state.get('thisPlayer', {}) if isinstance(player_state, dict) else {}
        player_budget = {}
        if isinstance(player, dict):
            for key in (
                "megaCredits",
                "steel",
                "titanium",
                "heat",
                "plants",
                "microbes",
                "floaters",
                "lunaArchivesScience",
                "spireScience",
                "seeds",
                "auroraiData",
                "graphene",
                "kuiperAsteroids",
            ):
                player_budget[key] = int(player.get(key, 0) or 0)

        title = waiting_for.get("title", "")
        if isinstance(title, dict):
            title = title.get("message", "")

        signature_payload = {
            "type": str(waiting_for.get("type", "") or ""),
            "title": str(title or ""),
            "buttonLabel": str(waiting_for.get("buttonLabel", "") or ""),
            "amount": int(waiting_for.get("amount", 0) or 0),
            "min": int(waiting_for.get("min", 0) or 0),
            "max": int(waiting_for.get("max", 0) or 0),
            "canPass": bool(waiting_for.get("canPass", False)),
            "paymentOptions": waiting_for.get("paymentOptions", {}) if isinstance(waiting_for.get("paymentOptions", {}), dict) else {},
            "reserveUnits": waiting_for.get("reserveUnits", {}) if isinstance(waiting_for.get("reserveUnits", {}), dict) else {},
            "cards": card_signature,
            "options": option_signature,
            "playerBudget": player_budget,
        }
        try:
            serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            serialized = str(signature_payload)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return str(digest)

    def _prompt_cache_key(self, player_id: str, player_state: Dict[str, Any]) -> str:
        signature = self._build_prompt_signature(player_state)
        if not signature:
            return ''
        return f"{str(player_id)}:{signature}"

    def _get_rejected_actions_for_prompt(self, player_id: str, player_state: Dict[str, Any]) -> set[int]:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return set()
        return set(int(a) for a in self._rejected_actions_by_prompt.get(cache_key, set()))

    def _remember_rejected_action(self, player_id: str, player_state: Dict[str, Any], action_index: int) -> None:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return
        if cache_key not in self._rejected_actions_by_prompt:
            self._rejected_actions_by_prompt[cache_key] = set()
            self._rejected_action_prompt_order.append(cache_key)
        self._rejected_actions_by_prompt[cache_key].add(int(action_index))
        self._prune_rejected_action_cache()

    def _clear_rejected_action(self, player_id: str, player_state: Dict[str, Any], action_index: int) -> None:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return
        blocked = self._rejected_actions_by_prompt.get(cache_key)
        if not blocked:
            return
        blocked.discard(int(action_index))
        if not blocked:
            self._rejected_actions_by_prompt.pop(cache_key, None)

    def _prune_rejected_action_cache(self) -> None:
        max_entries = max(64, int(self.rejected_action_memory_size))
        while len(self._rejected_actions_by_prompt) > max_entries:
            if not self._rejected_action_prompt_order:
                self._rejected_actions_by_prompt.clear()
                return
            oldest_key = self._rejected_action_prompt_order.popleft()
            self._rejected_actions_by_prompt.pop(oldest_key, None)
        while len(self._rejected_action_prompt_order) > (max_entries * 2):
            self._rejected_action_prompt_order.popleft()

    def _get_fallback_retry_count_for_prompt(self, player_id: str, player_state: Dict[str, Any]) -> int:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return 0
        return int(self._fallback_random_retries_by_prompt.get(cache_key, 0) or 0)

    def _bump_fallback_retry_count_for_prompt(
        self,
        player_id: str,
        player_state: Dict[str, Any],
        amount: int = 1,
    ) -> int:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return 0
        if cache_key not in self._fallback_random_retries_by_prompt:
            self._fallback_random_retries_by_prompt[cache_key] = 0
            self._fallback_retry_prompt_order.append(cache_key)
        next_value = int(self._fallback_random_retries_by_prompt.get(cache_key, 0) or 0) + max(0, int(amount))
        self._fallback_random_retries_by_prompt[cache_key] = int(next_value)
        self._prune_fallback_retry_cache()
        return int(next_value)

    def _clear_fallback_retry_count_for_prompt(self, player_id: str, player_state: Dict[str, Any]) -> None:
        cache_key = self._prompt_cache_key(player_id, player_state)
        if not cache_key:
            return
        self._fallback_random_retries_by_prompt.pop(cache_key, None)

    def _prune_fallback_retry_cache(self) -> None:
        max_entries = max(64, int(self.rejected_action_memory_size))
        while len(self._fallback_random_retries_by_prompt) > max_entries:
            if not self._fallback_retry_prompt_order:
                self._fallback_random_retries_by_prompt.clear()
                return
            oldest_key = self._fallback_retry_prompt_order.popleft()
            self._fallback_random_retries_by_prompt.pop(oldest_key, None)
        while len(self._fallback_retry_prompt_order) > (max_entries * 2):
            self._fallback_retry_prompt_order.popleft()

    def _snapshot_hate_draft_counters(self) -> Dict[str, int]:
        return {
            "draft_decisions_total": int(self.decision_stats.get("draft_decisions_total", 0) or 0),
            "draft_decisions_low_hand_ev": int(self.decision_stats.get("draft_decisions_low_hand_ev", 0) or 0),
            "hate_draft_picks": int(self.decision_stats.get("hate_draft_picks", 0) or 0),
            "hate_draft_picks_low_hand_ev": int(self.decision_stats.get("hate_draft_picks_low_hand_ev", 0) or 0),
        }

    def _build_play_game_telemetry(
        self,
        game_outcome: Dict[str, Any],
        counter_snapshot_before: Dict[str, int],
    ) -> Dict[str, Any]:
        counters_after = self._snapshot_hate_draft_counters()
        draft_decisions = max(
            0,
            int(counters_after.get("draft_decisions_total", 0)) - int(counter_snapshot_before.get("draft_decisions_total", 0)),
        )
        draft_decisions_low_hand_ev = max(
            0,
            int(counters_after.get("draft_decisions_low_hand_ev", 0)) - int(counter_snapshot_before.get("draft_decisions_low_hand_ev", 0)),
        )
        hate_draft_picks = max(
            0,
            int(counters_after.get("hate_draft_picks", 0)) - int(counter_snapshot_before.get("hate_draft_picks", 0)),
        )
        hate_draft_picks_low_hand_ev = max(
            0,
            int(counters_after.get("hate_draft_picks_low_hand_ev", 0)) - int(counter_snapshot_before.get("hate_draft_picks_low_hand_ev", 0)),
        )
        hate_draft_rate = (float(hate_draft_picks) / float(draft_decisions)) if draft_decisions > 0 else 0.0
        hate_draft_rate_low_hand_ev = (
            float(hate_draft_picks_low_hand_ev) / float(draft_decisions_low_hand_ev)
            if draft_decisions_low_hand_ev > 0
            else 0.0
        )
        return {
            "agent_id": self.id,
            "completed": bool(game_outcome.get("completed", False)),
            "rank": int(game_outcome.get("rank", 4) or 4),
            "vp": int(game_outcome.get("vp", 0) or 0),
            "draft_decisions": int(draft_decisions),
            "draft_decisions_low_hand_ev": int(draft_decisions_low_hand_ev),
            "hate_draft_picks": int(hate_draft_picks),
            "hate_draft_picks_low_hand_ev": int(hate_draft_picks_low_hand_ev),
            "hate_draft_rate": float(hate_draft_rate),
            "hate_draft_rate_low_hand_ev": float(hate_draft_rate_low_hand_ev),
        }

    async def play_game(self, game_instance: GameInstance, player_name: str) -> Dict[str, Any]:
        """Play a complete game"""
        episode_steps: deque = deque(maxlen=self.config.max_episode_steps)
        counter_snapshot_before = self._snapshot_hate_draft_counters()
        game_outcome: Dict[str, Any] = {"completed": False, "rank": 4, "vp": 0}
        max_transport_retries = max(1, self._safe_env_int("AGENT_TRANSPORT_RETRY_LIMIT", 6))
        transport_retry_backoff_sec = max(0.5, self._safe_env_float("AGENT_TRANSPORT_RETRY_BACKOFF_SEC", 3.0))
        try:
            # Join the game
            player_id = await game_instance.join_player(player_name)
            self._set_recurrent_state_for_player(player_id, self._zero_recurrent_state())
            logger.info(f"Agent {self.id[:8]} joined game as {player_name} (ID: {player_id})")
            logger.info(
                "Agent %s debug links: game=%s player_api(public)=%s player_api(internal)=%s",
                self.id[:8],
                game_instance.get_public_game_url(),
                game_instance.get_public_player_api_url(player_id),
                game_instance.get_internal_player_api_url(player_id),
            )
            await self._maybe_apply_startup_seat_stagger(player_name)

            # Optional deterministic startup autosubmit. Keep disabled by default
            # so startup choices can be learned by policy from STARTUP_PLAN actions.
            if self.startup_autosubmit:
                await self._run_initial_setup(game_instance, player_id)
             
            # Game loop with transient-error resilience: the game still exists on
            # the server after a disconnect, so we back off and retry instead of
            # immediately killing the episode.
            consecutive_transport_errors = 0
            while True:
                try:
                    player_state = await self._timed_get_player_state(game_instance, player_id)
                    consecutive_transport_errors = 0
                except ServerTransportError:
                    consecutive_transport_errors += 1
                    if consecutive_transport_errors > max_transport_retries:
                        raise
                    backoff = min(
                        transport_retry_backoff_sec * float(consecutive_transport_errors),
                        15.0,
                    )
                    logger.warning(
                        "Agent %s transient transport error polling state (%d/%d). "
                        "Backing off %.1fs before retry.",
                        self.id[:8],
                        consecutive_transport_errors,
                        max_transport_retries,
                        backoff,
                    )
                    await self._sleep_if_needed(backoff)
                    continue
                
                # Check if game is over
                if player_state.get('game', {}).get('phase') == 'end':
                    break
                
                # If we are waiting for input, make a move.
                # Otherwise, we wait for our turn.
                if player_state.get('waitingFor'):
                    try:
                        submitted_action = await self._make_move(game_instance, player_id, player_state, episode_steps)
                    except ServerTransportError:
                        consecutive_transport_errors += 1
                        if consecutive_transport_errors > max_transport_retries:
                            raise
                        backoff = min(
                            transport_retry_backoff_sec * float(consecutive_transport_errors),
                            15.0,
                        )
                        logger.warning(
                            "Agent %s transient transport error during move (%d/%d). "
                            "Backing off %.1fs before retry.",
                            self.id[:8],
                            consecutive_transport_errors,
                            max_transport_retries,
                            backoff,
                        )
                        await self._sleep_if_needed(backoff)
                        continue
                    consecutive_transport_errors = 0
                    # After a successful action, repoll immediately so chained prompts
                    # don't pay an extra AGENT_POLL_INTERVAL_SEC delay.
                    if submitted_action:
                        continue
                
                # Wait before polling again to avoid busy-waiting
                await self._sleep_if_needed(self._poll_interval_for_state(player_state))
            
            # Record game completion
            self.games_played += 1
            final_state = await game_instance.get_final_state()
            game_outcome = await self._record_game_result(final_state, player_name, game_instance)

            # Queue PPO trajectory for coordinator-driven optimization.
            if self.train_from_self_play and game_outcome.get("completed", False):
                reward = self._compute_terminal_reward(
                    game_outcome.get("rank", 4),
                    game_outcome.get("vp", 0),
                    game_outcome.get("completed", False),
                )
                reward *= self.self_play_reward_scale
                if self.ppo_enable:
                    await self._queue_episode_rollout(episode_steps, reward)
                else:
                    await self._train_from_episode(episode_steps, reward)
            return self._build_play_game_telemetry(game_outcome, counter_snapshot_before)
            
        except asyncio.CancelledError:
            reason = "unknown"
            game_id = ""
            task_name = ""
            try:
                current_task = asyncio.current_task()
                if current_task is not None:
                    reason = str(getattr(current_task, "_cancel_reason", "") or "unknown")
                    game_id = str(getattr(current_task, "_cancel_game_id", "") or "")
                    task_name = str(current_task.get_name() or "")
            except Exception:
                pass
            if reason == "game_timeout":
                logger.warning(
                    "Agent %s play_game task cancelled due to game timeout (game_id=%s task=%s)",
                    self.id[:8],
                    game_id or "unknown",
                    task_name or "unnamed",
                )
            else:
                logger.warning(
                    "Agent %s play_game task cancelled (reason=%s game_id=%s task=%s)",
                    self.id[:8],
                    reason,
                    game_id or "unknown",
                    task_name or "unnamed",
                )
            raise
        except Exception as e:
            logger.error(f"Agent {self.id[:8]} failed during game: {e}")
            raise
        finally:
            try:
                self._clear_recurrent_state_for_player(locals().get("player_id"))
            except Exception:
                pass

    async def _sleep_if_needed(self, seconds: float):
        delay = max(0.0, float(seconds or 0.0))
        if delay > 0.0:
            await asyncio.sleep(delay)

    def _poll_interval_for_state(self, player_state: Optional[Dict[str, Any]]) -> float:
        waiting_for = {}
        if isinstance(player_state, dict):
            waiting_for = player_state.get("waitingFor", {}) or {}
        return float(self.active_poll_interval_sec) if bool(waiting_for) else float(self.idle_poll_interval_sec)

    async def _timed_get_player_state(self, game_instance: GameInstance, player_id: str) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            return await game_instance.get_player_state(player_id)
        finally:
            self._record_pipeline_timing("get_player_state_sec", time.perf_counter() - started)

    async def _timed_send_player_input(
        self,
        game_instance: GameInstance,
        player_id: str,
        action_input: Dict[str, Any],
    ) -> bool:
        started = time.perf_counter()
        try:
            return bool(await game_instance.send_player_input(player_id, action_input))
        finally:
            self._record_pipeline_timing("send_player_input_sec", time.perf_counter() - started)

    @staticmethod
    def _seat_index_from_player_name(player_name: str) -> int:
        raw = str(player_name or "").strip()
        match = re.match(r"^A(\d+)_", raw)
        if not match:
            return 0
        try:
            return max(0, int(match.group(1)) - 1)
        except Exception:
            return 0

    async def _maybe_apply_startup_seat_stagger(self, player_name: str):
        base_ms = max(0, self._safe_env_int("AGENT_STARTUP_SEAT_STAGGER_MS", 0))
        jitter_ms = max(0, self._safe_env_int("AGENT_STARTUP_SEAT_STAGGER_JITTER_MS", 0))
        if base_ms <= 0 and jitter_ms <= 0:
            return
        seat_index = self._seat_index_from_player_name(player_name)
        stagger_ms = float(seat_index * base_ms)
        if jitter_ms > 0:
            stagger_ms += float(random.randint(0, int(jitter_ms)))
        if stagger_ms <= 0.0:
            return
        logger.debug(
            "Applying startup seat stagger for %s: seat=%d delay_ms=%.1f",
            player_name,
            seat_index + 1,
            stagger_ms,
        )
        await asyncio.sleep(stagger_ms / 1000.0)

    async def _run_initial_setup(self, game_instance: GameInstance, player_id: str, max_attempts: int = 12) -> bool:
        """
        Attempt to submit initial setup choices once near game start.
        Returns True only when startup selections were successfully submitted.
        """
        attempts = max(1, int(max_attempts))
        for _ in range(attempts):
            try:
                player_state = await self._timed_get_player_state(game_instance, player_id)
            except Exception as e:
                if isinstance(e, ServerTransportError):
                    raise
                await self._sleep_if_needed(self.poll_interval_sec)
                continue

            waiting_for = player_state.get('waitingFor', {}) if isinstance(player_state, dict) else {}
            waiting_type = str(waiting_for.get('type', ''))
            if waiting_type in ['initialCards', 'selectInitialCards']:
                setup_action = self.action_decoder.build_initial_setup_response(player_state)
                if not setup_action:
                    return False
                logger.info(f"Agent {self.id[:8]} startup setup action: {setup_action}")
                if await self._timed_send_player_input(game_instance, player_id, setup_action):
                    self._record_action_choice(800, setup_action, player_state)
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return True
                await self._sleep_if_needed(max(self.failure_pause_sec, self.poll_interval_sec))
                continue

            # If startup input is not currently visible, let normal loop handle the rest.
            if waiting_for:
                return False

            await self._sleep_if_needed(self.poll_interval_sec)

        return False

    def _log_stuck_context(self, game_instance: GameInstance, player_id: str, player_state: Dict[str, Any], reason: str):
        now = time.monotonic()
        last = self._last_stuck_log_by_player.get(player_id, 0.0)
        if (now - last) < max(0.0, self.stuck_log_cooldown_sec):
            return
        self._last_stuck_log_by_player[player_id] = now

        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        waiting_type = waiting_for.get('type', 'unknown')
        try:
            waiting_preview = json.dumps(waiting_for, ensure_ascii=True, separators=(',', ':'))
        except Exception:
            waiting_preview = str(waiting_for)
        if len(waiting_preview) > 1200:
            waiting_preview = waiting_preview[:1200] + "...(truncated)"

        logger.warning(
            "Agent %s stuck (%s). waitingFor.type=%s, GameURL=%s, PlayerAPI(public)=%s, PlayerAPI(internal)=%s, waitingFor=%s",
            self.id[:8],
            reason,
            waiting_type,
            game_instance.get_public_game_url(),
            game_instance.get_public_player_api_url(player_id),
            game_instance.get_internal_player_api_url(player_id),
            waiting_preview,
        )

    async def _make_move(
        self,
        game_instance: GameInstance,
        player_id: str,
        player_state: Dict[str, Any],
        episode_steps: List[Dict[str, Any]],
    ) -> bool:
        """Make a single move in the game, with robust fallbacks."""
        try:
            # Update turn-action counter (resets automatically on phase change)
            self._maybe_reset_turn_action_count(player_id, player_state)
            turn_action_count = self._get_turn_action_count(player_id)

            availability_started = time.perf_counter()
            raw_action_descriptors = self.action_decoder.get_legal_action_descriptors(player_state)
            raw_available_actions = [int(item.get("action_index", -1)) for item in raw_action_descriptors]
            filtered_available_actions = self._filter_pass_actions(raw_available_actions, player_state)
            filtered_action_set = set(int(item) for item in filtered_available_actions)
            filtered_action_descriptors = [
                item for item in raw_action_descriptors
                if int(item.get("action_index", -1)) in filtered_action_set
            ]
            if player_id:
                rejected_actions = self._get_rejected_actions_for_prompt(player_id, player_state)
                if rejected_actions:
                    before = len(filtered_action_descriptors)
                    filtered_action_descriptors = [
                        item for item in filtered_action_descriptors
                        if int(item.get("action_index", -1)) not in rejected_actions
                    ]
                    blocked_count = max(0, before - len(filtered_action_descriptors))
                    if blocked_count > 0:
                        self._bump_decision_stat('policy_actions_blocked_by_reject_cache', blocked_count)
            if not filtered_action_descriptors:
                filtered_action_descriptors = list(raw_action_descriptors)
            if not filtered_action_descriptors:
                logger.warning("Agent %s found no legal action descriptors for player %s", self.id[:8], player_id)
                return False
            self._record_pipeline_timing("action_availability_sec", time.perf_counter() - availability_started)

            loop = asyncio.get_running_loop()
            encode_started = time.perf_counter()
            try:
                planner_state = await loop.run_in_executor(
                    _get_inference_executor(),
                    self.state_encoder.encode,
                    player_state,
                    turn_action_count,
                    filtered_action_descriptors,
                )
            finally:
                self._record_pipeline_timing("encode_state_sec", time.perf_counter() - encode_started)
            self._bump_decision_stat('total_decisions')
            
            # Log what we're waiting for
            waiting_for = player_state.get('waitingFor', {})
            waiting_type = waiting_for.get('type', 'unknown')
            is_initial_cards_prompt = waiting_type in ['initialCards', 'selectInitialCards']
            logger.debug(f"Agent {self.id[:8]} making move for input type: {waiting_type}")

            if self.startup_autosubmit and waiting_type in ['initialCards', 'selectInitialCards']:
                initial_action = self.action_decoder.build_initial_setup_response(player_state)
                if initial_action:
                    self._bump_decision_stat('policy_attempts')
                    self._bump_decision_stat('policy_sampled_actions')
                    logger.info(f"Agent {self.id[:8]} attempting startup setup action: {initial_action}")
                    if await self._timed_send_player_input(game_instance, player_id, initial_action):
                        self._bump_decision_stat('policy_successes')
                        self._record_action_choice(800, initial_action, player_state)
                        logger.info(f"Agent {self.id[:8]} startup setup action succeeded")
                        await self._sleep_if_needed(self.post_move_sleep_sec)
                        return True
                    self._bump_decision_stat('policy_rejections')
                    self._log_stuck_context(game_instance, player_id, player_state, "startup_setup_rejected")
                    await self._sleep_if_needed(self.failure_pause_sec)

            # 1. Try a policy-driven action
            policy_action, policy_action_idx, sampled_from_policy, action_meta = await self._get_action_from_network(
                planner_state,
                player_state,
                filtered_action_descriptors,
                raw_available_actions,
                player_id=player_id,
                force_random=False,
            )
            tried_action_indices = set()
            if policy_action:
                self._bump_decision_stat('policy_attempts')
                if sampled_from_policy:
                    self._bump_decision_stat('policy_sampled_actions')
                else:
                    self._bump_decision_stat('epsilon_random_actions')
                logger.debug(f"Agent {self.id[:8]} attempting policy action: {policy_action}")
                if await self._timed_send_player_input(game_instance, player_id, policy_action):
                    self._bump_decision_stat('policy_successes')
                    # Track how many actions this player has taken this turn so the
                    # state encoder can expose first-vs-second-action information.
                    self._increment_turn_action_count(player_id)
                    if policy_action_idx is not None:
                        self._record_action_choice(int(policy_action_idx), policy_action, player_state)
                        self._clear_rejected_action(player_id, player_state, int(policy_action_idx))
                    self._clear_fallback_retry_count_for_prompt(player_id, player_state)
                    if action_meta is not None:
                        self._set_recurrent_state_for_player(
                            player_id,
                            action_meta.get("recurrent_state_out"),
                        )
                    logger.debug(f"Agent {self.id[:8]} policy action succeeded {policy_action}")
                    if sampled_from_policy and policy_action_idx is not None:
                        if action_meta is not None:
                            rare_award = float(action_meta.get("rare_award_funding", 0.0))
                            rare_milestone = float(action_meta.get("rare_milestone_timing", 0.0))
                            rare_draft = float(action_meta.get("rare_draft_keep_buy", 0.0))
                            rare_high_cost = float(action_meta.get("rare_high_cost_payment", 0.0))
                            if (rare_award + rare_milestone + rare_draft + rare_high_cost) > 0.0:
                                self._bump_decision_stat('rare_state_samples')
                            if rare_award > 0.0:
                                self._bump_decision_stat('rare_award_funding')
                            if rare_milestone > 0.0:
                                self._bump_decision_stat('rare_milestone_timing')
                            if rare_draft > 0.0:
                                self._bump_decision_stat('rare_draft_keep_buy')
                            if rare_high_cost > 0.0:
                                self._bump_decision_stat('rare_high_cost_payment')
                        # Always record steps; the deque(maxlen=max_episode_steps)
                        # automatically evicts the oldest entries so we keep the
                        # last N steps (including the true terminal step).
                        step_reward = 0.0
                        reward_tr_component = 0.0
                        reward_cards_vp_component = 0.0
                        reward_city_greenery_component = 0.0
                        reward_city_future_component = 0.0
                        reward_milestones_awards_component = 0.0
                        reward_other_component = 0.0
                        reward_shaping_coef = self._current_reward_shaping_coef()
                        if self.train_from_self_play:
                            post_action_state = game_instance.peek_cached_state(player_id)
                            reward_breakdown = calculate_step_reward_decomposition(
                                before_state=player_state,
                                after_state=post_action_state,
                                action_input=policy_action,
                            )
                            step_reward_scale = float(reward_breakdown.get("step_reward_scale", 1.0))
                            weighted_tr = float(self.reward_tr_weight) * float(reward_breakdown.get("tr_component", 0.0))
                            weighted_cards_vp = float(self.reward_cards_vp_weight) * float(reward_breakdown.get("cards_vp_component", 0.0))
                            weighted_city_greenery = float(self.reward_city_greenery_weight) * float(reward_breakdown.get("city_greenery_component", 0.0))
                            weighted_city_future = float(self.reward_city_greenery_weight) * float(reward_breakdown.get("city_future_component", 0.0))
                            weighted_milestones_awards = float(self.reward_milestones_awards_weight) * float(reward_breakdown.get("milestones_awards_component", 0.0))
                            weighted_other = float(self.reward_other_weight) * float(reward_breakdown.get("other_component", 0.0))
                            weighted_raw = (
                                weighted_tr
                                + weighted_cards_vp
                                + weighted_city_greenery
                                + weighted_city_future
                                + weighted_milestones_awards
                                + weighted_other
                            )
                            weighted_scaled = max(-0.35, min(0.35, weighted_raw)) * step_reward_scale
                            step_reward = float(reward_shaping_coef * weighted_scaled)
                            reward_tr_component = float(weighted_tr * step_reward_scale)
                            reward_cards_vp_component = float(weighted_cards_vp * step_reward_scale)
                            reward_city_greenery_component = float(weighted_city_greenery * step_reward_scale)
                            reward_city_future_component = float(weighted_city_future * step_reward_scale)
                            reward_milestones_awards_component = float(weighted_milestones_awards * step_reward_scale)
                            reward_other_component = float(weighted_other * step_reward_scale)
                            if self.reward_debug_enabled:
                                self._reward_debug_counter += 1
                                shaped_l1 = (
                                    abs(reward_tr_component)
                                    + abs(reward_cards_vp_component)
                                    + abs(reward_city_greenery_component)
                                    + abs(reward_city_future_component)
                                    + abs(reward_milestones_awards_component)
                                )
                                if (
                                    self.games_played > 10
                                    and shaped_l1 < float(self.reward_debug_threshold)
                                    and (self._reward_debug_counter % int(self.reward_debug_log_every)) == 0
                                ):
                                    logger.warning(
                                        "Low VP shaping components: tr=%.5f cards=%.5f city_greenery=%.5f city_future=%.5f milestones_awards=%.5f other=%.5f coef=%.3f scaled=%.5f raw=%.5f",
                                        reward_tr_component,
                                        reward_cards_vp_component,
                                        reward_city_greenery_component,
                                        reward_city_future_component,
                                        reward_milestones_awards_component,
                                        reward_other_component,
                                        reward_shaping_coef,
                                        float(reward_breakdown.get("scaled_total", 0.0)),
                                        float(reward_breakdown.get("raw_total", 0.0)),
                                    )
                            if float(reward_breakdown.get("hate_draft_decision", 0.0)) > 0.0:
                                self._bump_decision_stat("draft_decisions_total")
                            if float(reward_breakdown.get("hate_draft_low_hand_ev", 0.0)) > 0.0:
                                self._bump_decision_stat("draft_decisions_low_hand_ev")
                            if reward_breakdown.get("hate_draft_bonus_applied"):
                                self._bump_decision_stat("hate_draft_picks")
                                if float(reward_breakdown.get("hate_draft_low_hand_ev", 0.0)) > 0.0:
                                    self._bump_decision_stat("hate_draft_picks_low_hand_ev")
                            if reward_breakdown.get("sniping_milestone_applied"):
                                self._bump_decision_stat("milestone_snipes")
                            if reward_breakdown.get("sniping_award_applied"):
                                self._bump_decision_stat("award_snipes")
                        if action_meta is not None:
                            episode_steps.append(
                                {
                                    "state_bundle": planner_state,
                                    "action_position": int(action_meta.get("chosen_action_position", 0)),
                                    "action_index": int(policy_action_idx),
                                    "reward": float(step_reward),
                                    "logp_old": float(action_meta.get("logp_old", 0.0)),
                                    "value_old": float(action_meta.get("value_old", 0.0)),
                                    "legal_actions": list(action_meta.get("legal_actions", [])),
                                    "phase_index": int(action_meta.get("phase_index", 0)),
                                    "recurrent_state": list(action_meta.get("recurrent_state", [])),
                                    "aux_targets": dict(action_meta.get("aux_targets", {})),
                                    "aux_predictions": list(action_meta.get("aux_predictions", [])),
                                    "rare_state_weight": float(action_meta.get("rare_state_weight", 1.0)),
                                    "rare_award_funding": float(action_meta.get("rare_award_funding", 0.0)),
                                    "rare_milestone_timing": float(action_meta.get("rare_milestone_timing", 0.0)),
                                    "rare_draft_keep_buy": float(action_meta.get("rare_draft_keep_buy", 0.0)),
                                    "rare_high_cost_payment": float(action_meta.get("rare_high_cost_payment", 0.0)),
                                    "reward_tr_component": float(reward_tr_component),
                                    "reward_cards_vp_component": float(reward_cards_vp_component),
                                    "reward_city_greenery_component": float(reward_city_greenery_component),
                                    "reward_city_future_component": float(reward_city_future_component),
                                    "reward_milestones_awards_component": float(reward_milestones_awards_component),
                                    "reward_other_component": float(reward_other_component),
                                    "reward_shaping_coef": float(reward_shaping_coef),
                                }
                            )
                    self._maybe_capture_decision_snapshot(
                        game_instance=game_instance,
                        player_id=player_id,
                        player_state=player_state,
                        action_input=policy_action,
                        action_index=policy_action_idx,
                        action_meta=action_meta,
                        sampled_from_policy=sampled_from_policy,
                        send_outcome="accepted",
                        turn_action_count=turn_action_count,
                        state_vector=None,
                    )
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return True
                else:
                    self._bump_decision_stat('policy_rejections')
                    self._bump_decision_stat('action_rejected_by_server')
                    if policy_action_idx is not None:
                        action_idx = int(policy_action_idx)
                        tried_action_indices.add(action_idx)
                        self._remember_rejected_action(player_id, player_state, action_idx)
                    logger.warning(f"Agent {self.id[:8]} policy action was rejected by game")
                    self._log_stuck_context(game_instance, player_id, player_state, "policy_action_rejected")
                    self._maybe_capture_decision_snapshot(
                        game_instance=game_instance,
                        player_id=player_id,
                        player_state=player_state,
                        action_input=policy_action,
                        action_index=policy_action_idx,
                        action_meta=action_meta,
                        sampled_from_policy=sampled_from_policy,
                        send_outcome="rejected",
                        turn_action_count=turn_action_count,
                        state_vector=None,
                    )
                    pause_after_reject = self.failure_pause_sec
                    if is_initial_cards_prompt:
                        pause_after_reject = max(pause_after_reject, self.initial_cards_reject_pause_sec)
                    await self._sleep_if_needed(pause_after_reject)

            logger.warning(f"Policy action failed for agent {self.id[:8]}. Trying random actions.")
            self._bump_decision_stat('fallback_decisions')

            # 2. Try a broader set of alternative actions, excluding already-rejected choices.
            pass_base = int(self.action_decoder.action_types.get('PASS', 900))
            raw_available_actions: List[int] = []
            if isinstance(action_meta, dict):
                cached_raw_actions = action_meta.get("available_actions_raw", [])
                if isinstance(cached_raw_actions, list):
                    raw_available_actions = [
                        int(action_idx)
                        for action_idx in cached_raw_actions
                        if isinstance(action_idx, (int, np.integer))
                        or (isinstance(action_idx, str) and str(action_idx).isdigit())
                    ]
            if not raw_available_actions:
                raw_available_actions = self.action_decoder.get_available_actions(player_state)
            can_legally_pass = any(int(a) >= pass_base for a in raw_available_actions)

            available_actions = self._filter_pass_actions(raw_available_actions, player_state)
            available_actions = [a for a in available_actions if int(a) not in tried_action_indices]
            prompt_rejected_actions = self._get_rejected_actions_for_prompt(player_id, player_state)
            if prompt_rejected_actions:
                candidate_actions = [a for a in available_actions if int(a) not in prompt_rejected_actions]
                blocked_count = max(0, int(len(available_actions) - len(candidate_actions)))
                if blocked_count > 0:
                    self._bump_decision_stat('policy_actions_blocked_by_reject_cache', blocked_count)
                available_actions = candidate_actions
            if not available_actions and not can_legally_pass:
                # In mandatory selection flows we cannot pass; retry non-pass actions even if tried.
                available_actions = [
                    a
                    for a in self._filter_pass_actions(raw_available_actions, player_state)
                    if int(a) < pass_base
                    and int(a) not in tried_action_indices
                    and int(a) not in prompt_rejected_actions
                ]

            if not available_actions:
                # If no actions are available, pass only when pass is legal.
                self._bump_decision_stat('no_available_actions')
                if can_legally_pass:
                    self._bump_decision_stat('fallback_passes')
                    self._record_action_choice(pass_base)
                    self._log_stuck_context(game_instance, player_id, player_state, "no_available_actions_pass")
                    sent_pass = await self._timed_send_player_input(
                        game_instance,
                        player_id,
                        self.action_decoder._create_pass_action(),
                    )
                    if sent_pass:
                        self._clear_fallback_retry_count_for_prompt(player_id, player_state)
                    await self._sleep_if_needed(self.post_move_sleep_sec)
                    return bool(sent_pass)
                else:
                    self._log_stuck_context(game_instance, player_id, player_state, "no_available_actions_no_pass")
                    await self._sleep_if_needed(self.failure_pause_sec or self.poll_interval_sec)
                    return False
                 
            random.shuffle(available_actions)
            prompt_retry_budget = int(self.max_fallback_random_retries_per_prompt)
            prompt_retry_used = self._get_fallback_retry_count_for_prompt(player_id, player_state)
            remaining_prompt_budget = max(0, int(prompt_retry_budget - prompt_retry_used))
            max_attempts = min(len(available_actions), int(self.max_fallback_attempts), int(remaining_prompt_budget))
            if is_initial_cards_prompt:
                max_attempts = min(max_attempts, int(self.initial_cards_fallback_max_attempts))

            if max_attempts <= 0 and remaining_prompt_budget <= 0:
                logger.warning(
                    "Fallback retry budget exhausted for agent %s on current prompt (budget=%d).",
                    self.id[:8],
                    prompt_retry_budget,
                )

            for i in range(max_attempts):
                random_action_idx = available_actions[i]
                random_action = self.action_decoder.decode_action(random_action_idx, player_state)
                 
                if random_action:
                    self._bump_decision_stat('fallback_random_attempts')
                    self._bump_fallback_retry_count_for_prompt(player_id, player_state, 1)
                    if await self._timed_send_player_input(game_instance, player_id, random_action):
                        self._bump_decision_stat('fallback_random_successes')
                        self._record_action_choice(int(random_action_idx), random_action, player_state)
                        self._clear_rejected_action(player_id, player_state, int(random_action_idx))
                        self._clear_fallback_retry_count_for_prompt(player_id, player_state)
                        logger.info(f"Random action succeeded for agent {self.id[:8]}.")
                        await self._sleep_if_needed(self.post_move_sleep_sec)
                        return True
                    self._remember_rejected_action(player_id, player_state, int(random_action_idx))
                    prompt_rejected_actions.add(int(random_action_idx))

            if can_legally_pass:
                logger.warning(f"All random actions failed for agent {self.id[:8]}. Passing.")
                self._bump_decision_stat('fallback_passes')
                self._record_action_choice(pass_base)
                self._log_stuck_context(game_instance, player_id, player_state, "all_random_actions_failed_pass")
                sent_pass = await self._timed_send_player_input(
                    game_instance,
                    player_id,
                    self.action_decoder._create_pass_action(),
                )
                if sent_pass:
                    self._clear_fallback_retry_count_for_prompt(player_id, player_state)
                await self._sleep_if_needed(self.post_move_sleep_sec)
                return bool(sent_pass)

            # Mandatory prompt and no legal pass: do not resend blacklisted actions.
            self._log_stuck_context(game_instance, player_id, player_state, "all_random_actions_failed_no_pass")
            await self._sleep_if_needed(self.failure_pause_sec or self.poll_interval_sec)
            return False

        except Exception as e:
            if isinstance(e, ServerTransportError):
                raise
            logger.error(f"Error making move for agent {self.id[:8]}: {e}", exc_info=True)
            return False

    def _filter_pass_actions(self, available_actions: List[int], player_state: Dict[str, Any]) -> List[int]:
        if not available_actions:
            return available_actions
        pass_base = self.action_decoder.action_types.get('PASS', 900)
        non_pass_actions = [a for a in available_actions if a < pass_base]
        if not non_pass_actions:
            return available_actions

        waiting_for = player_state.get('waitingFor', {}) if player_state else {}
        waiting_type = str(waiting_for.get('type', ''))
        sell_option_actions = set()

        def _is_sell_patents_action(action_idx: int) -> bool:
            return int(action_idx) == 702 or int(action_idx) in sell_option_actions

        # Keep pass available during explicit selection flows (draft/research/buy/keep).
        if waiting_type in ['card', 'selectCard', 'projectCard', 'selectProjectCardToPlay', 'initialCards']:
            title = waiting_for.get('title', '')
            if isinstance(title, dict):
                title = title.get('message', '')
            title_l = str(title).lower()
            button_label = str(waiting_for.get('buttonLabel', '') or '').lower()
            if (
                waiting_for.get('showOnlyInLearnerMode', False)
                or waiting_for.get('selectBlueCardAction', False)
                or 'prelude' in title_l
                or 'research' in title_l
                or 'draft' in title_l
                or 'select' in title_l
                or button_label in ['keep', 'buy', 'select', 'choose', 'discard', 'confirm', 'ok', 'save']
                or (
                    'min' in waiting_for
                    and 'max' in waiting_for
                    and button_label in ['keep', 'buy', 'select', 'choose', 'confirm', 'ok', 'save', 'research']
                )
            ):
                return available_actions

        if waiting_for.get('type') == 'or':
            options = waiting_for.get('options', [])
            select_option_base = self.action_decoder.action_types.get('SELECT_OPTION', 200)
            filtered = list(non_pass_actions)
            pass_option_actions = set()
            for i, option in enumerate(options):
                title = option.get('title', '')
                if isinstance(title, dict):
                    title = title.get('message', '')
                title_l = str(title).lower()
                if 'pass' in title_l:
                    pass_action = select_option_base + i
                    if pass_action in filtered:
                        filtered.remove(pass_action)
                    pass_option_actions.add(pass_action)
                if 'sell patents' in title_l:
                    sell_option_actions.add(select_option_base + i)

            non_pass_non_pass_option = [a for a in non_pass_actions if int(a) not in pass_option_actions]
            productive_actions = [a for a in non_pass_non_pass_option if not _is_sell_patents_action(int(a))]

            # If the only alternative to pass is sell patents, keep pass.
            if not productive_actions:
                return available_actions
            # When at least one non-sell productive action exists, hide sell-patents
            # choices so they are only used as a last-resort liquidity tool.
            filtered_non_sell = [a for a in filtered if not _is_sell_patents_action(int(a))]
            return filtered_non_sell if filtered_non_sell else (filtered if filtered else available_actions)

        productive_non_sell = [a for a in non_pass_actions if not _is_sell_patents_action(int(a))]
        if productive_non_sell:
            return productive_non_sell

        # If the only non-pass action is sell patents, keep pass to avoid forced selling.
        if non_pass_actions and all(_is_sell_patents_action(int(a)) for a in non_pass_actions):
            return available_actions

        return non_pass_actions

    def _bump_decision_stat(self, key: str, amount: int = 1):
        self.decision_stats[key] = int(self.decision_stats.get(key, 0)) + int(amount)
        if key == "total_decisions":
            self._maybe_log_pipeline_timing()

    def _bump_decision_stat_float(self, key: str, amount: float = 0.0):
        current = float(self.decision_stats.get(key, 0.0) or 0.0)
        self.decision_stats[key] = float(current + float(amount))

    def _record_pipeline_timing(self, stage_key: str, elapsed_sec: float):
        if not stage_key:
            return
        elapsed = max(0.0, float(elapsed_sec or 0.0))
        totals = self.decision_stats.setdefault("timing_totals_sec", {})
        counts = self.decision_stats.setdefault("timing_counts", {})
        totals[stage_key] = float(totals.get(stage_key, 0.0) or 0.0) + elapsed
        counts[stage_key] = int(counts.get(stage_key, 0) or 0) + 1

    def _pipeline_timing_avg_ms(self, stage_key: str) -> float:
        totals = self.decision_stats.get("timing_totals_sec", {}) or {}
        counts = self.decision_stats.get("timing_counts", {}) or {}
        total = float(totals.get(stage_key, 0.0) or 0.0)
        count = int(counts.get(stage_key, 0) or 0)
        if count <= 0:
            return 0.0
        return (total * 1000.0) / float(count)

    def _maybe_log_pipeline_timing(self):
        total_decisions = int(self.decision_stats.get("total_decisions", 0) or 0)
        interval = max(1, int(self.timing_log_every_n_decisions))
        if total_decisions <= 0 or (total_decisions % interval) != 0:
            return
        counts = self.decision_stats.get("timing_counts", {}) or {}
        logger.info(
            "Agent %s timing over %d decisions: "
            "encode=%.2fms(%d) avail=%.2fms(%d) forward=%.2fms(%d) decode=%.2fms(%d) send=%.2fms(%d) get=%.2fms(%d)",
            self.id[:8],
            total_decisions,
            self._pipeline_timing_avg_ms("encode_state_sec"),
            int(counts.get("encode_state_sec", 0) or 0),
            self._pipeline_timing_avg_ms("action_availability_sec"),
            int(counts.get("action_availability_sec", 0) or 0),
            self._pipeline_timing_avg_ms("network_forward_sec"),
            int(counts.get("network_forward_sec", 0) or 0),
            self._pipeline_timing_avg_ms("decode_action_sec"),
            int(counts.get("decode_action_sec", 0) or 0),
            self._pipeline_timing_avg_ms("send_player_input_sec"),
            int(counts.get("send_player_input_sec", 0) or 0),
            self._pipeline_timing_avg_ms("get_player_state_sec"),
            int(counts.get("get_player_state_sec", 0) or 0),
        )
        self.decision_stats["timing_log_events"] = int(self.decision_stats.get("timing_log_events", 0) or 0) + 1

    @staticmethod
    def _iter_action_payloads(action_input: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(action_input, dict):
            return []
        payloads: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = [action_input]
        while stack:
            payload = stack.pop()
            if not isinstance(payload, dict):
                continue
            payloads.append(payload)
            nested_response = payload.get('response')
            if isinstance(nested_response, dict):
                stack.append(nested_response)
            nested_responses = payload.get('responses')
            if isinstance(nested_responses, list):
                for item in nested_responses:
                    if isinstance(item, dict):
                        stack.append(item)
        return payloads

    def _current_reward_shaping_coef(self) -> float:
        initial = max(0.0, float(self.reward_shaping_initial_coef))
        final = max(0.0, float(self.reward_shaping_final_coef))
        anneal_games = max(1, int(self.reward_shaping_anneal_games))
        progress = max(0.0, min(1.0, float(self.games_played) / float(anneal_games)))
        return float(initial + ((final - initial) * progress))

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("message", "")
        return str(value or "").strip().lower()

    def _estimate_payment_values(
        self,
        payment: Optional[Dict[str, Any]],
        player_state: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not isinstance(payment, dict):
            return {
                "steel_units": 0.0,
                "titanium_units": 0.0,
                "mc_units": 0.0,
                "steel_value_spent": 0.0,
                "titanium_value_spent": 0.0,
                "metal_value_spent": 0.0,
                "total_value_spent": 0.0,
            }
        player = {}
        if isinstance(player_state, dict):
            player = player_state.get("thisPlayer", {}) or {}
        steel_value = float(player.get("steelValue", 2) or 2)
        titanium_value = float(player.get("titaniumValue", 3) or 3)
        try:
            steel_units = float(payment.get("steel", 0) or 0)
        except Exception:
            steel_units = 0.0
        try:
            titanium_units = float(payment.get("titanium", 0) or 0)
        except Exception:
            titanium_units = 0.0
        mc_units = 0.0
        for key in ("megaCredits", "megacredits", "mega_credit", "mc", "credits"):
            if key in payment:
                try:
                    mc_units += float(payment.get(key, 0) or 0)
                except Exception:
                    continue
        steel_value_spent = max(0.0, steel_units) * max(0.0, steel_value)
        titanium_value_spent = max(0.0, titanium_units) * max(0.0, titanium_value)
        metal_value_spent = steel_value_spent + titanium_value_spent
        total_value_spent = metal_value_spent + max(0.0, mc_units)
        return {
            "steel_units": float(max(0.0, steel_units)),
            "titanium_units": float(max(0.0, titanium_units)),
            "mc_units": float(max(0.0, mc_units)),
            "steel_value_spent": float(steel_value_spent),
            "titanium_value_spent": float(titanium_value_spent),
            "metal_value_spent": float(metal_value_spent),
            "total_value_spent": float(total_value_spent),
        }

    def _estimate_action_payment_value(
        self,
        action_input: Optional[Dict[str, Any]],
        player_state: Optional[Dict[str, Any]],
    ) -> float:
        total = 0.0
        for payload in self._iter_action_payloads(action_input):
            payment_summary = self._estimate_payment_values(payload.get("payment"), player_state)
            total += float(payment_summary.get("total_value_spent", 0.0))
        return float(total)

    def _infer_rare_state_flags(
        self,
        player_state: Optional[Dict[str, Any]],
        action_input: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        waiting_for = {}
        if isinstance(player_state, dict):
            waiting_for = player_state.get("waitingFor", {}) or {}
        waiting_type = self._normalize_text(waiting_for.get("type", ""))
        waiting_title = self._normalize_text(waiting_for.get("title", ""))
        option_text = " ".join(
            self._normalize_text((opt or {}).get("title", ""))
            for opt in (waiting_for.get("options", []) or [])
            if isinstance(opt, dict)
        )
        action_text = self._normalize_text((action_input or {}).get("type", ""))
        combined = " ".join([waiting_type, waiting_title, option_text, action_text]).strip()

        is_award_funding = (
            ("award" in combined and "fund" in combined)
            or action_text in ("fundaward", "fund_award", "award")
            or waiting_type in ("fundaward", "award")
        )
        is_milestone_timing = (
            ("milestone" in combined and ("claim" in combined or "fund" in combined))
            or action_text in ("claimmilestone", "milestone")
            or waiting_type in ("milestone", "claimmilestone")
        )
        is_draft_keep_buy = (
            waiting_type in ("initialcards", "selectinitialcards", "draft", "draftcards")
            or ("draft" in combined)
            or ("keep" in combined and "card" in combined)
            or ("buy" in combined and "card" in combined)
        )
        payment_value = self._estimate_action_payment_value(action_input, player_state)
        is_high_cost_payment = payment_value >= float(self.rare_high_cost_payment_threshold)

        rare_count = (
            int(is_award_funding)
            + int(is_milestone_timing)
            + int(is_draft_keep_buy)
            + int(is_high_cost_payment)
        )
        weight = 1.0 + (float(self.rare_state_priority_bonus) * float(rare_count))
        return {
            "award_funding": 1.0 if is_award_funding else 0.0,
            "milestone_timing": 1.0 if is_milestone_timing else 0.0,
            "draft_keep_buy": 1.0 if is_draft_keep_buy else 0.0,
            "high_cost_payment": 1.0 if is_high_cost_payment else 0.0,
            "weight": float(max(1.0, weight)),
            "payment_value": float(payment_value),
            "rare_count": float(rare_count),
        }

    def _categorize_action(self, action_index: int) -> str:
        pass_base = int(self.action_decoder.action_types.get('PASS', 900))
        mask_base = int(self.action_decoder.action_types.get('SELECT_CARD_MASK', -1))
        mask_limit = int(getattr(self.action_decoder, 'card_selection_mask_limit', 0) or 0)
        startup_base = int(self.action_decoder.action_types.get('STARTUP_PLAN', -1))
        startup_limit = int(getattr(self.action_decoder, 'startup_plan_limit', 0) or 0)
        if action_index >= pass_base:
            return 'pass'
        if action_index < 100:
            return 'play_card'
        if action_index < 200:
            return 'standard_project'
        if startup_base >= 0 and startup_limit > 0 and startup_base <= action_index < (startup_base + startup_limit):
            return 'startup_plan'
        if action_index < 300:
            return 'select_option'
        if mask_base >= 0 and mask_limit > 0 and mask_base <= action_index < (mask_base + mask_limit):
            return 'card_selection_mask'
        if action_index == 700:
            return 'convert_plants'
        if action_index == 701:
            return 'convert_heat'
        if action_index == 702:
            return 'sell_patents'
        return 'other'

    def _describe_action(self, action_index: int, player_state: Dict[str, Any]) -> str:
        """Provide a human-readable description for an action index based on current state."""
        waiting_for = player_state.get('waitingFor', {})
        input_type = waiting_for.get('type', '')
        
        # Base components from action_decoder
        pass_base = int(self.action_decoder.action_types.get('PASS', 900))
        mask_base = int(self.action_decoder.action_types.get('SELECT_CARD_MASK', -1))
        mask_limit = int(getattr(self.action_decoder, 'card_selection_mask_limit', 0) or 0)
        startup_base = int(self.action_decoder.action_types.get('STARTUP_PLAN', -1))
        startup_limit = int(getattr(self.action_decoder, 'startup_plan_limit', 0) or 0)

        if action_index >= pass_base:
            return "PASS"
        
        # PLAY_CARD range (0-99)
        if action_index < 100:
            cards = waiting_for.get('cards', [])
            if 0 <= action_index < len(cards):
                card_name = cards[action_index].get('name', f"idx:{action_index}")
                return f"PLAY_CARD({card_name})"
            # Fallback for nested OR menus
            if input_type == 'or':
                for opt in waiting_for.get('options', []):
                    if opt.get('type') in ['projectCard', 'selectProjectCardToPlay', 'card', 'selectCard']:
                        opt_cards = opt.get('cards', [])
                        if 0 <= action_index < len(opt_cards):
                            return f"PLAY_CARD({opt_cards[action_index].get('name')})"
            return f"PLAY_CARD({action_index})"

        # STANDARD_PROJECT range (100-199)
        if action_index < 200:
            idx = action_index - 100
            sp_names = getattr(self.action_decoder, 'standard_projects', [])
            if 0 <= idx < len(sp_names):
                return f"STANDARD_PROJECT({sp_names[idx]})"
            sp_cards = waiting_for.get('cards', [])
            if 0 <= idx < len(sp_cards):
                return f"STANDARD_PROJECT({sp_cards[idx].get('name')})"
            return f"STANDARD_PROJECT({idx})"

        # SELECT_OPTION range (200-299)
        if 200 <= action_index < 300:
            idx = action_index - 200
            options = waiting_for.get('options', [])
            if 0 <= idx < len(options):
                title = options[idx].get('title', '')
                if isinstance(title, dict): title = title.get('message', '')
                title = str(title).strip()
                if not title: title = options[idx].get('type', f"idx:{idx}")
                return f"SELECT_OPTION({title})"
            return f"SELECT_OPTION({idx})"

        # Specialized indices (700+)
        if action_index == 700:
            if input_type == 'selectResources': return "SELECT_RESOURCES"
            return "CONVERT_PLANTS"
        if action_index == 701: return "CONVERT_HEAT"
        if action_index == 702: return "SELL_PATENTS"
        if action_index == 710: return "PROD_TO_LOSE"
        if 720 <= action_index < 730: return f"SELECT_COLONY({action_index-720})"
        if 730 <= action_index < 740: return f"SELECT_PARTY({action_index-730})"
        if 740 <= action_index < 750: return f"SELECT_DELEGATE({action_index-740})"
        if 750 <= action_index < 760: return f"SELECT_GLOBAL_EVENT({action_index-750})"
        if 760 <= action_index < 770: return f"SELECT_UNDERGROUND_TOKEN({action_index-760})"
        if action_index == 800: return "STARTUP_FALLBACK"
        if action_index == 810: return "ARES_GLOBAL_PARAMS"
        if 820 <= action_index < 830: return f"SELECT_RESOURCE_TYPE({action_index-820})"
        if action_index == 830: return "AND_CHOICE"
        if 840 <= action_index < 850: return f"SELECT_POLICY({action_index-840})"
        
        # STARTUP_PLAN range
        if startup_base >= 0 and startup_limit > 0 and startup_base <= action_index < (startup_base + startup_limit):
            return f"STARTUP_PLAN({action_index - startup_base})"

        # SELECT_CARD_MASK range
        if mask_base >= 0 and mask_limit > 0 and mask_base <= action_index < (mask_base + mask_limit):
            return f"CARD_SELECTION_MASK({action_index - mask_base})"

        # AWARD / PLAYER selection (600+)
        if 600 <= action_index < 700:
            if input_type == 'selectPlayer':
                players = waiting_for.get('players', [])
                idx = action_index - 600
                if 0 <= idx < len(players):
                    name = players[idx].get('name', players[idx].get('color', f"idx:{idx}"))
                    return f"PLAYER({name})"
            if input_type == 'or':
                idx = action_index - 600
                for opt in waiting_for.get('options', []):
                    opt_title = opt.get('title', '')
                    if isinstance(opt_title, dict): opt_title = opt_title.get('message', '')
                    if 'award' in str(opt_title).lower():
                        award_opts = opt.get('options', [])
                        if 0 <= idx < len(award_opts):
                            title = award_opts[idx].get('title', '')
                            if isinstance(title, dict): title = title.get('message', '')
                            return f"AWARD({title})"
            return f"OTHER({action_index})"

        # AMOUNT selection (500+)
        if 500 <= action_index < 600:
            return f"SELECT_AMOUNT({action_index - 500})"

        # SPACE selection (300+)
        if 300 <= action_index < 500:
            return f"SELECT_SPACE({action_index - 300})"

        return f"OTHER({action_index})"

    def _build_policy_ranking(
        self,
        player_state: Dict[str, Any],
        available_actions: List[int],
        policy_logits: torch.Tensor,
        policy_probs: torch.Tensor,
        masked_distribution: Optional[torch.Tensor],
        chosen_action_index: Optional[int],
    ) -> List[Dict[str, Any]]:
        ranking: List[Dict[str, Any]] = []
        logits_vec = policy_logits.detach().cpu().reshape(-1)
        probs_vec = policy_probs.detach().cpu().reshape(-1)
        masked_vec = masked_distribution.detach().cpu().reshape(-1) if isinstance(masked_distribution, torch.Tensor) else None

        for pos, action_idx in enumerate(available_actions):
            idx = int(action_idx)
            if pos < 0 or pos >= int(probs_vec.numel()):
                continue
            try:
                decoded_action = self.action_decoder.decode_action(idx, player_state)
            except Exception:
                decoded_action = None
            ranking.append(
                {
                    "action_index": idx,
                    "label": self._describe_action(idx, player_state),
                    "decoded_action": decoded_action,
                    "raw_probability": float(probs_vec[pos].item()),
                    "masked_probability": float(masked_vec[pos].item()) if masked_vec is not None and pos < int(masked_vec.numel()) else 0.0,
                    "logit": float(logits_vec[pos].item()) if pos < int(logits_vec.numel()) else 0.0,
                    "chosen": bool(chosen_action_index is not None and idx == int(chosen_action_index)),
                    "legal": True,
                }
            )

        ranking.sort(
            key=lambda item: (
                float(item.get("masked_probability", 0.0)),
                float(item.get("raw_probability", 0.0)),
                float(item.get("logit", 0.0)),
            ),
            reverse=True,
        )
        return ranking

    def _maybe_capture_decision_snapshot(
        self,
        game_instance: GameInstance,
        player_id: Optional[str],
        player_state: Dict[str, Any],
        action_input: Optional[Dict[str, Any]],
        action_index: Optional[int],
        action_meta: Optional[Dict[str, Any]],
        sampled_from_policy: bool,
        send_outcome: str,
        turn_action_count: int,
        state_vector: Optional[np.ndarray],
    ) -> None:
        if not isinstance(action_meta, dict):
            return

        request = reserve_pending_capture_request(
            agent_id=self.id,
            game_id=getattr(game_instance, "game_id", None),
            player_id=player_id,
        )
        if request is None:
            return

        try:
            try:
                game_url = game_instance.get_public_game_url()
            except Exception:
                game_url = ""
            snapshot = build_decision_snapshot(
                request=request,
                agent_id=self.id,
                game_id=getattr(game_instance, "game_id", None),
                game_url=game_url,
                player_id=player_id,
                player_state=player_state,
                action_input=action_input,
                action_index=action_index,
                action_meta=action_meta,
                sampled_from_policy=sampled_from_policy,
                send_outcome=send_outcome,
                turn_action_count=turn_action_count,
                state_vector=state_vector,
            )
            saved = save_snapshot(snapshot)
            complete_capture_request(
                request_id=str(request.get("request_id", "") or ""),
                snapshot_id=str(saved.get("snapshot_id", "") or ""),
                snapshot_path=str(saved.get("snapshot_path", "") or ""),
            )
            logger.info(
                "Captured decision snapshot %s for agent %s",
                str(saved.get("snapshot_id", "") or ""),
                self.id[:8],
            )
        except Exception as exc:
            fail_capture_request(str(request.get("request_id", "") or ""), str(exc))
            logger.warning("Failed to capture decision snapshot for agent %s: %s", self.id[:8], exc)

    def _extract_standard_project_name(
        self,
        action_index: int,
        action_input: Optional[Dict[str, Any]],
        player_state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Best-effort extraction of selected standard project name for telemetry."""
        known_names = set(getattr(self.action_decoder, "standard_projects", []) or [])

        # Direct payload naming.
        if action_input:
            if action_input.get('type') == 'standardProject':
                project = str(action_input.get('project', '') or '').strip()
                return project or None
            if action_input.get('type') == 'card':
                selected = action_input.get('cards', []) or []
                if selected:
                    candidate = str(selected[0]).strip()
                    if candidate in known_names:
                        return candidate
            if action_input.get('type') == 'projectCard':
                candidate = str(action_input.get('card', '') or '').strip()
                if candidate in known_names:
                    return candidate

        waiting_for = (player_state or {}).get('waitingFor', {}) if player_state else {}
        input_type = str(waiting_for.get('type', ''))
        title = waiting_for.get('title', '')
        if isinstance(title, dict):
            title = title.get('message', '')
        title_l = str(title).lower()

        if input_type in ['card', 'selectCard'] and 'standard project' in title_l:
            cards = waiting_for.get('cards', []) or []
            normalized = int(action_index)
            if 100 <= normalized < 200:
                normalized -= 100
            if 0 <= normalized < len(cards):
                name = str(cards[normalized].get('name', '') or '').strip()
                if name:
                    return name
            enabled = [c for c in cards if not c.get('isDisabled', False)]
            if enabled:
                name = str(enabled[0].get('name', '') or '').strip()
                return name or None
            return None

        if input_type == 'or':
            options = waiting_for.get('options', []) or []
            for option in options:
                opt_title = option.get('title', '')
                if isinstance(opt_title, dict):
                    opt_title = opt_title.get('message', '')
                opt_title_l = str(opt_title).lower()
                if option.get('type') in ['card', 'selectCard'] and 'standard project' in opt_title_l:
                    cards = option.get('cards', []) or []
                    normalized = int(action_index)
                    if 100 <= normalized < 200:
                        normalized -= 100
                    if 0 <= normalized < len(cards):
                        name = str(cards[normalized].get('name', '') or '').strip()
                        if name:
                            return name
                    enabled = [c for c in cards if not c.get('isDisabled', False)]
                    if enabled:
                        name = str(enabled[0].get('name', '') or '').strip()
                        return name or None
                    break
        return None

    def _record_action_choice(
        self,
        action_index: int,
        action_input: Optional[Dict[str, Any]] = None,
        player_state: Optional[Dict[str, Any]] = None,
    ):
        counts = self.decision_stats.setdefault('action_type_counts', {})
        category = self._categorize_action(int(action_index))
        counts[category] = int(counts.get(category, 0)) + 1
        if category == 'standard_project':
            project_name = self._extract_standard_project_name(int(action_index), action_input, player_state)
            if project_name:
                project_counts = self.decision_stats.setdefault('standard_project_counts', {})
                project_counts[project_name] = int(project_counts.get(project_name, 0)) + 1

        # Track project card plays and resource spend, including nested OR/AND payloads.
        if not isinstance(action_input, dict):
            return

        for payload in self._iter_action_payloads(action_input):
            if not isinstance(payload, dict):
                continue

            action_type = str(payload.get('type', '') or '')
            if category != 'standard_project' and (
                action_type == 'projectCard'
                or (action_type == 'card' and 'card' in payload)
            ):
                self._bump_decision_stat('card_play_actions')

            payment = payload.get('payment')
            if isinstance(payment, dict):
                try:
                    steel_units = int(payment.get('steel', 0) or 0)
                except Exception:
                    steel_units = 0
                try:
                    titanium_units = int(payment.get('titanium', 0) or 0)
                except Exception:
                    titanium_units = 0
                payment_summary = self._estimate_payment_values(payment, player_state)
                payment_total = float(payment_summary.get("total_value_spent", 0.0))
                metal_total = float(payment_summary.get("metal_value_spent", 0.0))
                if payment_total > 0.0:
                    self._bump_decision_stat_float('project_payment_value_total', payment_total)
                if metal_total > 0.0:
                    self._bump_decision_stat_float('metal_payment_value_total', metal_total)
                steel_value_spent = float(payment_summary.get("steel_value_spent", 0.0))
                titanium_value_spent = float(payment_summary.get("titanium_value_spent", 0.0))
                if steel_value_spent > 0.0:
                    self._bump_decision_stat_float('steel_payment_value_total', steel_value_spent)
                if titanium_value_spent > 0.0:
                    self._bump_decision_stat_float('titanium_payment_value_total', titanium_value_spent)
                if steel_units > 0:
                    self._bump_decision_stat('steel_spent', steel_units)
                if titanium_units > 0:
                    self._bump_decision_stat('titanium_spent', titanium_units)
    
    def _sync_forward_and_probs(
        self,
        planner_state: Dict[str, Any],
        phase_index: int,
        recurrent_state: "torch.Tensor",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any, torch.Tensor]:
        """Forward pass dispatched to ThreadPoolExecutor.

        Runs on ``self._inference_device`` (GPU when available) with FP16
        autocast for CUDA to cut latency and VRAM.  On CUDA OOM, clears the
        cache and retries on CPU so the game keeps running.
        """
        device = self._inference_device
        return self._sync_forward_impl(planner_state, phase_index, recurrent_state, device)

    def _sync_forward_impl(
        self,
        planner_state: Dict[str, Any],
        phase_index: int,
        recurrent_state: "torch.Tensor",
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any, torch.Tensor]:
        with self._model_device_lock:
            device = self._ensure_network_device_consistency(device)
            state_tensor = bundle_to_torch(planner_state, device=device, planner_config=self.config.planner_config())
            phase_tensor = torch.tensor(
                [phase_index], dtype=torch.long, device=device,
            )
            recurrent_state_tensor = recurrent_state.unsqueeze(0)
            if not _devices_match(recurrent_state_tensor.device, device):
                recurrent_state_tensor = recurrent_state_tensor.to(device)

            try:
                with torch.no_grad():
                    use_amp = device.type == "cuda"
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = self._forward_network(
                            state_tensor,
                            phase_indices=phase_tensor,
                            recurrent_state=recurrent_state_tensor,
                        )

                    policy_logits = out["policy_logits"].float()
                    value = out["value"].float()
                    recurrent_state_out = out.get("recurrent_state")
                    if recurrent_state_out is not None:
                        recurrent_state_out = recurrent_state_out.float()
                    aux_predictions = out.get("aux_predictions")

                    policy_temperature = self._effective_policy_temperature()
                    policy_logits = policy_logits / max(policy_temperature, 1e-3)
                    policy_probs = F.softmax(policy_logits, dim=-1)

                if device.type != "cpu":
                    policy_logits = policy_logits.cpu()
                    value = value.cpu()
                    policy_probs = policy_probs.cpu()

                return policy_logits, value, recurrent_state_out, aux_predictions, policy_probs

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if device.type == "cpu":
                    raise
                err_msg = str(exc)
                if "CUDA" not in err_msg and "out of memory" not in err_msg:
                    raise
                logger.warning("CUDA OOM during inference; clearing cache and falling back to CPU.")
                torch.cuda.empty_cache()
                cpu = torch.device("cpu")
                _move_network_and_optimizer_to(self.network, self.optimizer, cpu)
                self._inference_device = cpu
                return self._sync_forward_impl(planner_state, phase_index, recurrent_state, cpu)

    async def _get_action_from_network(
        self,
        planner_state: Dict[str, Any],
        player_state: Dict[str, Any],
        action_descriptors: List[Dict[str, Any]],
        raw_available_actions: List[int],
        player_id: Optional[str] = None,
        force_random: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool, Optional[Dict[str, Any]]]:
        """Get action from neural network"""
        try:
            phase_index = int(self._extract_phase_index(player_state))
            recurrent_state_in = self._get_recurrent_state_for_player(player_id)

            forward_started = time.perf_counter()
            try:
                if self._inference_batcher is not None and isinstance(planner_state, np.ndarray):
                    policy_logits, value, recurrent_state_out, aux_predictions, policy_probs = (
                        await self._inference_batcher.infer(planner_state, phase_index, recurrent_state_in)
                    )
                else:
                    loop = asyncio.get_running_loop()
                    executor = _get_inference_executor()
                    policy_logits, value, recurrent_state_out, aux_predictions, policy_probs = await loop.run_in_executor(
                        executor,
                        self._sync_forward_and_probs,
                        planner_state,
                        phase_index,
                        recurrent_state_in,
                    )
            finally:
                self._record_pipeline_timing("network_forward_sec", time.perf_counter() - forward_started)

            if recurrent_state_out is not None and player_id:
                self._set_recurrent_state_for_player(player_id, recurrent_state_out)

            policy_temperature = self._effective_policy_temperature()
            available_actions = [int(item.get("action_index", -1)) for item in action_descriptors]
            waiting_for = player_state.get('waitingFor', {})
            self._bump_decision_stat('available_action_observations')
            self._bump_decision_stat('sum_available_actions', len(available_actions))
            self._bump_decision_stat('action_mask_observations')
            self._bump_decision_stat('action_legal_count_total', len(available_actions))
            prefer_project_cards = False
            waiting_type = str(waiting_for.get('type', '') or '').lower()
            if waiting_type in ['projectcard', 'selectprojectcardtoplay']:
                prefer_project_cards = True
            elif waiting_type == 'or':
                options = waiting_for.get('options', [])
                has_project_card_option = any(
                    str(option.get('type', '') or '') in ['projectCard', 'selectProjectCardToPlay']
                    for option in options
                )
                def _opt_title_l(option: Dict[str, Any]) -> str:
                    raw = option.get('title', '')
                    if isinstance(raw, dict):
                        raw = raw.get('message', '')
                    return str(raw or '').lower()
                has_standard_project_option = any(
                    str(option.get('type', '') or '') in ['card', 'selectCard']
                    and 'standard project' in _opt_title_l(option)
                    for option in options
                )
                prefer_project_cards = bool(has_project_card_option and has_standard_project_option)
               
            if not available_actions:
                return None, None, False, None
                 
            if logger.isEnabledFor(logging.DEBUG):
                action_types = [
                    self._describe_action(action_idx, player_state)
                    for action_idx in available_actions
                ]
                logger.debug(f"Available actions: {action_types}")
             
            # Optional: adjust weights for OR menus based on option titles to avoid passing.
            # When PPO is enabled, these heuristic probability reweightings are
            # disabled because they create a mismatch between the behavior policy
            # (used to sample actions) and the policy PPO reconstructs from raw
            # masked logits during optimization.  The network's learnable
            # action_type_bias already handles general category preferences.
            action_weight_adjustments = None
            if self.ppo_enable:
                # Disable heuristic reweighting under PPO to preserve on-policy
                # consistency.  The network learns its own biases via
                # action_type_bias in the forward pass.
                prefer_project_cards = False
            elif waiting_for and waiting_for.get('type') == 'or':
                options = waiting_for.get('options', [])
                adjustments = {}
                select_option_base = 200
                for i, opt in enumerate(options):
                    title = opt.get('title', '')
                    if isinstance(title, dict):
                        title = title.get('message', '')
                    title_l = str(title).lower()
                    idx = select_option_base + i
                    # Downweight passing and selling patents
                    if 'pass' in title_l:
                        adjustments[idx] = 0.2
                    elif 'sell patents' in title_l:
                        adjustments[idx] = 0.08
                    # Upweight productive actions
                    elif 'play project card' in title_l:
                        adjustments[idx] = 1.8
                    elif 'perform an action' in title_l or 'take action' in title_l:
                        # Blue card actions entry
                        adjustments[idx] = 1.7
                    elif 'standard project' in title_l:
                        adjustments[idx] = 0.85
                    elif 'convert heat' in title_l or 'convert 8 heat' in title_l or 'convert plants' in title_l:
                        adjustments[idx] = 1.3
                if adjustments:
                    action_weight_adjustments = adjustments

            # Ensure eval mode for inference
            try:
                self.network.eval()
            except Exception:
                pass

            # Sample action based on policy
            action_position, action_index, sampled_from_policy, sampled_distribution = self._sample_action(
                policy_probs.squeeze(),
                available_actions,
                force_random=force_random,
                action_weight_adjustments=action_weight_adjustments,
                prefer_project_cards=prefer_project_cards,
            )
              
            # Convert to game input
            decode_started = time.perf_counter()
            action_input = self.action_decoder.decode_action(action_index, player_state)
            self._record_pipeline_timing("decode_action_sec", time.perf_counter() - decode_started)
            aux_targets = self._compute_aux_targets(player_state)
            rare_flags = self._infer_rare_state_flags(player_state, action_input)
            recurrent_out_vec: List[float] = []
            if isinstance(recurrent_state_out, torch.Tensor):
                recurrent_out_vec = recurrent_state_out.detach().cpu().reshape(-1).tolist()
            aux_pred_vec: List[float] = []
            if isinstance(aux_predictions, torch.Tensor):
                aux_pred_vec = aux_predictions.detach().cpu().reshape(-1).tolist()
            action_family_counts: Dict[str, int] = {}
            for descriptor in action_descriptors:
                family = str(descriptor.get("family", "other") or "other").strip() or "other"
                action_family_counts[family] = int(action_family_counts.get(family, 0)) + 1
            action_meta: Optional[Dict[str, Any]] = {
                "phase_index": int(phase_index),
                "recurrent_state": recurrent_state_in.detach().cpu().reshape(-1).tolist(),
                "recurrent_state_out": recurrent_out_vec,
                "aux_targets": aux_targets,
                "aux_predictions": aux_pred_vec,
                "rare_state_weight": float(rare_flags.get("weight", 1.0)),
                "rare_award_funding": float(rare_flags.get("award_funding", 0.0)),
                "rare_milestone_timing": float(rare_flags.get("milestone_timing", 0.0)),
                "rare_draft_keep_buy": float(rare_flags.get("draft_keep_buy", 0.0)),
                "rare_high_cost_payment": float(rare_flags.get("high_cost_payment", 0.0)),
                "payment_value_estimate": float(rare_flags.get("payment_value", 0.0)),
                "available_actions_raw": [int(a) for a in raw_available_actions],
                "available_actions_filtered": [int(a) for a in available_actions],
                "action_descriptors": list(action_descriptors),
                "chosen_action_position": int(action_position),
                "chosen_action_label": self._describe_action(int(action_index), player_state),
                "sampled_from_policy": bool(sampled_from_policy),
                "bundle_summary": {
                    "world_token_count": int(planner_state["world_tokens"].shape[0]),
                    "hand_token_count": int(planner_state["hand_tokens"].shape[0]),
                    "action_token_count": int(planner_state["action_tokens"].shape[0]),
                    "legal_action_count": int(sum(1 for item in available_actions if int(item) >= 0)),
                    "action_family_counts": action_family_counts,
                },
            }
            if has_pending_capture_request(agent_id=self.id):
                transformer_stats = dict(getattr(self.network, "last_transformer_stats", {}) or {})
                transformer_stats.update(
                    {
                        "planner_hand_limit": int(getattr(self.config, "planner_hand_limit", 0)),
                        "planner_tableau_limit": int(getattr(self.config, "planner_tableau_limit", 0)),
                        "planner_opponent_limit": int(getattr(self.config, "planner_opponent_limit", 0)),
                        "planner_token_dim": int(getattr(self.config, "planner_token_dim", 0)),
                        "planner_global_dim": int(getattr(self.config, "planner_global_dim", 0)),
                        "planner_opportunity_limit": int(getattr(self.config, "planner_opportunity_limit", 0)),
                    }
                )
                ranking = self._build_policy_ranking(
                    player_state=player_state,
                    available_actions=available_actions,
                    policy_logits=policy_logits.squeeze(),
                    policy_probs=policy_probs.squeeze(),
                    masked_distribution=sampled_distribution,
                    chosen_action_index=int(action_index),
                )
                action_meta["transformer_stats"] = transformer_stats
                action_meta["policy_ranking"] = ranking
                action_meta["policy_top_actions"] = ranking[:12]
                action_meta["prompt_card_rankings"] = self.state_encoder.build_prompt_card_rankings(player_state)
            if sampled_distribution is not None:
                action_meta["value_old"] = float(value.squeeze().item())
                action_meta["legal_actions"] = [int(a) for a in available_actions]
                action_meta["policy_temperature"] = float(policy_temperature)
            if sampled_from_policy and sampled_distribution is not None:
                chosen_pos = max(0, min(int(action_position), int(sampled_distribution.numel()) - 1))
                action_prob = float(sampled_distribution[chosen_pos].item())
                action_meta["logp_old"] = float(np.log(max(1e-8, action_prob)))

            return action_input, action_index, sampled_from_policy, action_meta
             
        except Exception as e:
            logger.error(f"Error getting action from network: {e}")
            return None, None, False, None
    
    def _sample_action(
        self,
        policy_probs: torch.Tensor,
        available_actions: List[int],
        force_random: bool = False,
        action_weight_adjustments: Optional[Dict[int, float]] = None,
        prefer_project_cards: bool = False,
    ) -> Tuple[int, int, bool, Optional[torch.Tensor]]:
        """Sample action from policy, restricted to available actions.

        Hard-coded probability multipliers have been removed from this method.
        Per-category biases are now applied as learnable logit offsets in
        TerraformingMarsNetwork.forward() (self.action_type_bias), initialised
        to the log of the old multipliers so initial behaviour is unchanged but
        gradients can adjust the values over time.

        What remains here:
          1. ε-greedy random exploration.
          2. Legal-action masking (zero out unavailable actions).
          3. Contextual OR-menu title adjustments (action_weight_adjustments).
          4. Mild prefer_project_cards boost (kept small; network learns the rest).
        """
        masked_probs = torch.clamp(policy_probs.reshape(-1).float(), min=0.0)
        valid_positions = list(range(min(int(masked_probs.numel()), len(available_actions))))
        if not valid_positions:
            fallback_idx = int(np.random.choice(available_actions))
            return 0, fallback_idx, False, None
        if int(masked_probs.numel()) > len(available_actions):
            masked_probs = masked_probs[:len(available_actions)]
        total = float(masked_probs.sum().item())
        if total > 0:
            masked_probs = masked_probs / total
        else:
            masked_probs = torch.ones((len(valid_positions),), dtype=torch.float32, device=policy_probs.device)
            masked_probs = masked_probs / masked_probs.sum()

        # Small residual boost when the prompt is explicitly a project-card play
        # (prefer_project_cards=True means the server is asking to pick a card).
        # The network's action_type_bias handles the general preference; this is
        # a context-specific nudge based on waiting-for type, not a hand-tuned value.
        if prefer_project_cards:
            for pos, action_idx in enumerate(available_actions[:len(masked_probs)]):
                if 0 <= int(action_idx) < 100:
                    masked_probs[pos] *= float(self.project_card_priority_weight)

        # Apply OR-menu title adjustments (e.g. downweight pass/sell options).
        if action_weight_adjustments:
            for pos, action_idx in enumerate(available_actions[:len(masked_probs)]):
                if int(action_idx) in action_weight_adjustments:
                    masked_probs[pos] *= float(action_weight_adjustments[int(action_idx)])

        # Numerical stability
        masked_probs = masked_probs + 1e-8

        total_prob = float(masked_probs.sum().item())
        if total_prob <= 0:
            random_pos = int(np.random.choice(valid_positions))
            return random_pos, int(available_actions[random_pos]), False, None
        masked_probs = masked_probs / total_prob

        effective_epsilon = self._effective_policy_epsilon(force_random=force_random)
        if np.random.random() < float(effective_epsilon):
            random_pos = int(np.random.choice(valid_positions))
            return random_pos, int(available_actions[random_pos]), False, masked_probs

        try:
            chosen_pos = int(torch.multinomial(masked_probs, 1).item())
            return chosen_pos, int(available_actions[chosen_pos]), True, masked_probs
        except RuntimeError:
            pass_action_base = int(self.action_decoder.action_types.get('PASS', 900))
            non_pass_actions = [a for a in available_actions if a < pass_action_base]
            if non_pass_actions:
                fallback_idx = int(np.random.choice(non_pass_actions))
                fallback_pos = max(0, available_actions.index(fallback_idx))
                return fallback_pos, fallback_idx, False, None
            fallback_pos = int(np.random.choice(valid_positions))
            return fallback_pos, int(available_actions[fallback_pos]), False, None
    
    async def _record_game_result(self, final_state: Dict[str, Any], player_name: str, game_instance: GameInstance) -> Dict[str, Any]:
        """Record the result of a completed game"""
        try:
            # Find our player in the final state
            our_player = None
            for player in final_state.get('players', []):
                if player.get('name') == player_name:
                    our_player = player
                    break
            
            vp = 0
            rank = 4

            # Prefer authoritative data from the JSON player view to avoid parsing dynamic HTML
            try:
                our_player_id = (our_player or {}).get('id')
                if our_player_id and game_instance:
                    # Use the same base URL/session as the game instance for reliable in-cluster access
                    internal_base = getattr(game_instance, 'base_url', os.getenv('INTERNAL_TM_URL', os.getenv('PUBLIC_TM_URL', 'http://localhost:8081')))
                    session = game_instance._get_session()
                    async with session.get(f"{internal_base}/api/player", params={'id': our_player_id}) as r2:
                        if r2.status == 200:
                            view = await r2.json()
                            players_view = view.get('players', []) or []
                            # Sort like the UI: by total VP desc, then megacredits desc
                            def vp_total(p: Dict[str, Any]) -> int:
                                return int(((p.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0))
                            def mc_val(p: Dict[str, Any]) -> int:
                                return int(p.get('megaCredits', 0) or 0)
                            sorted_players = sorted(players_view, key=lambda p: (vp_total(p), mc_val(p)), reverse=True)
                            for idx, p in enumerate(sorted_players, start=1):
                                if p.get('name') == player_name:
                                    vp = vp_total(p)
                                    rank = idx
                                    break
                        else:
                            logger.warning(f"Agent {self.id[:8]} failed to fetch player view JSON (HTTP {r2.status}).")
            except Exception as _:
                pass



            # Final fallback to whatever is available in the final state
            if our_player and vp == 0:
                vp = int(((our_player.get('victoryPointsBreakdown', {}) or {}).get('total', 0) or 0))
                # Rank may not be present; keep previous value if missing
                rank = int(our_player.get('rank', rank) or rank)

            # Update aggregates
            self.total_victory_points += int(vp)
            if int(rank) == 1:
                self.wins += 1
            logger.info(f"Agent {self.id[:8]} finished rank {rank} with {vp} VP")
            return {"completed": True, "rank": int(rank), "vp": int(vp)}
        
        except Exception as e:
            logger.error(f"Error recording game result: {e}")
            return {"completed": False, "rank": 4, "vp": 0}

    def _compute_terminal_reward(self, rank: int, vp: int, completed: bool = True) -> float:
        """Convert game outcome into a bounded terminal reward for policy updates."""
        return calculate_terminal_reward(rank=rank, victory_points=vp, completed=completed)

    async def _queue_episode_rollout(self, episode_steps: List[Dict[str, Any]], terminal_reward: float):
        """Queue rollout transitions for coordinator-driven PPO optimization."""
        if not episode_steps:
            return

        steps = list(episode_steps)  # deque already capped at max_episode_steps
        rollout_steps: List[PPORolloutStep] = []
        for idx, step in enumerate(steps):
            try:
                state_bundle = step.get("state_bundle")
                if not isinstance(state_bundle, dict):
                    continue
                reward = float(step.get("reward", 0.0))
                if idx == len(steps) - 1:
                    reward += float(terminal_reward)
                aux_raw = step.get("aux_targets", {})
                if not isinstance(aux_raw, dict):
                    aux_raw = {}
                aux_pred_raw = list(step.get("aux_predictions", []) or [])
                while len(aux_pred_raw) < int(self.config.planner_aux_output_dim):
                    aux_pred_raw.append(0.0)

                rollout_steps.append(
                    PPORolloutStep(
                        state_bundle=state_bundle,
                        action=int(step.get("action_position", 0)),
                        action_index=int(step.get("action_index", step.get("action_position", 0))),
                        logp_old=float(step.get("logp_old", 0.0)),
                        value_old=float(step.get("value_old", 0.0)),
                        reward=reward,
                        done=(idx == len(steps) - 1),
                        legal_actions=[int(a) for a in step.get("legal_actions", [])],
                        phase_index=int(step.get("phase_index", 0)),
                        recurrent_state=np.asarray(step.get("recurrent_state", []), dtype=np.float32).reshape(-1),
                        aux_targets=np.asarray(aux_raw.get("planner_vector", []), dtype=np.float32).reshape(-1),
                        aux_predictions=np.asarray(aux_pred_raw[:int(self.config.planner_aux_output_dim)], dtype=np.float32).reshape(-1),
                        rare_state_weight=float(step.get("rare_state_weight", 1.0) or 1.0),
                        rare_award_funding=float(step.get("rare_award_funding", 0.0) or 0.0),
                        rare_milestone_timing=float(step.get("rare_milestone_timing", 0.0) or 0.0),
                        rare_draft_keep_buy=float(step.get("rare_draft_keep_buy", 0.0) or 0.0),
                        rare_high_cost_payment=float(step.get("rare_high_cost_payment", 0.0) or 0.0),
                        reward_tr_component=float(step.get("reward_tr_component", 0.0) or 0.0),
                        reward_cards_vp_component=float(step.get("reward_cards_vp_component", 0.0) or 0.0),
                        reward_city_greenery_component=float(step.get("reward_city_greenery_component", 0.0) or 0.0),
                        reward_city_future_component=float(step.get("reward_city_future_component", 0.0) or 0.0),
                        reward_milestones_awards_component=float(step.get("reward_milestones_awards_component", 0.0) or 0.0),
                        reward_other_component=float(step.get("reward_other_component", 0.0) or 0.0),
                        reward_shaping_coef=float(step.get("reward_shaping_coef", 0.0) or 0.0),
                        state_schema_version=self.state_schema_version,
                    )
                )
            except Exception:
                continue

        if not rollout_steps:
            return

        async with self.training_lock:
            for step in rollout_steps:
                self.rollout_buffer.append(step)

        logger.info(
            "Agent %s queued PPO rollout: steps=%d terminal_reward=%.3f buffer_size=%d",
            self.id[:8],
            len(rollout_steps),
            float(terminal_reward),
            len(self.rollout_buffer),
        )

    async def optimize_from_rollout_buffer(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run PPO optimization from buffered rollout data. Offloaded to thread pool to keep event loop responsive."""
        if not self.ppo_enable:
            return {}

        take = int(max_steps) if max_steps is not None else int(self.ppo_rollout_steps)
        take = max(1, take)
        expected_schema_version = str(self.state_schema_version or "v1")
        schema_filtered = 0
        steps: List[PPORolloutStep] = []

        async with self.training_lock:
            if not self.rollout_buffer:
                return {}
            while self.rollout_buffer and len(steps) < take:
                candidate = self.rollout_buffer.popleft()
                if str(getattr(candidate, "state_schema_version", "")) != expected_schema_version:
                    schema_filtered += 1
                    continue
                steps.append(candidate)

        if not steps:
            return {"rollout/steps": 0, "rollout/schema_filtered": int(schema_filtered)}

        current_entropy_coef = float(self._current_ppo_entropy_coef())
        policy_temp = max(float(self._effective_policy_temperature()), 1e-3)

        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            _get_ppo_executor(),
            _run_ppo_update_sync,
            self,
            steps,
            current_entropy_coef,
            policy_temp,
        )

        metrics["rollout/schema_filtered"] = int(schema_filtered)
        if metrics:
            logger.info(
                "Agent %s PPO update: steps=%d policy_loss=%.4f value_loss=%.4f approx_kl=%.4f entropy_coef=%.4f",
                self.id[:8],
                int(metrics.get("rollout/steps", 0)),
                float(metrics.get("ppo/policy_loss", 0.0)),
                float(metrics.get("ppo/value_loss", 0.0)),
                float(metrics.get("ppo/approx_kl", 0.0)),
                float(metrics.get("ppo/entropy_coef", self.ppo_hparams.entropy_coef)),
            )
        return metrics

    def _current_ppo_entropy_coef(self) -> float:
        start = max(0.0, float(self.ppo_entropy_coef_start))
        end = max(0.0, float(self.ppo_entropy_coef_end))
        anneal_games = max(1, int(self.ppo_entropy_coef_anneal_games))
        if abs(start - end) <= 1e-12:
            return float(start)
        games_played = max(0.0, float(getattr(self, "games_played", 0)))
        progress = min(1.0, games_played / float(anneal_games))
        return float(start + ((end - start) * progress))

    def _adapt_ppo_learning_rate(self, approx_kl: float) -> Dict[str, float]:
        target_kl = float(self.ppo_hparams.target_kl)
        prev_lr = float(self.optimizer.param_groups[0].get("lr", float(self.ppo_learning_rate)))

        # Keep learning-rate adaptation deterministic and bounded.
        up_factor = max(1.0, float(self.ppo_lr_adapt_up))
        down_factor = min(1.0, max(1e-6, float(self.ppo_lr_adapt_down)))
        min_lr = max(1e-7, float(self.ppo_lr_min))
        max_lr = max(min_lr, float(self.ppo_lr_max))

        next_lr = prev_lr
        if target_kl > 0.0:
            if approx_kl > (1.5 * target_kl):
                next_lr = prev_lr * down_factor
            elif approx_kl < (0.5 * target_kl):
                next_lr = prev_lr * up_factor

        next_lr = max(min_lr, min(max_lr, float(next_lr)))
        if abs(next_lr - prev_lr) > 1e-12:
            for group in self.optimizer.param_groups:
                group["lr"] = float(next_lr)

        return {
            "ppo/learning_rate_prev": float(prev_lr),
            "ppo/learning_rate_next": float(next_lr),
            "ppo/lr_adjustment_applied": 1.0 if abs(next_lr - prev_lr) > 1e-12 else 0.0,
        }

    def get_rollout_buffer_size(self) -> int:
        return int(len(self.rollout_buffer))

    async def clear_rollout_buffer(self) -> int:
        """Clear queued PPO rollout samples and return the number of discarded steps."""
        async with self.training_lock:
            cleared = int(len(self.rollout_buffer))
            self.rollout_buffer.clear()
            return cleared

    async def _train_from_episode(self, episode_steps: List[Dict[str, Any]], terminal_reward: float):
        """Policy/value update from one self-play episode with terminal reward."""
        if not episode_steps:
            return

        # Protect optimizer/network updates when the same agent appears in concurrent games.
        async with self.training_lock:
            steps = list(episode_steps)  # deque already capped at max_episode_steps
            device = next(self.network.parameters()).device
            states = pad_bundle_batch(
                [step.get("state_bundle", {}) for step in steps],
                device,
                planner_config=self.config.planner_config(),
            )
            actions = torch.tensor(
                [int(step.get("action_position", step.get("action", 0))) for step in steps],
                dtype=torch.long,
                device=device,
            )
            phase_indices = torch.tensor(
                [int(step.get("phase_index", 0)) for step in steps],
                dtype=torch.long,
                device=device,
            )
            step_rewards = [
                float(step.get("reward", 0.0))
                for step in steps
            ]
            recurrent_states: Optional[torch.Tensor] = None
            recurrent_dim = max(0, int(getattr(self.network, "recurrent_size", 0)))
            if recurrent_dim > 0:
                recurrent_states = torch.zeros((len(steps), recurrent_dim), dtype=torch.float32, device=device)
                for row_idx, step in enumerate(steps):
                    vec = np.asarray(step.get("recurrent_state", []), dtype=np.float32).reshape(-1)
                    if vec.size <= 0:
                        continue
                    use = min(recurrent_dim, int(vec.size))
                    recurrent_states[row_idx, :use] = torch.from_numpy(vec[:use]).to(device)

            aux_dim = max(1, int(getattr(self.config, "planner_aux_output_dim", 280)))
            aux_targets = torch.zeros((len(steps), aux_dim), dtype=torch.float32, device=device)
            for row_idx, step in enumerate(steps):
                raw_aux = step.get("aux_targets", {}) or {}
                if isinstance(raw_aux, dict):
                    raw_aux = raw_aux.get("planner_vector", [])
                aux_vec = np.asarray(raw_aux, dtype=np.float32).reshape(-1)
                take = min(aux_dim, int(aux_vec.size))
                if take > 0:
                    aux_targets[row_idx, :take] = torch.from_numpy(aux_vec[:take]).to(device)

            # Dense+terminal reward: back-propagate step shaping and final outcome.
            running_return = float(terminal_reward)
            returns = []
            for immediate_reward in reversed(step_rewards):
                running_return = float(immediate_reward) + (float(self.config.discount_factor) * running_return)
                returns.append(running_return)
            returns.reverse()
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            if returns_t.numel() > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

            self.network.train()
            out = self._forward_network(
                states,
                phase_indices=phase_indices,
                recurrent_state=recurrent_states,
            )
            policy_logits = out["policy_logits"]
            values = out["value"]
            aux_predictions = out.get("aux_predictions")
            policy_logits = policy_logits / max(float(self.config.temperature), 1e-3)
            log_probs = F.log_softmax(policy_logits.masked_fill(~states["action_mask"], -1e9), dim=-1)
            chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            values = values.squeeze(-1)

            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            policy_loss = -(chosen_log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            if aux_predictions is not None:
                aux_loss = F.mse_loss(aux_predictions, aux_targets)
            else:
                aux_loss = torch.tensor(0.0, dtype=torch.float32, device=value_loss.device)
            total_loss = (
                policy_loss
                + float(self.config.value_loss_coef) * value_loss
                + float(self.ppo_hparams.aux_coef) * aux_loss
                - float(self.config.entropy_coef) * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.max_grad_norm))
            self.optimizer.step()
            self.network.eval()

            logger.info(
                f"Agent {self.id[:8]} self-play update: "
                f"steps={len(steps)}, reward={terminal_reward:.3f}, shaped_sum={sum(step_rewards):.3f}, "
                f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, aux_loss={aux_loss.item():.4f}"
            )
    
    def get_fitness_score(self) -> float:
        """Calculate fitness score for evolutionary selection"""
        if self.games_played == 0:
            return 0.0
        
        avg_vp = self.total_victory_points / self.games_played
        win_rate = self.wins / self.games_played

        # Dashboard/summary fitness: keep configurable, defaulting to win-rate emphasis.
        try:
            vp_weight = float(os.getenv("AGENT_FITNESS_VP_WEIGHT", "0.45"))
        except Exception:
            vp_weight = 0.45
        try:
            win_weight = float(os.getenv("AGENT_FITNESS_WIN_WEIGHT", "0.55"))
        except Exception:
            win_weight = 0.55
        vp_weight = max(0.0, vp_weight)
        win_weight = max(0.0, win_weight)
        denom = max(1e-6, vp_weight + win_weight)
        vp_w = vp_weight / denom
        win_w = win_weight / denom
        fitness = avg_vp * vp_w + win_rate * 100 * win_w
        return fitness

    def get_behavior_stats(self) -> Dict[str, Any]:
        """Return action-decision telemetry used by the monitoring dashboard."""
        total_decisions = int(self.decision_stats.get('total_decisions', 0))
        policy_attempts = int(self.decision_stats.get('policy_attempts', 0))
        policy_successes = int(self.decision_stats.get('policy_successes', 0))
        policy_rejections = int(self.decision_stats.get('policy_rejections', 0))
        policy_sampled_actions = int(self.decision_stats.get('policy_sampled_actions', 0))
        epsilon_random_actions = int(self.decision_stats.get('epsilon_random_actions', 0))
        fallback_decisions = int(self.decision_stats.get('fallback_decisions', 0))
        fallback_random_attempts = int(self.decision_stats.get('fallback_random_attempts', 0))
        fallback_random_successes = int(self.decision_stats.get('fallback_random_successes', 0))
        fallback_passes = int(self.decision_stats.get('fallback_passes', 0))
        no_available_actions = int(self.decision_stats.get('no_available_actions', 0))
        sum_available_actions = int(self.decision_stats.get('sum_available_actions', 0))
        available_action_observations = int(self.decision_stats.get('available_action_observations', 0))
        card_play_actions = int(self.decision_stats.get('card_play_actions', 0))
        steel_spent = int(self.decision_stats.get('steel_spent', 0))
        titanium_spent = int(self.decision_stats.get('titanium_spent', 0))
        project_payment_value_total = float(self.decision_stats.get('project_payment_value_total', 0.0) or 0.0)
        metal_payment_value_total = float(self.decision_stats.get('metal_payment_value_total', 0.0) or 0.0)
        steel_payment_value_total = float(self.decision_stats.get('steel_payment_value_total', 0.0) or 0.0)
        titanium_payment_value_total = float(self.decision_stats.get('titanium_payment_value_total', 0.0) or 0.0)
        rare_state_samples = int(self.decision_stats.get('rare_state_samples', 0))
        rare_award_funding = int(self.decision_stats.get('rare_award_funding', 0))
        rare_milestone_timing = int(self.decision_stats.get('rare_milestone_timing', 0))
        rare_draft_keep_buy = int(self.decision_stats.get('rare_draft_keep_buy', 0))
        rare_high_cost_payment = int(self.decision_stats.get('rare_high_cost_payment', 0))
        hate_draft_picks = int(self.decision_stats.get('hate_draft_picks', 0))
        draft_decisions_total = int(self.decision_stats.get('draft_decisions_total', 0))
        draft_decisions_low_hand_ev = int(self.decision_stats.get('draft_decisions_low_hand_ev', 0))
        hate_draft_picks_low_hand_ev = int(self.decision_stats.get('hate_draft_picks_low_hand_ev', 0))
        milestone_snipes = int(self.decision_stats.get('milestone_snipes', 0))
        award_snipes = int(self.decision_stats.get('award_snipes', 0))
        action_mask_observations = int(self.decision_stats.get('action_mask_observations', 0))
        action_legal_count_total = int(self.decision_stats.get('action_legal_count_total', 0))
        action_rejected_by_server = int(self.decision_stats.get('action_rejected_by_server', 0))
        policy_actions_blocked_by_reject_cache = int(self.decision_stats.get('policy_actions_blocked_by_reject_cache', 0))
        timing_totals_sec = dict(self.decision_stats.get("timing_totals_sec", {}) or {})
        timing_counts = dict(self.decision_stats.get("timing_counts", {}) or {})
        timing_log_events = int(self.decision_stats.get("timing_log_events", 0) or 0)

        def _ratio(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator > 0 else 0.0

        action_counts = dict(self.decision_stats.get('action_type_counts', {}))
        standard_project_counts = dict(self.decision_stats.get('standard_project_counts', {}))
        total_recorded_actions = sum(int(v) for v in action_counts.values())
        action_mix = {
            key: _ratio(int(value), total_recorded_actions)
            for key, value in action_counts.items()
        }
        total_standard_projects = sum(int(v) for v in standard_project_counts.values())
        standard_project_mix = {
            key: _ratio(int(value), total_standard_projects)
            for key, value in standard_project_counts.items()
        }

        return {
            'total_decisions': total_decisions,
            'policy_attempts': policy_attempts,
            'policy_successes': policy_successes,
            'policy_rejections': policy_rejections,
            'policy_sampled_actions': policy_sampled_actions,
            'epsilon_random_actions': epsilon_random_actions,
            'fallback_decisions': fallback_decisions,
            'fallback_random_attempts': fallback_random_attempts,
            'fallback_random_successes': fallback_random_successes,
            'fallback_passes': fallback_passes,
            'no_available_actions': no_available_actions,
            'avg_available_actions': _ratio(sum_available_actions, available_action_observations),
            'policy_success_rate': _ratio(policy_successes, policy_attempts),
            'policy_sample_rate': _ratio(policy_sampled_actions, total_decisions),
            'epsilon_random_rate': _ratio(epsilon_random_actions, total_decisions),
            'fallback_decision_rate': _ratio(fallback_decisions, total_decisions),
            'fallback_random_success_rate': _ratio(fallback_random_successes, fallback_random_attempts),
            'fallback_pass_rate': _ratio(fallback_passes, total_decisions),
            'card_play_actions': card_play_actions,
            'card_plays_per_game': _ratio(card_play_actions, int(self.games_played)),
            'steel_spent': steel_spent,
            'steel_spent_per_game': _ratio(steel_spent, int(self.games_played)),
            'titanium_spent': titanium_spent,
            'titanium_spent_per_game': _ratio(titanium_spent, int(self.games_played)),
            'project_payment_value_total': float(project_payment_value_total),
            'metal_payment_value_total': float(metal_payment_value_total),
            'steel_payment_value_total': float(steel_payment_value_total),
            'titanium_payment_value_total': float(titanium_payment_value_total),
            'project_payment_value_per_game': (float(project_payment_value_total) / float(self.games_played)) if int(self.games_played) > 0 else 0.0,
            'metal_payment_value_per_game': (float(metal_payment_value_total) / float(self.games_played)) if int(self.games_played) > 0 else 0.0,
            'metal_conversion_efficiency': (float(metal_payment_value_total) / float(project_payment_value_total)) if project_payment_value_total > 0.0 else 0.0,
            'rare_state_samples': rare_state_samples,
            'rare_state_rate': _ratio(rare_state_samples, total_decisions),
            'rare_award_funding': rare_award_funding,
            'rare_milestone_timing': rare_milestone_timing,
            'rare_draft_keep_buy': rare_draft_keep_buy,
            'rare_high_cost_payment': rare_high_cost_payment,
            'hate_draft_picks': hate_draft_picks,
            'draft_decisions_total': draft_decisions_total,
            'draft_decisions_low_hand_ev': draft_decisions_low_hand_ev,
            'hate_draft_picks_low_hand_ev': hate_draft_picks_low_hand_ev,
            'hate_draft_rate': _ratio(hate_draft_picks, draft_decisions_total),
            'hate_draft_rate_low_hand_ev': _ratio(hate_draft_picks_low_hand_ev, draft_decisions_low_hand_ev),
            'milestone_snipes': milestone_snipes,
            'award_snipes': award_snipes,
            'action_legal_count_mean': _ratio(action_legal_count_total, action_mask_observations),
            'action_mask_coverage_rate': _ratio(action_mask_observations, total_decisions),
            'action_rejected_by_server': action_rejected_by_server,
            'policy_actions_blocked_by_reject_cache': policy_actions_blocked_by_reject_cache,
            'reject_cache_entries': int(len(self._rejected_actions_by_prompt)),
            'rollout_buffer_size': int(len(self.rollout_buffer)),
            'action_counts': action_counts,
            'action_mix': action_mix,
            'standard_project_counts': standard_project_counts,
            'standard_project_mix': standard_project_mix,
            'timing_totals_sec': timing_totals_sec,
            'timing_counts': timing_counts,
            'timing_log_events': timing_log_events,
            'timing_avg_ms': {
                key: ((float(timing_totals_sec.get(key, 0.0) or 0.0) * 1000.0) / float(max(1, int(timing_counts.get(key, 0) or 0))))
                for key in timing_totals_sec.keys()
            },
        }
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate network weights for evolutionary training"""
        with torch.no_grad():
            for param in self.network.parameters():
                if np.random.random() < mutation_rate:
                    # Add Gaussian noise
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
    
    def crossover(self, other_agent: 'RLAgent') -> 'RLAgent':
        """Create offspring by crossing over with another agent"""
        max_epsilon = max(0.0, self._safe_env_float("EVOLUTION_MAX_EPSILON", 0.12))
        min_epsilon = max(0.0, self._safe_env_float("EVOLUTION_MIN_EPSILON", 0.001))
        max_temperature = max(1e-3, self._safe_env_float("EVOLUTION_MAX_TEMPERATURE", 1.2))
        min_temperature = max(1e-3, self._safe_env_float("EVOLUTION_MIN_TEMPERATURE", 0.6))
        sampled_epsilon = np.random.uniform(
            min(self.config.epsilon, other_agent.config.epsilon),
            max(self.config.epsilon, other_agent.config.epsilon),
        )
        sampled_temperature = np.random.uniform(
            min(self.config.temperature, other_agent.config.temperature),
            max(self.config.temperature, other_agent.config.temperature),
        )
        clamped_epsilon = float(max(min_epsilon, min(max_epsilon, sampled_epsilon)))
        clamped_temperature = float(max(min_temperature, min(max_temperature, sampled_temperature)))

        parent_a = asdict(self.config)
        parent_b = asdict(other_agent.config)
        child_payload: Dict[str, Any] = {}
        for key, default_value in asdict(AgentConfig()).items():
            if key in ("learning_rate", "epsilon", "temperature"):
                continue
            pick_a = parent_a.get(key, default_value)
            pick_b = parent_b.get(key, default_value)
            child_payload[key] = pick_a if random.random() < 0.5 else pick_b
        child_payload["learning_rate"] = float(np.random.choice([self.config.learning_rate, other_agent.config.learning_rate]))
        child_payload["epsilon"] = clamped_epsilon
        child_payload["temperature"] = clamped_temperature
        child_config = AgentConfig(**child_payload)
        
        child = RLAgent(child_config)
        
        # Mix network weights
        with torch.no_grad():
            for child_param, parent1_param, parent2_param in zip(
                child.network.parameters(),
                self.network.parameters(),
                other_agent.network.parameters()
            ):
                # Random crossover point for each parameter
                crossover_mask = torch.rand_like(child_param) < 0.5
                child_param.data = torch.where(crossover_mask, parent1_param.data, parent2_param.data)
        
        return child
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'games_played': self.games_played,
            'total_victory_points': self.total_victory_points,
            'wins': self.wins
        }, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # Our checkpoints include optimizer/config metadata, so we need full unpickling
        # for trusted local artifacts.
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Backward compatibility for torch versions without weights_only argument.
            checkpoint = torch.load(path, map_location='cpu')

        # Restore saved AgentConfig so policy behavior and training hyperparameters
        # remain consistent after resume/load.
        config_payload = checkpoint.get('config', {})
        if isinstance(config_payload, dict):
            defaults = asdict(AgentConfig())
            merged_config = defaults.copy()
            for key, value in config_payload.items():
                if key in defaults:
                    merged_config[key] = value
            # Compatibility-only mapping for older checkpoint/config payloads.
            legacy_config_map = {
                "tableau_token_count": "planner_tableau_limit",
                "hand_token_count": "planner_hand_limit",
                "opponent_token_count": "planner_opponent_limit",
                "card_token_dim": "planner_token_dim",
                "state_size": None,
                "num_layers": None,
                "transformer_enabled": None,
                "transformer_embed_dim": None,
            }
            for legacy_key, new_key in legacy_config_map.items():
                if legacy_key not in config_payload:
                    continue
                if new_key and new_key in defaults:
                    merged_config[new_key] = config_payload[legacy_key]
            token_count_envs = {
                "planner_tableau_limit": "AGENT_PLANNER_TABLEAU_LIMIT",
                "planner_hand_limit": "AGENT_PLANNER_HAND_LIMIT",
                "planner_opponent_limit": "AGENT_PLANNER_OPPONENT_LIMIT",
            }
            applied_token_overrides: Dict[str, int] = {}
            for key, env_name in token_count_envs.items():
                raw_value = str(os.getenv(env_name, "")).strip()
                if not raw_value:
                    continue
                merged_config[key] = self._safe_env_int(env_name, int(merged_config.get(key, defaults[key])))
                applied_token_overrides[key] = int(merged_config[key])
            self.config = AgentConfig(**merged_config)
            if applied_token_overrides:
                logger.info(
                    "Applied token-count env override(s) while loading %s: %s",
                    path,
                    applied_token_overrides,
                )

        # Rebuild model/optimizer from restored config before loading state dicts.
        self.network = TerraformingMarsNetwork(self.config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.ppo_learning_rate)
        planner_config = self.config.planner_config()
        self.state_encoder = StateEncoder(planner_config=planner_config)
        self.action_decoder = ActionDecoder(planner_config=planner_config)

        state_dict = checkpoint.get('network_state_dict', {}) or {}
        
        # Handle backward compatibility: old checkpoints had 5 aux outputs, new model has 74
        old_aux_weight_key = 'aux_head.2.weight'
        old_aux_bias_key = 'aux_head.2.bias'
        if old_aux_weight_key in state_dict and old_aux_bias_key in state_dict:
            old_weight = state_dict[old_aux_weight_key]
            old_bias = state_dict[old_aux_bias_key]
            old_size = old_weight.shape[0]  # Number of outputs in old checkpoint
            
            current_aux_weight = self.network.aux_head[2].weight
            current_aux_bias = self.network.aux_head[2].bias
            current_size = current_aux_weight.shape[0]  # Should be 74
            
            if old_size != current_size:
                logger.info(
                    "Detected aux head size mismatch: checkpoint has %d outputs, current model has %d. "
                    "Migrating old checkpoint...",
                    old_size, current_size
                )
                
                # Create new weight and bias tensors with current model size
                new_weight = current_aux_weight.clone()  # Initialize with current random weights
                new_bias = current_aux_bias.clone()
                
                # Map old outputs to new positions:
                # Old format: [milestone_claimability (scalar), award_ev, playable_cards, steel_target, titanium_target]
                # New format: [milestone_claimability (70), award_ev, playable_cards, steel_target, titanium_target]
                # So indices 1-4 map to indices 70-73 in new format
                if old_size == 5:
                    # Copy old outputs to appropriate positions
                    # milestone_claimability (old index 0) -> initialize first milestone (index 0) with old value
                    # For milestones, we'll initialize with a small value based on the old scalar
                    milestone_init_value = old_weight[0:1, :].mean(dim=0, keepdim=True)
                    new_weight[0:70, :] = milestone_init_value.expand(70, -1) * 0.1  # Scale down for milestones
                    
                    # Copy other 4 outputs to positions 70-73
                    new_weight[70:74, :] = old_weight[1:5, :]
                    new_bias[70:74] = old_bias[1:5]
                    
                    # Initialize milestone biases with small values
                    new_bias[0:70] = old_bias[0:1].expand(70) * 0.1
                    
                    logger.info(
                        "Migrated aux head: old milestone scalar -> 70 milestone outputs, "
                        "old outputs [1-4] -> new outputs [70-73]"
                    )
                else:
                    # Unknown old format, just copy what we can
                    copy_size = min(old_size, current_size)
                    new_weight[:copy_size, :] = old_weight[:copy_size, :]
                    new_bias[:copy_size] = old_bias[:copy_size]
                    logger.warning(
                        "Unknown aux head format (old_size=%d). Copied first %d outputs.",
                        old_size, copy_size
                    )
                
                # Replace in state_dict
                state_dict[old_aux_weight_key] = new_weight
                state_dict[old_aux_bias_key] = new_bias

        # Backward compatibility: older checkpoints stored value head as
        # value_head.{0,2}.* (Sequential). The current model uses value_trunk +
        # value_head linear output, so drop legacy keys to avoid noisy warnings.
        legacy_value_head_keys: List[str] = []
        for key in list(state_dict.keys()):
            if not key.startswith("value_head."):
                continue
            key_parts = key.split(".")
            if len(key_parts) >= 3 and key_parts[1].isdigit():
                legacy_value_head_keys.append(key)
        if legacy_value_head_keys:
            for key in legacy_value_head_keys:
                state_dict.pop(key, None)
            logger.info(
                "Dropped %d legacy value-head parameter(s) from %s to load new value architecture.",
                len(legacy_value_head_keys),
                path,
            )
        
        load_result = self.network.load_state_dict(state_dict, strict=False)
        missing = list(getattr(load_result, "missing_keys", []) or [])
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if missing or unexpected:
            # Keys that are legitimately absent from older checkpoints and will
            # fall back to their default initialisation values safely.
            expected_missing_keys = {"action_type_bias"}
            expected_missing_prefixes = (
                "card_attention_module.",
                "transformer_fusion.",
                "value_trunk.",
                "value_head.",
            )
            expected_missing = [
                key for key in missing
                if key in expected_missing_keys
                or any(key.startswith(prefix) for prefix in expected_missing_prefixes)
            ]
            if not unexpected and expected_missing and len(expected_missing) == len(missing):
                logger.info(
                    "Loaded checkpoint %s; %d param(s) initialised from defaults: %s",
                    path,
                    len(missing),
                    missing,
                )
            else:
                logger.warning(
                    "Model load used non-strict mode for %s (missing=%d unexpected=%d)",
                    path,
                    len(missing),
                    len(unexpected),
                )

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state:
            saved_group_sizes: List[int] = []
            current_group_sizes: List[int] = []
            try:
                saved_groups = optimizer_state.get("param_groups", []) if isinstance(optimizer_state, dict) else []
                current_groups = self.optimizer.state_dict().get("param_groups", [])
                saved_group_sizes = [
                    int(len(group.get("params", [])))
                    for group in saved_groups
                    if isinstance(group, dict)
                ]
                current_group_sizes = [
                    int(len(group.get("params", [])))
                    for group in current_groups
                    if isinstance(group, dict)
                ]
            except Exception:
                saved_group_sizes = []
                current_group_sizes = []

            if saved_group_sizes and current_group_sizes and saved_group_sizes != current_group_sizes:
                logger.info(
                    "Skipping optimizer restore for %s due to parameter-group mismatch (saved=%s current=%s).",
                    path,
                    saved_group_sizes,
                    current_group_sizes,
                )
            else:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as e:
                    logger.warning(f"Failed to restore optimizer state from {path}: {e}")
        # Keep PPO optimizer LR env-driven even after restoring optimizer state.
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.ppo_learning_rate)

        self.network.eval()
        self._inference_device = _resolve_inference_device()
        self._move_network_to_inference_device()
        if self._inference_batcher is not None:
            self._inference_batcher.shutdown()
        self._inference_batcher = None
        self._init_inference_batcher()

        self.train_from_self_play = (
            self.config.train_from_self_play and os.getenv("SELF_PLAY_LEARNING", "1") != "0"
        )
        self.games_played = int(checkpoint.get('games_played', 0))
        self.total_victory_points = int(checkpoint.get('total_victory_points', 0))
        self.wins = int(checkpoint.get('wins', 0))
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            'id': self.id,
            'config': asdict(self.config),
            'games_played': self.games_played,
            'total_victory_points': self.total_victory_points,
            'wins': self.wins,
            'fitness_score': self.get_fitness_score()
        }
    

    
