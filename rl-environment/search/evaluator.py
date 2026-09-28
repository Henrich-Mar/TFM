"""Neural position evaluation for search, outside the live decision path.

Wraps the agent's encoder/decoder/network so any ``PlayerViewModel`` (root,
opponent continuation, or leaf) yields action priors, a value estimate, and a
next recurrent state.  Nothing here mutates live agent memory: branches carry
their own recurrent maps, and all forwards run under ``torch.no_grad``.
"""
from __future__ import annotations

import logging
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from models.planner_common import pad_bundle_batch

logger = logging.getLogger("rl.search")


class EvaluationError(RuntimeError):
    pass


@dataclass
class EvalItem:
    player_state: Dict[str, Any]
    player_id: str
    turn_count: int = 0
    recurrent_in: Optional[torch.Tensor] = None


@dataclass
class EvalResult:
    descriptors: List[Dict[str, Any]]
    probabilities: List[float]
    value: float
    recurrent_out: torch.Tensor
    phase_index: int


class PositionEvaluator:
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache: OrderedDict[str, EvalResult] = OrderedDict()
        self._cache_limit = 512

    def _cache_key(self, item: EvalItem) -> str:
        digest = hashlib.sha256()
        digest.update(str(item.player_id).encode("utf-8", errors="replace"))
        digest.update(str(int(item.turn_count)).encode("ascii"))
        digest.update(
            json.dumps(item.player_state, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8", errors="replace"
            )
        )
        if isinstance(item.recurrent_in, torch.Tensor):
            digest.update(item.recurrent_in.detach().float().cpu().reshape(-1).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _clone_result(result: EvalResult) -> EvalResult:
        return EvalResult(
            descriptors=list(result.descriptors),
            probabilities=list(result.probabilities),
            value=float(result.value),
            recurrent_out=result.recurrent_out.clone() if isinstance(result.recurrent_out, torch.Tensor) else result.recurrent_out,
            phase_index=int(result.phase_index),
        )

    def _remember(self, key: str, result: EvalResult) -> None:
        self._cache[key] = self._clone_result(result)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # recurrent memory helpers (branch-local copies of live memory)
    # ------------------------------------------------------------------

    def zero_recurrent(self) -> torch.Tensor:
        return self.agent._zero_recurrent_state()

    def clone_recurrent_map(self, player_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
        memory: Dict[str, torch.Tensor] = {}
        for player_id in player_ids:
            key = str(player_id)
            memory[key] = self.agent._get_recurrent_state_for_player(key).reshape(-1).clone()
        return memory

    @staticmethod
    def memory_for(memory: Dict[str, torch.Tensor], player_id: str) -> Optional[torch.Tensor]:
        cached = memory.get(str(player_id))
        if isinstance(cached, torch.Tensor):
            return cached.reshape(-1)
        return None

    @staticmethod
    def update_memory(
        memory: Dict[str, torch.Tensor],
        player_id: str,
        recurrent_out: Optional[torch.Tensor],
    ) -> None:
        if recurrent_out is None:
            return
        memory[str(player_id)] = recurrent_out.detach().float().reshape(-1)

    # ------------------------------------------------------------------
    # forwards
    # ------------------------------------------------------------------

    def _encode(self, item: EvalItem) -> Optional[Dict[str, Any]]:
        agent = self.agent
        try:
            descriptors = agent.action_decoder.get_legal_action_descriptors(item.player_state)
        except Exception as exc:
            logger.debug("search evaluation: illegal prompt rejected: %s", exc)
            return None
        if not descriptors:
            return None
        try:
            phase_index = int(agent._extract_phase_index(item.player_state))
            bundle = agent.state_encoder.encode(item.player_state, int(item.turn_count), descriptors)
        except Exception as exc:
            logger.warning("search evaluation: encode failed: %s", exc)
            return None
        return {
            "descriptors": descriptors,
            "phase_index": phase_index,
            "bundle": bundle,
            "item": item,
        }

    def evaluate(self, item: EvalItem) -> EvalResult:
        results = self.evaluate_batch([item])
        if not results or results[0] is None:
            raise EvaluationError("position could not be evaluated")
        return results[0]

    def evaluate_batch(self, items: Sequence[EvalItem]) -> List[Optional[EvalResult]]:
        """Evaluate a batch of positions. ``None`` marks items that produced no
        legal action set (those branches are dropped by the caller)."""
        results: List[Optional[EvalResult]] = [None] * len(items)
        rows: List[tuple] = []
        duplicate_indices: Dict[str, List[int]] = {}
        for index, item in enumerate(items):
            key = self._cache_key(item)
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                results[index] = self._clone_result(cached)
                continue
            if key in duplicate_indices:
                self.cache_hits += 1
                duplicate_indices[key].append(index)
                continue
            self.cache_misses += 1
            duplicate_indices[key] = [index]
            encoded = self._encode(item)
            if encoded is not None:
                rows.append((index, key, encoded))
        if not rows:
            return results
        try:
            outputs = self._forward_batch([row for _, _, row in rows])
            for (index, key, _), output in zip(rows, outputs):
                self._remember(key, output)
                for duplicate_index in duplicate_indices[key]:
                    results[duplicate_index] = self._clone_result(output)
            return results
        except Exception as exc:
            logger.debug("batched search forward failed (%s); falling back to per-item", exc)
        for index, key, row in rows:
            try:
                output = self._forward_batch([row])[0]
                self._remember(key, output)
                for duplicate_index in duplicate_indices[key]:
                    results[duplicate_index] = self._clone_result(output)
            except Exception as fallback_exc:
                logger.debug("search evaluation forward failed for one leaf: %s", fallback_exc)
        return results

    def _forward_batch(self, rows: List[Dict[str, Any]]) -> List[EvalResult]:
        agent = self.agent
        device = agent._inference_device
        recurrent_size = max(
            1,
            int(getattr(agent.network, "recurrent_size", max(16, agent.config.hidden_size // 2))),
        )
        bundles = [row["bundle"] for row in rows]
        counts = [len(row["descriptors"]) for row in rows]
        phases = torch.tensor([int(row["phase_index"]) for row in rows], dtype=torch.long, device=device)
        recurrent_rows: List[torch.Tensor] = []
        for row in rows:
            item = row["item"]
            rec = item.recurrent_in
            if rec is None:
                vec = torch.zeros((recurrent_size,), dtype=torch.float32, device=device)
            else:
                vec = rec.detach().float().reshape(-1)
                if vec.device != device:
                    vec = vec.to(device)
                if int(vec.numel()) != recurrent_size:
                    padded = torch.zeros((recurrent_size,), dtype=torch.float32, device=device)
                    take = min(int(vec.numel()), recurrent_size)
                    padded[:take] = vec[:take]
                    vec = padded
            recurrent_rows.append(vec)
        recurrent_batch = torch.stack(recurrent_rows, dim=0)

        with agent._model_device_lock:
            batch = pad_bundle_batch(
                bundles,
                device=device,
                planner_config=agent.config.planner_config(),
            )
            with torch.no_grad():
                use_amp = getattr(device, "type", "cpu") == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = agent._forward_network(
                        batch,
                        phase_indices=phases,
                        recurrent_state=recurrent_batch,
                    )
                policy_logits = out["policy_logits"].float()
                value = out["value"].float()
                recurrent_out = out.get("recurrent_state")
                if recurrent_out is not None:
                    recurrent_out = recurrent_out.float()
        if getattr(device, "type", "cpu") != "cpu":
            policy_logits = policy_logits.cpu()
            value = value.cpu()
            if recurrent_out is not None:
                recurrent_out = recurrent_out.cpu()

        outputs: List[EvalResult] = []
        for position, row in enumerate(rows):
            action_count = int(counts[position])
            logits_row = policy_logits[position, :action_count]
            probabilities = F.softmax(logits_row, dim=-1).tolist()
            row_value = float(value[position].reshape(-1)[0].item())
            row_recurrent = (
                recurrent_out[position].reshape(-1).clone()
                if recurrent_out is not None
                else None
            )
            outputs.append(
                EvalResult(
                    descriptors=row["descriptors"],
                    probabilities=[float(p) for p in probabilities],
                    value=row_value,
                    recurrent_out=row_recurrent,
                    phase_index=int(row["phase_index"]),
                )
            )
        return outputs
