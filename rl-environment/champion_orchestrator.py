#!/usr/bin/env python3
"""
Global champion orchestrator for multi-coordinator training setups.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from game_interface import GameServerCluster
from models.agent import RLAgent
from tournament_manager import TournamentManager


LOG_LEVEL_NAME = str(os.getenv("LOG_LEVEL", "INFO")).strip().upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("champion_orchestrator")


@dataclass
class CandidateRef:
    candidate_id: str
    coordinator_id: str
    source_type: str  # coord | incumbent
    coord_root: str
    generation: int
    checkpoint_path: str
    checkpoint_relpath: str


@dataclass
class CoordProgress:
    coordinator_id: str
    coord_root: str
    next_generation: Optional[int]
    latest_saved_generation: Optional[int]
    generation_signal: int


@dataclass
class OrchestratorConfig:
    coord_sources: Dict[str, str]
    trainer_coord_id: str
    output_root: str
    poll_interval_sec: float
    trigger_every_n_gens: int
    top_k_per_coord: int
    games_per_candidate: int
    global_game_concurrency: int
    tournament_concurrency: int
    min_games_for_promotion: int
    min_completion_rate: float
    win_rate_margin: float
    keep_generations_trainer: int
    keep_generations_worker: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _checkpoint_fitness_from_name(path: str) -> float:
    base = os.path.basename(path)
    try:
        return float(base.split("_fitness_")[-1].replace(".pth", ""))
    except Exception:
        return -1.0


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_coord_sources(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in [part.strip() for part in str(raw or "").split(",") if part.strip()]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        coord_id = key.strip()
        coord_root = os.path.abspath(value.strip())
        if not coord_id or not coord_root:
            continue
        out[coord_id] = coord_root
    return out


def _available_generations(models_root: str) -> List[int]:
    generations: List[int] = []
    for path in glob.glob(os.path.join(models_root, "generation_*")):
        base = os.path.basename(path)
        try:
            generations.append(int(base.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(generations))


def _all_generation_checkpoints(models_root: str, generation: int) -> List[str]:
    pattern = os.path.join(models_root, f"generation_{generation}", "agent_*_fitness_*.pth")
    return sorted(glob.glob(pattern), key=_checkpoint_fitness_from_name, reverse=True)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        logger.warning("Failed reading JSON %s", path, exc_info=True)
    return None


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


def _atomic_copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp.{uuid.uuid4().hex}"
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def discover_coord_progress(coord_id: str, coord_root: str) -> CoordProgress:
    state_path = os.path.join(coord_root, "checkpoints", "state.json")
    state = _load_json(state_path) or {}
    next_generation: Optional[int] = None
    if "next_generation" in state:
        next_generation = _safe_int(state.get("next_generation"), -1)
        if next_generation < 0:
            next_generation = None

    generations = _available_generations(coord_root)
    latest_saved_generation = generations[-1] if generations else None
    if next_generation is not None:
        generation_signal = int(next_generation)
    elif latest_saved_generation is not None:
        generation_signal = int(latest_saved_generation) + 1
    else:
        generation_signal = -1
    return CoordProgress(
        coordinator_id=str(coord_id),
        coord_root=str(coord_root),
        next_generation=next_generation,
        latest_saved_generation=latest_saved_generation,
        generation_signal=generation_signal,
    )


def discover_candidates_for_coord(coord_id: str, coord_root: str, top_k: int) -> List[CandidateRef]:
    candidates: List[CandidateRef] = []
    generations = list(reversed(_available_generations(coord_root)))
    for generation in generations:
        checkpoints = _all_generation_checkpoints(coord_root, generation)
        for checkpoint_path in checkpoints:
            abs_path = os.path.abspath(checkpoint_path)
            rel_path = os.path.relpath(abs_path, coord_root).replace("\\", "/")
            candidate_id = f"{coord_id}:g{generation}:{os.path.basename(abs_path)}"
            candidates.append(
                CandidateRef(
                    candidate_id=candidate_id,
                    coordinator_id=str(coord_id),
                    source_type="coord",
                    coord_root=os.path.abspath(coord_root),
                    generation=int(generation),
                    checkpoint_path=abs_path,
                    checkpoint_relpath=rel_path,
                )
            )
            if len(candidates) >= max(1, int(top_k)):
                return candidates
    return candidates


def rank_candidate_metrics(candidate_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for candidate_id, metrics in candidate_metrics.items():
        games_completed = max(0, int(metrics.get("games_completed", 0)))
        wins = max(0, int(metrics.get("wins", 0)))
        vp_sum = float(metrics.get("vp_sum", 0.0) or 0.0)
        rank_sum = float(metrics.get("rank_sum", 0.0) or 0.0)
        win_rate = float(wins) / float(games_completed) if games_completed > 0 else 0.0
        avg_rank = float(rank_sum) / float(games_completed) if games_completed > 0 else 999.0
        avg_vp = float(vp_sum) / float(games_completed) if games_completed > 0 else 0.0
        enriched = dict(metrics)
        enriched.update(
            {
                "candidate_id": str(candidate_id),
                "win_rate": float(win_rate),
                "avg_rank": float(avg_rank),
                "avg_vp": float(avg_vp),
            }
        )
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            -float(item.get("win_rate", 0.0)),
            float(item.get("avg_rank", 999.0)),
            -float(item.get("avg_vp", 0.0)),
            str(item.get("candidate_id", "")),
        )
    )
    return ranked


def decide_promotion(
    ranked_candidates: Sequence[Dict[str, Any]],
    *,
    incumbent_candidate_id: Optional[str],
    completed_games: int,
    completion_rate: float,
    min_games_for_promotion: int,
    min_completion_rate: float,
    win_rate_margin: float,
) -> Dict[str, Any]:
    if not ranked_candidates:
        return {
            "applied": False,
            "winner_candidate_id": None,
            "reason": "no_candidates",
        }

    leader = dict(ranked_candidates[0])
    leader_id = str(leader.get("candidate_id", ""))
    leader_win_rate = float(leader.get("win_rate", 0.0))

    if completed_games < int(min_games_for_promotion):
        return {
            "applied": False,
            "winner_candidate_id": str(incumbent_candidate_id or leader_id),
            "reason": "insufficient_completed_games",
        }
    if float(completion_rate) < float(min_completion_rate):
        return {
            "applied": False,
            "winner_candidate_id": str(incumbent_candidate_id or leader_id),
            "reason": "insufficient_completion_rate",
        }

    if not incumbent_candidate_id:
        return {
            "applied": True,
            "winner_candidate_id": leader_id,
            "reason": "bootstrap_no_incumbent",
        }

    incumbent = next(
        (item for item in ranked_candidates if str(item.get("candidate_id", "")) == str(incumbent_candidate_id)),
        None,
    )
    incumbent_win_rate = float((incumbent or {}).get("win_rate", 0.0))
    required_win_rate = float(incumbent_win_rate) + float(win_rate_margin)

    if leader_id == str(incumbent_candidate_id):
        return {
            "applied": False,
            "winner_candidate_id": str(incumbent_candidate_id),
            "reason": "incumbent_still_best",
        }
    if leader_win_rate >= required_win_rate:
        return {
            "applied": True,
            "winner_candidate_id": leader_id,
            "reason": "margin_passed",
            "required_win_rate": float(required_win_rate),
        }
    return {
        "applied": False,
        "winner_candidate_id": str(incumbent_candidate_id),
        "reason": "margin_not_met",
        "required_win_rate": float(required_win_rate),
    }


def _candidate_by_id(candidates: Sequence[CandidateRef]) -> Dict[str, CandidateRef]:
    return {str(candidate.candidate_id): candidate for candidate in candidates}


def _load_template_agent(candidate: CandidateRef) -> RLAgent:
    agent = RLAgent()
    agent.load_model(candidate.checkpoint_path)
    agent.id = str(candidate.candidate_id)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    return agent


def _clone_template_agent(template: RLAgent, candidate_id: str) -> RLAgent:
    clone = RLAgent(template.config)
    clone.network.load_state_dict(template.network.state_dict())
    clone.network.eval()
    clone.id = str(candidate_id)
    clone.train_from_self_play = False
    clone.config.train_from_self_play = False
    return clone


async def evaluate_candidates(
    candidates: Sequence[CandidateRef],
    *,
    game_servers: Sequence[str],
    games_per_candidate: int,
    global_game_concurrency: int,
    round_seed: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    if not candidates:
        return [], 0, 0

    candidate_map = _candidate_by_id(candidates)
    candidate_ids = sorted(candidate_map.keys())
    metrics: Dict[str, Dict[str, Any]] = {
        candidate_id: {
            "games_total": 0,
            "games_completed": 0,
            "wins": 0,
            "vp_sum": 0.0,
            "rank_sum": 0.0,
        }
        for candidate_id in candidate_ids
    }

    templates: Dict[str, RLAgent] = {}
    for candidate_id in candidate_ids:
        templates[candidate_id] = _load_template_agent(candidate_map[candidate_id])

    total_planned_games = len(candidate_ids) * max(1, int(games_per_candidate))
    successful_games = 0
    global_semaphore = asyncio.Semaphore(max(1, int(global_game_concurrency)))

    async def _run_single_eval_game(lineup_candidate_ids: List[str], tournament: TournamentManager) -> Any:
        async with global_semaphore:
            lineup_agents = [
                _clone_template_agent(templates[candidate_id], candidate_id=candidate_id)
                for candidate_id in lineup_candidate_ids
            ]
            return await tournament._run_single_game(
                lineup_agents,
                tournament_id=f"orch_{uuid.uuid4().hex[:10]}",
            )

    async with GameServerCluster(list(game_servers)) as cluster:
        await cluster.health_check()
        tournament = TournamentManager(cluster)
        tasks: List[asyncio.Task] = []
        for candidate_id in candidate_ids:
            opponents = [token for token in candidate_ids if token != candidate_id]
            for game_index in range(max(1, int(games_per_candidate))):
                rng = random.Random(f"{round_seed}:{candidate_id}:{game_index}")
                sampled: List[str] = []
                if opponents:
                    if len(opponents) >= 3:
                        sampled.extend(rng.sample(opponents, k=3))
                    else:
                        sampled.extend(opponents)
                while len(sampled) < 3:
                    sampled.append(rng.choice(opponents or [candidate_id]))
                lineup_ids = [candidate_id] + sampled[:3]
                rng.shuffle(lineup_ids)
                tasks.append(asyncio.create_task(_run_single_eval_game(lineup_ids, tournament)))

        for task in asyncio.as_completed(tasks):
            try:
                game_result = await task
            except Exception:
                logger.warning("Champion evaluation game failed", exc_info=True)
                continue

            if bool(getattr(game_result, "completed", False)):
                successful_games += 1
            for player in list(getattr(game_result, "players", []) or []):
                candidate_id = str(player.get("agent_id", ""))
                if candidate_id not in metrics:
                    continue
                metrics[candidate_id]["games_total"] += 1
                completed = bool(player.get("completed", False))
                if completed:
                    metrics[candidate_id]["games_completed"] += 1
                    rank = _safe_int(player.get("rank", 4), 4)
                    vp = float(player.get("victory_points", 0.0) or 0.0)
                    metrics[candidate_id]["rank_sum"] += float(rank)
                    metrics[candidate_id]["vp_sum"] += float(vp)
                    if int(rank) == 1:
                        metrics[candidate_id]["wins"] += 1

    ranked = rank_candidate_metrics(metrics)
    return ranked, successful_games, total_planned_games


def _default_state() -> Dict[str, Any]:
    return {
        "last_evaluated_generation_by_coord": {},
        "incumbent_checkpoint_path": "",
        "last_round_id": "",
    }


def _state_path(output_root: str) -> str:
    return os.path.join(output_root, "orchestrator_state.json")


def load_orchestrator_state(output_root: str) -> Dict[str, Any]:
    state = _load_json(_state_path(output_root))
    if not state:
        return _default_state()
    merged = _default_state()
    merged.update(state)
    return merged


def save_orchestrator_state(output_root: str, state: Dict[str, Any]) -> None:
    _atomic_write_json(_state_path(output_root), dict(state))


def _current_manifest_path(output_root: str) -> str:
    return os.path.join(output_root, "champion", "current", "champion_manifest.json")


def _current_checkpoint_path(output_root: str) -> str:
    return os.path.join(output_root, "champion", "current", "champion.pth")


def load_current_incumbent(output_root: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    manifest = _load_json(_current_manifest_path(output_root))
    checkpoint_path = _current_checkpoint_path(output_root)
    if not os.path.isfile(checkpoint_path):
        return None, None
    return manifest, checkpoint_path


def _candidate_to_payload(candidate: CandidateRef, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": str(candidate.candidate_id),
        "coordinator_id": str(candidate.coordinator_id),
        "source_type": str(candidate.source_type),
        "generation": int(candidate.generation),
        "checkpoint_relpath": str(candidate.checkpoint_relpath),
        "checkpoint_path": str(candidate.checkpoint_path),
        "metrics": {
            "games_total": int(metrics.get("games_total", 0)),
            "games_completed": int(metrics.get("games_completed", 0)),
            "wins": int(metrics.get("wins", 0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "avg_rank": float(metrics.get("avg_rank", 999.0)),
            "avg_vp": float(metrics.get("avg_vp", 0.0)),
        },
    }


def _publish_round(
    *,
    output_root: str,
    round_id: str,
    active_winner: CandidateRef,
    active_winner_metrics: Dict[str, Any],
    ranked_payloads: Sequence[Dict[str, Any]],
    promotion: Dict[str, Any],
    incumbent_before: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    history_dir = os.path.join(output_root, "champion", "history", round_id)
    current_dir = os.path.join(output_root, "champion", "current")
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(current_dir, exist_ok=True)

    history_checkpoint = os.path.join(history_dir, "champion.pth")
    _atomic_copy_file(active_winner.checkpoint_path, history_checkpoint)

    winner_payload = _candidate_to_payload(active_winner, active_winner_metrics)
    manifest_payload = {
        "version": 1,
        "round_id": str(round_id),
        "evaluated_at_utc": _utc_now_iso(),
        "winner": winner_payload,
        "incumbent_before": incumbent_before if isinstance(incumbent_before, dict) else None,
        "promotion": {
            "applied": bool(promotion.get("applied", False)),
            "reason": str(promotion.get("reason", "")),
            "required_win_rate": float(promotion.get("required_win_rate", 0.0) or 0.0),
        },
        "candidates": list(ranked_payloads),
    }

    if bool(promotion.get("applied", False)) or not os.path.isfile(_current_checkpoint_path(output_root)):
        _atomic_copy_file(active_winner.checkpoint_path, _current_checkpoint_path(output_root))
    _atomic_write_json(_current_manifest_path(output_root), manifest_payload)
    _atomic_write_json(os.path.join(history_dir, "champion_manifest.json"), manifest_payload)
    return manifest_payload


def prune_generation_dirs(coord_root: str, keep_last: int, pinned_generations: Iterable[int]) -> List[int]:
    keep_last = max(1, int(keep_last))
    generations = _available_generations(coord_root)
    pinned: Set[int] = set()
    for value in list(pinned_generations):
        try:
            pinned.add(int(value))
        except Exception:
            continue
    keep_generations = set(generations[-keep_last:]).union(pinned)
    removed: List[int] = []
    for generation in generations:
        if generation in keep_generations:
            continue
        generation_dir = os.path.join(coord_root, f"generation_{generation}")
        if not os.path.isdir(generation_dir):
            continue
        try:
            shutil.rmtree(generation_dir, ignore_errors=False)
            removed.append(int(generation))
        except Exception:
            logger.warning("Failed pruning generation_%d under %s", int(generation), coord_root, exc_info=True)
    return sorted(removed)


def _load_config() -> OrchestratorConfig:
    coord_sources = parse_coord_sources(os.getenv("ORCH_COORD_SOURCES", ""))
    if not coord_sources:
        raise ValueError("ORCH_COORD_SOURCES is empty. Example: coord-1=/app/coord-models/coord-1,coord-2=/app/coord-models/coord-2")
    trainer_coord_id = str(os.getenv("ORCH_TRAINER_COORD_ID", "coord-1")).strip() or "coord-1"
    if trainer_coord_id not in coord_sources:
        trainer_coord_id = sorted(coord_sources.keys())[0]

    return OrchestratorConfig(
        coord_sources=coord_sources,
        trainer_coord_id=trainer_coord_id,
        output_root=os.path.abspath(os.getenv("ORCH_OUTPUT_ROOT", "/app/rl-models-global")),
        poll_interval_sec=max(1.0, float(os.getenv("ORCH_POLL_INTERVAL_SEC", "20"))),
        trigger_every_n_gens=max(1, _safe_int(os.getenv("ORCH_TRIGGER_EVERY_N_GENS", "1"), 1)),
        top_k_per_coord=max(1, _safe_int(os.getenv("ORCH_TOP_K_PER_COORD", "2"), 2)),
        games_per_candidate=max(1, _safe_int(os.getenv("ORCH_GAMES_PER_CANDIDATE", "6"), 6)),
        global_game_concurrency=max(1, _safe_int(os.getenv("ORCH_GLOBAL_GAME_CONCURRENCY", "2"), 2)),
        tournament_concurrency=max(1, _safe_int(os.getenv("ORCH_TOURNAMENT_CONCURRENCY", "2"), 2)),
        min_games_for_promotion=max(1, _safe_int(os.getenv("ORCH_MIN_GAMES_FOR_PROMOTION", "24"), 24)),
        min_completion_rate=min(1.0, max(0.0, float(os.getenv("ORCH_MIN_COMPLETION_RATE", "0.90")))),
        win_rate_margin=max(0.0, float(os.getenv("ORCH_WIN_RATE_MARGIN", "0.03"))),
        keep_generations_trainer=max(1, _safe_int(os.getenv("ORCH_KEEP_GENERATIONS_TRAINER", "30"), 30)),
        keep_generations_worker=max(1, _safe_int(os.getenv("ORCH_KEEP_GENERATIONS_WORKER", "15"), 15)),
    )


def _build_incumbent_candidate(output_root: str, manifest: Optional[Dict[str, Any]], checkpoint_path: Optional[str]) -> Optional[CandidateRef]:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None
    winner = dict((manifest or {}).get("winner", {}) or {})
    coordinator_id = str(winner.get("coordinator_id", "incumbent") or "incumbent")
    generation = _safe_int(winner.get("generation", -1), -1)
    candidate_id = str(winner.get("candidate_id", "incumbent:current") or "incumbent:current")
    checkpoint_relpath = str(winner.get("checkpoint_relpath", "champion/current/champion.pth") or "champion/current/champion.pth")
    return CandidateRef(
        candidate_id=candidate_id,
        coordinator_id=coordinator_id,
        source_type="incumbent",
        coord_root=os.path.abspath(output_root),
        generation=int(generation),
        checkpoint_path=os.path.abspath(checkpoint_path),
        checkpoint_relpath=checkpoint_relpath,
    )


def _coordinator_keep_window(config: OrchestratorConfig, coord_id: str) -> int:
    if str(coord_id) == str(config.trainer_coord_id):
        return int(config.keep_generations_trainer)
    return int(config.keep_generations_worker)


async def run_round(config: OrchestratorConfig, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    progress = {
        coord_id: discover_coord_progress(coord_id, coord_root)
        for coord_id, coord_root in config.coord_sources.items()
    }
    trainer_progress = progress.get(config.trainer_coord_id)
    if trainer_progress is None:
        logger.warning("Trainer coordinator %s is missing from coord sources", config.trainer_coord_id)
        return None

    last_eval_map = dict(state.get("last_evaluated_generation_by_coord", {}) or {})
    last_eval_trainer = _safe_int(last_eval_map.get(config.trainer_coord_id, -1), -1)
    if trainer_progress.generation_signal < 0:
        logger.info("Trainer has no generation signal yet. Waiting.")
        return None
    if trainer_progress.generation_signal < (last_eval_trainer + int(config.trigger_every_n_gens)):
        return None

    all_candidates: List[CandidateRef] = []
    for coord_id, coord_root in config.coord_sources.items():
        discovered = discover_candidates_for_coord(coord_id, coord_root, config.top_k_per_coord)
        all_candidates.extend(discovered)

    incumbent_manifest, incumbent_checkpoint = load_current_incumbent(config.output_root)
    incumbent_candidate = _build_incumbent_candidate(config.output_root, incumbent_manifest, incumbent_checkpoint)
    incumbent_candidate_id: Optional[str] = None
    if incumbent_candidate is not None:
        incumbent_candidate_id = str(incumbent_candidate.candidate_id)
        existing = {os.path.abspath(candidate.checkpoint_path) for candidate in all_candidates}
        if os.path.abspath(incumbent_candidate.checkpoint_path) not in existing:
            all_candidates.append(incumbent_candidate)

    if not all_candidates:
        logger.warning("No candidate checkpoints discovered; marking trainer generation as evaluated to avoid busy-loop.")
        state["last_evaluated_generation_by_coord"] = {
            coord_id: int(item.generation_signal)
            for coord_id, item in progress.items()
        }
        save_orchestrator_state(config.output_root, state)
        return None

    round_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["TOURNAMENT_CONCURRENCY"] = str(config.tournament_concurrency)
    os.environ["GLOBAL_GAME_CONCURRENCY"] = str(config.global_game_concurrency)
    game_servers = [s.strip() for s in str(os.getenv("GAME_SERVERS", "")).split(",") if s.strip()]
    if not game_servers:
        raise ValueError("GAME_SERVERS is empty; orchestrator cannot run cross-play")

    logger.info(
        "Running champion round %s with %d candidate(s), games_per_candidate=%d",
        round_id,
        len(all_candidates),
        int(config.games_per_candidate),
    )
    ranked, successful_games, planned_games = await evaluate_candidates(
        all_candidates,
        game_servers=game_servers,
        games_per_candidate=config.games_per_candidate,
        global_game_concurrency=config.global_game_concurrency,
        round_seed=round_id,
    )
    completion_rate = (float(successful_games) / float(planned_games)) if planned_games > 0 else 0.0
    candidate_lookup = _candidate_by_id(all_candidates)
    ranked_payloads: List[Dict[str, Any]] = []
    for item in ranked:
        candidate_id = str(item.get("candidate_id", ""))
        candidate = candidate_lookup.get(candidate_id)
        if candidate is None:
            continue
        ranked_payloads.append(_candidate_to_payload(candidate, item))

    promotion = decide_promotion(
        ranked_candidates=ranked,
        incumbent_candidate_id=incumbent_candidate_id,
        completed_games=int(successful_games),
        completion_rate=float(completion_rate),
        min_games_for_promotion=int(config.min_games_for_promotion),
        min_completion_rate=float(config.min_completion_rate),
        win_rate_margin=float(config.win_rate_margin),
    )
    winner_candidate_id = str(promotion.get("winner_candidate_id") or "")
    active_winner = candidate_lookup.get(winner_candidate_id)
    if active_winner is None and ranked:
        active_winner = candidate_lookup.get(str(ranked[0].get("candidate_id", "")))
    if active_winner is None:
        logger.warning("No winner candidate resolved for round %s", round_id)
        return None

    winner_metrics = next(
        (item for item in ranked if str(item.get("candidate_id", "")) == str(active_winner.candidate_id)),
        {
            "games_total": 0,
            "games_completed": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_rank": 999.0,
            "avg_vp": 0.0,
        },
    )
    incumbent_before = dict(incumbent_manifest.get("winner", {}) or {}) if isinstance(incumbent_manifest, dict) else None
    manifest_payload = _publish_round(
        output_root=config.output_root,
        round_id=round_id,
        active_winner=active_winner,
        active_winner_metrics=winner_metrics,
        ranked_payloads=ranked_payloads,
        promotion=promotion,
        incumbent_before=incumbent_before,
    )

    state["last_evaluated_generation_by_coord"] = {
        coord_id: int(item.generation_signal)
        for coord_id, item in progress.items()
    }
    state["incumbent_checkpoint_path"] = os.path.abspath(_current_checkpoint_path(config.output_root))
    state["last_round_id"] = str(round_id)
    save_orchestrator_state(config.output_root, state)

    # Post-round prune of coordinator generation directories.
    active_champion = dict(manifest_payload.get("winner", {}) or {})
    champion_coord = str(active_champion.get("coordinator_id", "") or "")
    champion_generation = _safe_int(active_champion.get("generation", -1), -1)
    candidates_by_coord: Dict[str, Set[int]] = {}
    for candidate in all_candidates:
        if candidate.source_type != "coord":
            continue
        candidates_by_coord.setdefault(str(candidate.coordinator_id), set()).add(int(candidate.generation))
    for coord_id, coord_root in config.coord_sources.items():
        pinned = set(candidates_by_coord.get(coord_id, set()))
        if coord_id == champion_coord and champion_generation >= 0:
            pinned.add(int(champion_generation))
        removed = prune_generation_dirs(
            coord_root=coord_root,
            keep_last=_coordinator_keep_window(config, coord_id),
            pinned_generations=pinned,
        )
        if removed:
            logger.info("Pruned %s generations for %s: %s", len(removed), coord_id, removed)

    logger.info(
        "Champion round %s complete: planned_games=%d successful_games=%d completion_rate=%.3f winner=%s promotion=%s(%s)",
        round_id,
        int(planned_games),
        int(successful_games),
        float(completion_rate),
        active_winner.candidate_id,
        bool(promotion.get("applied", False)),
        str(promotion.get("reason", "")),
    )
    return manifest_payload


async def orchestrator_loop(*, once: bool = False) -> None:
    config = _load_config()
    os.makedirs(config.output_root, exist_ok=True)
    state = load_orchestrator_state(config.output_root)

    while True:
        try:
            await run_round(config, state)
        except Exception:
            logger.exception("Champion orchestrator round failed")

        if once:
            break
        await asyncio.sleep(float(config.poll_interval_sec))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global champion orchestrator.")
    parser.add_argument("--once", action="store_true", help="Run one round check and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(orchestrator_loop(once=bool(args.once)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
