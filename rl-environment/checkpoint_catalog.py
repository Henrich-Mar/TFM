"""Discover and rank trained checkpoints for standalone play.

The training stack writes checkpoints under several incompatible layouts
(legacy ``rl-models/generation_*``, self-play ``<root>/checkpoints/candidate_*``,
``<root>/checkpoints/champion.pth``, behaviour-cloning ``bc_best.pth``, and the
cross-coordinator ``rl-models-global/champion`` tournament). Strength evidence is
also scattered: ``v2_benchmark`` JSON reports, the ``champion_manifest.json``
tournament manifest, and the legacy ``agent_*_config.json`` sidecars.

This module gathers all of it into one ranked list so a human can pick the
strongest checkpoint to bring to a table with other humans, instead of
hand-pasting a path or trusting a filename.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CheckpointCandidate",
    "default_search_bases",
    "discover_checkpoints",
    "format_candidate_row",
    "rank_candidates",
    "select_best_checkpoint",
]

# Directory names under the repository root that hold checkpoint stores.
DEFAULT_SEARCH_BASES: Tuple[str, ...] = (
    "rl-v2",
    "rl-v3",
    "rl-v4",
    "rl-alphago",
    "rl-models",
    "rl-models-global",
)

# Per-base glob patterns for checkpoints and their sidecar evidence.
_CHECKPOINT_PATTERNS: Tuple[str, ...] = (
    "checkpoints/*.pth",
    "pretrain*/*.pth",
    "generation_*/agent_*_fitness_*.pth",
    "champion/current/champion.pth",
    "champion/history/*/champion.pth",
    "*.pth",
)
_BENCHMARK_PATTERN = "benchmarks/benchmark_*.json"
_MANIFEST_PATTERN = "champion/current/champion_manifest.json"

# Evidence tiers. Higher wins. A checkpoint with hard strength evidence always
# outranks an unvalidated learner, no matter how many decisions it has seen.
TIER_TOURNAMENT_WINNER = 60
TIER_TOURNAMENT_CANDIDATE = 50
TIER_GATE_PASS = 40
TIER_BENCHMARK = 30
TIER_SIDECAR_METRICS = 20
TIER_SIDECAR = 15
TIER_DECISIONS = 10
TIER_UNRANKED = 5

_TIER_LABELS: Dict[int, str] = {
    TIER_TOURNAMENT_WINNER: "tournament winner",
    TIER_TOURNAMENT_CANDIDATE: "tournament candidate",
    TIER_GATE_PASS: "benchmark",
    TIER_BENCHMARK: "benchmark",
    TIER_SIDECAR_METRICS: "generation metrics",
    TIER_SIDECAR: "generation sidecar",
    TIER_DECISIONS: "training decisions",
    TIER_UNRANKED: "unverified",
}

# How much a benchmark baseline tells us about strength against humans. The
# scripted teacher is the closest stand-in for a real opponent, so a checkpoint
# measured against it outranks one measured against an older, weaker champion
# snapshot even if the raw first-place rate is higher. Without this ordering a
# 100% win rate against ``random`` would outrank a real 59% win rate against the
# teacher.
_BASELINE_RANKS: Dict[str, int] = {
    "teacher": 4,
    "award_teacher": 3,
    "champion": 2,
    "random": 1,
}
_UNRANKED_BASELINE = 0

_GENERATION_FITNESS_RE = re.compile(r"agent_(\d+)_fitness_(-?\d+(?:\.\d+)?)\.pth$")
_DECISIONS_RE = re.compile(r"decisions_(\d+)")
_TRAILING_NUMBER_RE = re.compile(r"(\d{4,})")

_ROLE_CHAMPION = "champion"
_ROLE_LATEST_LEARNER = "latest learner"
_ROLE_CANDIDATE = "candidate"
_ROLE_BC = "behaviour cloning"
_ROLE_HISTORY = "history"
_ROLE_GENERATION = "generation"
_ROLE_FALLBACK = "checkpoint"

# A benchmark report below this completion rate is a partial/aborted run and is
# not treated as strength evidence.
_MIN_BENCHMARK_COMPLETION_RATE = 0.5

# Benchmarks are ranked by the lower bound of the 95% Wilson interval on the
# pooled first-place rate, so a lucky 32-game screen cannot outrank a long run.
_WILSON_Z = 1.96

# Files replaced in place on promotion; reports older than the file are stale.
_MUTABLE_ROLES = frozenset({_ROLE_CHAMPION, _ROLE_LATEST_LEARNER})
_STALE_TOLERANCE_SECONDS = 60.0
_IDENTITY_PROBE_BYTES = 1 << 20


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_search_bases(root: Optional[str] = None) -> List[str]:
    """Return the checkpoint store directories to scan under ``root``."""
    base = os.path.abspath(root or repo_root())
    found: List[str] = []
    for name in DEFAULT_SEARCH_BASES:
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            found.append(candidate)
    return found


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fitness_from_name(path: str) -> float:
    """Legacy ``agent_<i>_fitness_<score>.pth`` fitness, or ``-inf``."""
    match = _GENERATION_FITNESS_RE.search(os.path.basename(path))
    if not match:
        return float("-inf")
    try:
        return float(match.group(2))
    except ValueError:
        return float("-inf")


def _decisions_from_name(stem: str) -> Optional[int]:
    match = _DECISIONS_RE.search(stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    match = _TRAILING_NUMBER_RE.search(stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _classify_role(stem: str) -> str:
    lowered = stem.lower()
    if lowered.startswith("champion"):
        return _ROLE_CHAMPION
    if lowered == "latest_learner":
        return _ROLE_LATEST_LEARNER
    if lowered.startswith("bc_"):
        return _ROLE_BC
    if lowered.startswith("history_policy"):
        return _ROLE_HISTORY
    if lowered.startswith("candidate"):
        return _ROLE_CANDIDATE
    if _GENERATION_FITNESS_RE.search(f"{stem}.pth"):
        return _ROLE_GENERATION
    return _ROLE_FALLBACK


@dataclass
class CheckpointCandidate:
    """One selectable checkpoint plus the evidence used to rank it."""

    path: str
    store: str
    role: str
    tier: int = TIER_UNRANKED
    mtime: float = 0.0
    size_bytes: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    decisions: Optional[int] = None
    fitness: Optional[float] = None
    verified: bool = False

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def verified_label(self) -> str:
        return _TIER_LABELS.get(self.tier, "unverified")

    @property
    def baseline(self) -> str:
        return str(self.metrics.get("baseline") or "")

    @property
    def baseline_rank(self) -> float:
        return float(_BASELINE_RANKS.get(self.baseline, _UNRANKED_BASELINE))

    def score_key(self) -> Tuple[float, ...]:
        """Ranking key; larger is better.

        Order of trust: strength evidence, then how human-like the measured
        opponent was, then the measured result. The role only breaks ties, so a
        stale or weak ``champion.pth`` cannot outrank a stronger candidate.
        """
        primary = _as_float(self.metrics.get("primary")) or 0.0
        secondary = _as_float(self.metrics.get("secondary")) or 0.0
        # Prefer the champion file over the identical candidate it was copied from.
        role_bonus = 1.0 if self.role == _ROLE_CHAMPION else 0.0
        return (
            float(self.tier),
            self.baseline_rank,
            primary,
            secondary,
            role_bonus,
            float(self.mtime),
        )

    def summary(self) -> str:
        parts: List[str] = [self.verified_label]
        win_rate = _as_float(self.metrics.get("tournament_win_rate"))
        first_place = _as_float(self.metrics.get("first_place_rate"))
        elo = _as_float(self.metrics.get("elo"))
        mean_rank = _as_float(self.metrics.get("mean_rank"))
        if mean_rank is None:
            mean_rank = _as_float(self.metrics.get("tournament_avg_rank"))
        if win_rate is not None:
            parts.append(f"win rate {win_rate * 100:.0f}%")
        if first_place is not None:
            opponent = f" vs {self.baseline}" if self.baseline else ""
            detail = []
            lower = _as_float(self.metrics.get("first_place_lower_95"))
            if lower is not None:
                detail.append(f"95% low {lower * 100:.0f}%")
            games = self.metrics.get("completed_games")
            if games:
                runs = self.metrics.get("benchmark_runs") or 1
                detail.append(f"{games} games/{runs} run{'s' if runs != 1 else ''}")
            suffix = f" ({', '.join(detail)})" if detail else ""
            parts.append(f"1st place {first_place * 100:.0f}%{opponent}{suffix}")
        if elo is not None:
            parts.append(f"elo {elo:.0f}")
        gate = self.metrics.get("gate_passed")
        if gate is not None:
            parts.append("gate PASS" if gate else "gate FAIL")
        if mean_rank is not None:
            parts.append(f"avg rank {mean_rank:.2f}")
        if self.decisions is not None:
            parts.append(f"{self.decisions:,} decisions")
        return " | ".join(parts)


def _mtime_of(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _size_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _resolve(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(path))
    except Exception:
        return os.path.abspath(path)


def _relpath(path: str, root: str) -> str:
    absolute = os.path.abspath(path)
    relative = os.path.relpath(absolute, root)
    if relative == ".." or relative.startswith(f"..{os.sep}"):
        return absolute
    return relative


def _store_label(path: str, root: str) -> str:
    relative = _relpath(path, root)
    parts = relative.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] in DEFAULT_SEARCH_BASES:
        return "/".join(parts[:2])
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else relative


def _iter_checkpoint_files(search_bases: Sequence[str]) -> List[Tuple[str, str]]:
    """Return ``(store base, checkpoint path)`` pairs, de-duplicated."""
    found: List[Tuple[str, str]] = []
    seen: set = set()
    for base in search_bases:
        for pattern in _CHECKPOINT_PATTERNS:
            for match in glob.glob(os.path.join(base, pattern)):
                if not os.path.isfile(match):
                    continue
                key = _resolve(match)
                if key in seen:
                    continue
                seen.add(key)
                found.append((base, match))
    return sorted(found, key=lambda item: item[1])


def _benchmark_entry(report_path: str) -> Optional[Dict[str, Any]]:
    report = _load_json(report_path)
    if not report:
        return None
    if "first_place_rate" not in report and "gate_passed" not in report:
        return None
    completed = int(report.get("completed_games", 0) or 0)
    planned = int(report.get("planned_games", 0) or 0)
    completion_rate = _as_float(report.get("completion_rate"))
    if completion_rate is None and planned > 0:
        completion_rate = completed / float(planned)
    return {
        "report_path": report_path,
        "mtime": _mtime_of(report_path),
        "baseline": str(report.get("baseline", "")),
        "stage": report.get("stage"),
        "first_place_rate": _as_float(report.get("first_place_rate")),
        "mean_rank": _as_float(report.get("mean_rank")),
        "pairwise_score": _as_float(report.get("pairwise_score")),
        "gate_passed": bool(report.get("gate_passed", False)),
        "completed_games": completed,
        "planned_games": planned,
        "completion_rate": completion_rate,
        "checkpoint": str(report.get("checkpoint", "") or "").strip(),
    }


def _report_stem(entry: Dict[str, Any]) -> str:
    checkpoint = str(entry.get("checkpoint") or "").replace("\\", "/")
    if checkpoint:
        return os.path.splitext(checkpoint.rsplit("/", 1)[-1])[0]
    # Fall back to the report filename: benchmark_<stem>_stage<N>_<baseline>.json
    return os.path.basename(entry["report_path"])[len("benchmark_"):].split("_stage")[0]


def _collect_benchmarks(
    search_bases: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """Index every benchmark report by resolved checkpoint path and by (store, stem).

    Reports usually carry container paths (``/app/...``) that never resolve
    locally, so the stem index is the common match. It is scoped to the store
    the report lives in: a ``champion`` report from one store says nothing about
    another store's ``champion.pth``.
    """
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    by_stem: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for base in search_bases:
        for report_path in glob.glob(os.path.join(base, _BENCHMARK_PATTERN)):
            entry = _benchmark_entry(report_path)
            if entry is None:
                continue
            checkpoint = entry.get("checkpoint") or ""
            if checkpoint:
                by_path.setdefault(_resolve(checkpoint), []).append(entry)
            stem = _report_stem(entry)
            if stem:
                by_stem.setdefault((base, stem), []).append(entry)
    return by_path, by_stem


def _reports_for(
    candidate: "CheckpointCandidate",
    base: str,
    by_path: Dict[str, List[Dict[str, Any]]],
    by_stem: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    seen: set = set()
    reports: List[Dict[str, Any]] = []
    for entry in by_path.get(candidate.path, []) + by_stem.get((base, candidate.stem), []):
        if entry["report_path"] in seen:
            continue
        seen.add(entry["report_path"])
        reports.append(entry)
    return reports


def _wilson_lower(successes: float, games: float, z: float = _WILSON_Z) -> float:
    """Lower bound of the Wilson score interval; penalises small samples."""
    if games <= 0:
        return 0.0
    rate = successes / games
    denominator = 1.0 + z * z / games
    centre = rate + z * z / (2.0 * games)
    margin = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games))
    return max(0.0, (centre - margin) / denominator)


def _is_complete(entry: Dict[str, Any]) -> bool:
    return (_as_float(entry.get("completion_rate")) or 0.0) >= _MIN_BENCHMARK_COMPLETION_RATE


def _drop_stale_reports(candidate: "CheckpointCandidate", reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ignore reports older than a checkpoint that is overwritten in place.

    ``champion.pth`` and ``latest_learner.pth`` are replaced on promotion, so a
    report written before the current file measured a different network.
    ``candidate_*`` files are immutable, so their reports are always kept (this
    also stops a fresh clone, where every mtime is "now", from discarding all
    evidence).
    """
    if candidate.role not in _MUTABLE_ROLES or not candidate.mtime:
        return reports
    fresh = [entry for entry in reports if entry["mtime"] + _STALE_TOLERANCE_SECONDS >= candidate.mtime]
    stale = len(reports) - len(fresh)
    if stale:
        candidate.evidence.append(f"{stale} stale benchmark(s) ignored (file overwritten since)")
    return fresh


def _pool_benchmarks(reports: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pool every completed report on the hardest baseline into one estimate.

    Taking only the best single run rewards lucky screens. Pooling all runs on
    the same baseline and ranking by the Wilson lower bound rewards checkpoints
    that are strong *and* well measured.
    """
    complete = [entry for entry in reports if _is_complete(entry)]
    if not complete:
        return None
    baseline = max(
        (str(entry.get("baseline", "")) for entry in complete),
        key=lambda name: _BASELINE_RANKS.get(name, _UNRANKED_BASELINE),
    )
    group = [entry for entry in complete if str(entry.get("baseline", "")) == baseline]
    games = 0
    wins = 0.0
    rank_games = 0
    rank_total = 0.0
    gate_games = 0
    for entry in group:
        played = int(entry.get("completed_games") or 0)
        if played <= 0:
            continue
        mean_rank = _as_float(entry.get("mean_rank"))
        games += played
        wins += (_as_float(entry.get("first_place_rate")) or 0.0) * played
        if mean_rank is not None:
            rank_games += played
            rank_total += mean_rank * played
        if entry.get("gate_passed"):
            gate_games += played
    if games <= 0:
        return None
    return {
        "baseline": baseline,
        "stage": group[0].get("stage"),
        "first_place_rate": wins / games,
        "first_place_lower_95": _wilson_lower(wins, games),
        "mean_rank": rank_total / rank_games if rank_games else None,
        # The gate is judged on the majority of measured games, not the best run.
        "gate_passed": gate_games * 2 >= games,
        "completed_games": games,
        "runs": len(group),
        "report_paths": [entry["report_path"] for entry in group],
    }


def _apply_benchmark(candidate: CheckpointCandidate, reports: Sequence[Dict[str, Any]]) -> None:
    for entry in reports:
        if not _is_complete(entry):
            # A partially completed run says nothing about strength, so record it
            # for context without letting it count as verification evidence.
            candidate.evidence.append(
                f"aborted benchmark vs {entry.get('baseline')} "
                f"({entry.get('completed_games')}/{entry.get('planned_games')} games)"
            )
    pooled = _pool_benchmarks(reports)
    if pooled is None:
        return
    mean_rank = pooled["mean_rank"]
    candidate.metrics.update(
        {
            "first_place_rate": pooled["first_place_rate"],
            "first_place_lower_95": pooled["first_place_lower_95"],
            "mean_rank": mean_rank,
            "gate_passed": pooled["gate_passed"],
            "completed_games": pooled["completed_games"],
            "benchmark_runs": pooled["runs"],
            "baseline": pooled["baseline"],
            "stage": pooled["stage"],
            "report_paths": pooled["report_paths"],
        }
    )
    candidate.metrics["primary"] = pooled["first_place_lower_95"]
    # Lower mean rank is better, so invert it into a "higher is better" value.
    candidate.metrics["secondary"] = -mean_rank if mean_rank is not None else 0.0
    candidate.tier = max(candidate.tier, TIER_GATE_PASS if pooled["gate_passed"] else TIER_BENCHMARK)
    candidate.evidence.append(
        f"benchmark stage{pooled['stage']} vs {pooled['baseline']} "
        f"({pooled['completed_games']} games over {pooled['runs']} run(s))"
    )


def _same_weights(left: str, right: str) -> bool:
    """Cheap identity probe for two equally sized checkpoint files."""
    try:
        with open(left, "rb") as a, open(right, "rb") as b:
            if a.read(_IDENTITY_PROBE_BYTES) != b.read(_IDENTITY_PROBE_BYTES):
                return False
            size = os.fstat(a.fileno()).st_size
            offset = max(0, size - _IDENTITY_PROBE_BYTES)
            a.seek(offset)
            b.seek(offset)
            return a.read() == b.read()
    except OSError:
        return False


def _find_alias(
    candidate: CheckpointCandidate,
    base: str,
    peers: Sequence[Tuple[str, CheckpointCandidate]],
) -> Optional[CheckpointCandidate]:
    """Find the immutable ``candidate_*`` file a promoted checkpoint was copied from.

    Promotion uses ``shutil.copy2``, so the copy keeps the source's size and
    mtime; the content probe rules out coincidences.
    """
    for peer_base, peer in peers:
        if peer is candidate or peer_base != base or peer.role != _ROLE_CANDIDATE:
            continue
        if peer.size_bytes != candidate.size_bytes or abs(peer.mtime - candidate.mtime) > 2.0:
            continue
        if _same_weights(candidate.path, peer.path):
            return peer
    return None


def _apply_manifest(candidate: CheckpointCandidate, metrics: Dict[str, Any], *, winner: bool) -> None:
    win_rate = _as_float(metrics.get("win_rate"))
    avg_rank = _as_float(metrics.get("avg_rank"))
    avg_vp = _as_float(metrics.get("avg_vp"))
    candidate.metrics.update(
        {
            "tournament_win_rate": win_rate,
            "tournament_avg_rank": avg_rank,
            "tournament_avg_vp": avg_vp,
            "tournament_games": metrics.get("games_completed"),
        }
    )
    candidate.metrics["primary"] = win_rate if win_rate is not None else 0.0
    candidate.metrics["secondary"] = -avg_rank if avg_rank is not None else 0.0
    candidate.tier = TIER_TOURNAMENT_WINNER if winner else TIER_TOURNAMENT_CANDIDATE
    candidate.evidence.append(
        f"{'tournament winner' if winner else 'tournament candidate'} "
        f"({metrics.get('games_completed', 0)} games, win rate {win_rate if win_rate is not None else 0.0:.2f})"
    )


def _collect_manifests(search_bases: Sequence[str]) -> Dict[str, Tuple[Dict[str, Any], bool]]:
    """Map resolved checkpoint path -> (metrics, is_winner) from tournament manifests."""
    entries: Dict[str, Tuple[Dict[str, Any], bool]] = {}
    for base in search_bases:
        for manifest_path in glob.glob(os.path.join(base, _MANIFEST_PATTERN)):
            manifest = _load_json(manifest_path)
            if not manifest:
                continue
            winner = manifest.get("winner") if isinstance(manifest.get("winner"), dict) else {}
            winner_path = str(winner.get("checkpoint_path", "") or "").strip()
            if winner_path:
                metrics = winner.get("metrics") if isinstance(winner.get("metrics"), dict) else {}
                entries[_resolve(winner_path)] = (metrics, True)
            for candidate in manifest.get("candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                path = str(candidate.get("checkpoint_path", "") or "").strip()
                if not path:
                    continue
                key = _resolve(path)
                metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
                existing = entries.get(key)
                if existing is None or not existing[1]:
                    entries[key] = (metrics, False)
    return entries


def _apply_sidecar(candidate: CheckpointCandidate, sidecar: Dict[str, Any]) -> None:
    elo = _as_float(sidecar.get("elo"))
    if elo is None:
        advanced = sidecar.get("advanced_metrics")
        if isinstance(advanced, dict):
            elo = _as_float(advanced.get("elo"))
    eval_fitness = _as_float(sidecar.get("eval_fitness"))
    if eval_fitness is None:
        eval_fitness = _as_float(sidecar.get("eval_fitness_gated"))
    candidate.metrics.update(
        {
            "elo": elo,
            "eval_fitness": eval_fitness,
            "saved_generation": sidecar.get("saved_generation"),
        }
    )
    if elo is not None or eval_fitness is not None:
        candidate.tier = TIER_SIDECAR_METRICS
        candidate.metrics["primary"] = elo if elo is not None else (eval_fitness or 0.0)
        candidate.metrics["secondary"] = eval_fitness if eval_fitness is not None else 0.0
        parts = []
        if elo is not None:
            parts.append(f"elo {elo:.0f}")
        if eval_fitness is not None:
            parts.append(f"eval fitness {eval_fitness:.2f}")
        generation = sidecar.get("saved_generation")
        if generation is not None:
            parts.append(f"generation {generation}")
        candidate.evidence.append("generation " + ", ".join(parts))
    else:
        candidate.tier = TIER_SIDECAR
        candidate.evidence.append("generation sidecar without strength metrics")


def _sidecar_for(path: str) -> Optional[Dict[str, Any]]:
    base = os.path.basename(path)
    if not _GENERATION_FITNESS_RE.search(base):
        return None
    index = base.split("_", 2)[1]
    sidecar = os.path.join(os.path.dirname(path), f"agent_{index}_config.json")
    return _load_json(sidecar) if os.path.isfile(sidecar) else None


def _finalize_tier(candidate: CheckpointCandidate) -> None:
    """Downgrade candidates that fell through to a weaker evidence tier."""
    if candidate.tier == TIER_UNRANKED and candidate.decisions is not None:
        candidate.tier = TIER_DECISIONS
        candidate.metrics.setdefault("primary", float(candidate.decisions))
        candidate.evidence.append(f"{candidate.decisions:,} training decisions")
    if candidate.tier == TIER_UNRANKED and candidate.fitness is not None:
        candidate.tier = TIER_DECISIONS
        candidate.metrics.setdefault("primary", candidate.fitness)
        candidate.evidence.append(f"filename fitness {candidate.fitness:.2f}")


def _build_candidate(path: str, root: str) -> CheckpointCandidate:
    stem = os.path.splitext(os.path.basename(path))[0]
    return CheckpointCandidate(
        path=_resolve(path),
        store=_store_label(path, root),
        role=_classify_role(stem),
        mtime=_mtime_of(path),
        size_bytes=_size_of(path),
        decisions=_decisions_from_name(stem),
        fitness=_fitness_from_name(path),
    )


def discover_checkpoints(
    search_bases: Optional[Iterable[str]] = None,
    *,
    root: Optional[str] = None,
) -> List[CheckpointCandidate]:
    """Return every discoverable checkpoint, best first.

    ``search_bases`` defaults to every known checkpoint store under the
    repository root. Pass explicit directories to scan somewhere else.
    """
    base_root = os.path.abspath(root or repo_root())
    bases = [os.path.abspath(str(item)) for item in (search_bases or default_search_bases(base_root)) if str(item or "").strip()]
    bases = [item for item in bases if os.path.isdir(item)]
    if not bases:
        return []

    benchmarks_by_path, benchmarks_by_stem = _collect_benchmarks(bases)
    manifest_entries = _collect_manifests(bases)

    found = [(base, path, _build_candidate(path, base_root)) for base, path in _iter_checkpoint_files(bases)]
    peers = [(base, candidate) for base, _, candidate in found]

    candidates: List[CheckpointCandidate] = []
    for base, path, candidate in found:
        manifest_entry = manifest_entries.get(candidate.path)
        if manifest_entry is not None:
            metrics, winner = manifest_entry
            _apply_manifest(candidate, metrics, winner=winner)

        if candidate.tier < TIER_TOURNAMENT_CANDIDATE:
            reports = _drop_stale_reports(
                candidate, _reports_for(candidate, base, benchmarks_by_path, benchmarks_by_stem)
            )
            if candidate.role in _MUTABLE_ROLES:
                # A promoted champion is a byte copy of a measured candidate;
                # inherit that candidate's (always current) evidence.
                alias = _find_alias(candidate, base, peers)
                if alias is not None:
                    candidate.evidence.append(f"same weights as {alias.name}")
                    known = {entry["report_path"] for entry in reports}
                    reports = reports + [
                        entry
                        for entry in _reports_for(alias, base, benchmarks_by_path, benchmarks_by_stem)
                        if entry["report_path"] not in known
                    ]
            if reports:
                _apply_benchmark(candidate, reports)

        sidecar = _sidecar_for(path)
        if sidecar is not None and candidate.tier < TIER_GATE_PASS:
            _apply_sidecar(candidate, sidecar)

        _finalize_tier(candidate)
        candidate.verified = candidate.tier > TIER_DECISIONS
        candidates.append(candidate)

    return rank_candidates(candidates)


def rank_candidates(candidates: Sequence[CheckpointCandidate]) -> List[CheckpointCandidate]:
    """Sort candidates best-first (verified evidence first, then recency)."""
    return sorted(candidates, key=lambda item: item.score_key(), reverse=True)


def select_best_checkpoint(
    search_bases: Optional[Iterable[str]] = None,
    *,
    root: Optional[str] = None,
) -> Optional[CheckpointCandidate]:
    candidates = discover_checkpoints(search_bases, root=root)
    return candidates[0] if candidates else None


def format_candidate_row(candidate: CheckpointCandidate, index: int, root: Optional[str] = None) -> str:
    base_root = os.path.abspath(root or repo_root())
    stamp = ""
    if candidate.mtime:
        stamp = datetime.fromtimestamp(candidate.mtime).strftime("%Y-%m-%d %H:%M")
    return (
        f"{index:>2}. [{'v' if candidate.verified else ' '}] "
        f"{_relpath(candidate.path, base_root)}\n"
        f"    {candidate.role} | {candidate.summary()}"
        + (f" | {stamp}" if stamp else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List and rank trained checkpoints for standalone play")
    parser.add_argument(
        "--search-base",
        action="append",
        default=[],
        help="Directory to scan. Repeatable. Defaults to every known store in the repository.",
    )
    parser.add_argument("--root", default="", help="Repository root used to shorten displayed paths.")
    parser.add_argument("--top", type=int, default=0, help="Only print the first N candidates.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    candidates = discover_checkpoints(args.search_base or None, root=args.root or None)
    if args.top > 0:
        candidates = candidates[: args.top]
    if not candidates:
        print("No checkpoints found.")
        return

    if args.json:
        print(json.dumps(
            [
                {
                    "path": item.path,
                    "store": item.store,
                    "role": item.role,
                    "verified": item.verified,
                    "tier": item.tier,
                    "evidence": item.evidence,
                    "metrics": item.metrics,
                }
                for item in candidates
            ],
            indent=2,
        ))
        return

    print(f"Found {len(candidates)} checkpoints. Ranked best first ([v] = strength-verified):")
    for index, candidate in enumerate(candidates, start=1):
        print(format_candidate_row(candidate, index, args.root or None))


if __name__ == "__main__":
    main()
