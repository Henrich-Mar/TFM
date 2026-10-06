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
        opponent was, then the checkpoint's own role, then the measured result.
        """
        primary = _as_float(self.metrics.get("primary")) or 0.0
        secondary = _as_float(self.metrics.get("secondary")) or 0.0
        # A checkpoint in an explicit champion role is a better default pick than
        # an equally-scored intermediate candidate.
        role_bonus = 1.0 if self.role == _ROLE_CHAMPION else 0.0
        return (
            float(self.tier),
            self.baseline_rank,
            role_bonus,
            primary,
            secondary,
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
            parts.append(f"1st place {first_place * 100:.0f}%{opponent}")
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


def _iter_checkpoint_files(search_bases: Sequence[str]) -> List[str]:
    found: List[str] = []
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
                found.append(match)
    return sorted(found)


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


def _collect_benchmarks(search_bases: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Index benchmark reports by resolved checkpoint path and by checkpoint stem.

    A checkpoint can have several reports (one per baseline/stage/label); keep
    the strongest completed, gate-passing one.
    """
    by_path: Dict[str, Dict[str, Any]] = {}
    by_stem: Dict[str, Dict[str, Any]] = {}
    for base in search_bases:
        for report_path in glob.glob(os.path.join(base, _BENCHMARK_PATTERN)):
            entry = _benchmark_entry(report_path)
            if entry is None:
                continue
            checkpoint = entry.get("checkpoint") or ""
            stem = os.path.splitext(os.path.basename(checkpoint))[0] if checkpoint else ""
            if not stem:
                # Fall back to the report filename: benchmark_<stem>_stage<N>_<baseline>.json
                stem = os.path.basename(report_path)[len("benchmark_"):].split("_stage")[0]
            if checkpoint:
                key = _resolve(checkpoint)
                existing = by_path.get(key)
                if existing is None or _benchmark_sort_key(entry) > _benchmark_sort_key(existing):
                    by_path[key] = dict(entry)
            if stem:
                existing_stem = by_stem.get(stem)
                if existing_stem is None or _benchmark_sort_key(entry) > _benchmark_sort_key(existing_stem):
                    by_stem[stem] = dict(entry)
    return by_path, by_stem


def _benchmark_sort_key(entry: Dict[str, Any]) -> Tuple[float, ...]:
    """Pick the most trustworthy report for a checkpoint.

    A full, gate-passing run against the hardest baseline says more than a high
    win rate against a weaker one.
    """
    completion = _as_float(entry.get("completion_rate")) or 0.0
    baseline = str(entry.get("baseline", ""))
    return (
        1.0 if completion >= _MIN_BENCHMARK_COMPLETION_RATE else 0.0,
        float(_BASELINE_RANKS.get(baseline, _UNRANKED_BASELINE)),
        1.0 if entry.get("gate_passed") else 0.0,
        float(entry.get("planned_games") or 0),
        _as_float(entry.get("first_place_rate")) or 0.0,
    )


def _benchmark_tier(entry: Dict[str, Any]) -> int:
    completion = _as_float(entry.get("completion_rate")) or 0.0
    if completion < _MIN_BENCHMARK_COMPLETION_RATE:
        return TIER_UNRANKED
    return TIER_GATE_PASS if entry.get("gate_passed") else TIER_BENCHMARK


def _apply_benchmark(candidate: CheckpointCandidate, entry: Dict[str, Any]) -> None:
    first_place = _as_float(entry.get("first_place_rate"))
    mean_rank = _as_float(entry.get("mean_rank"))
    pairwise = _as_float(entry.get("pairwise_score"))
    candidate.metrics.update(
        {
            "first_place_rate": first_place,
            "mean_rank": mean_rank,
            "pairwise_score": pairwise,
            "gate_passed": bool(entry.get("gate_passed", False)),
            "completed_games": entry.get("completed_games"),
            "baseline": entry.get("baseline"),
            "stage": entry.get("stage"),
            "report_path": entry.get("report_path"),
        }
    )
    games = f"{entry.get('completed_games')}/{entry.get('planned_games')}"
    if (_as_float(entry.get("completion_rate")) or 0.0) < _MIN_BENCHMARK_COMPLETION_RATE:
        # A partially completed run says nothing about strength, so record it for
        # context without letting it count as verification evidence.
        candidate.evidence.append(f"aborted benchmark vs {entry.get('baseline')} ({games} games)")
        return
    candidate.metrics["primary"] = first_place if first_place is not None else 0.0
    # Lower mean rank is better, so invert it into a "higher is better" value.
    candidate.metrics["secondary"] = -mean_rank if mean_rank is not None else 0.0
    candidate.evidence.append(
        f"benchmark stage{entry.get('stage')} vs {entry.get('baseline')} ({games} games)"
    )


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

    candidates: List[CheckpointCandidate] = []
    for path in _iter_checkpoint_files(bases):
        candidate = _build_candidate(path, base_root)

        manifest_entry = manifest_entries.get(candidate.path)
        if manifest_entry is not None:
            metrics, winner = manifest_entry
            _apply_manifest(candidate, metrics, winner=winner)

        benchmark = benchmarks_by_path.get(candidate.path) or benchmarks_by_stem.get(candidate.stem)
        if benchmark is not None and candidate.tier < TIER_TOURNAMENT_CANDIDATE:
            _apply_benchmark(candidate, benchmark)
            candidate.tier = max(candidate.tier, _benchmark_tier(benchmark))

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
