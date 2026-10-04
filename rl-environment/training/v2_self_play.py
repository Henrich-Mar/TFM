"""Four-seat PPO self-play for TFM RL v2.

Live seats share one policy and rollout buffer. Seeded matchup scheduling mixes
four-seat self-play with one live learner against three trusted frozen opponents.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import AwardFundingTeacherPolicy, HeuristicTeacherPolicy
from search.config import SearchConfig
from search.replay_store import SearchReplayStore
from search.search_agent import SearchPolicy
from tournament_manager import TournamentManager
from training.v2_benchmark import benchmark
from v2_runtime import initialize_v2_runtime


def _pretrain_report_allows_ppo(report_path: Path) -> bool:
    """Read reports written by either Python or Windows PowerShell."""
    if not report_path.is_file():
        return False
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    return bool(payload.get("ppo_gate_passed", False))


def _frozen_checkpoint_agent(path: str, agent_id: str) -> RLAgent:
    agent = RLAgent(agent_id=agent_id)
    agent.load_model(path)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.deterministic_actions = True
    return agent


def _bind_frozen_seat(leader: RLAgent, agent_id: str, *, decision_policy=None) -> RLAgent:
    """Create independent seat memory while sharing immutable network weights."""
    seat = RLAgent(agent_id=agent_id, config=leader.config, decision_policy=decision_policy)
    seat.bind_shared_learner(leader)
    seat.train_from_self_play = False
    seat.config.train_from_self_play = False
    seat.ppo_enable = False
    seat.deterministic_actions = True
    return seat


def _shared_champion_pool(path: str, count: int) -> List[RLAgent]:
    leader = _frozen_checkpoint_agent(path, "champion-0")
    seats = [leader]
    for index in range(1, max(1, int(count))):
        seats.append(_bind_frozen_seat(leader, f"champion-{index}"))
    return seats


def _shared_teacher_pool(count: int, seed: int) -> List[RLAgent]:
    # The award-aware teacher funds cheap awards, so its donations give the
    # learner positive examples of a commitment the stock teacher almost never
    # makes. Opt-in so existing training behaviour is unchanged by default.
    award_teacher = str(os.getenv("ALPHAGO_SELFPLAY_AWARD_TEACHER", "0")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    policy_factory = (
        (lambda offset: AwardFundingTeacherPolicy(int(seed) + offset, sample=False))
        if award_teacher
        else (lambda offset: HeuristicTeacherPolicy(int(seed) + offset, sample=False))
    )
    leader = RLAgent(
        agent_id="teacher-0",
        decision_policy=policy_factory(0),
    )
    leader.train_from_self_play = False
    leader.config.train_from_self_play = False
    leader.ppo_enable = False
    leader.deterministic_actions = True
    seats = [leader]
    for index in range(1, max(1, int(count))):
        seats.append(
            _bind_frozen_seat(
                leader,
                f"teacher-{index}",
                decision_policy=policy_factory(index),
            )
        )
    return seats


def _experiment_version() -> str:
    if str(os.getenv("TFM_RL_V4", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return "v4"
    if str(os.getenv("TFM_RL_V3", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return "v3"
    return "v2"


def _load_stage_options(stage: int) -> Dict:
    version = _experiment_version()
    root = Path(__file__).resolve().parents[1]
    path = root / f"game_options.{version}_stage{int(stage)}.json"
    if version == "v4" and not path.is_file():
        path = root / f"game_options.v3_stage{int(stage)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _SelfPlayGame:
    number: int
    seed: int
    stage: int
    lineup: List[RLAgent]
    search_training: bool = False
    lineup_kind: str = "live"
    random_ma_training: bool = False
    random_ma_mode: Optional[str] = None


class V2SelfPlayRunner:
    def __init__(
        self,
        bc_checkpoint: Optional[str],
        root: str,
        benchmark_interval: int = 50_000,
        seed: int = 100_000,
        initial_stage: Optional[int] = None,
        from_scratch: bool = False,
    ) -> None:
        if from_scratch == bool(bc_checkpoint):
            raise ValueError("choose exactly one of a BC checkpoint or random initialization")
        self.paths = initialize_v2_runtime()
        self.is_v4 = _experiment_version() == "v4"
        self.is_v3 = self.is_v4 or str(os.getenv("TFM_RL_V3", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self.version = _experiment_version()
        if self.is_v4:
            from training.v4_gates import assert_ppo_unlocked
            assert_ppo_unlocked()
        # Subsequent benchmark subprocess-equivalent calls are intentional resumes.
        os.environ[f"{self.version.upper()}_ALLOW_RESUME"] = "1"
        self.root = Path(root).expanduser().resolve()
        if self.root != Path(self.paths["root"]).resolve():
            raise RuntimeError(f"--root must exactly match TFM_RL_{self.version.upper()}_ROOT")
        self.checkpoints = self.root / "checkpoints"
        self.benchmarks = self.root / "benchmarks"
        self.metrics = self.root / "metrics"
        for path in (self.checkpoints, self.benchmarks, self.metrics):
            path.mkdir(parents=True, exist_ok=True)
        if bc_checkpoint is not None:
            report_path = Path(bc_checkpoint).with_name("pretrain_report.json")
            if not _pretrain_report_allows_ppo(report_path):
                raise RuntimeError("PPO is blocked until the BC pretrain_report.json has ppo_gate_passed=true")
        self.state_path = self.metrics / "selfplay_state.json"
        self.latest_learner_path = self.checkpoints / "latest_learner.pth"
        resume_state: Dict = {}
        if self.state_path.is_file() != self.latest_learner_path.is_file():
            raise RuntimeError(
                f"incomplete {self.version} resume state: selfplay_state.json and latest_learner.pth must both exist"
            )
        if self.state_path.is_file() and self.latest_learner_path.is_file():
            resume_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        learner_source = str(self.latest_learner_path) if resume_state else bc_checkpoint
        self.learner = RLAgent(agent_id=f"{self.version}-main-learner")
        if learner_source is not None:
            self.learner.load_model(learner_source)
        else:
            print(
                "[selfplay] initializing policy from random weights; no behavior-cloning gate applies",
                flush=True,
            )
        self.learner.train_from_self_play = True
        self.learner.config.train_from_self_play = True
        self.learner.ppo_enable = True
        self.learner.deterministic_actions = False
        if resume_state and self.learner.rollout_shard_store is not None:
            quarantine = self.learner.rollout_shard_store.quarantine_incompatible(
                expected_schema_version=str(self.learner.state_schema_version or "v1"),
                policy_version=int(self.learner.policy_version),
            )
            if int(quarantine["steps"]) > 0:
                print(
                    f"[selfplay] quarantined stale rollouts steps={quarantine['steps']} "
                    f"shards={quarantine['shards']} policy_version={self.learner.policy_version}",
                    flush=True,
                )
        configured_initial_stage = 0 if initial_stage is None else int(initial_stage)
        self.stage = int(resume_state.get("stage", configured_initial_stage) or 0)
        if self.stage not in (0, 1):
            raise ValueError(f"unsupported self-play stage: {self.stage}")
        from v2_runtime import assert_stage_allowed

        assert_stage_allowed(self.stage, context="v2 self-play")
        self.seed_cursor = int(resume_state.get("seed_cursor", seed) or seed)
        environment_root = Path(__file__).resolve().parents[1]
        benchmark_seed_path = environment_root / "benchmark_seeds.v1.json"
        configured_screen_seed_path = str(os.getenv("BENCHMARK_SCREEN_SEEDS_PATH", "") or "").strip()
        self.screen_seed_path = (
            Path(configured_screen_seed_path).expanduser().resolve()
            if configured_screen_seed_path
            else environment_root / "benchmark_screen_seeds.v1.json"
        )
        self.reserved_benchmark_seeds = set()
        for reserved_path in (benchmark_seed_path, self.screen_seed_path):
            reserved_payload = json.loads(reserved_path.read_text(encoding="utf-8"))
            self.reserved_benchmark_seeds.update(int(item) for item in reserved_payload.get("seeds", []))
        self.benchmark_interval = max(1, int(benchmark_interval))
        self.screen_min_completion_ratio = min(
            1.0,
            max(0.0, float(os.getenv("BENCHMARK_SCREEN_MIN_COMPLETION_RATIO", "0.90"))),
        )
        self.screen_teacher_min_first_place_rate = min(
            1.0,
            max(0.0, float(os.getenv("BENCHMARK_SCREEN_TEACHER_MIN_FIRST_PLACE_RATE", "0.25"))),
        )
        self.screen_champion_min_pairwise_score = min(
            1.0,
            max(0.0, float(os.getenv("BENCHMARK_SCREEN_CHAMPION_MIN_PAIRWISE_SCORE", "0.50"))),
        )
        self.history_snapshot_interval_updates = max(
            1,
            int(os.getenv("HISTORY_SNAPSHOT_INTERVAL_UPDATES", "8")),
        )
        self.lineup_live_fraction = self._fraction_env("SELFPLAY_LIVE_FRACTION", 0.40)
        self.lineup_champion_fraction = self._fraction_env("SELFPLAY_CHAMPION_FRACTION", 0.40)
        self.lineup_history_fraction = self._fraction_env("SELFPLAY_HISTORY_FRACTION", 0.0)
        self.lineup_teacher_fraction = self._fraction_env("SELFPLAY_TEACHER_FRACTION", 0.20)
        lineup_total = (
            self.lineup_live_fraction
            + self.lineup_champion_fraction
            + self.lineup_history_fraction
            + self.lineup_teacher_fraction
        )
        if abs(lineup_total - 1.0) > 1e-6:
            raise ValueError(
                "SELFPLAY_LIVE_FRACTION + SELFPLAY_CHAMPION_FRACTION + SELFPLAY_HISTORY_FRACTION + "
                f"SELFPLAY_TEACHER_FRACTION must equal 1.0; got {lineup_total:.6f}"
            )
        self.random_ma_selfplay_fraction = self._fraction_env(
            "ALPHAGO_RANDOM_MA_SELFPLAY_FRACTION", 0.0
        )
        self.random_ma_mode = str(
            os.getenv("ALPHAGO_RANDOM_MA_MODE", "Limited synergy")
        ).strip()
        if self.random_ma_mode not in {"Limited synergy", "Full random"}:
            raise ValueError(
                "ALPHAGO_RANDOM_MA_MODE must be 'Limited synergy' or 'Full random'"
            )
        self.random_ma_force_award_teacher = self._bool_env(
            "ALPHAGO_RANDOM_MA_FORCE_AWARD_TEACHER", False
        )
        if (
            self.random_ma_force_award_teacher
            and self.random_ma_selfplay_fraction > self.lineup_teacher_fraction + 1e-9
        ):
            raise ValueError(
                "ALPHAGO_RANDOM_MA_SELFPLAY_FRACTION cannot exceed "
                "SELFPLAY_TEACHER_FRACTION when award-teacher routing is forced"
            )
        if (
            self.random_ma_force_award_teacher
            and self.random_ma_selfplay_fraction > 0.0
            and not self._bool_env("ALPHAGO_SELFPLAY_AWARD_TEACHER", False)
        ):
            raise ValueError(
                "ALPHAGO_RANDOM_MA_FORCE_AWARD_TEACHER requires "
                "ALPHAGO_SELFPLAY_AWARD_TEACHER=1"
            )
        # History snapshots are training sparring partners only; promotion
        # still compares exclusively against the trusted champion and teacher.
        self.load_unpromoted_history = self.lineup_history_fraction > 0.0
        self.promotion_teacher_pairwise_margin = max(
            0.0,
            self._float_env("PROMOTION_TEACHER_PAIRWISE_MARGIN", 0.0),
        )
        self.rollback_pairwise_floor = min(
            1.0,
            max(0.0, self._float_env("PPO_ROLLBACK_PAIRWISE_FLOOR", 0.35)),
        )
        self.rollback_failure_limit = max(
            1,
            int(os.getenv("PPO_ROLLBACK_FAILURE_LIMIT", "2")),
        )
        self.rollback_confirm_full = str(os.getenv("PPO_ROLLBACK_CONFIRM_FULL", "1")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.rollback_confirm_pairwise = min(
            1.0,
            max(0.0, self._float_env("PPO_ROLLBACK_CONFIRM_PAIRWISE", 0.45)),
        )
        self.benchmark_candidate_stochastic = str(
            os.getenv("BENCHMARK_CANDIDATE_STOCHASTIC", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.consecutive_champion_screen_failures = int(
            resume_state.get("consecutive_champion_screen_failures", 0) or 0
        )
        self.champion_promotions = int(
            resume_state.get(
                "champion_promotions",
                sum(1 for report in (resume_state.get("reports", []) or []) if report.get("promoted")),
            )
            or 0
        )
        try:
            self.selfplay_concurrency = max(1, int(os.getenv("SELFPLAY_CONCURRENCY", "1")))
        except (TypeError, ValueError):
            self.selfplay_concurrency = 1
        resumed_decisions = int(resume_state.get("decisions", 0) or 0)
        self.decision_offset = resumed_decisions
        self.next_benchmark_decision = ((resumed_decisions // self.benchmark_interval) + 1) * self.benchmark_interval
        self.progress_path = self.metrics / "selfplay_progress.json"
        self.game_count = 0
        self.lifetime_games = max(
            int(resume_state.get("games", 0) or 0),
            int(getattr(self.learner, "games_played", 0) or 0),
        )
        self.learner.games_played = int(self.lifetime_games)
        self.search_game_count = 0
        self.random_ma_game_count = int(resume_state.get("random_ma_games", 0) or 0)
        self.random_ma_games_since_update = 0
        self.lineup_game_counts: Dict[str, int] = {
            "live": 0,
            "champion": 0,
            "history": 0,
            "teacher": 0,
            "search": 0,
        }
        self.lineup_games_since_update: Dict[str, int] = dict(self.lineup_game_counts)
        try:
            self.search_selfplay_fraction = min(
                1.0,
                max(0.0, float(os.getenv("ALPHAGO_SEARCH_SELFPLAY_FRACTION", "0"))),
            )
        except (TypeError, ValueError):
            self.search_selfplay_fraction = 0.0
        self.search_replay_store: Optional[SearchReplayStore] = None
        # Frozen teacher/champion seats never reach the strict on-policy PPO
        # buffer, so every award they correctly fund is thrown away. Donation
        # captures those decisions for the distillation stream instead.
        self.teacher_replay_store: Optional[SearchReplayStore] = None
        teacher_replay_dir = str(os.getenv("ALPHAGO_TEACHER_REPLAY_DIR", "") or "").strip()
        if teacher_replay_dir:
            try:
                teacher_replay_max_shards = max(
                    1, int(os.getenv("ALPHAGO_TEACHER_REPLAY_MAX_SHARDS", "2048"))
                )
            except (TypeError, ValueError):
                teacher_replay_max_shards = 2048
            self.teacher_replay_store = SearchReplayStore(
                Path(teacher_replay_dir).expanduser(),
                max_shards=teacher_replay_max_shards,
            )
            print(
                f"[selfplay] teacher donation replay enabled dir={teacher_replay_dir}",
                flush=True,
            )
        if self.search_selfplay_fraction > 0.0:
            replay_dir = Path(
                os.getenv("ALPHAGO_SEARCH_REPLAY_DIR", str(self.root / "search-replay"))
            ).expanduser()
            try:
                search_replay_max_shards = max(
                    1, int(os.getenv("ALPHAGO_SEARCH_REPLAY_MAX_SHARDS", "2048"))
                )
            except (TypeError, ValueError):
                search_replay_max_shards = 2048
            self.search_replay_store = SearchReplayStore(replay_dir, max_shards=search_replay_max_shards)
        self.champion_path = self.checkpoints / "champion.pth"
        if not self.champion_path.is_file():
            if bc_checkpoint is None:
                self.learner.save_model(str(self.champion_path))
            else:
                shutil.copy2(bc_checkpoint, self.champion_path)
        self.history: List[str] = [str(item) for item in (resume_state.get("history", []) or []) if Path(str(item)).is_file()]
        self.last_history_snapshot_policy_version = int(
            resume_state.get("last_history_snapshot_policy_version", 0) or 0
        )
        self.previous_reports: List[Dict] = list(resume_state.get("reports", []) or [])
        self.recovery_events: List[Dict] = list(resume_state.get("recovery_events", []) or [])
        self.champion_teacher_report: Optional[Dict] = resume_state.get("champion_teacher_report")
        if self.champion_teacher_report is None:
            promoted_teacher_reports = [
                report.get("teacher")
                for report in self.previous_reports
                if report.get("promoted") and isinstance(report.get("teacher"), dict)
            ]
            if promoted_teacher_reports:
                self.champion_teacher_report = promoted_teacher_reports[-1]
        if not self._teacher_baseline_is_compatible(self.champion_teacher_report):
            self.champion_teacher_report = None
        self.historical_pool: List[RLAgent] = []
        self.historical_seats_by_snapshot: List[List[RLAgent]] = []
        self.historical_snapshot_paths: List[str] = []
        self._historical_snapshot_cursors: List[int] = []
        self.champion_pool: List[RLAgent] = []
        self.teacher_pool: List[RLAgent] = []
        self._historical_cursor = 0
        self._champion_cursor = 0
        self._teacher_cursor = 0
        self._seat_cursor = 0
        seat_count = max(4, int(self.selfplay_concurrency) * 4)
        self.learning_seats = [
            self._make_learning_seat(f"{self.version}-self-{idx}")
            for idx in range(seat_count)
        ]
        servers = [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
        self.cluster = GameServerCluster(servers)
        self.cluster.base_game_options = _load_stage_options(self.stage)
        self.manager = TournamentManager(self.cluster)
        self.ppo_metrics_path = self.metrics / "ppo_updates.jsonl"
        self._refresh_frozen_pools()
        self._write_progress("started")
        print(
            f"[selfplay] started stage={self.stage} decisions={self._total_decisions()} "
            f"next_benchmark={self.next_benchmark_decision} concurrency={self.selfplay_concurrency} "
            f"server_slots={len(self.cluster.servers) * max(1, self.cluster.max_active_games_per_server)} "
            f"ppo_lr={self.learner.optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )
        print(
            f"[selfplay] lineup=live:{self.lineup_live_fraction:.2f} "
            f"champion_1v3:{self.lineup_champion_fraction:.2f} "
            f"history_1v3:{self.lineup_history_fraction:.2f} "
            f"teacher_1v3:{self.lineup_teacher_fraction:.2f} "
            f"search_game_fraction={self.search_selfplay_fraction:.3f} "
            f"random_ma_fraction={self.random_ma_selfplay_fraction:.3f} "
            f"random_ma_mode={self.random_ma_mode!r} "
            f"random_ma_force_award_teacher={self.random_ma_force_award_teacher} "
            f"history_snapshots={len(self.historical_snapshot_paths)} "
            f"eval_candidate={'sample' if self.benchmark_candidate_stochastic else 'argmax'}",
            flush=True,
        )
        champion_teacher = self.champion_teacher_report or {}
        print(
            "[selfplay] promotion teacher baseline "
            f"pairwise={champion_teacher.get('pairwise_score', 'none')} "
            f"margin={self.promotion_teacher_pairwise_margin:.3f}",
            flush=True,
        )

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        fallback = "1" if default else "0"
        return str(os.getenv(name, fallback)).strip().lower() in {
            "1", "true", "yes", "on",
        }

    @classmethod
    def _fraction_env(cls, name: str, default: float) -> float:
        return min(1.0, max(0.0, cls._float_env(name, default)))

    def _make_learning_seat(self, agent_id: str) -> RLAgent:
        seat = RLAgent(agent_id=agent_id, config=self.learner.config)
        seat.bind_shared_learner(self.learner)
        return seat

    def _reset_seat_memory(self, agent: RLAgent) -> None:
        for name in (
            "_recurrent_hidden_by_player",
            "_turn_action_count_by_player",
            "_last_phase_by_player",
        ):
            store = getattr(agent, name, None)
            if isinstance(store, dict):
                store.clear()

    def _take_learning_seats(self, count: int) -> List[RLAgent]:
        picked: List[RLAgent] = []
        for _ in range(int(count)):
            seat = self.learning_seats[self._seat_cursor % len(self.learning_seats)]
            self._seat_cursor += 1
            self._reset_seat_memory(seat)
            picked.append(seat)
        return picked

    def _take_historical_seats(self, rng: random.Random, count: int) -> List[RLAgent]:
        """Seat ``count`` distinct snapshots so no snapshot plays twice in one game."""
        pools = self.historical_seats_by_snapshot
        if len(pools) < int(count):
            raise RuntimeError(f"history pool has {len(pools)} snapshots; {count} required")
        picked: List[RLAgent] = []
        for snapshot_index in rng.sample(range(len(pools)), int(count)):
            cursor = self._historical_snapshot_cursors[snapshot_index]
            seat = pools[snapshot_index][cursor % len(pools[snapshot_index])]
            self._historical_snapshot_cursors[snapshot_index] = cursor + 1
            self._reset_seat_memory(seat)
            picked.append(seat)
        return picked

    def _history_lineup_available(self) -> bool:
        return len(getattr(self, "historical_seats_by_snapshot", []) or []) >= 3

    def _refresh_history_pool(self) -> None:
        """Load the most recent snapshots as frozen sparring partners."""
        recent = self.history[-8:]
        if not recent or not bool(getattr(self, "load_unpromoted_history", False)):
            self.historical_pool = []
            self.historical_seats_by_snapshot = []
            self.historical_snapshot_paths = []
            self._historical_snapshot_cursors = []
            return
        if recent == list(getattr(self, "historical_snapshot_paths", []) or []):
            return
        # Each game seats a snapshot at most once, so one seat per concurrent
        # game keeps every in-flight seat's recurrent memory independent.
        seats_per_snapshot = max(1, int(self.selfplay_concurrency))
        loaded = dict(
            zip(
                getattr(self, "historical_snapshot_paths", []) or [],
                getattr(self, "historical_seats_by_snapshot", []) or [],
            )
        )
        by_snapshot: List[List[RLAgent]] = []
        for path in recent:
            if path in loaded:
                by_snapshot.append(loaded[path])
                continue
            label = Path(path).stem
            leader = _frozen_checkpoint_agent(path, f"historical-{label}-0")
            seats = [leader]
            for seat_index in range(1, seats_per_snapshot):
                seats.append(_bind_frozen_seat(leader, f"historical-{label}-{seat_index}"))
            by_snapshot.append(seats)
        self.historical_seats_by_snapshot = by_snapshot
        self.historical_pool = [seat for seats in by_snapshot for seat in seats]
        self.historical_snapshot_paths = list(recent)
        self._historical_snapshot_cursors = [0] * len(by_snapshot)
        self._apply_v3_feature_scale()

    def _take_frozen_seats(self, kind: str, count: int) -> List[RLAgent]:
        if kind == "champion":
            pool = self.champion_pool
            cursor_name = "_champion_cursor"
        elif kind == "teacher":
            pool = self.teacher_pool
            cursor_name = "_teacher_cursor"
        else:
            raise ValueError(f"unsupported frozen opponent kind: {kind}")
        if len(pool) < int(count):
            raise RuntimeError(f"{kind} pool has {len(pool)} seats; {count} required")
        cursor = int(getattr(self, cursor_name, 0))
        picked: List[RLAgent] = []
        for _ in range(int(count)):
            seat = pool[cursor % len(pool)]
            cursor += 1
            self._reset_seat_memory(seat)
            picked.append(seat)
        setattr(self, cursor_name, cursor)
        return picked

    def _refresh_frozen_pools(self) -> None:
        """Refresh trusted champion/teacher seats with independent recurrent memory."""
        self._refresh_history_pool()
        trusted_seats = max(3, int(self.selfplay_concurrency) * 3)
        champion_path = getattr(self, "champion_path", None)
        if champion_path is not None and Path(champion_path).is_file():
            self.champion_pool = _shared_champion_pool(str(champion_path), trusted_seats)
        else:
            self.champion_pool = []
        self.teacher_pool = _shared_teacher_pool(
            trusted_seats,
            int(getattr(self, "seed_cursor", 0)) ^ 0x5EA7,
        )
        self._historical_cursor = 0
        self._champion_cursor = 0
        self._teacher_cursor = 0
        self._apply_v3_feature_scale()

    def _current_v3_feature_scale(self) -> float:
        if getattr(self, "is_v4", False):
            return 1.0
        if not getattr(self, "is_v3", False):
            return 0.0
        try:
            ramp_decisions = max(1, int(os.getenv("V3_FEATURE_RAMP_DECISIONS", "25000")))
        except (TypeError, ValueError):
            ramp_decisions = 25_000
        return max(0.0, min(float(self._total_decisions()) / float(ramp_decisions), 1.0))

    def _apply_v3_feature_scale(self) -> None:
        if not getattr(self, "is_v3", False):
            return
        scale = self._current_v3_feature_scale()
        self.learner.set_v3_feature_scale(scale)
        for agent in [
            *getattr(self, "learning_seats", []),
            *self.historical_pool,
            *getattr(self, "champion_pool", []),
        ]:
            agent.set_v3_feature_scale(scale)

    def _total_decisions(self) -> int:
        seats = getattr(self, "learning_seats", None) or []
        if seats:
            current_run = sum(
                int(seat.get_behavior_stats().get("total_decisions", 0))
                for seat in seats
            )
        else:
            current_run = int(self.learner.get_behavior_stats().get("total_decisions", 0))
        return int(self.decision_offset + current_run)

    def _write_progress(
        self,
        status: str,
        game_seed: Optional[int] = None,
        game_elapsed: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "schema_version": f"tfm_rl_{self.version}.selfplay_progress.v1",
            "status": str(status),
            "stage": int(self.stage),
            "games": int(getattr(self, "lifetime_games", self.game_count)),
            "search_games": int(self.search_game_count),
            "random_ma_games": int(getattr(self, "random_ma_game_count", 0)),
            "decisions": int(self._total_decisions()),
            "next_benchmark_decision": int(self.next_benchmark_decision),
            "seed_cursor": int(self.seed_cursor),
            "rollout_buffer": int(self.learner.get_rollout_buffer_size()),
            "game_seed": game_seed,
            "game_elapsed_sec": game_elapsed,
            "error": error,
            "updated_at": time.time(),
        }
        if self.is_v3:
            payload["v3_feature_scale"] = self._current_v3_feature_scale()
        temporary = self.progress_path.with_name(self.progress_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.progress_path)

    def _save_resume_state(self, reports: List[Dict], *, save_model: bool = True) -> None:
        """Atomically refresh the durable learner and self-play state."""
        if save_model:
            learner_temporary = self.latest_learner_path.with_name(self.latest_learner_path.name + ".tmp")
            self.learner.save_model(str(learner_temporary))
            os.replace(learner_temporary, self.latest_learner_path)
        state = {
            "schema_version": f"tfm_rl_{self.version}.selfplay_state.v1",
            "stage": self.stage,
            "decisions": self._total_decisions(),
            "seed_cursor": self.seed_cursor,
            "policy_version": self.learner.policy_version,
            "games": int(getattr(self, "lifetime_games", self.game_count)),
            "random_ma_games": int(getattr(self, "random_ma_game_count", 0)),
            "champion": str(self.champion_path),
            "history": self.history,
            "last_history_snapshot_policy_version": int(self.last_history_snapshot_policy_version),
            "champion_promotions": int(getattr(self, "champion_promotions", 0)),
            "consecutive_champion_screen_failures": int(
                getattr(self, "consecutive_champion_screen_failures", 0)
            ),
            "recovery_events": list(getattr(self, "recovery_events", [])),
            "champion_teacher_report": getattr(self, "champion_teacher_report", None),
            "reports": reports,
        }
        state_temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        state_temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(state_temporary, self.state_path)

    def _draw_lineup_kind(self, draw: float) -> str:
        live_fraction = float(getattr(self, "lineup_live_fraction", 1.0))
        champion_fraction = float(getattr(self, "lineup_champion_fraction", 0.0))
        history_fraction = float(getattr(self, "lineup_history_fraction", 0.0))
        if draw < live_fraction:
            return "live"
        if draw < live_fraction + champion_fraction:
            return "champion"
        if draw < live_fraction + champion_fraction + history_fraction:
            # Until three snapshots exist, history games fall back to the champion.
            return "history" if self._history_lineup_available() else "champion"
        return "teacher"

    def _lineup(self, game_seed: int) -> List[RLAgent]:
        """Choose a deterministic live, champion, history, or teacher training matchup."""
        rng = random.Random(int(game_seed) ^ 0x5F3759DF)
        kind = self._draw_lineup_kind(rng.random())
        if kind == "live":
            return self._take_learning_seats(4)
        live_seat = int(rng.randrange(4))
        if kind == "history":
            opponents = self._take_historical_seats(rng, 3)
        else:
            opponents = self._take_frozen_seats(kind, 3)
        lineup = list(opponents)
        lineup.insert(live_seat, self._take_learning_seats(1)[0])
        return lineup

    def _lineup_kind(self, game_seed: int, search_training: bool = False) -> str:
        if search_training:
            return "search"
        return self._draw_lineup_kind(random.Random(int(game_seed) ^ 0x5F3759DF).random())

    def _is_random_ma_game(
        self,
        game_seed: int,
        lineup_kind: str,
        *,
        search_training: bool = False,
    ) -> bool:
        """Select a reproducible random-MA cohort without changing lineup ratios."""
        fraction = float(getattr(self, "random_ma_selfplay_fraction", 0.0))
        if search_training or fraction <= 0.0:
            return False
        force_teacher = bool(getattr(self, "random_ma_force_award_teacher", False))
        if force_teacher:
            if lineup_kind != "teacher":
                return False
            teacher_fraction = float(getattr(self, "lineup_teacher_fraction", 0.0))
            if teacher_fraction <= 0.0:
                return False
            # The configured fraction is global. Restricting selection to the
            # teacher slice requires this conditional probability within it.
            fraction = min(1.0, fraction / teacher_fraction)
        draw = random.Random(int(game_seed) ^ 0x4D41524D).random()
        return draw < fraction

    def _remember_history(self, checkpoint: Path) -> None:
        checkpoint_text = str(checkpoint)
        self.history = [item for item in self.history if item != checkpoint_text]
        self.history.append(checkpoint_text)
        self.history = self.history[-8:]

    def _maybe_archive_history_snapshot(self, decisions: int) -> Optional[str]:
        """Archive learner weights for diagnostics without trusting them as opponents."""
        policy_version = int(self.learner.policy_version)
        if policy_version - int(self.last_history_snapshot_policy_version) < self.history_snapshot_interval_updates:
            return None
        target = self.checkpoints / (
            f"history_policy_{policy_version:06d}_decisions_{int(decisions):09d}.pth"
        )
        temporary = target.with_name(target.name + ".tmp")
        self.learner.save_model(str(temporary))
        os.replace(temporary, target)
        self._remember_history(target)
        self.last_history_snapshot_policy_version = policy_version
        if bool(getattr(self, "load_unpromoted_history", False)):
            self._refresh_history_pool()
        print(
            f"[selfplay] historical snapshot saved policy_version={policy_version} "
            f"decisions={decisions} retained={len(self.history)}",
            flush=True,
        )
        return str(target)

    def _append_ppo_metrics(self, metrics: Dict[str, Any], decisions: int, elapsed: float) -> None:
        payload: Dict[str, Any] = {
            "schema_version": f"tfm_rl_{self.version}.ppo_update.v1",
            "timestamp": time.time(),
            "decisions": int(decisions),
            "games": int(self.lifetime_games),
            "policy_version": int(self.learner.policy_version),
            "elapsed_sec": float(elapsed),
            "lineups": dict(self.lineup_games_since_update),
            "random_ma_games": int(getattr(self, "random_ma_games_since_update", 0)),
            "reward_shaping_coef": float(self.learner._current_reward_shaping_coef()),
            "entropy_coef": float(self.learner._current_ppo_entropy_coef()),
        }
        for key, value in dict(metrics or {}).items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                payload[str(key)] = value
            elif hasattr(value, "item"):
                payload[str(key)] = value.item()
        with self.ppo_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.lineup_games_since_update = {key: 0 for key in self.lineup_games_since_update}
        self.random_ma_games_since_update = 0

    def _rebind_learning_seats(self) -> None:
        for seat in self.learning_seats:
            seat.bind_shared_learner(self.learner)
            self._reset_seat_memory(seat)

    async def _rollback_to_champion(self, decisions: int, reason: str) -> Dict[str, Any]:
        failed_version = int(self.learner.policy_version)
        quarantine = await self.learner.quarantine_rollout_buffer(
            f"rollback_policy_{failed_version:06d}"
        )
        self.learner.load_model(str(self.champion_path))
        restored_version = int(self.learner.policy_version)
        self.learner.policy_version = max(failed_version, restored_version) + 1
        self.learner.games_played = int(self.lifetime_games)
        self._rebind_learning_seats()
        self._refresh_frozen_pools()
        self.consecutive_champion_screen_failures = 0
        event = {
            "applied": True,
            "reason": str(reason),
            "decisions": int(decisions),
            "failed_policy_version": failed_version,
            "restored_checkpoint_policy_version": restored_version,
            "new_policy_version": int(self.learner.policy_version),
            "quarantined_rollout_steps": int(quarantine.get("steps", 0)),
            "quarantine_path": str(quarantine.get("path", "")),
        }
        print(
            f"[selfplay] rollback applied reason={reason} failed_version={failed_version} "
            f"new_version={self.learner.policy_version} "
            f"quarantined_rollouts={quarantine.get('steps', 0)}",
            flush=True,
        )
        return event

    async def _benchmark(self, *args, **kwargs):
        """Run policy-aware gates without giving either neural policy an argmax edge."""
        baseline = kwargs.get("baseline")
        if baseline is None and len(args) >= 2:
            baseline = args[1]
        if bool(getattr(self, "benchmark_candidate_stochastic", False)):
            if baseline == "champion":
                # The promotion/rollback gate compares like with like: both
                # neural policies sample from their exact PPO behavior policy.
                kwargs.setdefault("stochastic", True)
            else:
                # The heuristic teacher is intentionally deterministic, while
                # the candidate uses the policy distribution PPO optimized.
                kwargs.setdefault("candidate_stochastic", True)
        return await benchmark(*args, **kwargs)

    def _teacher_baseline_is_compatible(self, report: Any) -> bool:
        if not isinstance(report, dict):
            return False
        expected = (
            "candidate_sample"
            if bool(getattr(self, "benchmark_candidate_stochastic", False))
            else "argmax"
        )
        return report.get("action_selection") == expected

    async def _ensure_champion_teacher_baseline(self) -> Dict[str, Any]:
        """Rebuild teacher strength for the champion under the current eval policy."""
        baseline = getattr(self, "champion_teacher_report", None)
        if self._teacher_baseline_is_compatible(baseline):
            return baseline
        print(
            "[selfplay] calibrating champion teacher baseline for current action selection",
            flush=True,
        )
        baseline = await self._benchmark(
            str(self.champion_path),
            "teacher",
            self.stage,
            str(self.benchmarks),
            report_label="champion_baseline",
        )
        self.champion_teacher_report = baseline
        return baseline

    async def _confirm_rollback(
        self,
        candidate_path: Optional[Path],
        decisions: int,
        trigger: str,
    ) -> Dict[str, Any]:
        """Re-test a screen-triggered rollback on the full 30-deal champion benchmark.

        The 8-deal screen swings far enough between near-identical policies
        that one bad screen is not sufficient evidence to discard ~100k decisions.
        """
        if candidate_path is None or not bool(getattr(self, "rollback_confirm_full", True)):
            return await self._rollback_to_champion(decisions, trigger)
        print(f"[selfplay] rollback trigger={trigger}; confirming on full champion benchmark", flush=True)
        confirmation = await self._benchmark(
            str(candidate_path),
            "champion",
            self.stage,
            str(self.benchmarks),
            champion=str(self.champion_path),
            report_label="rollback_confirm",
        )
        confirm_pairwise = float(confirmation.get("pairwise_score", 0.0) or 0.0)
        confirm_threshold = float(getattr(self, "rollback_confirm_pairwise", 0.45))
        planned = max(1, int(confirmation.get("planned_games", 0) or 0))
        completed = max(0, int(confirmation.get("completed_games", 0) or 0))
        trustworthy = (
            completed >= math.ceil(self.screen_min_completion_ratio * planned)
            and int(confirmation.get("rejection_count", 0) or 0) == 0
        )
        summary = {
            "trigger": trigger,
            "pairwise_score": confirm_pairwise,
            "threshold": confirm_threshold,
            "completed_games": completed,
            "planned_games": planned,
        }
        if trustworthy and confirm_pairwise >= confirm_threshold:
            self.consecutive_champion_screen_failures = 0
            print(
                f"[selfplay] rollback cleared full_pairwise={confirm_pairwise:.3f} "
                f"threshold={confirm_threshold:.3f}",
                flush=True,
            )
            return {"applied": False, "reason": "full_benchmark_cleared", "confirmation": summary}
        event = await self._rollback_to_champion(
            decisions,
            f"{trigger}_confirmed_full_pairwise_{confirm_pairwise:.3f}",
        )
        event["confirmation"] = summary
        return event

    async def _maybe_rollback_from_screen(
        self,
        report: Dict[str, Any],
        decisions: int,
        candidate_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if int(getattr(self, "champion_promotions", 0)) <= 0:
            return {"applied": False, "reason": "no_promoted_champion"}
        pairwise = float(report.get("pairwise_score", 0.0) or 0.0)
        threshold = float(self.screen_champion_min_pairwise_score)
        if pairwise >= threshold:
            self.consecutive_champion_screen_failures = 0
            return {"applied": False, "reason": "screen_passed"}
        self.consecutive_champion_screen_failures += 1
        if pairwise < float(self.rollback_pairwise_floor):
            return await self._confirm_rollback(
                candidate_path,
                decisions,
                f"pairwise_{pairwise:.3f}_below_floor_{self.rollback_pairwise_floor:.3f}",
            )
        if self.consecutive_champion_screen_failures >= int(self.rollback_failure_limit):
            return await self._confirm_rollback(
                candidate_path,
                decisions,
                f"champion_screen_failed_{self.consecutive_champion_screen_failures}_times",
            )
        return {
            "applied": False,
            "reason": "waiting_for_failure_limit",
            "consecutive_failures": int(self.consecutive_champion_screen_failures),
        }

    def _screen_report_promising(self, report: Dict, baseline: str) -> bool:
        planned = max(1, int(report.get("planned_games", 0) or 0))
        completed = max(0, int(report.get("completed_games", 0) or 0))
        if completed < math.ceil(self.screen_min_completion_ratio * planned):
            return False
        if int(report.get("rejection_count", 0) or 0) != 0:
            return False
        if baseline == "teacher":
            return float(report.get("first_place_rate", 0.0) or 0.0) >= self.screen_teacher_min_first_place_rate
        if baseline == "champion":
            return float(report.get("pairwise_score", 0.0) or 0.0) >= self.screen_champion_min_pairwise_score
        raise ValueError(f"unsupported screening baseline: {baseline}")

    @staticmethod
    def _rolling_screen_summary(reports: List[Dict], window: int = 3) -> str:
        """A single 32-game screen has roughly +/-0.15 noise; trends need a window."""
        recent = [report for report in reports if isinstance(report.get("screen_teacher"), dict)][-window:]

        def mean(key: str, field: str) -> str:
            values = [
                float((report.get(key) or {}).get(field, 0.0) or 0.0)
                for report in recent
                if isinstance(report.get(key), dict)
            ]
            return f"{sum(values) / len(values):.3f}" if values else "none"

        return (
            f"[selfplay] rolling screens last={len(recent)} "
            f"teacher_pairwise={mean('screen_teacher', 'pairwise_score')} "
            f"teacher_first={mean('screen_teacher', 'first_place_rate')} "
            f"teacher_vp={mean('screen_teacher', 'mean_relative_vp_margin')} "
            f"champion_pairwise={mean('screen_regression', 'pairwise_score')}"
        )

    def _teacher_relative_to_champion(self, report: Dict) -> Dict[str, Any]:
        """Reject candidates that beat the champion yet play worse against the teacher.

        Both full teacher reports use the same promotion seeds and seat
        rotations, so the comparison is paired on identical games.
        """
        baseline = getattr(self, "champion_teacher_report", None)
        margin = float(getattr(self, "promotion_teacher_pairwise_margin", 0.0))
        candidate_pairwise = float(report.get("pairwise_score", 0.0) or 0.0)
        if not isinstance(baseline, dict):
            return {
                "passed": True,
                "reason": "no_champion_teacher_baseline",
                "candidate_pairwise": candidate_pairwise,
                "champion_pairwise": None,
                "margin": margin,
                "paired": False,
            }
        champion_pairwise = float(baseline.get("pairwise_score", 0.0) or 0.0)
        passed = candidate_pairwise >= champion_pairwise - margin
        return {
            "passed": bool(passed),
            "reason": "not_worse_than_champion" if passed else "worse_than_champion_vs_teacher",
            "candidate_pairwise": candidate_pairwise,
            "champion_pairwise": champion_pairwise,
            "candidate_vp_margin": report.get("mean_relative_vp_margin"),
            "champion_vp_margin": baseline.get("mean_relative_vp_margin"),
            "margin": margin,
            "paired": list(report.get("seeds", []) or []) == list(baseline.get("seeds", []) or []),
        }

    def _reserve_selfplay_game(self) -> _SelfPlayGame:
        """Reserve one game using the current, unmodified learner policy."""
        self.game_count += 1
        seed = self.seed_cursor
        self.seed_cursor += 1
        while seed in self.reserved_benchmark_seeds:
            seed = self.seed_cursor
            self.seed_cursor += 1
        stage = self.stage
        search_training = (
            getattr(self, "search_replay_store", None) is not None
            and random.Random(int(seed) ^ 0xA17FA60).random()
            < float(getattr(self, "search_selfplay_fraction", 0.0))
        )
        lineup_kind = self._lineup_kind(seed, search_training=search_training)
        random_ma_training = self._is_random_ma_game(
            seed,
            lineup_kind,
            search_training=search_training,
        )
        lineup = self._take_learning_seats(4) if search_training else self._lineup(seed)
        teacher_replay_store = getattr(self, "teacher_replay_store", None)
        if teacher_replay_store is not None and not search_training:
            # Donate frozen-seat decisions only. The learner is excluded inside
            # the agent so its on-policy data is never double-counted.
            for seat in lineup:
                if not seat.train_from_self_play:
                    seat.teacher_replay_store = teacher_replay_store
        if search_training:
            self.search_game_count += 1
            for seat_index, seat in enumerate(lineup):
                config = SearchConfig.from_env()
                config.enabled = True
                config.selection = str(
                    os.getenv("ALPHAGO_SEARCH_SELFPLAY_SELECTION", "temperature")
                ).strip().lower()
                try:
                    config.temperature = float(os.getenv("ALPHAGO_SEARCH_SELFPLAY_TEMPERATURE", "1.0"))
                    config.temperature_until_generation = int(
                        os.getenv("ALPHAGO_SEARCH_SELFPLAY_TEMPERATURE_GENERATIONS", "4")
                    )
                except (TypeError, ValueError):
                    config.temperature = 1.0
                    config.temperature_until_generation = 4
                try:
                    config.root_noise_alpha = float(
                        os.getenv("ALPHAGO_SEARCH_SELFPLAY_ROOT_NOISE_ALPHA", "0.05")
                    )
                    config.root_noise_weight = float(
                        os.getenv("ALPHAGO_SEARCH_SELFPLAY_ROOT_NOISE_WEIGHT", "0.25")
                    )
                except (TypeError, ValueError):
                    config.root_noise_alpha = 0.05
                    config.root_noise_weight = 0.25
                config.seed = int(seed) * 4 + seat_index
                config.normalize()
                try:
                    replay_max_invalid_rate = float(
                        os.getenv("ALPHAGO_SEARCH_REPLAY_MAX_INVALID_RATE", "0.10")
                    )
                except (TypeError, ValueError):
                    replay_max_invalid_rate = 0.10
                seat.search_policy = SearchPolicy(
                    seat,
                    config,
                    replay_store=self.search_replay_store,
                    replay_max_invalid_rate=replay_max_invalid_rate,
                )
                # Search-generated games belong exclusively to the distillation
                # stream; even fallback policy actions must not enter strict PPO.
                seat.train_from_self_play = False
        return _SelfPlayGame(
            number=self.game_count,
            seed=seed,
            stage=stage,
            lineup=lineup,
            search_training=search_training,
            lineup_kind=lineup_kind,
            random_ma_training=random_ma_training,
            random_ma_mode=(
                str(getattr(self, "random_ma_mode", "Limited synergy"))
                if random_ma_training
                else None
            ),
        )

    async def _run_selfplay_game(self, game: _SelfPlayGame) -> Tuple[_SelfPlayGame, float]:
        """Run one pre-reserved game; PPO updates happen only after its batch completes."""
        started_at = time.monotonic()
        try:
            game_option_overrides = None
            if game.random_ma_training:
                game_option_overrides = {
                    "randomMA": str(game.random_ma_mode or "Limited synergy"),
                    "includeFanMA": False,
                    "modularMA": False,
                }
            await self.manager._run_single_game(
                game.lineup,
                tournament_id=f"{getattr(self, 'version', 'v2')}_selfplay_stage{game.stage}_{game.seed}",
                game_seed=game.seed,
                players_beginner=(game.stage == 0),
                game_option_overrides=game_option_overrides,
            )
            return game, time.monotonic() - started_at
        finally:
            if game.search_training:
                for seat in game.lineup:
                    seat.search_policy = None
                    seat.train_from_self_play = True

    async def _run_selfplay_batch(self) -> List[Tuple[_SelfPlayGame, float]]:
        """Run one batch on the current weights. PPO runs only after every game returns."""
        self._apply_v3_feature_scale()
        games = [self._reserve_selfplay_game() for _ in range(self.selfplay_concurrency)]
        for game in games:
            print(
                f"[selfplay] starting game={game.number} seed={game.seed} "
                f"stage={game.stage} lineup={game.lineup_kind} "
                f"search_training={game.search_training} "
                f"random_ma={game.random_ma_training} "
                f"decisions={self._total_decisions()}",
                flush=True,
            )
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    self._run_selfplay_game(game),
                    name=f"v2-selfplay:{game.number}:{game.seed}",
                )
                for game in games
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return list(results)

    async def _evaluate_and_promote(self, decisions: int) -> Dict:
        candidate_path = self.checkpoints / f"candidate_{decisions:09d}.pth"
        self.learner.save_model(str(candidate_path))
        shutil.copy2(candidate_path, self.latest_learner_path)
        screen_teacher_report = await self._benchmark(
            str(candidate_path),
            "teacher",
            self.stage,
            str(self.benchmarks),
            seeds_path=str(self.screen_seed_path),
            report_label="screen",
        )
        screen_regression_report = await self._benchmark(
            str(candidate_path),
            "champion",
            self.stage,
            str(self.benchmarks),
            seeds_path=str(self.screen_seed_path),
            champion=str(self.champion_path),
            report_label="screen",
        )
        teacher_screen_passed = self._screen_report_promising(screen_teacher_report, "teacher")
        regression_screen_passed = self._screen_report_promising(screen_regression_report, "champion")
        screen_passed = teacher_screen_passed and regression_screen_passed
        rollback = await self._maybe_rollback_from_screen(
            screen_regression_report,
            decisions,
            candidate_path=candidate_path,
        )
        teacher_report: Optional[Dict] = None
        teacher_relative: Optional[Dict[str, Any]] = None
        regression_report: Optional[Dict] = None
        if screen_passed:
            await self._ensure_champion_teacher_baseline()
            teacher_report = await self._benchmark(
                str(candidate_path), "teacher", self.stage, str(self.benchmarks)
            )
            teacher_relative = self._teacher_relative_to_champion(teacher_report)
            print(
                f"[selfplay] teacher gate absolute={bool(teacher_report.get('gate_passed', False))} "
                f"relative={teacher_relative['passed']} "
                f"candidate_pairwise={teacher_relative['candidate_pairwise']} "
                f"champion_pairwise={teacher_relative['champion_pairwise']}",
                flush=True,
            )
            if bool(teacher_report.get("gate_passed", False)) and teacher_relative["passed"]:
                regression_report = await self._benchmark(
                    str(candidate_path),
                    "champion",
                    self.stage,
                    str(self.benchmarks),
                    champion=str(self.champion_path),
                )
        else:
            print(
                f"[selfplay] full promotion benchmark skipped teacher_screen={teacher_screen_passed} "
                f"champion_screen={regression_screen_passed}",
                flush=True,
            )
        promoted = False
        if (
            teacher_report is not None
            and regression_report is not None
            and bool(teacher_report.get("gate_passed", False))
            and teacher_relative is not None
            and teacher_relative["passed"]
            and bool(regression_report.get("gate_passed", False))
        ):
            historical = self.checkpoints / f"champion_{decisions:09d}.pth"
            shutil.copy2(self.champion_path, historical)
            self._remember_history(historical)
            shutil.copy2(candidate_path, self.champion_path)
            self.champion_teacher_report = teacher_report
            self.champion_promotions = int(getattr(self, "champion_promotions", 0)) + 1
            self.consecutive_champion_screen_failures = 0
            if self.stage == 0:
                from v2_runtime import stage1_unlocked

                if not stage1_unlocked():
                    print(
                        "[selfplay] promotion passed, but Stage 1 remains blocked until "
                        "V2_ALLOW_STAGE1=1 after a strict action-space audit",
                        flush=True,
                    )
                else:
                    self.stage = 1
                    self.cluster.base_game_options = _load_stage_options(1)
            self._refresh_frozen_pools()
            promoted = True
        return {
            "decisions": decisions,
            "stage": self.stage,
            "candidate": str(candidate_path),
            "promoted": promoted,
            "screen_passed": screen_passed,
            "screen_teacher": screen_teacher_report,
            "screen_regression": screen_regression_report,
            "teacher": teacher_report,
            "teacher_relative": teacher_relative,
            "regression": regression_report,
            "rollback": rollback,
        }

    async def run(self, max_decisions: int, max_games: Optional[int] = None) -> None:
        max_decisions = int(max_decisions)
        game_limit = None if max_games is None else max(1, int(max_games))
        reports: List[Dict] = list(self.previous_reports)
        print(
            f"[selfplay] target_decisions={max_decisions} current={self._total_decisions()} "
            f"stage={self.stage}",
            flush=True,
        )
        try:
            while self._total_decisions() < max_decisions and (
                game_limit is None or int(self.game_count) < game_limit
            ):
                self._write_progress("batch_running")
                results = await self._run_selfplay_batch()
                self.lifetime_games += len(results)
                self.learner.games_played = int(self.lifetime_games)
                for game, _elapsed in results:
                    kind = str(getattr(game, "lineup_kind", "live"))
                    self.lineup_game_counts[kind] = int(self.lineup_game_counts.get(kind, 0)) + 1
                    self.lineup_games_since_update[kind] = int(
                        self.lineup_games_since_update.get(kind, 0)
                    ) + 1
                    if bool(getattr(game, "random_ma_training", False)):
                        self.random_ma_game_count = int(
                            getattr(self, "random_ma_game_count", 0)
                        ) + 1
                        self.random_ma_games_since_update = int(
                            getattr(self, "random_ma_games_since_update", 0)
                        ) + 1
                # A completed batch has released every cluster slot. Rebooting
                # the tmpfs-backed dedicated servers here purges all retained
                # remote games while PPO work proceeds on the GPU.
                recycle_task = None
                if self.cluster.rl_recycle_enabled:
                    recycle_task = asyncio.create_task(
                        self.cluster.recycle_idle_servers(),
                        name="v2-idle-server-recycle",
                    )
                decisions = self._total_decisions()
                rollout_size = self.learner.get_rollout_buffer_size()
                last_game, last_game_elapsed = results[-1]
                for game, game_elapsed in results:
                    print(
                        f"[selfplay] completed game={game.number} seed={game.seed} "
                        f"decisions={decisions} rollout={rollout_size} elapsed={game_elapsed:.1f}s",
                        flush=True,
                    )
                    self._write_progress("game_completed", game_seed=game.seed, game_elapsed=game_elapsed)
                # Rollout shards are already durable. Persist their decision/seed
                # accounting even when the process exits before the next PPO update.
                self._save_resume_state(reports, save_model=False)
                if rollout_size >= int(self.learner.ppo_rollout_steps):
                    optimize_started_at = time.monotonic()
                    print(
                        f"[selfplay] optimizing rollout steps={rollout_size} decisions={decisions}",
                        flush=True,
                    )
                    ppo_metrics = await self.learner.optimize_from_rollout_buffer(
                        self.learner.ppo_rollout_steps
                    )
                    self._maybe_archive_history_snapshot(decisions)
                    optimize_elapsed = time.monotonic() - optimize_started_at
                    self._append_ppo_metrics(ppo_metrics, decisions, optimize_elapsed)
                    print(
                        f"[selfplay] optimization complete elapsed={optimize_elapsed:.1f}s "
                        f"policy_version={self.learner.policy_version}",
                        flush=True,
                    )
                    self._write_progress(
                        "optimized",
                        game_seed=last_game.seed,
                        game_elapsed=last_game_elapsed,
                    )
                    self._save_resume_state(reports)
                if decisions >= self.next_benchmark_decision:
                    print(
                        f"[selfplay] benchmark starting decisions={decisions} stage={self.stage}",
                        flush=True,
                    )
                    report = await self._evaluate_and_promote(decisions)
                    reports.append(report)
                    print(self._rolling_screen_summary(reports), flush=True)
                    self.next_benchmark_decision += self.benchmark_interval
                    self._save_resume_state(reports)
                    self._write_progress(
                        "benchmark_completed",
                        game_seed=last_game.seed,
                        game_elapsed=last_game_elapsed,
                    )
                    print(
                        f"[selfplay] benchmark complete decisions={decisions} stage={self.stage} "
                        f"promoted={report['promoted']}",
                        flush=True,
                    )
                if recycle_task is not None:
                    await recycle_task
            self._write_progress("completed")
            print(f"[selfplay] completed target decisions={self._total_decisions()}", flush=True)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._write_progress("cancelled")
            raise
        except Exception as exc:
            self._write_progress("failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[selfplay] failed: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            self._save_resume_state(reports)
            # Covers a graceful stop between batches. The cluster refuses the
            # request if any cancellation path has not yet released its slot.
            if self.cluster.rl_recycle_enabled:
                try:
                    await self.cluster.recycle_idle_servers()
                except Exception as exc:
                    print(f"[selfplay] idle server recycle during shutdown failed: {exc}", flush=True)
            await self.cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four-seat TFM RL self-play")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--bc-checkpoint")
    initialization.add_argument(
        "--from-scratch",
        action="store_true",
        help="initialize a new policy with random weights and bypass only the BC pretrain gate",
    )
    parser.add_argument("--root", default=os.getenv("TFM_RL_V2_ROOT", "/app/v2"))
    parser.add_argument("--max-decisions", type=int, default=1_000_000)
    parser.add_argument("--benchmark-interval", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    runner = V2SelfPlayRunner(
        args.bc_checkpoint,
        args.root,
        args.benchmark_interval,
        args.seed,
        initial_stage=args.stage,
        from_scratch=args.from_scratch,
    )
    asyncio.run(runner.run(args.max_decisions))


if __name__ == "__main__":
    main()
