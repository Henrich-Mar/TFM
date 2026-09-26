"""Four-seat PPO self-play for TFM RL v2.

Every chair plays the live policy and writes into one rollout buffer. After the
first promotion, one chair is a frozen past checkpoint on 25% of games.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from game_interface import GameServerCluster
from models.agent import RLAgent
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


class V2SelfPlayRunner:
    def __init__(
        self,
        bc_checkpoint: str,
        root: str,
        benchmark_interval: int = 25_000,
        seed: int = 100_000,
        initial_stage: Optional[int] = None,
    ) -> None:
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
        self.learner.load_model(learner_source)
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
        benchmark_seed_payload = json.loads(
            (Path(__file__).resolve().parents[1] / "benchmark_seeds.v1.json").read_text(encoding="utf-8")
        )
        self.reserved_benchmark_seeds = {int(item) for item in benchmark_seed_payload.get("seeds", [])}
        self.benchmark_interval = max(1, int(benchmark_interval))
        try:
            self.selfplay_concurrency = max(1, int(os.getenv("SELFPLAY_CONCURRENCY", "1")))
        except (TypeError, ValueError):
            self.selfplay_concurrency = 1
        resumed_decisions = int(resume_state.get("decisions", 0) or 0)
        self.decision_offset = resumed_decisions
        self.next_benchmark_decision = ((resumed_decisions // self.benchmark_interval) + 1) * self.benchmark_interval
        self.progress_path = self.metrics / "selfplay_progress.json"
        self.game_count = 0
        self.champion_path = self.checkpoints / "champion.pth"
        if not self.champion_path.is_file():
            shutil.copy2(bc_checkpoint, self.champion_path)
        self.history: List[str] = [str(item) for item in (resume_state.get("history", []) or []) if Path(str(item)).is_file()]
        self.previous_reports: List[Dict] = list(resume_state.get("reports", []) or [])
        self.historical_pool: List[RLAgent] = []
        self._historical_cursor = 0
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
            "[selfplay] lineup=live_policy:4 historical_seat=25%_of_games_after_promotion",
            flush=True,
        )

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

    def _take_historical_seat(self) -> RLAgent:
        seat = self.historical_pool[self._historical_cursor % len(self.historical_pool)]
        self._historical_cursor += 1
        self._reset_seat_memory(seat)
        return seat

    def _refresh_frozen_pools(self) -> None:
        """Load every retained checkpoint, with extra copies when games outnumber them.

        A batch can seat one frozen opponent per concurrent game. Those chairs
        must be distinct objects, and every archived champion in the last eight
        has to be able to sit, not only the oldest few.
        """
        recent = self.history[-8:]
        if not recent:
            self.historical_pool = []
        else:
            copies = max(len(recent), int(self.selfplay_concurrency))
            self.historical_pool = [
                _frozen_checkpoint_agent(
                    recent[idx % len(recent)],
                    f"historical-{idx}",
                )
                for idx in range(copies)
            ]
        self._historical_cursor = 0
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
        for agent in [*getattr(self, "learning_seats", []), *self.historical_pool]:
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
            "games": int(self.game_count),
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

    def _save_resume_state(self, reports: List[Dict]) -> None:
        """Atomically refresh the durable learner and self-play state."""
        learner_temporary = self.latest_learner_path.with_name(self.latest_learner_path.name + ".tmp")
        self.learner.save_model(str(learner_temporary))
        os.replace(learner_temporary, self.latest_learner_path)
        state = {
            "schema_version": f"tfm_rl_{self.version}.selfplay_state.v1",
            "stage": self.stage,
            "decisions": self._total_decisions(),
            "seed_cursor": self.seed_cursor,
            "policy_version": self.learner.policy_version,
            "champion": str(self.champion_path),
            "history": self.history,
            "reports": reports,
        }
        state_temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        state_temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(state_temporary, self.state_path)

    def _lineup(self, game_seed: int) -> List[RLAgent]:
        """Four live seats. One frozen past checkpoint on 25% of games once history exists."""
        lineup = self._take_learning_seats(4)
        if not self.history or not self.historical_pool:
            return lineup
        rng = random.Random(int(game_seed) ^ 0x5F3759DF)
        if rng.random() >= 0.25:
            return lineup
        lineup[int(rng.randrange(4))] = self._take_historical_seat()
        return lineup

    def _reserve_selfplay_game(self) -> _SelfPlayGame:
        """Reserve one game using the current, unmodified learner policy."""
        self.game_count += 1
        seed = self.seed_cursor
        self.seed_cursor += 1
        while seed in self.reserved_benchmark_seeds:
            seed = self.seed_cursor
            self.seed_cursor += 1
        stage = self.stage
        lineup = self._lineup(seed)
        return _SelfPlayGame(number=self.game_count, seed=seed, stage=stage, lineup=lineup)

    async def _run_selfplay_game(self, game: _SelfPlayGame) -> Tuple[_SelfPlayGame, float]:
        """Run one pre-reserved game; PPO updates happen only after its batch completes."""
        started_at = time.monotonic()
        await self.manager._run_single_game(
            game.lineup,
            tournament_id=f"{getattr(self, 'version', 'v2')}_selfplay_stage{game.stage}_{game.seed}",
            game_seed=game.seed,
            players_beginner=(game.stage == 0),
        )
        return game, time.monotonic() - started_at

    async def _run_selfplay_batch(self) -> List[Tuple[_SelfPlayGame, float]]:
        """Run one batch on the current weights. PPO runs only after every game returns."""
        self._apply_v3_feature_scale()
        games = [self._reserve_selfplay_game() for _ in range(self.selfplay_concurrency)]
        for game in games:
            print(
                f"[selfplay] starting game={game.number} seed={game.seed} "
                f"stage={game.stage} decisions={self._total_decisions()}",
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
        teacher_report = await benchmark(
            str(candidate_path), "teacher", self.stage, str(self.benchmarks)
        )
        regression_report = await benchmark(
            str(candidate_path), "champion", self.stage, str(self.benchmarks), champion=str(self.champion_path)
        )
        promoted = False
        if bool(teacher_report.get("gate_passed", False)) and bool(regression_report.get("gate_passed", False)):
            historical = self.checkpoints / f"champion_{decisions:09d}.pth"
            shutil.copy2(self.champion_path, historical)
            self.history.append(str(historical))
            self.history = self.history[-8:]
            shutil.copy2(candidate_path, self.champion_path)
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
            "teacher": teacher_report,
            "regression": regression_report,
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
                self.learner.games_played = max(int(self.learner.games_played), int(self.game_count))
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
                if rollout_size >= int(self.learner.ppo_rollout_steps):
                    optimize_started_at = time.monotonic()
                    print(
                        f"[selfplay] optimizing rollout steps={rollout_size} decisions={decisions}",
                        flush=True,
                    )
                    await self.learner.optimize_from_rollout_buffer(self.learner.ppo_rollout_steps)
                    print(
                        f"[selfplay] optimization complete elapsed={time.monotonic() - optimize_started_at:.1f}s "
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
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--root", default=os.getenv("TFM_RL_V2_ROOT", "/app/v2"))
    parser.add_argument("--max-decisions", type=int, default=1_000_000)
    parser.add_argument("--benchmark-interval", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    runner = V2SelfPlayRunner(args.bc_checkpoint, args.root, args.benchmark_interval, args.seed)
    asyncio.run(runner.run(args.max_decisions))


if __name__ == "__main__":
    main()
