"""Single-main-learner curriculum and frozen-opponent loop for TFM RL v2."""
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
from models.decision_policy import HeuristicTeacherPolicy, RandomLegalPolicy
from tournament_manager import TournamentManager
from training.v2_benchmark import benchmark
from v2_runtime import initialize_v2_runtime


def _frozen_checkpoint_agent(path: str, agent_id: str) -> RLAgent:
    agent = RLAgent(agent_id=agent_id)
    agent.load_model(path)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.deterministic_actions = True
    return agent


def _teacher(agent_id: str, seed: int) -> RLAgent:
    agent = RLAgent(agent_id=agent_id, decision_policy=HeuristicTeacherPolicy(seed=seed, sample=False))
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    return agent


def _random(agent_id: str, seed: int) -> RLAgent:
    agent = RLAgent(agent_id=agent_id, decision_policy=RandomLegalPolicy(seed=seed))
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    return agent


def _load_stage_options(stage: int) -> Dict:
    path = Path(__file__).resolve().parents[1] / f"game_options.v2_stage{int(stage)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _SelfPlayGame:
    number: int
    seed: int
    stage: int
    lineup: List[RLAgent]


class V2SelfPlayRunner:
    def __init__(self, bc_checkpoint: str, root: str, benchmark_interval: int = 25_000, seed: int = 100_000) -> None:
        self.paths = initialize_v2_runtime()
        # Subsequent benchmark subprocess-equivalent calls are intentional v2 resumes.
        os.environ["V2_ALLOW_RESUME"] = "1"
        self.root = Path(root).expanduser().resolve()
        if self.root != Path(self.paths["root"]).resolve():
            raise RuntimeError("--root must exactly match TFM_RL_V2_ROOT")
        self.checkpoints = self.root / "checkpoints"
        self.benchmarks = self.root / "benchmarks"
        self.metrics = self.root / "metrics"
        for path in (self.checkpoints, self.benchmarks, self.metrics):
            path.mkdir(parents=True, exist_ok=True)
        report_path = Path(bc_checkpoint).with_name("pretrain_report.json")
        if not report_path.is_file() or not bool(json.loads(report_path.read_text(encoding="utf-8")).get("ppo_gate_passed", False)):
            raise RuntimeError("PPO is blocked until the BC pretrain_report.json has ppo_gate_passed=true")
        self.state_path = self.metrics / "selfplay_state.json"
        self.latest_learner_path = self.checkpoints / "latest_learner.pth"
        resume_state: Dict = {}
        if self.state_path.is_file() != self.latest_learner_path.is_file():
            raise RuntimeError(
                "incomplete v2 resume state: selfplay_state.json and latest_learner.pth must both exist"
            )
        if self.state_path.is_file() and self.latest_learner_path.is_file():
            resume_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        learner_source = str(self.latest_learner_path) if resume_state else bc_checkpoint
        self.learner = RLAgent(agent_id="v2-main-learner")
        self.learner.load_model(learner_source)
        self.learner.train_from_self_play = True
        self.learner.config.train_from_self_play = True
        self.learner.ppo_enable = True
        self.learner.deterministic_actions = False
        self.stage = int(resume_state.get("stage", 0) or 0)
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
        self.teacher_pool = [_teacher(f"teacher-{idx}", seed + idx) for idx in range(3)]
        self.random_pool = [_random(f"random-{idx}", seed + 100 + idx) for idx in range(3)]
        self.champion_pool: List[RLAgent] = []
        self.historical_pool: List[RLAgent] = []

        servers = [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
        self.cluster = GameServerCluster(servers)
        self.cluster.base_game_options = _load_stage_options(self.stage)
        self.manager = TournamentManager(self.cluster)
        if self.stage >= 1:
            self._refresh_frozen_pools()
        self._write_progress("started")
        print(
            f"[selfplay] started stage={self.stage} decisions={self._total_decisions()} "
            f"next_benchmark={self.next_benchmark_decision} concurrency={self.selfplay_concurrency}",
            flush=True,
        )

    def _refresh_frozen_pools(self) -> None:
        self.champion_pool = [
            _frozen_checkpoint_agent(str(self.champion_path), f"champion-{idx}")
            for idx in range(3)
        ]
        self.historical_pool = [
            _frozen_checkpoint_agent(
                self.history[-8:][idx % len(self.history[-8:])],
                f"historical-{idx}",
            )
            for idx in range(3)
        ] if self.history else []

    def _total_decisions(self) -> int:
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
            "schema_version": "tfm_rl_v2.selfplay_progress.v1",
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
        temporary = self.progress_path.with_name(self.progress_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.progress_path)

    def _opponents(self, game_seed: int) -> List[RLAgent]:
        opponents: List[RLAgent] = []
        rng = random.Random(int(game_seed) ^ 0x5F3759DF)
        for seat in range(3):
            draw = rng.random()
            if self.stage == 0:
                opponent = self.teacher_pool[seat] if draw < 0.5 else self.random_pool[seat]
                if isinstance(opponent.decision_policy, RandomLegalPolicy):
                    opponent.decision_policy.rng.seed(int(game_seed) + seat)
                opponents.append(opponent)
                continue
            if draw < 0.4:
                opponents.append(self.teacher_pool[seat])
            elif draw < 0.8 or not self.historical_pool:
                opponents.append(self.champion_pool[seat])
            else:
                opponents.append(self.historical_pool[seat % len(self.historical_pool)])
        return opponents

    def _reserve_selfplay_game(self) -> _SelfPlayGame:
        """Reserve one game using the current, unmodified learner policy."""
        self.game_count += 1
        seed = self.seed_cursor
        self.seed_cursor += 1
        while seed in self.reserved_benchmark_seeds:
            seed = self.seed_cursor
            self.seed_cursor += 1
        stage = self.stage
        lineup: List[RLAgent] = list(self._opponents(seed))
        lineup.insert(seed % 4, self.learner)
        return _SelfPlayGame(number=self.game_count, seed=seed, stage=stage, lineup=lineup)

    async def _run_selfplay_game(self, game: _SelfPlayGame) -> Tuple[_SelfPlayGame, float]:
        """Run one pre-reserved game; PPO updates happen only after its batch completes."""
        started_at = time.monotonic()
        await self.manager._run_single_game(
            game.lineup,
            tournament_id=f"v2_selfplay_stage{game.stage}_{game.seed}",
            game_seed=game.seed,
            players_beginner=(game.stage == 0),
        )
        return game, time.monotonic() - started_at

    async def _run_selfplay_batch(self) -> List[Tuple[_SelfPlayGame, float]]:
        """Run a bounded set of games against one frozen learner-policy version."""
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
        random_report: Optional[Dict] = None
        if self.stage == 0:
            random_report = await benchmark(str(candidate_path), "random", 0, str(self.benchmarks))
        teacher_report: Optional[Dict] = None
        regression_report: Optional[Dict] = await benchmark(
            str(candidate_path), "champion", self.stage, str(self.benchmarks), champion=str(self.champion_path)
        )
        promoted = False
        if (
            self.stage == 0
            and bool((random_report or {}).get("gate_passed", False))
            and bool(regression_report.get("gate_passed", False))
        ):
            self.stage = 1
            self.cluster.base_game_options = _load_stage_options(1)
            initial_champion = self.checkpoints / "champion_bc.pth"
            shutil.copy2(self.champion_path, initial_champion)
            self.history.append(str(initial_champion))
            shutil.copy2(candidate_path, self.champion_path)
            self._refresh_frozen_pools()
            promoted = True
        elif self.stage == 1:
            teacher_report = await benchmark(str(candidate_path), "teacher", 1, str(self.benchmarks))
            if bool(teacher_report.get("gate_passed", False)) and bool(regression_report.get("gate_passed", False)):
                historical = self.checkpoints / f"champion_{decisions:09d}.pth"
                shutil.copy2(self.champion_path, historical)
                self.history.append(str(historical))
                self.history = self.history[-8:]
                shutil.copy2(candidate_path, self.champion_path)
                self._refresh_frozen_pools()
                promoted = True
        return {
            "decisions": decisions,
            "stage": self.stage,
            "candidate": str(candidate_path),
            "promoted": promoted,
            "random": random_report,
            "teacher": teacher_report,
            "regression": regression_report,
        }

    async def run(self, max_decisions: int) -> None:
        max_decisions = int(max_decisions)
        reports: List[Dict] = list(self.previous_reports)
        print(
            f"[selfplay] target_decisions={max_decisions} current={self._total_decisions()} "
            f"stage={self.stage}",
            flush=True,
        )
        try:
            while self._total_decisions() < max_decisions:
                self._write_progress("batch_running")
                results = await self._run_selfplay_batch()
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
                if decisions >= self.next_benchmark_decision:
                    print(
                        f"[selfplay] benchmark starting decisions={decisions} stage={self.stage}",
                        flush=True,
                    )
                    report = await self._evaluate_and_promote(decisions)
                    reports.append(report)
                    self.next_benchmark_decision += self.benchmark_interval
                    state = {
                        "schema_version": "tfm_rl_v2.selfplay_state.v1",
                        "stage": self.stage,
                        "decisions": decisions,
                        "seed_cursor": self.seed_cursor,
                        "policy_version": self.learner.policy_version,
                        "champion": str(self.champion_path),
                        "history": self.history,
                        "reports": reports,
                    }
                    self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
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
            self._write_progress("completed")
            print(f"[selfplay] completed target decisions={self._total_decisions()}", flush=True)
        except Exception as exc:
            self._write_progress("failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[selfplay] failed: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            await self.cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-learner TFM RL v2 curriculum self-play")
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
