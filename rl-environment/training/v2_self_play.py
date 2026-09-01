"""Single-main-learner curriculum and frozen-opponent loop for TFM RL v2."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

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
        resumed_decisions = int(resume_state.get("decisions", 0) or 0)
        self.decision_offset = resumed_decisions
        self.next_benchmark_decision = ((resumed_decisions // self.benchmark_interval) + 1) * self.benchmark_interval
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

    async def _evaluate_and_promote(self, decisions: int) -> Dict:
        candidate_path = self.checkpoints / f"candidate_{decisions:09d}.pth"
        self.learner.save_model(str(candidate_path))
        shutil.copy2(candidate_path, self.latest_learner_path)
        random_report = await benchmark(str(candidate_path), "random", 0, str(self.benchmarks))
        teacher_report: Optional[Dict] = None
        regression_report: Optional[Dict] = await benchmark(
            str(candidate_path), "champion", self.stage, str(self.benchmarks), champion=str(self.champion_path)
        )
        promoted = False
        if (
            self.stage == 0
            and bool(random_report.get("gate_passed", False))
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
        reports: List[Dict] = list(self.previous_reports)
        try:
            while self._total_decisions() < int(max_decisions):
                seed = self.seed_cursor
                self.seed_cursor += 1
                while seed in self.reserved_benchmark_seeds:
                    seed = self.seed_cursor
                    self.seed_cursor += 1
                opponents = self._opponents(seed)
                candidate_seat = seed % 4
                lineup: List[RLAgent] = list(opponents)
                lineup.insert(candidate_seat, self.learner)
                await self.manager._run_single_game(
                    lineup,
                    tournament_id=f"v2_selfplay_stage{self.stage}_{seed}",
                    game_seed=seed,
                    players_beginner=(self.stage == 0),
                )
                if self.learner.get_rollout_buffer_size() >= int(self.learner.ppo_rollout_steps):
                    await self.learner.optimize_from_rollout_buffer(self.learner.ppo_rollout_steps)
                decisions = self._total_decisions()
                if decisions >= self.next_benchmark_decision:
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
