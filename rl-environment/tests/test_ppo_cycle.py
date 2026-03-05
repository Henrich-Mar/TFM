import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.ppo_cycle import optimize_population_with_ppo


class _FakePPOAgent:
    def __init__(self, available_steps: int) -> None:
        self.available_steps = int(available_steps)
        self.budgets: list[int] = []

    def get_rollout_buffer_size(self) -> int:
        return int(self.available_steps)

    async def optimize_from_rollout_buffer(self, max_steps: int):
        budget = int(max_steps)
        self.budgets.append(budget)
        return {
            "rollout/steps": budget,
            "rollout/schema_filtered": 0,
        }


def test_optimize_population_respects_total_rollout_budget(monkeypatch) -> None:
    monkeypatch.setenv("PPO_MIN_STEPS_PER_AGENT", "0")
    monkeypatch.setenv("PPO_PARALLEL_AGENTS", "1")

    agents = [_FakePPOAgent(100), _FakePPOAgent(100), _FakePPOAgent(100)]

    metrics = asyncio.run(
        optimize_population_with_ppo(
            population=agents,
            target_rollout_steps=10,
        )
    )

    assert metrics["rollout/steps_collected"] == 10
    assert [agent.budgets for agent in agents] == [[3], [3], [4]]
