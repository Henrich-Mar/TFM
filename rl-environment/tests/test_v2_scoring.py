from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import calculate_v2_terminal_reward


def test_v2_terminal_reward_uses_rank_and_bounded_relative_margin() -> None:
    assert calculate_v2_terminal_reward(1, 100, 80, True) > 1.0
    assert calculate_v2_terminal_reward(4, 50, 80, True) < -1.0
    assert calculate_v2_terminal_reward(2, 10_000, 0, True) == 0.35
