from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import (
    AWARD_FUND_CAP,
    TERMINAL_AWARD_CAP_PENALTY,
    calculate_v2_terminal_reward,
)


def test_v2_terminal_reward_uses_rank_and_bounded_relative_margin() -> None:
    assert calculate_v2_terminal_reward(1, 100, 80, True) > 1.0
    assert calculate_v2_terminal_reward(4, 50, 80, True) < -1.0
    assert calculate_v2_terminal_reward(2, 10_000, 0, True) == 0.35


def test_awards_up_to_the_cap_do_not_change_the_terminal_reward() -> None:
    """Funding the intended 1-2 awards must stay free at the terminal level."""
    baseline = calculate_v2_terminal_reward(1, 100, 80, True)
    for funded in range(0, max(1, AWARD_FUND_CAP) + 1):
        assert calculate_v2_terminal_reward(1, 100, 80, True, funded) == baseline


def test_awards_beyond_the_cap_are_charged_at_the_terminal_level() -> None:
    baseline = calculate_v2_terminal_reward(1, 100, 80, True)
    one_over = calculate_v2_terminal_reward(1, 100, 80, True, AWARD_FUND_CAP + 1)
    two_over = calculate_v2_terminal_reward(1, 100, 80, True, AWARD_FUND_CAP + 2)

    assert one_over == baseline - TERMINAL_AWARD_CAP_PENALTY
    assert two_over == baseline - (2 * TERMINAL_AWARD_CAP_PENALTY)
    # Must be large enough to matter against the 1st-to-2nd rank gap, which is
    # 0.75. A penalty below that cannot outvote simply finishing one place higher.
    assert TERMINAL_AWARD_CAP_PENALTY > 0.10


def test_terminal_cap_penalty_is_scored_even_when_the_game_is_won() -> None:
    """The penalty is not a consolation: winning still pays, over-funding still costs."""
    won_clean = calculate_v2_terminal_reward(1, 100, 80, True, AWARD_FUND_CAP)
    won_over = calculate_v2_terminal_reward(1, 100, 80, True, 3)
    assert won_clean > 0.0
    assert won_over < won_clean


def test_missing_award_count_is_treated_as_zero_not_as_a_crash() -> None:
    assert calculate_v2_terminal_reward(1, 100, 80, True, None) == calculate_v2_terminal_reward(1, 100, 80, True)
    assert calculate_v2_terminal_reward(1, 100, 80, True, "nonsense") == calculate_v2_terminal_reward(1, 100, 80, True)
