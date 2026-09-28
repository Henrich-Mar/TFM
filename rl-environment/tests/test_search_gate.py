"""Root eligibility gate and search configuration tests."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.config import MAX_BRANCHES_PER_BATCH, SearchConfig  # noqa: E402
from search.prompts import is_strategic_prompt, strategic_options  # noqa: E402
from search.search_agent import SearchPolicy  # noqa: E402
from models.agent import AgentConfig, RLAgent  # noqa: E402


def _or_state(count: int) -> dict:
    return {"waitingFor": {"type": "or", "options": [{"title": f"o{i}"} for i in range(count)]}}


def test_top_level_or_with_choices_is_strategic():
    assert is_strategic_prompt(_or_state(4))


def test_forced_single_option_is_not_strategic():
    assert not is_strategic_prompt(_or_state(1))
    assert not is_strategic_prompt(_or_state(0))


def test_continuation_prompts_are_not_strategic():
    for prompt_type in ("payment", "card", "space", "amount", "player", "projectCard", "initialCards"):
        assert not is_strategic_prompt({"waitingFor": {"type": prompt_type, "options": [{}, {}]}})
    assert not is_strategic_prompt({"waitingFor": None})
    assert not is_strategic_prompt({})
    assert strategic_options(None) == []


def test_gate_rejects_non_strategic_and_disabled_before_touching_server():
    agent = type(
        "AgentStub",
        (),
        {},
    )()
    cfg = SearchConfig(enabled=True, mode="lookahead")
    policy = SearchPolicy(agent, cfg)
    decision = asyncio.run(
        policy.decide(
            game_instance=type("G", (), {"base_url": "http://srv:8080"})(),
            player_id="p1",
            player_state={"waitingFor": {"type": "payment"}},
            planner_state={},
            action_descriptors=[{"action_index": 5, "decoded_action": {"type": "option"}}],
        )
    )
    assert decision is None
    assert policy.stats["fallback_ineligible"] == 1
    assert policy.stats["eligible_roots"] == 0

    disabled = SearchPolicy(agent, SearchConfig(enabled=False))
    decision = asyncio.run(
        disabled.decide(
            game_instance=type("G", (), {"base_url": "http://srv:8080"})(),
            player_id="p1",
            player_state=_or_state(3),
            planner_state={},
            action_descriptors=[{"action_index": 5, "decoded_action": {"type": "option"}}],
        )
    )
    assert decision is None
    assert disabled.stats["eligible_roots"] == 0


def test_config_reads_env_and_clamps_branch_budget(monkeypatch):
    env = {
        "ALPHAGO_SEARCH_ENABLED": "1",
        "ALPHAGO_SEARCH_MODE": "puct",
        "ALPHAGO_SEARCH_TOP_K": "16",
        "ALPHAGO_SEARCH_DETERMINIZATIONS": "16",
        "ALPHAGO_SEARCH_SIMULATIONS": "64",
        "ALPHAGO_SEARCH_DEPTH": "9",
        "ALPHAGO_SEARCH_PUCT_C": "2.0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = SearchConfig.from_env()
    assert cfg.enabled is True
    assert cfg.mode == "puct"
    assert cfg.simulations_per_move == 64
    assert cfg.max_root_turns_depth == 4
    assert cfg.puct_c == 2.0
    assert cfg.top_k * cfg.determinizations <= MAX_BRANCHES_PER_BATCH


def test_config_unknown_mode_falls_back_to_lookahead():
    cfg = SearchConfig(mode="mancala")
    cfg.normalize()
    assert cfg.mode == "lookahead"


def test_config_rejects_garbage_env_values(monkeypatch):
    monkeypatch.setenv("ALPHAGO_SEARCH_TOP_K", "not-a-number")
    monkeypatch.setenv("ALPHAGO_SEARCH_PUCT_C", "nan")
    cfg = SearchConfig.from_env()
    assert cfg.top_k == 8
    assert cfg.puct_c == 1.5


def _make_agent() -> RLAgent:
    with patch(
        "models.agent.require_backend_info",
        return_value={"module": "rust_tfm_rl", "api_version": "1.0", "crate_version": "test"},
    ):
        return RLAgent(AgentConfig())


def test_participant_ids_collect_all_players_from_observation():
    observation = {
        "id": "p1",
        "thisPlayer": {"id": "p1"},
        "players": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}, {"id": "p4"}],
    }
    ids = SearchPolicy._participant_ids(observation, "p1")
    assert set(ids) == {"p1", "p2", "p3", "p4"}
    assert ids.count("p1") == 1
