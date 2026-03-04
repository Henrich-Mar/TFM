import os
import sys
from pathlib import Path


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

from champion_orchestrator import (  # noqa: E402
    CandidateRef,
    _current_checkpoint_path,
    _current_manifest_path,
    _publish_round,
    decide_promotion,
    discover_candidates_for_coord,
    prune_generation_dirs,
    rank_candidate_metrics,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_candidate_discovery_uses_latest_then_fallback(tmp_path: Path) -> None:
    models = tmp_path / "coord-1"
    _touch(models / "generation_5" / "agent_0_fitness_101.00.pth")
    _touch(models / "generation_4" / "agent_0_fitness_99.50.pth")
    _touch(models / "generation_4" / "agent_1_fitness_88.10.pth")

    candidates = discover_candidates_for_coord("coord-1", str(models), top_k=2)
    assert len(candidates) == 2
    assert candidates[0].generation == 5
    assert candidates[1].generation == 4
    assert candidates[0].checkpoint_relpath.startswith("generation_5/")


def test_rank_order_is_deterministic() -> None:
    ranked = rank_candidate_metrics(
        {
            "a": {"games_completed": 10, "wins": 5, "vp_sum": 700, "rank_sum": 24},
            "b": {"games_completed": 10, "wins": 5, "vp_sum": 680, "rank_sum": 25},
            "c": {"games_completed": 10, "wins": 6, "vp_sum": 620, "rank_sum": 26},
        }
    )
    assert [item["candidate_id"] for item in ranked] == ["c", "a", "b"]


def test_promotion_bootstrap_and_margin_rules() -> None:
    ranked = [
        {"candidate_id": "leader", "win_rate": 0.60},
        {"candidate_id": "incumbent", "win_rate": 0.58},
    ]
    bootstrap = decide_promotion(
        ranked_candidates=ranked,
        incumbent_candidate_id=None,
        completed_games=24,
        completion_rate=0.95,
        min_games_for_promotion=24,
        min_completion_rate=0.90,
        win_rate_margin=0.03,
    )
    assert bootstrap["applied"] is True
    assert bootstrap["winner_candidate_id"] == "leader"

    insufficient = decide_promotion(
        ranked_candidates=ranked,
        incumbent_candidate_id="incumbent",
        completed_games=10,
        completion_rate=0.95,
        min_games_for_promotion=24,
        min_completion_rate=0.90,
        win_rate_margin=0.03,
    )
    assert insufficient["applied"] is False
    assert insufficient["reason"] == "insufficient_completed_games"

    margin_fail = decide_promotion(
        ranked_candidates=ranked,
        incumbent_candidate_id="incumbent",
        completed_games=24,
        completion_rate=0.95,
        min_games_for_promotion=24,
        min_completion_rate=0.90,
        win_rate_margin=0.05,
    )
    assert margin_fail["applied"] is False
    assert margin_fail["reason"] == "margin_not_met"

    margin_pass = decide_promotion(
        ranked_candidates=ranked,
        incumbent_candidate_id="incumbent",
        completed_games=24,
        completion_rate=0.95,
        min_games_for_promotion=24,
        min_completion_rate=0.90,
        win_rate_margin=0.01,
    )
    assert margin_pass["applied"] is True
    assert margin_pass["reason"] == "margin_passed"


def test_atomic_publish_writes_manifest_and_checkpoint(tmp_path: Path) -> None:
    checkpoint_src = tmp_path / "coord-1" / "generation_7" / "agent_0_fitness_123.45.pth"
    _touch(checkpoint_src)
    winner = CandidateRef(
        candidate_id="coord-1:g7:agent_0_fitness_123.45.pth",
        coordinator_id="coord-1",
        source_type="coord",
        coord_root=str(tmp_path / "coord-1"),
        generation=7,
        checkpoint_path=str(checkpoint_src),
        checkpoint_relpath="generation_7/agent_0_fitness_123.45.pth",
    )

    manifest = _publish_round(
        output_root=str(tmp_path / "global"),
        round_id="r1",
        active_winner=winner,
        active_winner_metrics={"games_total": 12, "games_completed": 12, "wins": 8, "win_rate": 2 / 3, "avg_rank": 1.7, "avg_vp": 83.0},
        ranked_payloads=[],
        promotion={"applied": True, "reason": "bootstrap_no_incumbent"},
        incumbent_before=None,
    )

    current_checkpoint = Path(_current_checkpoint_path(str(tmp_path / "global")))
    current_manifest = Path(_current_manifest_path(str(tmp_path / "global")))
    assert current_checkpoint.is_file()
    assert current_manifest.is_file()
    assert manifest["winner"]["generation"] == 7
    assert manifest["promotion"]["applied"] is True


def test_prune_keeps_latest_and_pins(tmp_path: Path) -> None:
    root = tmp_path / "coord-2"
    for generation in range(6):
        gen_dir = root / f"generation_{generation}"
        _touch(gen_dir / f"agent_0_fitness_{100 - generation:.2f}.pth")

    removed = prune_generation_dirs(str(root), keep_last=2, pinned_generations={1})
    assert removed == [0, 2, 3]
    assert (root / "generation_1").is_dir()
    assert (root / "generation_4").is_dir()
    assert (root / "generation_5").is_dir()
