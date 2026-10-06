import json
import os
import sys


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

from checkpoint_catalog import (  # noqa: E402
    TIER_BENCHMARK,
    TIER_DECISIONS,
    TIER_GATE_PASS,
    TIER_SIDECAR_METRICS,
    TIER_TOURNAMENT_WINNER,
    discover_checkpoints,
    format_candidate_row,
    rank_candidates,
    select_best_checkpoint,
)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _benchmark_report(checkpoint, *, first_place_rate, gate_passed, completed=120, planned=120, baseline="teacher"):
    return {
        "schema_version": "tfm_rl_v2.benchmark.v1",
        "checkpoint": checkpoint,
        "baseline": baseline,
        "stage": 1,
        "planned_games": planned,
        "completed_games": completed,
        "completion_rate": completed / planned,
        "rejection_count": 0,
        "first_place_rate": first_place_rate,
        "mean_rank": 1.9,
        "pairwise_score": 0.6,
        "gate_passed": gate_passed,
    }


def test_benchmarked_candidate_beats_a_higher_decision_unvalidated_learner(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-v2" / "checkpoints"
    benchmarks = root / "rl-v2" / "benchmarks"
    checkpoints.mkdir(parents=True)
    benchmarks.mkdir(parents=True)

    strong = checkpoints / "candidate_000100000.pth"
    learner = checkpoints / "latest_learner.pth"
    strong.write_bytes(b"strong")
    learner.write_bytes(b"learner")

    _write_json(
        benchmarks / "benchmark_candidate_000100000_stage1_teacher.json",
        _benchmark_report(str(strong.resolve()), first_place_rate=0.42, gate_passed=True),
    )

    ranked = discover_checkpoints([str(root / "rl-v2")], root=str(root))
    assert [item.name for item in ranked] == [strong.name, learner.name]
    assert ranked[0].verified is True
    assert ranked[0].tier == TIER_GATE_PASS
    assert ranked[0].metrics["first_place_rate"] == 0.42
    assert ranked[1].verified is False
    assert ranked[1].tier == TIER_DECISIONS


def test_teacher_result_outranks_a_higher_win_rate_against_a_weaker_baseline(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-alphago" / "checkpoints"
    checkpoints.mkdir(parents=True)
    versus_teacher = checkpoints / "candidate_003850348.pth"
    versus_champion = checkpoints / "candidate_000800867.pth"
    versus_teacher.write_bytes(b"a")
    versus_champion.write_bytes(b"b")

    benchmarks = root / "rl-alphago" / "benchmarks"
    _write_json(
        benchmarks / "benchmark_candidate_003850348_stage1_teacher_screen.json",
        _benchmark_report(str(versus_teacher.resolve()), first_place_rate=0.59, gate_passed=True, completed=32, planned=32),
    )
    _write_json(
        benchmarks / "benchmark_candidate_000800867_stage1_champion.json",
        _benchmark_report(str(versus_champion.resolve()), first_place_rate=1.0, gate_passed=True, baseline="champion"),
    )

    ranked = discover_checkpoints([str(root / "rl-alphago")], root=str(root))
    assert [item.name for item in ranked] == [versus_teacher.name, versus_champion.name]
    assert ranked[0].baseline == "teacher"
    assert ranked[0].baseline_rank > ranked[1].baseline_rank


def test_gate_failure_demotes_a_checkpoint_below_gate_passing_ones(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-alphago" / "checkpoints"
    checkpoints.mkdir(parents=True)
    failing = checkpoints / "champion.pth"
    passing = checkpoints / "candidate_003200389.pth"
    failing.write_bytes(b"a")
    passing.write_bytes(b"b")

    benchmarks = root / "rl-alphago" / "benchmarks"
    _write_json(
        benchmarks / "benchmark_champion_stage1_teacher.json",
        _benchmark_report(str(failing.resolve()), first_place_rate=0.75, gate_passed=False, baseline="award_teacher", completed=16, planned=16),
    )
    _write_json(
        benchmarks / "benchmark_candidate_003200389_stage1_teacher.json",
        _benchmark_report(str(passing.resolve()), first_place_rate=0.52, gate_passed=True),
    )

    ranked = discover_checkpoints([str(root / "rl-alphago")], root=str(root))
    assert [item.name for item in ranked] == [passing.name, failing.name]
    assert ranked[0].tier == TIER_GATE_PASS
    assert ranked[1].tier == TIER_BENCHMARK


def test_champion_is_measured_on_its_hardest_available_baseline(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-alphago" / "checkpoints"
    checkpoints.mkdir(parents=True)
    champion = checkpoints / "champion.pth"
    champion.write_bytes(b"champion")

    benchmarks = root / "rl-alphago" / "benchmarks"
    _write_json(
        benchmarks / "benchmark_champion_stage1_award_teacher_awardprobe.json",
        _benchmark_report(str(champion.resolve()), first_place_rate=0.75, gate_passed=False, baseline="award_teacher", completed=16, planned=16),
    )
    _write_json(
        benchmarks / "benchmark_champion_stage1_teacher.json",
        _benchmark_report(str(champion.resolve()), first_place_rate=0.15, gate_passed=False),
    )

    ranked = discover_checkpoints([str(root / "rl-alphago")], root=str(root))
    assert len(ranked) == 1
    assert ranked[0].baseline == "teacher"
    assert ranked[0].metrics["first_place_rate"] == 0.15
    assert "1st place 15% vs teacher" in ranked[0].summary()


def test_global_champion_manifest_winner_ranks_first(tmp_path):
    root = tmp_path / "repo"
    global_root = root / "rl-models-global"
    current = global_root / "champion" / "current"
    current.mkdir(parents=True)

    winner = current / "champion.pth"
    rival = root / "rl-v2" / "checkpoints" / "candidate_000999999.pth"
    winner.write_bytes(b"winner")
    rival.parent.mkdir(parents=True)
    rival.write_bytes(b"rival")

    _write_json(
        current / "champion_manifest.json",
        {
            "version": 1,
            "winner": {
                "checkpoint_path": str(winner.resolve()),
                "metrics": {"games_completed": 36, "win_rate": 0.61, "avg_rank": 1.4, "avg_vp": 62.0},
            },
            "candidates": [],
        },
    )
    _write_json(
        root / "rl-v2" / "benchmarks" / "benchmark_candidate_000999999_stage1_teacher.json",
        _benchmark_report(str(rival.resolve()), first_place_rate=0.80, gate_passed=True),
    )

    ranked = discover_checkpoints([str(global_root), str(root / "rl-v2")], root=str(root))
    assert ranked[0].name == "champion.pth"
    assert ranked[0].tier == TIER_TOURNAMENT_WINNER
    assert ranked[0].metrics["tournament_win_rate"] == 0.61
    assert ranked[1].name == rival.name


def test_legacy_generation_sidecar_elo_beats_a_sidecar_without_metrics(tmp_path):
    root = tmp_path / "repo"
    generations = root / "rl-models"
    (generations / "generation_7").mkdir(parents=True)
    (generations / "generation_6").mkdir(parents=True)

    scored = generations / "generation_7" / "agent_0_fitness_310.50.pth"
    unscored = generations / "generation_6" / "agent_0_fitness_290.00.pth"
    scored.write_bytes(b"scored")
    unscored.write_bytes(b"unscored")

    _write_json(generations / "generation_7" / "agent_0_config.json", {"elo": 1410.0, "eval_fitness": 310.5, "saved_generation": 7})
    _write_json(generations / "generation_6" / "agent_0_config.json", {"saved_generation": 6})

    ranked = discover_checkpoints([str(generations)], root=str(root))
    assert ranked[0].name == scored.name
    assert ranked[0].tier == TIER_SIDECAR_METRICS
    assert ranked[0].metrics["elo"] == 1410.0
    assert "elo 1410" in ranked[0].summary()
    assert ranked[1].name == unscored.name


def test_incomplete_benchmark_is_not_treated_as_strength_evidence(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-v2" / "checkpoints"
    checkpoints.mkdir(parents=True)
    aborted = checkpoints / "candidate_000000123.pth"
    other = checkpoints / "candidate_000000999.pth"
    aborted.write_bytes(b"aborted")
    other.write_bytes(b"other")

    _write_json(
        root / "rl-v2" / "benchmarks" / "benchmark_candidate_000000123_stage1_teacher.json",
        _benchmark_report(str(aborted.resolve()), first_place_rate=0.0, gate_passed=False, completed=4, planned=120),
    )

    ranked = discover_checkpoints([str(root / "rl-v2")], root=str(root))
    by_name = {item.name: item for item in ranked}
    aborted_entry = by_name[aborted.name]
    assert aborted_entry.verified is False
    assert aborted_entry.tier == TIER_DECISIONS
    assert any("aborted benchmark" in note for note in aborted_entry.evidence)
    # An aborted run must not outrank a checkpoint measured on a real baseline.
    measured = root / "rl-v2" / "benchmarks" / "benchmark_candidate_000000999_stage1_teacher.json"
    _write_json(measured, _benchmark_report(str(other.resolve()), first_place_rate=0.10, gate_passed=True))
    ranked = discover_checkpoints([str(root / "rl-v2")], root=str(root))
    assert ranked[0].name == other.name
    assert ranked[0].verified is True


def test_benchmark_stem_fallback_matches_when_report_checkpoint_path_is_stale(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-v3" / "checkpoints"
    checkpoints.mkdir(parents=True)
    moved = checkpoints / "candidate_000400000.pth"
    moved.write_bytes(b"moved")

    report = _benchmark_report("/old/container/path/candidate_000400000.pth", first_place_rate=0.33, gate_passed=True, baseline="award_teacher")
    _write_json(root / "rl-v3" / "benchmarks" / "benchmark_candidate_000400000_stage1_award_teacher.json", report)

    ranked = discover_checkpoints([str(root / "rl-v3")], root=str(root))
    assert ranked[0].verified is True
    assert ranked[0].metrics["baseline"] == "award_teacher"


def test_strongest_baseline_report_wins_when_checkpoint_has_several_reports(tmp_path):
    root = tmp_path / "repo"
    checkpoints = root / "rl-v2" / "checkpoints"
    checkpoints.mkdir(parents=True)
    candidate = checkpoints / "champion.pth"
    candidate.write_bytes(b"champion")
    benchmarks = root / "rl-v2" / "benchmarks"
    _write_json(
        benchmarks / "benchmark_champion_stage0_random.json",
        _benchmark_report(str(candidate.resolve()), first_place_rate=0.60, gate_passed=True, baseline="random"),
    )
    _write_json(
        benchmarks / "benchmark_champion_stage1_teacher_promotion.json",
        _benchmark_report(str(candidate.resolve()), first_place_rate=0.28, gate_passed=False),
    )

    ranked = discover_checkpoints([str(root / "rl-v2")], root=str(root))
    assert len(ranked) == 1
    assert ranked[0].metrics["first_place_rate"] == 0.28
    assert ranked[0].metrics["gate_passed"] is False
    assert ranked[0].baseline == "teacher"
    assert ranked[0].role == "champion"


def test_discovery_is_empty_without_checkpoints(tmp_path):
    root = tmp_path / "repo"
    (root / "rl-v2").mkdir(parents=True)
    assert discover_checkpoints([str(root / "rl-v2")], root=str(root)) == []
    assert select_best_checkpoint([str(root / "rl-v2")], root=str(root)) is None


def test_rank_candidates_is_stable_and_deterministic(tmp_path):
    root = str(tmp_path)
    paths = []
    for index in range(3):
        path = tmp_path / f"candidate_{index}.pth"
        path.write_bytes(b"x")
        paths.append(str(path.resolve()))

    first = discover_checkpoints([root], root=root)
    second = discover_checkpoints([root], root=root)
    assert [item.path for item in first] == [item.path for item in second]
    ranked = rank_candidates(first)
    assert ranked == first
    assert format_candidate_row(ranked[0], 1, root).startswith(" 1. [ ] ")


def test_default_search_bases_finds_every_store(tmp_path):
    root = tmp_path / "repo"
    for name in ("rl-v2", "rl-v4", "rl-models-global", "unrelated"):
        (root / name / "checkpoints").mkdir(parents=True)
    (root / "rl-v2" / "checkpoints" / "champion.pth").write_bytes(b"a")
    (root / "rl-v4" / "checkpoints" / "bc_best.pth").write_bytes(b"b")
    (root / "rl-models-global" / "champion" / "current").mkdir(parents=True)
    (root / "rl-models-global" / "champion" / "current" / "champion.pth").write_bytes(b"c")
    (root / "unrelated" / "checkpoints" / "ignored.pth").write_bytes(b"d")

    found = {os.path.basename(item.path) for item in discover_checkpoints(root=str(root))}
    assert found == {"champion.pth", "bc_best.pth"}
