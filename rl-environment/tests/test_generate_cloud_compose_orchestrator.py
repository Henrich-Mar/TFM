import os
import subprocess
import sys
from pathlib import Path


def test_generated_compose_includes_orchestrator_for_multi_coordinator(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_rl_cloud_compose.py"
    output_path = tmp_path / "docker-compose.generated.yml"

    cmd = [
        sys.executable,
        str(script_path),
        "--base-compose",
        str(repo_root / "docker-compose.rl_hard.yml"),
        "--output",
        str(output_path),
        "--public-host",
        "localhost",
        "--base-port",
        "8081",
        "--num-coordinators",
        "2",
        "--min-servers",
        "2",
        "--max-servers",
        "2",
        "--games-per-server",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    generated = output_path.read_text(encoding="utf-8")
    assert "rl-champion-orchestrator:" in generated
    assert "ORCH_COORD_SOURCES=coord-1=/app/coord-models/coord-1,coord-2=/app/coord-models/coord-2" in generated
    assert "GLOBAL_DASHBOARD_COORDINATOR_URLS=coord-1=http://rl-coordinator-1:5000,coord-2=http://rl-coordinator-2:5000" in generated
    assert generated.count("NUM_COORDINATORS=2") == 2
    assert "COORDINATOR_ID=coord-1" in generated
    assert "COORDINATOR_ID=coord-2" in generated
    assert "SAVE_EVERY_N_GENERATIONS=1" in generated
    assert "TRAINING_POOL_EXTRA_CHECKPOINTS=/app/rl-models-global/champion/current/champion.pth" in generated


def test_generated_compose_applies_per_coordinator_concurrency_caps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_rl_cloud_compose.py"
    output_path = tmp_path / "docker-compose.generated.yml"

    cmd = [
        sys.executable,
        str(script_path),
        "--base-compose",
        str(repo_root / "docker-compose.rl_hard.yml"),
        "--output",
        str(output_path),
        "--public-host",
        "localhost",
        "--base-port",
        "8081",
        "--num-coordinators",
        "6",
        "--min-servers",
        "54",
        "--max-servers",
        "54",
        "--games-per-server",
        "6",
        "--global-game-cap-per-coord",
        "20",
        "--tournament-cap-per-coord",
        "20",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))

    generated = output_path.read_text(encoding="utf-8")
    assert generated.count("GLOBAL_GAME_CONCURRENCY=20") == 6
    assert generated.count("TOURNAMENT_CONCURRENCY=20") == 6
    assert generated.count("MAX_ACTIVE_GAMES_PER_SERVER=5") == 6


def test_generated_compose_passes_through_runtime_tuning_env(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_rl_cloud_compose.py"
    output_path = tmp_path / "docker-compose.generated.yml"

    monkeypatch.setenv("PPO_PARALLEL_AGENTS", "1")
    monkeypatch.setenv("PPO_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("PPO_MINIBATCH_SIZE", "512")
    monkeypatch.setenv("AGENT_INFERENCE_BATCH_SIZE", "64")
    monkeypatch.setenv("AGENT_INFERENCE_THREADS", "8")
    monkeypatch.setenv("TRAINING_POOL_GENERATION_WINDOW", "10")
    monkeypatch.setenv("TRAINING_POOL_GAMES_PER_AGENT", "1")
    monkeypatch.setenv("FIXED_BENCHMARK_INTERVAL", "3")
    monkeypatch.setenv("MIN_EVAL_COMPLETION_RATIO", "0.80")
    monkeypatch.setenv("MIN_FROZEN_POOL_COMPLETION_RATE", "0.80")
    monkeypatch.setenv("RL_GAME_OPTIONS_FILE", "/app/game_options.fast_training.json")

    cmd = [
        sys.executable,
        str(script_path),
        "--base-compose",
        str(repo_root / "docker-compose.rl_hard.yml"),
        "--output",
        str(output_path),
        "--public-host",
        "localhost",
        "--base-port",
        "8081",
        "--min-servers",
        "2",
        "--max-servers",
        "2",
        "--games-per-server",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=os.environ.copy())

    generated = output_path.read_text(encoding="utf-8")
    assert "PPO_PARALLEL_AGENTS=1" in generated
    assert "PPO_EXECUTOR_WORKERS=1" in generated
    assert "PPO_MINIBATCH_SIZE=512" in generated
    assert "AGENT_INFERENCE_BATCH_SIZE=64" in generated
    assert "AGENT_INFERENCE_THREADS=8" in generated
    assert "TRAINING_POOL_GENERATION_WINDOW=10" in generated
    assert "TRAINING_POOL_GAMES_PER_AGENT=1" in generated
    assert "FIXED_BENCHMARK_INTERVAL=3" in generated
    assert "MIN_EVAL_COMPLETION_RATIO=0.80" in generated
    assert "MIN_FROZEN_POOL_COMPLETION_RATE=0.80" in generated
    assert "GAME_OPTIONS_FILE=/app/game_options.fast_training.json" in generated
