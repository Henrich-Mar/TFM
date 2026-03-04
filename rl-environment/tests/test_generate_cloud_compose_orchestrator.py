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
    assert "SAVE_EVERY_N_GENERATIONS=1" in generated
    assert "SAVE_EVERY_N_GENERATIONS=3" in generated
    assert "TRAINING_POOL_EXTRA_CHECKPOINTS=/app/rl-models-global/champion/current/champion.pth" in generated
