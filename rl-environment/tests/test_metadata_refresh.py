import json
import sys
from pathlib import Path

import pytest


if "rl-environment" not in sys.path:
    sys.path.append("rl-environment")

import metadata_refresh  # noqa: E402


def test_ensure_card_metadata_uses_existing_requirements(tmp_path: Path) -> None:
    metadata_path = tmp_path / "card_metadata.json"
    metadata_path.write_text(json.dumps({"Capital": {"requirements": [{"oceans": 4}]}}), encoding="utf-8")
    assert metadata_refresh.ensure_card_metadata(root=tmp_path, quiet=True) == metadata_path


def test_ensure_card_metadata_regenerates_when_generator_available(monkeypatch, tmp_path: Path) -> None:
    metadata_path = tmp_path / "card_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    tm_root = tmp_path / "terraforming-mars"
    tm_root.mkdir()
    generator_path = tmp_path / "scripts" / "generate_card_metadata_with_requirements.cjs"
    generator_path.parent.mkdir()
    generator_path.write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(metadata_refresh, "can_generate_card_metadata", lambda root=None: True)
    monkeypatch.setattr(metadata_refresh, "_node_binary", lambda: "node")

    def _fake_run(command, cwd, check):
        assert command[0] == "node"
        metadata_path.write_text(json.dumps({"AI Central": {"requirements": [{"tag": "science", "count": 3}]}}), encoding="utf-8")
        return None

    monkeypatch.setattr(metadata_refresh.subprocess, "run", _fake_run)
    assert metadata_refresh.ensure_card_metadata(root=tmp_path, quiet=True) == metadata_path


def test_ensure_card_metadata_fails_when_refresh_required_but_impossible(tmp_path: Path) -> None:
    metadata_path = tmp_path / "card_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        metadata_refresh.ensure_card_metadata(root=tmp_path, quiet=True)

