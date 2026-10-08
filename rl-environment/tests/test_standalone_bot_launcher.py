import os
import sys

import pytest


if "rl-environment" not in sys.path:
    sys.path.insert(0, "rl-environment")

pytest.importorskip("tkinter")

from standalone_bot import _build_arg_parser, _checkpoint_search_roots, _configure_action_selection  # noqa: E402
from standalone_bot_tk import StandaloneBotLauncher  # noqa: E402


def _args(*argv):
    return _build_arg_parser().parse_args(list(argv))


def test_explicit_search_roots_override_the_models_folder():
    args = _args("--models", "C:/models", "--search-root", "C:/a", "--search-root", "C:/b")
    assert _checkpoint_search_roots(args) == ["C:/a", "C:/b"]


def test_models_folder_is_used_when_no_search_root_is_given():
    args = _args("--models", "C:/models")
    assert _checkpoint_search_roots(args) == ["C:/models"]


def test_default_search_roots_are_used_when_nothing_is_configured():
    roots = _checkpoint_search_roots(_args())
    assert roots and all(os.path.isdir(root) for root in roots)
    assert any(root.endswith("rl-alphago") for root in roots)


def test_docker_mount_of_a_checkpoint_file_exposes_its_directory(tmp_path):
    # The coordinator image only mounts rl-environment and rl-models, so an
    # rl-alphago checkpoint must be bind-mounted to be usable in the container.
    checkpoint = tmp_path / "checkpoints" / "candidate_000000001.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    mounts, container_path = StandaloneBotLauncher._docker_mount_args(None, str(checkpoint), "/app/standalone-checkpoint")

    assert mounts[0] == "-v"
    assert mounts[1] == f"{checkpoint.parent}:/app/standalone-checkpoint/checkpoints:ro"
    assert container_path == "/app/standalone-checkpoint/checkpoints/candidate_000000001.pth"


def test_docker_mount_of_a_directory_maps_the_directory_itself(tmp_path):
    store = tmp_path / "rl-alphago"
    store.mkdir()

    mounts, container_path = StandaloneBotLauncher._docker_mount_args(None, str(store), "/app/standalone-roots")

    assert mounts[1] == f"{store}:/app/standalone-roots:ro"
    assert container_path == "/app/standalone-roots"


def test_docker_mount_of_a_missing_file_falls_back_to_its_parent(tmp_path):
    mounts, container_path = StandaloneBotLauncher._docker_mount_args(
        None, str(tmp_path / "gone" / "champion.pth"), "/app/standalone-checkpoint"
    )

    assert mounts[1].endswith(":ro")
    assert container_path.endswith("/gone/champion.pth") or container_path.endswith("/store/champion.pth")


def test_action_selection_defaults_to_argmax():
    assert _args().action_selection == "argmax"


@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_action_selection_configures_the_agent_like_the_benchmark(mode):
    from types import SimpleNamespace

    agent = SimpleNamespace(
        config=SimpleNamespace(train_from_self_play=True, epsilon=0.05, temperature=1.2),
        ppo_enable=True,
        deterministic_actions=False,
        policy_temperature_cap=1.0,
        policy_temperature_floor=0.75,
    )
    _configure_action_selection(agent, mode)
    assert agent.deterministic_actions is (mode == "argmax")
    assert agent.ppo_enable is (mode == "sample")
    if mode == "sample":
        assert agent.config.epsilon == 0.0
        assert agent.policy_temperature_floor == 1.0
