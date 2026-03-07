import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    rl_env = repo_root / "rl-environment"
    if str(rl_env) not in sys.path:
        sys.path.insert(0, str(rl_env))

    from metadata_refresh import _main  # type: ignore

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
