import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def card_metadata_path(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "card_metadata.json"


def terraforming_mars_root(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "terraforming-mars"


def generator_script_path(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "scripts" / "generate_card_metadata_with_requirements.cjs"


def metadata_has_requirements(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    for meta in payload.values():
        if not isinstance(meta, dict):
            continue
        requirements = meta.get("requirements")
        if isinstance(requirements, list):
            return True
    return False


def _node_binary() -> Optional[str]:
    return shutil.which("node")


def can_generate_card_metadata(root: Optional[Path] = None) -> bool:
    base = Path(root) if root is not None else repo_root()
    tm_root = terraforming_mars_root(base)
    generator_path = generator_script_path(base)
    if not tm_root.is_dir():
        return False
    if not generator_path.is_file():
        return False
    if _node_binary() is None:
        return False
    ts_node_bin = tm_root / "node_modules" / "ts-node" / "dist" / "bin.js"
    return ts_node_bin.is_file()


def ensure_card_metadata(root: Optional[Path] = None, quiet: bool = False) -> Path:
    base = Path(root) if root is not None else repo_root()
    metadata_path = card_metadata_path(base)
    tm_root = terraforming_mars_root(base)
    generator_path = generator_script_path(base)

    if tm_root.is_dir() and can_generate_card_metadata(base):
        command = [_node_binary() or "node", str(generator_path)]
        if not quiet:
            logger.info("Refreshing card metadata from %s", tm_root)
        subprocess.run(command, cwd=str(base), check=True)
        if not metadata_has_requirements(metadata_path):
            raise RuntimeError(f"Generated metadata at {metadata_path} does not include requirements")
        return metadata_path

    if metadata_has_requirements(metadata_path):
        if not quiet:
            logger.info("Using existing card metadata with requirements: %s", metadata_path)
        return metadata_path

    missing_bits = []
    if not tm_root.is_dir():
        missing_bits.append(f"missing checkout {tm_root}")
    if _node_binary() is None:
        missing_bits.append("missing node executable")
    ts_node_bin = tm_root / "node_modules" / "ts-node" / "dist" / "bin.js"
    if tm_root.is_dir() and not ts_node_bin.is_file():
        missing_bits.append(f"missing {ts_node_bin}")
    if not generator_path.is_file():
        missing_bits.append(f"missing generator {generator_path}")

    raise RuntimeError(
        "Card metadata refresh is required but cannot run, and existing card_metadata.json has no requirements. "
        + "; ".join(missing_bits)
    )


def _main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    try:
        path = ensure_card_metadata(quiet=quiet)
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if not quiet:
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
