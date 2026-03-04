import importlib
from typing import Any, Dict


_RUST_MODULE = None
_REQUIRED_FUNCTIONS = (
    "encode_state",
    "estimate_affordability",
    "can_afford_cards",
    "enumerate_card_selection_combos",
    "rank_startup_plans",
    "backend_info",
)


def get_rust_module(required: bool = True):
    global _RUST_MODULE
    if _RUST_MODULE is not None:
        return _RUST_MODULE
    try:
        _RUST_MODULE = importlib.import_module("rust_tfm_rl")
    except Exception as exc:
        if not required:
            return None
        raise RuntimeError(
            "Rust backend module 'rust_tfm_rl' is required but failed to import. "
            "Build/install the extension before running the RL agent."
        ) from exc

    missing = [name for name in _REQUIRED_FUNCTIONS if not hasattr(_RUST_MODULE, name)]
    if missing:
        raise RuntimeError(
            "Rust backend module is missing required functions: "
            + ", ".join(sorted(missing))
        )
    return _RUST_MODULE


def require_backend_info() -> Dict[str, Any]:
    module = get_rust_module(required=True)
    info = module.backend_info()
    if not isinstance(info, dict):
        raise RuntimeError("Rust backend returned invalid backend_info payload")
    if str(info.get("api_version", "")).strip() != "1.0":
        raise RuntimeError(
            f"Unsupported Rust backend api_version '{info.get('api_version')}'. Expected '1.0'."
        )
    return info
