"""Environment-driven configuration for the Python MCTS search consumer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_BRANCHES_PER_BATCH = 64
MAX_STEPS_PER_BRANCH = 32
SECONDS_PER_DAY = 86400.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(str(raw).strip())
    except ValueError:
        return float(default)
    if value != value or value in (float("inf"), float("-inf")):
        return float(default)
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SearchConfig:
    """One searched decision = top_k candidates x determinizations rolled out.

    Defaults follow the AlphaGo plan: inexpensive flat lookahead first,
    PUCT selectable once the fixed-seed benchmarks validate the loop.
    """

    enabled: bool = False
    mode: str = "lookahead"
    top_k: int = 8
    determinizations: int = 8
    simulations_per_move: int = 32
    max_root_turns_depth: int = 2
    puct_c: float = 1.5
    leaf_batch: int = 32
    max_rollout_steps: int = MAX_STEPS_PER_BRANCH
    max_replay_rounds: int = 40
    request_timeout_sec: float = 30.0
    decide_timeout_sec: float = 45.0
    selection: str = "argmax"
    temperature: float = 1.0
    seed: int = 0
    require_determinized: bool = True
    stats_log_every: int = 50
    root_noise_alpha: float = 0.05
    root_noise_weight: float = 0.0
    edge_kill_failures: int = 3
    value_horizon: int = 0
    root_prompt_types: str = "or,card,space"
    adaptive_simulations: bool = True
    simulations_two_actions: int = 8
    simulations_four_actions: int = 16
    early_stop: bool = True
    early_stop_min_simulations: int = 16
    temperature_until_generation: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "SearchConfig":
        cfg = cls(
            enabled=_env_bool("ALPHAGO_SEARCH_ENABLED", False),
            mode=_env_str("ALPHAGO_SEARCH_MODE", "lookahead"),
            top_k=_env_int("ALPHAGO_SEARCH_TOP_K", 8),
            determinizations=_env_int("ALPHAGO_SEARCH_DETERMINIZATIONS", 8),
            simulations_per_move=_env_int("ALPHAGO_SEARCH_SIMULATIONS", 32),
            max_root_turns_depth=_env_int("ALPHAGO_SEARCH_DEPTH", 2),
            puct_c=_env_float("ALPHAGO_SEARCH_PUCT_C", 1.5),
            leaf_batch=_env_int("ALPHAGO_SEARCH_LEAF_BATCH", 32),
            max_rollout_steps=_env_int("ALPHAGO_SEARCH_MAX_ROLLOUT_STEPS", MAX_STEPS_PER_BRANCH),
            max_replay_rounds=_env_int("ALPHAGO_SEARCH_MAX_REPLAY_ROUNDS", 40),
            request_timeout_sec=_env_float("ALPHAGO_SEARCH_REQUEST_TIMEOUT_SEC", 30.0),
            decide_timeout_sec=_env_float("ALPHAGO_SEARCH_DECIDE_TIMEOUT_SEC", 45.0),
            selection=_env_str("ALPHAGO_SEARCH_SELECTION", "argmax"),
            temperature=_env_float("ALPHAGO_SEARCH_TEMPERATURE", 1.0),
            seed=_env_int("ALPHAGO_SEARCH_SEED", 0),
            require_determinized=_env_bool("ALPHAGO_SEARCH_REQUIRE_DETERMINIZED", True),
            stats_log_every=_env_int("ALPHAGO_SEARCH_STATS_LOG_EVERY", 50),
            root_noise_alpha=_env_float("ALPHAGO_SEARCH_ROOT_NOISE_ALPHA", 0.05),
            root_noise_weight=_env_float("ALPHAGO_SEARCH_ROOT_NOISE_WEIGHT", 0.0),
            edge_kill_failures=_env_int("ALPHAGO_SEARCH_EDGE_KILL_FAILURES", 3),
            value_horizon=_env_int("ALPHAGO_SEARCH_VALUE_HORIZON", 0),
            root_prompt_types=_env_str("ALPHAGO_SEARCH_ROOT_PROMPTS", "or,card,space"),
            adaptive_simulations=_env_bool("ALPHAGO_SEARCH_ADAPTIVE_SIMULATIONS", True),
            simulations_two_actions=_env_int("ALPHAGO_SEARCH_SIMULATIONS_TWO_ACTIONS", 8),
            simulations_four_actions=_env_int("ALPHAGO_SEARCH_SIMULATIONS_FOUR_ACTIONS", 16),
            early_stop=_env_bool("ALPHAGO_SEARCH_EARLY_STOP", True),
            early_stop_min_simulations=_env_int("ALPHAGO_SEARCH_EARLY_STOP_MIN_SIMULATIONS", 16),
            temperature_until_generation=_env_int("ALPHAGO_SEARCH_TEMPERATURE_UNTIL_GENERATION", 0),
        )
        cfg.normalize()
        return cfg

    def normalize(self) -> None:
        if self.mode not in {"lookahead", "puct"}:
            self.mode = "lookahead"
        if self.selection not in {"argmax", "temperature"}:
            self.selection = "argmax"
        self.top_k = max(1, min(int(self.top_k), MAX_BRANCHES_PER_BATCH))
        self.determinizations = max(1, int(self.determinizations))
        # One lookahead batch of top_k x determinizations must fit the server
        # limit of 64 branches per replay request.
        while self.top_k * self.determinizations > MAX_BRANCHES_PER_BATCH and self.determinizations > 1:
            self.determinizations -= 1
        if self.top_k * self.determinizations > MAX_BRANCHES_PER_BATCH:
            self.top_k = MAX_BRANCHES_PER_BATCH
        self.simulations_per_move = max(1, int(self.simulations_per_move))
        self.max_root_turns_depth = max(1, min(int(self.max_root_turns_depth), 4))
        self.leaf_batch = max(1, min(int(self.leaf_batch), MAX_BRANCHES_PER_BATCH))
        self.max_rollout_steps = max(1, min(int(self.max_rollout_steps), MAX_STEPS_PER_BRANCH))
        self.max_replay_rounds = max(1, int(self.max_replay_rounds))
        self.puct_c = max(0.0, float(self.puct_c))
        self.temperature = max(1e-3, float(self.temperature))
        self.request_timeout_sec = max(1.0, float(self.request_timeout_sec))
        self.decide_timeout_sec = max(self.request_timeout_sec, float(self.decide_timeout_sec))
        self.root_noise_alpha = max(0.0, float(self.root_noise_alpha))
        self.root_noise_weight = min(1.0, max(0.0, float(self.root_noise_weight)))
        self.edge_kill_failures = max(1, min(int(self.edge_kill_failures), 8))
        self.value_horizon = max(0, min(int(self.value_horizon), 8))
        self.simulations_two_actions = max(1, min(int(self.simulations_two_actions), self.simulations_per_move))
        self.simulations_four_actions = max(
            self.simulations_two_actions,
            min(int(self.simulations_four_actions), self.simulations_per_move),
        )
        self.early_stop_min_simulations = max(
            1,
            min(int(self.early_stop_min_simulations), self.simulations_per_move),
        )
        self.temperature_until_generation = max(0, int(self.temperature_until_generation))
        types = {t.strip() for t in str(self.root_prompt_types or "").split(",") if t.strip()}
        self.root_prompt_types = ",".join(sorted(types)) if types else "card,or,space"

    @property
    def lookahead_branch_count(self) -> int:
        return int(self.top_k) * int(self.determinizations)

    def simulation_budget(self, legal_action_count: int) -> int:
        """Return the PUCT budget for a root with ``legal_action_count``.

        Forced roots are handled by :class:`SearchPolicy` before a simulator
        session is opened. Keeping the zero result here makes the rule explicit
        and lets tests and future callers share the same policy.
        """
        legal = max(0, int(legal_action_count))
        if legal <= 1:
            return 0
        if not self.adaptive_simulations:
            return int(self.simulations_per_move)
        if legal == 2:
            return int(self.simulations_two_actions)
        if legal <= 4:
            return int(self.simulations_four_actions)
        return int(self.simulations_per_move)
