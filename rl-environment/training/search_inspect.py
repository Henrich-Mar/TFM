"""Play one fixed-seed game with MCTS search and dump the search trees.

For every strategic prompt the candidate searches, this prints and saves a
human-readable tree: policy prior (P), visit count (N), mean simulated value
(Q), per-action rollout means, and the chosen action marked with ``*``.
Intended for a single-server inspection run, not for training.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy, RandomLegalPolicy
from search.config import SearchConfig
from search.search_agent import SearchDecision, SearchPolicy
from tournament_manager import TournamentManager
from v2_runtime import initialize_v2_runtime


class TracingSearchPolicy(SearchPolicy):
    def __init__(self, agent: Any, config: SearchConfig) -> None:
        super().__init__(agent, config)
        self.traces: List[Dict[str, Any]] = []
        self._stats_before = dict(self.stats)

    async def decide(self, **kwargs) -> Optional[SearchDecision]:
        player_state = kwargs.get("player_state") or {}
        before = dict(self.stats)
        decision = await super().decide(**kwargs)
        game = player_state.get("game", {}) or {}
        this_player = player_state.get("thisPlayer", {}) or {}
        entry: Dict[str, Any] = {
            "decision": len(self.traces) + 1,
            "player_id": str(kwargs.get("player_id", "")),
            "color": str(this_player.get("color", "") or ""),
            "generation": game.get("generation"),
            "prompt": player_state.get("waitingFor", {}).get("type", ""),
            "searched": decision is not None,
        }
        if decision is not None:
            entry["chosen_action_index"] = decision.action_index
            entry["chosen_label"] = decision.meta.get("chosen_action_label", "")
            entry["legal_action_count"] = len(decision.meta.get("legal_actions", []))
            entry["root_value"] = decision.meta.get("value_old")
            entry["mcts"] = decision.meta.get("mcts", {})
            entry["search_telemetry"] = decision.meta.get("search_telemetry", {})
        else:
            deltas = {
                key: int(self.stats.get(key, 0) - before.get(key, 0))
                for key in self.stats
                if int(self.stats.get(key, 0) - before.get(key, 0)) > 0
            }
            entry["fallback"] = deltas
        self.traces.append(entry)
        print(render_entry(entry), flush=True)
        return decision


def render_entry(entry: Dict[str, Any]) -> str:
    lines: List[str] = []
    header = (
        f"=== decision #{entry['decision']} | player={entry['player_id']} "
        f"color={entry.get('color') or '?'} gen={entry.get('generation')} "
        f"prompt={entry.get('prompt')}"
    )
    if not entry.get("searched"):
        lines.append(header)
        lines.append(f"    no search ({entry.get('fallback', {})}) -> policy fallback")
        return "\n".join(lines)
    mcts = entry.get("mcts", {}) or {}
    mode = str(mcts.get("mode", "?"))
    if mode in {"lookahead", "portfolio"}:
        counts = f"samples valid={mcts.get('valid_samples')} invalid={mcts.get('invalid_samples')}"
    else:
        counts = (
            f"sims={mcts.get('simulations')}/{mcts.get('simulations_selected')} "
            f"invalid={mcts.get('invalid_rollouts')} ({float(mcts.get('invalid_rate') or 0.0) * 100.0:.0f}%) "
            f"killed={mcts.get('killed_edges')}"
        )
        boot = mcts.get("bootstrapped") or {}
        boot_n = sum(int(value or 0) for value in boot.values()) if isinstance(boot, dict) else 0
        if boot_n:
            counts += f" bootstrapped={boot_n}"
    lines.append(
        f"{header} | {mode} {counts} | root_value={float(entry.get('root_value') or 0.0):+.3f}"
        f" | legal={entry.get('legal_action_count')}"
    )

    def walk(rows: List[Dict[str, Any]], depth: int) -> None:
        ordered = sorted(rows, key=lambda row: (
            -float(row.get("visits", 0)),
            -float(row.get("mean_value") if row.get("mean_value") is not None else row.get("q", 0.0)),
        ))
        for row in ordered:
            marker = "* " if row.get("action_index") == entry.get("chosen_action_index") else "  "
            value = row.get("mean_value")
            if value is None:
                value = row.get("q", 0.0)
            flags = "".join(
                flag for flag, on in (("dead", row.get("dead")),) if on
            )
            if not flags and int(row.get("failures", 0) or 0) > 0:
                flags = f"fail x{int(row.get('failures', 0))}"
            suffix = f" [{flags}]" if flags else ""
            lines.append(
                f"    {'|   ' * depth}{marker}P={float(row.get('prior', 0.0)):.2f} "
                f"N={int(row.get('visits', 0)):>3} V={float(value):+.3f}  {row.get('label') or ''}{suffix}"
            )
            children = row.get("children")
            if isinstance(children, list) and children:
                walk(children, depth + 1)

    walk(list(mcts.get("candidates", [])), 0)
    return "\n".join(lines)


async def inspect(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("TM_GAME_TIMEOUT_SEC", "1800")
    initialize_v2_runtime()
    cfg = SearchConfig.from_env()
    cfg.enabled = True
    if args.mode:
        cfg.mode = args.mode
    if args.top_k:
        cfg.top_k = int(args.top_k)
    if args.simulations:
        cfg.simulations_per_move = int(args.simulations)
    if args.determinizations:
        cfg.determinizations = int(args.determinizations)
    if args.depth:
        cfg.max_root_turns_depth = int(args.depth)
    if args.puct_c is not None:
        cfg.puct_c = float(args.puct_c)
    if args.selection:
        cfg.selection = args.selection
    if args.seed_of_run is not None:
        cfg.seed = int(args.seed_of_run)
    cfg.normalize()

    candidate = RLAgent(agent_id="mcts-inspect")
    candidate.load_model(args.checkpoint)
    candidate.train_from_self_play = False
    candidate.config.train_from_self_play = False
    candidate.ppo_enable = False
    candidate.deterministic_actions = True
    policy = TracingSearchPolicy(candidate, cfg)
    candidate.search_policy = policy

    baseline_agents: List[RLAgent] = []
    for idx in range(3):
        if args.baseline == "random":
            agent = RLAgent(agent_id=f"random-{idx}", decision_policy=RandomLegalPolicy(args.game_seed + idx))
        else:
            agent = RLAgent(agent_id=f"teacher-{idx}", decision_policy=HeuristicTeacherPolicy(args.game_seed + idx, sample=False))
        agent.train_from_self_play = False
        agent.config.train_from_self_play = False
        agent.ppo_enable = False
        baseline_agents.append(agent)

    cluster = GameServerCluster(
        [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
    )
    options_path = ROOT / f"game_options.v2_stage{int(args.stage)}.json"
    cluster.base_game_options = json.loads(options_path.read_text(encoding="utf-8"))
    manager = TournamentManager(cluster)
    lineup = list(baseline_agents)
    lineup.insert(args.seat, candidate)
    print(
        f"[inspect] checkpoint={Path(args.checkpoint).name} seed={args.game_seed} seat={args.seat} "
        f"mode={cfg.mode} top_k={cfg.top_k} det={cfg.determinizations} sims={cfg.simulations_per_move} "
        f"depth={cfg.max_root_turns_depth} c_puct={cfg.puct_c}",
        flush=True,
    )
    try:
        result = await manager._run_single_game(
            lineup,
            tournament_id=f"mcts_inspect_{args.game_seed}_{args.seat}",
            game_seed=args.game_seed,
            players_beginner=(int(args.stage) == 0),
        )
    finally:
        await cluster.close()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"mcts_trace_seed{args.game_seed}_seat{args.seat}"
    trace_text = "\n\n".join(render_entry(entry) for entry in policy.traces)
    summary = (
        f"\n=== summary ===\n"
        f"games_completed={bool(result.completed)}\n"
        f"final={json.dumps([{'agent_id': row.get('agent_id'), 'rank': row.get('rank'), 'vp': row.get('victory_points')} for row in result.players], indent=2)}\n"
        f"{json.dumps(policy.snapshot_stats(), indent=2)}\n"
    )
    stem.with_suffix(".txt").write_text(trace_text + "\n" + summary, encoding="utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": {
                    "mode": cfg.mode,
                    "top_k": cfg.top_k,
                    "determinizations": cfg.determinizations,
                    "simulations_per_move": cfg.simulations_per_move,
                    "max_root_turns_depth": cfg.max_root_turns_depth,
                    "puct_c": cfg.puct_c,
                    "selection": cfg.selection,
                    "seed": cfg.seed,
                    "root_prompt_types": cfg.root_prompt_types,
                    "adaptive_simulations": cfg.adaptive_simulations,
                    "simulations_two_actions": cfg.simulations_two_actions,
                    "simulations_four_actions": cfg.simulations_four_actions,
                    "leaf_batch": cfg.leaf_batch,
                    "early_stop": cfg.early_stop,
                    "early_stop_min_simulations": cfg.early_stop_min_simulations,
                },
                "game_seed": args.game_seed,
                "seat": args.seat,
                "completed": bool(result.completed),
                "players": [
                    {"agent_id": row.get("agent_id"), "rank": row.get("rank"), "vp": row.get("victory_points")}
                    for row in result.players
                ],
                "search_stats": policy.snapshot_stats(),
                "traces": policy.traces,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary, flush=True)
    print(f"[inspect] trace saved: {stem.with_suffix('.txt')}  ({len(policy.traces)} decisions recorded)")
    return {"traces": len(policy.traces)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump MCTS search trees from one fixed-seed game")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--game-seed", type=int, default=920003)
    parser.add_argument("--seat", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--stage", type=int, default=1, choices=(0, 1))
    parser.add_argument("--baseline", choices=("teacher", "random"), default="teacher")
    parser.add_argument("--mode", choices=("lookahead", "puct"), default=None)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--puct-c", type=float, default=None)
    parser.add_argument("--selection", choices=("argmax", "temperature"), default=None)
    parser.add_argument("--seed-of-run", dest="seed_of_run", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.getenv("V2_METRICS_DIR", str(ROOT / "alphago" / "metrics")),
    )
    args = parser.parse_args()
    asyncio.run(inspect(args))


if __name__ == "__main__":
    main()
