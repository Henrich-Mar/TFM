"""Play a human seat against N MCTS-searched copies of a trained checkpoint.

The game is created on the same ruleset the checkpoint was trained and gated on
(``game_options.v2_stage{N}.json``), every bot seat runs the checkpoint with a
``SearchPolicy`` attached, and the human seat is left open.

Examples:
  python -m play_vs_champion_mcts --checkpoint /app/alphago/checkpoints/champion.pth
  python -m play_vs_champion_mcts --mode puct --top-k 8 --determinizations 8 --simulations 32
"""
import argparse
import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from game_interface import GameServerCluster, GameInstance
from metadata_refresh import ensure_card_metadata
from models.agent import RLAgent
from search import SearchConfig, SearchPolicy

ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINTS = (
    Path("/app/alphago/checkpoints/champion.pth"),
    ROOT.parent / "rl-alphago" / "checkpoints" / "champion.pth",
)


def _default_checkpoint() -> str:
    for candidate in DEFAULT_CHECKPOINTS:
        if candidate.is_file():
            return str(candidate)
    return str(DEFAULT_CHECKPOINTS[-1])


def _load_options(stage: int) -> Dict[str, Any]:
    path = ROOT / f"game_options.v2_stage{stage}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_search_config(args: argparse.Namespace) -> SearchConfig:
    cfg = SearchConfig.from_env()
    cfg.enabled = not args.no_search
    cfg.mode = args.mode
    cfg.top_k = int(args.top_k)
    cfg.determinizations = int(args.determinizations)
    cfg.simulations_per_move = int(args.simulations)
    cfg.max_root_turns_depth = int(args.depth)
    cfg.leaf_batch = int(args.leaf_batch)
    cfg.decide_timeout_sec = max(cfg.decide_timeout_sec, float(args.decide_timeout_sec))
    cfg.selection = args.search_selection
    cfg.temperature = float(args.search_temperature)
    cfg.root_noise_weight = float(args.search_root_noise_weight)
    if args.puct_c is not None:
        cfg.puct_c = float(args.puct_c)
    if args.root_prompts:
        cfg.root_prompt_types = args.root_prompts
    cfg.normalize()
    return cfg


def _make_bot(index: int, checkpoint: str, cfg: SearchConfig, stochastic: bool) -> RLAgent:
    agent = RLAgent(agent_id=f"champion-mcts-{index}")
    agent.load_model(checkpoint)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.deterministic_actions = not stochastic
    if cfg.enabled:
        agent.search_policy = SearchPolicy(agent, cfg)
    return agent


async def _print_standings(game_instance: GameInstance, human_name: str) -> None:
    """A finished game's public view strips player detail, so ask each seat directly."""
    final_state = await game_instance.get_final_state()
    players = final_state.get("players", []) or []
    if not players:
        print("\nNo final standings returned.", flush=True)
        return

    rows = []
    for player in players:
        player_id = str(player.get("id", "") or "")
        name = str(player.get("name", "") or "")
        breakdown: Dict[str, Any] = {}
        megacredits = 0
        if player_id:
            try:
                seat_view = await game_instance.get_player_state(player_id)
            except Exception:
                seat_view = {}
            this_player = seat_view.get("thisPlayer", {}) or {}
            name = name or str(this_player.get("name", "") or "")
            breakdown = dict(this_player.get("victoryPointsBreakdown", {}) or {})
            megacredits = int(this_player.get("megaCredits", 0) or this_player.get("megacredits", 0) or 0)
        rows.append({"name": name, "vp": int(breakdown.get("total", 0) or 0), "mc": megacredits, "breakdown": breakdown})

    rows.sort(key=lambda row: (row["vp"], row["mc"]), reverse=True)
    print("\nFinal standings:", flush=True)
    for position, row in enumerate(rows, start=1):
        marker = " <- you" if row["name"] == human_name else ""
        breakdown = row["breakdown"]
        print(
            f"  {position}. {row['name']:<24} VP={row['vp']:>3} "
            f"TR={int(breakdown.get('terraformRating', 0) or 0):>2} "
            f"MS={int(breakdown.get('milestones', 0) or 0):>2} "
            f"AW={int(breakdown.get('awards', 0) or 0):>2} "
            f"MC={row['mc']:>3}{marker}",
            flush=True,
        )


async def _progress_printer(agents: List[RLAgent], stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
        if stop.is_set():
            return
        print("-" * 70, flush=True)
        for agent in agents:
            policy = getattr(agent, "search_policy", None)
            if policy is None:
                continue
            stats = policy.snapshot_stats()
            rollout = stats.get("rollout", {}) or {}
            print(
                f"  {agent.id:<22} searched={stats.get('searched_decisions', 0):>4} "
                f"fallbacks={stats.get('search_fallbacks', 0):>3} "
                f"mean_search={stats.get('mean_search_sec', 0.0):>6.2f}s "
                f"invalid_rate={rollout.get('invalid_rate', 0.0):.3f}",
                flush=True,
            )
        print("-" * 70, flush=True)


async def _run_match(args: argparse.Namespace) -> None:
    ensure_card_metadata(quiet=True)
    checkpoint = str(args.checkpoint or "").strip() or _default_checkpoint()
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    cfg = _build_search_config(args)
    bots = [_make_bot(idx, checkpoint, cfg, args.stochastic) for idx in range(args.bots)]
    bot_names = [f"MCTS_{idx + 1}" for idx in range(args.bots)]
    player_names = [args.human_name] + bot_names

    options = _load_options(args.stage)
    options["showTimers"] = bool(args.show_timers)
    if args.fast_mode is not None:
        options["fastModeOption"] = bool(args.fast_mode)
    if args.seed is not None:
        options["seed"] = int(args.seed)

    print(f"Checkpoint: {checkpoint}")
    if cfg.enabled:
        print(
            f"MCTS: mode={cfg.mode} top_k={cfg.top_k} determinizations={cfg.determinizations} "
            f"simulations={cfg.simulations_per_move} depth={cfg.max_root_turns_depth} "
            f"leaf_batch={cfg.leaf_batch} decide_timeout={cfg.decide_timeout_sec}s "
            f"selection={cfg.selection} prompts={cfg.root_prompt_types}"
        )
    else:
        print("MCTS: disabled (policy-only decisions)")
    selection = "stochastic" if args.stochastic else "greedy argmax"
    print(f"Bots: {args.bots} ({selection}{' + search' if cfg.enabled else ''})")
    print(f"Ruleset: stage {args.stage} options from game_options.v2_stage{args.stage}.json")

    servers = [item.strip() for item in args.servers.split(",") if item.strip()]
    cluster = GameServerCluster(servers)
    cluster.base_game_options = options

    stop = asyncio.Event()
    printer: Any = None
    async with cluster:
        await cluster.health_check()
        game_instance: GameInstance = await cluster.create_game(
            game_id=str(uuid.uuid4()),
            player_names=player_names,
            game_options={},
        )
        human_player_id = await game_instance.join_player(args.human_name)
        public_base = game_instance._resolve_public_base()  # pylint: disable=protected-access

        print(f"\nGame ID: {game_instance.game_id}")
        print(f"Spectator URL: {public_base}/game?id={game_instance.game_id}")
        print(f"Your player URL: {public_base}/player?id={human_player_id}")
        print("\nOpen your player URL to take your seat. Bots will not move until the game starts.")
        print("Keep this process running. Press Ctrl+C to stop.\n")

        printer = asyncio.create_task(_progress_printer(bots, stop, args.progress_interval_sec))
        bot_tasks = [
            asyncio.create_task(agent.play_game(game_instance, bot_name))
            for agent, bot_name in zip(bots, bot_names)
        ]
        try:
            results = await asyncio.gather(*bot_tasks, return_exceptions=True)
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                print(f"\nBot task failures: {len(failures)}")
                for err in failures[:5]:
                    print(f"  - {err}")
        except asyncio.CancelledError:
            for task in bot_tasks:
                task.cancel()
            raise
        finally:
            stop.set()
            if printer is not None:
                printer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await printer
            await _print_standings(game_instance, args.human_name)
            for agent in bots:
                policy = getattr(agent, "search_policy", None)
                if policy is None:
                    continue
                print(f"\n{agent.id} search stats:\n{json.dumps(policy.snapshot_stats(), indent=2)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human seat vs MCTS-searched copies of a trained TFM checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path. Default: alphago champion.")
    parser.add_argument("--bots", type=int, default=3, help="AI opponents (default: 3).")
    parser.add_argument("--human-name", type=str, default="Human", help="Your in-game name.")
    parser.add_argument("--servers", type=str, default=os.getenv("GAME_SERVERS", "localhost:8080"))
    parser.add_argument("--stage", type=int, default=1, choices=(0, 1), help="Ruleset preset.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed game seed.")
    parser.add_argument("--show-timers", action="store_true", help="Show turn timers in the UI.")
    fast = parser.add_mutually_exclusive_group()
    fast.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=None)
    fast.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.set_defaults(fast_mode=None, help="Override the stage preset's fastModeOption.")

    search = parser.add_argument_group("mcts")
    search.add_argument(
        "--no-search",
        action="store_true",
        help="Disable MCTS entirely; bots sample the policy directly, as self-play seats do.",
    )
    search.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of argmax. Matches the sampled-candidate "
             "configuration every logged benchmark (pairwise 0.664) was measured at.",
    )
    search.add_argument("--mode", type=str, default="lookahead", choices=("lookahead", "puct"))
    search.add_argument("--top-k", type=int, default=6, help="Policy candidates searched per root.")
    search.add_argument("--determinizations", type=int, default=4)
    search.add_argument("--simulations", type=int, default=16, help="PUCT simulation budget.")
    search.add_argument("--depth", type=int, default=2, help="Max root turns deep.")
    search.add_argument("--leaf-batch", type=int, default=32)
    search.add_argument("--puct-c", type=float, default=None)
    search.add_argument(
        "--search-selection",
        type=str,
        default="argmax",
        choices=("argmax", "temperature"),
        help="How a searched root picks its move.",
    )
    search.add_argument("--search-temperature", type=float, default=1.0)
    search.add_argument("--search-root-noise-weight", type=float, default=0.0)
    search.add_argument("--decide-timeout-sec", type=float, default=90.0)
    search.add_argument("--root-prompts", type=str, default="", help="Override root prompt types.")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.bots < 1 or args.bots > 3:
        raise ValueError("--bots must be between 1 and 3 for a 4-player game")
    try:
        asyncio.run(_run_match(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:  # noqa: BLE001
        print(f"\nFailed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()