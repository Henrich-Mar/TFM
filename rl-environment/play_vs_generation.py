"""
Launch a human-vs-AI Terraforming Mars game using saved RL checkpoints.

Examples:
  python play_vs_generation.py --generation 42 --servers localhost:8081 --human-name You
  python play_vs_generation.py --random-generation --servers localhost:8081 --seed 123
  python play_vs_generation.py --generation 42 --agent-indices 0,1,2 --servers localhost:8081
"""
import argparse
import asyncio
import glob
import os
import random
import uuid
from typing import List, Optional

from game_interface import GameServerCluster, GameInstance
from models.agent import RLAgent


def _default_models_root() -> str:
    env_path = os.getenv("RL_MODELS_DIR")
    if env_path:
        return env_path
    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(base_dir, "rl-models"),
        os.path.abspath(os.path.join(base_dir, "..", "rl-models")),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _parse_generation_dirs(models_root: str) -> List[int]:
    generations: List[int] = []
    pattern = os.path.join(models_root, "generation_*")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        try:
            generations.append(int(base.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(generations))


def _fitness_from_name(path: str) -> float:
    base = os.path.basename(path)
    try:
        return float(base.split("_fitness_")[-1].replace(".pth", ""))
    except Exception:
        return -1.0


def _find_checkpoint_for_index(models_root: str, generation: int, agent_index: int) -> Optional[str]:
    gen_dir = os.path.join(models_root, f"generation_{generation}")
    pattern = os.path.join(gen_dir, f"agent_{agent_index}_fitness_*.pth")
    matches = sorted(glob.glob(pattern), key=_fitness_from_name, reverse=True)
    return matches[0] if matches else None


def _all_generation_checkpoints(models_root: str, generation: int) -> List[str]:
    gen_dir = os.path.join(models_root, f"generation_{generation}")
    pattern = os.path.join(gen_dir, "agent_*_fitness_*.pth")
    return sorted(glob.glob(pattern), key=_fitness_from_name, reverse=True)


def _load_agent(checkpoint_path: str) -> RLAgent:
    agent = RLAgent()
    agent.load_model(checkpoint_path)
    # Human-vs-AI should be pure inference.
    agent.train_from_self_play = False
    return agent


def _choose_generation(args: argparse.Namespace, rng: random.Random) -> int:
    if args.generation is not None:
        return int(args.generation)

    generations = _parse_generation_dirs(args.models)
    if not generations:
        raise FileNotFoundError(f"No generation directories found under: {args.models}")

    return rng.choice(generations)


def _parse_agent_indices(raw: str) -> List[int]:
    items = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    out: List[int] = []
    for item in items:
        out.append(int(item))
    return out


def _choose_checkpoints(
    models_root: str,
    generation: int,
    bot_count: int,
    rng: random.Random,
    requested_indices: Optional[List[int]],
) -> List[str]:
    chosen: List[str] = []
    all_ckpts = _all_generation_checkpoints(models_root, generation)
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints found for generation {generation} under {models_root}")

    if requested_indices:
        for idx in requested_indices:
            ckpt = _find_checkpoint_for_index(models_root, generation, idx)
            if ckpt is None:
                raise FileNotFoundError(
                    f"No checkpoint found for generation {generation}, agent index {idx}"
                )
            chosen.append(ckpt)
    else:
        if len(all_ckpts) >= bot_count:
            chosen = rng.sample(all_ckpts, k=bot_count)
        else:
            chosen = list(all_ckpts)

    while len(chosen) < bot_count:
        chosen.append(rng.choice(all_ckpts))

    return chosen[:bot_count]


def _resolve_public_base(game_instance: GameInstance) -> str:
    try:
        return game_instance._resolve_public_base()  # pylint: disable=protected-access
    except Exception:
        return str(os.getenv("PUBLIC_TM_URL", "http://localhost:8081")).rstrip("/")


def _print_standings(final_state: dict):
    players = final_state.get("players", []) or []
    if not players:
        print("No final standings returned.")
        return

    def vp_total(player: dict) -> int:
        return int(((player.get("victoryPointsBreakdown", {}) or {}).get("total", 0) or 0))

    rows = sorted(
        players,
        key=lambda p: (vp_total(p), int(p.get("megaCredits", 0) or 0)),
        reverse=True,
    )
    print("\nFinal standings:")
    for i, p in enumerate(rows, start=1):
        print(
            f"  {i}. {p.get('name', 'Unknown'):<20} "
            f"VP={vp_total(p):>3} TR={int(p.get('terraformRating', 0) or 0):>2} "
            f"MC={int(p.get('megaCredits', 0) or 0):>3}"
        )


async def _run_match(args: argparse.Namespace):
    rng = random.Random(args.seed)
    generation = _choose_generation(args, rng)
    checkpoints = _choose_checkpoints(
        models_root=args.models,
        generation=generation,
        bot_count=args.bots,
        rng=rng,
        requested_indices=_parse_agent_indices(args.agent_indices) if args.agent_indices else None,
    )

    print(f"Using generation: {generation}")
    for i, ckpt in enumerate(checkpoints, start=1):
        print(f"  Bot {i}: {ckpt}")

    bot_agents = [_load_agent(ckpt) for ckpt in checkpoints]
    bot_names = [f"AI_{agent.id[:8]}" for agent in bot_agents]
    player_names = [args.human_name] + bot_names

    server_addrs = [s.strip() for s in args.servers.split(",") if s.strip()]
    async with GameServerCluster(server_addrs) as cluster:
        await cluster.health_check()
        game_instance = await cluster.create_game(
            game_id=str(uuid.uuid4()),
            player_names=player_names,
            game_options={
                "soloMode": False,
                "showTimers": args.show_timers,
                "fastModeOption": args.fast_mode,
                "undoOption": False,
            },
        )

        human_player_id = await game_instance.join_player(args.human_name)
        public_base = _resolve_public_base(game_instance)

        game_url = f"{public_base}/game?id={game_instance.game_id}"
        player_url = f"{public_base}/player?id={human_player_id}"
        print(f"\nGame created: {game_instance.game_id}")
        print(f"Spectator URL: {game_url}")
        print(f"Your player URL: {player_url}")
        print("\nKeep this script running while you play. Press Ctrl+C to stop.")

        bot_tasks = [
            asyncio.create_task(agent.play_game(game_instance, bot_name))
            for agent, bot_name in zip(bot_agents, bot_names)
        ]

        try:
            results = await asyncio.gather(*bot_tasks, return_exceptions=True)
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                print(f"\nBot task failures: {len(failures)}")
                for err in failures[:3]:
                    print(f"  - {err}")
        finally:
            try:
                final_state = await game_instance.get_final_state()
                _print_standings(final_state)
            except Exception as e:
                print(f"\nCould not fetch final state: {e}")
            await game_instance.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Play Terraforming Mars vs trained RL agents from saved generations."
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Generation number to load. If omitted with --random-generation, one is chosen.",
    )
    parser.add_argument(
        "--random-generation",
        action="store_true",
        help="Pick a random generation from models root.",
    )
    parser.add_argument(
        "--agent-indices",
        type=str,
        default="",
        help="Comma-separated saved agent indices (e.g. 0,1,2). If omitted, random checkpoints are used.",
    )
    parser.add_argument("--bots", type=int, default=3, help="Number of AI opponents (default: 3).")
    parser.add_argument("--human-name", type=str, default="Human", help="Your in-game player name.")
    parser.add_argument("--servers", type=str, default=os.getenv("GAME_SERVERS", "localhost:8080"))
    parser.add_argument(
        "--models",
        type=str,
        default=_default_models_root(),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible selection.")
    parser.add_argument("--fast-mode", action="store_true", help="Enable TM fast mode for quicker turns.")
    parser.add_argument("--show-timers", action="store_true", help="Enable TM turn timers in UI.")

    args = parser.parse_args()
    if args.bots < 1:
        raise ValueError("--bots must be at least 1")
    if args.bots > 3:
        raise ValueError("--bots cannot exceed 3 for a 4-player game")
    if args.generation is None and not args.random_generation:
        parser.error("Provide --generation N or --random-generation")

    try:
        asyncio.run(_run_match(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
