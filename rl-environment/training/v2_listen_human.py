"""Serve an opt-in passive collector for games played in the TFM web client."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

import uvicorn

from api import server
from training.human_game_listener import HumanGameListener
from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy
from v2_runtime import initialize_v2_runtime


async def _run_bot_game(game_instance, bots: list[RLAgent], bot_names: list[str]) -> None:
    try:
        await asyncio.gather(
            *(agent.play_game(game_instance, bot_name) for agent, bot_name in zip(bots, bot_names)),
            return_exceptions=True,
        )
    finally:
        await game_instance.cleanup()


async def _run_and_mark_bot_game(game_instance, bots: list[RLAgent], bot_names: list[str]) -> None:
    try:
        await _run_bot_game(game_instance, bots, bot_names)
    finally:
        server.mark_human_listener_game_complete(game_instance.game_id)


async def _create_human_games(args: argparse.Namespace, background_tasks: set[asyncio.Task]) -> list[dict]:
    if int(args.games) <= 0:
        return []
    servers = [item.strip() for item in os.getenv("GAME_SERVERS", "localhost:8080").split(",") if item.strip()]
    cluster = GameServerCluster(servers)
    games: list[dict] = []
    bot_tasks: list[asyncio.Task] = []
    try:
        for game_number in range(1, int(args.games) + 1):
            bot_names = [f"ListenerBot{game_number}-{seat}" for seat in range(1, int(args.bot_count) + 1)]
            game_instance = await cluster.create_game(
                game_id=str(uuid.uuid4()),
                player_names=[args.human_name, *bot_names],
                game_options={
                    "soloMode": False,
                    "randomMA": "No randomization",
                    "fastModeOption": False,
                    "removeNegativeGlobalEventsOption": True,
                    "undoOption": False,
                    "seed": int(args.seed_start) + game_number - 1,
                    "_players_beginner": bool(args.beginner),
                },
            )
            human_player_id = await game_instance.join_player(args.human_name)
            bots = []
            for seat, bot_name in enumerate(bot_names, start=1):
                bot = RLAgent(
                    agent_id=f"human-listener-bot-{game_number}-{seat}",
                    decision_policy=HeuristicTeacherPolicy(seed=int(args.seed_start) + (game_number * 10) + seat, temperature=0.18, sample=True),
                )
                bot.train_from_self_play = False
                bot.config.train_from_self_play = False
                bot.ppo_enable = False
                bots.append(bot)
            task = asyncio.create_task(_run_and_mark_bot_game(game_instance, bots, bot_names))
            bot_tasks.append(task)
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            games.append(
                {
                    "game_number": game_number,
                    "game_id": game_instance.game_id,
                    "seed": int(args.seed_start) + game_number - 1,
                    "player_url": game_instance.get_public_player_url(human_player_id),
                    "game_url": game_instance.get_public_game_url(),
                    "status": "active",
                }
            )
    except Exception:
        await cluster.close()
        raise
    # The games retain the session managed by the cluster, so close it only
    # after every bot task has finished.
    async def close_when_done() -> None:
        if bot_tasks:
            await asyncio.gather(*bot_tasks, return_exceptions=True)
        await cluster.close()
    cleanup_task = asyncio.create_task(close_when_done())
    background_tasks.add(cleanup_task)
    cleanup_task.add_done_callback(background_tasks.discard)
    return games


async def _serve(args: argparse.Namespace) -> None:
    listener = HumanGameListener(
        args.dataset,
        event_dir=args.event_dir or None,
        player_ids=args.player_id,
        player_names=args.player_name,
    )
    server.configure_human_game_listener(listener, token=args.token)
    api_server = uvicorn.Server(uvicorn.Config(server.app, host=args.host, port=args.port, log_level="info"))
    api_task = asyncio.create_task(api_server.serve())
    while not api_server.started and not api_task.done():
        await asyncio.sleep(0.05)
    if api_task.done():
        await api_task
        return

    background_tasks: set[asyncio.Task] = set()
    try:
        games = await _create_human_games(args, background_tasks)
        if games:
            server.set_human_listener_games(games)
            links_path = Path(args.links_file).expanduser().resolve()
            links_path.parent.mkdir(parents=True, exist_ok=True)
            links_path.write_text(json.dumps(games, indent=2), encoding="utf-8")
            print(json.dumps({"games": games, "links_file": str(links_path)}, indent=2), flush=True)
        await api_task
    finally:
        api_server.should_exit = True
        for task in list(background_tasks):
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*list(background_tasks), return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record human TFM web-client decisions without controlling the game")
    parser.add_argument("--dataset", default=os.getenv("V2_TEACHER_DATASET_DIR", "/app/v2/teacher-dataset"))
    parser.add_argument("--event-dir", default=os.getenv("V2_HUMAN_LISTENER_EVENT_DIR", ""))
    parser.add_argument("--player-id", action="append", default=[], help="Player ID to record; repeat for more than one")
    parser.add_argument("--player-name", action="append", default=[], help="Player name to record; repeat for more than one")
    parser.add_argument("--token", default=os.getenv("TFM_HUMAN_LISTENER_TOKEN", ""))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--games", type=int, default=0, help="Create this many human-vs-teacher games and print player links")
    parser.add_argument("--human-name", default="You")
    parser.add_argument("--bot-count", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument("--beginner", action="store_true", help="Use beginner corporations for a Stage 0 collection run")
    parser.add_argument("--links-file", default="/app/v2/human-game-links.json")
    args = parser.parse_args()

    initialize_v2_runtime()
    if args.player_name and args.human_name not in args.player_name:
        args.player_name.append(args.human_name)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
