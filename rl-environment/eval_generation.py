"""
Evaluate a saved generation from rl-models by running quick tournaments.

Usage (PowerShell/CMD):
  python -m rl-environment.eval_generation --generation 16 --agent 0 --games 3 --servers localhost:8080

Notes:
- Expects saved checkpoints under ../rl-models/generation_{N}/agent_{i}_fitness_*.pth
- Starts minimal, fast games (disabled expansions, no drafts) to get quick signal.
"""
import argparse
import asyncio
import glob
import os
from typing import List

from game_interface import GameServerCluster
from models.agent import RLAgent
from tournament_manager import TournamentManager


def default_models_root() -> str:
    env_path = os.getenv("RL_MODELS_DIR")
    if env_path:
        return env_path
    base_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
    candidates = [
        os.path.join(parent_dir, "rl-models"),
        os.path.join(base_dir, "rl-models"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    if os.path.basename(base_dir).lower() == "rl-environment":
        return os.path.join(parent_dir, "rl-models")
    return candidates[0]


def find_checkpoint(models_root: str, generation: int, agent_index: int) -> str:
    gen_dir = os.path.join(models_root, f"generation_{generation}")
    pattern = os.path.join(gen_dir, f"agent_{agent_index}_fitness_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for generation {generation}, agent {agent_index} under {gen_dir}")
    # Pick best (highest fitness) by filename sort works due to fixed-width formatting; safest is to parse number
    def fitness_from_name(p: str) -> float:
        base = os.path.basename(p)
        try:
            f = base.split("_fitness_")[-1].replace(".pth", "")
            return float(f)
        except Exception:
            return -1.0
    matches.sort(key=fitness_from_name, reverse=True)
    return matches[0]


async def evaluate_agents(server_addrs: List[str], agents: List[RLAgent], games: int) -> None:
    cluster = GameServerCluster(server_addrs)
    try:
        await cluster.health_check()
        tm = TournamentManager(cluster)

        # Create a small bracket with these agents; duplicate if fewer than 4
        eval_agents = agents[:]
        while len(eval_agents) < 4:
            eval_agents.append(agents[0])

        from collections import defaultdict
        totals = defaultdict(lambda: {"vp": 0, "games": 0, "wins": 0})
        total_completed = 0
        total_planned = 0
        for _ in range(max(1, games)):
            brackets = tm.create_tournaments(eval_agents, tournament_size=4, games_per_evaluation=1)
            result = await tm.run_tournament(brackets[0])
            total_completed += result.get('completed_games', 0)
            total_planned += result.get('total_planned_games', 0)
            for g in result.get("games", []):
                for p in g.get("players", []):
                    aid = p.get("agent_id")
                    totals[aid]["vp"] += int(p.get("victory_points", 0))
                    totals[aid]["games"] += 1
                    if int(p.get("rank", 4)) == 1:
                        totals[aid]["wins"] += 1

        # Summarize
        print("Evaluation completed:")
        print(f"  Games: {total_completed}/{total_planned}")
        for aid, t in totals.items():
            avg_vp = t["vp"] / max(1, t["games"])
            wr = t["wins"] / max(1, t["games"])
            print(f"  Agent {aid[:8]}: games={t['games']}, avg_vp={avg_vp:.1f}, win_rate={wr*100:.1f}%")
    finally:
        await cluster.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--agent", type=int, default=0, help="agent index within saved top set (default: 0)")
    parser.add_argument("--games", type=int, default=3, help="games per matchup")
    parser.add_argument("--models", type=str, default=default_models_root())
    parser.add_argument("--servers", type=str, default="localhost:8080", help="comma-separated game servers")
    args = parser.parse_args()

    ckpt = find_checkpoint(args.models, args.generation, args.agent)
    print(f"Loading checkpoint: {ckpt}")

    # Create 4 copies of the loaded agent for self-play evaluation
    base_agent = RLAgent()
    base_agent.load_model(ckpt)
    agents = [base_agent]
    for _ in range(3):
        clone = RLAgent(base_agent.config)
        clone.network.load_state_dict(base_agent.network.state_dict())
        agents.append(clone)

    server_addrs = [s.strip() for s in args.servers.split(",") if s.strip()]
    asyncio.run(evaluate_agents(server_addrs, agents, args.games))


if __name__ == "__main__":
    main()


