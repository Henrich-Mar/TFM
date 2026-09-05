"""Run one fixed-seed, decision-level trace for a frozen V2 checkpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import models.agent as agent_module
from game_interface import GameServerCluster
from models.agent import RLAgent
from models.decision_policy import HeuristicTeacherPolicy, RandomLegalPolicy
from tournament_manager import TournamentManager
from v2_runtime import initialize_v2_runtime


def _frozen(path: str, agent_id: str) -> RLAgent:
    agent = RLAgent(agent_id=agent_id)
    agent.load_model(path)
    agent.train_from_self_play = False
    agent.config.train_from_self_play = False
    agent.ppo_enable = False
    agent.deterministic_actions = True
    return agent


def _state_summary(player_state: Dict[str, Any]) -> Dict[str, Any]:
    player = player_state.get("thisPlayer", {}) or {}
    game = player_state.get("game", {}) or {}
    return {
        "generation": int(game.get("generation", 0) or 0),
        "phase": str(game.get("phase", "") or ""),
        "global": {
            "oxygen": game.get("oxygenLevel"), "temperature": game.get("temperature"),
            "oceans": game.get("oceans"), "venus": game.get("venusScaleLevel"),
        },
        "self": {
            "mc": player.get("megaCredits"), "tr": player.get("terraformRating"),
            "vp": (player.get("victoryPointsBreakdown", {}) or {}).get("total"),
            "production": {key: player.get(key) for key in (
                "megaCreditProduction", "steelProduction", "titaniumProduction",
                "plantProduction", "energyProduction", "heatProduction",
            )},
        },
        "opponents_visible_to_policy": [
            {
                "color": item.get("color"), "mc": item.get("megaCredits"),
                "tr": item.get("terraformRating"),
                "vp": (item.get("victoryPointsBreakdown", {}) or {}).get("total"),
                "tableau_count": len(item.get("tableau", []) or []),
            }
            for item in (player_state.get("players", []) or [])
            if isinstance(item, dict) and not _same_player(item, player)
        ],
    }


def _same_player(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Mirror StateEncoder identity fallbacks for server payload variants."""
    for key in ("id", "color", "name"):
        left = str(candidate.get(key, "") or "").strip().lower()
        right = str(current.get(key, "") or "").strip().lower()
        if left and right:
            return left == right
    return False


def _chosen_family(action_meta: Dict[str, Any], action_index: Any) -> str:
    for descriptor in action_meta.get("action_descriptors", []) or []:
        if int(descriptor.get("action_index", -1)) == int(action_index if action_index is not None else -1):
            return str(descriptor.get("family", "other") or "other")
    return "other"


async def trace(args: argparse.Namespace) -> Dict[str, Any]:
    initialize_v2_runtime()
    candidate = _frozen(args.checkpoint, "trace-candidate")
    if args.baseline == "teacher":
        opponents = [RLAgent(agent_id=f"teacher-{index}", decision_policy=HeuristicTeacherPolicy(args.seed + index, sample=False)) for index in range(3)]
    else:
        opponents = [RLAgent(agent_id=f"random-{index}", decision_policy=RandomLegalPolicy(args.seed + index)) for index in range(3)]
    for opponent in opponents:
        opponent.train_from_self_play = False
        opponent.config.train_from_self_play = False
        opponent.ppo_enable = False

    events: List[Dict[str, Any]] = []

    def capture(_self: RLAgent, **kwargs: Any) -> None:
        meta = dict(kwargs.get("action_meta") or {})
        if not meta or str(kwargs.get("send_outcome", "")) != "accepted":
            return None
        action_index = kwargs.get("action_index")
        event = {
            "generation": _state_summary(kwargs["player_state"]).get("generation", 0),
            "phase": _state_summary(kwargs["player_state"]).get("phase", ""),
            "family": _chosen_family(meta, action_index),
            "action_index": action_index,
            "action_label": meta.get("chosen_action_label"),
            "action_input": kwargs.get("action_input"),
            "legal_action_count": (meta.get("bundle_summary", {}) or {}).get("legal_action_count"),
            "state": _state_summary(kwargs["player_state"]),
            "policy_top_actions": list(meta.get("policy_top_actions", []) or [])[:3],
        }
        events.append(event)
        return None

    # Capture model rankings without creating persistent Debug Snapshot files.
    original_pending = agent_module.has_pending_capture_request
    agent_module.has_pending_capture_request = lambda **_kwargs: True
    candidate._maybe_capture_decision_snapshot = types.MethodType(capture, candidate)
    servers = [item.strip() for item in args.game_servers.split(",") if item.strip()]
    cluster = GameServerCluster(servers)
    try:
        options = Path(__file__).resolve().parents[1] / f"game_options.v2_stage{args.stage}.json"
        cluster.base_game_options = json.loads(options.read_text(encoding="utf-8"))
        lineup: List[RLAgent] = list(opponents)
        lineup.insert(args.candidate_seat, candidate)
        result = await TournamentManager(cluster)._run_single_game(
            lineup, tournament_id=f"v2_trace_{args.seed}", game_seed=args.seed,
            players_beginner=(args.stage == 0),
        )
    finally:
        agent_module.has_pending_capture_request = original_pending
        await cluster.close()

    by_family = Counter(event["family"] for event in events)
    by_generation: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for event in events:
        by_generation[str(event["generation"])][event["family"]] += 1
    candidate_result = next(row for row in result.players if row.get("agent_id") == candidate.id)
    # First decision in each generation is compact but demonstrates the policy
    # across the full arc; include all milestone/award/standard-project moves.
    selected: List[Dict[str, Any]] = []
    seen_generations = set()
    for event in events:
        keep = event["generation"] not in seen_generations or event["family"] in {
            "claim_milestone", "fund_award", "standard_project", "convert_plants", "convert_heat",
        }
        if keep and len(selected) < 48:
            selected.append(event)
            seen_generations.add(event["generation"])
    return {
        "schema_version": "tfm_rl_v2.candidate_game_trace.v1",
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "seed": args.seed,
        "stage": args.stage,
        "candidate_seat": args.candidate_seat,
        "completed": bool(result.completed),
        "candidate_result": candidate_result,
        "final_players": result.players,
        "decision_count": len(events),
        "families": dict(by_family),
        "families_by_generation": {key: dict(value) for key, value in by_generation.items()},
        "representative_decisions": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one frozen V2 candidate game")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", choices=("teacher", "random"), default="teacher")
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--game-servers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = asyncio.run(trace(args))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "completed": report["completed"], "candidate_result": report["candidate_result"],
        "decision_count": report["decision_count"], "families": report["families"],
    }, indent=2))


if __name__ == "__main__":
    main()
