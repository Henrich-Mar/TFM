"""
Shared fitness, behavior metrics, and gating logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from scoring import calculate_selection_score


def snapshot_population_behavior(population: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for agent in population:
        stats = getattr(agent, "decision_stats", {}) or {}
        action_counts = dict(stats.get("action_type_counts", {}) or {})
        snapshot[agent.id] = {
            "games_played": float(getattr(agent, "games_played", 0) or 0),
            "card_play_actions": float(stats.get("card_play_actions", 0) or 0),
            "standard_project_actions": float(action_counts.get("standard_project", 0) or 0),
            "steel_spent": float(stats.get("steel_spent", 0) or 0),
            "titanium_spent": float(stats.get("titanium_spent", 0) or 0),
        }
    return snapshot


def compute_generation_behavior_metrics(
    population: Sequence[Any],
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
    payment_reject_before: int,
    payment_reject_after: int,
    current_generation: int,
    game_cluster: Any,
    tournament_results: Sequence[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    per_agent: Dict[str, Dict[str, float]] = {}
    totals = {
        "games_played": 0.0,
        "card_play_actions": 0.0,
        "standard_project_actions": 0.0,
        "steel_spent": 0.0,
        "titanium_spent": 0.0,
        "vp_terraforming": 0.0,
        "vp_milestones": 0.0,
        "vp_awards": 0.0,
        "vp_greenery": 0.0,
        "vp_city": 0.0,
        "vp_cards": 0.0,
        "vp_total": 0.0,
        "town_placements": 0.0,
        "greenery_placements": 0.0,
    }

    per_agent_endscreen: Dict[str, Dict[str, float]] = {}
    for tournament_result in (tournament_results or []):
        if not isinstance(tournament_result, dict):
            continue
        for game_result in tournament_result.get("games", []) or []:
            if not isinstance(game_result, dict):
                continue
            for player_result in game_result.get("players", []) or []:
                if not isinstance(player_result, dict):
                    continue
                agent_id = str(player_result.get("agent_id", "") or "")
                if not agent_id:
                    continue
                bucket = per_agent_endscreen.setdefault(
                    agent_id,
                    {
                        "games": 0.0,
                        "vp_terraforming": 0.0,
                        "vp_milestones": 0.0,
                        "vp_awards": 0.0,
                        "vp_greenery": 0.0,
                        "vp_city": 0.0,
                        "vp_cards": 0.0,
                        "vp_total": 0.0,
                        "town_placements": 0.0,
                        "greenery_placements": 0.0,
                    },
                )
                bucket["games"] += 1.0
                bucket["vp_terraforming"] += float(player_result.get("vp_terraforming", player_result.get("terraform_rating", 0)) or 0)
                bucket["vp_milestones"] += float(player_result.get("vp_milestones", 0) or 0)
                bucket["vp_awards"] += float(player_result.get("vp_awards", 0) or 0)
                bucket["vp_greenery"] += float(player_result.get("vp_greenery", 0) or 0)
                bucket["vp_city"] += float(player_result.get("vp_city", 0) or 0)
                bucket["vp_cards"] += float(player_result.get("vp_cards", 0) or 0)
                bucket["vp_total"] += float(player_result.get("victory_points", 0) or 0)
                bucket["town_placements"] += float(player_result.get("town_placements", 0) or 0)
                bucket["greenery_placements"] += float(player_result.get("greenery_placements", 0) or 0)

    for agent in population:
        agent_id = agent.id
        prev = before.get(agent_id, {})
        curr = after.get(agent_id, {})
        endscreen = per_agent_endscreen.get(agent_id, {})
        games_delta = max(0.0, float(curr.get("games_played", 0.0)) - float(prev.get("games_played", 0.0)))
        card_plays_delta = max(0.0, float(curr.get("card_play_actions", 0.0)) - float(prev.get("card_play_actions", 0.0)))
        standard_projects_delta = max(0.0, float(curr.get("standard_project_actions", 0.0)) - float(prev.get("standard_project_actions", 0.0)))
        steel_spent_delta = max(0.0, float(curr.get("steel_spent", 0.0)) - float(prev.get("steel_spent", 0.0)))
        titanium_spent_delta = max(0.0, float(curr.get("titanium_spent", 0.0)) - float(prev.get("titanium_spent", 0.0)))
        endscreen_games = max(0.0, float(endscreen.get("games", 0.0)))
        vp_terraforming = max(0.0, float(endscreen.get("vp_terraforming", 0.0)))
        vp_milestones = max(0.0, float(endscreen.get("vp_milestones", 0.0)))
        vp_awards = max(0.0, float(endscreen.get("vp_awards", 0.0)))
        vp_greenery = max(0.0, float(endscreen.get("vp_greenery", 0.0)))
        vp_city = max(0.0, float(endscreen.get("vp_city", 0.0)))
        vp_cards = float(endscreen.get("vp_cards", 0.0))
        vp_total = max(0.0, float(endscreen.get("vp_total", 0.0)))
        town_placements = max(0.0, float(endscreen.get("town_placements", 0.0)))
        greenery_placements = max(0.0, float(endscreen.get("greenery_placements", 0.0)))
        vp_denom = endscreen_games if endscreen_games > 0.0 else games_delta

        totals["games_played"] += games_delta
        totals["card_play_actions"] += card_plays_delta
        totals["standard_project_actions"] += standard_projects_delta
        totals["steel_spent"] += steel_spent_delta
        totals["titanium_spent"] += titanium_spent_delta
        totals["vp_terraforming"] += vp_terraforming
        totals["vp_milestones"] += vp_milestones
        totals["vp_awards"] += vp_awards
        totals["vp_greenery"] += vp_greenery
        totals["vp_city"] += vp_city
        totals["vp_cards"] += vp_cards
        totals["vp_total"] += vp_total
        totals["town_placements"] += town_placements
        totals["greenery_placements"] += greenery_placements

        action_denom = card_plays_delta + standard_projects_delta
        per_agent[agent_id] = {
            "games_played": games_delta,
            "endscreen_games": endscreen_games,
            "card_play_actions": card_plays_delta,
            "standard_project_actions": standard_projects_delta,
            "steel_spent": steel_spent_delta,
            "titanium_spent": titanium_spent_delta,
            "vp_terraforming": vp_terraforming,
            "vp_milestones": vp_milestones,
            "vp_awards": vp_awards,
            "vp_greenery": vp_greenery,
            "vp_city": vp_city,
            "vp_cards": vp_cards,
            "vp_total": vp_total,
            "town_placements": town_placements,
            "greenery_placements": greenery_placements,
            "card_plays_per_game": (card_plays_delta / games_delta) if games_delta > 0 else 0.0,
            "standard_project_ratio": (standard_projects_delta / action_denom) if action_denom > 0 else 0.0,
            "steel_spent_per_game": (steel_spent_delta / games_delta) if games_delta > 0 else 0.0,
            "titanium_spent_per_game": (titanium_spent_delta / games_delta) if games_delta > 0 else 0.0,
            "vp_terraforming_per_game": (vp_terraforming / vp_denom) if vp_denom > 0 else 0.0,
            "vp_milestones_per_game": (vp_milestones / vp_denom) if vp_denom > 0 else 0.0,
            "vp_awards_per_game": (vp_awards / vp_denom) if vp_denom > 0 else 0.0,
            "vp_greenery_per_game": (vp_greenery / vp_denom) if vp_denom > 0 else 0.0,
            "vp_city_per_game": (vp_city / vp_denom) if vp_denom > 0 else 0.0,
            "vp_cards_per_game": (vp_cards / vp_denom) if vp_denom > 0 else 0.0,
            "vp_total_per_game": (vp_total / vp_denom) if vp_denom > 0 else 0.0,
            "town_placements_per_game": (town_placements / vp_denom) if vp_denom > 0 else 0.0,
            "greenery_placements_per_game": (greenery_placements / vp_denom) if vp_denom > 0 else 0.0,
        }

    card_actions_total = totals["card_play_actions"] + totals["standard_project_actions"]
    games_total = totals["games_played"]
    payment_reject_delta = max(0, int(payment_reject_after) - int(payment_reject_before))
    generation_metrics: Dict[str, Any] = {
        "generation": int(current_generation),
        "total_games_evaluated": int(games_total),
        "card_play_actions": int(totals["card_play_actions"]),
        "standard_project_actions": int(totals["standard_project_actions"]),
        "card_plays_per_game": (float(totals["card_play_actions"]) / games_total) if games_total > 0 else 0.0,
        "standard_project_ratio": (float(totals["standard_project_actions"]) / card_actions_total) if card_actions_total > 0 else 0.0,
        "steel_spent": int(totals["steel_spent"]),
        "titanium_spent": int(totals["titanium_spent"]),
        "steel_spent_per_game": (float(totals["steel_spent"]) / games_total) if games_total > 0 else 0.0,
        "titanium_spent_per_game": (float(totals["titanium_spent"]) / games_total) if games_total > 0 else 0.0,
        "vp_terraforming": float(totals["vp_terraforming"]),
        "vp_milestones": float(totals["vp_milestones"]),
        "vp_awards": float(totals["vp_awards"]),
        "vp_greenery": float(totals["vp_greenery"]),
        "vp_city": float(totals["vp_city"]),
        "vp_cards": float(totals["vp_cards"]),
        "vp_total": float(totals["vp_total"]),
        "town_placements": float(totals["town_placements"]),
        "greenery_placements": float(totals["greenery_placements"]),
        "vp_terraforming_per_game": (float(totals["vp_terraforming"]) / games_total) if games_total > 0 else 0.0,
        "vp_milestones_per_game": (float(totals["vp_milestones"]) / games_total) if games_total > 0 else 0.0,
        "vp_awards_per_game": (float(totals["vp_awards"]) / games_total) if games_total > 0 else 0.0,
        "vp_greenery_per_game": (float(totals["vp_greenery"]) / games_total) if games_total > 0 else 0.0,
        "vp_city_per_game": (float(totals["vp_city"]) / games_total) if games_total > 0 else 0.0,
        "vp_cards_per_game": (float(totals["vp_cards"]) / games_total) if games_total > 0 else 0.0,
        "vp_total_per_game": (float(totals["vp_total"]) / games_total) if games_total > 0 else 0.0,
        "town_placements_per_game": (float(totals["town_placements"]) / games_total) if games_total > 0 else 0.0,
        "greenery_placements_per_game": (float(totals["greenery_placements"]) / games_total) if games_total > 0 else 0.0,
        "payment_reject_count": int(payment_reject_delta),
        "input_reject_count_total": int(getattr(game_cluster, "input_reject_count", 0)),
        "payment_reject_count_total": int(getattr(game_cluster, "payment_reject_count", 0)),
    }
    return generation_metrics, per_agent


@dataclass
class PromotionGateConfig:
    enabled: bool
    min_card_plays_per_game: float
    max_standard_project_ratio: float
    min_steel_spent_per_game: float
    min_titanium_spent_per_game: float
    max_payment_reject_count: int
    penalty_points: float
    global_payment_penalty_points: float


def apply_promotion_gates(
    population: Sequence[Any],
    raw_scores: List[float],
    per_agent_behavior: Dict[str, Dict[str, float]],
    generation_metrics: Dict[str, Any],
    gate_config: PromotionGateConfig,
) -> Tuple[List[float], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    gated_scores = [float(score) for score in raw_scores]
    per_agent_gate: Dict[str, Dict[str, Any]] = {}
    total_penalty = 0.0
    failed_agents = 0

    global_payment_reject = int(generation_metrics.get("payment_reject_count", 0))
    global_payment_gate_failed = global_payment_reject > int(gate_config.max_payment_reject_count)

    for idx, agent in enumerate(population):
        agent_id = agent.id
        behavior = per_agent_behavior.get(agent_id, {})
        fail_reasons: List[str] = []
        severity_penalty = 0.0
        card_plays_per_game = float(behavior.get("card_plays_per_game", 0.0))
        standard_project_ratio = float(behavior.get("standard_project_ratio", 0.0))
        steel_spent_per_game = float(behavior.get("steel_spent_per_game", 0.0))
        titanium_spent_per_game = float(behavior.get("titanium_spent_per_game", 0.0))
        steel_floor_cfg = max(0.0, float(gate_config.min_steel_spent_per_game))
        titanium_floor_cfg = max(0.0, float(gate_config.min_titanium_spent_per_game))
        resource_floor_enabled = (steel_floor_cfg > 0.0) or (titanium_floor_cfg > 0.0)
        steel_lazy = (steel_floor_cfg > 0.0) and (steel_spent_per_game < steel_floor_cfg)
        titanium_lazy = (titanium_floor_cfg > 0.0) and (titanium_spent_per_game < titanium_floor_cfg)
        lazy_resource_fail = resource_floor_enabled and steel_lazy and titanium_lazy
        if gate_config.enabled and float(behavior.get("games_played", 0.0)) > 0.0:
            if card_plays_per_game < float(gate_config.min_card_plays_per_game):
                fail_reasons.append("card_plays_per_game")
            if standard_project_ratio > float(gate_config.max_standard_project_ratio):
                fail_reasons.append("standard_project_ratio")
            # Resource guard is lazy-only: fail only when both steel and titanium
            # utilization are simultaneously below configured floors.
            if lazy_resource_fail:
                fail_reasons.append("resource_spend_lazy")

        penalty = 0.0
        if gate_config.enabled:
            penalty += float(gate_config.penalty_points) * float(len(fail_reasons))
            if float(behavior.get("games_played", 0.0)) > 0.0:
                card_floor = max(1e-6, float(gate_config.min_card_plays_per_game))
                sp_cap = max(1e-6, float(gate_config.max_standard_project_ratio))
                steel_floor = max(0.0, float(gate_config.min_steel_spent_per_game))
                titanium_floor = max(0.0, float(gate_config.min_titanium_spent_per_game))

                card_deficit = min(max(0.0, card_floor - card_plays_per_game) / card_floor, 2.0)
                sp_excess = min(max(0.0, standard_project_ratio - sp_cap) / sp_cap, 2.0)
                steel_deficit = 0.0
                if steel_floor > 0.0:
                    steel_deficit = min(max(0.0, steel_floor - steel_spent_per_game) / steel_floor, 2.0)
                titanium_deficit = 0.0
                if titanium_floor > 0.0:
                    titanium_deficit = min(max(0.0, titanium_floor - titanium_spent_per_game) / titanium_floor, 2.0)
                resource_lazy_deficit = 0.0
                if lazy_resource_fail:
                    # Combined underutilization pressure, not "spend as much as possible".
                    resource_lazy_deficit = min((steel_deficit + titanium_deficit) / 2.0, 2.0)
                severity_penalty += float(gate_config.penalty_points) * (
                    (2.0 * card_deficit)
                    + (1.5 * sp_excess)
                    + (0.8 * resource_lazy_deficit)
                )
                penalty += severity_penalty
            if global_payment_gate_failed:
                penalty += float(gate_config.global_payment_penalty_points)

        raw_score = float(raw_scores[idx]) if idx < len(raw_scores) else 0.0
        gated_score = raw_score - penalty
        gated_scores[idx] = gated_score
        total_penalty += penalty

        passed = len(fail_reasons) == 0 and not global_payment_gate_failed
        if not passed:
            failed_agents += 1
        per_agent_gate[agent_id] = {
            "passed": bool(passed),
            "fail_reasons": fail_reasons,
            "raw_fitness": raw_score,
            "gated_fitness": gated_score,
            "penalty": penalty,
            "severity_penalty": severity_penalty,
            "behavior": behavior,
        }

    summary: Dict[str, Any] = {
        "enabled": bool(gate_config.enabled),
        "thresholds": {
            "min_card_plays_per_game": float(gate_config.min_card_plays_per_game),
            "max_standard_project_ratio": float(gate_config.max_standard_project_ratio),
            "min_steel_spent_per_game": float(gate_config.min_steel_spent_per_game),
            "min_titanium_spent_per_game": float(gate_config.min_titanium_spent_per_game),
            "resource_spend_mode": "lazy_both_below_floor",
            "max_payment_reject_count": int(gate_config.max_payment_reject_count),
        },
        "penalty_points": float(gate_config.penalty_points),
        "global_payment_penalty_points": float(gate_config.global_payment_penalty_points),
        "global_payment_gate_failed": bool(global_payment_gate_failed),
        "global_payment_reject_count": int(global_payment_reject),
        "failed_agents": int(failed_agents),
        "passed_agents": int(max(0, len(population) - failed_agents)),
        "pass_rate": (float(max(0, len(population) - failed_agents)) / float(len(population))) if population else 0.0,
        "total_penalty": float(total_penalty),
    }
    return gated_scores, summary, per_agent_gate


def calculate_selection_fitness(population: Sequence[Any], tournament_results: List[Dict[str, Any]]) -> List[float]:
    agent_scores = {agent.id: 0.0 for agent in population}
    agent_games = {agent.id: 0 for agent in population}

    for tournament_result in tournament_results:
        for game_result in tournament_result.get("games", []):
            for player_result in game_result.get("players", []):
                agent_id = player_result.get("agent_id")
                if agent_id not in agent_scores:
                    continue
                total_score = calculate_selection_score(
                    rank=player_result.get("rank", 4),
                    victory_points=player_result.get("victory_points", 0),
                    completed=player_result.get("completed", False),
                )
                agent_scores[agent_id] += total_score
                agent_games[agent_id] += 1

    fitness_scores: List[float] = []
    for agent in population:
        played = int(agent_games.get(agent.id, 0))
        fitness_scores.append((agent_scores[agent.id] / played) if played > 0 else 0.0)
    return fitness_scores
