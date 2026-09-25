"""Pluggable legal-action policies used by TFM RL v2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import math
import random

from .state_encoder import StateEncoder
from .action_decoder import (
    _card_keep_cost,
    _card_starting_megacredits,
    _card_tags,
    _card_vp,
    _find_prompt_card,
    _is_paid_card_purchase_prompt,
    _startup_plan_contents,
)
from .v4_flags import v4_enabled


@dataclass(frozen=True)
class ActionScore:
    action_position: int
    action_index: int
    score: float
    probability: float
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyDecision:
    chosen_action_position: int
    chosen_action_index: int
    actions: List[ActionScore]
    confidence: float
    policy_version: str
    used_fallback: bool = False
    is_forced: bool = False


class DecisionPolicy(Protocol):
    def score_actions(
        self,
        state: Dict[str, Any],
        legal_descriptors: Sequence[Dict[str, Any]],
    ) -> PolicyDecision:
        ...


def _softmax(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temp = max(1e-4, float(temperature))
    peak = max(float(item) for item in scores)
    weights = [math.exp(max(-60.0, min(60.0, (float(item) - peak) / temp))) for item in scores]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(scores)] * len(scores)
    return [float(item / total) for item in weights]


def _sample_position(probabilities: Sequence[float], rng: random.Random) -> int:
    marker = rng.random()
    running = 0.0
    for idx, probability in enumerate(probabilities):
        running += float(probability)
        if marker <= running:
            return idx
    return max(0, len(probabilities) - 1)


class RandomLegalPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(int(seed))

    def score_actions(self, state: Dict[str, Any], legal_descriptors: Sequence[Dict[str, Any]]) -> PolicyDecision:
        descriptors = list(legal_descriptors or [])
        if not descriptors:
            raise ValueError("RandomLegalPolicy requires at least one legal action")
        probability = 1.0 / len(descriptors)
        chosen = self.rng.randrange(len(descriptors))
        actions = [
            ActionScore(
                action_position=int(row.get("action_position", idx)),
                action_index=int(row.get("action_index", -1)),
                score=0.0,
                probability=probability,
                reasons=["uniform legal baseline"],
            )
            for idx, row in enumerate(descriptors)
        ]
        return PolicyDecision(
            chosen_action_position=int(actions[chosen].action_position),
            chosen_action_index=int(actions[chosen].action_index),
            actions=actions,
            confidence=0.0,
            policy_version="random-legal.v1",
            is_forced=len({int(row.get("action_index", -1)) for row in descriptors}) <= 1,
        )


class HeuristicTeacherPolicy:
    """Transparent v1 teacher. It is intentionally competent, not optimal."""

    _SUPPORTED_FAMILIES = {
        "play_card", "standard_project", "select_option", "select_space",
        "fund_award", "claim_milestone", "convert_plants", "convert_heat",
        "select_payment", "select_amount", "card_subset", "startup_plan",
        "sell_patents", "pass", "card_prompt", "other",
    }

    _MILESTONE_TRACKS = {
        "builder": ("building_tags", 8.0, 2.0),
        "gardener": ("greenery", 3.0, 1.0),
        "mayor": ("city", 3.0, 1.0),
        "planner": ("cards_in_play", 16.0, 2.0),
        "terraformer": ("tr", 35.0, 3.0),
    }
    _AWARD_TRACKS = {
        "landlord": ("tiles", 1.0),
        "banker": ("mc_production", 2.0),
        "scientist": ("science_tags", 2.0),
        "thermalist": ("heat", 2.0),
        "miner": ("miner", 2.0),
    }

    def __init__(
        self,
        seed: int = 0,
        temperature: float = 0.18,
        sample: bool = True,
        reachability: bool = True,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.temperature = max(1e-4, float(temperature))
        self.sample = bool(sample)
        self.reachability = bool(reachability)
        self.decisions = 0
        self.fallbacks = 0
        self._card_ranker = StateEncoder()
        self._active_card_rankings: Dict[str, Dict[str, Any]] = {}

    @property
    def fallback_rate(self) -> float:
        return float(self.fallbacks / self.decisions) if self.decisions else 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _track_taken(row: Dict[str, Any]) -> bool:
        return bool(
            row.get("playerName")
            or row.get("playerColor")
            or row.get("color")
            or row.get("funded_by")
        )

    def _card_record(self, name: str, card: Dict[str, Any]) -> Dict[str, Any]:
        cache = getattr(self._card_ranker, "card_metadata_by_name", None) or {}
        meta = dict(cache.get(name) or {}) if name else {}
        if not meta:
            tags = card.get("tags")
            meta = {
                "tags": tags if isinstance(tags, list) else [],
                "description": card.get("description") or "",
                "cardType": card.get("cardType") or card.get("type") or "",
            }
        elif not meta.get("description") and card.get("description"):
            meta["description"] = card.get("description")
        return meta

    def _tag_count(self, name: str, card: Dict[str, Any], needle: str) -> float:
        meta = self._card_record(name, card)
        tags = meta.get("tags")
        if isinstance(tags, list) and tags:
            return float(sum(1 for tag in tags if needle in str(tag).lower()))
        raw = card.get("tags")
        if isinstance(raw, dict):
            return float(sum(1 for key, present in raw.items() if present and needle in str(key).lower()))
        if isinstance(raw, list):
            return float(sum(1 for tag in raw if needle in str(tag).lower()))
        return 0.0

    def _card_track_delta(self, name: str, card: Dict[str, Any], *, include_planner: bool) -> Dict[str, float]:
        from .card_catalog import resolve_behavior

        meta = self._card_record(name, card)
        immediate = (resolve_behavior(meta).get("immediate") or {})
        global_params = immediate.get("global") or {}
        production = immediate.get("production") or {}
        stock = immediate.get("stock") or {}
        cities = self._safe_float(immediate.get("place_city"))
        greeneries = self._safe_float(immediate.get("place_greenery"))
        oceans = self._safe_float(global_params.get("oceans"))
        temperature = self._safe_float(global_params.get("temperature"))
        oxygen = self._safe_float(global_params.get("oxygen"))
        card_type = str(meta.get("cardType") or meta.get("type") or card.get("cardType") or card.get("type") or "")
        is_event = "event" in card_type.lower()
        return {
            "building_tags": self._tag_count(name, card, "building"),
            "science_tags": self._tag_count(name, card, "science"),
            "greenery": greeneries,
            "city": cities,
            "cards_in_play": 0.0 if (is_event or not include_planner) else 1.0,
            "tr": temperature + oxygen + oceans + greeneries,
            "tiles": cities + greeneries + oceans,
            "mc_production": self._safe_float(production.get("megacredits")),
            "heat": self._safe_float(stock.get("heat")),
            "miner": self._safe_float(stock.get("steel")) + self._safe_float(stock.get("titanium")),
        }

    @staticmethod
    def _standard_project_delta(label: str) -> Dict[str, float]:
        text = str(label or "").lower()
        delta = {
            "building_tags": 0.0, "science_tags": 0.0, "greenery": 0.0, "city": 0.0,
            "cards_in_play": 0.0, "tr": 0.0, "tiles": 0.0, "mc_production": 0.0,
            "heat": 0.0, "miner": 0.0,
        }
        if "greenery" in text:
            delta["greenery"] = 1.0
            delta["tiles"] = 1.0
            delta["tr"] = 1.0
        elif "city" in text:
            delta["city"] = 1.0
            delta["tiles"] = 1.0
        elif "asteroid" in text:
            delta["tr"] = 1.0
        elif "aquifer" in text or "ocean" in text:
            delta["tr"] = 1.0
            delta["tiles"] = 1.0
        return delta

    def _reachability_bonus(
        self,
        state: Dict[str, Any],
        delta: Dict[str, float],
    ) -> tuple[float, List[str]]:
        if not self.reachability:
            return 0.0, []
        game = state.get("game", {}) or {}
        generation = max(1.0, self._safe_float(game.get("generation", 1), 1.0))
        generations_left = max(0.0, 14.0 - generation)
        milestones = [row for row in (game.get("milestones") or []) if isinstance(row, dict)]
        awards = [row for row in (game.get("awards") or []) if isinstance(row, dict)]
        claimed = sum(1 for row in milestones if self._track_taken(row))
        funded = sum(1 for row in awards if self._track_taken(row))
        total = 0.0
        reasons: List[str] = []
        if claimed < 3:
            for milestone in milestones:
                key = str(milestone.get("name", "") or "").strip().lower()
                spec = self._MILESTONE_TRACKS.get(key)
                if spec is None or self._track_taken(milestone):
                    continue
                counter, threshold, budget = spec
                step = self._safe_float(delta.get(counter))
                if step <= 0.0:
                    continue
                own_score, _opponent = self._milestone_standing(state, milestone)
                gap = threshold - own_score
                if gap <= 0.0 or gap > generations_left * budget:
                    continue
                progress = own_score / threshold
                piece = min(0.8, 0.35 * 5.0 * progress / gap * step)
                total += piece
                reasons.append(f"reach-{key}={piece:.2f}")
        if funded < 3:
            player = state.get("thisPlayer", {}) or {}
            for award in awards:
                key = str(award.get("name", "") or "").strip().lower()
                spec = self._AWARD_TRACKS.get(key)
                if spec is None or self._track_taken(award):
                    continue
                counter, budget = spec
                step = self._safe_float(delta.get(counter))
                if step <= 0.0:
                    continue
                own_score, opponent_best, projected, _lead = self._award_standing(player, award, state)
                points = float(projected)
                if points <= 0.0 or own_score > opponent_best:
                    continue
                threshold = max(opponent_best + 1.0, 1.0)
                gap = threshold - own_score
                if gap <= 0.0 or gap > generations_left * budget:
                    continue
                progress = own_score / threshold
                piece = min(0.8, 0.35 * points * progress / gap * step)
                total += piece
                reasons.append(f"reach-{key}={piece:.2f}")
        return min(0.8, total), reasons

    def _score_card(self, state: Dict[str, Any], descriptor: Dict[str, Any]) -> tuple[float, List[str]]:
        waiting = state.get("waitingFor", {}) or {}
        player = state.get("thisPlayer", {}) or {}
        name = str(descriptor.get("card_name", "") or "")
        card = next(
            (item for item in (waiting.get("cards", []) or []) if isinstance(item, dict) and str(item.get("name", "")) == name),
            {},
        )
        cost = self._safe_float(card.get("calculatedCost", card.get("cost", 0)))
        vp = self._safe_float(card.get("victoryPoints", 0))
        generation = max(1.0, self._safe_float((state.get("game", {}) or {}).get("generation", 1), 1.0))
        mc = self._safe_float(player.get("megaCredits", 0))
        # The server card payload does not carry tags; resolve them from the
        # card metadata cache so the teacher scores the same tag information
        # the state encoder feeds to the policy network.
        tags_map = self._card_ranker._get_card_tags(name, fallback=card.get("tags", {}))
        tags = {str(tag).lower() for tag, present in tags_map.items() if present}
        affordability = 1.1 if cost <= mc else -1.5 - min(1.0, (cost - mc) / 15.0)
        phase = max(0.0, min(1.0, generation / 14.0))
        engine_tags = len(tags.intersection({"science", "earth", "building", "space", "plant"}))
        score = affordability + (0.45 * vp * (0.4 + phase)) + (0.12 * engine_tags * (1.2 - phase))
        reused = self._active_card_rankings.get(name, {})
        if reused:
            score += 0.6 * self._safe_float(reused.get("selection_score", 0.0))
        if cost <= 8:
            score += 0.18
        reasons = [f"affordability={affordability:.2f}", f"vp={vp:.1f}", f"cost={cost:.0f}"]
        if reused:
            reasons.extend([
                f"existing-card-score={self._safe_float(reused.get('selection_score', 0.0)):.2f}",
                f"requirement-readiness={self._safe_float(reused.get('readiness_score', 1.0)):.2f}",
            ])
        bonus, bonus_reasons = self._reachability_bonus(
            state,
            self._card_track_delta(name, card, include_planner=True),
        )
        return score + bonus, reasons + bonus_reasons

    def _score_card_subset(
        self,
        state: Dict[str, Any],
        descriptor: Dict[str, Any],
    ) -> tuple[float, List[str], bool]:
        """Score a draft or purchase by card quality, with no catalog-position bonus."""
        from scoring import _card_quality

        waiting = state.get("waitingFor", {}) or {}
        player = state.get("thisPlayer", {}) or {}
        decoded = descriptor.get("decoded_action", {}) or {}
        names = []
        if isinstance(decoded, dict):
            for item in decoded.get("cards", []) or []:
                if isinstance(item, str) and item.strip():
                    names.append(item.strip())
                elif isinstance(item, dict) and str(item.get("name", "") or "").strip():
                    names.append(str(item.get("name")).strip())
        prompt_cards = [card for card in (waiting.get("cards", []) or []) if isinstance(card, dict)]
        by_name = {str(card.get("name", "") or "").strip(): card for card in prompt_cards}
        selected = []
        for name in names:
            card = by_name.get(name) or _find_prompt_card(waiting, name)
            if not card:
                raise ValueError(f"draft teacher could not resolve selected card {name!r}")
            selected.append(card)
        paid = _is_paid_card_purchase_prompt(waiting)
        card_cost = self._safe_float(player.get("cardCost", 3), 3.0) if paid else 0.0
        threshold = 0.0 if not paid else min(0.90, 0.60 + 0.05 * card_cost)
        score = 0.0
        for card in selected:
            score += float(_card_quality(card, player)) - threshold
        fee = card_cost * float(len(selected))
        remaining = max(0.0, self._safe_float(player.get("megaCredits", 0)) - fee)
        score -= 0.02 * max(0.0, 14.0 - remaining)
        reasons = [
            f"cards={len(selected)}",
            f"threshold={threshold:.2f}",
            f"remaining-mc={remaining:.0f}",
            "card-quality",
        ]
        for card in selected:
            card_name = str(card.get("name", "") or "")
            bonus, bonus_reasons = self._reachability_bonus(
                state,
                self._card_track_delta(card_name, card, include_planner=False),
            )
            score += bonus
            reasons.extend(bonus_reasons)
        return score, reasons, False

    def _score_startup_plan(
        self,
        state: Dict[str, Any],
        descriptor: Dict[str, Any],
    ) -> tuple[float, List[str], bool]:
        waiting = state.get("waitingFor", {}) or {}
        decoded = descriptor.get("decoded_action", {}) or {}
        contents = _startup_plan_contents(decoded if isinstance(decoded, dict) else {}, waiting)
        corp_name = str(contents.get("corp", "") or "").strip()
        projects = [str(name).strip() for name in (contents.get("project") or []) if str(name).strip()]
        corp_card = _find_prompt_card(waiting, corp_name) if corp_name else {}
        if not corp_card and corp_name:
            corp_card = {"name": corp_name}
        keep_cost = float(_card_keep_cost(corp_card, default=3)) if corp_card else 3.0
        start_mc = float(_card_starting_megacredits(corp_card, default=40)) if corp_card else 40.0
        keep_spend = keep_cost * float(len(projects))
        remaining = max(0.0, start_mc - keep_spend)
        corp_tags = {
            str(tag).lower()
            for tag, present in _card_tags(corp_card).items()
            if present
        } if corp_card else set()

        keep_quality = 0.0
        synergy = 0.0
        for name in projects:
            card = _find_prompt_card(waiting, name) or {"name": name}
            cost = self._safe_float(card.get("calculatedCost", card.get("cost", 0)))
            vp = float(_card_vp(card))
            tags = {str(tag).lower() for tag, present in _card_tags(card).items() if present}
            keep_quality += (0.35 * vp) + max(0.0, 14.0 - cost) * 0.04
            if cost > 14.0:
                keep_quality -= (cost - 14.0) * 0.03
            synergy += 0.18 * float(len(tags.intersection(corp_tags)))

        spend_penalty = 0.045 * keep_spend
        cash_bonus = min(1.2, remaining / 40.0)
        rank_bonus = max(0.0, 0.4 - (0.01 * self._safe_float(descriptor.get("action_position", 0))))
        score = 0.6 + keep_quality + synergy + cash_bonus - spend_penalty + rank_bonus
        reasons = [
            f"corp={corp_name or '?'}",
            f"keeps={len(projects)}",
            f"keep-spend={keep_spend:.0f}",
            f"remaining-mc={remaining:.0f}",
            f"synergy={synergy:.2f}",
        ]
        return score, reasons, False

    def _find_award(self, game: Dict[str, Any], award_name: str) -> Dict[str, Any]:
        target = str(award_name or "").strip().lower()
        if not target:
            return {}
        for award in game.get("awards", []) or []:
            if not isinstance(award, dict):
                continue
            name = str(award.get("name", award.get("title", "")) or "").strip().lower()
            if name == target:
                return award
        return {}

    def _find_milestone(self, game: Dict[str, Any], milestone_name: str) -> Dict[str, Any]:
        target = str(milestone_name or "").strip().lower()
        if not target:
            return {}
        for milestone in game.get("milestones", []) or []:
            if not isinstance(milestone, dict):
                continue
            name = str(milestone.get("name", milestone.get("title", "")) or "").strip().lower()
            if name == target:
                return milestone
        return {}

    @staticmethod
    def _row_identity(row: Dict[str, Any]) -> tuple[str, str]:
        color = str(row.get("playerColor", row.get("color", "")) or "").strip().lower()
        name = str(row.get("playerName", row.get("name", "")) or "").strip().lower()
        return color, name

    def _milestone_standing(
        self,
        state: Dict[str, Any],
        milestone: Dict[str, Any],
    ) -> tuple[float, float]:
        player = state.get("thisPlayer", {}) or {}
        own_color = str(player.get("color", "") or "").strip().lower()
        own_name = str(player.get("name", "") or "").strip().lower()
        players = [row for row in (state.get("players", []) or []) if isinstance(row, dict)]
        if not players:
            players = [row for row in ((state.get("game", {}) or {}).get("players", []) or []) if isinstance(row, dict)]

        own_score = 0.0
        opponent_best = 0.0
        for index, row in enumerate(milestone.get("scores", []) or []):
            if not isinstance(row, dict):
                continue
            color, name = self._row_identity(row)
            if not color and not name and index < len(players):
                color, name = self._row_identity(players[index])
            score = self._safe_float(row.get("playerScore", row.get("score", 0)))
            if (own_color and color == own_color) or (own_name and name == own_name):
                own_score = score
            else:
                opponent_best = max(opponent_best, score)
        return own_score, opponent_best

    def _score_claim_milestone(
        self,
        state: Dict[str, Any],
        descriptor: Dict[str, Any],
    ) -> tuple[float, List[str], bool]:
        game = state.get("game", {}) or {}
        player = state.get("thisPlayer", {}) or {}
        name = str(descriptor.get("milestone_name", "") or descriptor.get("label", "") or "").strip()
        if not name or name.lower() in {"claim milestone", "claim a milestone", "milestone"}:
            raise ValueError("milestone teacher requires an exact milestone identity")
        milestone = self._find_milestone(game, name)
        if not milestone:
            raise ValueError(f"milestone teacher could not resolve {name!r}")
        if milestone.get("playerName") or milestone.get("playerColor") or milestone.get("color"):
            return -3.0, [f"milestone={name}", "already claimed"], False

        own_score, opponent_best = self._milestone_standing(state, milestone)
        generation = max(1.0, self._safe_float(game.get("generation", 1), 1.0))
        phase = max(0.0, min(generation / 14.0, 1.0))
        deny_risk = max(0.0, min(opponent_best / 3.0, 1.0))
        claim_surplus = max(0.0, min((own_score - 3.0) / 3.0, 1.0))
        waiting = state.get("waitingFor", {}) or {}
        effective_cost = self._safe_float(
            descriptor.get("milestone_cost", waiting.get("cost", player.get("milestoneCost", 8.0))),
            8.0,
        )
        mc = self._safe_float(player.get("megaCredits", 0.0))
        score = 2.9 + (0.45 * phase) + (0.55 * deny_risk) + (0.15 * claim_surplus) - (0.05 * effective_cost)
        if mc < effective_cost:
            score -= 2.0 + min(1.0, (effective_cost - mc) / 8.0)
        reasons = [
            f"milestone={name}",
            f"own={own_score:.1f}",
            f"opp={opponent_best:.1f}",
            f"deny={deny_risk:.2f}",
            f"cost={effective_cost:.0f}",
        ]
        return score, reasons, False

    def _award_standing(
        self,
        player: Dict[str, Any],
        award: Dict[str, Any],
        state: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, float, float, float]:
        """Return (own_score, opp_best, projected_vp, lead_gap).

        Live TM payloads use ``{color, score}``. Older/test fixtures may use
        ``playerColor`` / ``playerName`` / ``playerScore``. When identity is
        missing entirely, fall back to ``players`` order.
        """
        own_color = str(player.get("color", "") or "").strip().lower()
        own_name = str(player.get("name", "") or "").strip().lower()
        players = []
        if isinstance(state, dict):
            players = [row for row in (state.get("players", []) or []) if isinstance(row, dict)]
            if not players:
                game = state.get("game", {}) or {}
                if isinstance(game, dict):
                    players = [row for row in (game.get("players", []) or []) if isinstance(row, dict)]

        rows: List[tuple[str, str, float]] = []
        raw_scores = [row for row in (award.get("scores", []) or []) if isinstance(row, dict)]
        for idx, row in enumerate(raw_scores):
            color = str(
                row.get("playerColor", row.get("color", "")) or ""
            ).strip().lower()
            name = str(row.get("playerName", row.get("name", "")) or "").strip().lower()
            score = self._safe_float(row.get("playerScore", row.get("score", 0)))
            if not color and not name and idx < len(players):
                color = str(players[idx].get("color", "") or "").strip().lower()
                name = str(players[idx].get("name", "") or "").strip().lower()
            if not color and not name:
                continue
            rows.append((color, name, score))
        if not rows:
            return 0.0, 0.0, 0.0, 0.0

        own_score = 0.0
        opp_best = 0.0
        for color, name, score in rows:
            is_own = (own_color and color == own_color) or (own_name and name == own_name)
            if is_own:
                own_score = score
            else:
                opp_best = max(opp_best, score)

        ordered = sorted((score for _, _, score in rows), reverse=True)
        top = ordered[0] if ordered else 0.0
        top_count = sum(1 for score in ordered if score == top)
        second = next((score for score in ordered if score < top), 0.0)

        projected = 0.0
        if top > 0.0 and own_score == top:
            projected = 5.0
        elif second > 0.0 and own_score == second and top_count == 1:
            projected = 2.0

        if own_score == top and top_count == 1:
            lead_gap = own_score - second
        else:
            lead_gap = own_score - opp_best
        return own_score, opp_best, projected, lead_gap

    def _estimate_award_cost(self, game: Dict[str, Any]) -> float:
        funded = 0
        for award in game.get("awards", []) or []:
            if not isinstance(award, dict):
                continue
            # Live model uses ``color`` for the funder; fixtures may use playerColor.
            if (
                award.get("playerName")
                or award.get("playerColor")
                or award.get("color")
                or award.get("funded_by")
            ):
                funded += 1
        return float([8.0, 14.0, 20.0][min(funded, 2)])

    def _score_fund_award(
        self,
        state: Dict[str, Any],
        descriptor: Dict[str, Any],
    ) -> tuple[float, List[str], bool]:
        player = state.get("thisPlayer", {}) or {}
        game = state.get("game", {}) or {}
        generation = max(1.0, self._safe_float(game.get("generation", 1), 1.0))
        mc = self._safe_float(player.get("megaCredits", 0))
        award_name = str(descriptor.get("award_name", "") or descriptor.get("label", "") or "").strip()
        award = self._find_award(game, award_name)
        if (
            award.get("playerName")
            or award.get("playerColor")
            or award.get("color")
            or award.get("funded_by")
        ):
            return -3.0, [f"award={award_name or '?'}", "already funded"], False

        own_score, opp_best, projected_vp, lead_gap = self._award_standing(player, award, state)
        cost = self._estimate_award_cost(game)
        phase = min(1.0, generation / 12.0)
        cost_vp = cost / 5.0
        # Mild early tax: first award at 8 MC can still be correct with a real lead.
        commitment_tax = cost_vp * max(0.0, 1.0 - phase) * 0.35
        if projected_vp >= 5.0:
            confidence = 0.70 + min(0.25, max(0.0, lead_gap) / 10.0)
        elif projected_vp >= 2.0:
            confidence = 0.40 + min(0.20, max(0.0, lead_gap + 1.0) / 10.0)
        else:
            confidence = 0.10
        expected_vp = projected_vp * confidence
        expected_net = expected_vp - cost_vp - commitment_tax
        affordability = 0.15 if cost <= mc else -1.6 - min(1.0, (cost - mc) / 12.0)
        # Base timing keeps mid/late funding competitive with ordinary card plays.
        timing = 0.35 + (1.10 * phase)
        standing = (
            (0.70 * projected_vp)
            + (0.22 * lead_gap)
            + (0.06 * own_score)
            - (0.20 * max(0.0, opp_best - own_score))
        )
        score = timing + affordability + standing + (1.10 * expected_net)
        if projected_vp <= 0.0:
            score -= 2.2
        elif projected_vp >= 5.0 and lead_gap >= 1.0:
            score += 0.85
        if lead_gap <= -2.0:
            score -= 1.1
        reasons = [
            f"award={award_name or '?'}",
            f"own={own_score:.0f}",
            f"opp={opp_best:.0f}",
            f"projected-vp={projected_vp:.0f}",
            f"cost={cost:.0f}",
            f"expected-net={expected_net:.2f}",
        ]
        return score, reasons, False

    def _score_descriptor(self, state: Dict[str, Any], descriptor: Dict[str, Any]) -> tuple[float, List[str], bool]:
        family = str(descriptor.get("family", "other") or "other")
        label = str(descriptor.get("label", "") or "").lower()
        player = state.get("thisPlayer", {}) or {}
        game = state.get("game", {}) or {}
        generation = max(1.0, self._safe_float(game.get("generation", 1), 1.0))
        mc = self._safe_float(player.get("megaCredits", 0))
        plants = self._safe_float(player.get("plants", 0))
        heat = self._safe_float(player.get("heat", 0))

        # Nested OR menus sometimes surface convert actions as select_option.
        if family == "select_option":
            if "convert" in label and "heat" in label:
                family = "convert_heat"
            elif "convert" in label and "plant" in label:
                family = "convert_plants"

        if family == "play_card":
            score, reasons = self._score_card(state, descriptor)
            return score + 0.45, reasons + ["project-card tempo"], False
        if family == "startup_plan":
            return self._score_startup_plan(state, descriptor)
        if family in {"card_subset", "card_prompt"}:
            if v4_enabled():
                return self._score_card_subset(state, descriptor)
            decoded = descriptor.get("decoded_action", {}) or {}
            card_count = len(decoded.get("cards", []) or []) if isinstance(decoded, dict) else 0
            existing_rank_bonus = max(0.0, 1.5 - (0.02 * self._safe_float(descriptor.get("action_position", 0))))
            return 0.8 + 0.08 * card_count + existing_rank_bonus, ["existing ranked startup/subset heuristic"], False
        if family == "claim_milestone":
            return self._score_claim_milestone(state, descriptor)
        if family == "fund_award":
            return self._score_fund_award(state, descriptor)
        if family == "convert_plants":
            oxygen = self._safe_float(game.get("oxygenLevel", game.get("oxygen", 0)))
            return (2.3 if plants >= 8 and oxygen < 14 else -2.0), ["plant threshold", "oxygen capacity"], False
        if family == "convert_heat":
            temperature = self._safe_float(game.get("temperature", -30), -30)
            return (2.1 if heat >= 8 and temperature < 8 else -2.0), ["heat threshold", "temperature capacity"], False
        if family == "select_payment":
            decoded = descriptor.get("decoded_action", {}) or {}
            payment = decoded.get("payment", {}) if isinstance(decoded, dict) else {}
            mc_spend = self._safe_float(payment.get("megaCredits", 0)) if isinstance(payment, dict) else 0.0
            metal_spend = 2.0 * self._safe_float(payment.get("steel", 0)) + 3.0 * self._safe_float(payment.get("titanium", 0)) if isinstance(payment, dict) else 0.0
            return 1.0 + 0.04 * metal_spend - 0.015 * mc_spend, ["preserve flexible MC", "use matching metals"], False
        if family == "select_space":
            space = descriptor.get("space_features", {}) or {}
            if bool(space.get("board_context_available", False)):
                total = self._safe_float(space.get("total_value", 0.0))
                self_value = self._safe_float(space.get("self_value", 0.0))
                deny_value = self._safe_float(space.get("deny_value", 0.0))
                risk_value = self._safe_float(space.get("risk_value", 0.0))
                bonus_value = self._safe_float(space.get("bonus_value", 0.0))
                score = (0.4 + (2.4 * total) + (0.45 * self_value) + (0.35 * deny_value) - (0.45 * risk_value))
                reasons = [
                    f"space={space.get('space_id', '?')}",
                    f"placement-value={total:.2f}",
                    f"bonus={bonus_value:.1f}",
                    f"deny={deny_value:.2f}",
                    f"risk={risk_value:.2f}",
                ]
                return score, reasons, False
            return 0.2, ["space board context unavailable"], False
        if family == "standard_project":
            phase = min(1.0, generation / 14.0)
            score = -0.15 + 0.75 * phase
            if "greenery" in label and plants >= 4:
                score += 0.45
            if "power" in label and self._safe_float(player.get("energyProduction", 0)) <= 0:
                score += 0.25
            bonus, bonus_reasons = self._reachability_bonus(state, self._standard_project_delta(label))
            return score + bonus, ["standard-project opportunity cost", *bonus_reasons], False
        if family == "sell_patents":
            return -1.3 if mc > 3 else -0.25, ["avoid destroying option value"], False
        if family == "pass":
            playable = bool((state.get("waitingFor", {}) or {}).get("cards", []))
            return (-1.4 if playable else 0.05), ["pass only without valuable legal play"], False
        if family in {"select_option", "select_amount", "other"}:
            keyword_score = 0.0
            if "draw" in label or "production" in label or "increase" in label:
                keyword_score += 0.5
            if "discard" in label or "decrease" in label:
                keyword_score -= 0.4
            return keyword_score, ["generic prompt semantics"], family == "other"
        return 0.0, ["deterministic unsupported-family fallback"], True

    def score_actions(self, state: Dict[str, Any], legal_descriptors: Sequence[Dict[str, Any]]) -> PolicyDecision:
        descriptors = list(legal_descriptors or [])
        if not descriptors:
            raise ValueError("HeuristicTeacherPolicy requires at least one legal action")
        try:
            self._active_card_rankings = {
                str(row.get("name", "") or ""): row
                for row in self._card_ranker.build_prompt_card_rankings(state)
            }
        except Exception:
            self._active_card_rankings = {}
        scored = [self._score_descriptor(state, row) for row in descriptors]
        scores = [item[0] for item in scored]
        probabilities = _softmax(scores, self.temperature)
        fallback = all(bool(item[2]) for item in scored)
        self.decisions += 1
        if fallback:
            self.fallbacks += 1
        if self.sample and not fallback:
            chosen = _sample_position(probabilities, self.rng)
        else:
            chosen = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else 4.0
        is_forced = len({int(row.get("action_index", -1)) for row in descriptors}) <= 1
        # A one-action mask establishes legality, not strategic certainty.
        confidence = 0.0 if is_forced else max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, margin))))
        actions = [
            ActionScore(
                action_position=int(row.get("action_position", idx)),
                action_index=int(row.get("action_index", -1)),
                score=float(scores[idx]),
                probability=float(probabilities[idx]),
                reasons=list(scored[idx][1]),
            )
            for idx, row in enumerate(descriptors)
        ]
        return PolicyDecision(
            chosen_action_position=int(actions[chosen].action_position),
            chosen_action_index=int(actions[chosen].action_index),
            actions=actions,
            confidence=float(confidence),
            policy_version=(
                "heuristic-teacher.v6" if self.reachability and v4_enabled()
                else "heuristic-teacher.v5" if v4_enabled()
                else "heuristic-teacher.v1"
            ),
            used_fallback=bool(fallback),
            is_forced=bool(is_forced),
        )


class NeuralDecisionPolicy:
    """Adapter for callers that already produce one logit per legal action."""

    def __init__(self, scorer: Callable[[Dict[str, Any], Sequence[Dict[str, Any]]], Sequence[float]], version: str) -> None:
        self.scorer = scorer
        self.version = str(version)

    def score_actions(self, state: Dict[str, Any], legal_descriptors: Sequence[Dict[str, Any]]) -> PolicyDecision:
        descriptors = list(legal_descriptors or [])
        logits = [float(item) for item in self.scorer(state, descriptors)]
        if len(logits) != len(descriptors) or not descriptors:
            raise ValueError("NeuralDecisionPolicy scorer returned an invalid action shape")
        probabilities = _softmax(logits, 1.0)
        chosen = max(range(len(logits)), key=logits.__getitem__)
        actions = [
            ActionScore(int(row.get("action_position", idx)), int(row.get("action_index", -1)), logits[idx], probabilities[idx], ["neural logit"])
            for idx, row in enumerate(descriptors)
        ]
        ordered = sorted(probabilities, reverse=True)
        is_forced = len({int(row.get("action_index", -1)) for row in descriptors}) <= 1
        confidence = (ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
        return PolicyDecision(actions[chosen].action_position, actions[chosen].action_index, actions, confidence, self.version, is_forced=is_forced)
