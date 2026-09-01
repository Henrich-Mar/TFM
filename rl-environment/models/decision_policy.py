"""Pluggable legal-action policies used by TFM RL v2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import math
import random

from .state_encoder import StateEncoder


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
        )


class HeuristicTeacherPolicy:
    """Transparent v1 teacher. It is intentionally competent, not optimal."""

    _SUPPORTED_FAMILIES = {
        "play_card", "standard_project", "select_option", "select_space",
        "fund_award", "claim_milestone", "convert_plants", "convert_heat",
        "select_payment", "select_amount", "card_subset", "startup_plan",
        "sell_patents", "pass", "card_prompt", "other",
    }

    def __init__(self, seed: int = 0, temperature: float = 0.18, sample: bool = True) -> None:
        self.rng = random.Random(int(seed))
        self.temperature = max(1e-4, float(temperature))
        self.sample = bool(sample)
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
        tags_raw = card.get("tags", []) or []
        if isinstance(tags_raw, dict):
            tags = {str(key).lower() for key, value in tags_raw.items() if value}
        else:
            tags = {str(item).lower() for item in tags_raw}
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
        return score, reasons

    def _score_descriptor(self, state: Dict[str, Any], descriptor: Dict[str, Any]) -> tuple[float, List[str], bool]:
        family = str(descriptor.get("family", "other") or "other")
        label = str(descriptor.get("label", "") or "").lower()
        player = state.get("thisPlayer", {}) or {}
        game = state.get("game", {}) or {}
        generation = max(1.0, self._safe_float(game.get("generation", 1), 1.0))
        mc = self._safe_float(player.get("megaCredits", 0))
        plants = self._safe_float(player.get("plants", 0))
        heat = self._safe_float(player.get("heat", 0))

        if family == "play_card":
            score, reasons = self._score_card(state, descriptor)
            return score + 0.45, reasons + ["project-card tempo"], False
        if family in {"startup_plan", "card_subset", "card_prompt"}:
            decoded = descriptor.get("decoded_action", {}) or {}
            card_count = len(decoded.get("cards", []) or []) if isinstance(decoded, dict) else 0
            existing_rank_bonus = max(0.0, 1.5 - (0.02 * self._safe_float(descriptor.get("action_position", 0))))
            return 0.8 + 0.08 * card_count + existing_rank_bonus, ["existing ranked startup/subset heuristic"], False
        if family == "claim_milestone":
            return 3.6, ["secure five VP before opponents"], False
        if family == "fund_award":
            phase = min(1.0, generation / 12.0)
            return -0.3 + 1.8 * phase - min(0.8, max(0.0, 14.0 - mc) / 20.0), ["award timing", "cash opportunity cost"], False
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
            decoded = descriptor.get("decoded_action", {}) or {}
            bonus_count = len(decoded.get("bonuses", []) or []) if isinstance(decoded, dict) else 0
            adjacency = 0.35 if any(token in label for token in ("city", "greenery", "ocean")) else 0.0
            return 0.9 + 0.18 * bonus_count + adjacency, ["placement bonuses", "adjacency"], False
        if family == "standard_project":
            phase = min(1.0, generation / 14.0)
            score = -0.15 + 0.75 * phase
            if "greenery" in label and plants >= 4:
                score += 0.45
            if "power" in label and self._safe_float(player.get("energyProduction", 0)) <= 0:
                score += 0.25
            return score, ["standard-project opportunity cost"], False
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
        confidence = max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, margin))))
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
            policy_version="heuristic-teacher.v1",
            used_fallback=bool(fallback),
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
        confidence = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        return PolicyDecision(actions[chosen].action_position, actions[chosen].action_index, actions, confidence, self.version)
