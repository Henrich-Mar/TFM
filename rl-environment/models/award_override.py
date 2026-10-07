"""Rule-based award funding override for evaluation-only hybrid benchmarks.

This answers one question before any more training effort goes into awards:
would the current policy get stronger if it funded awards under a simple,
explicit rule? A frozen checkpoint plays normally, except that whenever a
fund-award option is legal and the rule fires, the rule's award is taken
instead of the policy's choice. Comparing that hybrid against the unmodified
checkpoint on the same seeds isolates the value of funding.

The override is never used in self-play training: it changes the behavior
policy, so its actions must not enter the strict on-policy PPO buffer.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .action_decoder import _find_prompt_card, _startup_plan_contents
from .award_tracks import card_features, plan_totals, track_for, track_value
from .decision_policy import HeuristicTeacherPolicy


@dataclass
class AwardOverrideRule:
    """Fund an award only when leading it under explicit thresholds."""

    #: Minimum lead over the best opponent. 0 allows a tie for first place.
    min_lead: float = 1.0
    #: Earliest generation in which the rule may fire.
    min_generation: int = 1
    #: Highest funding cost the rule accepts (8 = first award only, 14, 20).
    max_cost: float = 8.0
    #: MC that must remain after paying for the award.
    reserve_mc: float = 0.0
    #: Upper bound on awards funded by this seat in one game.
    max_own_awards: int = 1
    #: Weight of award-track gains still sitting in hand. A human funds on the
    #: lead they know their hand will extend, not only on the current one;
    #: 0 keeps the rule blind to the hand.
    hand_weight: float = 0.0
    #: Startup steering. At the initial-cards prompt, pick among the policy's
    #: ``init_top_k`` most likely corp/keep plans the one maximising
    #: ``log p + init_focus_weight * focus``, where focus measures how much the
    #: plan commits to one award/milestone on this map. 0 disables steering.
    init_focus_weight: float = 0.0
    init_top_k: int = 8

    # Telemetry; shared across concurrent games, so guarded by a lock.
    offered: int = field(default=0, compare=False)
    fired: int = field(default=0, compare=False)
    policy_agreed: int = field(default=0, compare=False)
    startup_decisions: int = field(default=0, compare=False)
    startup_overridden: int = field(default=0, compare=False)
    _card_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict, compare=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    _teacher: Optional[HeuristicTeacherPolicy] = field(default=None, compare=False, repr=False)

    _award_standing = HeuristicTeacherPolicy._award_standing
    _estimate_award_cost = HeuristicTeacherPolicy._estimate_award_cost
    _find_award = HeuristicTeacherPolicy._find_award
    _safe_float = staticmethod(HeuristicTeacherPolicy._safe_float)
    _track_taken = staticmethod(HeuristicTeacherPolicy._track_taken)

    @classmethod
    def from_spec(cls, spec: str) -> "AwardOverrideRule":
        """Parse ``"min_lead=1,min_generation=3,max_cost=8"`` style specs."""
        rule = cls()
        casts = {
            "min_lead": float,
            "min_generation": int,
            "max_cost": float,
            "reserve_mc": float,
            "max_own_awards": int,
            "hand_weight": float,
            "init_focus_weight": float,
            "init_top_k": int,
        }
        for chunk in str(spec or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            key, sep, value = chunk.partition("=")
            key = key.strip()
            if not sep or key not in casts:
                raise ValueError(f"invalid award override setting: {chunk!r}")
            setattr(rule, key, casts[key](value.strip()))
        return rule

    def config(self) -> Dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "min_lead", "min_generation", "max_cost", "reserve_mc", "max_own_awards",
                "hand_weight", "init_focus_weight", "init_top_k",
            )
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": self.config(),
                "decisions_with_fund_option": self.offered,
                "fired": self.fired,
                "fired_policy_already_chose_it": self.policy_agreed,
                "startup_decisions": self.startup_decisions,
                "startup_overridden": self.startup_overridden,
            }

    def _own_funded(self, game: Dict[str, Any], player: Dict[str, Any]) -> int:
        own = {
            str(player.get("color", "") or "").strip().lower(),
            str(player.get("name", "") or "").strip().lower(),
        } - {""}
        count = 0
        for award in game.get("awards", []) or []:
            if not isinstance(award, dict):
                continue
            funder = {
                str(award.get(key, "") or "").strip().lower()
                for key in ("playerColor", "color", "playerName", "funded_by")
            } - {""}
            if funder & own:
                count += 1
        return count

    def _helper(self) -> HeuristicTeacherPolicy:
        with self._lock:
            if self._teacher is None:
                self._teacher = HeuristicTeacherPolicy(sample=False)
            return self._teacher

    def _card_features(self, name: str, card: Dict[str, Any]) -> Dict[str, float]:
        """Track features of one card from its metadata (see models/award_tracks.py)."""
        cached = self._card_vectors.get(name)
        if cached is not None:
            return cached
        try:
            features = card_features(self._helper()._card_record(name, card))
        except Exception:
            features = {}
        self._card_vectors[name] = features
        return features

    def startup_focus(self, state: Dict[str, Any], descriptor: Dict[str, Any]) -> float:
        """How strongly one corp/keep plan commits to an award or milestone on this map.

        Works for any card-derivable award or milestone, so random-MA boards
        are steered as well as the base Tharsis set.
        """
        game = state.get("game", {}) or {}
        tracks = [
            track
            for row in [*(game.get("awards") or []), *(game.get("milestones") or [])]
            if isinstance(row, dict) and (track := track_for(row.get("name", row.get("title", "")))) is not None
        ]
        if not tracks:
            return 0.0
        waiting = state.get("waitingFor", {}) or {}
        decoded = descriptor.get("decoded_action") or {}
        contents = _startup_plan_contents(decoded if isinstance(decoded, dict) else {}, waiting)
        names = [contents.get("corp", ""), *(contents.get("prelude") or []), *(contents.get("project") or [])]
        features: Dict[str, float] = {}
        for name in names:
            name = str(name or "").strip()
            if not name:
                continue
            card = _find_prompt_card(waiting, name) or {"name": name}
            for key, value in self._card_features(name, card).items():
                features[key] = features.get(key, 0.0) + self._safe_float(value)
        totals = plan_totals(features)
        scores = sorted(
            (min(1.0, max(0.0, track_value(weights, totals)) / scale) for weights, scale in tracks),
            reverse=True,
        )
        # One owned track plus a supporting one (e.g. Builder + Miner).
        return scores[0] + (0.5 * scores[1] if len(scores) > 1 else 0.0)

    def _choose_startup(
        self,
        state: Dict[str, Any],
        descriptors: Sequence[Dict[str, Any]],
        policy_action_index: Optional[int],
        position_probs: Optional[Sequence[float]],
    ) -> Optional[Dict[str, Any]]:
        plans = [
            (position, row) for position, row in enumerate(descriptors)
            if str(row.get("family", "")) == "startup_plan"
        ]
        if len(plans) < 2:
            return None
        aligned = position_probs is not None and len(position_probs) == len(descriptors)
        uniform = 1.0 / float(len(plans))

        def prob(position: int) -> float:
            return max(1e-6, self._safe_float(position_probs[position])) if aligned else uniform

        ranked = sorted(plans, key=lambda item: prob(item[0]), reverse=True)[: max(1, int(self.init_top_k))]
        best = max(
            ranked,
            key=lambda item: math.log(prob(item[0]))
            + float(self.init_focus_weight) * self.startup_focus(state, item[1]),
        )[1]
        with self._lock:
            self.startup_decisions += 1
            if policy_action_index is None or int(best.get("action_index", -1)) != int(policy_action_index):
                self.startup_overridden += 1
        return best

    def _hand_potential(self, state: Dict[str, Any], award_name: str) -> float:
        """Award score the cards in hand would add if played (immediate effects only)."""
        track = track_for(award_name)
        if track is None:
            return 0.0
        hand = state.get("cardsInHand") or (state.get("thisPlayer", {}) or {}).get("cardsInHand") or []
        features: Dict[str, float] = {}
        for card in hand:
            card = card if isinstance(card, dict) else {"name": str(card or "")}
            name = str(card.get("name", "") or "")
            if not name:
                continue
            for key, value in self._card_features(name, card).items():
                features[key] = features.get(key, 0.0) + self._safe_float(value)
        return max(0.0, track_value(track[0], features))

    def choose(
        self,
        state: Dict[str, Any],
        descriptors: Sequence[Dict[str, Any]],
        policy_action_index: Optional[int] = None,
        position_probs: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the descriptor to take instead of the policy's, or ``None`` to defer.

        Handles startup steering (initial-cards prompt) and award funding.
        """
        if float(self.init_focus_weight) > 0.0 and any(
            str(row.get("family", "")) == "startup_plan" for row in descriptors or []
        ):
            return self._choose_startup(state, descriptors, policy_action_index, position_probs)
        options = [row for row in descriptors or [] if str(row.get("family", "")) == "fund_award"]
        if not options:
            return None
        with self._lock:
            self.offered += 1
        player = state.get("thisPlayer", {}) or {}
        game = state.get("game", {}) or {}
        generation = int(self._safe_float(game.get("generation", 1), 1.0))
        if generation < int(self.min_generation):
            return None
        if self._own_funded(game, player) >= int(self.max_own_awards):
            return None
        cost = self._estimate_award_cost(game)
        if cost > float(self.max_cost):
            return None
        if self._safe_float(player.get("megaCredits", 0)) < cost + float(self.reserve_mc):
            return None
        best: Optional[Dict[str, Any]] = None
        best_lead = float("-inf")
        for row in options:
            award = self._find_award(game, str(row.get("award_name", "") or row.get("label", "") or ""))
            if not award or self._track_taken(award):
                continue
            _own, _opp, projected_vp, lead_gap = self._award_standing(player, award, state)
            if projected_vp < 5.0:
                continue
            if float(self.hand_weight) > 0.0:
                lead_gap += float(self.hand_weight) * self._hand_potential(state, str(award.get("name", "") or ""))
            if lead_gap < float(self.min_lead):
                continue
            if lead_gap > best_lead:
                best, best_lead = row, lead_gap
        if best is None:
            return None
        with self._lock:
            self.fired += 1
            if policy_action_index is not None and int(best.get("action_index", -1)) == int(policy_action_index):
                self.policy_agreed += 1
        return best


def parse_award_override(spec: Optional[str]) -> Optional[AwardOverrideRule]:
    if spec is None or not str(spec).strip() or str(spec).strip().lower() in {"0", "off", "none"}:
        return None
    if str(spec).strip().lower() in {"1", "on", "default"}:
        return AwardOverrideRule()
    return AwardOverrideRule.from_spec(spec)


__all__: List[str] = ["AwardOverrideRule", "parse_award_override"]
