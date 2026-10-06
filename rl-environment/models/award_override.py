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

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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

    # Telemetry; shared across concurrent games, so guarded by a lock.
    offered: int = field(default=0, compare=False)
    fired: int = field(default=0, compare=False)
    policy_agreed: int = field(default=0, compare=False)
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
            for key in ("min_lead", "min_generation", "max_cost", "reserve_mc", "max_own_awards", "hand_weight")
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": self.config(),
                "decisions_with_fund_option": self.offered,
                "fired": self.fired,
                "fired_policy_already_chose_it": self.policy_agreed,
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

    def _hand_potential(self, state: Dict[str, Any], award_name: str) -> float:
        """Sum the award-track gain of every card in hand (immediate effects only)."""
        track = HeuristicTeacherPolicy._AWARD_TRACKS.get(str(award_name or "").strip().lower())
        if track is None:
            return 0.0
        hand = state.get("cardsInHand") or (state.get("thisPlayer", {}) or {}).get("cardsInHand") or []
        with self._lock:
            if self._teacher is None:
                self._teacher = HeuristicTeacherPolicy(sample=False)
            teacher = self._teacher
        total = 0.0
        for card in hand:
            card = card if isinstance(card, dict) else {"name": str(card or "")}
            name = str(card.get("name", "") or "")
            if not name:
                continue
            try:
                delta = teacher._card_track_delta(name, card, include_planner=False)
            except Exception:
                continue
            total += max(0.0, self._safe_float(delta.get(track[0])))
        return total

    def choose(
        self,
        state: Dict[str, Any],
        descriptors: Sequence[Dict[str, Any]],
        policy_action_index: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the fund-award descriptor to take, or ``None`` to defer."""
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
