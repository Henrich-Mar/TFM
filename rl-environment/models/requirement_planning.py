from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_CITY_TILE_TYPES = {2, 3, 20, 37, 43}
_GREENERY_TILE_TYPES = {0, 36}
_OCEAN_TILE_TYPES = {1, 20, 21, 22, 36, 43}
_MOON_MINE_TILE_TYPES = {29, 35}
_MOON_HABITAT_TILE_TYPES = {30}
_MOON_ROAD_TILE_TYPES = {31}
_STANDARD_RESOURCE_KEYS = ("megaCredits", "steel", "titanium", "plants", "energy", "heat")
_PRODUCTION_FIELD_BY_RESOURCE = {
    "megacredits": "megaCreditProduction",
    "megacredit": "megaCreditProduction",
    "m$": "megaCreditProduction",
    "steel": "steelProduction",
    "titanium": "titaniumProduction",
    "plants": "plantProduction",
    "plant": "plantProduction",
    "energy": "energyProduction",
    "heat": "heatProduction",
}
_GLOBAL_ROW_ORDER = {"temperature": 0, "oxygen": 1, "oceans": 2, "venus": 3, "tr": 4}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_tag_name(value: Any) -> str:
    raw = _normalize_text(value)
    if not raw:
        return ""
    return raw.capitalize() if raw[0].islower() else raw


def _normalize_key(value: Any) -> str:
    return _normalize_text(value).lower().replace("-", "").replace("_", "").replace(" ", "")


def _normalize_resource_type(value: Any) -> str:
    raw = _normalize_key(value)
    return {
        "microbes": "microbe",
        "animals": "animal",
        "floaters": "floater",
        "fighters": "fighter",
        "asteroids": "asteroid",
    }.get(raw, raw)


def _space_owner(space: Dict[str, Any]) -> str:
    return _normalize_key(space.get("color", space.get("playerColor", space.get("owner", ""))))


def _space_type_lower(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _tile_number(space: Dict[str, Any]) -> Optional[int]:
    try:
        raw = space.get("tileType")
        if isinstance(raw, bool):
            return None
        return int(raw)
    except Exception:
        return None


def _tile_flags(space: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    tile_num = _tile_number(space)
    tile_name = _space_type_lower(space.get("tileType"))
    is_city = tile_num in _CITY_TILE_TYPES if tile_num is not None else False
    is_greenery = tile_num in _GREENERY_TILE_TYPES if tile_num is not None else False
    is_ocean = tile_num in _OCEAN_TILE_TYPES if tile_num is not None else False
    if "city" in tile_name or tile_name in ("capital", "new_holland"):
        is_city = True
    if "greenery" in tile_name or "wetland" in tile_name:
        is_greenery = True
    if "ocean" in tile_name or "wetland" in tile_name:
        is_ocean = True
    return is_city, is_greenery, is_ocean


class RequirementPlanner:
    def __init__(self, card_metadata_by_name: Optional[Dict[str, Dict[str, Any]]] = None):
        self.card_metadata_by_name = card_metadata_by_name or {}

    def get_card_requirements(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        live = card.get("requirements")
        if isinstance(live, list):
            return [dict(item) for item in live if isinstance(item, dict)]
        name = _normalize_text(card.get("name"))
        meta = self.card_metadata_by_name.get(name, {}) if name else {}
        meta_requirements = meta.get("requirements")
        if isinstance(meta_requirements, list):
            return [dict(item) for item in meta_requirements if isinstance(item, dict)]
        return []

    def evaluate_card(self, card: Dict[str, Any], player_state: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.evaluate_requirements(self.get_card_requirements(card), player_state, card=card)
        plan["requirements"] = self.get_card_requirements(card)
        return plan

    def evaluate_requirements(
        self,
        requirements: Sequence[Dict[str, Any]],
        player_state: Dict[str, Any],
        card: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reqs = [dict(item) for item in requirements if isinstance(item, dict)]
        rows = [self._evaluate_descriptor(item, reqs, player_state) for item in reqs]
        rows.sort(key=self._row_sort_key)
        unsatisfied = [row for row in rows if not row["satisfied"]]
        all_satisfied = not unsatisfied
        readiness = 1.0 if all_satisfied else max(0.0, 1.0 - self._avg_gap_ratio(unsatisfied))
        reachability = 1.0 if all_satisfied else max(
            0.0,
            1.0 - (0.55 * self._avg_gap_ratio(unsatisfied)) - (0.15 * self._advisory_gap_ratio(unsatisfied)),
        )
        primary = unsatisfied[0] if unsatisfied else (rows[0] if rows else {})
        server_override = False
        masked_by_server = False
        if isinstance(card, dict) and isinstance(card.get("isDisabled"), bool):
            if (not bool(card.get("isDisabled"))) and (not all_satisfied):
                server_override = True
            elif bool(card.get("isDisabled")) and all_satisfied:
                masked_by_server = True
        for row in rows:
            row["server_override"] = server_override
            row["masked_by_server"] = masked_by_server
        return {
            "requirement_plan": rows,
            "all_satisfied": all_satisfied,
            "blocking_count": len(unsatisfied),
            "primary_gap_label": str(primary.get("label", "")),
            "primary_gap_axis": str(primary.get("type", "")),
            "reachability_score": float(max(0.0, min(1.0, reachability))),
            "readiness_score": float(max(0.0, min(1.0, readiness))),
            "plan_summary": self._build_plan_summary(rows, unsatisfied),
            "server_override": server_override,
            "masked_by_server": masked_by_server,
        }

    def _evaluate_descriptor(
        self,
        descriptor: Dict[str, Any],
        all_requirements: Sequence[Dict[str, Any]],
        player_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_player = dict(player_state.get("thisPlayer", {}) or {})
        game = dict(player_state.get("game", {}) or {})
        players = [dict(item) for item in (player_state.get("players", []) or []) if isinstance(item, dict)]
        if "tag" in descriptor:
            return self._tag_requirement(descriptor, current_player, players)
        if any(key in descriptor for key in ("oxygen", "temperature", "oceans", "venus", "tr")):
            return self._global_requirement(descriptor, current_player, game)
        if "production" in descriptor:
            return self._production_requirement(descriptor, current_player)
        if "party" in descriptor:
            return self._party_requirement(descriptor, current_player, game)
        if "chairman" in descriptor:
            return self._chairman_requirement(descriptor, current_player, game)
        if "partyLeader" in descriptor:
            return self._party_leader_requirement(descriptor, current_player, game)
        if "plantsRemoved" in descriptor:
            return self._advisory_row(descriptor, "plantsRemoved", "plants removed by another player", 1, "Public state does not expose this requirement.")
        if "cities" in descriptor:
            return self._city_requirement(descriptor, all_requirements, current_player, game)
        if "greeneries" in descriptor:
            return self._board_tile_requirement(descriptor, game, current_player, "greeneries")
        if "colonies" in descriptor:
            return self._simple_count_requirement(descriptor, current_player, players, "colonies")
        if "floaters" in descriptor:
            return self._simple_count_requirement(descriptor, current_player, players, "floaters")
        if "habitatRate" in descriptor:
            return self._moon_rate_requirement(descriptor, game, "habitatRate")
        if "miningRate" in descriptor:
            return self._moon_rate_requirement(descriptor, game, "miningRate")
        if "logisticRate" in descriptor:
            return self._moon_rate_requirement(descriptor, game, "logisticRate")
        if "habitatTiles" in descriptor:
            return self._moon_tile_requirement(descriptor, game, current_player, "habitatTiles")
        if "miningTiles" in descriptor:
            return self._moon_tile_requirement(descriptor, game, current_player, "miningTiles")
        if "roadTiles" in descriptor:
            return self._moon_tile_requirement(descriptor, game, current_player, "roadTiles")
        if "undergroundTokens" in descriptor:
            return self._simple_count_requirement(descriptor, current_player, players, "undergroundTokens")
        if "corruption" in descriptor:
            return self._simple_count_requirement(descriptor, current_player, players, "corruption")
        if "resourceTypes" in descriptor:
            return self._simple_count_requirement(descriptor, current_player, players, "resourceTypes")
        return self._advisory_row(descriptor, "unknown", "unknown requirement", _safe_int(descriptor.get("count", 1), 1), f"Unsupported requirement: {descriptor}")

    def _tag_requirement(
        self,
        descriptor: Dict[str, Any],
        current_player: Dict[str, Any],
        players: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tag = _normalize_tag_name(descriptor.get("tag"))
        all_players = bool(descriptor.get("all", False))
        is_max = bool(descriptor.get("max", False))
        target = _safe_int(descriptor.get("count", 1), 1)
        current = self._tag_count(current_player, tag, include_wild=not is_max)
        if all_players:
            own_id = _normalize_text(current_player.get("id"))
            own_color = _normalize_key(current_player.get("color"))
            for player in players:
                if self._same_player(player, own_id, own_color):
                    continue
                current += self._tag_count(player, tag, include_wild=False)
        return self._finalize_row(descriptor, "tag", f"{tag.lower()} tags", target, current, is_max, all_players)

    def _global_requirement(
        self,
        descriptor: Dict[str, Any],
        current_player: Dict[str, Any],
        game: Dict[str, Any],
    ) -> Dict[str, Any]:
        field_map = {
            "temperature": ("temperature", game.get("temperature", -30), "temperature"),
            "oxygen": ("oxygen", game.get("oxygenLevel", game.get("oxygen", 0)), "oxygen"),
            "oceans": ("oceans", game.get("oceans", 0), "oceans"),
            "venus": ("venus", game.get("venusScaleLevel", 0), "venus"),
            "tr": ("tr", current_player.get("terraformRating", 20), "TR"),
        }
        for key, (_, raw_current, label) in field_map.items():
            if key not in descriptor:
                continue
            target = _safe_int(descriptor.get("count", descriptor.get(key, 0)), _safe_int(descriptor.get(key, 0), 0))
            row = self._finalize_row(
                descriptor,
                key,
                label,
                target,
                _safe_int(raw_current, 0),
                bool(descriptor.get("max", False)),
                bool(descriptor.get("all", False)),
            )
            if key == "temperature":
                row["remaining_steps"] = int(math.ceil(max(0, row["remaining"]) / 2.0))
            return row
        return self._advisory_row(descriptor, "global", "global parameter", 0, "Unknown global requirement.")

    def _production_requirement(self, descriptor: Dict[str, Any], current_player: Dict[str, Any]) -> Dict[str, Any]:
        resource = _normalize_key(descriptor.get("production"))
        field = _PRODUCTION_FIELD_BY_RESOURCE.get(resource)
        if not field:
            return self._advisory_row(descriptor, "production", f"production {descriptor.get('production')}", _safe_int(descriptor.get("count", 0), 0), "Unknown production resource.")
        return self._finalize_row(
            descriptor,
            "production",
            f"{resource} production",
            _safe_int(descriptor.get("count", 0), 0),
            _safe_int(current_player.get(field, 0), 0),
            bool(descriptor.get("max", False)),
            bool(descriptor.get("all", False)),
        )

    def _party_requirement(
        self,
        descriptor: Dict[str, Any],
        current_player: Dict[str, Any],
        game: Dict[str, Any],
    ) -> Dict[str, Any]:
        turmoil = dict(game.get("turmoil", {}) or {})
        parties = [dict(item) for item in (turmoil.get("parties", []) or []) if isinstance(item, dict)]
        party_name = _normalize_text(descriptor.get("party"))
        own_color = _normalize_key(current_player.get("color"))
        allied_party = _normalize_text(((current_player.get("alliedParty") or {}).get("partyName")))
        delegates = 0
        for party in parties:
            if _normalize_text(party.get("name")) != party_name:
                continue
            for row in party.get("delegates", []) or []:
                if isinstance(row, dict) and _normalize_key(row.get("color")) == own_color:
                    delegates += _safe_int(row.get("number", 0), 0)
        current = 1 if (_normalize_text(turmoil.get("ruling")) == party_name or allied_party == party_name or delegates >= 2) else 0
        return self._finalize_row(descriptor, "party", f"{party_name} ruling", 1, current, False, False)

    def _chairman_requirement(
        self,
        descriptor: Dict[str, Any],
        current_player: Dict[str, Any],
        game: Dict[str, Any],
    ) -> Dict[str, Any]:
        turmoil = dict(game.get("turmoil", {}) or {})
        current = 1 if _normalize_key(turmoil.get("chairman")) == _normalize_key(current_player.get("color")) else 0
        return self._finalize_row(descriptor, "chairman", "chairman", 1, current, False, False)

    def _party_leader_requirement(
        self,
        descriptor: Dict[str, Any],
        current_player: Dict[str, Any],
        game: Dict[str, Any],
    ) -> Dict[str, Any]:
        turmoil = dict(game.get("turmoil", {}) or {})
        own_color = _normalize_key(current_player.get("color"))
        parties = [dict(item) for item in (turmoil.get("parties", []) or []) if isinstance(item, dict)]
        current = sum(1 for party in parties if _normalize_key(party.get("partyLeader")) == own_color)
        target = _safe_int(descriptor.get("count", descriptor.get("partyLeader", 0)), 0)
        return self._finalize_row(descriptor, "partyLeader", "party leaderships", target, current, bool(descriptor.get("max", False)), False)

    def _city_requirement(
        self,
        descriptor: Dict[str, Any],
        all_requirements: Sequence[Dict[str, Any]],
        current_player: Dict[str, Any],
        game: Dict[str, Any],
    ) -> Dict[str, Any]:
        if bool(descriptor.get("nextTo", False)):
            if any("oceans" in item for item in all_requirements):
                current = self._count_cities_adjacent_to_oceans(game, current_player, bool(descriptor.get("all", False)))
                label = "cities next to oceans" if bool(descriptor.get("all", False)) else "your cities next to oceans"
                return self._finalize_row(
                    descriptor,
                    "cities",
                    label,
                    _safe_int(descriptor.get("count", descriptor.get("cities", 0)), 0),
                    current,
                    bool(descriptor.get("max", False)),
                    bool(descriptor.get("all", False)),
                )
            return self._advisory_row(descriptor, "cities", "cities with adjacency condition", _safe_int(descriptor.get("count", descriptor.get("cities", 0)), 0), "Unsupported nextTo requirement without an ocean companion requirement.")
        return self._board_tile_requirement(descriptor, game, current_player, "cities")

    def _board_tile_requirement(
        self,
        descriptor: Dict[str, Any],
        game: Dict[str, Any],
        current_player: Dict[str, Any],
        field: str,
    ) -> Dict[str, Any]:
        spaces = [dict(item) for item in (game.get("spaces", []) or []) if isinstance(item, dict)]
        own_color = _normalize_key(current_player.get("color"))
        all_players = bool(descriptor.get("all", False))
        current = 0
        for space in spaces:
            is_city, is_greenery, _ = _tile_flags(space)
            if field == "cities" and not is_city:
                continue
            if field == "greeneries" and not is_greenery:
                continue
            if all_players or _space_owner(space) == own_color:
                current += 1
        label = field if all_players else ("your cities" if field == "cities" else "your greeneries")
        return self._finalize_row(
            descriptor,
            field,
            label,
            _safe_int(descriptor.get("count", descriptor.get(field, 0)), 0),
            current,
            bool(descriptor.get("max", False)),
            all_players,
        )

    def _moon_rate_requirement(self, descriptor: Dict[str, Any], game: Dict[str, Any], field: str) -> Dict[str, Any]:
        moon = dict(game.get("moon", {}) or {})
        label = {"habitatRate": "moon habitat rate", "miningRate": "moon mining rate", "logisticRate": "moon logistics rate"}[field]
        return self._finalize_row(
            descriptor,
            field,
            label,
            _safe_int(descriptor.get("count", descriptor.get(field, 0)), 0),
            _safe_int(moon.get(field, 0), 0),
            bool(descriptor.get("max", False)),
            False,
        )

    def _moon_tile_requirement(
        self,
        descriptor: Dict[str, Any],
        game: Dict[str, Any],
        current_player: Dict[str, Any],
        field: str,
    ) -> Dict[str, Any]:
        all_players = bool(descriptor.get("all", False))
        label = {"habitatTiles": "moon habitat tiles", "miningTiles": "moon mine tiles", "roadTiles": "moon road tiles"}[field]
        return self._finalize_row(
            descriptor,
            field,
            label,
            _safe_int(descriptor.get("count", descriptor.get(field, 0)), 0),
            self._count_moon_tiles(game, current_player, field, all_players),
            bool(descriptor.get("max", False)),
            all_players,
        )

    def _simple_count_requirement(self, descriptor: Dict[str, Any], current_player: Dict[str, Any], players: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
        all_players = bool(descriptor.get("all", False))
        if field == "colonies":
            current = self._count_colonies(current_player, players, all_players)
            label = "colonies"
        elif field == "floaters":
            current = self._count_floaters(current_player, players, all_players)
            label = "floaters"
        elif field == "undergroundTokens":
            current = self._count_underworld_tokens(current_player, players, all_players)
            label = "underground tokens"
        elif field == "corruption":
            current = self._count_corruption(current_player, players, all_players)
            label = "corruption"
        else:
            current = self._count_resource_types(current_player)
            label = "resource types"
        return self._finalize_row(descriptor, field, label, _safe_int(descriptor.get("count", descriptor.get(field, 0)), 0), current, bool(descriptor.get("max", False)), all_players)

    def _count_colonies(self, current_player: Dict[str, Any], players: Sequence[Dict[str, Any]], all_players: bool) -> int:
        if not all_players:
            return _safe_int(current_player.get("coloniesCount", 0), 0)
        return self._sum_players(players, current_player, lambda player: _safe_int(player.get("coloniesCount", 0), 0))

    def _count_floaters(self, current_player: Dict[str, Any], players: Sequence[Dict[str, Any]], all_players: bool) -> int:
        if not all_players:
            return self._sum_resource_type(current_player.get("tableau", []), "floater")
        return self._sum_players(players, current_player, lambda player: self._sum_resource_type(player.get("tableau", []), "floater"))

    def _count_underworld_tokens(self, current_player: Dict[str, Any], players: Sequence[Dict[str, Any]], all_players: bool) -> int:
        token_count = lambda player: len([item for item in (((player.get("underworldData") or {}).get("tokens", []) or []))])
        if not all_players:
            return token_count(current_player)
        return self._sum_players(players, current_player, token_count)

    def _count_corruption(self, current_player: Dict[str, Any], players: Sequence[Dict[str, Any]], all_players: bool) -> int:
        corruption = lambda player: _safe_int(((player.get("underworldData") or {}).get("corruption", 0)), 0)
        if not all_players:
            return corruption(current_player)
        return self._sum_players(players, current_player, corruption)

    def _count_resource_types(self, current_player: Dict[str, Any]) -> int:
        total = sum(1 for key in _STANDARD_RESOURCE_KEYS if _safe_int(current_player.get(key, 0), 0) > 0)
        resource_types = {
            self._get_card_resource_type(card)
            for card in (current_player.get("tableau", []) or [])
            if isinstance(card, dict) and _safe_int(card.get("resources", 0), 0) > 0 and self._get_card_resource_type(card)
        }
        total += len(resource_types)
        if _safe_int(((current_player.get("underworldData") or {}).get("corruption", 0)), 0) > 0:
            total += 1
        return total

    def _count_moon_tiles(self, game: Dict[str, Any], current_player: Dict[str, Any], field: str, all_players: bool) -> int:
        moon = dict(game.get("moon", {}) or {})
        spaces = [dict(item) for item in (moon.get("spaces", []) or []) if isinstance(item, dict)]
        if not spaces:
            spaces = [dict(item) for item in (game.get("spaces", []) or []) if isinstance(item, dict) and str(item.get("id", "")).lower().startswith("m")]
        own_color = _normalize_key(current_player.get("color"))
        allowed = {
            "habitatTiles": _MOON_HABITAT_TILE_TYPES,
            "miningTiles": _MOON_MINE_TILE_TYPES,
            "roadTiles": _MOON_ROAD_TILE_TYPES,
        }[field]
        count = 0
        for space in spaces:
            if _tile_number(space) not in allowed:
                continue
            if all_players or _space_owner(space) == own_color:
                count += 1
        return count

    def _count_cities_adjacent_to_oceans(self, game: Dict[str, Any], current_player: Dict[str, Any], all_players: bool) -> int:
        spaces = [dict(item) for item in (game.get("spaces", []) or []) if isinstance(item, dict)]
        own_color = _normalize_key(current_player.get("color"))
        count = 0
        for space in spaces:
            is_city, _, _ = _tile_flags(space)
            if not is_city:
                continue
            if (not all_players) and _space_owner(space) != own_color:
                continue
            if self._is_adjacent_to_ocean(space, spaces):
                count += 1
        return count

    def _is_adjacent_to_ocean(self, space: Dict[str, Any], all_spaces: Sequence[Dict[str, Any]]) -> bool:
        return any(_tile_flags(adjacent)[2] for adjacent in self._adjacent_spaces(space, all_spaces))

    def _adjacent_spaces(self, space: Dict[str, Any], all_spaces: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = [item for item in all_spaces if _space_type_lower(item.get("spaceType")) != "colony"]
        if not candidates:
            return []
        max_y = max(_safe_int(item.get("y", 0), 0) for item in candidates)
        middle_row = max_y / 2.0
        x = _safe_int(space.get("x", 0), 0)
        y = _safe_int(space.get("y", 0), 0)
        left = (x - 1, y)
        right = (x + 1, y)
        top_left = [x, y - 1]
        top_right = [x, y - 1]
        bottom_left = [x, y + 1]
        bottom_right = [x, y + 1]
        if y < middle_row:
            bottom_left[0] -= 1
            top_right[0] += 1
        elif y == middle_row:
            bottom_right[0] += 1
            top_right[0] += 1
        else:
            bottom_right[0] += 1
            top_left[0] -= 1
        coords = [tuple(top_left), tuple(top_right), right, tuple(bottom_right), tuple(bottom_left), left]
        out: List[Dict[str, Any]] = []
        for adj_x, adj_y in coords:
            for candidate in candidates:
                if candidate is not space and _safe_int(candidate.get("x", 0), 0) == adj_x and _safe_int(candidate.get("y", 0), 0) == adj_y:
                    out.append(candidate)
                    break
        return out

    def _tag_count(self, player: Dict[str, Any], tag: str, include_wild: bool) -> int:
        raw = player.get("tags", {}) or {}
        tag_key = _normalize_key(tag)
        if isinstance(raw, dict):
            count = _safe_int(raw.get(tag_key, raw.get(tag.lower(), 0)), 0)
            if include_wild:
                count += _safe_int(raw.get("wild", 0), 0)
            return count
        if isinstance(raw, list):
            normalized = [_normalize_key(item) for item in raw]
            count = normalized.count(tag_key)
            if include_wild:
                count += normalized.count("wild")
            return count
        return 0

    def _get_card_resource_type(self, card: Dict[str, Any]) -> str:
        own = _normalize_resource_type(card.get("resourceType"))
        if own:
            return own
        name = _normalize_text(card.get("name"))
        meta = self.card_metadata_by_name.get(name, {}) if name else {}
        return _normalize_resource_type(meta.get("resourceType"))

    def _sum_resource_type(self, cards: Iterable[Any], resource_type: str) -> int:
        target = _normalize_resource_type(resource_type)
        total = 0
        for card in cards or []:
            if isinstance(card, dict) and self._get_card_resource_type(card) == target:
                total += _safe_int(card.get("resources", 0), 0)
        return total

    def _same_player(self, player: Dict[str, Any], own_id: str, own_color: str) -> bool:
        return bool((own_id and _normalize_text(player.get("id")) == own_id) or (own_color and _normalize_key(player.get("color")) == own_color))

    def _sum_players(self, players: Sequence[Dict[str, Any]], current_player: Dict[str, Any], getter) -> int:
        own_id = _normalize_text(current_player.get("id"))
        own_color = _normalize_key(current_player.get("color"))
        total = 0
        seen_self = False
        for player in players:
            if self._same_player(player, own_id, own_color):
                seen_self = True
            total += int(getter(player))
        if not seen_self:
            total += int(getter(current_player))
        return total

    def _finalize_row(self, descriptor: Dict[str, Any], requirement_type: str, label: str, target: int, current: int, is_max: bool, all_players: bool, advisory_only: bool = False, text: str = "") -> Dict[str, Any]:
        satisfied = current <= target if is_max else current >= target
        remaining = max(0, current - target) if is_max else max(0, target - current)
        row = {
            "type": requirement_type,
            "label": label,
            "satisfied": bool(satisfied),
            "target": int(target),
            "current": int(current),
            "remaining": int(remaining),
            "remaining_steps": None,
            "is_max": bool(is_max),
            "all_players": bool(all_players),
            "count": _safe_int(descriptor.get("count", target), target),
            "next_to": bool(descriptor.get("nextTo", False)),
            "text": _normalize_text(descriptor.get("text")) or text or self._format_row_text(requirement_type, label, target, current, remaining, is_max),
            "advisory_only": bool(advisory_only),
            "server_override": False,
            "masked_by_server": False,
        }
        if requirement_type == "temperature":
            row["remaining_steps"] = int(math.ceil(max(0, remaining) / 2.0))
        return row

    def _advisory_row(self, descriptor: Dict[str, Any], requirement_type: str, label: str, target: int, text: str) -> Dict[str, Any]:
        return self._finalize_row(descriptor, requirement_type, label, target, 0, False, bool(descriptor.get("all", False)), advisory_only=True, text=text)

    def _format_row_text(self, requirement_type: str, label: str, target: int, current: int, remaining: int, is_max: bool) -> str:
        comparator = "<=" if is_max else ">="
        base = f"{label} {comparator} {target}, now {current}"
        if remaining <= 0:
            return base
        if requirement_type == "temperature":
            return f"{base}, need +{int(math.ceil(remaining / 2.0))} steps"
        return f"{base}, {'exceed by' if is_max else 'need'} {remaining}"

    def _avg_gap_ratio(self, rows: Sequence[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return float(sum(self._row_gap_ratio(row) for row in rows)) / float(len(rows))

    def _advisory_gap_ratio(self, rows: Sequence[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        advisory = [row for row in rows if bool(row.get("advisory_only", False))]
        return float(len(advisory)) / float(len(rows)) if advisory else 0.0

    def _row_gap_ratio(self, row: Dict[str, Any]) -> float:
        requirement_type = str(row.get("type", ""))
        if requirement_type == "temperature":
            return min(float(_safe_int(row.get("remaining_steps", 0), 0)) / 10.0, 1.0)
        remaining = float(_safe_int(row.get("remaining", 0), 0))
        target = max(float(_safe_int(row.get("target", 0), 0)), 1.0)
        denom = {
            "oxygen": 7.0,
            "oceans": 9.0,
            "venus": 15.0,
            "tr": 20.0,
            "party": 1.0,
            "chairman": 1.0,
        }.get(requirement_type, max(target, 3.0))
        ratio = min(remaining / max(denom, 1.0), 1.0)
        return max(0.55, ratio) if bool(row.get("advisory_only", False)) else ratio

    def _build_plan_summary(self, rows: Sequence[Dict[str, Any]], unsatisfied: Sequence[Dict[str, Any]]) -> str:
        if not rows:
            return "No requirements."
        if not unsatisfied:
            return "All requirements satisfied."
        return "; ".join(str(row.get("text", "")) for row in list(unsatisfied)[:3] if str(row.get("text", ""))) or "Requirements not satisfied."

    def _row_sort_key(self, row: Dict[str, Any]) -> Tuple[int, int, float, str]:
        return (0 if not row["satisfied"] else 1, _GLOBAL_ROW_ORDER.get(str(row.get("type", "")), 50), -self._row_gap_ratio(row), str(row.get("label", "")))
