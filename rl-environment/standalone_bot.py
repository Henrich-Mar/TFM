"""
Run a single trained agent against an existing Terraforming Mars player slot.

Examples:
  python rl-environment/standalone_bot.py --player-url "http://localhost:8081/player?id=<PLAYER_ID>"
  python rl-environment/standalone_bot.py --base-url "http://localhost:8081" --player-id "<PLAYER_ID>"
"""
import argparse
import asyncio
import copy
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit, urlunsplit

import aiohttp

from metadata_refresh import ensure_card_metadata
from models.agent import RLAgent
from models.state_encoder import StateEncoder

logger = logging.getLogger(__name__)


def _iter_player_models(player_state: Dict[str, Any]):
    """Yield the private and public player models contained in an API response."""
    this_player = player_state.get("thisPlayer")
    if isinstance(this_player, dict):
        yield this_player
    for player in player_state.get("players", []) or []:
        if isinstance(player, dict):
            yield player


def _normalize_inbound_player_schema(player_state: Dict[str, Any]) -> Dict[str, Any]:
    """Add current field aliases without discarding the server's original data."""
    for player in _iter_player_models(player_state):
        if "megaCredits" not in player and "megacredits" in player:
            player["megaCredits"] = player["megacredits"]
        if "megaCreditProduction" not in player and "megacreditProduction" in player:
            player["megaCreditProduction"] = player["megacreditProduction"]
    return player_state


def _uses_lowercase_payment_mc(player_state: Dict[str, Any]) -> bool:
    return any(
        "megacredits" in player and "megaCredits" not in player
        for player in _iter_player_models(player_state)
    )


def _adapt_outbound_payment_schema(input_data: Dict[str, Any], lowercase_mc: bool) -> Dict[str, Any]:
    """Use the payment-key spelling advertised by the connected server.

    The older public server serializes player money as ``megacredits`` and
    validates that same spelling inside every payment payload. The local server
    uses ``megaCredits``. Only payment objects are rewritten; card/game state
    is never altered on the wire.
    """
    payload = copy.deepcopy(input_data)

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        payment = value.get("payment")
        if isinstance(payment, dict):
            if lowercase_mc and "megaCredits" in payment:
                payment["megacredits"] = payment.pop("megaCredits")
            elif not lowercase_mc and "megacredits" in payment:
                payment["megaCredits"] = payment.pop("megacredits")
        for nested in value.values():
            visit(nested)

    visit(payload)
    return payload


def _log_agent_metadata(agent: RLAgent):
    """Log card metadata, milestones, awards, and other agent information."""
    print("\n" + "=" * 70)
    print("AGENT METADATA INFORMATION")
    print("=" * 70)
    
    # Card metadata information
    state_encoder = agent.state_encoder
    card_metadata = state_encoder.card_metadata_by_name
    common_cards = state_encoder.common_cards
    
    print(f"\n[Card Metadata]")
    print(f"  Total cards loaded: {len(card_metadata)}")
    print(f"  Common cards list size: {len(common_cards)}")
    
    if card_metadata:
        # Show card metadata source
        env_path = os.getenv('TM_CARD_METADATA_PATH')
        if env_path:
            print(f"  Metadata source: {env_path} (from TM_CARD_METADATA_PATH)")
        else:
            print(f"  Metadata source: Auto-detected from default locations")
        
        # Show sample cards with their metadata
        print(f"\n  Sample cards (first 10):")
        for i, card_name in enumerate(common_cards[:10]):
            meta = card_metadata.get(card_name, {})
            tags = meta.get('tags', [])
            card_type = meta.get('type', 'unknown')
            cost_raw = meta.get('cost', None)
            cost_str = str(cost_raw) if cost_raw is not None else '?'
            tags_str = ', '.join(str(tag) for tag in tags[:3])
            if len(tags) > 3:
                tags_str += '...'
            print(f"    {i+1:2d}. {card_name:30s} | Type: {card_type:10s} | Cost: {cost_str:>3s} | Tags: {tags_str}")
        
        if len(common_cards) > 10:
            print(f"    ... and {len(common_cards) - 10} more cards")
        
        # Count cards by type
        type_counts = {}
        for card_name, meta in card_metadata.items():
            card_type = meta.get('type', 'unknown')
            type_counts[card_type] = type_counts.get(card_type, 0) + 1
        
        if type_counts:
            print(f"\n  Cards by type:")
            for card_type, count in sorted(type_counts.items()):
                print(f"    {card_type:15s}: {count:4d}")
    else:
        print("  WARNING: No card metadata loaded - using fallback card list")
    
    # Milestones information
    milestones = StateEncoder._ALL_MILESTONES
    print(f"\n[Milestones]")
    print(f"  Total milestones tracked: {len(milestones)}")
    print(f"  Sample milestones: {', '.join(milestones[:5])}...")
    
    # Awards information
    awards = StateEncoder._ALL_AWARDS
    print(f"\n[Awards]")
    print(f"  Total awards tracked: {len(awards)}")
    print(f"  Sample awards: {', '.join(awards[:5])}...")
    
    # Agent configuration
    print(f"\n[Agent Configuration]")
    print(f"  Agent ID: {agent.id[:8]}...")
    print(f"  Hidden size: {agent.config.hidden_size}")
    print(f"  Recurrent size: {agent.config.recurrent_size}")
    print(f"  Phase head count: {agent.config.phase_head_count}")
    print(f"  Planner token dim: {agent.config.planner_token_dim}")
    print(f"  Planner global dim: {agent.config.planner_global_dim}")
    print(f"  Planner limits: tableau={agent.config.planner_tableau_limit}, hand={agent.config.planner_hand_limit}, "
          f"opponents={agent.config.planner_opponent_limit}, opportunities={agent.config.planner_opportunity_limit}")
    print(f"  Learning rate: {agent.config.learning_rate}")
    print(f"  Temperature: {agent.config.temperature}")
    print(f"  Epsilon: {agent.config.epsilon}")
    
    # Network information
    network = agent.network
    print(f"\n[Network Architecture]")
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Action dimension: {network.action_dim}")
    print(f"  Recurrent size: {network.recurrent_size}")
    print(f"  Phase heads: {network.phase_head_count}")
    
    print("=" * 70 + "\n")


def _default_models_root() -> str:
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


def _fitness_from_name(path: str) -> float:
    base = os.path.basename(path)
    try:
        return float(base.split("_fitness_")[-1].replace(".pth", ""))
    except Exception:
        return float("-inf")


def _find_best_checkpoint(models_root: str) -> str:
    pattern = os.path.join(models_root, "generation_*", "agent_*_fitness_*.pth")
    matches = sorted(set(glob.glob(pattern)), key=_fitness_from_name, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No checkpoints found under: {models_root}")
    return matches[0]


def _normalize_base_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("base URL is required")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {scheme}")
    netloc = parsed.netloc or parsed.path
    if not netloc:
        raise ValueError(f"Invalid URL: {value}")
    return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")


def _extract_query_id(url: str) -> Optional[str]:
    try:
        parsed = urlsplit(str(url or "").strip())
        query = parse_qs(parsed.query)
    except Exception:
        return None
    values = query.get("id", [])
    for value in values:
        token = str(value or "").strip()
        if token:
            return token
    return None


def _parse_player_url(player_url: str) -> Tuple[str, str]:
    raw = str(player_url or "").strip()
    if not raw:
        raise ValueError("player URL is empty")
    parsed = urlsplit(raw)
    if not parsed.scheme:
        parsed = urlsplit(f"https://{raw}")
    base_url = _normalize_base_url(urlunsplit((parsed.scheme, parsed.netloc, "", "", "")))
    player_id = _extract_query_id(raw)
    if not player_id:
        raise ValueError("player URL must include query parameter id=<PLAYER_ID>")
    return base_url, player_id


class StandaloneGameClient:
    def __init__(
        self,
        base_url: str,
        player_id: str,
        session: aiohttp.ClientSession,
        game_id: Optional[str] = None,
        min_action_interval_sec: float = 1.0,
    ):
        self.base_url = _normalize_base_url(base_url)
        self._player_id = str(player_id)
        self.session = session
        self.game_id = str(game_id or "").strip()
        self.cluster = None
        self.player_ids: Dict[str, str] = {}
        self._min_action_interval_sec = max(0.0, float(min_action_interval_sec))
        self._next_allowed_action_monotonic = 0.0
        self._payment_uses_lowercase_mc = False

    def _resolve_public_base(self) -> str:
        return self.base_url

    def get_public_game_url(self) -> str:
        if self.game_id:
            return f"{self._resolve_public_base()}/game?id={self.game_id}"
        return self._resolve_public_base()

    def get_public_player_api_url(self, player_id: str) -> str:
        return f"{self._resolve_public_base()}/api/player?id={player_id}"

    def get_public_player_url(self, player_id: str) -> str:
        return f"{self._resolve_public_base()}/player?id={player_id}"

    def get_internal_player_api_url(self, player_id: str) -> str:
        return self.get_public_player_api_url(player_id)

    async def join_player(self, player_name: str) -> str:
        if player_name:
            self.player_ids[str(player_name)] = self._player_id
        return self._player_id

    async def get_player_state(self, player_id: str) -> Dict[str, Any]:
        async with self.session.get(
            f"{self.base_url}/api/player",
            params={"id": player_id},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to get player state: HTTP {response.status}")
            player_state = await response.json()
            if isinstance(player_state, dict):
                self._payment_uses_lowercase_mc = _uses_lowercase_payment_mc(player_state)
                return _normalize_inbound_player_schema(player_state)
            return player_state

    async def _throttle_action_send(self):
        if self._min_action_interval_sec <= 0.0:
            return
        now = asyncio.get_running_loop().time()
        wait = float(self._next_allowed_action_monotonic) - float(now)
        if wait > 0.0:
            await asyncio.sleep(wait)
            now = asyncio.get_running_loop().time()
        self._next_allowed_action_monotonic = float(now) + float(self._min_action_interval_sec)

    async def send_player_input(self, player_id: str, input_data: Dict[str, Any]) -> bool:
        await self._throttle_action_send()
        wire_input = _adapt_outbound_payment_schema(input_data, self._payment_uses_lowercase_mc)
        try:
            async with self.session.post(
                f"{self.base_url}/player/input",
                params={"id": player_id},
                json=wire_input,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return True
                response_text = await response.text()
                logger.warning(
                    "Input rejected. status=%s player=%s response=%s payload=%s",
                    response.status,
                    player_id,
                    response_text[:500],
                    json.dumps(wire_input, ensure_ascii=False, sort_keys=True)[:4000],
                )
                return False
        except Exception as e:
            logger.error("Failed to send input for player %s: %s", player_id, e)
            return False

    async def get_final_state(self) -> Dict[str, Any]:
        if not self.game_id:
            return {}
        try:
            async with self.session.get(
                f"{self.base_url}/api/game",
                params={"id": self.game_id},
            ) as response:
                if response.status != 200:
                    logger.warning("Failed to fetch final game state: HTTP %s", response.status)
                    return {}
                return await response.json()
        except Exception as e:
            logger.warning("Failed to fetch final game state: %s", e)
            return {}

    async def cleanup(self):
        return


def _resolve_target(
    player_url: str,
    base_url: str,
    player_id: str,
) -> Tuple[str, str]:
    parsed_base = ""
    parsed_player_id = ""
    if str(player_url or "").strip():
        parsed_base, parsed_player_id = _parse_player_url(player_url)
    final_base = str(base_url or "").strip() or parsed_base
    final_player = str(player_id or "").strip() or parsed_player_id
    if not final_base:
        raise ValueError("Provide --player-url or --base-url")
    if not final_player:
        raise ValueError("Provide --player-url or --player-id")
    return _normalize_base_url(final_base), final_player


async def _run(args: argparse.Namespace):
    ensure_card_metadata(quiet=True)
    if args.no_random_fallback:
        # Human matches should never progress through an unrelated random move
        # after a rejected neural action. The next poll may still select a
        # different policy action, but no random fallback is submitted.
        os.environ["MAX_FALLBACK_RANDOM_RETRIES_PER_PROMPT"] = "0"
    checkpoint = str(args.checkpoint or "").strip() or _find_best_checkpoint(args.models)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    base_url, player_id = _resolve_target(
        player_url=args.player_url,
        base_url=args.base_url,
        player_id=args.player_id,
    )
    game_id = str(args.game_id or "").strip()
    if not game_id and str(args.game_url or "").strip():
        game_id = str(_extract_query_id(args.game_url) or "").strip()

    min_action_delay_ms = max(1000, int(args.min_action_delay_ms))
    min_action_interval_sec = float(min_action_delay_ms) / 1000.0
    poll_interval_sec = max(0.1, float(args.poll_interval_ms) / 1000.0)

    print(f"Using checkpoint: {checkpoint}")
    print(f"Target base URL: {base_url}")
    print(f"Target player ID: {player_id}")
    if game_id:
        print(f"Target game ID: {game_id}")
    print(f"Minimum action interval: {min_action_delay_ms} ms")
    print(f"Poll interval: {int(poll_interval_sec * 1000)} ms")
    if args.no_random_fallback:
        print("Random fallback actions: disabled")

    timeout = aiohttp.ClientTimeout(total=max(5.0, float(args.request_timeout_sec)))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        game = StandaloneGameClient(
            base_url=base_url,
            player_id=player_id,
            session=session,
            game_id=game_id,
            min_action_interval_sec=min_action_interval_sec,
        )

        agent = RLAgent()
        agent.load_model(checkpoint)
        agent.train_from_self_play = False
        agent.post_move_sleep_sec = max(float(agent.post_move_sleep_sec), min_action_interval_sec)
        agent.failure_pause_sec = max(float(agent.failure_pause_sec), min_action_interval_sec)
        agent.poll_interval_sec = max(float(agent.poll_interval_sec), poll_interval_sec)

        # Log card metadata and state encoder information
        _log_agent_metadata(agent)

        initial_state = await game.get_player_state(player_id)
        resolved_player_name = str(
            args.player_name
            or ((initial_state.get("thisPlayer", {}) or {}).get("name"))
            or f"RemoteBot_{agent.id[:8]}"
        )
        game.player_ids[resolved_player_name] = player_id

        print(f"Resolved player name: {resolved_player_name}")
        print(f"Player URL: {game.get_public_player_url(player_id)}")
        if game_id:
            print(f"Game URL: {game.get_public_game_url()}")
        print("Bot attached. Keep this process running until the game finishes.")

        await agent.play_game(game, resolved_player_name)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach the best trained agent checkpoint to an existing TM player URL."
    )
    parser.add_argument(
        "--player-url",
        type=str,
        default="",
        help="Full player URL, e.g. http://localhost:8081/player?id=<PLAYER_ID>",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base server URL, e.g. http://localhost:8081",
    )
    parser.add_argument(
        "--player-id",
        type=str,
        default="",
        help="Player ID token from the player URL",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default="",
        help="Optional game ID for final standings lookup",
    )
    parser.add_argument(
        "--game-url",
        type=str,
        default="",
        help="Optional game URL (used only to extract game ID)",
    )
    parser.add_argument(
        "--player-name",
        type=str,
        default="",
        help="Optional player name override. Default: fetched from /api/player",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Explicit checkpoint path (.pth). If omitted, best checkpoint is auto-selected.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=_default_models_root(),
        help="Models root used when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--min-action-delay-ms",
        type=int,
        default=1000,
        help="Minimum delay between action submissions. Values below 1000 are clamped to 1000.",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=1000,
        help="Polling interval for /api/player state checks.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=60.0,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--no-random-fallback",
        action="store_true",
        help="Do not submit random fallback actions after a rejected policy action (recommended for human games).",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    level_name = str(args.log_level or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
