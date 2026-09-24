from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_canonical import (  # noqa: E402
    canonical_payload_text,
    canonicalize_legal_actions,
    merge_equivalent_action_mass,
    normalize_payload,
)
from models.action_contract import Action  # noqa: E402
from models.action_decoder import ActionDecoder  # noqa: E402
from models.state_encoder import StateEncoder  # noqa: E402
from models.agent import AgentConfig, TerraformingMarsNetwork  # noqa: E402
from models.card_catalog import (  # noqa: E402
    ACTION_EFFECTS,
    CARD_CAPACITY,
    FLAGS,
    IMMEDIATE,
    METADATA_DIM,
    CardCatalogError,
    CardEncodingError,
    bind_action_card_mask,
    build_catalog,
    get_catalog,
    iter_subset_masks,
    load_metadata,
    metadata_vector,
    resolve_behavior,
    validate_checkpoint_catalog,
    write_catalog,
)
from models.decision_policy import HeuristicTeacherPolicy  # noqa: E402
from models.planner_common import ensure_bundle  # noqa: E402
from scoring import _card_quality  # noqa: E402
from training.v2_pretrain import _target_family  # noqa: E402
from training.v4_diagnose_checkpoint import _placement_report  # noqa: E402
from training.v4_gates import evaluate_ppo_gate, validation_candidate_status  # noqa: E402
from training.v4_human import partition_human_samples, reencode_human_events  # noqa: E402
from training.v4_warm_start import REQUIRED_ARCHITECTURE, _require_source_architecture  # noqa: E402


def _meta(name: str, **overrides):
    payload = {
        "name": name,
        "type": "event",
        "tags": ["space"],
        "cost": 10,
        "category": "project",
        "description": "",
        "requirements": [],
    }
    payload.update(overrides)
    return payload


def test_catalog_is_deterministic_and_maps_every_current_card_once() -> None:
    metadata = load_metadata()
    first = build_catalog(metadata)
    second = build_catalog(metadata)
    assert first.sha256 == second.sha256
    assert len(first.entries) == 989
    assert [item["id"] for item in first.entries] == list(range(1, 990))
    assert [item["name"] for item in first.entries] == sorted(metadata)
    assert len({item["name"] for item in first.entries}) == 989


def test_catalog_file_matches_regenerated_hash(tmp_path: Path) -> None:
    destination = write_catalog(tmp_path / "card_catalog.v1.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "card_catalog.v1"
    assert payload["capacity"] == CARD_CAPACITY
    assert payload["unknown_id"] == 0
    assert payload["sha256"] == get_catalog(force_reload=True).sha256
    assert len(payload["cards"]) == 989


def test_same_cost_and_tags_still_receive_different_ids() -> None:
    catalog = build_catalog({
        "Alpha": _meta("Alpha"),
        "Beta": _meta("Beta"),
    })
    assert catalog.id_for_name("Alpha") != catalog.id_for_name("Beta")
    assert catalog.id_for_name("Missing") == 0
    assert catalog.metadata_vector_for_id(0) == [0.0] * METADATA_DIM


def test_capacity_rejects_the_2048th_named_card() -> None:
    cards = {f"Card {index}": _meta(f"Card {index}") for index in range(CARD_CAPACITY - 1)}
    assert len(build_catalog(cards).entries) == CARD_CAPACITY - 1
    cards["One Too Many"] = _meta("One Too Many")
    with pytest.raises(CardCatalogError, match="capacity"):
        build_catalog(cards)


def test_structured_behavior_overrides_description_fallback() -> None:
    described = resolve_behavior({"description": "Increase your plant production 1 step."})
    assert described["source"] == "description"
    assert described["immediate"]["production"]["plants"] == 1.0
    structured = {
        "name": "Structured",
        "type": "automated",
        "tags": ["plant"],
        "cost": 9,
        "description": "Increase your plant production 9 steps.",
        "behavior": {"immediate": {"production": {"plants": 3}}},
    }
    behavior = resolve_behavior(structured)
    assert behavior["source"] == "structured"
    assert behavior["immediate"]["production"]["plants"] == 3.0
    vector = metadata_vector(structured)
    assert len(vector) == METADATA_DIM
    assert vector[IMMEDIATE.start + 3] == round(3.0 / 8.0, 6)
    assert vector[FLAGS.start + 5] == 0.0
    fallback = metadata_vector({"name": "Adapted Lichen", "type": "automated", "tags": ["plant"], "cost": 9, "description": "Increase your plant production 1 step."})
    assert fallback[FLAGS.start + 5] == 1.0
    assert fallback[IMMEDIATE.start + 3] == round(1.0 / 8.0, 6)
    assert len(vector[ACTION_EFFECTS]) == 28


def test_every_four_card_subset_has_a_distinct_mask() -> None:
    cards = [{"name": f"Card {index}"} for index in range(4)]
    catalog = build_catalog({card["name"]: _meta(card["name"]) for card in cards})
    masks = []
    for pattern in iter_subset_masks(4):
        names = [cards[index]["name"] for index, selected in enumerate(pattern) if selected]
        descriptor = {"family": "card_subset", "decoded_action": {"type": "card", "cards": names}}
        _, mask = bind_action_card_mask(cards, [descriptor], hand_limit=24, catalog=catalog)
        masks.append(tuple(bool(item) for item in mask[0]))
    assert len(masks) == 16
    assert len(set(masks)) == 16
    assert masks[0] == (False, False, False, False)


def test_reordering_candidates_and_masks_preserves_the_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    cards = [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}, {"name": "Delta"}]
    catalog = build_catalog({card["name"]: _meta(card["name"], cost=index + 1) for index, card in enumerate(cards)})
    selected = ["Beta", "Delta"]
    descriptor = {"family": "card_subset", "decoded_action": {"cards": selected}}
    ids, mask = bind_action_card_mask(cards, [descriptor], hand_limit=24, catalog=catalog)
    reordered = [cards[2], cards[0], cards[3], cards[1]]
    reordered_ids, reordered_mask = bind_action_card_mask(reordered, [descriptor], hand_limit=24, catalog=catalog)
    assert set(int(ids[index]) for index, selected_bit in enumerate(mask[0]) if selected_bit) == set(
        int(reordered_ids[index]) for index, selected_bit in enumerate(reordered_mask[0]) if selected_bit
    )
    network = _card_network()
    network.eval()
    left = _forward_subset(network, cards, ids, mask)
    right = _forward_subset(network, reordered, reordered_ids, reordered_mask)
    assert torch.allclose(left, right)


def test_play_and_subset_references_resolve_and_omission_fails() -> None:
    cards = [{"name": f"Card {index}"} for index in range(25)]
    catalog = build_catalog({card["name"]: _meta(card["name"]) for card in cards})
    kept = cards[:24]
    play = {"family": "play_card", "card_name": "Card 0", "decoded_action": {"card": "Card 0"}}
    _, play_mask = bind_action_card_mask(kept, [play], hand_limit=24, catalog=catalog)
    assert int(play_mask.sum()) == 1
    assert bool(play_mask[0, 0])
    buy_nothing = {"family": "card_subset", "decoded_action": {"type": "card", "cards": []}}
    other = {"family": "pass", "decoded_action": {"type": "pass"}}
    _, mixed = bind_action_card_mask(kept, [buy_nothing, other], hand_limit=24, catalog=catalog)
    assert not bool(mixed[0].any())
    assert not bool(mixed[1].any())
    omitted = {"family": "play_card", "decoded_action": {"card": "Card 24"}}
    with pytest.raises(CardEncodingError, match="24-card"):
        bind_action_card_mask(cards, [omitted], hand_limit=24, catalog=catalog)


def test_action_menu_includes_blue_card_actions_as_prompt_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    project = {"name": "Solar Wind Power"}
    blue = {"name": "Equatorial Magnetizer", "type": "active"}
    state = {
        "cardsInHand": [project],
        "waitingFor": {
            "type": "or",
            "options": [
                {"type": "projectCard", "title": "Play project card", "cards": [project]},
                {
                    "type": "card",
                    "title": "Take action",
                    "selectBlueCardAction": True,
                    "cards": [blue],
                    "min": 1,
                    "max": 1,
                },
                {"type": "card", "title": "Standard projects", "cards": [{"name": "Power Plant"}]},
            ],
        },
    }
    names = [card["name"] for card in StateEncoder()._get_candidate_hand_cards(state)]
    assert names[:2] == ["Solar Wind Power", "Equatorial Magnetizer"]
    assert "Power Plant" not in names
    catalog = build_catalog({
        "Solar Wind Power": _meta("Solar Wind Power"),
        "Equatorial Magnetizer": _meta("Equatorial Magnetizer"),
    })
    descriptor = {
        "family": "play_card",
        "card_name": "Equatorial Magnetizer",
        "decoded_action": {
            "type": "or",
            "index": 1,
            "response": {"type": "card", "cards": ["Equatorial Magnetizer"]},
        },
    }
    _, mask = bind_action_card_mask(
        StateEncoder()._get_candidate_hand_cards(state),
        [descriptor],
        hand_limit=24,
        catalog=catalog,
    )
    assert bool(mask[0, names.index("Equatorial Magnetizer")])


def test_duplicate_lunar_beam_payloads_collapse_and_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    payload = {"type": "card", "card": "Lunar Beam", "payment": {"megaCredits": 6, "steel": 2, "titanium": 0}}
    shuffled = {"payment": {"titanium": 0, "steel": 2, "megaCredits": 6}, "card": "Lunar Beam", "type": "card"}
    kept, aliases = canonicalize_legal_actions([
        Action(0, "play_card", payload, "Lunar Beam"),
        Action(1, "play_card", shuffled, "Lunar Beam duplicate"),
    ])
    assert len(kept) == 1
    assert kept[0].action_id == 0
    assert aliases == [{"dropped_action_id": 1, "kept_action_id": 0}]
    posted = json.loads(canonical_payload_text(kept[0].payload))
    assert posted == normalize_payload(payload)
    assert canonical_payload_text(posted) == canonical_payload_text(payload)

    decoder = ActionDecoder()
    decoder._get_available_action_indices = lambda state: [0, 1]
    decoder.decode_action = lambda action_index, state: dict(shuffled if action_index else payload)
    decoder._semantic_family = lambda *args, **kwargs: "play_card"
    decoder._descriptor_labels = lambda *args, **kwargs: {
        "label": "Lunar Beam",
        "card_name": "Lunar Beam",
        "project_name": "",
        "award_name": "",
        "milestone_name": "",
    }
    legal = decoder.enumerate_legal_actions({
        "game": {"phase": "action"},
        "waitingFor": {"type": "projectCard", "title": "Play a card"},
        "thisPlayer": {},
    })
    assert legal.status == "active"
    assert [action.action_id for action in legal.actions] == [0]
    assert decoder.last_canonical_aliases[0]["dropped_action_id"] == 1


def test_v4_preserves_the_first_49_action_features(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"thisPlayer": {"megaCredits": 20}, "game": {"generation": 4}, "waitingFor": {}}
    label = {"label": "Pass", "card_name": "", "project_name": "", "award_name": "", "milestone_name": ""}
    legacy = ActionDecoder()._build_action_token(state, 900, "pass", label, {"type": "pass"}, None)
    monkeypatch.setenv("TFM_RL_V4", "1")
    card_aware = ActionDecoder()._build_action_token(state, 900, "pass", label, {"type": "pass"}, None)
    assert np.allclose(legacy[1:50], card_aware[1:50])
    assert np.allclose(card_aware[50:], 0.0)


def test_teacher_draft_scores_follow_card_quality_not_position(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    teacher = HeuristicTeacherPolicy(seed=1, sample=False)
    player = {"megaCredits": 30, "cardCost": 3, "steel": 0, "titanium": 0}
    strong = {"name": "Strong", "cost": 4, "victoryPoints": 5, "tags": ["Science", "Building", "Space"]}
    weak = {"name": "Weak", "cost": 39, "victoryPoints": 0, "tags": []}
    state = {"thisPlayer": player, "waitingFor": {"title": "Select cards to buy", "cards": [strong, weak], "min": 0, "max": 1}}
    early = {"family": "card_subset", "action_position": 0, "decoded_action": {"cards": ["Weak"]}}
    late = {"family": "card_subset", "action_position": 9, "decoded_action": {"cards": ["Strong"]}}
    same_late = {"family": "card_subset", "action_position": 0, "decoded_action": {"cards": ["Strong"]}}
    weak_score, _, _ = teacher._score_card_subset(state, early)
    strong_score, _, _ = teacher._score_card_subset(state, late)
    strong_again, _, _ = teacher._score_card_subset(state, same_late)
    assert strong_score == strong_again
    assert strong_score > weak_score
    threshold = min(0.90, 0.60 + 0.05 * 3)
    expected = _card_quality(strong, player) - threshold - 0.02 * max(0.0, 14 - (30 - 3))
    assert strong_score == pytest.approx(expected)
    free_state = {"thisPlayer": player, "waitingFor": {"title": "Draft", "cards": [strong], "min": 1, "max": 1}}
    free_score, _, _ = teacher._score_card_subset(
        free_state,
        {"family": "card_subset", "action_position": 4, "decoded_action": {"cards": ["Strong"]}},
    )
    assert free_score == pytest.approx(_card_quality(strong, player) - 0.02 * max(0.0, 14 - 30))


def test_milestone_teacher_ranks_identity_against_cards_and_ignores_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 30},
        "players": [{"name": "A1", "color": "red"}, {"name": "A2", "color": "blue"}],
        "game": {
            "generation": 8,
            "milestones": [
                {"name": "Gardener", "scores": [{"color": "red", "score": 3}, {"color": "blue", "score": 3}]},
                {"name": "Builder", "scores": [{"color": "red", "score": 4}, {"color": "blue", "score": 1}]},
            ],
        },
        "waitingFor": {"cards": [
            {"name": "Strong", "calculatedCost": 0, "victoryPoints": 8},
            {"name": "Weak", "calculatedCost": 40, "victoryPoints": 0},
        ]},
    }
    descriptors = [
        {"action_index": 650, "action_position": 0, "family": "claim_milestone", "label": "Gardener", "milestone_name": "Gardener", "decoded_action": {"type": "option"}},
        {"action_index": 651, "action_position": 1, "family": "claim_milestone", "label": "Builder", "milestone_name": "Builder", "decoded_action": {"type": "option"}},
        {"action_index": 1, "action_position": 2, "family": "play_card", "label": "Strong", "card_name": "Strong", "decoded_action": {"type": "projectCard", "card": "Strong"}},
        {"action_index": 2, "action_position": 3, "family": "play_card", "label": "Weak", "card_name": "Weak", "decoded_action": {"type": "projectCard", "card": "Weak"}},
    ]
    teacher = HeuristicTeacherPolicy(seed=7, sample=False)
    first = teacher.score_actions(state, descriptors)
    second = teacher.score_actions(state, list(reversed(descriptors)))
    scores = {row.action_index: row.score for row in first.actions}
    reversed_scores = {row.action_index: row.score for row in second.actions}
    assert scores[650] > scores[651]
    assert scores[1] > scores[650] > scores[2]
    assert scores == reversed_scores


def test_v4_milestone_and_placement_tails_expose_decision_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    decoder = ActionDecoder()
    state = {
        "thisPlayer": {"name": "A1", "color": "red", "megaCredits": 16},
        "game": {"generation": 7, "milestones": [{
            "name": "Gardener",
            "scores": [{"color": "red", "score": 4}, {"color": "blue", "score": 2}],
        }]},
        "waitingFor": {},
    }
    milestone = decoder._v3_named_action_features(state, "claim_milestone", "Gardener")
    assert len(milestone) == 14
    assert milestone[6] > 0.0
    assert milestone[9] == 1.0
    city = decoder._v4_family_tail(state, "select_space", {}, {}, {"intent": "city"}, [], "")
    greenery = decoder._v4_family_tail(state, "select_space", {}, {}, {"intent": "greenery"}, [], "")
    ocean = decoder._v4_family_tail(state, "select_space", {}, {}, {"intent": "ocean"}, [], "")
    assert city[-2:] == [1.0, 0.0]
    assert greenery[-2:] == [0.0, 1.0]
    assert ocean[-2:] == [0.0, 0.0]


def test_v4_award_features_normalize_identity_and_funded_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    monkeypatch.setenv("TFM_RL_V3", "1")
    state = {
        "thisPlayer": {"name": "Alice", "color": "red", "megaCredits": 20},
        "game": {
            "generation": 9,
            "awards": [
                {
                    "name": "Landlord",
                    "funded_by": "Bob",
                    "scores": [
                        {"playerColor": "red", "playerScore": 6},
                        {"playerColor": "blue", "playerScore": 4},
                    ],
                },
                {
                    "name": "Banker",
                    "scores": [
                        {"name": "Alice", "score": 6},
                        {"name": "Bob", "score": 4},
                    ],
                },
            ],
        },
    }
    decoder = ActionDecoder()
    token = decoder._build_action_token(
        state,
        600,
        "fund_award",
        {"label": "Banker", "award_name": "Banker"},
        None,
    )
    tail = token[50:64]
    assert tail[6] == pytest.approx(6 / 20)
    assert tail[7] == pytest.approx(4 / 20)
    assert tail[9] == pytest.approx(1.0)
    assert tail[10] == pytest.approx(14 / 20)
    assert tail[11] == pytest.approx(1.0)

    world_features = StateEncoder()._award_token_features(state["game"], state["thisPlayer"], "Landlord")
    assert world_features[2] == pytest.approx(1.0)


def test_legacy_bundle_is_rejected_by_v4_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    with pytest.raises(ValueError, match="planner.card_aware.v1"):
        ensure_bundle({
            "world_tokens": np.zeros((1, 64), dtype=np.float32),
            "world_token_types": np.zeros((1,), dtype=np.int64),
            "world_mask": np.ones((1,), dtype=np.bool_),
            "hand_tokens": np.zeros((0, 64), dtype=np.float32),
            "hand_mask": np.zeros((0,), dtype=np.bool_),
            "action_tokens": np.ones((1, 64), dtype=np.float32),
            "action_mask": np.ones((1,), dtype=np.bool_),
            "action_indices": np.zeros((1,), dtype=np.int64),
            "action_positions": np.zeros((1,), dtype=np.int64),
            "global_scalars": np.zeros((16,), dtype=np.float32),
        })
    with pytest.raises(CardCatalogError, match="incompatible checkpoint"):
        validate_checkpoint_catalog({"experiment_version": "tfm-rl-v2"})
    with pytest.raises(CardCatalogError, match="catalog hash"):
        validate_checkpoint_catalog({"experiment_version": "tfm-rl-v4", "card_catalog_sha256": "deadbeef"})


def test_card_modules_receive_gradients_and_ignore_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    network = _card_network()
    network.train()
    with torch.no_grad():
        network.card_metadata_table[1] = 1.0
        network.card_identity.weight[1].fill_(0.2)
        network.card_identity.weight[2].fill_(-0.4)
    batch = _batch(hand=2, hand_ids=[1, 0], action_mask=[[True, False]], hand_values=[1.0, 50.0])
    first = network(batch)["policy_logits"].detach().clone()
    changed = _batch(hand=2, hand_ids=[2, 0], action_mask=[[True, False]], hand_values=[1.0, 50.0])
    second = network(changed)["policy_logits"]
    assert not torch.allclose(first, second)
    padded = _batch(hand=2, hand_ids=[1, 7], action_mask=[[True, False]], hand_values=[1.0, -25.0], hand_mask=[True, False])
    padded_logits = network(padded)["policy_logits"]
    original = network(_batch(hand=2, hand_ids=[1, 0], action_mask=[[True, False]], hand_values=[1.0, 50.0], hand_mask=[True, False]))["policy_logits"]
    assert torch.allclose(padded_logits, original)
    buy_nothing = network(_batch(hand=1, hand_ids=[1], action_mask=[[False]], hand_values=[1.0], family_column=11))["policy_logits"]
    other = network(_batch(hand=1, hand_ids=[1], action_mask=[[False]], hand_values=[1.0], family_column=13))["policy_logits"]
    assert not torch.allclose(buy_nothing, other)
    loss = network(batch)["policy_logits"].sum()
    loss.backward()
    assert network.card_identity.weight.grad[1].abs().sum() > 0
    assert network.card_metadata_projection.weight.grad.abs().sum() > 0
    assert network.card_set_projection.weight.grad.abs().sum() > 0


def test_warm_start_requires_the_h512_source_shape() -> None:
    with pytest.raises(RuntimeError, match="hidden_size"):
        _require_source_architecture({"hidden_size": 256, **{key: value for key, value in REQUIRED_ARCHITECTURE.items() if key != "hidden_size"}})


def test_partial_warm_start_reuses_matching_weights_and_resets_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOTSTRAP_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("RESUME_TRAINING", raising=False)
    monkeypatch.setenv("TFM_RL_V4", "0")
    monkeypatch.setenv("TFM_RL_V3", "0")
    monkeypatch.setenv("TFM_RL_V4_ROOT", str(tmp_path))
    source_network = TerraformingMarsNetwork(AgentConfig(
        hidden_size=512,
        recurrent_size=128,
        transformer_heads=4,
        transformer_layers=3,
        transformer_dropout=0.0,
        planner_token_dim=64,
    ))
    source_path = tmp_path / "bc_best.pth"
    source_state = source_network.state_dict()
    torch.save({
        "experiment_version": "tfm-rl-v2",
        "config": {
            "hidden_size": 512,
            "recurrent_size": 128,
            "transformer_layers": 3,
            "transformer_heads": 4,
            "planner_token_dim": 64,
        },
        "network_state_dict": source_state,
        "policy_version": 12,
        "games_played": 40,
    }, source_path)
    from training.v4_warm_start import warm_start
    report = warm_start(str(source_path), str(tmp_path / "bootstrap" / "warm_start.pth"))
    loaded = torch.load(tmp_path / "bootstrap" / "warm_start.pth", map_location="cpu", weights_only=False)
    assert loaded["experiment_version"] == "tfm-rl-v4"
    assert loaded["policy_version"] == 0
    assert loaded["games_played"] == 0
    assert loaded["wins"] == 0
    assert loaded["optimizer_state_dict"]["state"] == {}
    assert loaded["card_catalog_sha256"] == get_catalog().sha256
    assert torch.equal(loaded["network_state_dict"]["world_projection.weight"], source_state["world_projection.weight"])
    assert "card_identity.weight" not in source_state
    assert "card_identity.weight" in report["initialized_parameters"]
    assert "world_projection.weight" in report["reused_parameters"]
    assert report["optimizer_reset"] is True
    assert report["policy_version_reset"] is True
    assert report["training_statistics_reset"] is True


def test_human_reencode_merges_mass_and_invalidates_missing_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    payload = {"type": "card", "cards": ["Alpha"]}
    duplicate = {"cards": ["Alpha"], "type": "card"}
    events = [{
        "episode_id": "finished",
        "player_state": {"waitingFor": {"type": "card"}, "game": {}},
        "response": payload,
        "action_descriptors": [
            {"decoded_action": payload, "action_index": 0},
            {"decoded_action": duplicate, "action_index": 1},
        ],
        "probabilities": [0.25, 0.75],
    }, {
        "episode_id": "open",
        "player_state": {"waitingFor": {"type": "card"}, "game": {}},
        "response": payload,
        "action_descriptors": [{"decoded_action": payload, "action_index": 0}],
        "probabilities": [1.0],
    }]

    class _Decoder:
        last_canonical_aliases = []

        def enumerate_legal_actions(self, state):
            from models.action_contract import LegalActionSet
            return LegalActionSet(status="active", actions=[Action(1000, "card_subset", normalize_payload(payload), "Alpha")])

        def _build_action_descriptor_from_action(self, action, position, state):
            return {"action_index": action.action_id, "family": action.family, "decoded_action": dict(action.payload)}

    samples = reencode_human_events(
        events,
        [{"episode_id": "finished", "completed": True, "value_target": 4.5}],
        decoder=_Decoder(),
        encode=lambda state, descriptors: {"planner_schema_version": "planner.card_aware.v1"},
    )
    assert samples[0]["value_target_valid"] is True
    assert samples[0]["value_target"] == 4.5
    assert samples[0]["teacher_probabilities"] == [1.0]
    assert samples[1]["value_target_valid"] is False
    merged, probabilities, aliases = merge_equivalent_action_mass(events[0]["action_descriptors"], events[0]["probabilities"])
    assert len(merged) == 1
    assert probabilities == [1.0]
    assert aliases[0]["kept_position"] == 0


def test_ppo_stays_blocked_until_every_held_out_gate_passes() -> None:
    checkpoint_sha = "a" * 64
    report = {
        "selected_validation_gate_passed": True,
        "selected_checkpoint_sha256": checkpoint_sha,
        "placement_gate": {"mode": "top1", "diagnostic_qualified": False},
        "test": {
            "teacher_top1": 0.85,
            "teacher_top3": 0.97,
            "family_top1": {family: threshold for family, threshold in {
                "play_card": 0.85,
                "card_subset": 0.90,
                "select_option": 0.95,
                "select_space": 0.87,
                "claim_milestone": 0.90,
                "fund_award": 0.80,
                "select_payment": 0.90,
            }.items()},
            "family_top3": {"select_space": 0.97},
            "family_counts": {"select_space": 100, "claim_milestone": 100, "fund_award": 100},
        },
        "human_evaluation": {"top3": 0.80, "samples": 20, "held_out_games": ["g1", "g2"]},
        "duplicate_executable_actions": 0,
        "unresolved_known_card_references": 0,
        "smoke": {"server_rejected_actions": None},
    }
    passed, reasons = evaluate_ppo_gate(report)
    assert not passed
    assert any("smoke" in reason for reason in reasons)
    report["smoke"] = {
        "completed": True,
        "server_rejected_actions": 0,
        "checkpoint_sha256": checkpoint_sha,
        "seed": 910001,
        "configuration": {
            "baseline": "teacher",
            "stage": 1,
            "candidate_seat": 0,
            "deterministic_actions": True,
            "ppo_enabled": False,
        },
    }
    report["smoke"]["checkpoint_sha256"] = "b" * 64
    passed, reasons = evaluate_ppo_gate(report)
    assert not passed
    assert any("does not match" in reason for reason in reasons)
    report["smoke"]["checkpoint_sha256"] = checkpoint_sha
    passed, reasons = evaluate_ppo_gate(report)
    assert passed
    assert reasons == []


def test_family_metrics_use_teacher_argmax_not_sampled_action() -> None:
    sample = {
        "chosen_action_position": 1,
        "target_action_position": 0,
        "action_descriptors": [
            {"family": "play_card"},
            {"family": "claim_milestone"},
        ],
    }
    assert _target_family(sample, expected_position=0) == "play_card"


def test_human_games_are_held_out_whole_and_keep_unit_weight() -> None:
    samples = [
        {"game_id": f"game-{game}", "episode_id": f"episode-{game}", "sample_weight": 4.0, "row": row}
        for game in range(10)
        for row in range(3)
    ]
    first, provenance = partition_human_samples(samples)
    second, repeated = partition_human_samples(samples)
    assert provenance == repeated
    assert len(provenance["held_out_games"]) == 2
    assert len(provenance["training_games"]) == 8
    assert all(item["sample_weight"] == 1.0 for item in first)
    by_game = {}
    for item in first:
        by_game.setdefault(item["game_id"], set()).add(item["human_split"])
    assert all(len(splits) == 1 for splits in by_game.values())
    assert [item["human_split"] for item in first] == [item["human_split"] for item in second]


def test_gate_aware_selection_prefers_family_clearance_over_higher_average() -> None:
    def metrics(space: float, award: float, milestone: float, aggregate: float) -> dict:
        return {
            "teacher_top1": aggregate,
            "teacher_top3": 0.99,
            "family_top1": {
                "play_card": 0.90,
                "card_subset": 0.93,
                "select_option": 0.99,
                "select_space": space,
                "claim_milestone": milestone,
                "fund_award": award,
                "select_payment": 0.97,
            },
            "family_top3": {"select_space": 0.99},
            "family_counts": {"select_space": 100, "claim_milestone": 100, "fund_award": 100},
        }
    clearing = validation_candidate_status(metrics(0.88, 0.81, 0.91, 0.89), "top1")
    average_only = validation_candidate_status(metrics(0.84, 0.79, 0.89, 0.91), "top1")
    assert clearing["passed"] is True
    assert average_only["passed"] is False
    assert tuple(clearing["selection_key"]) > tuple(average_only["selection_key"])


def test_placement_top3_gate_requires_diagnostic_and_near_tie_evidence() -> None:
    descriptor = lambda space_id: {
        "family": "select_space",
        "label": space_id,
        "decoded_action": {"type": "space", "spaceId": space_id},
        "space_features": {"space_id": space_id, "intent": "city", "total_value": 0.5},
    }
    rows = []
    for index in range(100):
        rows.append({
            "sample_id": str(index),
            "target_rank": 1 if index < 80 else (2 if index < 97 else 4),
            "candidate_count": 6,
            "logit_margin": 0.1,
            "teacher_score_gap": 0.1 if index < 97 else 0.5,
            "target_descriptor": descriptor("target"),
            "predicted_descriptor": descriptor("predicted"),
        })
    report = _placement_report(rows)
    assert report["tile"]["top3"] == pytest.approx(0.97)
    assert report["near_tie_rate"] == pytest.approx(17 / 20)
    assert report["qualified_for_top3_gate"] is True


def _card_network() -> TerraformingMarsNetwork:
    return TerraformingMarsNetwork(AgentConfig(
        hidden_size=64,
        recurrent_size=32,
        transformer_heads=4,
        transformer_layers=1,
        transformer_dropout=0.0,
        planner_token_dim=64,
        planner_hand_limit=24,
    ))


def _batch(
    hand: int,
    hand_ids: list[int],
    action_mask: list[list[bool]],
    hand_values: list[float],
    family_column: int = 11,
    hand_mask: list[bool] | None = None,
) -> dict:
    token = torch.zeros((1, 1, 64))
    token[0, 0, family_column + 1] = 1.0
    hand_tokens = torch.zeros((1, hand, 64))
    for index, value in enumerate(hand_values):
        hand_tokens[0, index, 1] = value
    visible = hand_mask if hand_mask is not None else [True] * hand
    return {
        "world_tokens": torch.zeros((1, 1, 64)),
        "world_token_types": torch.zeros((1, 1), dtype=torch.long),
        "world_mask": torch.ones((1, 1), dtype=torch.bool),
        "world_card_ids": torch.zeros((1, 1), dtype=torch.long),
        "hand_tokens": hand_tokens,
        "hand_mask": torch.tensor(visible, dtype=torch.bool).view(1, hand),
        "hand_card_ids": torch.tensor(hand_ids, dtype=torch.long).view(1, hand),
        "action_tokens": token,
        "action_mask": torch.ones((1, 1), dtype=torch.bool),
        "action_card_mask": torch.tensor(action_mask, dtype=torch.bool).view(1, 1, hand),
        "global_scalars": torch.zeros((1, 16)),
        "terminal_mask": torch.zeros((1,), dtype=torch.bool),
    }


def _forward_subset(network, cards, ids, mask) -> torch.Tensor:
    hand = len(cards)
    hand_tokens = torch.zeros((1, hand, 64))
    for index, card in enumerate(cards):
        hand_tokens[0, index, 1] = float(sum(ord(character) for character in card["name"]))
    batch = {
        "world_tokens": torch.zeros((1, 1, 64)),
        "world_token_types": torch.zeros((1, 1), dtype=torch.long),
        "world_mask": torch.ones((1, 1), dtype=torch.bool),
        "world_card_ids": torch.zeros((1, 1), dtype=torch.long),
        "hand_tokens": hand_tokens,
        "hand_mask": torch.ones((1, hand), dtype=torch.bool),
        "hand_card_ids": torch.tensor(ids, dtype=torch.long).view(1, hand),
        "action_tokens": torch.zeros((1, 1, 64)),
        "action_mask": torch.ones((1, 1), dtype=torch.bool),
        "action_card_mask": torch.tensor(mask, dtype=torch.bool).view(1, 1, hand),
        "global_scalars": torch.zeros((1, 16)),
        "terminal_mask": torch.zeros((1,), dtype=torch.bool),
    }
    return network(batch)["policy_logits"]


def test_v4_init_checkpoint_copies_warm_start_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFM_RL_V4", "1")
    monkeypatch.setenv("TFM_RL_V3", "1")
    source = TerraformingMarsNetwork(AgentConfig(
        hidden_size=64,
        recurrent_size=16,
        transformer_heads=4,
        transformer_layers=1,
        planner_token_dim=64,
    ))
    with torch.no_grad():
        source.world_projection.weight.fill_(0.25)
    path = tmp_path / "warm_start.pth"
    torch.save({
        "experiment_version": "tfm-rl-v4",
        "card_catalog_sha256": get_catalog().sha256,
        "network_state_dict": source.state_dict(),
    }, path)
    fresh = TerraformingMarsNetwork(AgentConfig(
        hidden_size=64,
        recurrent_size=16,
        transformer_heads=4,
        transformer_layers=1,
        planner_token_dim=64,
    ))
    from training.v4_pretrain import load_v4_init_weights
    assert load_v4_init_weights(fresh, str(path)) == str(path.resolve())
    assert torch.equal(fresh.world_projection.weight, source.world_projection.weight)
    with pytest.raises(CardCatalogError, match="incompatible checkpoint"):
        torch.save({"experiment_version": "tfm-rl-v2", "network_state_dict": source.state_dict()}, path)
        load_v4_init_weights(fresh, str(path))


def test_v4_collection_pins_base_game_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAME_OPTIONS_FILE", "/app/game_options.v3_stage1.json")
    from training.v4_collect_teacher import pin_stage_options
    import os
    pinned = pin_stage_options(0)
    assert pinned.endswith("game_options.v2_stage0.json")
    assert os.environ["GAME_OPTIONS_FILE"] == pinned
    assert Path(pinned).is_file()
