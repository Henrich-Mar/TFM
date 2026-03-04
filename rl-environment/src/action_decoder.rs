use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::domain::PlayerState;
use crate::utils::{get_card_cost, get_card_tags, get_card_vp, to_f32};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CardSelectionPayload {
    pub cards: Vec<Value>,
    pub min_cards: Option<usize>,
    pub max_cards: Option<usize>,
    pub player: Option<Value>,
    pub player_state: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StartupCandidate {
    pub index: usize,
    pub score: f32,
    pub corp: String,
    pub prelude: Vec<String>,
    pub ceo: Vec<String>,
    pub project: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StartupRankPayload {
    pub candidates: Vec<StartupCandidate>,
}

fn player_from_value(player: &Value) -> PlayerState {
    serde_json::from_value(player.clone()).unwrap_or_default()
}

pub fn estimate_affordability(player: Option<&PlayerState>, card: &Value) -> f32 {
    let mut mc = 0.0;
    let mut steel = 0.0;
    let mut titanium = 0.0;
    let mut steel_value = 2.0;
    let mut titanium_value = 3.0;

    if let Some(p) = player {
        mc = p.mega_credits.unwrap_or(0.0);
        steel = p.steel.unwrap_or(0.0);
        titanium = p.titanium.unwrap_or(0.0);
        steel_value = p.steel_value.unwrap_or(2.0);
        titanium_value = p.titanium_value.unwrap_or(3.0);
    }

    let tags = get_card_tags(card.get("tags"));
    let mut purchasing_power = mc;
    if tags.get("Building").copied().unwrap_or(0) > 0 {
        purchasing_power += steel * steel_value;
    }
    if tags.get("Space").copied().unwrap_or(0) > 0 {
        purchasing_power += titanium * titanium_value;
    }

    let cost = get_card_cost(card);
    if cost <= 0.0 {
        return 1.0;
    }
    if purchasing_power >= cost {
        return 1.0;
    }

    let diff = (cost - purchasing_power) / 20.0;
    if diff >= 1.0 {
        0.0
    } else {
        1.0 - diff
    }
}

pub fn can_afford_card(player: &PlayerState, card: &Value) -> bool {
    let cost = get_card_cost(card);
    if cost <= 0.0 {
        return true;
    }

    let mut purchasing_power = player.mega_credits.unwrap_or(0.0);
    let tags = get_card_tags(card.get("tags"));

    if tags.get("Building").copied().unwrap_or(0) > 0 {
        purchasing_power += player.steel.unwrap_or(0.0) * player.steel_value.unwrap_or(2.0);
    }
    if tags.get("Space").copied().unwrap_or(0) > 0 {
        purchasing_power += player.titanium.unwrap_or(0.0) * player.titanium_value.unwrap_or(3.0);
    }

    // Helion: can pay with heat as money.
    let corp_name = player
        .corporation
        .as_ref()
        .and_then(|c| c.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    if corp_name.contains("helion") {
        purchasing_power += player.heat.unwrap_or(0.0);
    }

    purchasing_power >= cost
}

pub fn can_afford_cards(player: &PlayerState, cards: &[Value]) -> Vec<bool> {
    cards.iter().map(|card| can_afford_card(player, card)).collect()
}

fn score_card_for_selection(card: &Value, player: Option<&PlayerState>) -> f32 {
    let cost = get_card_cost(card);
    let vp = get_card_vp(card);
    let tags = get_card_tags(card.get("tags"));

    let mut mc = 0.0;
    let mut steel = 0.0;
    let mut titanium = 0.0;
    let mut steel_value = 2.0;
    let mut titanium_value = 3.0;
    if let Some(p) = player {
        mc = p.mega_credits.unwrap_or(0.0);
        steel = p.steel.unwrap_or(0.0);
        titanium = p.titanium.unwrap_or(0.0);
        steel_value = p.steel_value.unwrap_or(2.0);
        titanium_value = p.titanium_value.unwrap_or(3.0);
    }

    let mut purchasing_power = mc;
    if tags.get("Building").copied().unwrap_or(0) > 0 {
        purchasing_power += steel * steel_value;
    }
    if tags.get("Space").copied().unwrap_or(0) > 0 {
        purchasing_power += titanium * titanium_value;
    }

    let affordable = if purchasing_power >= cost {
        1.0
    } else {
        (1.0 - ((cost - purchasing_power) / 20.0)).max(0.0)
    };
    let cheapness = (1.0 - (cost / 40.0).min(1.0)).max(0.0);

    let mut tag_score = 0.0;
    if tags.get("Science").copied().unwrap_or(0) > 0 {
        tag_score += 0.45;
    }
    if tags.get("Building").copied().unwrap_or(0) > 0 {
        tag_score += 0.35;
    }
    if tags.get("Space").copied().unwrap_or(0) > 0 {
        tag_score += 0.35;
    }
    if tags.get("Earth").copied().unwrap_or(0) > 0 {
        tag_score += 0.2;
    }
    if tags.get("Plant").copied().unwrap_or(0) > 0 {
        tag_score += 0.2;
    }

    let name = card.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let deterministic_tiebreak = ((name.chars().map(|ch| ch as i32).sum::<i32>() % 97) as f32) * 1e-5;
    (vp * 0.45) + (affordable * 1.4) + (cheapness * 0.8) + tag_score + deterministic_tiebreak
}

fn combinations(
    items: &[usize],
    pick_count: usize,
    start: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == pick_count {
        out.push(current.clone());
        return;
    }
    let remaining_needed = pick_count.saturating_sub(current.len());
    if items.len().saturating_sub(start) < remaining_needed {
        return;
    }
    let mut i = start;
    while i < items.len() {
        current.push(items[i]);
        combinations(items, pick_count, i + 1, current, out);
        current.pop();
        i += 1;
    }
}

pub fn enumerate_card_selection_combos(payload: &CardSelectionPayload, limit: usize) -> Vec<Vec<usize>> {
    if payload.cards.is_empty() {
        return Vec::new();
    }

    let player = if let Some(player_obj) = payload.player.as_ref() {
        Some(player_from_value(player_obj))
    } else if let Some(player_state) = payload.player_state.as_ref() {
        player_state
            .get("thisPlayer")
            .map(player_from_value)
            .or_else(|| Some(player_from_value(player_state)))
    } else {
        None
    };

    let mut enabled_indices = Vec::new();
    for (idx, card) in payload.cards.iter().enumerate() {
        let disabled = card.get("isDisabled").and_then(|v| v.as_bool()).unwrap_or(false);
        if !disabled {
            enabled_indices.push(idx);
        }
    }
    if enabled_indices.is_empty() {
        return Vec::new();
    }

    let min_cards = payload.min_cards.unwrap_or(0);
    let max_cards = payload.max_cards.unwrap_or(enabled_indices.len());
    let min_pick = min_cards.min(enabled_indices.len());
    let max_pick = max_cards.min(enabled_indices.len()).max(min_pick);
    if max_pick == 0 {
        return if min_pick == 0 { vec![Vec::new()] } else { Vec::new() };
    }

    let mut scored_indices: Vec<(usize, f32)> = enabled_indices
        .iter()
        .map(|idx| {
            (
                *idx,
                score_card_for_selection(&payload.cards[*idx], player.as_ref()),
            )
        })
        .collect();
    scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let candidate_budget = scored_indices
        .len()
        .min(std::cmp::max(12, std::cmp::min(max_pick, 14)));
    let mut candidate_indices: Vec<usize> = scored_indices
        .iter()
        .take(candidate_budget)
        .map(|item| item.0)
        .collect();
    candidate_indices.sort_unstable();

    let mut score_map: HashMap<usize, f32> = HashMap::new();
    for idx in &candidate_indices {
        score_map.insert(*idx, score_card_for_selection(&payload.cards[*idx], player.as_ref()));
    }

    let mut ranked: Vec<(f32, usize, Vec<usize>)> = Vec::new();
    if min_pick == 0 {
        ranked.push((0.0, 0, Vec::new()));
    }

    for pick_count in std::cmp::max(1, min_pick)..=max_pick {
        if pick_count > candidate_indices.len() {
            break;
        }
        let mut combos = Vec::new();
        combinations(
            &candidate_indices,
            pick_count,
            0,
            &mut Vec::new(),
            &mut combos,
        );
        for combo in combos {
            let mut combo_score = 0.0;
            for idx in &combo {
                combo_score += *score_map.get(idx).unwrap_or(&0.0);
            }
            combo_score += 0.03 * (pick_count as f32);
            ranked.push((combo_score, pick_count, combo));
        }
    }

    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut out = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let cap = std::cmp::max(1, limit);
    for (_, _, combo) in ranked {
        if combo.len() < min_pick || combo.len() > max_pick {
            continue;
        }
        if seen.insert(combo.clone()) {
            out.push(combo);
        }
        if out.len() >= cap {
            break;
        }
    }

    if !out.is_empty() {
        return out;
    }

    let fallback_target = std::cmp::max(min_pick, 1);
    let fallback_indices: Vec<usize> = scored_indices
        .iter()
        .take(std::cmp::min(fallback_target, scored_indices.len()))
        .map(|item| item.0)
        .collect();
    if !fallback_indices.is_empty() {
        return vec![fallback_indices];
    }
    if min_pick == 0 {
        vec![Vec::new()]
    } else {
        Vec::new()
    }
}

pub fn rank_startup_plans(payload: &StartupRankPayload, max_plans: usize) -> String {
    let mut ranked = payload.candidates.clone();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.index.cmp(&b.index))
    });

    let mut seen: HashSet<String> = HashSet::new();
    let mut selected: Vec<Value> = Vec::new();
    for candidate in ranked {
        let mut prelude = candidate.prelude.clone();
        let mut ceo = candidate.ceo.clone();
        let mut project = candidate.project.clone();
        prelude.sort();
        ceo.sort();
        project.sort();
        let signature = format!(
            "{}|{}|{}|{}",
            candidate.corp,
            prelude.join(","),
            ceo.join(","),
            project.join(",")
        );
        if seen.insert(signature) {
            selected.push(json!({
                "index": candidate.index,
                "score": candidate.score,
            }));
        }
        if selected.len() >= std::cmp::max(1, max_plans) {
            break;
        }
    }

    serde_json::to_string(&selected).unwrap_or_else(|_| "[]".to_string())
}

pub fn affordability_for_score(player: Option<&Value>, card: &Value) -> f32 {
    let mut mc = 0.0;
    let mut steel = 0.0;
    let mut titanium = 0.0;
    let mut steel_value = 2.0;
    let mut titanium_value = 3.0;
    if let Some(player_value) = player {
        mc = to_f32(player_value.get("megaCredits"), 0.0);
        steel = to_f32(player_value.get("steel"), 0.0);
        titanium = to_f32(player_value.get("titanium"), 0.0);
        steel_value = to_f32(player_value.get("steelValue"), 2.0);
        titanium_value = to_f32(player_value.get("titaniumValue"), 3.0);
    }

    let mut purchasing_power = mc;
    let tags = get_card_tags(card.get("tags"));
    if tags.get("Building").copied().unwrap_or(0) > 0 {
        purchasing_power += steel * steel_value;
    }
    if tags.get("Space").copied().unwrap_or(0) > 0 {
        purchasing_power += titanium * titanium_value;
    }

    let cost = get_card_cost(card);
    if cost <= 0.0 {
        return 1.0;
    }
    if purchasing_power >= cost {
        return 1.0;
    }
    (1.0 - ((cost - purchasing_power) / 20.0)).max(0.0)
}
