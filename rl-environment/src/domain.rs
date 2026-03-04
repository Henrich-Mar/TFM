use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GameState {
    pub oxygen_level: Option<i32>,
    pub temperature: Option<i32>,
    pub venus_scale_level: Option<i32>,
    pub generation: Option<i32>,
    pub moon: Option<MoonParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MoonParams {
    pub logistics_rate: Option<i32>,
    pub mining_rate: Option<i32>,
    pub habitat_rate: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PlayerState {
    pub id: Option<String>,
    pub name: Option<String>,
    pub color: Option<String>,
    pub mega_credits: Option<f32>,
    pub steel: Option<f32>,
    pub titanium: Option<f32>,
    pub plants: Option<f32>,
    pub energy: Option<f32>,
    pub heat: Option<f32>,
    pub terraform_rating: Option<f32>,
    pub mega_credit_production: Option<f32>,
    pub steel_production: Option<f32>,
    pub titanium_production: Option<f32>,
    pub plant_production: Option<f32>,
    pub energy_production: Option<f32>,
    pub heat_production: Option<f32>,
    pub steel_value: Option<f32>,
    pub titanium_value: Option<f32>,
    pub card_cost: Option<i32>,
    pub corporation: Option<Value>,
    pub victory_points_breakdown: Option<VpBreakdown>,
    pub tableau: Option<Vec<Value>>,
    pub cards_in_hand: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VpBreakdown {
    pub total: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FullPayload {
    pub game: Option<GameState>,
    pub this_player: Option<PlayerState>,
    pub players: Option<Vec<PlayerState>>,
    pub cards_in_hand: Option<Vec<Value>>,
    pub waiting_for: Option<Value>,
}

