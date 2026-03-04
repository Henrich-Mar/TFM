use crate::domain::FullPayload;

pub fn encode_state_impl(json_str: &str, turn_action_count: i32, state_size: usize) -> Result<Vec<f32>, String> {
    let payload: FullPayload = serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let target_size = state_size.max(1);
    let mut features = Vec::with_capacity(target_size);

    // 1. Global parameters
    if let Some(game) = &payload.game {
        let oxygen_level = game.oxygen_level.unwrap_or(0);
        let oxygen_index = oxygen_level / 2;
        features.push((oxygen_index as f32) / 7.0);

        let temp_level = game.temperature.unwrap_or(-30);
        let temp_index = (temp_level + 30) / 2;
        features.push((temp_index as f32) / 19.0);

        let venus_level = game.venus_scale_level.unwrap_or(0);
        let venus_index = venus_level / 2;
        features.push((venus_index as f32) / 15.0);

        let generation = game.generation.unwrap_or(1);
        features.push((generation as f32) / 14.0);

        if let Some(moon) = &game.moon {
            features.push((moon.logistics_rate.unwrap_or(0) as f32) / 8.0);
            features.push((moon.mining_rate.unwrap_or(0) as f32) / 8.0);
            features.push((moon.habitat_rate.unwrap_or(0) as f32) / 8.0);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    } else {
        features.extend_from_slice(&[0.0; 7]);
    }

    // 2. Player resources
    if let Some(player) = &payload.this_player {
        let max_res = [300.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 100.0];
        let tr = (player.terraform_rating.unwrap_or(20.0) - 20.0).max(0.0);
        let vp = player
            .victory_points_breakdown
            .as_ref()
            .map(|v| v.total.unwrap_or(0.0))
            .unwrap_or(0.0);

        let res = [
            player.mega_credits.unwrap_or(0.0),
            player.steel.unwrap_or(0.0),
            player.titanium.unwrap_or(0.0),
            player.plants.unwrap_or(0.0),
            player.energy.unwrap_or(0.0),
            player.heat.unwrap_or(0.0),
            tr,
            vp,
        ];

        for i in 0..8 {
            features.push((res[i] / max_res[i]).min(1.0));
        }
    } else {
        features.extend_from_slice(&[0.0; 8]);
    }

    // 3. Player production
    if let Some(player) = &payload.this_player {
        let max_prod = [100.0, 30.0, 30.0, 30.0, 30.0, 30.0];
        let prod = [
            player.mega_credit_production.unwrap_or(0.0),
            player.steel_production.unwrap_or(0.0),
            player.titanium_production.unwrap_or(0.0),
            player.plant_production.unwrap_or(0.0),
            player.energy_production.unwrap_or(0.0),
            player.heat_production.unwrap_or(0.0),
        ];

        for i in 0..6 {
            features.push((prod[i] + 10.0) / (max_prod[i] + 10.0));
        }
    } else {
        features.extend_from_slice(&[0.0; 6]);
    }

    // 4. Action-slot context signal (first vs second action in a turn).
    let action_slot = (turn_action_count.max(0) as f32 / 3.0).min(1.0);
    features.push(action_slot);

    if features.len() > target_size {
        features.truncate(target_size);
    } else {
        features.resize(target_size, 0.0);
    }
    Ok(features)
}
