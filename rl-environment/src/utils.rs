use std::collections::HashMap;

use serde_json::Value;

pub fn to_f32(value: Option<&Value>, default: f32) -> f32 {
    if let Some(v) = value {
        if let Some(n) = v.as_f64() {
            return n as f32;
        }
        if let Some(s) = v.as_str() {
            return s.parse::<f32>().unwrap_or(default);
        }
    }
    default
}

pub fn to_i32(value: Option<&Value>, default: i32) -> i32 {
    if let Some(v) = value {
        if let Some(n) = v.as_i64() {
            return n as i32;
        }
        if let Some(n) = v.as_f64() {
            return n as i32;
        }
        if let Some(s) = v.as_str() {
            return s.parse::<i32>().unwrap_or(default);
        }
    }
    default
}

pub fn normalize_tag_name(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut chars = trimmed.chars();
    let first = chars.next().unwrap_or_default();
    if first.is_lowercase() {
        let mut out = first.to_uppercase().collect::<String>();
        out.push_str(chars.as_str());
        return out;
    }
    trimmed.to_string()
}

pub fn normalize_resource_type(value: Option<&str>) -> String {
    let raw = value
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .replace('-', "")
        .replace('_', "")
        .replace(' ', "");
    if raw.is_empty() {
        return String::new();
    }
    match raw.as_str() {
        "microbe" | "microbes" => "microbe".to_string(),
        "animal" | "animals" => "animal".to_string(),
        "floater" | "floaters" => "floater".to_string(),
        "science" => "science".to_string(),
        "fighter" | "fighters" => "fighter".to_string(),
        "asteroid" | "astro" => "asteroid".to_string(),
        _ => raw,
    }
}

pub fn get_card_cost(card: &Value) -> f32 {
    to_f32(
        card.get("calculatedCost")
            .or_else(|| card.get("cost"))
            .or_else(|| card.get("cardCost")),
        0.0,
    )
    .max(0.0)
}

pub fn get_card_vp(card: &Value) -> f32 {
    to_f32(card.get("victoryPoints"), 0.0).max(0.0)
}

pub fn get_card_tags(fallback_tags: Option<&Value>) -> HashMap<String, i32> {
    let mut tags = HashMap::new();
    if let Some(fallback) = fallback_tags {
        if let Some(obj) = fallback.as_object() {
            for (key, val) in obj {
                let normalized = normalize_tag_name(key);
                if normalized.is_empty() {
                    continue;
                }
                if let Some(v_bool) = val.as_bool() {
                    if v_bool {
                        tags.insert(normalized, 1);
                    }
                } else if let Some(v_i64) = val.as_i64() {
                    if v_i64 > 0 {
                        tags.insert(normalized, v_i64 as i32);
                    }
                }
            }
            return tags;
        }

        if let Some(arr) = fallback.as_array() {
            for item in arr {
                if let Some(name) = item.as_str() {
                    let normalized = normalize_tag_name(name);
                    if !normalized.is_empty() {
                        tags.insert(normalized, 1);
                    }
                }
            }
            return tags;
        }
    }
    tags
}
