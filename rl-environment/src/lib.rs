use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

mod action_decoder;
mod domain;
mod encoder;
mod utils;

#[pyfunction]
fn encode_state<'py>(
    py: Python<'py>,
    json_str: &str,
    turn_action_count: i32,
    state_size: usize,
) -> PyResult<&'py PyArray1<f32>> {
    match encoder::encode_state_impl(json_str, turn_action_count, state_size) {
        Ok(features) => Ok(features.into_pyarray(py)),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn estimate_affordability(json_player_str: &str, json_card_str: &str) -> PyResult<f32> {
    let player: Option<domain::PlayerState> = serde_json::from_str(json_player_str).ok();
    let card: Value = serde_json::from_str(json_card_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid card json: {}", e)))?;
    Ok(action_decoder::estimate_affordability(player.as_ref(), &card))
}

#[pyfunction]
fn can_afford_cards(json_player_str: &str, json_cards_str: &str) -> PyResult<Vec<bool>> {
    let player: domain::PlayerState = serde_json::from_str(json_player_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid player json: {}", e)))?;
    let cards: Vec<Value> = serde_json::from_str(json_cards_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid cards json: {}", e)))?;
    Ok(action_decoder::can_afford_cards(&player, &cards))
}

#[pyfunction]
fn enumerate_card_selection_combos(json_payload_str: &str, limit: usize) -> PyResult<Vec<Vec<usize>>> {
    let payload: action_decoder::CardSelectionPayload = serde_json::from_str(json_payload_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid card selection payload: {}", e)))?;
    Ok(action_decoder::enumerate_card_selection_combos(
        &payload,
        limit.max(1),
    ))
}

#[pyfunction]
fn rank_startup_plans(json_payload_str: &str, max_plans: usize) -> PyResult<String> {
    let payload: action_decoder::StartupRankPayload = serde_json::from_str(json_payload_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid startup rank payload: {}", e)))?;
    Ok(action_decoder::rank_startup_plans(&payload, max_plans.max(1)))
}

#[pyfunction]
fn backend_info(py: Python<'_>) -> PyResult<PyObject> {
    let info = PyDict::new(py);
    info.set_item("module", "rust_tfm_rl")?;
    info.set_item("api_version", "1.0")?;
    info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    info.set_item(
        "capabilities",
        vec![
            "encode_state",
            "estimate_affordability",
            "can_afford_cards",
            "enumerate_card_selection_combos",
            "rank_startup_plans",
            "backend_info",
        ],
    )?;
    Ok(info.into())
}

#[pymodule]
fn rust_tfm_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_affordability, m)?)?;
    m.add_function(wrap_pyfunction!(can_afford_cards, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_card_selection_combos, m)?)?;
    m.add_function(wrap_pyfunction!(rank_startup_plans, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}
