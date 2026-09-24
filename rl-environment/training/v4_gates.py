"""Promotion gates that keep V4 PPO blocked until held-out evidence is complete."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


FAMILY_TOP1_GATES = {
    "play_card": 0.85,
    "card_subset": 0.90,
    "select_option": 0.95,
    "claim_milestone": 0.90,
    "fund_award": 0.80,
    "select_payment": 0.90,
}
MIN_HELD_OUT_FAMILY_SAMPLES = {
    "select_space": 100,
    "claim_milestone": 100,
    "fund_award": 100,
}
SMOKE_SEED = 910001
SMOKE_CONFIGURATION = {
    "baseline": "teacher",
    "stage": 1,
    "candidate_seat": 0,
    "deterministic_actions": True,
    "ppo_enabled": False,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _teacher_gate_status(
    metrics: Mapping[str, Any],
    placement_mode: str,
    *,
    minimum_family_samples: int | None = 100,
) -> Dict[str, Any]:
    if placement_mode not in {"top1", "top3"}:
        raise ValueError(f"unknown placement gate mode: {placement_mode!r}")
    top1 = metrics.get("family_top1") or {}
    top3 = metrics.get("family_top3") or {}
    counts = metrics.get("family_counts") or {}
    checks: List[Tuple[str, float, float]] = [
        ("teacher top-1", float(metrics.get("teacher_top1", 0.0) or 0.0), 0.85),
        ("teacher top-3", float(metrics.get("teacher_top3", 0.0) or 0.0), 0.97),
    ]
    checks.extend(
        (f"{family} top-1", float(top1.get(family, 0.0) or 0.0), threshold)
        for family, threshold in FAMILY_TOP1_GATES.items()
    )
    placement_threshold = 0.97 if placement_mode == "top3" else 0.87
    placement_values = top3 if placement_mode == "top3" else top1
    checks.append((
        f"select_space {placement_mode}",
        float(placement_values.get("select_space", 0.0) or 0.0),
        placement_threshold,
    ))

    reasons: List[str] = []
    margins: List[float] = []
    passed_count = 0
    for label, observed, threshold in checks:
        margin = (observed / threshold) - 1.0 if threshold else observed
        margins.append(margin)
        if observed >= threshold:
            passed_count += 1
        else:
            reasons.append(f"{label} {observed:.3f} is below {threshold:.2f}")

    if minimum_family_samples is not None:
        for family, default_minimum in MIN_HELD_OUT_FAMILY_SAMPLES.items():
            required = int(minimum_family_samples if minimum_family_samples >= 0 else default_minimum)
            observed = int(counts.get(family, 0) or 0)
            margins.append((observed / required) - 1.0 if required else float(observed))
            if observed >= required:
                passed_count += 1
            else:
                reasons.append(f"{family} has {observed} held-out targets; requires {required}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "passed_checks": passed_count,
        "total_checks": len(checks) + (len(MIN_HELD_OUT_FAMILY_SAMPLES) if minimum_family_samples is not None else 0),
        "worst_normalized_margin": min(margins) if margins else -1.0,
        "selection_key": (
            int(passed_count),
            float(min(margins) if margins else -1.0),
            float(metrics.get("teacher_top1", 0.0) or 0.0),
            float(metrics.get("teacher_top3", 0.0) or 0.0),
        ),
    }


def validation_candidate_status(
    metrics: Mapping[str, Any],
    placement_mode: str,
    *,
    allow_small_dataset: bool = False,
) -> Dict[str, Any]:
    return _teacher_gate_status(
        metrics,
        placement_mode,
        minimum_family_samples=0 if allow_small_dataset else 100,
    )


def evaluate_ppo_gate(report: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if report.get("selected_validation_gate_passed") is not True:
        reasons.append("selected checkpoint did not clear every validation gate")
    placement = report.get("placement_gate") or {}
    placement_mode = str(placement.get("mode", "top1") or "top1")
    if placement_mode == "top3" and not bool(placement.get("diagnostic_qualified", False)):
        reasons.append("select_space top-3 mode lacks a qualifying saved-checkpoint diagnostic")
    try:
        teacher_status = _teacher_gate_status(report.get("test") or {}, placement_mode)
        reasons.extend(teacher_status["reasons"])
    except ValueError as exc:
        reasons.append(str(exc))

    human = report.get("human_evaluation") or {}
    if float(human.get("top3", 0.0) or 0.0) < 0.80:
        reasons.append("held-out human top-3 is below 80%")
    held_out_games = sorted({str(value) for value in (human.get("held_out_games") or []) if str(value)})
    if len(held_out_games) != 2:
        reasons.append("human evaluation must contain exactly two held-out source games")
    if int(human.get("samples", 0) or 0) <= 0:
        reasons.append("held-out human evaluation has no decisions")

    if report.get("duplicate_executable_actions") != 0:
        reasons.append("duplicate executable actions remain")
    if report.get("unresolved_known_card_references") != 0:
        reasons.append("unresolved known-card references remain")

    selected_sha = str(report.get("selected_checkpoint_sha256", "") or "")
    if len(selected_sha) != 64 or any(character not in "0123456789abcdef" for character in selected_sha.lower()):
        reasons.append("selected checkpoint hash is missing")
    smoke = report.get("smoke") or {}
    if smoke.get("completed") is not True:
        reasons.append("fixed-seed smoke run has not completed")
    if smoke.get("server_rejected_actions") != 0:
        reasons.append("fixed-seed smoke run has not recorded zero server-rejected actions")
    if str(smoke.get("checkpoint_sha256", "") or "") != selected_sha:
        reasons.append("fixed-seed smoke evidence does not match the selected checkpoint")
    configuration = smoke.get("configuration")
    if smoke.get("seed") is None or not isinstance(configuration, dict):
        reasons.append("fixed-seed smoke evidence is missing seed/configuration provenance")
    else:
        try:
            observed_seed = int(smoke.get("seed"))
        except (TypeError, ValueError):
            reasons.append("fixed-seed smoke seed is invalid")
        else:
            if observed_seed != SMOKE_SEED:
                reasons.append(f"fixed-seed smoke used {observed_seed}; expected {SMOKE_SEED}")
        mismatches = [
            key for key, expected in SMOKE_CONFIGURATION.items()
            if configuration.get(key) != expected
        ]
        if mismatches:
            reasons.append("fixed-seed smoke configuration mismatch: " + ", ".join(mismatches))
    return (not reasons), reasons


def pretrain_report_path(path: str | None = None) -> Path:
    configured = path or os.getenv("V4_PRETRAIN_REPORT", "/app/v4/pretrain-v2/pretrain_report.json")
    return Path(configured).expanduser().resolve()


def assert_ppo_unlocked(path: str | None = None, checkpoint_path: str | None = None) -> Dict[str, Any]:
    report_path = pretrain_report_path(path)
    if not report_path.is_file():
        raise RuntimeError(f"PPO blocked until a V4 pretrain report exists: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed, reasons = evaluate_ppo_gate(report)
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            reasons.append(f"selected checkpoint does not exist: {checkpoint}")
        elif file_sha256(checkpoint) != str(report.get("selected_checkpoint_sha256", "") or ""):
            reasons.append("bootstrap checkpoint hash does not match the promoted pretrain report")
    else:
        reasons.append("PPO bootstrap checkpoint was not supplied for hash verification")
    if reasons or not passed:
        raise RuntimeError("PPO blocked: " + "; ".join(reasons))
    return report


def record_smoke_result(evidence: Mapping[str, Any], path: str | None = None) -> Dict[str, Any]:
    report_path = pretrain_report_path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"V4 pretrain report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {"completed", "server_rejected_actions", "checkpoint_sha256", "seed", "configuration"}
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"smoke evidence is missing required fields: {missing}")
    report["smoke"] = dict(evidence)
    passed, reasons = evaluate_ppo_gate(report)
    report["ppo_gate_passed"] = passed
    report["ppo_gate_reasons"] = reasons
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
