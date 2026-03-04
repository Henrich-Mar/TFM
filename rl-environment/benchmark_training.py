#!/usr/bin/env python3
"""
RL Training Benchmark — stress-test GPU inference, PPO training, and capacity
planning to find optimal parameters for a given cloud machine.

Run inside the coordinator container (where PyTorch + CUDA are available):
    python benchmark_training.py

Or from the host via:
    bash scripts/benchmark_training.sh
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the models package is importable.
# Inside Docker: /app is the workdir with models/ right there.
# From repo root: rl-environment/ contains models/.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from models.agent import AgentConfig, TerraformingMarsNetwork  # noqa: E402
from models.ppo import (  # noqa: E402
    PPOHyperParameters,
    PPORolloutStep,
    optimize_ppo_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WIDTH = 78  # terminal table width


def _hr(char: str = "─") -> str:
    return char * WIDTH


def _header(title: str) -> str:
    return f"\n{'═' * WIDTH}\n  {title}\n{'═' * WIDTH}"


def _kv(key: str, value: Any, indent: int = 2) -> str:
    return f"{' ' * indent}{key:<40s} {value}"


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def _fmt_mem(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# 1. Hardware Detection
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    cpu_model: str
    cpu_count: int
    ram_total_mb: int
    gpu_name: str
    gpu_vram_mb: int
    cuda_version: str
    torch_version: str
    has_cuda: bool


def detect_hardware(cpu_only: bool = False) -> HardwareInfo:
    # CPU
    cpu_model = platform.processor() or "unknown"
    try:
        # Try reading /proc/cpuinfo for a better name
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    cpu_count = os.cpu_count() or 1

    # RAM
    ram_total_mb = 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    ram_total_mb = int(parts[1]) // 1024
                    break
    except Exception:
        ram_total_mb = 16384  # fallback

    # GPU
    has_cuda = torch.cuda.is_available() and not cpu_only
    gpu_name = "N/A"
    gpu_vram_mb = 0
    cuda_version = "N/A"
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        cuda_version = torch.version.cuda or "unknown"

    return HardwareInfo(
        cpu_model=cpu_model,
        cpu_count=cpu_count,
        ram_total_mb=ram_total_mb,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        has_cuda=has_cuda,
    )


def print_hardware(hw: HardwareInfo) -> None:
    print(_header("HARDWARE"))
    print(_kv("CPU", hw.cpu_model))
    print(_kv("CPU cores", hw.cpu_count))
    print(_kv("RAM", _fmt_mem(hw.ram_total_mb)))
    print(_kv("GPU", hw.gpu_name))
    print(_kv("GPU VRAM", _fmt_mem(hw.gpu_vram_mb)))
    print(_kv("CUDA", hw.cuda_version))
    print(_kv("PyTorch", hw.torch_version))
    print(_kv("CUDA available", hw.has_cuda))


# ---------------------------------------------------------------------------
# 2. Build model from env vars (matching the compose config)
# ---------------------------------------------------------------------------

def build_agent_config() -> AgentConfig:
    """Build AgentConfig from environment variables, same as the coordinator."""
    return AgentConfig(
        state_size=int(os.getenv("AGENT_STATE_SIZE", "1024")),
        hidden_size=int(os.getenv("AGENT_HIDDEN_SIZE", "1024")),
        num_layers=int(os.getenv("AGENT_NUM_LAYERS", "8")),
        recurrent_size=int(os.getenv("AGENT_RECURRENT_SIZE", "128")),
        phase_head_count=int(os.getenv("AGENT_PHASE_HEAD_COUNT", "6")),
        card_token_dim=int(os.getenv("AGENT_CARD_TOKEN_DIM", "20")),
        tableau_token_count=int(os.getenv("AGENT_TABLEAU_TOKEN_COUNT", "10")),
        hand_token_count=int(os.getenv("AGENT_HAND_TOKEN_COUNT", "4")),
        opponent_token_count=int(os.getenv("AGENT_OPPONENT_TOKEN_COUNT", "6")),
        transformer_embed_dim=int(os.getenv("AGENT_TRANSFORMER_EMBED_DIM", "256")),
        transformer_heads=int(os.getenv("AGENT_TRANSFORMER_HEADS", "16")),
        transformer_layers=int(os.getenv("AGENT_TRANSFORMER_LAYERS", "4")),
    )


# ---------------------------------------------------------------------------
# 3. Inference Benchmark
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    batch_size: int
    device: str
    latency_per_decision_ms: float
    throughput_decisions_per_sec: float
    vram_used_mb: float


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_inference(
    config: AgentConfig,
    batch_sizes: List[int],
    device: torch.device,
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> List[InferenceResult]:
    """Benchmark forward-pass latency at different batch sizes."""
    network = TerraformingMarsNetwork(config).to(device).eval()
    state_size = config.state_size
    results: List[InferenceResult] = []

    for bs in batch_sizes:
        states = torch.randn(bs, state_size, device=device)
        phase_idx = torch.randint(0, config.phase_head_count, (bs,), device=device)
        recurrent = torch.zeros(bs, config.recurrent_size, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                network(states, phase_indices=phase_idx, recurrent_state=recurrent)
        _sync_cuda()

        # Track VRAM before measurement
        vram_before = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            vram_before = torch.cuda.memory_allocated(device) / (1024 * 1024)

        # Benchmark
        _sync_cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                network(states, phase_indices=phase_idx, recurrent_state=recurrent)
        _sync_cuda()
        elapsed = time.perf_counter() - t0

        vram_used = 0.0
        if device.type == "cuda":
            vram_used = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        latency_per_call = elapsed / bench_iters
        latency_per_decision = latency_per_call / bs
        throughput = bs / latency_per_call

        results.append(InferenceResult(
            batch_size=bs,
            device=str(device),
            latency_per_decision_ms=latency_per_decision * 1000,
            throughput_decisions_per_sec=throughput,
            vram_used_mb=vram_used,
        ))

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def print_inference_results(results: List[InferenceResult]) -> None:
    print(_header("INFERENCE BENCHMARK"))
    header = f"  {'Batch':>6s}  {'Device':>6s}  {'Latency/dec':>12s}  {'Throughput':>14s}  {'VRAM':>8s}"
    print(header)
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 12}  {'─' * 14}  {'─' * 8}")
    for r in results:
        print(
            f"  {r.batch_size:>6d}  {r.device:>6s}  "
            f"{r.latency_per_decision_ms:>10.3f}ms  "
            f"{r.throughput_decisions_per_sec:>11.0f}/s  "
            f"{_fmt_mem(r.vram_used_mb):>8s}"
        )


# ---------------------------------------------------------------------------
# 4. PPO Training Benchmark
# ---------------------------------------------------------------------------

@dataclass
class PPOTrainingResult:
    minibatch_size: int
    rollout_steps: int
    epochs: int
    wall_time_sec: float
    steps_per_sec: float
    vram_peak_mb: float
    updates_total: int


def _generate_synthetic_rollout(
    count: int, state_size: int, action_dim: int = 1000
) -> List[PPORolloutStep]:
    """Create synthetic rollout data for benchmarking."""
    rng = np.random.RandomState(42)
    steps: List[PPORolloutStep] = []
    for _ in range(count):
        num_legal = rng.randint(2, 20)
        legal_actions = sorted(rng.choice(action_dim, size=num_legal, replace=False).tolist())
        steps.append(PPORolloutStep(
            state=rng.randn(state_size).astype(np.float32),
            action=int(rng.choice(legal_actions)),
            logp_old=float(rng.uniform(-5, 0)),
            value_old=float(rng.uniform(-1, 1)),
            reward=float(rng.uniform(-1, 1)),
            done=bool(rng.rand() < 0.02),
            legal_actions=legal_actions,
            phase_index=int(rng.randint(0, 6)),
            recurrent_state=rng.randn(128).astype(np.float32),
            aux_milestone_claimability=rng.rand(70).astype(np.float32),
            aux_award_ev=float(rng.rand()),
            aux_playable_cards=float(rng.rand()),
            aux_steel_target=float(rng.rand()),
            aux_titanium_target=float(rng.rand()),
        ))
    return steps


def benchmark_ppo_training(
    config: AgentConfig,
    minibatch_sizes: List[int],
    rollout_steps: int = 8192,
    ppo_epochs: int = 4,
) -> List[PPOTrainingResult]:
    """Benchmark PPO optimize_ppo_policy at different minibatch sizes."""
    results: List[PPOTrainingResult] = []
    state_size = config.state_size

    print(f"\n  Generating {rollout_steps} synthetic rollout steps...")
    synthetic_steps = _generate_synthetic_rollout(rollout_steps, state_size)

    for mbs in minibatch_sizes:
        print(f"  Benchmarking minibatch_size={mbs}...", end="", flush=True)

        network = TerraformingMarsNetwork(config)
        optimizer = torch.optim.Adam(network.parameters(), lr=1.2e-4)
        ppo_params = PPOHyperParameters(
            epochs=ppo_epochs,
            minibatch_size=mbs,
            clip_eps=0.15,
            value_clip_eps=0.15,
            entropy_coef=0.015,
            value_coef=1.0,
            target_kl=100.0,  # Disable early stopping for benchmark
        )

        # Reset VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        metrics = optimize_ppo_policy(
            network=network,
            optimizer=optimizer,
            steps=synthetic_steps,
            ppo=ppo_params,
        )
        elapsed = time.perf_counter() - t0

        vram_peak = 0.0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        num_updates = int(metrics.get("ppo/update_steps", 0))
        total_samples_processed = num_updates * mbs
        steps_per_sec = total_samples_processed / max(elapsed, 1e-6)

        results.append(PPOTrainingResult(
            minibatch_size=mbs,
            rollout_steps=rollout_steps,
            epochs=ppo_epochs,
            wall_time_sec=elapsed,
            steps_per_sec=steps_per_sec,
            vram_peak_mb=vram_peak,
            updates_total=num_updates,
        ))

        del network, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f" {_fmt_time(elapsed)}, {steps_per_sec:.0f} samples/s, VRAM peak {_fmt_mem(vram_peak)}")

    return results


def print_ppo_results(results: List[PPOTrainingResult]) -> None:
    print(_header("PPO TRAINING BENCHMARK"))
    print(f"  Rollout = {results[0].rollout_steps} steps, {results[0].epochs} epochs per run")
    print()
    header = (
        f"  {'Minibatch':>10s}  {'Wall time':>10s}  {'Throughput':>14s}  "
        f"{'VRAM peak':>10s}  {'Updates':>8s}"
    )
    print(header)
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 14}  {'─' * 10}  {'─' * 8}")
    for r in results:
        print(
            f"  {r.minibatch_size:>10d}  {_fmt_time(r.wall_time_sec):>10s}  "
            f"{r.steps_per_sec:>11.0f}/s  "
            f"{_fmt_mem(r.vram_peak_mb):>10s}  {r.updates_total:>8d}"
        )


# ---------------------------------------------------------------------------
# 5. Capacity Planning / Recommendations
# ---------------------------------------------------------------------------

@dataclass
class Recommendations:
    # Inference
    best_inference_batch_size: int
    inference_throughput: float
    recommended_inference_threads: int
    # PPO
    best_ppo_minibatch_size: int
    ppo_throughput: float
    ppo_vram_peak_mb: float
    # Capacity
    recommended_servers: int
    recommended_games_per_server: int
    recommended_global_concurrency: int
    recommended_profile: str
    # Computed estimates
    estimated_decisions_per_hour: int
    gpu_bottleneck: bool


def compute_recommendations(
    hw: HardwareInfo,
    inference_results: List[InferenceResult],
    ppo_results: List[PPOTrainingResult],
) -> Recommendations:
    # --- Inference: pick best throughput ---
    best_inf = max(inference_results, key=lambda r: r.throughput_decisions_per_sec)

    # --- Inference threads: for GPU inference, 4 is good; for CPU scale with cores ---
    if hw.has_cuda:
        inf_threads = min(8, max(4, hw.cpu_count // 4))
    else:
        inf_threads = min(32, hw.cpu_count)

    # --- PPO: pick best throughput that fits in VRAM ---
    viable_ppo = ppo_results
    if hw.has_cuda and hw.gpu_vram_mb > 0:
        # Leave ~2GB for inference (16 agents on GPU)
        vram_budget = hw.gpu_vram_mb - 2048
        viable_vram = [r for r in ppo_results if r.vram_peak_mb <= vram_budget]
        if viable_vram:
            viable_ppo = viable_vram
    best_ppo = max(viable_ppo, key=lambda r: r.steps_per_sec)

    # --- Capacity planning based on RAM and CPU ---
    # Reserve 10% RAM for OS + overhead
    usable_ram_mb = int(hw.ram_total_mb * 0.85)
    # Infra overhead (postgres, redis, coordinator): ~4 GB
    infra_mb = 4096
    server_mem_mb = 1400
    available_for_servers = max(0, usable_ram_mb - infra_mb)
    max_servers_by_ram = max(1, available_for_servers // server_mem_mb)
    # CPU: ~2 servers per core is reasonable for Node.js
    max_servers_by_cpu = max(1, int(hw.cpu_count * 1.5))
    recommended_servers = min(max_servers_by_ram, max_servers_by_cpu, 128)

    # Games per server based on available resources
    if hw.ram_total_mb >= 196_608 or hw.cpu_count >= 48:
        games_per_server = 6
    elif hw.ram_total_mb >= 98_304 or hw.cpu_count >= 24:
        games_per_server = 5
    elif hw.ram_total_mb >= 49_152 or hw.cpu_count >= 16:
        games_per_server = 4
    else:
        games_per_server = 3

    global_concurrency = recommended_servers * games_per_server

    # --- Estimate decisions per hour ---
    # Assume ~200 decisions per game, ~3 min per game on average  
    # GPU inference is rarely the bottleneck; game simulation is.
    games_per_hour = global_concurrency * (60 / 3)  # ~3 min per game
    decisions_per_hour = int(games_per_hour * 200)

    # --- Is GPU the bottleneck? ---
    # If inference throughput < decisions needed per second, GPU is bottleneck
    decisions_per_sec_needed = decisions_per_hour / 3600
    gpu_bottleneck = best_inf.throughput_decisions_per_sec < decisions_per_sec_needed

    # Profile recommendation
    if hw.cpu_count >= 16 and hw.ram_total_mb >= 49_152:
        profile = "saturate"
    else:
        profile = "balanced"

    return Recommendations(
        best_inference_batch_size=best_inf.batch_size,
        inference_throughput=best_inf.throughput_decisions_per_sec,
        recommended_inference_threads=inf_threads,
        best_ppo_minibatch_size=best_ppo.minibatch_size,
        ppo_throughput=best_ppo.steps_per_sec,
        ppo_vram_peak_mb=best_ppo.vram_peak_mb,
        recommended_servers=recommended_servers,
        recommended_games_per_server=games_per_server,
        recommended_global_concurrency=global_concurrency,
        recommended_profile=profile,
        estimated_decisions_per_hour=decisions_per_hour,
        gpu_bottleneck=gpu_bottleneck,
    )


def print_recommendations(rec: Recommendations, hw: HardwareInfo) -> None:
    print(_header("RECOMMENDATIONS"))

    print("\n  Bottleneck analysis:")
    if rec.gpu_bottleneck:
        print("  ⚠️  GPU inference may be the bottleneck!")
        print("      Consider reducing game concurrency or increasing batch deadline.")
    else:
        print("  ✅  GPU has plenty of headroom. Game simulation (Node.js CPU) is the bottleneck.")
        print("      Focus on maximizing server count and games-per-server.")

    print(f"\n  Inference throughput:  {rec.inference_throughput:,.0f} decisions/sec")
    print(f"  PPO training speed:   {rec.ppo_throughput:,.0f} samples/sec")
    print(f"  Est. decisions/hour:  {rec.estimated_decisions_per_hour:,}")

    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  RECOMMENDED ENVIRONMENT VARIABLES                            │")
    print("  ├─────────────────────────────────────────────────────────────────┤")

    env_vars = [
        ("RL_TRAINING_PROFILE", rec.recommended_profile),
        ("RL_MAX_SERVERS", str(rec.recommended_servers)),
        ("RL_GAMES_PER_SERVER", str(rec.recommended_games_per_server)),
        ("PPO_MINIBATCH_SIZE", str(rec.best_ppo_minibatch_size)),
        ("AGENT_INFERENCE_BATCH_SIZE", str(rec.best_inference_batch_size)),
        ("AGENT_INFERENCE_THREADS", str(rec.recommended_inference_threads)),
        ("RL_SERVER_MEM_MB", "1400"),
        ("RL_AGENT_POLL_INTERVAL_SEC", "0.03" if rec.recommended_profile == "saturate" else "0.20"),
        ("RL_AGENT_FAILURE_PAUSE_SEC", "0.05" if rec.recommended_profile == "saturate" else "0.20"),
    ]

    for key, value in env_vars:
        print(f"  │  export {key}={value}")
    print("  └─────────────────────────────────────────────────────────────────┘")

    print("\n  Copy-paste block:")
    print("  " + _hr("─"))
    for key, value in env_vars:
        print(f"  export {key}={value}")
    print("  " + _hr("─"))

    # Training speed estimate
    print(f"\n  With {rec.recommended_servers} servers × {rec.recommended_games_per_server} games = "
          f"{rec.recommended_global_concurrency} concurrent games")
    print(f"  Expected ~{rec.estimated_decisions_per_hour:,} decisions/hour")
    pop_size = int(os.getenv("POPULATION_SIZE", "16"))
    games_per_eval = int(os.getenv("GAMES_PER_EVAL", "6"))
    expected_games_per_gen = pop_size * games_per_eval
    game_instances = math.ceil(expected_games_per_gen / 4)
    waves = math.ceil(game_instances / max(1, rec.recommended_global_concurrency))
    est_gen_time_min = waves * 3  # ~3 min per wave
    print(f"  With POP={pop_size}, GAMES_PER_EVAL={games_per_eval}: "
          f"~{game_instances} game instances/gen, {waves} waves, ~{est_gen_time_min} min/gen")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RL training pipeline to find optimal parameters."
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only benchmarks (skip GPU even if available).",
    )
    parser.add_argument(
        "--inference-batch-sizes",
        type=str,
        default="1,4,8,16,32,64",
        help="Comma-separated batch sizes for inference benchmark.",
    )
    parser.add_argument(
        "--ppo-minibatch-sizes",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated minibatch sizes for PPO benchmark.",
    )
    parser.add_argument(
        "--ppo-rollout-steps",
        type=int,
        default=8192,
        help="Number of synthetic rollout steps for PPO benchmark.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="PPO epochs for benchmark runs.",
    )
    parser.add_argument(
        "--inference-iters",
        type=int,
        default=50,
        help="Number of forward-pass iterations per batch size.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference benchmark.",
    )
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Skip the PPO training benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("\n" + "█" * WIDTH)
    print("  TFM RL Training Benchmark")
    print("█" * WIDTH)

    # 1. Hardware
    hw = detect_hardware(cpu_only=args.cpu_only)
    print_hardware(hw)

    # 2. Build model config
    config = build_agent_config()
    print(_header("MODEL CONFIG"))
    print(_kv("state_size", config.state_size))
    print(_kv("hidden_size", config.hidden_size))
    print(_kv("num_layers", config.num_layers))
    print(_kv("transformer", f"{config.transformer_embed_dim}d, "
              f"{config.transformer_heads}h, {config.transformer_layers}L"))
    print(_kv("card tokens", f"{config.tableau_token_count}+{config.hand_token_count}"
              f"+{config.opponent_token_count} × {config.card_token_dim}d"))

    # Count parameters
    tmp_net = TerraformingMarsNetwork(config)
    param_count = sum(p.numel() for p in tmp_net.parameters())
    print(_kv("parameters", f"{param_count:,}"))
    del tmp_net

    # 3. Inference benchmark
    inference_results: List[InferenceResult] = []
    if not args.skip_inference:
        batch_sizes = [int(x) for x in args.inference_batch_sizes.split(",")]

        if hw.has_cuda:
            gpu_results = benchmark_inference(
                config, batch_sizes,
                device=torch.device("cuda"),
                bench_iters=args.inference_iters,
            )
            inference_results.extend(gpu_results)

        # Always include CPU baseline for comparison
        cpu_batch_sizes = [bs for bs in batch_sizes if bs <= 16]
        if not cpu_batch_sizes:
            cpu_batch_sizes = [1, 8]
        cpu_results = benchmark_inference(
            config, cpu_batch_sizes,
            device=torch.device("cpu"),
            warmup_iters=3,
            bench_iters=10,
        )
        inference_results.extend(cpu_results)
        print_inference_results(inference_results)

    # 4. PPO training benchmark
    ppo_results: List[PPOTrainingResult] = []
    if not args.skip_ppo:
        minibatch_sizes = [int(x) for x in args.ppo_minibatch_sizes.split(",")]
        ppo_results = benchmark_ppo_training(
            config,
            minibatch_sizes,
            rollout_steps=args.ppo_rollout_steps,
            ppo_epochs=args.ppo_epochs,
        )
        print_ppo_results(ppo_results)

    # 5. Recommendations
    if inference_results and ppo_results:
        # Use only GPU results for recommendations if available
        gpu_inf = [r for r in inference_results if r.device != "cpu"]
        if not gpu_inf:
            gpu_inf = inference_results
        rec = compute_recommendations(hw, gpu_inf, ppo_results)
        print_recommendations(rec, hw)
    elif inference_results:
        print("\n  ⚠️  Skipped PPO benchmark — recommendations may be incomplete.")
    elif ppo_results:
        print("\n  ⚠️  Skipped inference benchmark — recommendations may be incomplete.")

    print(f"\n{'█' * WIDTH}")
    print("  Benchmark complete!")
    print(f"{'█' * WIDTH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
