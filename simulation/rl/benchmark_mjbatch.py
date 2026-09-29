#!/usr/bin/env python3
"""
ANSA OS - Unitree H1: mjbatch High-Throughput Performance Benchmark
=============================================================================
Demonstrates running 256 to 512 parallel H1 robots concurrently on CPU.
Measures simulation throughput in FPS (Frames Per Second).
=============================================================================
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add module paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_batched_env import H1BatchedEnv


def run_benchmark(num_sims: int = 256, num_steps: int = 200):
    print("=" * 78)
    print("      ANSA OS - UNITREE H1: MJBATCH HIGH-THROUGHPUT BENCHMARK")
    print(f"      Testing {num_sims} Parallel Humanoid Robots Concurrently on CPU")
    print("=" * 78)

    print(f"\n[1/3] Initializing {num_sims} parallel H1 simulation instances with mjbatch...")
    t0 = time.time()
    env = H1BatchedEnv(num_sims=num_sims, frameskip=10, enable_domain_randomization=True)
    init_time = time.time() - t0
    print(f"  ✓ Initialized {num_sims} environments in {init_time:.2f}s")
    print(f"  ✓ Bound qpos shape : {env.qpos.shape}")
    print(f"  ✓ Bound ctrl shape : {env.ctrl.shape}")
    print(f"  ✓ Obs vector shape : {env.get_observations().shape}")

    print(f"\n[2/3] Running {num_steps} batched simulation steps (each step has frameskip=10)...")
    total_samples = 0
    t_start = time.time()

    for step in range(num_steps):
        # Generate random exploratory control actions for all robots at once
        actions = np.random.uniform(-0.5, 0.5, size=(num_sims, env.num_actuators)).astype(np.float32)

        obs, rewards, dones, info = env.step(actions)
        total_samples += num_sims

        if (step + 1) % 50 == 0 or (step + 1) == num_steps:
            elapsed = time.time() - t_start
            fps = total_samples / elapsed
            physics_fps = fps * env.frameskip
            print(f"  Step {step+1:3d}/{num_steps} | Simulated Samples: {total_samples:7,d} | Throughput: {fps:8.1f} Env-Steps/s ({physics_fps:9.1f} Physics-FPS)")

    total_time = time.time() - t_start
    final_fps = total_samples / total_time
    final_physics_fps = final_fps * env.frameskip

    # Comparison metrics
    standard_fps = 750.0  # Measured from our earlier 4-env SubprocVecEnv run
    speedup = final_fps / standard_fps

    print("\n" + "=" * 78)
    print("                          BENCHMARK RESULTS")
    print("=" * 78)
    print(f"  Parallel Robots Simulated : {num_sims} simultaneous instances")
    print(f"  Total Simulated Steps     : {total_samples:,} environment steps")
    print(f"  Total Physics Steps       : {total_samples * env.frameskip:,} physics steps")
    print(f"  Total Elapsed Time        : {total_time:.2f} seconds")
    print(f"  Effective Environment FPS : {final_fps:,.1f} steps/second")
    print(f"  Raw Physics Throughput    : {final_physics_fps:,.1f} physics-evaluations/second")
    print(f"  Speedup vs Standard Loop  : {speedup:.1f}x FASTER than 4-worker multiprocessing!")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="H1 mjbatch Performance Benchmark")
    parser.add_argument("--sims", type=int, default=256, help="Number of parallel robots (e.g. 128, 256, 512)")
    parser.add_argument("--steps", type=int, default=200, help="Number of batched steps to evaluate")
    args = parser.parse_args()

    run_benchmark(num_sims=args.sims, num_steps=args.steps)
