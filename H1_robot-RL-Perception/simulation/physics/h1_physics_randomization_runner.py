import csv
import os
import subprocess
import sys

BENCHMARK = "simulation/physics/h1_physics_randomization_benchmark.py"
OUTPUT = "simulation/physics/h1_physics_randomization_results.csv"

results = []

for episode, seed in enumerate(range(42, 52), start=1):
    print(f"\n{'=' * 80}")
    print(f"EPISODE {episode}/10  |  SEED {seed}")
    print(f"{'=' * 80}")

    env = os.environ.copy()
    env["H1_RANDOM_SEED"] = str(seed)
    env["H1_HEADLESS_DURATION"] = "8"

    result = subprocess.run(
        [sys.executable, BENCHMARK],
        env=env,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    metrics = {}

    for line in result.stdout.splitlines():
        if line.startswith("BENCHMARK_"):
            key, value = line.split("=", 1)
            metrics[key] = value.strip()

    if result.returncode != 0:
        print(f"Episode failed to execute. Return code: {result.returncode}")
        if result.stderr:
            print(result.stderr)

    row = {
        "episode": episode,
        "seed": seed,
        "friction": "",
        "gravity": "",
        "damping_scale": "",
        "peak_tilt_deg": metrics.get("BENCHMARK_PEAK_TILT_DEG", ""),
        "final_tilt_deg": metrics.get("BENCHMARK_FINAL_TILT_DEG", ""),
        "min_contacts": metrics.get("BENCHMARK_MIN_CONTACTS", ""),
        "min_fz": metrics.get("BENCHMARK_MIN_FZ", ""),
        "max_tau": metrics.get("BENCHMARK_MAX_TAU", ""),
        "safety_reason": metrics.get("BENCHMARK_SAFETY_REASON", ""),
        "return_code": result.returncode,
    }

    for line in result.stdout.splitlines():
        if line.startswith("Friction"):
            row["friction"] = line.split(":", 1)[1].strip()
        elif line.startswith("Gravity"):
            row["gravity"] = line.split(":", 1)[1].strip().split()[0]
        elif line.startswith("Damping scale"):
            row["damping_scale"] = line.split(":", 1)[1].strip()

    results.append(row)

fieldnames = [
    "episode",
    "seed",
    "friction",
    "gravity",
    "damping_scale",
    "peak_tilt_deg",
    "final_tilt_deg",
    "min_contacts",
    "min_fz",
    "max_tau",
    "safety_reason",
    "return_code",
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# ============================================================================
# FINAL VALIDATION
# ============================================================================

failures = []

if len(results) != 10:
    failures.append(f"Expected 10 episodes, got {len(results)}")

for row in results:
    ep = row["episode"]

    if row["return_code"] != 0:
        failures.append(
            f"Episode {ep}: return code {row['return_code']}"
        )

    if not row["friction"]:
        failures.append(f"Episode {ep}: missing friction")

    if not row["gravity"]:
        failures.append(f"Episode {ep}: missing gravity")

    if not row["damping_scale"]:
        failures.append(f"Episode {ep}: missing damping scale")

    required_metrics = [
        "peak_tilt_deg",
        "final_tilt_deg",
        "min_contacts",
        "min_fz",
        "max_tau",
    ]

    for metric in required_metrics:
        if not row[metric]:
            failures.append(
                f"Episode {ep}: missing {metric}"
            )

    if row["min_contacts"]:
        if int(float(row["min_contacts"])) < 4:
            failures.append(
                f"Episode {ep}: min contacts = {row['min_contacts']}"
            )

    if row["safety_reason"]:
        failures.append(
            f"Episode {ep}: safety reason = {row['safety_reason']}"
        )

print(f"\n{'=' * 80}")
print("RANDOMIZATION BENCHMARK VALIDATION")
print(f"{'=' * 80}")

if failures:
    print("STATUS : FAILED")
    for failure in failures:
        print(f" - {failure}")
    print(f"CSV    : {OUTPUT}")
    sys.exit(1)

print("STATUS : PASS")
print("Episodes : 10/10")
print("Return codes : 0/10 failures")
print("Physics parameters : 10/10 present")
print("Required metrics : 10/10 present")
print("Minimum contacts : >= 4 in all episodes")
print("Safety stops : 0")
print(f"CSV      : {OUTPUT}")
print(f"{'=' * 80}")
