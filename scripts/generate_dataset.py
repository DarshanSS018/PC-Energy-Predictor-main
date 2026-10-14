"""Generate a SYNTHETIC PC telemetry + power dataset.

The synopsis allows "PC monitoring sources or a suitable dataset". This script
produces a physically-motivated stand-in so the whole pipeline runs end to end.
For the final report, replace data/pc_energy.csv with real measurements
(see scripts/collect_data.py) and re-run scripts/train.py -- nothing else changes.

Power model (watts): idle + CPU (util x freq^2, nonlinear) + GPU + RAM + disk
+ network + temperature leakage + noise.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "cpu_usage", "gpu_usage", "ram_usage", "cpu_freq_ghz",
    "cpu_temp_c", "gpu_temp_c", "disk_activity", "network_activity",
]
TARGET = "power_watts"


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Workload "regimes": idle, office, gaming, compute-heavy
    regime = rng.choice(4, size=n, p=[0.25, 0.30, 0.25, 0.20])
    cpu_mu = np.array([6, 25, 55, 85])[regime]
    gpu_mu = np.array([2, 10, 80, 35])[regime]
    cpu = np.clip(rng.normal(cpu_mu, 10), 1, 100)
    gpu = np.clip(rng.normal(gpu_mu, 12), 0, 100)
    ram = np.clip(rng.normal(35 + 0.35 * cpu + 0.1 * gpu, 8), 10, 98)

    # Boost clock rises with load (1.2 GHz idle -> ~4.6 GHz)
    freq = np.clip(1.2 + 3.4 * (cpu / 100) ** 0.6 + rng.normal(0, 0.15, n), 0.8, 5.0)
    disk = np.clip(rng.gamma(1.5, 8, n) + 0.1 * cpu, 0, 100)
    net = np.clip(rng.gamma(1.2, 6, n), 0, 100)

    # Dynamic power; frequency enters ~quadratically (voltage scales with f)
    cpu_p = 0.55 * cpu * (freq / 3.0) ** 2
    gpu_p = 1.9 * gpu * (0.6 + 0.4 * gpu / 100)
    ram_p = 0.12 * ram
    io_p = 0.05 * disk + 0.03 * net
    base = 28.0

    # Temperature follows power (thermal lag handled via noise), then leaks back
    heat = cpu_p + gpu_p * 0.6
    cpu_t = np.clip(32 + 0.42 * heat + rng.normal(0, 2.5, n), 28, 98)
    gpu_t = np.clip(30 + 0.28 * gpu_p + rng.normal(0, 2.5, n), 28, 92)
    leak = 0.08 * np.maximum(cpu_t - 60, 0) ** 1.3

    power = base + cpu_p + gpu_p + ram_p + io_p + leak + rng.normal(0, 3.0, n)

    return pd.DataFrame({
        "cpu_usage": cpu.round(1), "gpu_usage": gpu.round(1), "ram_usage": ram.round(1),
        "cpu_freq_ghz": freq.round(2), "cpu_temp_c": cpu_t.round(1), "gpu_temp_c": gpu_t.round(1),
        "disk_activity": disk.round(1), "network_activity": net.round(1),
        TARGET: power.round(1),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "pc_energy.csv"))
    a = ap.parse_args()
    df = generate(a.rows)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows -> {a.out}")
    print(df.describe().T[["min", "mean", "max"]])
