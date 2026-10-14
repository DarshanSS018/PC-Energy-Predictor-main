"""Collect REAL telemetry from this PC for the dataset (optional, for the final report).

Features come from psutil. Power (the target) needs a source, in order of preference:
  1. Linux Intel RAPL  (/sys/class/powercap/intel-rapl:0/energy_uj)  -> CPU package watts
  2. A manual value you type from a plug-in wattmeter (--manual)
  3. A column exported from HWiNFO / LibreHardwareMonitor / HWMonitor that you merge later
GPU usage/temp need NVIDIA's `nvidia-smi` (used automatically if present), else 0.

Usage:  python scripts/collect_data.py --samples 600 --interval 1 --out data/real_pc.csv
Then:   python scripts/train.py --data data/real_pc.csv
Run different workloads while it records (idle, browsing, video, gaming, compiling).
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def gpu_stats():
    if not shutil.which("nvidia-smi"):
        return 0.0, 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=2).strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return 0.0, 0.0


def cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)
    except Exception:
        pass
    return float("nan")  # fill manually / from HWiNFO export if unavailable


def read_rapl():
    try:
        return int(RAPL.read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--manual", action="store_true", help="type watts from a wattmeter each sample")
    ap.add_argument("--out", default="data/real_pc.csv")
    a = ap.parse_args()

    rows, prev_e, prev_t = [], read_rapl(), time.time()
    psutil.cpu_percent(None)
    d0, n0 = psutil.disk_io_counters(), psutil.net_io_counters()
    for i in range(a.samples):
        time.sleep(a.interval)
        now = time.time()
        d1, n1 = psutil.disk_io_counters(), psutil.net_io_counters()
        dt = now - prev_t
        # activity = MB/s, capped later in preprocessing
        disk = ((d1.read_bytes + d1.write_bytes) - (d0.read_bytes + d0.write_bytes)) / 1e6 / dt
        net = ((n1.bytes_sent + n1.bytes_recv) - (n0.bytes_sent + n0.bytes_recv)) / 1e6 / dt
        d0, n0 = d1, n1

        power = float("nan")
        e = read_rapl()
        if e is not None and prev_e is not None and e >= prev_e:
            power = (e - prev_e) / 1e6 / dt  # microjoules -> watts
        prev_e, prev_t = e, now
        if a.manual:
            power = float(input(f"[{i+1}/{a.samples}] watts: "))

        g_use, g_temp = gpu_stats()
        freq = psutil.cpu_freq()
        rows.append({
            "cpu_usage": psutil.cpu_percent(None),
            "gpu_usage": g_use,
            "ram_usage": psutil.virtual_memory().percent,
            "cpu_freq_ghz": round((freq.current if freq else 0) / 1000, 2),
            "cpu_temp_c": cpu_temp(),
            "gpu_temp_c": g_temp,
            "disk_activity": round(disk, 2),
            "network_activity": round(net, 2),
            "power_watts": power,
        })
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{a.samples} samples")
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {len(df)} rows -> {a.out}  (missing power rows: {df.power_watts.isna().sum()})")


if __name__ == "__main__":
    main()
