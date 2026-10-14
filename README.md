# PC Energy Consumption Predictor

Green Computing (22CS75) project: predicts PC power draw (watts) from system telemetry using
Linear Regression, Random Forest and Gradient Boosting, served through FastAPI with an HTML/CSS/JS UI.

## Quick start
```bash
pip install -r requirements.txt
python scripts/generate_dataset.py      # synthetic dataset -> data/pc_energy.csv
python scripts/train.py                 # trains 3 models, writes models/ and reports/
uvicorn app.main:app --reload           # open http://127.0.0.1:8000
```

## Structure
| Path | Purpose | Synopsis role |
|---|---|---|
| `scripts/collect_data.py` | Record real telemetry (psutil, RAPL, nvidia-smi) | Data collection |
| `scripts/generate_dataset.py` | Synthetic stand-in dataset | Data collection |
| `scripts/train.py` | Cleaning, split, training, metrics, 6 plots | Preprocessing, ML, Analysis |
| `app/main.py`, `app/static/` | `/api/predict`, `/api/metrics`, web UI | Application |
| `models/`, `reports/` | Saved models, metrics.json, figures for the report | Documentation |

## IMPORTANT: dataset
The shipped `data/pc_energy.csv` is **synthetic** (generated from a physically-motivated formula plus noise).
R² ≈ 0.99 therefore reflects how well models recover that formula, **not** real-world accuracy.
Do not present it as measured data in the report. For real data:
1. Run `python scripts/collect_data.py --samples 600 --out data/real_pc.csv` while doing varied workloads.
   Power comes from Intel RAPL on Linux (CPU package only), a wattmeter via `--manual`, or a HWiNFO /
   LibreHardwareMonitor export you merge in. Alternatively use a public dataset with the same column names.
2. `python scripts/train.py --data data/real_pc.csv` and restart the server. Nothing else changes.

Columns: `cpu_usage, gpu_usage, ram_usage, cpu_freq_ghz, cpu_temp_c, gpu_temp_c, disk_activity, network_activity, power_watts`.

## Metrics (synopsis list)
MAE, MSE, RMSE, R², training time and prediction time are computed in `train.py` and shown in the UI.

## API
`POST /api/predict` with `{"cpu_usage":75,"gpu_usage":60,"ram_usage":70,"cpu_freq_ghz":3.2,"cpu_temp_c":70}`
returns watts, per-model predictions, kWh/CO₂/cost estimates and energy-saving tips.
Interactive docs at `/docs`. CO₂ factor (0.71 kg/kWh) and tariff (₹8/kWh) are assumptions in `app/main.py`; adjust them.

## Remaining for the report
Real-data run, screenshots (`reports/` + UI), literature review, architecture diagram, green-computing analysis.
