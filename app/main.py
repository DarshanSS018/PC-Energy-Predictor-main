"""FastAPI backend: serves predictions, model metrics and the web UI.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if not (MODELS / "meta.json").exists():
    raise SystemExit("No trained models found. Run: python scripts/train.py")

META = json.loads((MODELS / "meta.json").read_text())
METRICS = json.loads((MODELS / "metrics.json").read_text())
FEATURES = META["features"]
LOADED = {m["model"]: joblib.load(MODELS / (m["model"].lower().replace(" ", "_") + ".joblib"))
          for m in METRICS}

# Illustrative only: grid carbon intensity (kg CO2 / kWh) and tariff (INR / kWh).
# India's grid average is roughly 0.7-0.8 kg/kWh; edit to match your own tariff.
CO2_KG_PER_KWH = 0.71
INR_PER_KWH = 8.0

app = FastAPI(title="PC Energy Consumption Predictor", version="1.0")


class PCInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU utilisation %")
    gpu_usage: float = Field(..., ge=0, le=100, description="GPU utilisation %")
    ram_usage: float = Field(..., ge=0, le=100, description="RAM utilisation %")
    cpu_freq_ghz: float = Field(..., ge=0.5, le=6.5, description="CPU frequency (GHz)")
    cpu_temp_c: float = Field(..., ge=20, le=110, description="CPU temperature (C)")
    gpu_temp_c: float = Field(30.0, ge=20, le=110, description="GPU temperature (C)")
    disk_activity: float = Field(0.0, ge=0, le=100, description="Disk activity")
    network_activity: float = Field(0.0, ge=0, le=100, description="Network activity")
    model: str | None = None
    hours_per_day: float = Field(4.0, ge=0, le=24)


def level(watts: float) -> dict:
    lo, hi = META["target_min_w"], META["target_max_w"]
    t = (watts - lo) / max(hi - lo, 1e-9)
    if t < 0.15: return {"label": "Low", "tone": "good"}
    if t < 0.40: return {"label": "Moderate", "tone": "ok"}
    return {"label": "High", "tone": "bad"}


def tips(x: PCInput, watts: float) -> list[str]:
    out = []
    if x.gpu_usage > 70: out.append("GPU is the dominant load: cap frame rate / use V-Sync, or lower graphics settings.")
    if x.cpu_usage > 80: out.append("CPU is heavily loaded: close background processes or switch to a balanced power plan.")
    if x.cpu_freq_ghz > 4.0 and x.cpu_usage < 60: out.append("High clock at moderate load: disabling aggressive turbo boost can save power.")
    if x.cpu_temp_c > 80: out.append("CPU is running hot, which increases leakage power: clean fans/dust and check airflow.")
    if x.ram_usage > 85: out.append("RAM is nearly full: close unused tabs/apps to avoid swap activity.")
    if x.cpu_usage < 15 and x.gpu_usage < 10 and watts > META["target_mean_w"] * 0.6:
        out.append("Near-idle load but notable draw: check for background tasks, and enable sleep/display timeout.")
    if not out:
        if level(watts)["tone"] == "bad":
            out.append("Sustained high draw: use a balanced/eco power plan, limit boost clocks, and take breaks with sleep enabled.")
        else:
            out.append("Usage looks efficient. Keep power-saving mode on and use sleep for long idle periods.")
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "best_model": META["best_model"], "models": list(LOADED)}


@app.get("/api/meta")
def meta():
    return META


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(x: PCInput):
    name = x.model or META["best_model"]
    if name not in LOADED:
        raise HTTPException(400, f"Unknown model '{name}'. Choose from {list(LOADED)}")
    row = pd.DataFrame([{f: getattr(x, f) for f in FEATURES}])
    preds = {n: float(m.predict(row)[0]) for n, m in LOADED.items()}
    watts = max(preds[name], 0.0)
    kwh_day = watts * x.hours_per_day / 1000
    return {
        "model": name,
        "predicted_watts": round(watts, 1),
        "all_models": {n: round(max(v, 0), 1) for n, v in preds.items()},
        "level": level(watts),
        "energy": {
            "kwh_per_day": round(kwh_day, 3),
            "kwh_per_month": round(kwh_day * 30, 2),
            "co2_kg_per_month": round(kwh_day * 30 * CO2_KG_PER_KWH, 2),
            "cost_inr_per_month": round(kwh_day * 30 * INR_PER_KWH, 1),
        },
        "tips": tips(x, watts),
        "note": f"Trained on '{META['data_source']}'. Output is an estimate, not a measurement.",
    }


app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
