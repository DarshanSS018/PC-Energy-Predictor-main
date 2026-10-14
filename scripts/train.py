"""Preprocess -> feature selection -> train/compare models -> plots + saved artifacts.

Usage: python scripts/train.py [--data data/pc_energy.csv]
Outputs: models/*.joblib, models/metrics.json, models/meta.json, reports/*.png
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["cpu_usage", "gpu_usage", "ram_usage", "cpu_freq_ghz",
            "cpu_temp_c", "gpu_temp_c", "disk_activity", "network_activity"]
TARGET = "power_watts"
SEED = 42


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    # impute missing sensor columns with the median (e.g. no GPU temp on laptops)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    # clip percentage columns, drop impossible rows
    for c in ["cpu_usage", "gpu_usage", "ram_usage"]:
        df[c] = df[c].clip(0, 100)
    df = df[df[TARGET] > 0]
    # remove extreme target outliers (> 4 std)
    z = (df[TARGET] - df[TARGET].mean()) / df[TARGET].std()
    df = df[z.abs() < 4]
    print(f"Preprocessing: {n0} -> {len(df)} rows")
    return df.reset_index(drop=True)


def evaluate(name, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pred_t = time.perf_counter() - t0
    mse = mean_squared_error(yte, pred)
    m = {
        "model": name,
        "MAE": mean_absolute_error(yte, pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(yte, pred),
        "train_time_s": train_t,
        "predict_time_ms_per_1000": pred_t / max(len(Xte), 1) * 1000 * 1000,
    }
    print(f"{name:20s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.4f} "
          f"train={train_t:.2f}s")
    return m, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "pc_energy.csv"))
    a = ap.parse_args()

    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
    df = preprocess(pd.read_csv(a.data))
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                       learning_rate=0.08, random_state=SEED),
    }
    results, preds = [], {}
    for name, mdl in models.items():
        m, p = evaluate(name, mdl, Xtr, ytr, Xte, yte)
        results.append(m); preds[name] = p
        joblib.dump(mdl, ROOT / "models" / (name.lower().replace(" ", "_") + ".joblib"))

    # Best = lowest RMSE (accuracy); report cost alongside so the trade-off is visible
    best = min(results, key=lambda r: r["RMSE"])["model"]

    # Feature importance (RF impurity-based) + permutation-free linear coefficients
    rf = models["Random Forest"]
    imp = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))
    lr_coef = dict(zip(FEATURES, models["Linear Regression"][-1].coef_.round(3).tolist()))

    ranges = {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES}
    meta = {
        "features": FEATURES, "target": TARGET, "best_model": best,
        "feature_importance": imp, "linear_coefficients_standardised": lr_coef,
        "feature_ranges": ranges, "feature_means": X.mean().round(3).to_dict(),
        "n_rows": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "target_mean_w": float(y.mean()), "target_min_w": float(y.min()),
        "target_max_w": float(y.max()),
        "data_source": Path(a.data).name,
    }
    (ROOT / "models" / "metrics.json").write_text(json.dumps(results, indent=2))
    (ROOT / "models" / "meta.json").write_text(json.dumps(meta, indent=2))

    # ----- plots -----
    sns.set_theme(style="whitegrid", context="notebook")
    rep = ROOT / "reports"

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(rep / "correlation.png", dpi=140); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a_, c in zip(ax.ravel(), ["cpu_usage", "gpu_usage", "ram_usage", "cpu_freq_ghz", "cpu_temp_c", "disk_activity"]):
        a_.scatter(df[c], df[TARGET], s=4, alpha=.35, color="#2a9d8f"); a_.set_xlabel(c); a_.set_ylabel("Power (W)")
    fig.suptitle("Feature vs power"); fig.tight_layout(); fig.savefig(rep / "feature_vs_power.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    lo, hi = float(yte.min()), float(yte.max())
    for a_, (n, p) in zip(ax, preds.items()):
        a_.scatter(yte, p, s=5, alpha=.4, color="#264653"); a_.plot([lo, hi], [lo, hi], "r--", lw=1)
        a_.set_title(n); a_.set_xlabel("Actual (W)"); a_.set_ylabel("Predicted (W)")
    fig.tight_layout(); fig.savefig(rep / "actual_vs_predicted.png", dpi=140); plt.close(fig)

    res = pd.DataFrame(results).set_index("model")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    res[["MAE", "RMSE"]].plot.bar(ax=ax[0], rot=15, color=["#2a9d8f", "#e76f51"]); ax[0].set_title("Error (W) - lower is better")
    res["R2"].plot.bar(ax=ax[1], rot=15, color="#264653"); ax[1].set_title("R2 - higher is better"); ax[1].set_ylim(min(0.9, res.R2.min() - .02), 1)
    res["train_time_s"].plot.bar(ax=ax[2], rot=15, color="#e9c46a"); ax[2].set_title("Training time (s)")
    for a_ in ax: a_.set_xlabel("")
    fig.tight_layout(); fig.savefig(rep / "model_comparison.png", dpi=140); plt.close(fig)

    s = pd.Series(imp).sort_values()
    plt.figure(figsize=(7, 4.5)); s.plot.barh(color="#2a9d8f"); plt.title("Random Forest feature importance")
    plt.tight_layout(); plt.savefig(rep / "feature_importance.png", dpi=140); plt.close()

    plt.figure(figsize=(7, 4.5))
    for n, p in preds.items(): sns.kdeplot(np.asarray(yte) - p, label=n, fill=True, alpha=.25)
    plt.xlabel("Residual (W)"); plt.title("Error distribution"); plt.legend(); plt.tight_layout()
    plt.savefig(rep / "residuals.png", dpi=140); plt.close()

    print(f"\nBest model by RMSE: {best}\nSaved models, metrics and 6 plots.")


if __name__ == "__main__":
    main()
