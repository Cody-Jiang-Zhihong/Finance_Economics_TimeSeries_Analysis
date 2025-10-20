#!/usr/bin/env python3
# Author: Cody
# Evaluate predictions (MAE, RMSE, MAPE) and plot actual vs predicted

import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR = "outputs/predictions"
OUT_DIR = "outputs"

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape(y_true, y_pred)),
    }

def plot_actual_vs_pred(df: pd.DataFrame, title: str, fname: str):
    os.makedirs("outputs/figures", exist_ok=True)
    plt.figure(figsize=(10,4))
    df.iloc[:,0].plot(label="Actual")
    df.iloc[:,1].plot(label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs/figures", fname))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=PRED_DIR)
    args = ap.parse_args()

    os.makedirs("outputs/figures", exist_ok=True)

    # Try to locate features for VAR comparison
    try:
        truth_full = pd.read_parquet("outputs/features.parquet")
    except Exception:
        truth_full = None

    rows = []
    for path in glob.glob(os.path.join(args.pred_dir, "*.parquet")):
        name = os.path.basename(path)
        df = pd.read_parquet(path)

        lower = [c.lower() for c in df.columns]
        if "y_true" in lower and "y_pred" in lower:
            y_true = df[df.columns[lower.index("y_true")]].astype(float)
            y_pred = df[df.columns[lower.index("y_pred")]].astype(float)
            met = compute_metrics(y_true, y_pred)
            met["file"] = name
            rows.append(met)
            plot_actual_vs_pred(pd.concat([y_true, y_pred], axis=1), f"Actual vs Predicted: {name}", name.replace(".parquet",".png"))
        else:
            # e.g., VAR forecasts
            if truth_full is not None and name in ["var_forecast_levels.parquet", "var_forecast_diffz.parquet"]:
                pred = df.copy()
                common = [c for c in pred.columns if c in truth_full.columns]
                for c in common:
                    y_true = truth_full.loc[pred.index, c].astype(float).dropna()
                    y_pred = pred.loc[y_true.index, c].astype(float)
                    if len(y_true) > 0 and len(y_pred) > 0:
                        met = compute_metrics(y_true, y_pred)
                        met["file"] = f"{name.replace('.parquet','')}_{c}"
                        rows.append(met)
                        plot_actual_vs_pred(pd.concat([y_true, y_pred], axis=1), f"{name.replace('.parquet','').upper()}: {c}", f"{name.replace('.parquet','')}_{c}.png")

    metrics_df = pd.DataFrame(rows)
    if len(metrics_df) > 0:
        metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
        print("[OK] Wrote metrics to outputs/metrics_summary.csv and saved figures to outputs/figures")
    else:
        print("[Info] No evaluable prediction files found under outputs/predictions")

if __name__ == "__main__":
    main()
