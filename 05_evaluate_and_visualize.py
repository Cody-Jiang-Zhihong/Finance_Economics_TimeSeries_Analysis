
#!/usr/bin/env python3
# Author: Cody
# Evaluate predictions (MAE, RMSE, MAPE) and plot actual vs predicted (matplotlib)

import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import compute_metrics

PRED_DIR = "outputs/predictions"
OUT_DIR = "outputs"

def eval_file(path: str) -> dict:
    df = pd.read_parquet(path)
    # normalize column names
    cols = [c.lower() for c in df.columns]
    if "y_true" in cols and "y_pred" in cols:
        y_true = df[df.columns[cols.index("y_true")]].astype(float)
        y_pred = df[df.columns[cols.index("y_pred")]].astype(float)
        met = compute_metrics(y_true, y_pred)
        return met
    else:
        # e.g., VAR forecast over multiple columns — skip in this loop
        return {}

def plot_actual_vs_pred(df: pd.DataFrame, title: str, fname: str):
    plt.figure(figsize=(10,4))
    df.iloc[:,0].plot(label="Actual")
    df.iloc[:,1].plot(label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    figpath = os.path.join("outputs/figures", fname)
    plt.savefig(figpath)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=PRED_DIR)
    args = ap.parse_args()

    os.makedirs("outputs/figures", exist_ok=True)

    metrics_rows = []
    for path in glob.glob(os.path.join(args.pred_dir, "*.parquet")):
        name = os.path.basename(path)
        df = pd.read_parquet(path)

        # Try standard y_true/y_pred pairs
        if set(["y_true","y_pred"]).issubset(set([c.lower() for c in df.columns])):
            cols_lower = [c.lower() for c in df.columns]
            actual = df[df.columns[cols_lower.index("y_true")]]
            pred = df[df.columns[cols_lower.index("y_pred")]]
            met = eval_file(path)
            met["file"] = name
            metrics_rows.append(met)

            # plot
            plotdf = pd.concat([actual, pred], axis=1)
            plot_actual_vs_pred(plotdf, f"Actual vs Predicted: {name}", fname=name.replace(".parquet",".png"))

        # VAR forecast (multi-column) — compute columnwise metrics if we also have ground truth
        else:
            # If this is VAR output, attempt to compare against cleaned data
            if name == "var_forecast.parquet":
                try:
                    truth = pd.read_parquet("outputs/features.parquet")  # contains original columns
                    pred = df.copy()
                    common = [c for c in pred.columns if c in truth.columns]
                    for c in common:
                        y_true = truth.loc[pred.index, c].astype(float).dropna()
                        y_pred = pred.loc[y_true.index, c].astype(float)
                        met = compute_metrics(y_true, y_pred)
                        met["file"] = f"var_{c}"
                        metrics_rows.append(met)

                        plot_actual_vs_pred(pd.concat([y_true, y_pred], axis=1), f"VAR Forecast: {c}", f"var_{c}.png")
                except Exception as e:
                    print("[Warn] VAR evaluation skipped:", e)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    print("[OK] Wrote metrics to outputs/metrics_summary.csv and saved figures to outputs/figures")

if __name__ == "__main__":
    main()
