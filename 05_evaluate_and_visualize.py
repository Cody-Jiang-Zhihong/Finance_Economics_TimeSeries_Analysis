#!/usr/bin/env python3
# Author: Cody
# Week 5 + 6: Evaluate forecasts (MAE, RMSE, MAPE) and visualize with confidence bands

import argparse
import os
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR_DEFAULT = "outputs/predictions"
OUT_DIR_DEFAULT = "outputs"
FIG_SUBDIR = "figures"
CLEAN_PATH_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "cleaned.parquet")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with safe handling of zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    perc_err = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.nanmean(perc_err))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape(y_true, y_pred),
    }


def plot_actual_vs_pred(df: pd.DataFrame, title: str, out_path: str) -> None:
    """
    Plot actual vs predicted with a simple 95% confidence band
    based on residual standard deviation.
    df: DataFrame with two columns: [actual, predicted]
    """
    if df.shape[1] < 2:
        return

    actual = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    pred = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    # Align indices and drop NaNs
    common_idx = actual.index.intersection(pred.index)
    actual = actual.loc[common_idx].dropna()
    pred = pred.loc[common_idx].dropna()
    common_idx = actual.index.intersection(pred.index)
    actual = actual.loc[common_idx]
    pred = pred.loc[common_idx]

    if len(actual) == 0:
        return

    # Residual-based constant-variance 95% band
    residuals = actual - pred
    sigma = residuals.std(ddof=1)
    upper = lower = None
    if np.isfinite(sigma) and sigma > 0:
        upper = pred + 1.96 * sigma
        lower = pred - 1.96 * sigma

    plt.figure(figsize=(10, 4))
    actual.plot(label="Actual")
    pred.plot(label="Predicted")
    if upper is not None:
        plt.fill_between(pred.index, lower, upper, alpha=0.2, label="95% band")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_model_indicator(name_no_ext: str) -> Dict[str, str]:
    """
    Try to parse model name and indicator name from a file stem.
    Examples:
      naive_Stock Index      -> model='naive', indicator='Stock Index'
      linlags_Close Price    -> model='linlags', indicator='Close Price'
      var_levels             -> model='var',   indicator='' (handled later)
    """
    parts = name_no_ext.split("_", 1)
    if len(parts) == 1:
        return {"model": parts[0], "indicator": ""}
    return {"model": parts[0], "indicator": parts[1]}


def evaluate_predictions(pred_dir: str, out_dir: str, clean_path: str) -> None:
    """Main evaluation routine: iterate over prediction files, compute metrics, plot."""
    if not os.path.isdir(pred_dir):
        print(f"[Info] Prediction directory '{pred_dir}' not found. Run 04_models.py first.")
        return

    fig_dir = os.path.join(out_dir, FIG_SUBDIR)
    os.makedirs(fig_dir, exist_ok=True)

    # Load cleaned data if available (for VAR evaluation)
    if os.path.exists(clean_path):
        cleaned_df = pd.read_parquet(clean_path)
    else:
        cleaned_df = None
        print(
            f"[Warn] Cleaned dataset '{clean_path}' not found. "
            "VAR forecasts will be visualized without metrics."
        )

    rows: List[Dict[str, float]] = []

    parquet_files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if not parquet_files:
        print(f"[Info] No .parquet files found under '{pred_dir}'.")
        return

    for path in parquet_files:
        name = os.path.basename(path)
        stem = name.replace(".parquet", "")
        print(f"[Diag] Processing prediction file: {name}")
        try:
            dfp = pd.read_parquet(path)
        except Exception as e:
            print(f"[Warn] Skipping {name} (failed to read parquet): {e}")
            continue

        if dfp.empty:
            print(f"[Info] {name} is empty, skipping.")
            continue

        info = parse_model_indicator(stem)
        cols_lower = [c.lower() for c in dfp.columns]

        # ---- Case 1: standard univariate predictions with y_true / y_pred ----
        if "y_true" in cols_lower and "y_pred" in cols_lower:
            idx_true = cols_lower.index("y_true")
            idx_pred = cols_lower.index("y_pred")
            y_true = pd.to_numeric(dfp.iloc[:, idx_true], errors="coerce")
            y_pred = pd.to_numeric(dfp.iloc[:, idx_pred], errors="coerce")

            # Drop NaNs and align
            common_idx = y_true.index.intersection(y_pred.index)
            y_true = y_true.loc[common_idx].dropna()
            y_pred = y_pred.loc[common_idx].dropna()
            common_idx = y_true.index.intersection(y_pred.index)
            y_true = y_true.loc[common_idx]
            y_pred = y_pred.loc[common_idx]

            if len(y_true) == 0:
                print(f"[Info] {name}: no valid overlapping y_true/y_pred, skipping.")
                continue

            metrics = compute_metrics(y_true.values, y_pred.values)
            metrics["file"] = stem
            metrics["model"] = info["model"]
            metrics["indicator"] = info["indicator"] or "target"

            rows.append(metrics)

            out_path = os.path.join(fig_dir, f"{stem}.png")
            df_plot = pd.DataFrame(
                {"Actual": y_true, "Predicted": y_pred},
                index=y_true.index,
            )
            plot_actual_vs_pred(df_plot, stem, out_path)
            continue

        # ---- Case 2: VAR multi-series forecast (no explicit y_true/y_pred) ----
        is_var_file = "var" in name.lower()
        if is_var_file and cleaned_df is not None:
            # For each overlapping column, compare forecast vs actual
            for c in dfp.columns:
                if c not in cleaned_df.columns:
                    continue

                pred_series = pd.to_numeric(dfp[c], errors="coerce")
                actual_series = pd.to_numeric(cleaned_df[c], errors="coerce")

                # Align on time index
                common_idx = pred_series.index.intersection(actual_series.index)
                pred_series = pred_series.loc[common_idx].dropna()
                actual_series = actual_series.loc[common_idx].dropna()
                common_idx = pred_series.index.intersection(actual_series.index)
                pred_series = pred_series.loc[common_idx]
                actual_series = actual_series.loc[common_idx]

                if len(actual_series) < 5:
                    continue

                met = compute_metrics(actual_series.values, pred_series.values)
                met["file"] = f"{stem}_{c}"
                met["model"] = "VAR"
                met["indicator"] = c
                rows.append(met)

                out_path = os.path.join(fig_dir, f"{stem}_{c}.png")
                df_plot = pd.DataFrame(
                    {"Actual": actual_series, "Predicted": pred_series},
                    index=actual_series.index,
                )
                title = f"{stem.upper()}: {c}"
                plot_actual_vs_pred(df_plot, title, out_path)
        elif is_var_file and cleaned_df is None:
            # Just plot the VAR forecasts themselves without metrics
            for c in dfp.columns:
                series = pd.to_numeric(dfp[c], errors="coerce").dropna()
                if series.empty:
                    continue
                plt.figure(figsize=(10, 4))
                series.plot(label="Forecast")
                plt.title(f"{stem.upper()}: {c}")
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(fig_dir, f"{stem}_{c}.png")
                plt.savefig(out_path)
                plt.close()
        else:
            print(f"[Info] {name}: no y_true/y_pred columns and not treated as VAR. Skipping metrics.")

    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_csv = os.path.join(out_dir, "metrics_summary.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"[OK] Wrote metrics to {metrics_csv} and saved figures to {fig_dir}")

        # ✅ Week 5: ranking by accuracy
        try:
            print("\n[Summary] Best models per indicator (by RMSE):")
            by_ind = metrics_df.sort_values("RMSE").groupby("indicator", dropna=True)
            summary_rows = []
            for ind, g in by_ind:
                best = g.iloc[0]
                summary_rows.append(
                    {
                        "indicator": ind,
                        "best_model": best["model"],
                        "RMSE": best["RMSE"],
                        "MAE": best["MAE"],
                        "MAPE": best["MAPE"],
                        "file": best["file"],
                    }
                )
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                print(summary_df.to_string(index=False))
        except Exception as e:
            print("[Warn] Failed to print ranking summary:", e)
    else:
        print("[Info] No evaluable prediction series found; metrics_summary.csv not created.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate forecasts and visualize predictions.")
    ap.add_argument("--pred_dir", default=PRED_DIR_DEFAULT, help="Directory containing prediction .parquet files.")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Base output directory.")
    ap.add_argument("--clean_path", default=CLEAN_PATH_DEFAULT, help="Path to cleaned.parquet for VAR evaluation.")
    args = ap.parse_args()

    evaluate_predictions(args.pred_dir, args.out_dir, args.clean_path)


if __name__ == "__main__":
    main()
