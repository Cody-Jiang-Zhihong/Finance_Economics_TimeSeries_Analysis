
#!/usr/bin/env python3
# Author: Cody
# Week 2: Exploratory Data Analysis (EDA) - Matplotlib only (hardened for mixed/str columns)

import argparse, os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.utils import to_monthly

INP = "outputs/cleaned.parquet"
OUT_FIG = "outputs/figures"

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert object/string columns that look numeric into floats.
       Handles commas, percent signs, and whitespace."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str).str.replace(r"[,\s]", "", regex=True)
            # convert percentages like "3.2%" to 0.032
            is_pct = s.str.endswith("%")
            s_pct = s.str.replace("%", "", regex=False)
            num = pd.to_numeric(s_pct, errors="coerce")
            num[is_pct] = num[is_pct] / 100.0
            df[c] = num
    return df

def plot_series(df, cols, title, fname):
    # Only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[Warn] Skipping '{title}' — none of {cols} found.")
        return
    # Coerce a copy to numeric
    df_plot = df[cols].copy()
    for c in df_plot.columns:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    # Drop columns that are all NaN after coercion
    df_plot = df_plot.dropna(how="all", axis=1)
    if df_plot.shape[1] == 0:
        print(f"[Warn] Skipping '{title}' — no numeric data after coercion.")
        return
    plt.figure(figsize=(10,4))
    for c in df_plot.columns:
        df_plot[c].plot(label=c)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.savefig(os.path.join(OUT_FIG, fname))
    plt.close()

def plot_corr_heatmap(df, fname):
    num = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1).dropna()
    if num.shape[1] == 0:
        print("[Warn] Skipping correlation heatmap — no numeric columns.")
        return
    corr = num.corr()
    fig = plt.figure(figsize=(max(6, 0.6 * len(corr.columns)), 5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.savefig(os.path.join(OUT_FIG, fname))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP, help="Cleaned parquet path")
    ap.add_argument("--monthly", action="store_true", help="Resample to monthly for macro decomposition")
    args = ap.parse_args()

    os.makedirs(OUT_FIG, exist_ok=True)

    df = pd.read_parquet(args.input)
    # Coerce any numeric-like object columns to floats
    df = coerce_numeric_columns(df)

    if args.monthly:
        dfm = to_monthly(df)
        dfm.to_parquet("outputs/monthly.parquet")
    else:
        dfm = df.copy()

    # Detect groups by name (case-insensitive)
    lower_cols = {c.lower(): c for c in dfm.columns}
    def has_any(c, keys): return any(k in c.lower() for k in keys)

    gdp_cols   = [c for c in dfm.columns if has_any(c, ["gdp"])]
    infl_cols  = [c for c in dfm.columns if has_any(c, ["infl", "cpi"])]
    unemp_cols = [c for c in dfm.columns if has_any(c, ["unemp"])]
    rate_cols  = [c for c in dfm.columns if has_any(c, ["rate","ffr","yield","treasury"])]
    stock_cols = [c for c in dfm.columns if has_any(c, ["close","price","sp500","index","adj_close"])]

    plot_series(dfm, gdp_cols[:3], "GDP-related series", "gdp_series.png")
    plot_series(dfm, infl_cols[:3], "Inflation-related series", "inflation_series.png")
    plot_series(dfm, unemp_cols[:3], "Unemployment-related series", "unemployment_series.png")
    plot_series(dfm, rate_cols[:3], "Interest rate-related series", "rates_series.png")
    plot_series(dfm, stock_cols[:3], "Stock-related series", "stocks_series.png")

    # Correlation heatmap (numeric only)
    plot_corr_heatmap(dfm, "correlation_heatmap.png")

    # Seasonal decomposition on first macro column (if monthly and enough data)
    try:
        macro_candidates = gdp_cols + infl_cols + unemp_cols + rate_cols
        macro_candidates = [c for c in macro_candidates if c in dfm.columns and pd.api.types.is_numeric_dtype(dfm[c])]
        if len(macro_candidates) > 0:
            series = dfm[macro_candidates[0]].dropna()
            if len(series) >= 36:
                res = seasonal_decompose(series, model="additive", period=12)
                fig = res.plot()
                fig.set_size_inches(10,6)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_FIG, "seasonal_decompose.png"))
                plt.close(fig)
    except Exception as e:
        print("[Warn] Seasonal decomposition skipped:", e)

    print("[OK] EDA figures saved to outputs/figures")

if __name__ == "__main__":
    main()
