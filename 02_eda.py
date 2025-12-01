#!/usr/bin/env python3
# Author: Cody
# Week 2: Exploratory Data Analysis (EDA) - Hardened

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

INP = "outputs/cleaned.parquet"
OUT_FIG = "outputs/figures"

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string numeric-like columns to floats. Handles commas and percent signs."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str).str.replace(r"[,\s]", "", regex=True)
            is_pct = s.str.endswith("%")
            s = s.str.replace("%", "", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if is_pct.any():
                num[is_pct] = num[is_pct] / 100.0
            df[c] = num
    return df

def plot_series(df, cols, title, fname):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[Warn] Skipping '{title}' — none of {cols} found.")
        return
    df_plot = df[cols].copy()
    for c in df_plot.columns:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(how="all", axis=1)
    if df_plot.shape[1] == 0:
        print(f"[Warn] Skipping '{title}' — no numeric data after coercion.")
        return
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.figure(figsize=(10,4))
    for c in df_plot.columns:
        df_plot[c].plot(label=c)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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

def plot_cross_correlation(df, col_x, col_y, max_lag=24, fname="ccf.png"):
    """
    Plot cross-correlation function between two series with lags from -max_lag to +max_lag.
    """
    if col_x not in df.columns or col_y not in df.columns:
        print(f"[Warn] Cross-corr skipped — {col_x} or {col_y} not in columns.")
        return

    s1 = pd.to_numeric(df[col_x], errors="coerce").dropna()
    s2 = pd.to_numeric(df[col_y], errors="coerce").dropna()

    common_idx = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common_idx], s2.loc[common_idx]

    if len(s1) < max_lag * 2:
        print("[Warn] Cross-corr skipped — series too short.")
        return

    s1 = (s1 - s1.mean()) / s1.std(ddof=0)
    s2 = (s2 - s2.mean()) / s2.std(ddof=0)

    lags = range(-max_lag, max_lag + 1)
    ccs = []
    for k in lags:
        if k < 0:
            # s1 leads, s2 lags
            cc = (s1[:k] * s2[-k:]).mean()
        elif k > 0:
            # s2 leads, s1 lags
            cc = (s1[k:] * s2[:-k]).mean()
        else:
            cc = (s1 * s2).mean()
        ccs.append(cc)

    os.makedirs(OUT_FIG, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.stem(list(lags), ccs, use_line_collection=True)
    plt.axhline(0, linestyle="--")
    plt.title(f"Cross-correlation: {col_x} vs {col_y}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, fname))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP, help="Cleaned parquet path")
    ap.add_argument("--monthly", action="store_true", help="Resample to month-end for macro decomposition")
    args = ap.parse_args()

    os.makedirs(OUT_FIG, exist_ok=True)
    df = pd.read_parquet(args.input)
    df = coerce_numeric_columns(df)

    if args.monthly:
        dfm = df.resample("ME").mean(numeric_only=True)
        dfm.to_parquet("outputs/monthly.parquet")
    else:
        dfm = df.copy()

    # Buckets by name
    def has_any(c, keys): return any(k in c.lower() for k in keys)
    gdp_cols   = [c for c in dfm.columns if has_any(c, ["gdp"])]
    infl_cols  = [c for c in dfm.columns if has_any(c, ["infl","cpi"])]
    unemp_cols = [c for c in dfm.columns if has_any(c, ["unemp"])]
    rate_cols  = [c for c in dfm.columns if has_any(c, ["rate","ffr","yield","treasury","interest"])]
    stock_cols = [c for c in dfm.columns if has_any(c, ["close","price","sp500","index","adj_close"])]

    plot_series(dfm, gdp_cols[:3], "GDP-related series", "gdp_series.png")
    plot_series(dfm, infl_cols[:3], "Inflation-related series", "inflation_series.png")
    plot_series(dfm, unemp_cols[:3], "Unemployment-related series", "unemployment_series.png")
    plot_series(dfm, rate_cols[:3], "Interest rate-related series", "rates_series.png")
    plot_series(dfm, stock_cols[:3], "Stock-related series", "stocks_series.png")

    plot_corr_heatmap(dfm, "correlation_heatmap.png")

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

    #//Rajdeep Commit//
        # --- ACF & PACF plots for series ---
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    keywords = ["gdp", "infl", "unemp", "rate"]
    for key in keywords:
        match = [c for c in dfm.columns if key.lower() in c.lower()]
        if not match:
            print(f"[Info] No match for '{key}'")
            continue
        col = match[0]

        series = pd.to_numeric(dfm[col], errors='coerce').dropna()
        series = series.resample('ME').mean().interpolate()

        if len(series) < 24:
            print(f"[Warn] Skipping {col} (too few data points).")
            continue

        print(f"[OK] ACF/PACF: {col}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(series, ax=axes[0], lags=40)
        axes[0].set_title(f"ACF: {col}")
        plot_pacf(series, ax=axes[1], lags=40, method='ywm')
        axes[1].set_title(f"PACF: {col}")
        for ax in axes:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Show more y-ticks
        plt.tight_layout()

        os.makedirs(OUT_FIG, exist_ok=True)
        plt.savefig(os.path.join(OUT_FIG, f"acf_pacf_{col}.png"))
        plt.close()
        
        # --- Cross-correlation: one macro vs one stock (if available) ---
        if stock_cols and rate_cols:
            cc_x, cc_y = rate_cols[0], stock_cols[0]
            print(f"[OK] Cross-corr between {cc_x} (rates) and {cc_y} (stocks)")
            plot_cross_correlation(dfm, cc_x, cc_y, max_lag=12, fname="ccf_rates_vs_stocks.png")
        elif stock_cols and infl_cols:
            cc_x, cc_y = infl_cols[0], stock_cols[0]
            print(f"[OK] Cross-corr between {cc_x} (inflation) and {cc_y} (stocks)")
            plot_cross_correlation(dfm, cc_x, cc_y, max_lag=12, fname="ccf_infl_vs_stocks.png")
        else:
            print("[Info] No suitable pair for cross-correlation found.")


    print("[OK] EDA figures saved to outputs/figures")

if __name__ == "__main__":
    main()
