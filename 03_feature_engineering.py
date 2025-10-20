#!/usr/bin/env python3
# Author: Cody
# Week 3: Feature Engineering (lags, rolling stats, returns, spreads, differencing)

import argparse, os
import pandas as pd
import numpy as np

INP = "outputs/cleaned.parquet"
OUTP = "outputs/features.parquet"

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
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

def add_returns(df: pd.DataFrame, col: str, log: bool = True) -> pd.DataFrame:
    df = df.copy()
    if log:
        df[f"{col}_logret"] = np.log(df[col]).diff()
    else:
        df[f"{col}_ret"] = df[col].pct_change()
    return df

def add_rolling_stats(df: pd.DataFrame, col: str, windows=(5,21,63)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{col}_sma_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_vol_{w}"] = df[col].rolling(w).std()
    return df

def add_lags(df: pd.DataFrame, col: str, lags=(1,5,21)) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        df[f"{col}_lag_{l}"] = df[col].shift(l)
    return df

def add_spread(df: pd.DataFrame, col_a: str, col_b: str, name=None) -> pd.DataFrame:
    df = df.copy()
    nm = name or f"spread_{col_a}_minus_{col_b}"
    df[nm] = df[col_a] - df[col_b]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP)
    ap.add_argument("--target", default=None, help="Target column for univariate features (optional)")
    ap.add_argument("--lags", type=int, nargs="+", default=[1,5,21])
    ap.add_argument("--windows", type=int, nargs="+", default=[5,21,63])
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df = coerce_numeric_columns(df)

    feat = df.copy()

    # Stock-like features
    stock_like = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c]) and any(k in c.lower() for k in ["close","price","sp500","index","adj"])]
    for c in stock_like:
        feat = add_returns(feat, c, log=True)
        feat = add_rolling_stats(feat, c, windows=tuple(args.windows))
        feat = add_lags(feat, c, lags=tuple(args.lags))

    # Term spread if columns hint (10y vs 2y)
    cols_lower = {c.lower(): c for c in feat.columns}
    cand_10y = [v for k,v in cols_lower.items() if ("10" in k and "y" in k) or ("ten_year" in k)]
    cand_2y  = [v for k,v in cols_lower.items() if (("2" in k and "y" in k) or "two_year" in k)]
    if cand_10y and cand_2y:
        feat = add_spread(feat, cand_10y[0], cand_2y[0], name="term_spread_10y_2y")

    # Macro differencing
    macro_like = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c]) and any(k in c.lower() for k in ["gdp","infl","cpi","unemp","rate","interest"])]
    for c in macro_like:
        feat[f"{c}_diff1"] = feat[c].diff()
        feat[f"{c}_diff2"] = feat[c].diff(2)

    # Ensure target lag features
    if args.target and args.target in feat.columns:
        feat = add_lags(feat, args.target, lags=tuple(args.lags))

    os.makedirs("outputs", exist_ok=True)
    feat.to_parquet(OUTP)
    feat.to_csv("outputs/features.csv")
    print("[OK] Feature set saved to outputs/features.parquet and features.csv")

if __name__ == "__main__":
    main()
