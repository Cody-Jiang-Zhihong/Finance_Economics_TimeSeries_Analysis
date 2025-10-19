
#!/usr/bin/env python3
# Author: Cody
# Week 3: Feature Engineering (lags, rolling stats, returns, spreads, differencing)

import argparse, os
import pandas as pd
import numpy as np
from src.utils import add_returns, add_rolling_stats, add_lags, add_spread, safe_cols

INP = "outputs/cleaned.parquet"
OUTP = "outputs/features.parquet"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP)
    ap.add_argument("--target", default=None, help="Target column for univariate features (optional)")
    ap.add_argument("--lags", type=int, nargs="+", default=[1,5,21])
    ap.add_argument("--windows", type=int, nargs="+", default=[5,21,63])
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    feat = df.copy()

    # Add returns/log returns to stock-like columns
    stock_like = [c for c in num_cols if any(k in c.lower() for k in ["close","price","sp500","index","adj"])]
    for c in stock_like:
        feat = add_returns(feat, c, log=True)
        feat = add_rolling_stats(feat, c, windows=tuple(args.windows))
        feat = add_lags(feat, c, lags=tuple(args.lags))

    # Add rates spreads if plausible (10y - 2y style) when columns hint
    cols_lower = {c.lower(): c for c in num_cols}
    # crude heuristics
    cand_10y = [cols_lower[c] for c in cols_lower if "10" in c and "y" in c or "ten_year" in c]
    cand_2y  = [cols_lower[c] for c in cols_lower if ("2" in c and "y" in c) or "two_year" in c]
    if len(cand_10y) > 0 and len(cand_2y) > 0:
        feat = add_spread(feat, cand_10y[0], cand_2y[0], name="term_spread_10y_2y")

    # Differencing for non-stationary macro columns
    macro_like = [c for c in num_cols if any(k in c.lower() for k in ["gdp","infl","cpi","unemp","rate"])]
    for c in macro_like:
        feat[f"{c}_diff1"] = feat[c].diff()
        feat[f"{c}_diff2"] = feat[c].diff(2)

    # If target provided, ensure lag features for it
    if args.target and args.target in feat.columns:
        feat = add_lags(feat, args.target, lags=tuple(args.lags))

    feat.to_parquet(OUTP)
    feat.to_csv("outputs/features.csv")
    print("[OK] Feature set saved to outputs/features.parquet and features.csv")

if __name__ == "__main__":
    main()
