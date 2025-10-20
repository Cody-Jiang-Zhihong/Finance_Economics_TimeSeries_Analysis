#!/usr/bin/env python3
# Week 1: Data Understanding & Cleaning (Robust + Windows-friendly)
# Author: Cody & Adam (consolidated)
#
# Features:
# - Flexible date parsing (auto or via --date_col)
# - Duplicate handling on index or specified column
# - Numeric coercion for object columns (handles commas, %)
# - Missingness summary and largest gap spans per column
# - Optional resampling (--resample ME/D/Q) with aggregation (--agg mean/last/sum)
# - Simple z-score outlier report for selected columns
# - Cleaned outputs to outputs/cleaned.parquet and outputs/cleaned.csv
#
# Usage example (Windows PowerShell):
#   python 01_data_understanding_cleaning_final.py --csv ".\finance_economics_dataset.csv" --start 2000-01-01 --end 2008-12-31 --resample ME --agg mean

import argparse, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import detect_date_col, coerce_datetime, summarize_missingness, zscore_outliers, to_monthly, to_quarterly

OUT_DIR = "outputs"

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string numeric-like columns to floats. Handles commas and percent signs."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            s = s.replace(r"[,\s]", "", regex=True)
            is_pct = s.str.endswith("%")
            s = s.str.replace("%", "", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if is_pct.any():
                num[is_pct] = num[is_pct] / 100.0
            df[c] = num
    return df

def largest_gap_span(s: pd.Series) -> int:
    """Return length (in rows) of the largest consecutive NaN gap in a series."""
    if s.isna().sum() == 0:
        return 0
    isna = s.isna().astype(int)
    max_run, cur = 0, 0
    for v in isna:
        if v == 1:
            cur += 1
            if cur > max_run: max_run = cur
        else:
            cur = 0
    return int(max_run)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to Finance & Economics dataset CSV")
    ap.add_argument("--date_col", type=str, default=None, help="Explicit date column (optional)")
    ap.add_argument("--start", type=str, default="2000-01-01", help="Start date filter (inclusive)")
    ap.add_argument("--end", type=str, default="2008-12-31", help="End date filter (inclusive)")
    ap.add_argument("--drop_dupes_on", type=str, default=None, help="Column name for duplicate drop; default = date index")
    ap.add_argument("--resample", type=str, default=None, choices=["ME","D","Q"], help="Optional resample rule: ME=month-end, D=daily, Q=quarterly")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","last","sum"], help="Aggregation for resample")
    ap.add_argument("--outlier_cols", type=str, nargs="*", default=[], help="Columns to check for z-score outliers (threshold=4.0)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Identify date column
    date_col = args.date_col or detect_date_col(df)
    print(f"[Info] Using date column: {date_col}")
    df = coerce_datetime(df, date_col)

    # Filter to range
    df = df.loc[args.start:args.end]

    # Coerce numerics
    df = coerce_numeric_columns(df)

    # Basic structure
    print("\n=== Structure ===")
    print(df.info())
    print("\nHead:\n", df.head())
    print("\nTail:\n", df.tail())

    # Deduplicate
    if args.drop_dupes_on and args.drop_dupes_on in df.columns:
        before = len(df)
        df = df[~df[args.drop_dupes_on].duplicated(keep="first")]
        after = len(df)
        print(f"[Info] Dropped {before - after} duplicates using column '{args.drop_dupes_on}'.")
    else:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        after = len(df)
        print(f"[Info] Dropped {before - after} duplicate rows on date index.")

    # Missingness report
    miss = summarize_missingness(df)
    spans = {c: largest_gap_span(df[c]) for c in df.columns}
    miss["largest_nan_gap"] = miss.index.map(spans.get)
    miss.to_csv(os.path.join(OUT_DIR, "missingness_summary.csv"))
    print("\n=== Missingness summary saved to outputs/missingness_summary.csv ===")
    print(miss.head(20))

    # Optional resample
    if args.resample:
        if args.resample == "ME":
            if args.agg == "mean":
                df = df.resample("ME").mean(numeric_only=True)
            elif args.agg == "last":
                df = df.resample("ME").last()
            else:
                df = df.resample("ME").sum(numeric_only=True)
        elif args.resample == "D":
            if args.agg == "mean":
                df = df.resample("D").mean(numeric_only=True)
            elif args.agg == "last":
                df = df.resample("D").last()
            else:
                df = df.resample("D").sum(numeric_only=True)
        elif args.resample == "Q":
            rule = "Q"
            if args.agg == "mean":
                df = df.resample(rule).mean(numeric_only=True)
            elif args.agg == "last":
                df = df.resample(rule).last()
            else:
                df = df.resample(rule).sum(numeric_only=True)
        print(f"[Info] Resampled to {args.resample} with {args.agg} aggregation.")

    # Simple cleaning: forward-fill then backward-fill for gaps
    df_clean = df.sort_index().copy()
    df_clean = df_clean.ffill().bfill()

    # Outlier report (z-score > 4.0) for selected columns
    outlier_rows = []
    for col in args.outlier_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            s = df_clean[col].astype(float)
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd and not np.isnan(sd) and sd != 0.0:
                z = (s - mu) / sd
                mask = z.abs() > 4.0
                if mask.any():
                    flagged = df_clean.loc[mask, [col]].copy()
                    flagged["column"] = col
                    outlier_rows.append(flagged)
    if outlier_rows:
        out_df = pd.concat(outlier_rows, axis=0)
        out_df.to_csv(os.path.join(OUT_DIR, "outliers_report.csv"))
        print(f"[Info] Outliers flagged and saved to outputs/outliers_report.csv (threshold=4.0 z-score).")
    else:
        print("[Info] No outliers flagged (or no valid numeric columns provided in --outlier_cols).")

    # Save cleaned
    df_clean.to_parquet(os.path.join(OUT_DIR, "cleaned.parquet"))
    df_clean.to_csv(os.path.join(OUT_DIR, "cleaned.csv"))
    print("[OK] Saved cleaned data to outputs/cleaned.parquet and cleaned.csv")

if __name__ == "__main__":
    main()
