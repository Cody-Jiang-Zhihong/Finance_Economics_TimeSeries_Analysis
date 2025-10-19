
#!/usr/bin/env python3
# Author: Cody
# Week 1: Data Understanding & Cleaning

import argparse, os
import pandas as pd
import numpy as np
from src.utils import detect_date_col, coerce_datetime, summarize_missingness

DEFAULT_CSV = "/mnt/data/finance_economics_dataset.csv"
OUT_DIR = "outputs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to Finance & Economics dataset CSV")
    ap.add_argument("--start", type=str, default="2000-01-01", help="Start date filter (inclusive)")
    ap.add_argument("--end", type=str, default="2008-12-31", help="End date filter (inclusive)")
    ap.add_argument("--drop_dupes_on", type=str, default=None, help="Column name for duplicate drop; default = index(date)+all cols")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    date_col = detect_date_col(df)
    print(f"[Info] Using date column: {date_col}")
    df = coerce_datetime(df, date_col)

    # Range filter
    df = df.loc[args.start:args.end]

    # Basic structure
    print("\n=== Structure ===")
    print(df.info())
    print("\nHead:\n", df.head())
    print("\nTail:\n", df.tail())

    # Duplicates
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

    # Missingness
    miss = summarize_missingness(df)
    miss.to_csv(os.path.join(OUT_DIR, "missingness_summary.csv"))
    print("\n=== Missingness summary saved to outputs/missingness_summary.csv ===")
    print(miss.head(15))

    # Simple cleaning: forward-fill then backward-fill for gaps
    df_clean = df.sort_index().copy()
    df_clean = df_clean.ffill().bfill()

    # Save cleaned
    df_clean.to_parquet(os.path.join(OUT_DIR, "cleaned.parquet"))
    df_clean.to_csv(os.path.join(OUT_DIR, "cleaned.csv"))
    print("[OK] Saved cleaned data to outputs/cleaned.parquet and cleaned.csv")

if __name__ == "__main__":
    main()
