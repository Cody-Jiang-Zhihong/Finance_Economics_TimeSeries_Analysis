#!/usr/bin/env python3
# Author: Cody
# Models: Naive, MA, Linear(lags), VAR, LSTM — Hardened with diagnostics and robust VAR

import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# DL is optional; if TF not available, catch gracefully
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except Exception:
    TF_OK = False

INP = "outputs/features.parquet"
OUT_DIR = "outputs/predictions"

def build_sequences(series, lookback=21, horizon=1):
    X, y = [], []
    vals = series.values.astype(float)
    for i in range(lookback, len(vals)-horizon+1):
        X.append(vals[i-lookback:i])
        y.append(vals[i+horizon-1])
    return np.array(X), np.array(y)

def auto_pick_target(df: pd.DataFrame):
    # Prefer stock-like columns
    candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in ["close","price","sp500","index","adj_close"])]
    if candidates:
        return candidates[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None

def prepare_var_frame(df: pd.DataFrame, split_date: str, top_k: int = 6):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].copy()
    X = X.dropna(how="all", axis=1)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")
    X = X.sort_index().ffill().bfill()

    # Top-K by variance
    variances = X.var().sort_values(ascending=False)
    X = X[variances.index[:top_k]]

    # Correlation pruning |corr|>0.98
    corr = X.corr().abs()
    upper = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
    if to_drop:
        X = X.drop(columns=to_drop, errors="ignore")

    # Diff + z-score
    X_diff = X.diff().dropna()
    mu = X_diff.mean()
    sd = X_diff.std().replace(0, np.nan)
    Xz = (X_diff - mu) / sd
    Xz = Xz.dropna(axis=1, how="any")

    train = Xz.loc[Xz.index < split_date].copy()
    test = Xz.loc[Xz.index >= split_date].copy()
    return train, test, mu, sd, X.columns.tolist()

def backtransform_forecast(fcz: pd.DataFrame, mu: pd.Series, sd: pd.Series, last_levels: pd.Series) -> pd.DataFrame:
    diffs = fcz * sd + mu
    out = []
    running = last_levels.copy()
    for idx, row in diffs.iterrows():
        running = running + row
        out.append(running.copy())
    out = pd.DataFrame(out, index=fcz.index, columns=fcz.columns)
    return out

def fit_var_backoffs(train: pd.DataFrame, maxlags_list=[5,3,2,1]):
    last_err = None
    for k in maxlags_list:
        try:
            model = VAR(train)
            res = model.fit(maxlags=k, ic=None, trend="c")
            return res
        except Exception as e:
            last_err = e
    raise last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP)
    ap.add_argument("--target", default=None, help="Target column to forecast (if omitted, auto-pick)")
    ap.add_argument("--split_date", default="2007-01-01", help="Train/Test split date")
    ap.add_argument("--ma_window", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--var_topk", type=int, default=4, help="Max series count for VAR (reduce if PD error)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(args.input).dropna(how="all").copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    print(f"[Diag] features shape: {df.shape}")
    print("[Diag] numeric columns (first 20):", numeric_cols[:20])

    # ------- Target selection -------
    target = args.target if args.target in df.columns else auto_pick_target(df)
    if target:
        print(f"[Info] Using target: {target}")
    else:
        print("[Warn] No numeric target found; univariate models will be skipped.")

    # ------- Univariate Baselines + Linear + LSTM -------
    if target and pd.api.types.is_numeric_dtype(df[target]):
        series = df[target].dropna()
        if len(series) > 10:
            y_train = series.loc[series.index < args.split_date]
            y_test = series.loc[series.index >= args.split_date]

            if len(y_train) > 0 and len(y_test) > 0:
                # Naive
                naive_pred = pd.Series([y_train.iloc[-1]] * len(y_test), index=y_test.index)
                pd.DataFrame({"y_true": y_test, "y_pred": naive_pred}).to_parquet(os.path.join(OUT_DIR, f"naive_{target}.parquet"))

                # Moving Average
                ma = series.rolling(args.ma_window).mean()
                ma_pred = ma.loc[y_test.index].fillna(method="bfill")
                pd.DataFrame({"y_true": y_test, "y_pred": ma_pred}).to_parquet(os.path.join(OUT_DIR, f"ma{args.ma_window}_{target}.parquet"))

                # Linear Regression with lags
                lag_cols = [c for c in df.columns if c.startswith(f"{target}_lag_")]
                if lag_cols:
                    X_train = df.loc[y_train.index, lag_cols].dropna()
                    y_train_aligned = y_train.loc[X_train.index]
                    X_test = df.loc[y_test.index, lag_cols].fillna(method="ffill").fillna(0.0)
                    if len(X_train) > 10:
                        lr = LinearRegression()
                        lr.fit(X_train, y_train_aligned)
                        lr_pred = pd.Series(lr.predict(X_test), index=X_test.index)
                        pd.DataFrame({"y_true": y_test.loc[lr_pred.index], "y_pred": lr_pred}).to_parquet(os.path.join(OUT_DIR, f"linlags_{target}.parquet"))

                # LSTM (fixed: proper scaling + clean date comparison + time-series-aware training)
                if TF_OK:
                    try:
                        # 1) Full series, sorted, no NaNs
                        series_full = df[target].astype(float).sort_index().dropna()

                        # Use pandas Timestamp, not datetime.date / np.datetime64 mix
                        split_dt = pd.to_datetime(args.split_date)

                        series_train = series_full[series_full.index < split_dt]
                        series_test  = series_full[series_full.index >= split_dt]

                        # Need enough points to build sequences
                        if len(series_train) > args.lookback + 10 and len(series_test) > 0:
                            # 2) Scale using TRAIN stats only
                            mu = series_train.mean()
                            sigma = series_train.std()
                            if sigma == 0 or np.isnan(sigma):
                                sigma = 1.0

                            series_train_s = (series_train - mu) / sigma
                            series_test_s  = (series_test - mu) / sigma

                            def make_sequences(s: pd.Series, lookback: int):
                                vals = s.values.astype("float32")
                                Xs, ys, idxs = [], [], []
                                for i in range(lookback, len(vals)):
                                    Xs.append(vals[i - lookback:i])
                                    ys.append(vals[i])
                                    idxs.append(s.index[i])
                                return np.array(Xs), np.array(ys), np.array(idxs)

                            # 3) Build training sequences
                            X_train_dl, y_train_dl, dates_train = make_sequences(
                                series_train_s, args.lookback
                            )

                            # 4) Build test sequences with train tail as context
                            tail_for_context = series_train_s.iloc[-args.lookback:]
                            test_with_context = pd.concat([tail_for_context, series_test_s])
                            X_all_test, y_all_test, dates_all_test = make_sequences(
                                test_with_context, args.lookback
                            )

                            # Keep only the sequences whose target date >= split_dt (true test)
                            mask_test = dates_all_test >= split_dt
                            X_test_dl  = X_all_test[mask_test]
                            y_test_dl  = y_all_test[mask_test]
                            dates_test = dates_all_test[mask_test]

                            if len(X_train_dl) > 50 and len(X_test_dl) > 0:
                                model = Sequential(
                                    [
                                        LSTM(32, input_shape=(X_train_dl.shape[1], 1)),
                                        Dense(1),
                                    ]
                                )
                                model.compile(optimizer="adam", loss="mse")

                                # Time-based validation: last 10% of training data
                                val_idx = int(len(X_train_dl) * 0.9)
                                X_tr, X_val = X_train_dl[:val_idx], X_train_dl[val_idx:]
                                y_tr, y_val = y_train_dl[:val_idx], y_train_dl[val_idx:]

                                cb = EarlyStopping(
                                    monitor="val_loss",
                                    patience=5,
                                    restore_best_weights=True,
                                )

                                model.fit(
                                    X_tr[..., None],
                                    y_tr,
                                    validation_data=(X_val[..., None], y_val),
                                    epochs=args.epochs,
                                    batch_size=args.batch,
                                    callbacks=[cb],
                                    verbose=0,
                                    shuffle=False,  # critical for time series
                                )

                                # 5) Predict in scaled space, then unscale
                                y_pred_scaled = model.predict(X_test_dl[..., None], verbose=0).ravel()
                                y_true_scaled = y_test_dl

                                y_pred = y_pred_scaled * sigma + mu
                                y_true = y_true_scaled * sigma + mu

                                pd.DataFrame(
                                    {"y_true": y_true, "y_pred": y_pred},
                                    index=dates_test,
                                ).to_parquet(
                                    os.path.join(OUT_DIR, f"lstm_{target}.parquet")
                                )
                                print(f"[OK] LSTM predictions saved for {target}")
                            else:
                                print("[Warn] Not enough sequences for LSTM after splitting.")
                        else:
                            print("[Warn] Not enough data for LSTM (train/test) on target:", target)
                    except Exception as e:
                        print("[Warn] LSTM training skipped:", e)


        else:
            print("[Warn] Target series too short for univariate models.")
    else:
        print("[Info] Skipping univariate ML/DL — no suitable target. Pass --target \"Close Price\" etc.")

    # ------- VAR (Multivariate) -------
    try:
        train_m, test_m, mu, sd, kept_cols = prepare_var_frame(df, args.split_date, top_k=args.var_topk)
        if len(train_m) > 50 and len(test_m) > 0 and train_m.shape[1] >= 2:
            res = fit_var_backoffs(train_m, maxlags_list=[5,3,2,1])
            steps = len(test_m)
            fc = res.forecast(y=train_m.values[-res.k_ar:], steps=steps)
            fc_df = pd.DataFrame(fc, index=test_m.index, columns=train_m.columns)

            last_levels = df[kept_cols].sort_index().ffill().bfill().iloc[-1]
            levels = backtransform_forecast(fc_df, mu[fc_df.columns], sd[fc_df.columns], last_levels[fc_df.columns])

            fc_df.to_parquet(os.path.join(OUT_DIR, "var_forecast_diffz.parquet"))
            levels.to_parquet(os.path.join(OUT_DIR, "var_forecast_levels.parquet"))
            print(f"[OK] VAR forecasts saved for {len(fc_df.columns)} series (diff-z and levels).")
        else:
            print("[Warn] VAR skipped — not enough data/columns after cleaning.")
    except Exception as e:
        print("[Warn] VAR training skipped:", e)

    # Status file
    try:
        with open(os.path.join(OUT_DIR, "status.txt"), "w", encoding="utf-8") as f:
            f.write("Run completed. Check parquet files in this folder.\n")
    except Exception:
        pass
    print("[OK] Predictions (if any) saved under outputs/predictions")

if __name__ == "__main__":
    main()
