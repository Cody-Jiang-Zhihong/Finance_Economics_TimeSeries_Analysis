
#!/usr/bin/env python3
# Author: Cody
# Baseline & Advanced Models: Naive, MovingAverage, Linear (lags), VAR, LSTM (Keras)

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from src.utils import train_test_split_by_date, compute_metrics, safe_cols
from statsmodels.tsa.api import VAR
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

INP = "outputs/features.parquet"
OUT_DIR = "outputs/predictions"

def build_sequences(series, lookback=21, horizon=1):
    X, y = [], []
    vals = series.values.astype(float)
    for i in range(lookback, len(vals)-horizon+1):
        X.append(vals[i-lookback:i])
        y.append(vals[i+horizon-1])
    return np.array(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INP)
    ap.add_argument("--target", default=None, help="Target column to forecast (required for univariate ML/DL)")
    ap.add_argument("--split_date", default="2007-01-01", help="Train/Test split date")
    ap.add_argument("--ma_window", type=int, default=5, help="Window for Moving Average baseline")
    ap.add_argument("--lookback", type=int, default=21, help="Lookback window for LSTM/GRU")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = df.dropna(how="all").copy()

    # ---------- Baselines & Linear (univariate) ----------
    if args.target is not None and args.target in df.columns:
        target_series = df[args.target].dropna()
        train, test = train_test_split_by_date(target_series.to_frame(args.target), args.split_date)
        y_train, y_test = train[args.target], test[args.target]

        # Naive: last observed value
        naive_pred = pd.Series([y_train.iloc[-1]] * len(y_test), index=y_test.index)
        pd.DataFrame({"y_true": y_test, "y_pred": naive_pred}).to_parquet(os.path.join(OUT_DIR, f"naive_{args.target}.parquet"))

        # Moving Average baseline
        ma = target_series.rolling(args.ma_window).mean()
        ma_pred = ma.loc[y_test.index].fillna(method="bfill")
        pd.DataFrame({"y_true": y_test, "y_pred": ma_pred}).to_parquet(os.path.join(OUT_DIR, f"ma{args.ma_window}_{args.target}.parquet"))

        # Linear Regression with lags (use existing *_lag_* features if present)
        lag_cols = [c for c in df.columns if c.startswith(f"{args.target}_lag_")]
        X_train = df.loc[y_train.index, lag_cols].dropna()
        y_train_aligned = y_train.loc[X_train.index]
        X_test = df.loc[y_test.index, lag_cols].fillna(method="ffill").fillna(0.0)
        lr = LinearRegression()
        if len(X_train) > 0:
            lr.fit(X_train, y_train_aligned)
            lr_pred = pd.Series(lr.predict(X_test), index=X_test.index)
            pd.DataFrame({"y_true": y_test.loc[lr_pred.index], "y_pred": lr_pred}).to_parquet(os.path.join(OUT_DIR, f"linlags_{args.target}.parquet"))

        # LSTM (univariate)
        try:
            series = target_series.copy().astype(float).dropna()
            X, y = build_sequences(series, lookback=args.lookback, horizon=1)
            split_idx = np.searchsorted(series.index.values, np.datetime64(args.split_date))
            # ensure split within bounds relative to built sequences
            # Map indices from sequences to dates
            dates = series.index[args.lookback:args.lookback+len(y)]
            X_train, y_train = X[dates < np.datetime64(args.split_date)], y[dates < np.datetime64(args.split_date)]
            X_test, y_test_dl = X[dates >= np.datetime64(args.split_date)], y[dates >= np.datetime64(args.split_date)]
            dates_test = dates[dates >= np.datetime64(args.split_date)]

            if len(X_train) > 10 and len(X_test) > 0:
                model = Sequential([
                    LSTM(32, input_shape=(X_train.shape[1], 1)),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                model.fit(X_train[..., None], y_train, validation_split=0.2, epochs=args.epochs, batch_size=args.batch, callbacks=[cb], verbose=0)
                y_pred_dl = model.predict(X_test[..., None], verbose=0).ravel()
                dl_df = pd.DataFrame({"y_true": y_test_dl, "y_pred": y_pred_dl}, index=dates_test)
                dl_df.to_parquet(os.path.join(OUT_DIR, f"lstm_{args.target}.parquet"))
        except Exception as e:
            print("[Warn] LSTM training skipped:", e)

    else:
        print("[Info] --target not provided or missing in data; skipping univariate ML/DL.")

    # ---------- VAR (multivariate) ----------
    # Use numeric columns only
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_num = df[num_cols].dropna()
    try:
        train_m, test_m = train_test_split_by_date(df_num, args.split_date)
        if len(train_m) > 50 and len(test_m) > 0:
            model = VAR(train_m)
            res = model.fit(maxlags=5, ic="aic")
            lag_order = res.k_ar
            forecast_input = train_m.values[-lag_order:]
            steps = len(test_m)
            fc = res.forecast(y=forecast_input, steps=steps)
            fc_df = pd.DataFrame(fc, index=test_m.index, columns=train_m.columns)
            fc_df.to_parquet(os.path.join(OUT_DIR, "var_forecast.parquet"))
    except Exception as e:
        print("[Warn] VAR training skipped:", e)

    print("[OK] Predictions saved under outputs/predictions")

if __name__ == "__main__":
    main()
