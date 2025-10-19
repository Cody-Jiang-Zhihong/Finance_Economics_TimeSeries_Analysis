
import re
import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except Exception:
    # Fallback implementations (no external deps):
    import numpy as _np
    def mean_absolute_error(y_true, y_pred):
        y_true = _np.asarray(y_true, dtype=float)
        y_pred = _np.asarray(y_pred, dtype=float)
        return _np.nanmean(_np.abs(y_true - y_pred))
    def mean_squared_error(y_true, y_pred):
        y_true = _np.asarray(y_true, dtype=float)
        y_pred = _np.asarray(y_pred, dtype=float)
        return _np.nanmean((y_true - y_pred)**2)

# ----------------- Column & time helpers -----------------

COMMON_DATE_COLS = ["date", "time", "timestamp", "Date", "TIME", "DATE"]
GDP_ALIASES = ["gdp", "gdp_growth", "real_gdp", "gdp_growth_rate"]
INFL_ALIASES = ["inflation", "cpi", "cpi_yoy", "infl"]
UNEMP_ALIASES = ["unemployment", "unemp", "unemployment_rate"]
RATE_ALIASES = ["interest", "rate", "interest_rate", "federal_funds_rate", "ffr", "ten_year", "2y", "10y"]
STOCK_ALIASES = ["close", "adj_close", "price", "sp500", "stock", "index"]

def find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, orig in cols_lower.items():
            if cand.lower() == col_lower or cand.lower() in col_lower:
                return orig
    return None

def detect_date_col(df: pd.DataFrame) -> str:
    for c in COMMON_DATE_COLS:
        if c in df.columns:
            return c
    # fallback: try to infer
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    # if still not found, assume first column
    return df.columns[0]

def coerce_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    return df

def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = miss / len(df) * 100.0
    return pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_count", ascending=False)

def zscore_outliers(series: pd.Series, thresh: float = 4.0) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series([False]*len(s), index=series.index)
    z = (s - mu) / sigma
    return z.abs() > thresh

# ----------------- Frequency helpers -----------------

def to_monthly(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    if agg == "mean":
        return df.resample("ME").mean(numeric_only=True)
    if agg == "last":
        return df.resample("ME").last()
    if agg == "sum":
        return df.resample("ME").sum(numeric_only=True)
    return df.resample("ME").mean(numeric_only=True)

def to_quarterly(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    rule = "Q"
    if agg == "mean":
        return df.resample(rule).mean(numeric_only=True)
    if agg == "last":
        return df.resample(rule).last()
    if agg == "sum":
        return df.resample(rule).sum(numeric_only=True)
    return df.resample(rule).mean(numeric_only=True)

# ----------------- Metrics -----------------

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape(y_true, y_pred)),
    }

# ----------------- Feature Engineering -----------------

def add_returns(df: pd.DataFrame, col: str, log: bool = True) -> pd.DataFrame:
    df = df.copy()
    if log:
        df[f"{col}_logret"] = np.log(df[col]).diff()
    else:
        df[f"{col}_ret"] = df[col].pct_change()
    return df

def add_rolling_stats(df: pd.DataFrame, col: str, windows=(5, 21, 63)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{col}_sma_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_vol_{w}"] = df[col].rolling(w).std()
    return df

def add_lags(df: pd.DataFrame, col: str, lags=(1, 5, 21)) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        df[f"{col}_lag_{l}"] = df[col].shift(l)
    return df

def add_spread(df: pd.DataFrame, col_a: str, col_b: str, name: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    nm = name or f"spread_{col_a}_minus_{col_b}"
    df[nm] = df[col_a] - df[col_b]
    return df

def safe_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def train_test_split_by_date(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df.index < split_date].copy()
    test = df.loc[df.index >= split_date].copy()
    return train, test
