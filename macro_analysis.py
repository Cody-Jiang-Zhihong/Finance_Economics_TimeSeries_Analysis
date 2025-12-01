# src/macro_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# ---------------------------
# Data loading / preprocessing
# ---------------------------
def get_repo_root(this_file=__file__):
    return Path(this_file).resolve().parents[1]

def load_raw(filename="cleaned.parquet"):
    """
    Load cleaned data. Prefer parquet in outputs/, fallback to csv.
    Assumes repository layout:
      <repo>/outputs/cleaned.parquet  OR cleaned.csv
    Returns a dataframe with Date index.
    """
    repo_root = get_repo_root()
    outputs = repo_root / "outputs"
    p_parq = outputs / filename
    p_csv = outputs / "cleaned.csv"

    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"Neither {p_parq} nor {p_csv} found.")
    # ensure Date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
    return df

def resample_to_monthly(df):
    """
    Convert daily dataframe to monthly (month-end) aggregated series.
    Returns a monthly DataFrame with the main columns.
    """
    monthly = pd.DataFrame()
    # If column absent, this will raise; callers should ensure columns exist.
    monthly["Consumer Confidence Index"] = df["Consumer Confidence Index"].resample("ME").mean()
    monthly["Retail Sales (Billion USD)"] = df["Retail Sales (Billion USD)"].resample("ME").mean()
    monthly["GDP Growth (%)"] = df["GDP Growth (%)"].resample("ME").last()
    monthly["Unemployment Rate (%)"] = df["Unemployment Rate (%)"].resample("ME").last()
    monthly["Inflation Rate (%)"] = df["Inflation Rate (%)"].resample("ME").last()
    monthly = monthly.dropna()
    return monthly

# ---------------------------
# Business cycle classification
# ---------------------------
def classify_business_cycle(monthly):
    """
    Add Phase column to monthly DataFrame based on simple rules.
    Returns a copy of DF with 'Phase' column added.
    """
    df = monthly.copy()
    df["GDP_Growth_prev"] = df["GDP Growth (%)"].shift(1)
    df["Unemp_prev"] = df["Unemployment Rate (%)"].shift(1)
    df["Infl_prev"] = df["Inflation Rate (%)"].shift(1)

    def classify_row(row):
        if pd.isna(row["GDP_Growth_prev"]):
            return "Unclassified"
        # Recession: negative GDP & unemployment rising
        if row["GDP Growth (%)"] < 0 and row["Unemployment Rate (%)"] > row["Unemp_prev"]:
            return "Recession"
        # Early Recovery: GDP still negative but unemployment improving
        if row["GDP Growth (%)"] < 0 and row["Unemployment Rate (%)"] <= row["Unemp_prev"]:
            return "Early Recovery"
        # Late Recovery: GDP positive & inflation falling
        if row["GDP Growth (%)"] > 0 and row["Inflation Rate (%)"] < row["Infl_prev"]:
            return "Late Recovery"
        # Slowdown: GDP positive but inflation rising
        if row["GDP Growth (%)"] > 0 and row["Inflation Rate (%)"] > row["Infl_prev"]:
            return "Slowdown"
        # Strong growth
        if row["GDP Growth (%)"] > 1.5:
            return "Expansion"
        return "Unclassified"

    df["Phase"] = df.apply(classify_row, axis=1)
    # drop helper cols
    df = df.drop(columns=["GDP_Growth_prev", "Unemp_prev", "Infl_prev"])
    return df

# ---------------------------
# Derived indicators
# ---------------------------
def consumer_spending_momentum(df):
    out = df.copy()
    out["Retail Sales MoM %"] = out["Retail Sales (Billion USD)"].pct_change() * 100
    out["Confidence MoM %"] = out["Consumer Confidence Index"].pct_change() * 100
    return out

def labor_market_metrics(df):
    out = df.copy()
    out["Unemployment 6M Change"] = out["Unemployment Rate (%)"].diff(6)
    out["Labor Market Score"] = -out["Unemployment 6M Change"]  # higher = tighter
    return out

def inflation_and_real_retail(df):
    out = df.copy()
    out["Inflation MoM Change"] = out["Inflation Rate (%)"].diff()
    out["Real Retail Sales"] = out["Retail Sales (Billion USD)"] / (1 + out["Inflation Rate (%)"] / 100)
    return out

# ---------------------------
# Correlations & lag analysis
# ---------------------------
def correlation_matrix(df):
    return df.corr()

def lagged_correlation(series_x, series_y, max_lag=12):
    corrs = []
    for lag in range(0, max_lag + 1):
        c = series_x.corr(series_y.shift(-lag))
        corrs.append(c)
    return pd.Series(corrs, index=range(0, max_lag + 1))

def lead_lag_matrix(df, x_vars, y_vars, max_lag=12):
    out = {}
    for x in x_vars:
        mat = pd.DataFrame(index=range(0, max_lag + 1), columns=y_vars, dtype=float)
        for lag in range(0, max_lag + 1):
            for y in y_vars:
                mat.loc[lag, y] = df[x].corr(df[y].shift(-lag))
        out[x] = mat
    return out

# ---------------------------
# Group & lagged summary by phase
# ---------------------------
def phase_summary_stats(df):
    """
    df must have 'Phase'
    Returns per-phase mean/std for main variables
    """
    cols = ['Consumer Confidence Index', 'Retail Sales (Billion USD)',
            'GDP Growth (%)', 'Unemployment Rate (%)', 'Inflation Rate (%)']
    return df.groupby('Phase')[cols].agg(['mean', 'std', 'median', 'count'])

def lagged_phase_means(df, lags=(1,2,3,6)):
    """
    For each lag, shift indicators forward (so we see indicators BEFORE the phase) and compute means by Phase.
    Returns dict lag -> dataframe
    """
    results = {}
    for lag in lags:
        lagged = df.copy()
        # shift indicators so that values lag months earlier are associated with the current phase
        lagged[['Consumer Confidence Index','Retail Sales (Billion USD)',
                'GDP Growth (%)','Unemployment Rate (%)','Inflation Rate (%)']] = \
            lagged[['Consumer Confidence Index','Retail Sales (Billion USD)',
                    'GDP Growth (%)','Unemployment Rate (%)','Inflation Rate (%)']].shift(lag)
        results[lag] = lagged.groupby('Phase')[['Consumer Confidence Index','Retail Sales (Billion USD)',
                                                 'GDP Growth (%)','Unemployment Rate (%)','Inflation Rate (%)']].mean()
    return results

def anova_tests(df):
    """
    Run one-way ANOVA across phases for each variable (exclude 'Unclassified').
    Returns dict var -> (F, p)
    """
    res = {}
    phases = [p for p in df['Phase'].unique() if p != 'Unclassified']
    for col in ['Consumer Confidence Index', 'Retail Sales (Billion USD)', 'GDP Growth (%)',
                'Unemployment Rate (%)', 'Inflation Rate (%)']:
        groups = [df[df['Phase']==phase][col].dropna() for phase in phases]
        # Only run if at least 2 groups have data
        if sum([len(g)>0 for g in groups]) >= 2:
            f, p = f_oneway(*groups)
            res[col] = (f, p)
        else:
            res[col] = (np.nan, np.nan)
    return res

# ---------------------------
# Plotting helpers (matplotlib figures)
# ---------------------------
def plot_corr_heatmap(df, figsize=(10,8), vmin=-1, vmax=1):
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title("Correlation Matrix (monthly)")
    fig.tight_layout()
    return fig

def plot_dual_panel_retail(df, var_name, figsize=(10,8)):
    fig, axes = plt.subplots(2,1, figsize=figsize, sharex=True)
    axes[0].plot(df.index, df['Retail Sales (Billion USD)'], label='Retail Sales (Billion USD)')
    axes[0].set_title('Retail Sales (Billion USD)')
    axes[0].grid(True)
    axes[1].plot(df.index, df[var_name], label=var_name)
    axes[1].set_title(var_name)
    axes[1].grid(True)
    fig.tight_layout()
    return fig

def plot_lag_series(series_x, series_y, max_lag=12, figsize=(8,3.5)):
    s = lagged_correlation(series_x, series_y, max_lag=max_lag)
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(s.index, s.values, basefmt=" ", use_line_collection=True)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xlabel("Lag (months) — positive means predictor leads target")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Lagged corr: {series_x.name} → {series_y.name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, s

def plot_lead_lag_heatmap(df, predictor, targets, max_lag=12, figsize=(9,4)):
    mat = lead_lag_matrix(df, [predictor], targets, max_lag=max_lag)[predictor]
    # plot transpose so rows are targets and columns are lags
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat.T, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Targets")
    ax.set_title(f"Lead-Lag matrix ({predictor})")
    fig.tight_layout()
    return fig, mat
