# fixed_macro_analysis.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from IPython.display import display

# Statsmodels tests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# -------------------------
# 1. LOAD DATA (robust)
# -------------------------
def load_data(filename="cleaned.csv"):
    """
    Loads the macroeconomic dataset from the /outputs folder inside the repo.
    If __file__ is not defined (e.g. running in notebook), falls back to cwd.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except NameError:
        # in notebooks __file__ is not defined; assume working dir is repo root
        repo_root = Path.cwd()

    # try common paths
    candidate = repo_root / "Finance_Economics_TimeSeries_Analysis-main" / "outputs" / filename
    if not candidate.exists():
        # fallback: look in repo_root/outputs or repo_root/data
        candidate = repo_root / "outputs" / filename
    if not candidate.exists():
        candidate = repo_root / "data" / filename
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find {filename} in expected locations. Last tried: {candidate}")

    df = pd.read_csv(candidate)
    if "Date" not in df.columns:
        raise KeyError("Input CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


# -------------------------
# 2. RESAMPLE TO MONTHLY
# -------------------------
def resample_to_monthly(df):
    monthly = pd.DataFrame(index=df.resample("M").mean().index)  # ensure index exists

    # Indicators with daily variation → mean of month
    if "Consumer Confidence Index" in df.columns:
        monthly["Consumer Confidence Index"] = df["Consumer Confidence Index"].resample("M").mean()
    if "Retail Sales (Billion USD)" in df.columns:
        monthly["Retail Sales (Billion USD)"] = df["Retail Sales (Billion USD)"].resample("M").mean()

    # Indicators updated infrequently → last value of month
    for col in ["GDP Growth (%)", "Unemployment Rate (%)", "Inflation Rate (%)"]:
        if col in df.columns:
            monthly[col] = df[col].resample("M").last()

    monthly = monthly.dropna()
    return monthly


# -------------------------
# 3. PLOTTING HELPERS
# -------------------------
def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, center=0)
    plt.title("Macro Indicators Correlation Matrix (Monthly)")
    plt.tight_layout()
    plt.show()


def compare_with_retail(df, var_name):
    if var_name not in df.columns or "Retail Sales (Billion USD)" not in df.columns:
        print(f"Missing columns for comparison: {var_name} or Retail Sales not found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df.index, df["Retail Sales (Billion USD)"])
    axes[0].set_title("Retail Sales (Billion USD)")
    axes[1].plot(df.index, df[var_name])
    axes[1].set_title(var_name)
    plt.tight_layout()
    plt.show()


# -------------------------
# 4. ADF stationarity check
# -------------------------
def adf_check(series, signif=0.05):
    """Return (is_stationary_bool, pvalue) using Augmented Dickey-Fuller."""
    series = series.dropna()
    if len(series) < 3:
        return False, np.nan
    result = adfuller(series, autolag="AIC")
    pvalue = result[1]
    return (pvalue < signif), pvalue


# -------------------------
# 5. LAGGED CORRELATIONS
# -------------------------
def lagged_correlation(series_x, series_y, max_lag=12):
    corrs = []
    for lag in range(0, max_lag + 1):
        # compare x_t with y_{t+lag} => shift y backward (so y at t+lag aligns with x at t)
        c = series_x.corr(series_y.shift(-lag))
        corrs.append(c)
    return pd.Series(corrs, index=range(0, max_lag + 1))


import matplotlib.pyplot as plt
import numpy as np

def plot_lagged_corr(series_x, series_y, max_lag=12, title="Lagged Correlation"):
    """
    Compute and plot cross-correlation between two time series at different lags.
    Positive lag: X leads Y.
    Negative lag: Y leads X.
    Backward-compatible with older Matplotlib that does not support `use_line_collection`.
    """
    # Align time series
    df = (
        series_x.rename("x")
        .to_frame()
        .join(series_y.rename("y"), how="inner")
        .dropna()
    )

    x = df["x"]
    y = df["y"]

    # Compute correlations for both positive and negative lags
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []

    for lag in lags:
        if lag < 0:
            corr.append(x[:lag].corr(y[-lag:]))
        elif lag > 0:
            corr.append(x[lag:].corr(y[:-lag]))
        else:
            corr.append(x.corr(y))

    corr = np.array(corr)

    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    s_idx = lags
    s_vals = corr

    # ---- Matplotlib backward compatibility ----
    try:
        # Try new argument (Matplotlib 3.6+)
        ax.stem(s_idx, s_vals, basefmt=" ", use_line_collection=True)
    except TypeError:
        # Fallback for older versions
        ax.stem(s_idx, s_vals, basefmt=" ")

    # Draw zero line
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return {
        "lags": lags,
        "correlations": corr,
        "figure": fig,
        "axis": ax
    }



def lead_lag_matrix(df, x_vars, y_vars, max_lag=12):
    out = {}
    for x in x_vars:
        mat = pd.DataFrame(index=range(0, max_lag + 1), columns=y_vars, dtype=float)
        for lag in range(0, max_lag + 1):
            for y in y_vars:
                mat.loc[lag, y] = df[x].corr(df[y].shift(-lag))
        out[x] = mat
    return out


# -------------------------
# 6. GRANGER CAUSALITY HELPERS
# -------------------------
def run_granger_tests(df, predictor, target, maxlag=6, verbose=False):
    """
    Runs grangercausalitytests on columns [target, predictor].
    Returns dict with p-values by lag and a flag if any p < 0.05.
    """
    data = df[[target, predictor]].dropna()
    if len(data) < maxlag + 5:
        # insufficient data
        return {"pvalues_by_lag": None, "any_significant": None, "error": "insufficient data"}

    # statsmodels expects shape (n_obs, 2) with [endog, exog] ordering when passing DataFrame
    try:
        test_res = grangercausalitytests(data[[target, predictor]], maxlag=maxlag, verbose=verbose)
    except Exception as e:
        return {"pvalues_by_lag": None, "any_significant": None, "error": str(e)}

    pvals = {}
    for lag, res in test_res.items():
        # res is a tuple-like dict; the common access for F-test p-value:
        try:
            pval = res[0]["ssr_ftest"][1]
        except Exception:
            # try alternative indexing if structure different
            try:
                pval = res[0][0][1]
            except Exception:
                pval = np.nan
        pvals[lag] = pval
    any_sig = any((p < 0.05) for p in pvals.values() if p is not None and not np.isnan(p))
    return {"pvalues_by_lag": pvals, "any_significant": any_sig, "error": None}


def granger_causality_table(df, pairs, max_lag=4, alpha=0.05):
    results = []
    for cause, effect in pairs:
        sub = df[[effect, cause]].dropna()
        if len(sub) < max_lag + 5:
            results.append({
                "Cause": cause,
                "Effect": effect,
                "Best Lag": None,
                "Min p-value": None,
                "Significant (<0.05)": "Insufficient data"
            })
            continue
        try:
            test_result = grangercausalitytests(sub[[effect, cause]], maxlag=max_lag, verbose=False)
        except Exception as e:
            results.append({
                "Cause": cause,
                "Effect": effect,
                "Best Lag": None,
                "Min p-value": None,
                "Significant (<0.05)": f"Error: {str(e)}"
            })
            continue

        pvals = []
        for lag in range(1, max_lag + 1):
            try:
                pval = test_result[lag][0]["ssr_ftest"][1]
            except Exception:
                pval = np.nan
            pvals.append((lag, pval))

        # skip if all pvals nan
        valid = [t for t in pvals if not np.isnan(t[1])]
        if not valid:
            results.append({
                "Cause": cause,
                "Effect": effect,
                "Best Lag": None,
                "Min p-value": None,
                "Significant (<0.05)": "No valid p-values"
            })
            continue

        best_lag, min_pval = min(valid, key=lambda x: x[1])
        results.append({
            "Cause": cause,
            "Effect": effect,
            "Best Lag": int(best_lag),
            "Min p-value": round(float(min_pval), 5),
            "Significant (<0.05)": "YES" if min_pval < alpha else "NO"
        })
    return pd.DataFrame(results)


# -------------------------
# 7. BUSINESS CYCLE CLASSIFICATION
# -------------------------
def classify_business_cycle(df):
    """
    Simple rule-based classification. Returns a DataFrame with a 'Phase' column.
    """
    df = df.copy()
    # previous-months for comparison
    df["GDP_prev"] = df["GDP Growth (%)"].shift(1)
    df["Unemp_prev"] = df["Unemployment Rate (%)"].shift(1)
    df["Infl_prev"] = df["Inflation Rate (%)"].shift(1)

    def classify_row(row):
        if pd.isna(row["GDP_prev"]):
            return "Unclassified"
        gdp = row["GDP Growth (%)"]
        unemp = row["Unemployment Rate (%)"]
        infl = row["Inflation Rate (%)"]
        if gdp < 0 and unemp > row["Unemp_prev"]:
            return "Recession"
        if gdp < 0 and unemp <= row["Unemp_prev"]:
            return "Early Recovery"
        if gdp > 0 and infl < row["Infl_prev"]:
            return "Late Recovery"
        if gdp > 0 and infl > row["Infl_prev"]:
            return "Slowdown"
        if gdp > 1.5:
            return "Expansion"
        return "Unclassified"

    df["Phase"] = df.apply(classify_row, axis=1)
    return df


# -------------------------
# 8. ADDITIONAL METRICS
# -------------------------
def consumer_spending_momentum(df):
    df = df.copy()
    if "Retail Sales (Billion USD)" in df.columns:
        df["Retail Sales MoM %"] = df["Retail Sales (Billion USD)"].pct_change() * 100
    if "Consumer Confidence Index" in df.columns:
        df["Confidence MoM %"] = df["Consumer Confidence Index"].pct_change() * 100
    return df


def labor_market_strength(df):
    df = df.copy()
    if "Unemployment Rate (%)" in df.columns:
        df["Unemployment 6M Change"] = df["Unemployment Rate (%)"].diff(6)
        df["Labor Market Score"] = -df["Unemployment 6M Change"]
    return df


def inflation_metrics(df):
    df = df.copy()
    if "Inflation Rate (%)" in df.columns:
        df["Inflation MoM Change"] = df["Inflation Rate (%)"].diff()
    if "Retail Sales (Billion USD)" in df.columns and "Inflation Rate (%)" in df.columns:
        df["Real Retail Sales"] = df["Retail Sales (Billion USD)"] / (1 + df["Inflation Rate (%)"] / 100)
    return df


# -------------------------
# main execution
# -------------------------
def main(filename="cleaned.csv", show_plots=True):
    print("Loading dataset...")
    df = load_data(filename)

    print("Converting to monthly frequency...")
    monthly = resample_to_monthly(df)

    print("\n--- Monthly Data Sample ---")
    display(monthly.head())

    # Add computed metrics
    monthly = consumer_spending_momentum(monthly)
    monthly = labor_market_strength(monthly)
    monthly = inflation_metrics(monthly)

    # Business cycle classification
    monthly = classify_business_cycle(monthly)

    if show_plots:
        print("\nPlotting correlation matrix...")
        plot_correlation_matrix(monthly[[
            c for c in ["Consumer Confidence Index",
                        "Retail Sales (Billion USD)",
                        "GDP Growth (%)",
                        "Unemployment Rate (%)",
                        "Inflation Rate (%)"] if c in monthly.columns
        ]])

    predictors = [c for c in ["Consumer Confidence Index", "Retail Sales (Billion USD)"] if c in monthly.columns]
    targets = [c for c in ["GDP Growth (%)", "Unemployment Rate (%)", "Inflation Rate (%)"] if c in monthly.columns]

    # Lead-lag matrices & heatmaps
    if predictors and targets:
        ll_mats = lead_lag_matrix(monthly, predictors, targets, max_lag=12)
        for pred in predictors:
            print(f"\nLead-Lag matrix for predictor = {pred} (rows=lag months, columns=targets):")
            display(ll_mats[pred].round(3))
            if show_plots:
                plt.figure(figsize=(8, 3))
                sns.heatmap(ll_mats[pred].T, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
                plt.xlabel("Lag (months) — positive means predictor leads target")
                plt.title(f"Lead-Lag Correlations: {pred}")
                plt.tight_layout()
                plt.show()

        # example lagged correlation plots for top pairs if present
    if "Consumer Confidence Index" in monthly.columns and "GDP Growth (%)" in monthly.columns:
        _ = plot_lagged_corr(
            series_x=monthly["Consumer Confidence Index"],
            series_y=monthly["GDP Growth (%)"],
            max_lag=12,
            title="Consumer Confidence → GDP Growth"
        )

    if "Retail Sales (Billion USD)" in monthly.columns and "GDP Growth (%)" in monthly.columns:
        _ = plot_lagged_corr(
            series_x=monthly["Retail Sales (Billion USD)"],
            series_y=monthly["GDP Growth (%)"],
            max_lag=12,
            title="Retail Sales → GDP Growth"
        )


    # ADF checks
    print("\n--- ADF Stationarity Checks ---")
    for var in predictors + targets:
        stationary, pval = adf_check(monthly[var])
        print(f"ADF test for {var}: stationary={stationary}, pvalue={pval:.3f}")

    # Granger causality table
    pairs_to_test = [
        ("Consumer Confidence Index", "Retail Sales (Billion USD)"),
        ("Retail Sales (Billion USD)", "GDP Growth (%)"),
        ("Consumer Confidence Index", "GDP Growth (%)"),
        ("Unemployment Rate (%)", "Inflation Rate (%)"),
        ("Inflation Rate (%)", "Consumer Confidence Index")
    ]
    # filter pairs for available columns
    filtered_pairs = [(a, b) for (a, b) in pairs_to_test if a in monthly.columns and b in monthly.columns]
    if filtered_pairs:
        causality_results = granger_causality_table(monthly, filtered_pairs, max_lag=4)
        print("\nGranger Causality Results:\n")
        print(causality_results)
    else:
        causality_results = pd.DataFrame()
        print("No valid pairs for Granger tests (missing columns).")

    # Summary outputs
    print("\nComputed columns:")
    print([col for col in monthly.columns if col not in [
        "Consumer Confidence Index",
        "Retail Sales (Billion USD)",
        "GDP Growth (%)",
        "Unemployment Rate (%)",
        "Inflation Rate (%)"
    ]])

    print("\nBusiness cycle phases distribution:")
    if "Phase" in monthly.columns:
        print(monthly["Phase"].value_counts())
    else:
        print("Phase column not found.")

    return {
        "raw": df,
        "monthly": monthly,
        "causality_results": causality_results
    }


if __name__ == "__main__":
    out = main()
