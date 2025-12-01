# Author: Cody
# Streamlit dashboard: Week 5–8 visualization (data, EDA, forecasts, metrics)

import os
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Finance & Economics Dashboard", layout="wide")

st.title("📈 Finance & Economics Time-Series Dashboard")

CLEAN_PATH = Path("outputs/cleaned.parquet")
FEAT_PATH = Path("outputs/features.parquet")
PRED_DIR = Path("outputs/predictions")
FIG_DIR = Path("outputs/figures")
METRICS_PATH = Path("outputs/metrics_summary.csv")


@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_metrics(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return None


def get_prediction_files() -> List[Path]:
    if not PRED_DIR.exists():
        return []
    return sorted(Path(p) for p in glob.glob(str(PRED_DIR / "*.parquet")))


def numeric_nonempty_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns that have enough non-NaN and some variation."""
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        non_na = s.notna().sum()
        if non_na <= 5:
            continue
        if s.dropna().nunique() <= 1:
            continue
        cols.append(c)
    return cols


def plot_timeseries(df: pd.DataFrame, cols: List[str], title: str):
    if not cols:
        st.info("Please select at least one series.")
        return
    sub = df[cols]
    # Keep rows where not all selected columns are NaN
    sub = sub.dropna(how="all")
    if sub.empty:
        st.info("Selected series are empty after dropping NaNs.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    sub.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_actual_pred_ci(y_true: pd.Series, y_pred: pd.Series, title: str):
    # Align & clean
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[idx].dropna()
    y_pred = y_pred.loc[idx].dropna()
    idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[idx]
    y_pred = y_pred.loc[idx]

    if len(y_true) == 0:
        st.info("No overlapping non-NaN data between actual and predicted.")
        return

    residuals = y_true - y_pred
    sigma = residuals.std(ddof=1)
    upper = lower = None
    if np.isfinite(sigma) and sigma > 0:
        upper = y_pred + 1.96 * sigma
        lower = y_pred - 1.96 * sigma

    fig, ax = plt.subplots(figsize=(12, 4))
    y_true.plot(ax=ax, label="Actual")
    y_pred.plot(ax=ax, label="Predicted")
    if upper is not None:
        ax.fill_between(y_pred.index, lower, upper, alpha=0.2, label="95% band")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    st.pyplot(fig)


def show_static_figures():
    """Show key EDA figures saved in outputs/figures by 02_eda.py."""
    if not FIG_DIR.exists():
        st.info("No figures directory found. Run 02_eda.py first.")
        return

    png_files = sorted(FIG_DIR.glob("*.png"))
    if not png_files:
        st.info("No .png figures found in outputs/figures.")
        return

    # Group by rough types based on filename
    series_imgs = [p for p in png_files if any(k in p.name for k in ["gdp_series","inflation_series","unemployment_series","rates_series","stocks_series"])]
    decomp_imgs = [p for p in png_files if "seasonal_decompose" in p.name]
    corr_imgs = [p for p in png_files if "correlation_heatmap" in p.name]
    acf_imgs = [p for p in png_files if "acf_pacf" in p.name]
    other_imgs = [p for p in png_files if p not in series_imgs + decomp_imgs + corr_imgs + acf_imgs]

    if series_imgs:
        with st.expander("Macro & market trends (time-series)", expanded=True):
            for p in series_imgs:
                st.caption(p.name)
                st.image(str(p))

    if decomp_imgs:
        with st.expander("Seasonal decomposition", expanded=True):
            for p in decomp_imgs:
                st.caption(p.name)
                st.image(str(p))

    if corr_imgs:
        with st.expander("Correlation heatmaps", expanded=True):
            for p in corr_imgs:
                st.caption(p.name)
                st.image(str(p))

    if acf_imgs:
        with st.expander("ACF / PACF plots", expanded=False):
            for p in acf_imgs:
                st.caption(p.name)
                st.image(str(p))

    if other_imgs:
        with st.expander("Other figures (e.g., cross-correlations)", expanded=False):
            for p in other_imgs:
                st.caption(p.name)
                st.image(str(p))


def main():
    tab_data, tab_eda, tab_forecast = st.tabs(
        ["📊 Data & Features", "🔍 EDA & Insights", "🔮 Forecasts & Metrics"]
    )

    # ------------- Tab 1: Data & Features -------------
    with tab_data:
        st.subheader("Cleaned dataset")
        cleaned = load_parquet_safe(CLEAN_PATH)
        if cleaned is None:
            st.warning("Cleaned dataset not found. Run 01_data_understanding_cleaning.py first.")
        else:
            st.dataframe(cleaned.head())
            num_cols = numeric_nonempty_columns(cleaned)
            if num_cols:
                sel_cols = st.multiselect(
                    "Select series to visualize (from cleaned data):",
                    options=num_cols,
                    default=num_cols[: min(3, len(num_cols))],
                )
                plot_timeseries(cleaned, sel_cols, "Selected indicators (cleaned data)")
            else:
                st.info("No numeric columns with enough data in cleaned dataset.")

        st.markdown("---")
        st.subheader("Feature-engineered dataset")
        features = load_parquet_safe(FEAT_PATH)
        if features is None:
            st.info("Feature dataset not found. Run 03_feature_engineering.py.")
        else:
            st.dataframe(features.head())
            feat_num_cols = numeric_nonempty_columns(features)
            if feat_num_cols:
                # try default to one logret if exists
                default_feat = [c for c in feat_num_cols if "logret" in c.lower()]
                default_sel = default_feat[:1] if default_feat else feat_num_cols[: min(3, len(feat_num_cols))]
                sel_feat_cols = st.multiselect(
                    "Select series to visualize (from features):",
                    options=feat_num_cols,
                    default=default_sel,
                    key="feat_sel",
                )
                plot_timeseries(features, sel_feat_cols, "Selected derived features")
            else:
                st.info("No numeric feature columns with enough data.")

    # ------------- Tab 2: EDA & Insights -------------
    with tab_eda:
        st.subheader("Exploratory Data Analysis Figures")
        st.markdown(
            "These figures come from **02_eda.py** (Week 2 & Week 6): long-term trends, "
            "correlation structure, seasonal patterns, and autocorrelation."
        )
        show_static_figures()

    # ------------- Tab 3: Forecasts & Metrics -------------
    with tab_forecast:
        st.subheader("Forecast visualizations & model metrics")

        metrics_df = load_metrics(METRICS_PATH)
        if metrics_df is not None and not metrics_df.empty:
            st.markdown("**Overall metrics (from 05_evaluate_and_visualize.py):**")
            st.dataframe(metrics_df)

            # Ranking by RMSE per indicator (Week 5 requirement)
            try:
                st.markdown("**Best model per indicator (by RMSE):**")
                by_ind = metrics_df.sort_values("RMSE").groupby("indicator", dropna=True)
                rows = []
                for ind, g in by_ind:
                    best = g.iloc[0]
                    rows.append(
                        {
                            "indicator": ind,
                            "best_model": best.get("model", ""),
                            "RMSE": best["RMSE"],
                            "MAE": best["MAE"],
                            "MAPE": best["MAPE"],
                            "file": best["file"],
                        }
                    )
                if rows:
                    best_df = pd.DataFrame(rows)
                    st.dataframe(best_df)
            except Exception as e:
                st.warning(f"Could not compute ranking summary: {e}")
        else:
            st.info("metrics_summary.csv not found. Run 05_evaluate_and_visualize.py to generate metrics.")

        st.markdown("---")
        st.markdown("### Interactive forecast plots")

        pred_files = get_prediction_files()
        if not pred_files:
            st.info("No prediction files found. Run 04_models.py to generate forecasts.")
            return

        file_labels = [p.name for p in pred_files]
        chosen_label = st.selectbox("Select prediction file:", file_labels)
        chosen_path = pred_files[file_labels.index(chosen_label)]

        dfp = load_parquet_safe(chosen_path)
        if dfp is None or dfp.empty:
            st.warning("Selected prediction file is empty or unreadable.")
            return

        st.write("**Prediction file preview:**")
        st.dataframe(dfp.head())

        cols_lower = [c.lower() for c in dfp.columns]

        # Case 1: univariate with y_true/y_pred
        if "y_true" in cols_lower and "y_pred" in cols_lower:
            idx_true = cols_lower.index("y_true")
            idx_pred = cols_lower.index("y_pred")
            y_true = dfp.iloc[:, idx_true]
            y_pred = dfp.iloc[:, idx_pred]
            st.markdown("**Univariate forecast (y_true / y_pred detected)**")
            plot_actual_pred_ci(y_true, y_pred, f"{chosen_label} — Actual vs Predicted with 95% band")

        # Case 2: VAR or multi-series forecast (no explicit y_true/y_pred)
        else:
            st.markdown("**Multi-series forecast (e.g., VAR)**")
            cols = list(dfp.columns)
            if not cols:
                st.info("No columns to visualize.")
                return

            cleaned = load_parquet_safe(CLEAN_PATH)
            col = st.selectbox("Select series to plot:", cols)

            pred_series = pd.to_numeric(dfp[col], errors="coerce").dropna()

            if cleaned is not None and col in cleaned.columns:
                hist_series = pd.to_numeric(cleaned[col], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(12, 4))
                if not hist_series.empty:
                    hist_series.plot(ax=ax, label="History")
                pred_series.plot(ax=ax, label="Forecast")
                ax.set_title(f"{chosen_label} — {col}")
                ax.set_xlabel("Date")
                ax.legend(loc="best")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                pred_series.plot(ax=ax, label="Forecast")
                ax.set_title(f"{chosen_label} — {col}")
                ax.set_xlabel("Date")
                ax.legend(loc="best")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
