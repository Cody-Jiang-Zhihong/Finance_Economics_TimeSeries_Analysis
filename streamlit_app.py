
# Author: Cody
# Optional Streamlit dashboard to explore data & forecasts

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Finance & Economics Dashboard", layout="wide")

st.title("Finance & Economics Time-Series Dashboard")

clean_path = Path("outputs/cleaned.parquet")
feat_path = Path("outputs/features.parquet")

if clean_path.exists():
    df = pd.read_parquet(clean_path)
    st.subheader("Raw / Cleaned Data (first 500 rows)")
    st.dataframe(df.head(500))
else:
    st.warning("Cleaned data not found. Run 01_data_understanding_cleaning.py first.")
    st.stop()

if feat_path.exists():
    feats = pd.read_parquet(feat_path)
else:
    feats = df.copy()

# Variable selection
num_cols = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
sel = st.multiselect("Select variables to plot", options=num_cols[:200], default=num_cols[:3])

if len(sel) > 0:
    st.subheader("Time Series Plot")
    fig, ax = plt.subplots(figsize=(10,4))
    feats[sel].plot(ax=ax)
    st.pyplot(fig)

st.subheader("Forecasts (if available)")
pred_dir = Path("outputs/predictions")
if pred_dir.exists():
    files = list(pred_dir.glob("*.parquet"))
    names = [f.name for f in files]
    if names:
        choice = st.selectbox("Pick a prediction file", names)
        dfp = pd.read_parquet(pred_dir / choice)
        cols_lower = [c.lower() for c in dfp.columns]
        if "y_true" in cols_lower and "y_pred" in cols_lower:
            actual = dfp[dfp.columns[cols_lower.index("y_true")]]
            pred = dfp[dfp.columns[cols_lower.index("y_pred")]]
            fig, ax = plt.subplots(figsize=(10,4))
            actual.plot(ax=ax, label="Actual")
            pred.plot(ax=ax, label="Predicted")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Multi-variate forecast (e.g., VAR). Showing head:")
            st.dataframe(dfp.head())
    else:
        st.info("No prediction files found yet. Run 04_models.py.")
else:
    st.info("No prediction directory found. Run 04_models.py.")
