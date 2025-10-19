# Finance & Economics Time‑Series Analysis (2000–2008)

End‑to‑end pipeline for exploring, cleaning, feature‑engineering, modeling, and visualizing a finance + macroeconomics dataset.
Works on Windows/PowerShell and Mac/Linux. Includes baseline models, VAR, LSTM, and an optional Streamlit dashboard.

---

## Dataset
- Default location: `/mnt/data/finance_economics_dataset.csv` (update with your path if different)
- Must contain a **date** column (e.g., `date`, `Date`, `timestamp`) and numeric indicators such as prices, CPI/inflation, unemployment, interest rates, etc.

**Time window:** scripts default to `2000‑01‑01` → `2008‑12‑31`

---

## Project Structure
```
finance_econ_project/
├─ 01_data_understanding_cleaning.py
├─ 02_eda.py
├─ 03_feature_engineering.py
├─ 04_models.py
├─ 05_evaluate_and_visualize.py
├─ streamlit_app.py
├─ src/
│  └─ utils.py
├─ outputs/
│  ├─ cleaned.parquet / cleaned.csv
│  ├─ monthly.parquet
│  ├─ features.parquet / features.csv
│  ├─ predictions/  (model outputs)
│  └─ figures/      (PNG charts)
└─ requirements.txt
```

---

## Quickstart (Windows PowerShell)
From your project root (where the scripts live):

```powershell
# 0) (Optional) Create venv & install deps
pip install -r .\requirements.txt

# 1) Week 1 – Cleaning (adjust --csv if your path differs)
python 01_data_understanding_cleaning.py --csv ".\finance_economics_dataset.csv"

# 2) Week 2 – EDA (use --monthly for macro seasonality plots)
python 02_eda.py --monthly

# 3) Week 3 – Feature Engineering
python 03_feature_engineering.py

# 4) Models
#    Choose a numeric target column (see examples below)
python 04_models.py --target "Close Price" --split_date 2007-01-01 --var_topk 4

# 5) Evaluation + viz
python 05_evaluate_and_visualize.py

# 6) Dashboard
streamlit run streamlit_app.py
```

**Typical numeric targets in this dataset**
- `"Close Price"` (recommended)
- `"Open Price"`, `"Real Estate Index"`, `"Crude Oil Price (USD per Barrel)"`, `"Gold Price (USD per Ounce)"`

> If your column has spaces, keep quotes. The target **must be numeric** in `outputs/features.parquet`.

---

## What Each Script Does

### 01_data_understanding_cleaning.py
- Loads the CSV, auto‑detects a date column, sets it as index, filters to 2000–2008.
- Drops duplicate dates, summarizes missingness → `outputs/missingness_summary.csv`.
- Fills gaps with forward/backward fills → `outputs/cleaned.parquet` + `.csv`.

**Key flags**
```
--csv <path>           Path to your dataset
--start YYYY-MM-DD     Start date (default 2000-01-01)
--end   YYYY-MM-DD     End date (default 2008-12-31)
```

### 02_eda.py
- Coerces numeric‑like strings to numbers (handles commas, `%`).
- Plots groups: GDP/Inflation/Unemployment/Rates/Stocks (skips non‑numeric automatically).
- Correlation heatmap; optional monthly resample for seasonal decomposition.
- Saves figures under `outputs/figures/`.

**Key flags**
```
--input outputs/cleaned.parquet
--monthly       Resample to month-end for macro seasonality plots
```

### 03_feature_engineering.py
- Adds **log returns**, **rolling means/volatilities**, and **lags** for price‑like series.
- Adds **spreads** (e.g., term spread if 10y/2y present).
- Adds first/second differences for macro indicators.
- Writes `outputs/features.parquet` + `.csv`.

**Key flags**
```
--input outputs/cleaned.parquet
--target <col>         (optional) add lags for a specific target
--lags 1 5 21          lag steps (default)
--windows 5 21 63      rolling windows (default)
```

### 04_models.py
- **Univariate baselines:** Naive (last value), Moving Average.
- **Linear (lags):** uses `<target>_lag_*` features if present.
- **LSTM:** simple univariate DL model (requires enough samples).
- **VAR (multivariate):** numeric‑only, drop constant cols, ffill/bfill, top‑K by variance, correlation pruning, differencing + z‑score, lag backoff. Saves:
  - `outputs/predictions/naive_<target>.parquet`
  - `outputs/predictions/ma5_<target>.parquet`
  - `outputs/predictions/linlags_<target>.parquet` (if lag features exist)
  - `outputs/predictions/lstm_<target>.parquet` (if enough data)
  - `outputs/predictions/var_forecast_diffz.parquet`
  - `outputs/predictions/var_forecast_levels.parquet`

**Key flags**
```
--target "Close Price"     REQUIRED for univariate models
--split_date 2007-01-01    Train/Test split
--var_topk 4               Limit VAR dimensionality (try 3–6)
```

### 05_evaluate_and_visualize.py
- Computes **MAE, RMSE, MAPE** for saved predictions.
- Plots **Actual vs Predicted** to `outputs/figures/`.
- Aggregates metrics to `outputs/metrics_summary.csv`.

### streamlit_app.py (Optional)
- Browse the cleaned/features tables.
- Plot selected variables.
- Preview prediction files interactively.

---

## Troubleshooting

**“Skipping univariate ML/DL — no suitable target.”**  
- You must pass `--target "<exact column name>"` and it must be **numeric** in `features.parquet`.
- List candidates:
  ```powershell
  python -c "import pandas as pd; df=pd.read_parquet('outputs/features.parquet'); num=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]; print(num[:80])"
  ```

**VAR error: “leading minor not positive definite”**  
- Reduce dimensionality: `--var_topk 4` (or 3).  
- Use a diverse set (avoid near-duplicates like both Close and Adj Close).  
- As a last resort, manually restrict series for VAR inside `04_models.py` by keeping a small list:
  ```python
  keep_for_var = ["Close Price", "Interest Rate (%)", "Inflation Rate (%)", "Unemployment Rate (%)"]
  ```

**No numeric data to plot (EDA)**  
- The EDA script auto‑coerces numeric‑like strings; if your schema is unusual, verify the cleaned file:
  ```powershell
  python -c "import pandas as pd; print(pd.read_parquet('outputs/cleaned.parquet').dtypes)"
  ```

---

## Tips
- Keep target & features aligned by date; avoid leaking future info.
- For daily → monthly, use `--monthly` in EDA; models run on the **engineered features** at native frequency.
- For LSTM, you typically want longer history (`--lookback 21` by default) and enough training samples.

---

## License
For academic/project use. NULL for now.
