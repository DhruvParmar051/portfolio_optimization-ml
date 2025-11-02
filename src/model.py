"""
model.py

Expanding-window ARIMA backtest per stock for portfolio modeling (parallel + optimized).

This module:
1. Loads cleaned & preprocessed stock-price data.
2. Performs an expanding-window backtest (750-day minimum training window).
3. Re-fits ARIMA(p,d,q) models using efficient parallelization.
4. Evaluates out-of-sample RMSE and stores forecasts.
5. Supports resume via checkpointing to handle large-scale runs.

Optimizations:
- Parallelized across stocks (via joblib)
- Caches best (p,d,q) order per stock to avoid repeated grid search
- Saves intermediate stock results to disk incrementally
"""

# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import logging
from itertools import product
from joblib import Parallel, delayed, dump, load
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

DATA_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "arima_expanding")

ROLLING_START = 750           # initial expanding window length
FORECAST_HORIZON = 30         # forecast next 30 days
MAX_P, MAX_D, MAX_Q = 2, 1, 2 # smaller grid for speed
N_JOBS = max(1, os.cpu_count() // 2)  # parallel cores

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Utility functions
# ============================================================

def select_best_order(y_train, cache_path=None):
    """
    Select ARIMA(p,d,q) order minimizing AIC.
    Uses cache if available to avoid recomputation.
    """
    if cache_path and os.path.exists(cache_path):
        return load(cache_path)
    best_aic = np.inf
    best_order = (1, 0, 0)
    for p, d, q in product(range(MAX_P + 1), range(MAX_D + 1), range(MAX_Q + 1)):
        try:
            model = ARIMA(y_train, order=(p, d, q))
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p, d, q)
        except Exception:
            continue
    if cache_path:
        dump(best_order, cache_path)
    return best_order


def expanding_window_forecast(stock, df_stock):
    """
    Perform expanding-window ARIMA backtest on a single stock.
    Saves checkpoint as soon as completed.
    """
    results = []
    df_stock = df_stock.sort_values("Date").reset_index(drop=True)
    y, dates = df_stock["Close"], df_stock["Date"]
    n = len(y)

    if n <= ROLLING_START + FORECAST_HORIZON:
        logging.warning(f"{stock}: insufficient data ({n} obs), skipping.")
        return None

    cache_path = os.path.join(MODEL_DIR, f"{stock}_order.pkl")
    best_order = select_best_order(y.iloc[:ROLLING_START], cache_path)

    logging.info(f"{stock}: Using ARIMA{best_order} with {n} data points.")

    for end_idx in range(ROLLING_START, n - FORECAST_HORIZON, FORECAST_HORIZON):
        train = y.iloc[:end_idx]
        test = y.iloc[end_idx:end_idx + FORECAST_HORIZON]
        test_dates = dates.iloc[end_idx:end_idx + FORECAST_HORIZON]

        try:
            model = ARIMA(train, order=best_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, forecast))

            results.append(pd.DataFrame({
                "Date": test_dates.values,
                "Stock": stock,
                "Actual": test.values,
                "Forecast": forecast.values,
                "RMSE": rmse,
                "Order_p": best_order[0],
                "Order_d": best_order[1],
                "Order_q": best_order[2],
                "Train_End_Date": dates.iloc[end_idx - 1]
            }))
        except Exception as e:
            logging.error(f"{stock}: failed at {end_idx} → {e}")
            continue

    if not results:
        return None

    out_df = pd.concat(results, ignore_index=True)
    out_path = os.path.join(MODEL_DIR, f"{stock}_forecasts.parquet")
    out_df.to_parquet(out_path, index=False)
    logging.info(f"{stock}: saved forecasts ({len(out_df)} rows).")
    return out_df


# ============================================================
# Main pipeline
# ============================================================

def run_expanding_arima():
    """Run expanding-window ARIMA in parallel for all stocks."""
    logging.info("Loading preprocessed data...")
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Stock" not in df.columns or "Close" not in df.columns:
        raise ValueError("Expected columns ['Stock', 'Date', 'Close'].")

    logging.info(f"Dataset loaded: {df.shape}, running on {N_JOBS} CPU cores.")
    stocks = sorted(df["Stock"].unique())

    completed = {f.split('_forecasts.parquet')[0] for f in os.listdir(MODEL_DIR) if f.endswith("_forecasts.parquet")}
    stocks = [s for s in stocks if s not in completed]

    logging.info(f"Remaining stocks to process: {len(stocks)} (skipping {len(completed)})")

    if not stocks:
        logging.info("All stocks already processed. Skipping retraining.")
        return

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(expanding_window_forecast)(s, df[df["Stock"] == s]) for s in stocks
    )

    results = [r for r in results if r is not None]
    if not results:
        logging.warning("No results produced.")
        return

    all_df = pd.concat(results, ignore_index=True)
    summary = all_df.groupby("Stock")["RMSE"].mean().reset_index()

    forecasts_path = os.path.join(MODEL_DIR, "arima_expanding_forecasts.parquet")
    summary_path = os.path.join(MODEL_DIR, "arima_expanding_summary.csv")

    all_df.to_parquet(forecasts_path, index=False)
    summary.to_csv(summary_path, index=False)

    logging.info(f"Expanding-window ARIMA complete.")
    logging.info(f"Forecasts saved → {forecasts_path}")
    logging.info(f"Summary saved → {summary_path}")
