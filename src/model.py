"""
model.py

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
This module defines and trains an ARIMA model for time-series forecasting
on each stock’s adjusted closing prices.
=======
Trains and evaluates per-stock ARIMA models using time-series data.
The script loads train/validation splits, fits ARIMA models, forecasts
validation periods, and saves the results.
>>>>>>> 9f88a9c (Model dala hai guyysss)

Pipeline Steps:
1. Load X/y train–validation splits from `data/splits/`
2. Fit ARIMA model on the training target (`y_train`)
3. Forecast the validation horizon length
4. Evaluate model performance using RMSE
5. Save model summaries and predictions

Author: Dhruv
Date: 2025-11-01
"""
>>>>>>> c97fbfd (Kuch to kiya hai)

# ======================================================================
# Imports
# ======================================================================

import os
import numpy as np
<<<<<<< HEAD
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

def load_data():
    """Load pre-split training and validation datasets."""
    logging.info("Loading training and validation splits...")

    X_train = pd.read_parquet(X_TRAIN_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH)["target"]
    X_val = pd.read_parquet(X_VAL_PATH)
    y_val = pd.read_parquet(Y_VAL_PATH)["target"]

    logging.info(f"Loaded: X_train={X_train.shape}, X_val={X_val.shape}")
    return X_train, y_train, X_val, y_val


def train_arima(y_train, order=(1, 1, 1)):
    """Fit an ARIMA model on training data."""
    logging.info(f"Training ARIMA model with order={order}...")
    model = ARIMA(y_train, order=order)
    fitted_model = model.fit()
    logging.info("Model training completed.")
    return fitted_model


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
=======
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
DATA_DIR = os.path.join(os.getcwd(), "data", "processed_data")
MODEL_OUTPUT_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def fit_arima_per_stock(train_df, valid_df, order=(1, 1, 1)):
    """
    Fits an ARIMA model for each stock and evaluates performance.

    Args:
        train_df (pd.DataFrame): training dataset with columns ['Date', 'Ticker', 'Adj Close']
        valid_df (pd.DataFrame): validation dataset
        order (tuple): ARIMA order (p, d, q)

    Returns:
        pd.DataFrame: results with metrics per stock
    """
    results = []

    tickers = train_df["Ticker"].unique()
    for ticker in tickers:
        logging.info(f"Training ARIMA{order} for {ticker}...")

        train_data = train_df[train_df["Ticker"] == ticker].sort_values("Date")
        valid_data = valid_df[valid_df["Ticker"] == ticker].sort_values("Date")

        try:
            model = ARIMA(train_data["Adj Close"], order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(valid_data))

            mae = mean_absolute_error(valid_data["Adj Close"], forecast)
            rmse = np.sqrt(mean_squared_error(valid_data["Adj Close"], forecast))

            results.append({
                "Ticker": ticker,
                "MAE": mae,
                "RMSE": rmse
            })

            # Save model summary
            with open(os.path.join(MODEL_OUTPUT_DIR, f"{ticker}_arima_summary.txt"), "w") as f:
                f.write(str(model_fit.summary()))

        except Exception as e:
            logging.error(f"ARIMA failed for {ticker}: {e}")

    return pd.DataFrame(results)


def main():
    logging.info("Loading training and validation datasets...")
    train_path = os.path.join(DATA_DIR, "train.parquet")
    valid_path = os.path.join(DATA_DIR, "valid.parquet")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    logging.info("Fitting ARIMA models per stock...")
    metrics_df = fit_arima_per_stock(train_df, valid_df, order=(1, 1, 1))

    metrics_path = os.path.join(MODEL_OUTPUT_DIR, "arima_results.csv")
    metrics_df.to_csv(metrics_path, index=False)

    logging.info(f"ARIMA training complete. Results saved to: {metrics_path}")


if __name__ == "__main__":
    main()
>>>>>>> c97fbfd (Kuch to kiya hai)
