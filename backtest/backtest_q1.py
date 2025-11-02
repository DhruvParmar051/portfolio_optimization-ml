"""
backtest_q1_2025.py

Fetches Q1 2025 stock price data, runs it through the same cleaning,
feature engineering, and preprocessing pipeline, then compares actual
returns to ARIMA expanding forecasts for model validation.

All results (predictions + metrics) are saved under backtest/results/.

Author: Dhruv
Date: 2025-11-02
"""

# ============================================================
# Imports
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.getcwd()
BACKTEST_DIR = os.path.join(BASE_DIR, "backtest")
DATA_DIR = os.path.join(BACKTEST_DIR, "data")
RESULTS_DIR = os.path.join(BACKTEST_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "models", "arima_expanding_forecasts.parquet")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# 1. Fetch Q1 2025 Data
# ============================================================

def fetch_q1_data(start_date="2025-01-01", end_date="2025-03-31"):
    """Downloads Q1 2025 data for tickers in ARIMA model results."""
    logging.info("Loading list of modeled stocks...")
    model_df = pd.read_parquet(MODEL_PATH)
    tickers = sorted(model_df["Stock"].unique().tolist())

    logging.info(f"Fetching Q1 2025 price data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True, threads=True)

    combined = []
    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue
        df = data[t].copy().reset_index()
        df["Stock"] = t
        combined.append(df)
    q1 = pd.concat(combined, ignore_index=True)

    raw_path = os.path.join(DATA_DIR, "raw_q1.parquet")
    q1.to_parquet(raw_path, index=False)
    logging.info(f"Raw Q1 data saved → {raw_path}")
    return q1

# ============================================================
# 2. Minimal Cleaning + Feature Preparation
# ============================================================

def prepare_q1_data(df):
    """Simple preprocessing to match structure of ARIMA model input."""
    df = df.dropna(subset=["Close"])
    df = df.sort_values(["Stock", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Return"] = df.groupby("Stock")["Close"].pct_change()
    cleaned_path = os.path.join(DATA_DIR, "cleaned_q1.parquet")
    df.to_parquet(cleaned_path, index=False)
    logging.info(f"Cleaned Q1 data saved → {cleaned_path}")
    return df

# ============================================================
# 3. Run Backtest
# ============================================================

def run_backtest(start_date="2025-01-01", end_date="2025-03-31"):
    """Compare ARIMA forecasts vs actuals for Q1 2025."""
    logging.info("=== Running ARIMA Backtest for Q1 2025 ===")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model_df = pd.read_parquet(MODEL_PATH)
    q1_raw = fetch_q1_data(start_date, end_date)
    q1_df = prepare_q1_data(q1_raw)

    model_df["Date"] = pd.to_datetime(model_df["Date"])
    q1_df["Date"] = pd.to_datetime(q1_df["Date"])

    # Merge on Stock + Date
    merged = pd.merge(
        model_df,
        q1_df[["Date", "Stock", "Close"]],
        on=["Stock", "Date"],
        how="inner",
        suffixes=("_Forecast", "_Actual")
    )

    if merged.empty:
        logging.warning("No overlapping dates found between forecasts and Q1 data.")
        return

    merged["Error"] = merged["Forecast"] - merged["Close"]
    merged["Absolute_Error"] = merged["Error"].abs()
    merged["Squared_Error"] = merged["Error"] ** 2
    merged["Directional_Accuracy"] = np.sign(merged["Forecast"].diff()) == np.sign(merged["Close"].diff())

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(merged["Close"], merged["Forecast"]))
    mae = mean_absolute_error(merged["Close"], merged["Forecast"])
    mape = np.mean(np.abs(merged["Error"] / merged["Close"])) * 100
    da = merged["Directional_Accuracy"].mean() * 100

    metrics = pd.DataFrame({
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE (%)": [mape],
        "Directional_Accuracy (%)": [da]
    })

    # Save results
    pred_path = os.path.join(RESULTS_DIR, "backtest_q1_2025_predictions.csv")
    metrics_path = os.path.join(RESULTS_DIR, "backtest_q1_2025_metrics.csv")

    merged.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    logging.info(f"Backtest predictions saved → {pred_path}")
    logging.info(f"Backtest metrics saved → {metrics_path}")
    logging.info("=== Backtest Complete ===")
    logging.info(f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, Dir.Acc={da:.2f}%")

    return metrics