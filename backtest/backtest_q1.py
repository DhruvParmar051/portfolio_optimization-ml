"""
backtest_q1.py

Performs out-of-sample backtesting for Q1 2025
on expanding-window ARIMA forecasts.

Loads per-stock forecast files from models/,
fetches actual Yahoo Finance prices for Q1 2025,
and compares predictions to realized returns.

Author: Dhruv
Date: 2025-11-03
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")
BACKTEST_DIR = os.path.join(BASE_DIR, "backtest")
DATA_DIR = os.path.join(BACKTEST_DIR, "data")
RESULTS_DIR = os.path.join(BACKTEST_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

START_DATE = "2025-01-01"
END_DATE = "2025-06-30"


# ============================================================
# Utility Functions
# ============================================================

def get_modeled_stocks():
    """Return list of valid stock tickers from model forecast files."""
    stocks = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith("_forecasts.parquet"):
            name = file.replace("_forecasts.parquet", "")
            # Skip non-stock or invalid filenames
            if not name.isalpha() or len(name) > 5:
                continue
            stocks.append(name)
    logging.info(f"Loaded {len(stocks)} modeled stocks.")
    return stocks


def fetch_q1_data(stocks):
    """Download Q1 2025 adjusted close prices from Yahoo Finance."""
    if not stocks:
        logging.error("No stocks to fetch.")
        return None

    logging.info(f"Fetching Q1 2025 price data for {len(stocks)} tickers...")
    data = yf.download(
        tickers=stocks,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        threads=True,
        auto_adjust=True,
        progress=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        df_list = []
        for ticker in stocks:
            if ticker in data.columns.get_level_values(0):
                sub = data[ticker].reset_index()
                sub["Stock"] = ticker
                df_list.append(sub)
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = data.reset_index()
        df["Stock"] = stocks[0]

    df = df.rename(columns={"Date": "Date", "Close": "Close"})
    df = df[["Date", "Stock", "Close"]]
    df.to_parquet(os.path.join(DATA_DIR, "raw_q1.parquet"), index=False)
    logging.info(f"Raw Q1 data saved → {os.path.join(DATA_DIR, 'raw_q1.parquet')}")

    df["Return"] = df.groupby("Stock")["Close"].pct_change()
    df = df.dropna()
    df.to_parquet(os.path.join(DATA_DIR, "cleaned_q1.parquet"), index=False)
    logging.info(f"Cleaned Q1 data saved → {os.path.join(DATA_DIR, 'cleaned_q1.parquet')}")
    return df


def evaluate_forecasts(q1_df):
    """Evaluate all stock forecasts against actual Q1 returns."""
    results = []

    for stock_file in os.listdir(MODEL_DIR):
        if not stock_file.endswith("_forecasts.parquet"):
            continue

        stock = stock_file.replace("_forecasts.parquet", "")
        f_path = os.path.join(MODEL_DIR, stock_file)
        forecast_df = pd.read_parquet(f_path)

        if "Date" not in forecast_df.columns or "Forecast" not in forecast_df.columns:
            logging.warning(f"{stock}: Missing required columns in forecast file.")
            continue

        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
        actual_df = q1_df[q1_df["Stock"] == stock].copy()

        if actual_df.empty:
            continue

        merged = pd.merge(
            forecast_df, actual_df,
            on=["Date", "Stock"], how="inner"
        )

        if merged.empty:
            continue

        merged["Error"] = merged["Forecast"] - merged["Return"]
        rmse = np.sqrt(np.mean(merged["Error"] ** 2))
        mae = np.mean(np.abs(merged["Error"]))
        mape = np.mean(np.abs(merged["Error"] / merged["Return"].replace(0, np.nan))) * 100

        results.append({
            "Stock": stock,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Count": len(merged)
        })

    if not results:
        logging.warning("No overlapping forecasts and actuals found.")
        return None

    result_df = pd.DataFrame(results).sort_values("RMSE")
    result_path = os.path.join(RESULTS_DIR, "backtest_metrics.csv")
    result_df.to_csv(result_path, index=False)
    logging.info(f"Backtest metrics saved → {result_path}")
    return result_df


# ============================================================
# Main Function
# ============================================================

def run_backtest():
    """Run Q1 2025 ARIMA forecast backtest."""
    logging.info("=== Running ARIMA Backtest for Q1 2025 ===")
    stocks = get_modeled_stocks()
    q1_df = fetch_q1_data(stocks)
    if q1_df is None:
        return None
    metrics = evaluate_forecasts(q1_df)
    if metrics is not None:
        logging.info("Backtest summary:")
        logging.info(metrics.head(10).to_string(index=False))
    else:
        logging.warning("Backtest produced no valid metrics.")
    return metrics