"""
backtest_q1.py

Performs quarterly backtesting for ARIMA forecasts (individual-stock models).

Workflow:
1. Load latest ARIMA forecasts from models directory.
2. Detect last forecast date (typically Dec 2024).
3. Fetch next-quarter data from Yahoo Finance (e.g., Jan–Mar 2025).
4. Match forecasted vs. actual prices using ±5 trading days tolerance.
5. Compute RMSE, MAE, and Directional Accuracy per stock.
6. Save detailed and summary results to backtest/results/.

Author: Dhruv
Date: 2025-11-03
"""

# ============================================================
# Imports
# ============================================================
import os
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import timedelta

# ============================================================
# Paths and setup
# ============================================================
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "arima_expanding_forecasts.parquet")

BACKTEST_DIR = os.path.join(BASE_DIR, "backtest")
DATA_DIR = os.path.join(BACKTEST_DIR, "data")
RESULTS_DIR = os.path.join(BACKTEST_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Utility functions
# ============================================================
def detect_forecast_period():
    """Detect last forecast date and infer next quarter range."""
    df = pd.read_parquet(MODEL_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    next_start = last_date + timedelta(days=1)
    next_end = next_start + timedelta(days=90)
    stocks = df["Stock"].unique().tolist()
    logging.info(f"Detected last forecast date = {last_date.date()}, running backtest for {next_start.date()} → {next_end.date()}")
    return stocks, next_start, next_end, df


def fetch_next_quarter_data(stocks, start, end):
    """Fetch next-quarter price data for given tickers."""
    logging.info(f"Fetching price data for {len(stocks)} tickers ({start.date()}–{end.date()})...")
    data = yf.download(
        tickers=stocks,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        threads=True,
        auto_adjust=True
    )

    all_data = []
    for ticker in stocks:
        try:
            df_t = data[ticker].reset_index()[["Date", "Close"]]
            df_t["Stock"] = ticker
            all_data.append(df_t)
        except Exception:
            continue

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.to_parquet(os.path.join(DATA_DIR, "q1_raw.parquet"))
    logging.info(f"Saved Q1 raw data → {os.path.join(DATA_DIR, 'q1_raw.parquet')}")
    return df_all


def evaluate_forecasts(forecasts_df, actual_df, tolerance_days=5):
    """Match forecasts to actuals and compute backtest metrics."""
    results = []
    forecasts_df["Date"] = pd.to_datetime(forecasts_df["Date"])
    actual_df["Date"] = pd.to_datetime(actual_df["Date"])

    for stock in forecasts_df["Stock"].unique():
        f_df = forecasts_df[forecasts_df["Stock"] == stock].copy()
        a_df = actual_df[actual_df["Stock"] == stock].copy()
        if a_df.empty or f_df.empty:
            continue

        # For each forecasted date, find nearest actual date within ±tolerance_days
        merged = []
        for _, row in f_df.iterrows():
            diff = (a_df["Date"] - row["Date"]).abs()
            nearest_idx = diff.idxmin()
            if diff.min().days <= tolerance_days:
                merged.append({
                    "Stock": stock,
                    "Forecast_Date": row["Date"],
                    "Forecast": row["Forecast"],
                    "Actual_Date": a_df.loc[nearest_idx, "Date"],
                    "Actual": a_df.loc[nearest_idx, "Close"]
                })

        if not merged:
            continue

        dfm = pd.DataFrame(merged)
        dfm["Error"] = dfm["Actual"] - dfm["Forecast"]
        dfm["Pct_Error"] = dfm["Error"] / dfm["Actual"]
        dfm["Direction_Acc"] = np.sign(dfm["Forecast"].diff()) == np.sign(dfm["Actual"].diff())

        rmse = np.sqrt(np.mean(dfm["Error"] ** 2))
        mae = np.mean(np.abs(dfm["Error"]))
        direction_acc = dfm["Direction_Acc"].mean() * 100

        results.append({
            "Stock": stock,
            "RMSE": rmse,
            "MAE": mae,
            "Directional_Accuracy(%)": direction_acc,
            "Matched_Samples": len(dfm)
        })

    if not results:
        logging.warning("No valid matches found — backtest metrics empty.")
        return None

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "backtest_summary.csv"), index=False)
    logging.info(f"Saved backtest summary → {os.path.join(RESULTS_DIR, 'backtest_summary.csv')}")
    return summary


# ============================================================
# Main entry point
# ============================================================
def run_backtest():
    """Run Q1 2025 (or next-quarter) ARIMA backtest."""
    logging.info("=== Running ARIMA Backtest for Next Quarter ===")

    stocks, start, end, forecasts_df = detect_forecast_period()
    actual_df = fetch_next_quarter_data(stocks, start, end)
    summary = evaluate_forecasts(forecasts_df, actual_df)

    if summary is not None:
        logging.info("\n" + summary.describe().to_string())
    else:
        logging.warning("Backtest completed but no valid metrics were produced (likely no overlapping forecast dates).")

    return summary
