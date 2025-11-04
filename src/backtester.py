"""
backtest.py

Executes the full backtesting workflow for the portfolio optimization pipeline.

Pipeline Steps:
1. Fetch backtest data for a specified date range
2. Clean and validate it
3. Perform feature engineering (within range)
4. Preprocess features using trained scalers/encoders
5. Run ARIMA predictions on backtest data
6. Optimize portfolio based on forecasts
7. Save and log results

Author: Dhruv Parmar
Date: 2025-11-03
"""

import os
import logging
import pandas as pd

# === Import project modules ===
from data_fetch import data_fetch
from data_cleaning import data_cleaning
from feature_engineering import feature_engineering
from preprocessor import preprocessor
from model import run_expanding_arima
from optimize_portfolio import portfolio_optimization

# ======================================================================
# Configuration
# ======================================================================
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2025-03-31"

BACKTEST_RESULTS_DIR = os.path.join(os.getcwd(), "data", "backtest_results")
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "backtest.log")),
        logging.StreamHandler(),
    ],
)

# ======================================================================
# Backtest Pipeline
# ======================================================================
def run_backtest(symbols=None):
    """
    Execute the full backtesting pipeline end-to-end.
    """
    try:
        logging.info("=== Starting Backtest Pipeline ===")
        logging.info(f"Backtest period: {BACKTEST_START} → {BACKTEST_END}")

        # [1] Fetch raw data
        raw_df = data_fetch(
            tickers_subset=symbols,
            start=BACKTEST_START,
            end=BACKTEST_END,
            backtest=True,
        )
        logging.info(f"✅ Data fetched: {raw_df.shape}")

        # [2] Clean data
        cleaned_df = data_cleaning(backtest=True)
        logging.info(f"✅ Data cleaned: {cleaned_df.shape}")

        # [3] Feature engineering
        features_df = feature_engineering(start_date=BACKTEST_START, end_date=BACKTEST_END)
        if features_df is None or features_df.empty:
            logging.warning("⚠️ No data available after feature engineering — check date range.")
            return
        logging.info(f"✅ Features created: {features_df.shape}")

        # [4] Preprocess features (use trained scaler/encoder)
        preprocessed_df = preprocessor(training_mode=False, backtest=True)
        if preprocessed_df is None or preprocessed_df.empty:
            logging.warning("⚠️ No preprocessed data available — aborting backtest.")
            return
        logging.info(f"✅ Data preprocessed: {preprocessed_df.shape}")

        # [5] Model inference (ARIMA expanding-window)
        # predictions_df = run_expanding_arima(backtest=True)
        # if predictions_df is None or predictions_df.empty:
        #     logging.warning("⚠️ No predictions generated — aborting backtest.")
        #     return
        # logging.info(f"✅ Model predictions complete: {predictions_df.shape}")

        # [6] Portfolio optimization
        results_df = portfolio_optimization(backtest=True)
        logging.info("✅ Portfolio optimization finished successfully.")

        # [7] Save final results
        output_path = os.path.join(
            BACKTEST_RESULTS_DIR,
            f"backtest_results_{BACKTEST_START}_{BACKTEST_END}.parquet",
        )
        if isinstance(results_df, pd.DataFrame):
            results_df.to_parquet(output_path, index=False)
        else:
            pd.DataFrame([results_df]).to_parquet(output_path, index=False)

        logging.info(f"✅ Backtest results saved → {output_path}")
        logging.info("=== Backtest Pipeline executed successfully ===")

    except Exception as e:
        logging.exception("❌ Backtest pipeline failed.")
        raise e


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == "__main__":
    run_backtest(symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
