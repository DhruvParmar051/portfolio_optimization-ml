"""
model.py

This module defines and trains an ARIMA model for time-series forecasting
on each stock’s adjusted closing prices.

Steps:
1. Loads the training and validation data
2. Fits ARIMA(p, d, q) per stock
3. Generates forecasts
4. Evaluates performance (MAE, RMSE)
5. Saves model summaries and metrics

Author: Dhruv
"""

import os
import pandas as pd
import numpy as np
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
