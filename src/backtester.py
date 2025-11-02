"""
backtest.py

Aggregates ARIMA forecasts into a daily portfolio strategy.

Approach:
1. Load per-stock forecasts from models/arima/
2. Rank stocks by predicted next-day return
3. Build equal-weight long-only portfolio (top-N stocks)
4. Compute cumulative return vs. baseline
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = os.path.join(os.getcwd(), "models", "arima")
BACKTEST_DIR = os.path.join(os.getcwd(), "backtest_results")
os.makedirs(BACKTEST_DIR, exist_ok=True)

TOP_N = 20


def load_forecasts():
    """Load all per-stock forecast files."""
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_forecast.parquet")]
    data = []
    for f in files:
        stock = f.replace("_forecast.parquet", "")
        df = pd.read_parquet(os.path.join(MODEL_DIR, f))
        df["Stock"] = stock
        data.append(df)
    return pd.concat(data, ignore_index=True)


def backtest_portfolio(forecasts: pd.DataFrame):
    """Compute simple top-N long portfolio performance."""
    forecasts["Signal"] = forecasts["y_val_pred"]
    forecasts["Actual"] = forecasts["y_val_actual"]

    # Rank stocks by forecasted return
    forecasts["Rank"] = forecasts.groupby(forecasts.index)["Signal"].rank(ascending=False)
    topN = forecasts[forecasts["Rank"] <= TOP_N]

    avg_return = topN.groupby(forecasts.index)["Actual"].mean()

    cum_return = (1 + avg_return).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(cum_return.index, cum_return.values, label=f"Top {TOP_N} ARIMA Portfolio")
    plt.title("ARIMA Backtest Cumulative Return")
    plt.xlabel("Time (validation steps)")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BACKTEST_DIR, "arima_backtest.png"))
    plt.close()

    logging.info(f"Backtest complete. Final cumulative return: {cum_return.iloc[-1]:.2f}")
    cum_return.to_csv(os.path.join(BACKTEST_DIR, "arima_cum_returns.csv"))

    return cum_return


def run_backtest():
    forecasts = load_forecasts()
    logging.info(f"Loaded {forecasts.shape[0]} forecast records.")
    backtest_portfolio(forecasts)


