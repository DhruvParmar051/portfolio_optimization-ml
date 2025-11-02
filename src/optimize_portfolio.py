"""
optimize_portfolio.py

Performs portfolio optimization based on ARIMA forecasted returns.

Pipeline Steps:
1. Load ARIMA forecast results per stock
2. Merge forecasts into a daily forecast matrix
3. Estimate expected returns and covariance
4. Solve for optimal weights using mean–variance optimization
5. Backtest portfolio returns and compute performance metrics

Author: Dhruv
Date: 2025-11-02
"""

# ======================================================================
# Imports
# ======================================================================

import os
import numpy as np
import pandas as pd
import logging
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================================================================
# Paths
# ======================================================================

FORECAST_DIR = os.path.join(os.getcwd(), "models", "arima_expanding")
OUTPUT_DIR = os.path.join(os.getcwd(), "results", "portfolio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================
# Portfolio Optimization Core
# ======================================================================

def load_forecasts() -> pd.DataFrame:
    """Load all ARIMA forecast parquet files into a unified DataFrame."""
    logging.info("Loading ARIMA forecast files...")
    records = []
    for file in os.listdir(FORECAST_DIR):
        if file.endswith("_forecasts.parquet"):
            stock = file.replace("_forecasts.parquet", "")
            df = pd.read_parquet(os.path.join(FORECAST_DIR, file))
            df["Stock"] = stock
            records.append(df)
    if not records:
        raise FileNotFoundError("No ARIMA forecast files found.")
    forecasts = pd.concat(records, ignore_index=True)
    forecasts["Date"] = pd.to_datetime(forecasts["Date"])
    logging.info(f"Loaded forecasts for {forecasts['Stock'].nunique()} stocks.")
    return forecasts


def pivot_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format forecasts into wide-format (Date x Stock)."""
    pivoted = forecasts.pivot(index="Date", columns="Stock", values="Forecast").sort_index()
    pivoted = pivoted.fillna(method="ffill").dropna(how="all")
    return pivoted


def calculate_expected_returns(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily forecasted returns from forecasted prices."""
    returns = forecast_prices.pct_change().dropna(how="all")
    return returns


def mean_variance_opt(expected_returns: pd.Series, cov_matrix: pd.DataFrame):
    """Solve mean-variance optimization for maximum Sharpe ratio portfolio."""
    n = len(expected_returns)
    init_w = np.ones(n) / n

    def portfolio_stats(weights):
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol, ret / vol

    def neg_sharpe(weights):
        _, _, sharpe = portfolio_stats(weights)
        return -sharpe

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(neg_sharpe, init_w, bounds=bounds, constraints=constraints)
    if not result.success:
        logging.warning("Optimization failed; returning equal weights.")
        return init_w
    return result.x


def backtest_portfolio(forecast_returns: pd.DataFrame, window: int = 30):
    """
    Rolling backtest with periodic rebalancing.

    Parameters:
        forecast_returns: DataFrame of forecasted returns (Date x Stock)
        window: rebalancing frequency (days)
    """
    weights_record = []
    portfolio_returns = []

    for i in range(window, len(forecast_returns)):
        hist = forecast_returns.iloc[i-window:i]
        mu = hist.mean()
        cov = hist.cov()
        w_opt = mean_variance_opt(mu, cov)
        weights_record.append(w_opt)

        next_ret = forecast_returns.iloc[i].fillna(0)
        portfolio_ret = np.dot(w_opt, next_ret)
        portfolio_returns.append(portfolio_ret)

    weights_record = np.array(weights_record)
    portfolio_series = pd.Series(portfolio_returns, index=forecast_returns.index[window:])

    logging.info("Backtest completed.")
    return portfolio_series, weights_record


def evaluate_performance(portfolio_returns: pd.Series):
    """Compute key portfolio performance metrics."""
    cumulative = (1 + portfolio_returns).cumprod()
    ann_return = (cumulative.iloc[-1]) ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    summary = {
        "Cumulative Return": cumulative.iloc[-1] - 1,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe
    }
    logging.info(f"Performance: {summary}")
    return summary, cumulative


# ======================================================================
# Entry Point
# ======================================================================

def optimize_portfolio():
    """Run portfolio optimization and backtesting."""
    try:
        forecasts = load_forecasts()
        pivoted = pivot_forecasts(forecasts)
        forecast_returns = calculate_expected_returns(pivoted)

        portfolio_returns, weights = backtest_portfolio(forecast_returns, window=30)
        summary, cumulative = evaluate_performance(portfolio_returns)

        cumulative.to_csv(os.path.join(OUTPUT_DIR, "portfolio_cumulative.csv"))
        pd.DataFrame(weights).to_csv(os.path.join(OUTPUT_DIR, "portfolio_weights.csv"), index=False)

        logging.info("Portfolio optimization completed successfully.")
        logging.info(f"Results saved → {OUTPUT_DIR}")

    except Exception as e:
        logging.exception("Portfolio optimization pipeline failed.")
