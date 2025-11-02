"""
optimize_portfolio.py

Mean-variance portfolio optimization using ARIMA model forecasts.

Steps:
1. Load forecasted prices from ARIMA expanding-window results.
2. Compute expected returns & covariance matrix.
3. Optimize portfolio weights to maximize the Sharpe ratio.
4. Save optimized weights & summary metrics.

Author: Dhruv
Date: 2025-11-02
"""

# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

# ============================================================
# Configuration
# ============================================================

PREDICTIONS_DIR = os.path.join(os.getcwd(), "models", "arima_expanding")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RISK_FREE_RATE = 0.0  # daily risk-free rate
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Utility Functions
# ============================================================

def load_forecasts():
    """Load all ARIMA forecast parquet files."""
    logging.info("Loading ARIMA forecast files...")

    all_files = [
        os.path.join(PREDICTIONS_DIR, f)
        for f in os.listdir(PREDICTIONS_DIR)
        if f.endswith("_forecasts.parquet")
    ]

    if not all_files:
        raise RuntimeError("No forecast files found.")

    df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    logging.info(f"Loaded forecasts: {df.shape}")
    return df


def compute_expected_returns(forecasts_df: pd.DataFrame):
    """Compute mean returns and covariance from ARIMA forecasts."""
    logging.info("Computing expected returns and covariance...")

    forecasts_df = forecasts_df.sort_values(["Stock", "Date"]).drop_duplicates(subset=["Stock", "Date"])

    pivot = forecasts_df.pivot(index="Date", columns="Stock", values="Forecast")

    # convert to daily percentage returns
    returns = pivot.pct_change(fill_method=None).dropna(how="all")

    # remove bad stocks
    valid_stocks = returns.columns[returns.isna().mean() < 0.1]
    returns = returns[valid_stocks].fillna(0)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    logging.info(f"Computed expected returns for {len(mean_returns)} stocks.")
    logging.info(f"Covariance matrix shape: {cov_matrix.shape}")

    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Compute portfolio metrics (return, volatility, Sharpe)."""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe_ratio


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """Optimize portfolio for maximum Sharpe ratio."""
    logging.info("Optimizing portfolio...")

    n = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # constraints: sum(weights) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 0.05) for _ in range(n))  # cap any stock ≤5%

    result = minimize(
        lambda w: -portfolio_performance(w, *args)[2],  # maximize Sharpe
        x0=np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    if not result.success:
        logging.warning(f"Optimization failed: {result.message}")

    return result.x


def save_results(weights, mean_returns, cov_matrix):
    """Save optimized portfolio weights and performance summary."""
    weights_df = pd.DataFrame({
        "Stock": mean_returns.index,
        "Weight": weights
    }).sort_values(by="Weight", ascending=False)

    weights_path = os.path.join(RESULTS_DIR, "optimized_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    logging.info(f"Saved optimized weights -> {weights_path}")

    port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, RISK_FREE_RATE)
    daily_sharpe = sharpe
    annualized_sharpe = sharpe * np.sqrt(252)

    summary = pd.DataFrame([{
        "Expected_Return": port_return,
        "Volatility": port_vol,
        "Sharpe_Ratio_Daily": daily_sharpe,
        "Sharpe_Ratio_Annualized": annualized_sharpe
    }])

    summary_path = os.path.join(RESULTS_DIR, "portfolio_summary.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Saved portfolio summary -> {summary_path}")

    logging.info("=== Portfolio Optimization Complete ===")
    print(summary.to_string(index=False))


# ============================================================
# Main
# ============================================================

def portfolio_optimization():
    logging.info("=== Running Portfolio Optimization ===")

    forecasts = load_forecasts()
    mean_returns, cov_matrix = compute_expected_returns(forecasts)
    weights = optimize_portfolio(mean_returns, cov_matrix)
    save_results(weights, mean_returns, cov_matrix)

