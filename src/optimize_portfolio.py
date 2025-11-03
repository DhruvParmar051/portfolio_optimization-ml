"""
optimize_portfolio.py

Mean-variance portfolio optimization using ARIMA model forecasts.

Steps:
1. Load forecasted prices (normal or backtest forecasts).
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

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtest_results")
BACKTEST_FORECASTS_DIR = os.path.join(RESULTS_DIR, "backtest_forecasts")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

RISK_FREE_RATE = 0.0  # daily risk-free rate
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================
# Utility Functions
# ============================================================

def load_forecasts(backtest: bool = False):
    """Load forecast parquet files from the appropriate directory."""
    source_dir = BACKTEST_FORECASTS_DIR if backtest else MODELS_DIR
    logging.info(f"Loading forecasts from: {source_dir}")

    all_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith("_forecasts.parquet")
    ]

    if not all_files:
        raise RuntimeError(f"No forecast files found in {source_dir}")

    df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    logging.info(f"Loaded {len(all_files)} forecast files → Shape: {df.shape}")
    return df


def compute_expected_returns(forecasts_df: pd.DataFrame):
    """Compute mean returns and covariance from ARIMA forecasts."""
    logging.info("Computing expected returns and covariance...")

    forecasts_df = forecasts_df.sort_values(["Stock", "Date"]).drop_duplicates(subset=["Stock", "Date"])
    pivot = forecasts_df.pivot(index="Date", columns="Stock", values="Forecast")

    # Convert to daily percentage returns
    returns = pivot.pct_change(fill_method=None).dropna(how="all")

    # Filter out bad stocks
    valid_stocks = returns.columns[returns.isna().mean() < 0.1]
    returns = returns[valid_stocks].fillna(0)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    logging.info(f"Computed expected returns for {len(mean_returns)} stocks.")
    logging.info(f"Covariance matrix shape: {cov_matrix.shape}")

    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Compute portfolio metrics (return, volatility, Sharpe)."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe_ratio


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """Optimize portfolio for maximum Sharpe ratio."""
    logging.info("Optimizing portfolio weights...")

    n = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 0.05) for _ in range(n))  # Limit any stock to ≤5%

    result = minimize(
        lambda w: -portfolio_performance(w, *args)[2],
        x0=np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    if not result.success:
        logging.warning(f"Optimization failed: {result.message}")

    return result.x


def save_results(weights, mean_returns, cov_matrix, backtest: bool = False):
    """Save optimized portfolio weights and performance summary."""
    save_dir = BACKTEST_RESULTS_DIR if backtest else RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    weights_df = pd.DataFrame({
        "Stock": mean_returns.index,
        "Weight": weights
    }).sort_values(by="Weight", ascending=False)

    weights_path = os.path.join(save_dir, "optimized_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    logging.info(f"Saved optimized weights → {weights_path}")

    port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, RISK_FREE_RATE)
    summary = pd.DataFrame([{
        "Expected_Return": port_return,
        "Volatility": port_vol,
        "Sharpe_Ratio_Daily": sharpe,
        "Sharpe_Ratio_Annualized": sharpe * np.sqrt(252)
    }])

    summary_path = os.path.join(save_dir, "portfolio_summary.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Saved portfolio summary → {summary_path}")

    logging.info("=== Portfolio Optimization Complete ===")
    print(summary.to_string(index=False))


# ============================================================
# Main Entry Point
# ============================================================

def portfolio_optimization(backtest: bool = False):
    """
    Run portfolio optimization using ARIMA forecasts.
    If backtest=True, use forecasts generated during backtesting.
    """
    mode = "BACKTEST" if backtest else "NORMAL"
    logging.info(f"=== Running Portfolio Optimization [{mode}] ===")

    forecasts = load_forecasts(backtest)
    mean_returns, cov_matrix = compute_expected_returns(forecasts)
    weights = optimize_portfolio(mean_returns, cov_matrix)
    save_results(weights, mean_returns, cov_matrix, backtest)
