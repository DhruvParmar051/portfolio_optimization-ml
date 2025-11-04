"""
optimize_portfolio.py

Enhanced portfolio optimization module.

Now includes:
- Forecast-based (expected) Sharpe ratio
- Realized Sharpe ratio (using actual returns from next period, if available)

Author: Dhruv Parmar
Date: 2025-11-04
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
FORECASTS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BACKTEST_RESULTS_DIR = os.path.join(BASE_DIR, "data", "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

RISK_FREE_RATE = 0.0  # daily risk-free rate
TRADING_DAYS = 252    # number of trading days in a year

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Utility Functions
# ============================================================

def load_forecasts():
    """Load all ARIMA forecast parquet files."""
    logging.info("Loading ARIMA forecast files...")
    all_files = [
        os.path.join(FORECASTS_DIR, f)
        for f in os.listdir(FORECASTS_DIR)
        if f.endswith("_forecasts.parquet")
    ]
    if not all_files:
        raise RuntimeError("No forecast files found.")
    df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    logging.info(f"Loaded forecasts: {df.shape}")
    return df


def compute_expected_returns(forecasts_df: pd.DataFrame):
    """Compute mean returns and covariance from ARIMA forecasts."""
    forecasts_df = forecasts_df.sort_values(["Stock", "Date"]).drop_duplicates(subset=["Stock", "Date"])
    pivot = forecasts_df.pivot(index="Date", columns="Stock", values="Forecast")
    returns = pivot.pct_change(fill_method=None).dropna(how="all")
    valid_stocks = returns.columns[returns.isna().mean() < 0.1]
    returns = returns[valid_stocks].fillna(0)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    logging.info(f"Computed daily returns for {len(mean_returns)} valid stocks.")
    logging.info(f"Returns matrix shape: {returns.shape}")
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Compute portfolio performance metrics."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """Optimize portfolio for maximum Sharpe ratio."""
    logging.info("Optimizing portfolio...")
    n = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 0.05) for _ in range(n))  # ≤5% per stock
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


def compute_realized_sharpe(weights, realized_returns_path):
    """Compute realized Sharpe ratio using actual market returns."""
    if not os.path.exists(realized_returns_path):
        logging.warning("⚠️ No realized returns found — skipping realized Sharpe computation.")
        return np.nan, np.nan, np.nan

    df_real = pd.read_parquet(realized_returns_path)
    df_real = df_real.pivot(index="Date", columns="Stock", values="Close").pct_change().dropna()
    valid_stocks = [s for s in df_real.columns if s in weights.index]
    df_real = df_real[valid_stocks]
    weights_vec = np.array(weights.loc[valid_stocks])
    port_ret = df_real.dot(weights_vec)
    port_mean = port_ret.mean()
    port_vol = port_ret.std()
    sharpe_daily = (port_mean - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    sharpe_annual = sharpe_daily * np.sqrt(TRADING_DAYS)
    return port_mean, port_vol, sharpe_annual


def save_results(weights, mean_returns, cov_matrix):
    """Save optimized portfolio weights and summary with forecast & realized metrics."""
    weights_df = pd.DataFrame({
        "Stock": mean_returns.index,
        "Weight": weights
    }).sort_values(by="Weight", ascending=False)

    weights_path = os.path.join(RESULTS_DIR, "optimized_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    logging.info(f"Saved optimized weights → {weights_path}")

    # Forecast-based Sharpe
    port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, RISK_FREE_RATE)
    annualized_sharpe = sharpe * np.sqrt(TRADING_DAYS)

    # Realized Sharpe (optional)
    realized_path = os.path.join(BACKTEST_RESULTS_DIR, "backtest_results_2025-01-01_2025-03-31.parquet")
    realized_mean, realized_vol, realized_sharpe = compute_realized_sharpe(weights_df.set_index("Stock"), realized_path)

    summary = pd.DataFrame([{
        "Expected_Return": port_return,
        "Volatility": port_vol,
        "Sharpe_Ratio_Daily": sharpe,
        "Sharpe_Ratio_Annualized": annualized_sharpe,
        "Realized_Return": realized_mean,
        "Realized_Volatility": realized_vol,
        "Realized_Sharpe_Annualized": realized_sharpe
    }])

    summary_path = os.path.join(RESULTS_DIR, "portfolio_summary.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Saved portfolio summary → {summary_path}")

    logging.info("=== Portfolio Optimization Complete ===")
    print(summary.to_string(index=False))


# ============================================================
# Main Entry
# ============================================================

def portfolio_optimization():
    logging.info("=== Running Portfolio Optimization ===")
    forecasts = load_forecasts()
    mean_returns, cov_matrix = compute_expected_returns(forecasts)
    weights = optimize_portfolio(mean_returns, cov_matrix)
    save_results(weights, mean_returns, cov_matrix)
