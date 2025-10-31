"""
portfolio_optimizer.py

Generates an optimal portfolio allocation based on user input:
- Stocks of interest
- Total capital
- Risk tolerance (low / medium / high)
- Investment horizon (in months or years)

Uses the trained return and risk models to estimate expected return
and volatility, then applies Mean-Variance Optimization to find the
best allocation.

Finally, estimates the expected portfolio value at the end of the
investment period using compounded returns.

Author: Dhruv
Date: 2025-10-31
"""

# ======================================================================
# Imports
# ======================================================================

import os
import numpy as np
import pandas as pd
import joblib
import logging
from scipy.optimize import minimize

# ======================================================================
# Configuration
# ======================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = os.path.join(os.getcwd(), "models")
RETURN_MODEL_PATH = os.path.join(MODEL_DIR, "model_return.pkl")
VOL_MODEL_PATH = os.path.join(MODEL_DIR, "model_vol.pkl")

# ======================================================================
# Core Portfolio Optimizer
# ======================================================================

def portfolio_optimizer(selected_stocks, capital, risk_tolerance, time_period_months):
    """
    Generate an optimal portfolio based on user inputs.

    Parameters
    ----------
    selected_stocks : list[str]
        Stock symbols user is interested in
    capital : float
        Total amount to invest
    risk_tolerance : str
        'low', 'medium', or 'high'
    time_period_months : int
        Investment duration in months

    Returns
    -------
    pd.DataFrame
        Portfolio with optimal weights, expected returns, volatility, and
        projected investment outcome.
    """

    # ---------------------------------------------------------
    # Load trained models
    # ---------------------------------------------------------
    if not os.path.exists(RETURN_MODEL_PATH) or not os.path.exists(VOL_MODEL_PATH):
        raise FileNotFoundError("Models not found. Please run model_training.py first.")

    model_return = joblib.load(RETURN_MODEL_PATH)
    model_vol = joblib.load(VOL_MODEL_PATH)

    # ---------------------------------------------------------
    # Load most recent stock features
    # ---------------------------------------------------------
    data_path = os.path.join(os.getcwd(), "data", "processed", "final_features.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed data not found. Run the preprocessing pipeline first.")

    df = pd.read_parquet(data_path)
    df = df[df["Stock"].isin(selected_stocks)].groupby("Stock").tail(1)

    if df.empty:
        raise ValueError("None of the selected stocks found in processed data.")

    X = df.drop(columns=["future_return", "volatility", "Stock", "Date"], errors="ignore")

    expected_returns = model_return.predict(X)  # per period (e.g., daily or monthly)
    predicted_vol = model_vol.predict(X)

    # ---------------------------------------------------------
    # Optimization setup
    # ---------------------------------------------------------
    n = len(selected_stocks)
    cov_matrix = np.diag(predicted_vol ** 2)

    # Map user risk tolerance
    risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
    risk_factor = risk_map.get(risk_tolerance.lower(), 0.5)

    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Higher risk tolerance => prioritize returns more
        return - (risk_factor * portfolio_return - (1 - risk_factor) * portfolio_vol)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_guess = np.ones(n) / n

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    weights = result.x

    # ---------------------------------------------------------
    # Expected Portfolio Performance
    # ---------------------------------------------------------
    portfolio_expected_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Assuming expected_returns is monthly return, compound it:
    monthly_return = portfolio_expected_return
    total_return = (1 + monthly_return) ** time_period_months - 1
    expected_final_value = capital * (1 + total_return)
    expected_profit = expected_final_value - capital

    # ---------------------------------------------------------
    # Build Result Table
    # ---------------------------------------------------------
    df_result = pd.DataFrame({
        "Stock": selected_stocks,
        "Weight": weights,
        "Expected_Monthly_Return": expected_returns,
        "Predicted_Volatility": predicted_vol,
        "Investment_Amount": weights * capital
    })

    summary = {
        "Total_Capital": capital,
        "Expected_Annualized_Return_%": round(total_return * 100, 2),
        "Expected_Final_Value": round(expected_final_value, 2),
        "Expected_Profit": round(expected_profit, 2),
        "Portfolio_Volatility": round(portfolio_volatility, 4)
    }

    logging.info("✅ Portfolio optimization complete.")
    logging.info(f"Expected final portfolio value: ₹{expected_final_value:,.2f}")
    logging.info(f"Expected profit: ₹{expected_profit:,.2f}")

    return df_result, summary
