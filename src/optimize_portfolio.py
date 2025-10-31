"""
optimize_portfolio.py

Optimizes portfolio allocation for user-selected stocks
using trained global return and volatility models.
"""

import os, numpy as np, pandas as pd, joblib, logging
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = os.path.join(os.getcwd(), "models")
RETURN_MODEL_PATH = os.path.join(MODEL_DIR, "model_return.pkl")
VOL_MODEL_PATH = os.path.join(MODEL_DIR, "model_vol.pkl")


def portfolio_optimizer(selected_stocks, capital, risk_tolerance, months):
    if not (os.path.exists(RETURN_MODEL_PATH) and os.path.exists(VOL_MODEL_PATH)):
        raise FileNotFoundError("Trained models missing.")

    m_ret, m_vol = joblib.load(RETURN_MODEL_PATH), joblib.load(VOL_MODEL_PATH)
    df = pd.read_parquet(os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet"))
    df = df[df["Stock"].isin(selected_stocks)].groupby("Stock").tail(1)

    X = df.drop(columns=["Date", "Stock"], errors="ignore").select_dtypes(include="number").fillna(0)
    exp_ret, exp_vol = m_ret.predict(X), m_vol.predict(X)

    n = len(selected_stocks)
    cov = np.diag(exp_vol ** 2)
    risk_factor = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(risk_tolerance.lower(), 0.5)

    def objective(w):
        p_ret, p_vol = np.dot(w, exp_ret), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return - (risk_factor * p_ret - (1 - risk_factor) * p_vol)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    res = minimize(objective, np.ones(n)/n, bounds=[(0,1)]*n, constraints=cons)
    w = res.x

    port_ret = np.dot(w, exp_ret)
    total_ret = (1 + port_ret) ** months - 1
    final_val = capital * (1 + total_ret)
    profit = final_val - capital

    df_out = pd.DataFrame({
        "Stock": selected_stocks,
        "Weight": w,
        "Expected_Monthly_Return": exp_ret,
        "Predicted_Volatility": exp_vol,
        "Investment_Amount": w * capital
    })
    summary = {
        "Total_Capital": capital,
        "Expected_Final_Value": round(final_val, 2),
        "Expected_Profit": round(profit, 2),
        "Portfolio_Volatility": round(np.sqrt(np.dot(w.T, np.dot(cov, w))), 4)
    }
    logging.info(f"Portfolio optimized → ₹{final_val:,.2f} expected value.")
    return df_out, summary
