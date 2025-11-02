"""
model.py

Expanding-window ARIMA backtest per stock for portfolio modeling (parallel + optimized).

This module:
1. Loads cleaned & preprocessed stock-price data.
2. Performs an expanding-window backtest (750-day minimum training window).
3. Re-fits ARIMA(p,d,q) models using efficient parallelization.
4. Evaluates out-of-sample RMSE and stores forecasts.
5. Supports resume via checkpointing to handle large-scale runs.

Optimizations:
- Parallelized across stocks (via joblib)
- Caches best (p,d,q) order per stock to avoid repeated grid search
- Saves intermediate stock results to disk incrementally

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
from itertools import product
from joblib import Parallel, delayed, dump, load
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

DATA_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")
MODEL_DIR = os.path.join(os.getcwd(), "models", "arima_expanding")
os.makedirs(MODEL_DIR, exist_ok=True)

ROLLING_START = 750           # initial expanding window length
FORECAST_HORIZON = 30         # forecast next 30 days
MAX_P, MAX_D, MAX_Q = 2, 1, 2 # smaller grid for speed
N_JOBS = max(1, os.cpu_count() // 2)  # parallel cores

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Utility functions
# ============================================================

def load_data():
    """Load pre-split training and validation datasets."""
    logging.info("Loading training and validation splits...")

    X_train = pd.read_parquet(X_TRAIN_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH)["target"]
    X_val = pd.read_parquet(X_VAL_PATH)
    y_val = pd.read_parquet(Y_VAL_PATH)["target"]

    logging.info(f"Loaded: X_train={X_train.shape}, X_val={X_val.shape}")
    return X_train, y_train, X_val, y_val


def train_arima(y_train, order=(1, 1, 1)):
    """Fit an ARIMA model on training data."""
    logging.info(f"Training ARIMA model with order={order}...")
    model = ARIMA(y_train, order=order)
    fitted_model = model.fit()
    logging.info("Model training completed.")
    return fitted_model


def evaluate_model(model, y_val):
    """Forecast and evaluate the ARIMA model."""
    logging.info("Forecasting on validation data...")
    forecast = model.forecast(steps=len(y_val))
    rmse = np.sqrt(mean_squared_error(y_val, forecast))
    logging.info(f"Validation RMSE: {rmse:.4f}")
    return forecast, rmse


def save_model(model, name="arima_model.pkl"):
    """Save the trained model object."""
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    logging.info(f"Model saved at: {path}")


def save_predictions(y_val, forecast):
    """Save forecast vs actual comparison."""
    results = pd.DataFrame({"Actual": y_val.values, "Forecast": forecast})
    output_path = os.path.join(MODEL_DIR, "arima_predictions.parquet")
    results.to_parquet(output_path, index=False)
    logging.info(f"Predictions saved at: {output_path}")

# ======================================================================
# Main Pipeline
# ======================================================================

def run_arima_models():
    """Run the ARIMA training and evaluation pipeline."""
    try:
        X_train, y_train, X_val, y_val = load_data()

        # Train ARIMA on training target
        model = train_arima(y_train, order=(1, 1, 1))

        # Evaluate
        forecast, rmse = evaluate_model(model, y_val)

        # Save model and results
        save_model(model)
        save_predictions(y_val, forecast)

        logging.info("ARIMA training and evaluation pipeline completed successfully.")
    except Exception as e:
        logging.exception("ARIMA pipeline failed.")
