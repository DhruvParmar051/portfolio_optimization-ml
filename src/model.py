"""
model.py

Trains and evaluates per-stock ARIMA models using time-series data.
The script loads train/validation splits, fits ARIMA models, forecasts
validation periods, and saves the results.

Pipeline Steps:
1. Load X/y train–validation splits from `data/splits/`
2. Fit ARIMA model on the training target (`y_train`)
3. Forecast the validation horizon length
4. Evaluate model performance using RMSE
5. Save model summaries and predictions

Author: Dhruv
Date: 2025-11-01
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import joblib

# ======================================================================
# Logging Configuration
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================================================================
# Paths
# ======================================================================

DATA_DIR = os.path.join(os.getcwd(), "data", "splits")
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.parquet")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.parquet")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.parquet")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.parquet")

# ======================================================================
# Core ARIMA Functions
# ======================================================================

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
