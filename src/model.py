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

def model_training():
    """Train models for predicting returns and volatility."""

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed file not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # Ensure target columns exist
    if "future_return" not in df.columns or "volatility" not in df.columns:
        raise ValueError("Processed dataset must contain 'future_return' and 'volatility' columns")

    feature_cols = [col for col in df.columns if col not in ["future_return", "volatility", "Stock", "Date"]]
    X = df[feature_cols]
    y_return = df["future_return"]
    y_vol = df["volatility"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.2, random_state=42)

    # Train return prediction model
    model_return = RandomForestRegressor(n_estimators=100, random_state=42)
    model_return.fit(X_train, y_train)

    preds = model_return.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    logging.info(f"Return prediction model RMSE: {rmse:.4f}")

    # Train volatility prediction model
    model_vol = RandomForestRegressor(n_estimators=100, random_state=42)
    model_vol.fit(X_train, y_vol)

    # Save both models
    joblib.dump(model_return, os.path.join(MODEL_DIR, "model_return.pkl"))
    joblib.dump(model_vol, os.path.join(MODEL_DIR, "model_vol.pkl"))

    logging.info("Models trained and saved successfully.")
