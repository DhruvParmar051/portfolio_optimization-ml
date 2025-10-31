"""
model_training.py

This module trains machine learning models to predict
expected stock returns and risk (volatility) using
the preprocessed dataset. These models are later used
by the portfolio optimizer to recommend optimal portfolios.

Steps:
1. Load preprocessed data
2. Train ML model for return prediction
3. Train ML model for risk estimation
4. Save both models for use in portfolio_optimizer.py
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ======================================================================
# Configuration
# ======================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "processed", "final_features.parquet")
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================================
# Model Training Functions
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
