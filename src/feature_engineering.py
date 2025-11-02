"""
feature_engineering.py

Generates advanced time-series features for stock-level modeling.
Includes rolling statistics, momentum, volatility, lagged features,
and sector-level contextual signals.

Steps:
1. Load cleaned dataset
2. Compute rolling and lag features per stock (750-day context ready)
3. Add sector-level features (average sector return, volatility)
4. Create prediction target (Next_Return)
5. Save to featured_data/featured_data.parquet
"""

# ===========================================================
# Imports
# ===========================================================

import os
import logging
import numpy as np
import pandas as pd

# ===========================================================
# Configuration
# ===========================================================

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "featured_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "featured_data.parquet")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===========================================================
# Feature Engineering Functions
# ===========================================================

def compute_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log and percentage returns."""
    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()
    df["Log_Return"] = np.log1p(df["Daily_Return"])
    return df


def add_rolling_features(df: pd.DataFrame, long_window: int = 750) -> pd.DataFrame:
    """
    Compute rolling-window features up to 750 days.
    Includes moving averages, volatility, and rolling momentum.
    """
    df = df.sort_values(["Stock", "Date"])

    def _calc_features(group):
        group["MA_20"] = group["Close"].rolling(window=20, min_periods=5).mean()
        group["MA_50"] = group["Close"].rolling(window=50, min_periods=10).mean()
        group["MA_200"] = group["Close"].rolling(window=200, min_periods=20).mean()

        group["Volatility_20d"] = group["Daily_Return"].rolling(window=20, min_periods=5).std()
        group["Volatility_60d"] = group["Daily_Return"].rolling(window=60, min_periods=10).std()
        group["Volatility_250d"] = group["Daily_Return"].rolling(window=250, min_periods=20).std()

        group["Momentum_20d"] = group["Close"].pct_change(periods=20)
        group["Momentum_60d"] = group["Close"].pct_change(periods=60)
        group["Momentum_250d"] = group["Close"].pct_change(periods=250)

        group["Rolling_Max"] = group["Close"].rolling(window=long_window, min_periods=30).max()
        group["Rolling_Min"] = group["Close"].rolling(window=long_window, min_periods=30).min()

        return group

    df = df.groupby("Stock", group_keys=False).apply(_calc_features)
    logging.info("Rolling window (up to 750 days) features created successfully.")
    return df


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sector-level metrics (mean return, volatility) for context."""
    sector_agg = (
        df.groupby(["Date", "Sector"])
        .agg(Sector_Mean_Return=("Daily_Return", "mean"),
             Sector_Volatility=("Daily_Return", "std"))
        .reset_index()
    )
    df = df.merge(sector_agg, on=["Date", "Sector"], how="left")
    logging.info("Sector-level contextual features added successfully.")
    return df


def add_lag_features(df: pd.DataFrame, lags=[1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Add lagged versions of returns for autoregressive modeling."""
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df.groupby("Stock")["Daily_Return"].shift(lag)
    logging.info(f"Lag features ({len(lags)} lags) added successfully.")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Define target as next-day percent return."""
    df["Next_Return"] = df.groupby("Stock")["Daily_Return"].shift(-1)
    df = df.dropna(subset=["Next_Return"])
    logging.info("Target variable 'Next_Return' created.")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# ===========================================================
# Main Feature Engineering Pipeline
# ===========================================================

def feature_engineering():
    """Run complete feature engineering pipeline."""
    try:
        logging.info("Loading cleaned dataset...")
        df = pd.read_parquet(INPUT_PATH)
        logging.info(f"Loaded data shape: {df.shape}")

        df = compute_basic_returns(df)
        df = add_rolling_features(df, long_window=750)
        df = add_sector_features(df)
        df = add_lag_features(df)
        df = create_target(df)
        df = reduce_memory(df)

        df = df.dropna(subset=["Daily_Return", "Close", "Sector", "Stock"])
        df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

        df.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Feature engineering completed successfully → {OUTPUT_PATH}")
        logging.info(f"Final dataset shape: {df.shape}")

    except Exception as e:
        logging.exception(f"Feature engineering failed: {e}")

