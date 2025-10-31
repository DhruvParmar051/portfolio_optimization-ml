"""
feature_engineering.py

This script creates additional features from cleaned stock data.
It computes daily returns, volatility, moving averages, momentum,
and aggregates sector-level information.

Pipeline Steps:
1. Load cleaned dataset
2. Generate stock-level and sector-level features
3. Merge and save engineered data to 'data/feature_engineering/'
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import numpy as np
import logging

# ======================================================================
# Configuration and Logging
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "engineered_data.parquet")

# ======================================================================
# Feature Engineering Functions
# ======================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load cleaned stock data."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    logging.info(f"Loaded cleaned data: {df.shape}")
    return df


def create_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-series-based stock-level features."""
    df = df.sort_values(["Stock", "Date"])

    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()
    df["Volatility_7d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(7).std())
    df["Volatility_30d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(30).std())
    df["MA_20"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(50).mean())
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x / x.shift(10) - 1)
    df["Log_Volume"] = np.log1p(df["Volume"])

    logging.info("Stock-level features generated successfully.")
    return df


def create_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregated sector-level metrics."""
    sector_df = (
        df.groupby(["Date", "Sector"])
        .agg({"Close": "mean", "Volume": "mean", "Daily_Return": "mean"})
        .reset_index()
    )

    sector_df["Sector_Return_7d"] = sector_df.groupby("Sector")["Daily_Return"].transform(lambda x: x.rolling(7).mean())
    logging.info("Sector-level features generated successfully.")
    return sector_df


def merge_sector_features(df, sector_df):
    """Merge sector features back to stock data."""
    return df.merge(sector_df[["Date", "Sector", "Sector_Return_7d"]], on=["Date", "Sector"], how="left")


def save_data(df, path):
    """Save the feature-engineered dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Feature-engineered data saved at: {path}")

# ======================================================================
# Main Pipeline
# ======================================================================

def feature_engineering():
    """Run the complete feature engineering pipeline."""
    try:
        df = load_data(INPUT_PATH)
        df = create_stock_features(df)
        sector_df = create_sector_features(df)
        final_df = merge_sector_features(df, sector_df)
        save_data(final_df, OUTPUT_PATH)
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.exception("Feature engineering pipeline failed.")

