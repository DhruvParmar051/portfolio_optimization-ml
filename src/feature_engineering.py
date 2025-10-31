"""
Feature Engineering Pipeline for Multi-Stock Time-Series Data
-------------------------------------------------------------

This script performs feature generation on cleaned stock-market data.

Steps:
1. Load cleaned parquet data (must include ['Date', 'Stock', 'Sector', 'Close', 'Volume'])
2. Compute daily returns, volatility, rolling averages, momentum, and sector-level aggregates
3. Save the engineered dataset for modeling

Output: '../data/engineered_features.parquet'
"""

import os
import pandas as pd
import numpy as np
import logging

# --------------------------------------------------------------------
# Configuration and Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = os.path.join(os.getcwd(),'data','cleaned', 'cleaned_data.parquet')
OUTPUT_PATH = os.path.join(os.getcwd(),'data','feature_engineering', 'engineered_data.parquet')


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load cleaned stock data from parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logging.info(f"Loaded cleaned data: {df.shape}")
    return df


def generate_stock_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-series features per stock:
    - Daily returns
    - Rolling volatility (7, 30 days)
    - Moving averages (20, 50 days)
    - Momentum (price ratio)
    """

    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # Calculate daily returns per stock
    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()

    # Rolling volatility (standard deviation)
    df["Volatility_7d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(7).std())
    df["Volatility_30d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(30).std())

    # Moving averages for smoothing trends
    df["MA_20"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(50).mean())

    # Momentum indicator (current price vs. 10 days ago)
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x / x.shift(10) - 1)

    # Volume-based feature (log transform to stabilize scale)
    df["Log_Volume"] = np.log1p(df["Volume"])

    return df


def generate_sector_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features at sector level:
    - Mean close price, volume, and return per day per sector
    - Rolling 7-day sector return for trend smoothing
    """
    sector_df = (
        df.groupby(["Date", "Sector"])
        .agg({
            "Close": "mean",
            "Volume": "mean",
            "Daily_Return": "mean"
        })
        .reset_index()
    )

    # Rolling 7-day mean return to capture short-term sector momentum
    sector_df["Sector_Return_7d"] = (
        sector_df.groupby("Sector")["Daily_Return"].transform(lambda x: x.rolling(7).mean())
    )

    logging.info(f"Generated sector-level data: {sector_df.shape}")
    return sector_df


def merge_sector_features(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sector-level metrics back into stock-level data."""
    merged = df.merge(
        sector_df[["Date", "Sector", "Sector_Return_7d"]],
        on=["Date", "Sector"],
        how="left"
    )
    return merged


def save_engineered_data(df: pd.DataFrame, path: str) -> None:
    """Save final dataset to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Feature-engineered dataset saved at: {path}")


# --------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------

def main():
    """Main execution for feature engineering."""
    logging.info("Starting Feature Engineering Pipeline...")

    # Step 1: Load data
    df = load_cleaned_data(INPUT_PATH)

    # Step 2: Generate stock-level features
    logging.info("Generating stock-level features...")
    df = generate_stock_level_features(df)

    # Step 3: Generate sector-level features
    logging.info("Generating sector-level features...")
    sector_df = generate_sector_level_features(df)

    # Step 4: Merge sector features back
    logging.info("Merging sector-level features...")
    df_final = merge_sector_features(df, sector_df)

    # Step 5: Save final dataset
    save_engineered_data(df_final, OUTPUT_PATH)

    logging.info("Feature Engineering completed successfully.")


if __name__ == "__main__":
    main()
