"""
data_cleaning.py

Cleans raw S&P 500 stock data — validates schema, removes invalid or incomplete
series, handles missing data gracefully, and outputs a reliable dataset for
feature engineering and modeling.

Steps:
1. Load raw dataset with sector info
2. Validate structure (columns, datatypes)
3. Trim each stock from its first valid 'Close' price
4. Remove duplicate or corrupt rows
5. Save cleaned dataset to parquet file

"""

# ===========================================================
# Imports
# ===========================================================

import os
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ===========================================================
# Configuration
# ===========================================================

RAW_PATH = os.path.join(os.getcwd(), "data", "raw_data", "all_stocks_data_with_sector.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "cleaned_data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===========================================================
# Helper Functions
# ===========================================================

def validate_schema(df: pd.DataFrame):
    """Ensure dataset has essential columns."""
    required_cols = {"Date", "Close", "Stock", "Sector", "Industry"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def trim_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock, trim records to start from first valid 'Close' price.
    Removes NaNs before first valid value.
    """
    df = df.sort_values(["Stock", "Date"])
    cleaned = []
    for stock, group in df.groupby("Stock"):
        first_valid = group["Close"].first_valid_index()
        if first_valid is None:
            logging.warning(f"{stock}: No valid Close values — skipped.")
            continue
        group = group.loc[first_valid:]
        cleaned.append(group)
    trimmed = pd.concat(cleaned, ignore_index=True)
    logging.info(f"Trimmed data for {len(trimmed['Stock'].unique())} stocks.")
    return trimmed


def handle_missing_data(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Drop stocks with excessive missing Close values (>threshold fraction).
    """
    miss_ratio = df.groupby("Stock")["Close"].apply(lambda x: x.isna().mean())
    to_drop = miss_ratio[miss_ratio > threshold].index
    if len(to_drop) > 0:
        logging.warning(f"Dropping {len(to_drop)} stocks with >{threshold*100:.0f}% missing data.")
    df = df[~df["Stock"].isin(to_drop)]
    df = df.dropna(subset=["Close"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (Stock, Date) pairs."""
    before = len(df)
    df = df.drop_duplicates(subset=["Stock", "Date"])
    after = len(df)
    if before != after:
        logging.info(f"Removed {before - after} duplicate rows.")
    return df


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure proper data types and add metadata fields."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

# ===========================================================
# Main Cleaning Pipeline
# ===========================================================

def data_cleaning():
    """Run complete data cleaning pipeline."""
    try:
        logging.info("Loading raw dataset...")
        df = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded raw data: {df.shape}")

        df = validate_schema(df)
        df = trim_stock_data(df)
        df = handle_missing_data(df)
        df = remove_duplicates(df)
        df = enrich_metadata(df)

        df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

        df.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Cleaned dataset saved → {OUTPUT_PATH}")
        logging.info(f"Final shape: {df.shape}")

    except Exception as e:
        logging.exception(f"Data cleaning failed: {e}")
