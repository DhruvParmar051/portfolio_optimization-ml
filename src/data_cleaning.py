"""
data_cleaning.py

Cleans raw S&P 500 stock data — validates schema, removes invalid or incomplete
series, handles missing data gracefully, and outputs a reliable dataset for
feature engineering and modeling.

Supports both:
- Full data cleaning  → data/raw_data → data/cleaned_data
- Backtest cleaning   → data/backtest/raw_data → data/backtest/cleaned_data
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
# Logging
# ===========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===========================================================
# Core Helper Functions
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
    Trim each stock from its first valid 'Close' value onward.
    Removes NaNs before first valid record.
    """
    df = df.sort_values(["Stock", "Date"])
    cleaned = []
    for stock, group in df.groupby("Stock", group_keys=False):
        first_valid = group["Close"].first_valid_index()
        if first_valid is None:
            logging.warning(f"{stock}: No valid Close values — skipped.")
            continue
        group = group.loc[first_valid:]
        cleaned.append(group)
    trimmed = pd.concat(cleaned, ignore_index=True)
    logging.info(f"Trimmed to {len(trimmed['Stock'].unique())} valid stocks.")
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
    """Ensure correct dtypes and add metadata fields."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df


# ===========================================================
# Main Cleaning Function
# ===========================================================
def data_cleaning(backtest: bool = False):
    """
    Run complete cleaning pipeline.

    Parameters:
    - backtest: if True → use backtest paths
    """
    # Dynamic paths
    base_dir = os.path.join(os.getcwd(), "data", "backtest" if backtest else "")
    raw_path = os.path.join(base_dir, "raw_data", "backtest_data.parquet" if backtest else "all_stocks_data_with_sector.parquet")
    output_dir = os.path.join(base_dir, "cleaned_data")
    output_path = os.path.join(output_dir, "cleaned_data.parquet")
    os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"Loading raw dataset → {raw_path}")
        df = pd.read_parquet(raw_path)
        logging.info(f"Loaded raw data: shape={df.shape}")

        # Cleaning pipeline
        df = (
            df.pipe(validate_schema)
              .pipe(trim_stock_data)
              .pipe(handle_missing_data)
              .pipe(remove_duplicates)
              .pipe(enrich_metadata)
              .sort_values(["Stock", "Date"])
              .reset_index(drop=True)
        )

        df.to_parquet(output_path, index=False)
        logging.info(f"✅ Cleaned dataset saved → {output_path}")
        logging.info(f"Final shape: {df.shape}")

        return df

    except Exception as e:
        logging.exception(f"❌ Data cleaning failed: {e}")
        raise
