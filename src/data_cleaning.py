"""
data_cleaning.py

This script handles the cleaning and validation of the raw stock dataset.
It checks for missing data, identifies gaps, trims invalid sections, and
prepares a reliable dataset for downstream analysis.

Pipeline Steps:
1. Load the raw dataset
2. Identify missing stretches and analyze data gaps
3. Trim each stock’s data to start from its first valid entry
4. Save the cleaned dataset to 'data/cleaned/'
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")

# ======================================================================
# Configuration and Logging
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = os.path.join(os.getcwd(), "data", "raw", "all_stocks_data_with_sector.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned", "cleaned_data.parquet")

# ======================================================================
# Core Cleaning Functions
# ======================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the raw parquet dataset."""
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    logging.info(f"Loaded data with shape: {df.shape}")
    return df


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing flags and summarize missing periods."""
    df["is_missing"] = df["Close"].isna().astype(int)

    def _find_stretches(group):
        group["gap_id"] = (group["is_missing"].ne(group["is_missing"].shift())).cumsum()
        missing = group[group["is_missing"] == 1]
        return (
            missing.groupby("gap_id")
            .agg(
                Stock=("Stock", "first"),
                start_date=("Date", "min"),
                end_date=("Date", "max"),
                missing_days=("Date", "count"),
            )
            .reset_index(drop=True)
        )

    missing_summary = df.groupby("Stock", group_keys=False).apply(_find_stretches)
    logging.info(f"Detected {len(missing_summary)} missing stretches.")
    logging.info(f"Example missing stretches:\n{missing_summary.head(5)}")
    return df


def trim_invalid_starts(df: pd.DataFrame) -> pd.DataFrame:
    """Trim each stock’s data from its first valid 'Close' value."""
    df = df.sort_values(["Stock", "Date"])

    def _trim(group):
        idx = group["Close"].first_valid_index()
        return group.loc[idx:] if idx is not None else group

    trimmed = df.groupby("Stock", group_keys=False).apply(_trim)
    trimmed = trimmed.drop(columns=["is_missing"], errors="ignore")
    logging.info(f"Trimmed dataset shape: {trimmed.shape}")
    return trimmed


def save_data(df: pd.DataFrame, path: str):
    """Save cleaned dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Cleaned data saved at: {path}")

# ======================================================================
# Main Pipeline
# ======================================================================

def main():
    """Run the full data cleaning pipeline."""
    try:
        df = load_data(INPUT_PATH)
        df = analyze_missing_data(df)
        cleaned = trim_invalid_starts(df)
        save_data(cleaned, OUTPUT_PATH)
        logging.info("Data cleaning completed successfully.")
    except Exception as e:
        logging.exception("Data cleaning failed.")


if __name__ == "__main__":
    main()
