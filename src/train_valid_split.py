"""
train_valid_split.py

This script performs time-aware train–validation splitting for multi-stock data.
It ensures chronological consistency (no shuffling) and prevents data leakage
for time-series model training.

Pipeline Steps:
1. Load the cleaned data
2. Split each stock chronologically into train and validation sets
3. Save the splits into 'data/splits/<stock>/'
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import logging
from typing import List, Dict

# ======================================================================
# Configuration and Logging
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned", "cleaned_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "splits")

# ======================================================================
# Core Split Helper
# ======================================================================

class TimeSeriesSplitHelper:
    """Utility class to perform per-stock time-based splits."""

    def __init__(self, date_col="Date", stock_col="Stock", target_col="Close"):
        self.date_col = date_col
        self.stock_col = stock_col
        self.target_col = target_col

    def _sort_data(self, df):
        return df.sort_values(by=[self.stock_col, self.date_col]).reset_index(drop=True)

    def split(self, df, feature_cols: List[str], valid_ratio: float = 0.2) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Split data chronologically into train and validation sets per stock."""
        df = self._sort_data(df)
        splits = {}

        for stock, group in df.groupby(self.stock_col):
            n = len(group)
            n_valid = int(n * valid_ratio)
            n_train = n - n_valid

            if n_train < 1 or n_valid < 1:
                logging.warning(f"Skipping {stock}: not enough data points.")
                continue

            train, valid = group.iloc[:n_train], group.iloc[n_train:]

            splits[stock] = {
                "X_train": train[feature_cols],
                "y_train": train[self.target_col],
                "X_val": valid[feature_cols],
                "y_val": valid[self.target_col],
            }

            logging.info(f"{stock}: Train={n_train}, Validation={n_valid}")

        return splits

# ======================================================================
# Helper Functions
# ======================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load cleaned dataset for splitting."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    logging.info(f"Loaded data: {df.shape}")
    return df


def save_splits(splits: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """Save train/validation splits for each stock."""
    os.makedirs(output_dir, exist_ok=True)

    for stock, parts in splits.items():
        stock_dir = os.path.join(output_dir, stock)
        os.makedirs(stock_dir, exist_ok=True)

        parts["X_train"].to_parquet(os.path.join(stock_dir, "X_train.parquet"), index=False)
        parts["y_train"].to_frame("target").to_parquet(os.path.join(stock_dir, "y_train.parquet"), index=False)
        parts["X_val"].to_parquet(os.path.join(stock_dir, "X_val.parquet"), index=False)
        parts["y_val"].to_frame("target").to_parquet(os.path.join(stock_dir, "y_val.parquet"), index=False)

        logging.info(f"Saved splits for {stock} at {stock_dir}")

# ======================================================================
# Main Pipeline
# ======================================================================

def train_valid_split():
    """Run the time-series split pipeline."""
    try:
        df = load_data(INPUT_PATH)
        splitter = TimeSeriesSplitHelper()
        features = ["Open", "High", "Low", "Volume", "Adj Close"]
        splits = splitter.split(df, feature_cols=features, valid_ratio=0.2)
        save_splits(splits, OUTPUT_DIR)
        logging.info("Train–validation splitting completed successfully.")
    except Exception as e:
        logging.exception("Splitting pipeline failed.")

