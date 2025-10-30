"""
train_valid_split.py

This module performs time-aware train–validation splitting for multi-stock datasets.
It ensures chronological integrity (no shuffling) and prevents data leakage when
training and validating models on financial time series data.

Pipeline Steps:
1. Load cleaned data from parquet file
2. Sort data by stock and date
3. Split each stock’s data into train and validation sets
4. Save split datasets (optional)

Author: Mahak
Date: 2025-10-30
"""

import os
import pandas as pd
import logging
from typing import List, Dict

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned", "cleaned_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "splits")

# ----------------------------------------------------------------------
# Core Splitter Class
# ----------------------------------------------------------------------
class TimeSeriesSplitHelper:
    """
    Utility class for performing time-based train–validation splits per stock.
    Ensures that validation data is strictly posterior to training data.
    """

    def __init__(self, date_col: str = 'Date', stock_col: str = 'Stock', target_col: str = 'Close'):
        """
        Initialize the splitter with relevant column names.

        Parameters
        ----------
        date_col : str
            Name of the datetime column.
        stock_col : str
            Name of the stock identifier column.
        target_col : str
            Name of the target variable column.
        """
        self.date_col = date_col
        self.stock_col = stock_col
        self.target_col = target_col

    # ------------------------------------------------------------------
    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data by stock and date."""
        return df.sort_values(by=[self.stock_col, self.date_col]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def split(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        valid_ratio: float = 0.2
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Perform time-based train–validation split per stock.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset containing features, target, stock, and date columns.
        feature_cols : list of str
            Feature column names for training and validation.
        valid_ratio : float, optional
            Fraction of each stock’s data to reserve for validation (default = 0.2).

        Returns
        -------
        dict
            Dictionary with structure:
            {
                'AAPL': {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ...},
                'MSFT': {...},
                ...
            }
        """
        df = self._sort_data(df)
        split_data = {}

        for stock, stock_df in df.groupby(self.stock_col):
            n_total = len(stock_df)
            n_valid = int(n_total * valid_ratio)
            n_train = n_total - n_valid

            if n_train <= 0 or n_valid <= 0:
                logging.warning(f"Skipping {stock} (insufficient data).")
                continue

            train_df = stock_df.iloc[:n_train]
            valid_df = stock_df.iloc[n_train:]

            X_train = train_df[feature_cols]
            y_train = train_df[self.target_col]
            X_val = valid_df[feature_cols]
            y_val = valid_df[self.target_col]

            split_data[stock] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val
            }

            logging.info(f"✅ {stock}: Train={n_train}, Validation={n_valid}, Total={n_total}")

        return split_data

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load cleaned dataset for splitting.

    Parameters
    ----------
    path : str
        Path to the cleaned parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Input file not found at {path}")

    df = pd.read_parquet(path)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def save_splits(splits: Dict[str, Dict[str, pd.DataFrame]], output_dir: str) -> None:
    """
    Save train and validation splits per stock into parquet files.

    Parameters
    ----------
    splits : dict
        Output from TimeSeriesSplitHelper.split()
    output_dir : str
        Target directory to store the split files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for stock, data in splits.items():
        stock_dir = os.path.join(output_dir, stock)
        os.makedirs(stock_dir, exist_ok=True)

        data["X_train"].to_parquet(os.path.join(stock_dir, "X_train.parquet"), index=False)
        data["y_train"].to_frame("target").to_parquet(os.path.join(stock_dir, "y_train.parquet"), index=False)
        data["X_val"].to_parquet(os.path.join(stock_dir, "X_val.parquet"), index=False)
        data["y_val"].to_frame("target").to_parquet(os.path.join(stock_dir, "y_val.parquet"), index=False)

        logging.info(f"Saved splits for {stock} in {stock_dir}")

# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------
def main():
    """
    Execute the time-based train–validation splitting pipeline:
      1. Load cleaned data
      2. Split each stock chronologically
      3. Save train and validation splits
    """
    try:
        # Step 1: Load cleaned data
        df = load_data(INPUT_PATH)

        # Step 2: Initialize splitter
        splitter = TimeSeriesSplitHelper(date_col='Date', stock_col='Stock', target_col='Close')

        # Define features
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Adj Close']

        # Step 3: Perform split
        splits = splitter.split(df, feature_cols=feature_cols, valid_ratio=0.2)

        # Step 4: Save outputs
        save_splits(splits, OUTPUT_DIR)

        logging.info("Train–validation split pipeline completed successfully.")

    except Exception as e:
        logging.exception("Pipeline failed due to an error.")

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
