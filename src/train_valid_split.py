"""
train_valid_split.py

Performs chronological train–validation split for preprocessed S&P 500 data.
Now supports backtesting mode via a `backtest` flag.

Modes:
- Training mode (default): Ratio-based chronological split
- Backtest mode: Date-based split for testing model performance on future data

Steps:
1. Load preprocessed dataset
2. Sort data chronologically
3. Split either by ratio (train/valid) or date range (train/backtest)
4. Save resulting parquet files
"""

import os
import pandas as pd
import logging
from typing import Optional, Tuple

# ======================================================================
# Logging Configuration
# ======================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================================================================
# File Paths
# ======================================================================
INPUT_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================
# Configuration for Backtesting
# ======================================================================
BACKTEST_START_DATE = "2025-01-01"
BACKTEST_END_DATE = "2025-03-31"

# ======================================================================
# Core Functions
# ======================================================================
def load_preprocessed_data(path: str) -> pd.DataFrame:
    """Load preprocessed data for splitting."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed file not found: {path}")
    df = pd.read_parquet(path)
    logging.info(f"Loaded preprocessed data: shape={df.shape}")
    return df


def chronological_split(
    df: pd.DataFrame,
    valid_ratio: Optional[float] = 0.2,
    backtest: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically.

    Args:
        df (pd.DataFrame): Preprocessed data.
        valid_ratio (float, optional): Ratio for validation split (if backtest=False).
        backtest (bool): Whether to run date-based backtest split.
        start_date (str, optional): Override for backtest start date.
        end_date (str, optional): Override for backtest end date.

    Returns:
        (train_df, valid_df)
    """
    if "Date" not in df.columns:
        raise ValueError("The dataset must contain a 'Date' column for time-based split.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if backtest:
        # --- Date-based split for backtesting ---
        start_date = start_date or BACKTEST_START_DATE
        end_date = end_date or BACKTEST_END_DATE

        mask_valid = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        valid_df = df.loc[mask_valid].reset_index(drop=True)
        train_df = df.loc[df["Date"] < start_date].reset_index(drop=True)

        logging.info(
            f"Backtest split: Train={len(train_df)}, Test={len(valid_df)} "
            f"({start_date} → {end_date})"
        )
    else:
        # --- Ratio-based split for training ---
        cutoff_idx = int(len(df) * (1 - valid_ratio))
        train_df = df.iloc[:cutoff_idx].reset_index(drop=True)
        valid_df = df.iloc[cutoff_idx:].reset_index(drop=True)
        logging.info(f"Train/Valid split: Train={len(train_df)}, Valid={len(valid_df)}")

    return train_df, valid_df


def save_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame, backtest: bool = False):
    """Save train and validation/backtest splits to parquet files."""
    suffix = "_backtest" if backtest else ""
    train_path = os.path.join(OUTPUT_DIR, f"train{suffix}.parquet")
    valid_path = os.path.join(OUTPUT_DIR, f"valid{suffix}.parquet")

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    logging.info(f"Saved {'backtest' if backtest else 'train/valid'} splits → {OUTPUT_DIR}")
    logging.info(f"Train path: {train_path}")
    logging.info(f"Validation path: {valid_path}")

    return train_path, valid_path


# ======================================================================
# Entry Point
# ======================================================================
def train_valid_split(valid_ratio: float = 0.2, backtest: bool = False):
    """
    Execute full train-validation or backtest split pipeline.

    Args:
        valid_ratio (float): Ratio for validation split (used only when backtest=False)
        backtest (bool): If True, use fixed date range for backtest split
    """
    try:
        df = load_preprocessed_data(INPUT_PATH)
        train_df, valid_df = chronological_split(df, valid_ratio=valid_ratio, backtest=backtest)
        paths = save_splits(train_df, valid_df, backtest=backtest)
        logging.info("✅ Split pipeline completed successfully.")
        return train_df, valid_df, paths
    except Exception as e:
        logging.exception("❌ Train-validation/backtest split pipeline failed.")
        raise e

