"""
train_valid_split.py

Performs a chronological train–validation split for preprocessed S&P 500 data.

Steps:
1. Load preprocessed dataset
2. Sort data chronologically
3. Split based on `valid_ratio`
4. Save train and validation sets as parquet files
"""

import os
import pandas as pd
import logging

# ======================================================================
# Logging Configuration
# ======================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

<<<<<<< HEAD
# ======================================================================
# File Paths
# ======================================================================
INPUT_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")
=======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
>>>>>>> c97fbfd (Kuch to kiya hai)
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def chronological_split(df: pd.DataFrame, valid_ratio: float = 0.2):
    """Split dataset chronologically into train and validation sets."""
    if "Date" not in df.columns:
        raise ValueError("The dataset must contain a 'Date' column for time-based split.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cutoff_idx = int(len(df) * (1 - valid_ratio))
    train_df = df.iloc[:cutoff_idx].reset_index(drop=True)
    valid_df = df.iloc[cutoff_idx:].reset_index(drop=True)
    logging.info(f"Split completed: Train={len(train_df)}, Valid={len(valid_df)}")
    return train_df, valid_df


def save_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """Save the train and validation splits to parquet files."""
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    valid_path = os.path.join(OUTPUT_DIR, "valid.parquet")
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    logging.info(f"Train/Validation splits saved → {OUTPUT_DIR}")


# ======================================================================
# Entry Point
# ======================================================================
def train_valid_split(valid_ratio: float = 0.2):
    """Execute full train-validation split pipeline."""
    try:
        df = load_preprocessed_data(INPUT_PATH)
        train_df, valid_df = chronological_split(df, valid_ratio)
        save_splits(train_df, valid_df)
        logging.info("Train-validation split pipeline completed successfully.")
    except Exception as e:
        logging.exception("Train-validation split pipeline failed.")
