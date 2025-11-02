"""
train_valid_split.py

Splits preprocessed S&P 500 dataset into unified chronological
train/validation sets for model training.
"""

import os, pandas as pd, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_valid_split(valid_ratio=0.2):
    df = pd.read_parquet(INPUT_PATH).sort_values("Date")
    feature_cols = [c for c in df.columns if c not in ["Date", "Close"]]
    n = len(df)
    n_valid = int(n * valid_ratio)
    X_train, X_val = df.iloc[:n-n_valid][feature_cols], df.iloc[n-n_valid:][feature_cols]
    y_train, y_val = df.iloc[:n-n_valid]["Close"], df.iloc[n-n_valid:]["Close"]

    X_train.to_parquet(os.path.join(OUTPUT_DIR, "X_train.parquet"), index=False)
    y_train.to_frame("target").to_parquet(os.path.join(OUTPUT_DIR, "y_train.parquet"), index=False)
    X_val.to_parquet(os.path.join(OUTPUT_DIR, "X_val.parquet"), index=False)
    y_val.to_frame("target").to_parquet(os.path.join(OUTPUT_DIR, "y_val.parquet"), index=False)

    logging.info(f"Train={len(X_train)}, Val={len(X_val)} saved to {OUTPUT_DIR}")
