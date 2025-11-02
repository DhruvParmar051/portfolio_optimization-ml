"""
data_cleaning.py

Cleans raw stock data — removes invalid starts, handles missing data, and
saves a consistent dataset ready for feature engineering.
"""

import os, pandas as pd, logging, numpy as np, warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "raw_data", "all_stocks_data_with_sector.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")


def load_data(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    logging.info(f"Loaded raw data: {df.shape}")
    return df


def analyze_missing(df):
    df["is_missing"] = df["Close"].isna().astype(int)
    missing = df.groupby("Stock")["is_missing"].sum().reset_index()
    logging.info(f"Missing data summary:\n{missing.head(5)}")
    return df


def trim_invalid(df):
    df = df.sort_values(["Stock", "Date"])
    def _trim(g):
        idx = g["Close"].first_valid_index()
        return g.loc[idx:] if idx else g
    trimmed = df.groupby("Stock", group_keys=False).apply(_trim)
    logging.info(f"Trimmed data: {trimmed.shape}")
    return trimmed.drop(columns=["is_missing"], errors="ignore")


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Cleaned data saved → {path}")


def data_cleaning():
    df = load_data(INPUT_PATH)
    df = analyze_missing(df)
    df = trim_invalid(df)
    save(df, OUTPUT_PATH)
    logging.info("Cleaning complete.")
