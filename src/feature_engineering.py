"""
feature_engineering.py

Creates return, volatility, moving average, and momentum features
for each stock and sector.
"""

import os, pandas as pd, numpy as np, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")


def create_stock_features(df):
    df = df.sort_values(["Stock", "Date"])
    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()
    df["Volatility_7d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(7).std())
    df["Volatility_30d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(30).std())
    df["MA_20"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(50).mean())
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x / x.shift(10) - 1)
    df["Log_Volume"] = np.log1p(df["Volume"])
    logging.info("Stock-level features done.")
    return df


def create_sector_features(df):
    sector = df.groupby(["Date", "Sector"]).agg({"Close": "mean", "Daily_Return": "mean"}).reset_index()
    sector["Sector_Return_7d"] = sector.groupby("Sector")["Daily_Return"].transform(lambda x: x.rolling(7).mean())
    logging.info("Sector-level features done.")
    return df.merge(sector[["Date", "Sector", "Sector_Return_7d"]], on=["Date", "Sector"], how="left")


def feature_engineering():
    """Run the complete feature engineering pipeline."""
    try:
        df = load_data(INPUT_PATH)
        df = create_stock_features(df)
        sector_df = create_sector_features(df)
        final_df = merge_sector_features(df, sector_df)
        save_data(final_df, OUTPUT_PATH)
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.exception("Feature engineering pipeline failed.")

