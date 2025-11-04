"""
feature_engineering.py

Generates advanced time-series features for stock-level modeling.
Includes rolling statistics, momentum, volatility, lagged features,
and sector-level contextual signals.

Modes:
- Training: Uses full cleaned dataset
- Backtesting: Uses only a specified date range

Outputs:
- data/featured_data/featured_data.parquet  (for training)
- data/backtest/featured_data/featured_data.parquet  (for backtesting)
"""

# ===========================================================
# Imports
# ===========================================================
import os
import logging
import numpy as np
import pandas as pd

# ===========================================================
# Configuration
# ===========================================================
FEATURED_DIR = os.path.join(os.getcwd(), "data", "featured_data")
BACKTEST_DIR = os.path.join(os.getcwd(), "data", "backtest", "featured_data")
os.makedirs(FEATURED_DIR, exist_ok=True)
os.makedirs(BACKTEST_DIR, exist_ok=True)

FEATURED_PATH = os.path.join(FEATURED_DIR, "featured_data.parquet")
BACKTEST_FEATURED_PATH = os.path.join(BACKTEST_DIR, "featured_data.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===========================================================
# Feature Engineering Functions
# ===========================================================

def compute_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()
    df["Log_Return"] = np.log1p(df["Daily_Return"])
    return df


def add_rolling_features(df: pd.DataFrame, long_window: int = 750) -> pd.DataFrame:
    df = df.sort_values(["Stock", "Date"])

    def _calc_features(group):
        def safe_minp(window):
            return min(5, window)  # ensure min_periods <= window

        group["MA_20"] = group["Close"].rolling(window=min(20, long_window), min_periods=safe_minp(min(20, long_window))).mean()
        group["MA_50"] = group["Close"].rolling(window=min(50, long_window), min_periods=safe_minp(min(50, long_window))).mean()
        group["MA_200"] = group["Close"].rolling(window=min(200, long_window), min_periods=safe_minp(min(200, long_window))).mean()

        group["Volatility_20d"] = group["Daily_Return"].rolling(window=min(20, long_window), min_periods=safe_minp(min(20, long_window))).std()
        group["Volatility_60d"] = group["Daily_Return"].rolling(window=min(60, long_window), min_periods=safe_minp(min(60, long_window))).std()
        group["Volatility_250d"] = group["Daily_Return"].rolling(window=min(250, long_window), min_periods=safe_minp(min(250, long_window))).std()

        group["Momentum_20d"] = group["Close"].pct_change(periods=min(20, long_window))
        group["Momentum_60d"] = group["Close"].pct_change(periods=min(60, long_window))
        group["Momentum_250d"] = group["Close"].pct_change(periods=min(250, long_window))

        group["Rolling_Max"] = group["Close"].rolling(window=min(long_window, 250), min_periods=safe_minp(min(long_window, 250))).max()
        group["Rolling_Min"] = group["Close"].rolling(window=min(long_window, 250), min_periods=safe_minp(min(long_window, 250))).min()
        return group

    df = df.groupby("Stock", group_keys=False).apply(_calc_features)
    logging.info(f"Rolling window features created successfully (window={long_window}).")
    return df


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    sector_agg = (
        df.groupby(["Date", "Sector"])
        .agg(Sector_Mean_Return=("Daily_Return", "mean"),
             Sector_Volatility=("Daily_Return", "std"))
        .reset_index()
    )
    df = df.merge(sector_agg, on=["Date", "Sector"], how="left")
    logging.info("Sector-level contextual features added successfully.")
    return df


def add_lag_features(df: pd.DataFrame, lags=[1, 2, 3, 5, 10]) -> pd.DataFrame:
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df.groupby("Stock")["Daily_Return"].shift(lag)
    logging.info(f"Lag features ({len(lags)} lags) added successfully.")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Next_Return"] = df.groupby("Stock")["Daily_Return"].shift(-1)
    df = df.dropna(subset=["Next_Return"])
    logging.info("Target variable 'Next_Return' created.")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# ===========================================================
# Main Function
# ===========================================================
def feature_engineering(start_date: str = None, end_date: str = None):
    """
    Run complete feature engineering pipeline.
    If start_date & end_date provided, run in backtest mode.
    """
    try:
        if start_date and end_date:
            input_path = os.path.join(os.getcwd(), "data", "backtest", "cleaned_data", "cleaned_data.parquet")
            backtest = True
        else:
            input_path = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
            backtest = False

        logging.info(f"Loading cleaned dataset → {input_path}")
        df = pd.read_parquet(input_path)
        logging.info(f"Loaded data shape: {df.shape}")

        if backtest:
            logging.info(f"Running in BACKTEST mode: {start_date} → {end_date}")
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            long_window = 15  # smaller rolling window for short periods
            lags = [1, 2]
            save_path = BACKTEST_FEATURED_PATH
        else:
            logging.info("Running in TRAINING mode (full dataset).")
            long_window = 750
            lags = [1, 2, 3, 5, 10]
            save_path = FEATURED_PATH

        if df.empty:
            logging.warning("⚠️ No data available after applying date filter.")
            return pd.DataFrame()

        # --- Feature Pipeline ---
        df = compute_basic_returns(df)
        df = add_rolling_features(df, long_window=long_window)
        df = add_sector_features(df)
        df = add_lag_features(df, lags=lags)
        df = create_target(df)
        df = reduce_memory(df)

        df = df.dropna(subset=["Daily_Return", "Close", "Sector", "Stock"])
        df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

        if df.empty:
            logging.warning("⚠️ Feature engineering produced an empty dataset. Check rolling/lags.")
            return pd.DataFrame()

        df.to_parquet(save_path, index=False)
        logging.info(f"✅ Feature engineering completed successfully → {save_path}")
        logging.info(f"Final dataset shape: {df.shape}")

        return df

    except Exception as e:
        logging.exception(f"Feature engineering failed: {e}")
        raise
