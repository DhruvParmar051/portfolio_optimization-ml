"""
data_fetch.py

Fetches S&P 500 tickers (with sector info) and downloads historical OHLCV data
from Yahoo Finance. It supports both full-data downloads and custom date-range
fetches for backtesting (e.g., 2025-01-01 to 2025-03-31).

Outputs:
- data/raw_data/all_stocks_data_with_sector.parquet       (full dataset)
- data/backtest/raw_data/backtest_data.parquet            (backtest dataset)
"""

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from typing import List, Optional

# ======================================================================
# Configuration
# ======================================================================
RAW_DIR = os.path.join(os.getcwd(), "data", "raw_data")
BACKTEST_RAW_DIR = os.path.join(os.getcwd(), "data", "backtest", "raw_data")
RAW_PATH = os.path.join(RAW_DIR, "all_stocks_data_with_sector.parquet")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2025-01-01"

YF_RETRY_PAUSE = 2.0
YF_MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ======================================================================
# Helper Functions
# ======================================================================
def get_sp500_table() -> pd.DataFrame:
    """Fetch S&P 500 constituents with sectors from Wikipedia."""
    headers = {"User-Agent": "Mozilla/5.0"}
    logging.info("Fetching S&P 500 constituents from Wikipedia...")
    try:
        html = requests.get(WIKI_URL, headers=headers, timeout=15)
        html.raise_for_status()
        tables = pd.read_html(html.text)
        sp500 = tables[0]
        sp500["Symbol"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)
        logging.info("S&P 500 constituents fetched successfully.")
        return sp500
    except Exception:
        logging.exception("Failed to fetch S&P 500 table.")
        raise


def download_yahoo_batch(
    tickers: List[str], start: str = DEFAULT_START, end: str = DEFAULT_END
) -> pd.DataFrame:
    """Download OHLCV data for tickers via yfinance."""
    if not tickers:
        return pd.DataFrame()

    logging.info(f"Downloading Yahoo Finance data ({start} → {end})...")
    attempts = 0
    last_exception = None

    while attempts < YF_MAX_RETRIES:
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                progress=False,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                records = []
                for t in tickers:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t].copy().reset_index()
                    df["Stock"] = t
                    records.append(df)
                data = pd.concat(records, ignore_index=True)
            else:
                df = raw.reset_index()
                df["Stock"] = tickers[0]
                data = df

            logging.info("Yahoo Finance data downloaded successfully.")
            return data
        except Exception as exc:
            attempts += 1
            last_exception = exc
            logging.warning(f"yfinance download failed (attempt {attempts}/{YF_MAX_RETRIES}): {exc}")
            time.sleep(YF_RETRY_PAUSE)

    raise last_exception


def reshape_and_merge(yf_df: pd.DataFrame, sp500: pd.DataFrame) -> pd.DataFrame:
    """Merge Yahoo data with sector info and clean columns."""
    if yf_df.empty:
        raise ValueError("Empty Yahoo Finance DataFrame.")

    yf_df["Date"] = pd.to_datetime(yf_df["Date"], errors="coerce")
    key_map = sp500.rename(
        columns={"Symbol": "Stock", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry"}
    )
    meta = key_map[["Stock", "Sector", "Industry"]].drop_duplicates(subset=["Stock"])
    merged = yf_df.merge(meta, on="Stock", how="left")
    merged = merged.dropna(subset=["Date"]).sort_values(["Stock", "Date"])
    return merged


def save_data(df: pd.DataFrame, path: str):
    """Save DataFrame to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Saved data → {path}")


# ======================================================================
# Main Pipeline
# ======================================================================
def data_fetch(
    tickers_subset: Optional[List[str]] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    backtest: bool = False,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Full Yahoo Finance data fetch pipeline.
    - If backtest=True → saves to data/backtest/raw_data/backtest_data.parquet
    - Else → saves to data/raw_data/all_stocks_data_with_sector.parquet
    """
    save_path = (
        os.path.join(BACKTEST_RAW_DIR, "backtest_data.parquet")
        if backtest
        else RAW_PATH
    )

    if os.path.exists(save_path) and not overwrite:
        logging.info(f"File exists → {save_path}, overwrite=False. Loading existing file.")
        return pd.read_parquet(save_path)

    # Fetch tickers
    sp500 = get_sp500_table()
    tickers = tickers_subset if tickers_subset else sp500["Symbol"].tolist()

    # Download data
    yf_df = download_yahoo_batch(tickers, start=start, end=end)
    merged = reshape_and_merge(yf_df, sp500)

    # Save final merged dataset
    save_data(merged, save_path)
    return merged
