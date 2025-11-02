"""
data_fetch.py

Fetch S&P 500 tickers (with sector info) and download historical OHLCV data
from Yahoo Finance. The script writes a single parquet file containing all
tickers' price data plus sector/industry metadata.
"""

from typing import List
import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf

# Configuration
RAW_DIR = os.path.join(os.getcwd(), "data", "raw_data")
RAW_PATH = os.path.join(RAW_DIR, "all_stocks_data_with_sector.parquet")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YA_START = "2010-01-01"
YA_END = "2025-01-01"
YF_RETRY_PAUSE = 2.0
YF_MAX_RETRIES = 3

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_sp500_table() -> pd.DataFrame:
    """
    Fetch the S&P 500 constituents table from Wikipedia and return a DataFrame.
    The function normalizes ticker names (periods -> hyphens).
    """
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    logging.info("Fetching S&P 500 constituents from Wikipedia...")
    try:
        html = requests.get(WIKI_URL, headers=headers, timeout=15)
        html.raise_for_status()
        tables = pd.read_html(html.text)
        sp500 = tables[0]
        # normalize symbol punctuation (e.g., BRK.B -> BRK-B)
        if "Symbol" in sp500.columns:
            sp500["Symbol"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)
        logging.info("S&P 500 constituents fetched.")
        return sp500
    except Exception as exc:
        logging.exception("Failed to fetch S&P 500 table from Wikipedia.")
        raise


def download_yahoo_batch(tickers: List[str], start: str = YA_START, end: str = YA_END) -> pd.DataFrame:
    """
    Download OHLCV data for a list of tickers using yfinance.
    Returns a concatenated DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume, Stock.
    """
    if not tickers:
        return pd.DataFrame()

    logging.info("Downloading price data from Yahoo Finance...")
    # yfinance returns a multi-index when multiple tickers requested; handle robustly
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
                progress=False
            )
            # If single ticker, yf returns a DataFrame without the ticker level
            if isinstance(raw.columns, pd.MultiIndex):
                records = []
                for t in tickers:
                    if t not in raw.columns.get_level_values(0):
                        logging.warning("No data for ticker: %s", t)
                        continue
                    df = raw[t].copy()
                    df = df.reset_index().rename(columns={"index": "Date"})
                    df["Stock"] = t
                    records.append(df)
                if records:
                    combined = pd.concat(records, ignore_index=True)
                else:
                    combined = pd.DataFrame()
            else:
                # single ticker
                df = raw.reset_index().rename(columns={"index": "Date"})
                if "Close" in df.columns:
                    df["Stock"] = tickers[0]
                    combined = df
                else:
                    combined = pd.DataFrame()
            logging.info("Yahoo download complete.")
            return combined
        except Exception as exc:
            last_exception = exc
            attempts += 1
            logging.warning("yfinance download failed (attempt %d/%d): %s", attempts, YF_MAX_RETRIES, exc)
            time.sleep(YF_RETRY_PAUSE)
    logging.exception("yfinance download failed after %d attempts.", YF_MAX_RETRIES)
    raise last_exception


def reshape_and_merge(yf_df: pd.DataFrame, sp500: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and merge Yahoo data with the S&P500 constituents table.
    Ensures columns are Date, Open, High, Low, Close, Adj Close, Volume, Stock, Sector, Industry.
    """
    if yf_df.empty:
        logging.error("Empty Yahoo Finance DataFrame passed to reshape_and_merge.")
        return pd.DataFrame()

    logging.info("Reshaping and merging price data with sector metadata...")
    # Ensure Date column exists and is datetime
    if "Date" in yf_df.columns:
        yf_df["Date"] = pd.to_datetime(yf_df["Date"])
    else:
        # if index was date
        yf_df = yf_df.reset_index()
        if "Date" in yf_df.columns:
            yf_df["Date"] = pd.to_datetime(yf_df["Date"])

    # Normalize column names to expected set
    expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date", "Stock"}
    missing = expected - set(yf_df.columns)
    if missing:
        logging.debug("Missing expected columns: %s", missing)

    # Merge sector info
    key_map = sp500.rename(columns={"Symbol": "Stock", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry"})
    meta = key_map[["Stock", "Sector", "Industry"]].drop_duplicates(subset=["Stock"])
    merged = yf_df.merge(meta, on="Stock", how="left")
    logging.info("Merged price data with sector information.")
    return merged


def save_raw(df: pd.DataFrame, path: str = RAW_PATH):
    """
    Save DataFrame to parquet, creating directories as necessary.
    Overwrites any existing file to keep idempotent behavior.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info("Saved raw data to %s", path)


def data_fetch(tickers_subset: List[str] = None, overwrite: bool = True):
    """
    Full pipeline:
    - Fetch S&P500 table
    - Construct tickers list (or use tickers_subset if provided)
    - Download Yahoo history
    - Reshape + merge sector metadata
    - Save parquet to disk

    Parameters:
    - tickers_subset: optional list of tickers to download (useful for testing)
    - overwrite: whether to overwrite existing RAW_PATH
    """
    if os.path.exists(RAW_PATH) and not overwrite:
        logging.info("Raw file already exists and overwrite=False; loading existing file.")
        return pd.read_parquet(RAW_PATH)

    sp500 = get_sp500_table()
    tickers = tickers_subset if tickers_subset else sp500["Symbol"].tolist()

    # Download data (yfinance handles batching internally)
    yf_df = download_yahoo_batch(tickers, start=YA_START, end=YA_END)
    merged = reshape_and_merge(yf_df, sp500)

    if merged.empty:
        logging.error("No data fetched. Exiting data_fetch.")
        raise RuntimeError("No price data retrieved from Yahoo Finance.")

    save_raw(merged, RAW_PATH)
    return merged

