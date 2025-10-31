"""
data_fetching.py

This script downloads S&P 500 company data from Wikipedia, fetches their
historical stock prices from Yahoo Finance, and merges everything into
a single parquet file with sector and industry information.

Pipeline Steps:
1. Fetch S&P 500 tickers and sector data from Wikipedia
2. Download historical prices for each company via Yahoo Finance
3. Combine and merge sector info
4. Save the final dataset in the 'data/raw/' directory
"""

# ======================================================================
# Imports
# ======================================================================

import os
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import logging

# ======================================================================
# Configuration and Logging
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = os.path.join(os.getcwd(), "data", "raw_data", "all_stocks_data_with_sector.parquet")

# ======================================================================
# Data Fetching Functions
# ======================================================================

def get_sp500_tickers() -> pd.DataFrame:
    """Fetch S&P 500 tickers and sector information from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    logging.info("Fetching S&P 500 company list from Wikipedia...")
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]

    # Adjust ticker symbols for Yahoo Finance compatibility
    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-", regex=False)

    logging.info(f"Found {len(sp500)} tickers from the S&P 500 list.")
    return sp500


def download_stock_data(tickers):
    """Download historical stock data for all tickers from Yahoo Finance."""
    logging.info("Downloading historical stock data (this may take a while)...")

    all_data = yf.download(
        tickers=tickers,
        start="2010-01-01",
        end="2025-01-01",
        group_by="ticker",
        threads=True,
        auto_adjust=True
    )
    return all_data


def reshape_and_merge(all_data, sp500):
    """Reshape multi-indexed Yahoo data and merge with sector info."""
    combined_list = []

    for ticker in sp500["Symbol"]:
        try:
            df = all_data[ticker].copy()
            df["Stock"] = ticker
            df.reset_index(inplace=True)
            combined_list.append(df)
        except KeyError:
            logging.warning(f"No data found for {ticker}, skipping...")

    data = pd.concat(combined_list, ignore_index=True)
    logging.info(f"Combined dataset shape: {data.shape}")

    sector_info = sp500[["Symbol", "GICS Sector", "GICS Sub-Industry"]]
    sector_info.columns = ["Stock", "Sector", "Industry"]

    merged_data = data.merge(sector_info, on="Stock", how="left")
    return merged_data


def save_data(df, path):
    """Save final dataset to parquet file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Dataset saved successfully at: {path}")

# ======================================================================
# Main Pipeline
# ======================================================================

def data_fetch():
    """Run the complete data fetching pipeline."""
    try:
        sp500 = get_sp500_tickers()
        tickers = sp500["Symbol"].tolist()
        raw_data = download_stock_data(tickers)
        merged_data = reshape_and_merge(raw_data, sp500)
        save_data(merged_data, RAW_DATA_PATH)
        logging.info("Data fetching pipeline completed successfully.")
    except Exception as e:
        logging.exception("Data fetching failed due to an error.")

