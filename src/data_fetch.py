"""
data_fetch.py

Fetches S&P 500 tickers, downloads historical stock prices from Yahoo Finance,
adds sector info, and saves as a combined parquet dataset.

"""

import os
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_PATH = os.path.join(os.getcwd(), "data", "raw_data", "all_stocks_data_with_sector.parquet")


def get_sp500_tickers() -> pd.DataFrame:
    """Fetch S&P 500 tickers and sectors."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    logging.info("Fetching S&P 500 list from Wikipedia...")
    tables = pd.read_html(StringIO(requests.get(url, headers=headers).text))
    sp500 = tables[0]
    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-", regex=False)
    logging.info(f"Found {len(sp500)} tickers.")
    return sp500


def download_stock_data(tickers):
    """Download historical stock data."""
    logging.info("Downloading data from Yahoo Finance...")
    data = yf.download(
        tickers=tickers, start="2010-01-01", end="2025-01-01",
        group_by="ticker", threads=True, auto_adjust=True
    )
    return data


def reshape_and_merge(all_data, sp500):
    """Reshape multi-indexed Yahoo data and merge with sector info."""
    combined = []
    for ticker in sp500["Symbol"]:
        try:
            df = all_data[ticker].copy()
            df["Stock"] = ticker
            df.reset_index(inplace=True)
            combined.append(df)
        except KeyError:
            logging.warning(f"No data for {ticker}, skipping...")
    data = pd.concat(combined, ignore_index=True)
    sector_info = sp500.rename(columns={"Symbol": "Stock", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry"})
    merged = data.merge(sector_info[["Stock", "Sector", "Industry"]], on="Stock", how="left")
    return merged


def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"Saved raw data → {path}")


def data_fetch():
    """Run entire data fetching pipeline."""
    sp500 = get_sp500_tickers()
    tickers = sp500["Symbol"].tolist()
    raw = download_stock_data(tickers)
    merged = reshape_and_merge(raw, sp500)
    save_data(merged, RAW_DATA_PATH)
    logging.info("✅ Data fetching complete.")
