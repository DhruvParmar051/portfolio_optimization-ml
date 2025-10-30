import os
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# === STEP 1: Get S&P 500 company list with sector info ===
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

response = requests.get(url, headers=headers)
tables = pd.read_html(StringIO(response.text))
sp500 = tables[0]

# Clean ticker symbols (replace '.' with '-' for Yahoo)
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)
tickers = sp500['Symbol'].tolist()
print(f"Found {len(tickers)} tickers.")

# Keep sector and industry info
sector_info = sp500[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
sector_info.columns = ['Stock', 'Sector', 'Industry']

# === STEP 2: Download multi-stock data from Yahoo Finance ===
all_data = yf.download(
    tickers=tickers,
    start='2010-01-01',
    end='2025-01-01',
    group_by='ticker',
    threads=True,
    auto_adjust=True
)

# === STEP 3: Reshape and combine ===
all_data_list = []
for ticker in tickers:
    try:
        df = all_data[ticker].copy()
        df['Stock'] = ticker
        df.reset_index(inplace=True)
        all_data_list.append(df)
    except KeyError:
        print(f"No data for {ticker}, skipping...")

data = pd.concat(all_data_list, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# === STEP 4: Merge with sector info ===
data = data.merge(sector_info, on='Stock', how='left')

print("Sample after merging sector info:")
print(data[['Date', 'Stock', 'Sector', 'Industry']].head())

# === STEP 5: Save the dataset ===
output_path = os.path.join(os.getcwd(), '..','data', 'all_stocks_data_with_sector.parquet')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
data.to_parquet(output_path, index=False)

print("Dataset saved successfully as", output_path)
