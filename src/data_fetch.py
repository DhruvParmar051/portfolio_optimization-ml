import os
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

response = requests.get(url, headers=headers)
tables = pd.read_html(StringIO(response.text))
sp500 = tables[0]

tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
print(f"Found {len(tickers)} tickers.")

all_data = yf.download(
    tickers=tickers,
    start='2010-01-01',
    end='2025-01-01',
    group_by='ticker',
    threads=True,
    auto_adjust=True
)

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
print(data.head())

output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'all_stocks_data.parquet')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
data.to_parquet(output_path, index=False)
print("Dataset saved successfully as", output_path)

