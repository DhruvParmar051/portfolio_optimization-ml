"""
main.py

Orchestrates the full S&P500 portfolio optimization pipeline.
"""

import logging
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
from src.train_valid_split import train_valid_split
from src.model import train_models
from src.optimize_portfolio import portfolio_optimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    print("=== 📊 Portfolio Optimizer Setup ===")
    stocks = input("Enter stock tickers (comma separated): ").upper().replace(" ", "").split(",")
    capital = float(input("Enter your investment capital (in ₹): "))
    risk = input("Enter risk appetite (low / medium / high): ").lower()
    duration = float(input("Enter investment duration (in years): "))

    print("\n=== 🚀 Running Pipeline ===")
    data_fetch()
    data_cleaning()
    feature_engineering()
    preprocessor()
    train_valid_split(valid_ratio=0.2)
    train_models()

    print("\n=== 💼 Optimizing Portfolio ===")
    df_result, summary = portfolio_optimizer(stocks, capital, risk, int(duration * 12))
    print("\n--- Allocation ---")
    print(df_result.to_string(index=False))
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
