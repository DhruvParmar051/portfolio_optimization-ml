"""
main.py

Runs the complete portfolio optimization ML pipeline:
1. Data fetching
2. Cleaning
3. Feature engineering
4. Preprocessing
5. Train-validation split
6. Model training (per-stock next-day return prediction)
"""

import logging
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
from src.train_valid_split import train_valid_split
from src.model import train_per_stock_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("Starting Portfolio Optimization Pipeline")

    logging.info("[1/6] Fetching stock data...")
    data_fetch()

    logging.info("[2/6]  Cleaning data...")
    data_cleaning()

    logging.info("[3/6] Generating features...")
    feature_engineering()

    logging.info("[4/6] Preprocessing data...")
    preprocessor()

    logging.info("[5/6]  Creating train-validation splits...")
    train_valid_split(valid_ratio=0.2)

    logging.info("[6/6]  Training ML models per stock...")
    train_per_stock_models(valid_ratio=0.2)

    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
