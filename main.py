"""
main.py

Master pipeline runner for S&P 500 portfolio modeling with ARIMA-based forecasting.

Pipeline Steps:
1. Fetch raw S&P 500 data
2. Clean and validate data
3. Perform feature engineering
4. Preprocess (scaling, encoding)
5. Create chronological train-validation splits
6. Run expanding-window ARIMA modeling for backtesting

Author: Dhruv
Date: 2025-11-02
"""

import logging
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
from src.train_valid_split import train_valid_split
from src.model import run_expanding_arima
from src.optimize_portfolio import portfolio_optimization
from src.reporting import generate_report

# ======================================================================
# Logging Configuration
# ======================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================================================================
# Main Execution
# ======================================================================
def main():
    """Run the complete portfolio optimization pipeline."""
    try:
        logging.info("=== Starting Portfolio Modeling Pipeline ===")

        logging.info("[1/8] Fetching raw data...")
        data_fetch()

        logging.info("[2/8] Cleaning data...")
        data_cleaning()

        logging.info("[3/8] Feature engineering...")
        feature_engineering()

        logging.info("[4/8] Preprocessing...")
        preprocessor()

        logging.info("[5/8] Creating train-validation splits...")
        train_valid_split(valid_ratio=0.2)

        logging.info("[6/8] Running expanding-window ARIMA models...")
        run_expanding_arima()

        logging.info("[7/8] Optimizing portfolio and backtesting...")
        portfolio_optimization()
        logging.info("=== Pipeline executed successfully. ===")

        logging.info("[8/8]Generating visual report...")
        generate_report()
        
    except Exception as e:
        logging.exception("Pipeline execution failed.")


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == "__main__":
    main()
