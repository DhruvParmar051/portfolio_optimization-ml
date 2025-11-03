"""
main.py

Master pipeline orchestrating all stages of the portfolio modeling system.

Steps:
1. Data Fetch
2. Cleaning
3. Feature Engineering
4. Preprocessing
5. Train-Validation Split
6. Expanding-Window ARIMA Modeling
7. Portfolio Optimization
8. Q1 2025 Backtest
9. Reporting

Author: Dhruv
Date: 2025-11-03
"""

# ======================================================================
# Imports
# ======================================================================
import logging
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
from src.train_valid_split import train_valid_split
from src.model import run_expanding_arima
from src.optimize_portfolio import portfolio_optimization
from backtest.backtest_q1 import run_backtest
from src.reporting import generate_report

# ======================================================================
# Logging Configuration
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ======================================================================
# Main Pipeline
# ======================================================================
def main():
    """Run the complete portfolio optimization pipeline."""
    try:
        logging.info("=== Starting Portfolio Modeling Pipeline ===")

        logging.info("[1/9] Fetching raw data...")
        # data_fetch()

        logging.info("[2/9] Cleaning data...")
        # data_cleaning()

        logging.info("[3/9] Feature engineering...")
        # feature_engineering()

        logging.info("[4/9] Preprocessing...")
        # preprocessor()

        logging.info("[5/9] Creating train-validation splits...")
        # train_valid_split(valid_ratio=0.2)

        logging.info("[6/9] Running expanding-window ARIMA models...")
        # run_expanding_arima()

        logging.info("[7/9] Optimizing portfolio...")
        # portfolio_optimization()

        logging.info("[8/9] Running Backtest for Q1 2025...")
        metrics = run_backtest()
        if metrics is not None:
            logging.info("Backtest summary:")
            logging.info(metrics.to_string(index=False))
        else:
            logging.warning("Backtest completed but no valid metrics were produced (likely no overlapping forecast dates).")

        logging.info("[9/9] Generating visual report...")
        # generate_report()

        logging.info("=== Pipeline executed successfully. ===")

    except Exception as e:
        logging.exception("Pipeline execution failed.")


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == "__main__":
    main()
