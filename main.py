"""
main.py

Main orchestration script for the Portfolio Optimization Pipeline.

Pipeline Steps:
1. Fetch raw data
2. Clean data
3. Feature engineering
4. Preprocessing
5. Train-validation split
6. Expanding-window ARIMA modeling
7. Portfolio optimization
8. Backtesting (Q1 2025)
9. Visual reporting

Author: Dhruv
Date: 2025-11-02
"""

# ============================================================
# Imports
# ============================================================

import logging
import warnings
warnings.filterwarnings("ignore")

# Import pipeline modules
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
from src.train_valid_split import train_valid_split
from src.model import run_expanding_arima
from src.optimize_portfolio import portfolio_optimization
from src.reporting import generate_report

# Backtest import
from backtest.backtest_q1 import run_backtest


# ============================================================
# Main Function
# ============================================================

def main():
    """Run the complete portfolio optimization pipeline."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("=== Starting Portfolio Modeling Pipeline ===")

        # ------------------------------------------------------------------
        # [1/9] Fetch Raw Data
        # ------------------------------------------------------------------
        logging.info("[1/9] Fetching raw data...")
        # data_fetch()

        # ------------------------------------------------------------------
        # [2/9] Clean Data
        # ------------------------------------------------------------------
        logging.info("[2/9] Cleaning data...")
        # data_cleaning()

        # ------------------------------------------------------------------
        # [3/9] Feature Engineering
        # ------------------------------------------------------------------
        logging.info("[3/9] Feature engineering...")
        # feature_engineering()

        # ------------------------------------------------------------------
        # [4/9] Preprocessing
        # ------------------------------------------------------------------
        logging.info("[4/9] Preprocessing...")
        # preprocessor()

        # ------------------------------------------------------------------
        # [5/9] Train-Validation Split
        # ------------------------------------------------------------------
        logging.info("[5/9] Creating train-validation splits...")
        # train_valid_split(valid_ratio=0.2)

        # ------------------------------------------------------------------
        # [6/9] Expanding-Window ARIMA Modeling
        # ------------------------------------------------------------------
        logging.info("[6/9] Running expanding-window ARIMA models...")
        run_expanding_arima()

        # ------------------------------------------------------------------
        # [7/9] Portfolio Optimization
        # ------------------------------------------------------------------
        logging.info("[7/9] Optimizing portfolio...")
        portfolio_optimization()

        # ------------------------------------------------------------------
        # [8/9] Backtest Q1 2025
        # ------------------------------------------------------------------
        logging.info("[8/9] Running Backtest for Q1 2025...")
        metrics = run_backtest()
        if metrics is not None and not metrics.empty:
            logging.info(metrics.to_string(index=False))
        else:
            logging.warning("Backtest completed but no valid metrics were produced (likely no overlapping forecast dates).")


        # ------------------------------------------------------------------
        # [9/9] Visual Report Generation
        # ------------------------------------------------------------------
        logging.info("[9/9] Generating visual report...")
        # generate_report()

        logging.info("=== Pipeline executed successfully. ===")

    except Exception as e:
        logging.exception("Pipeline execution failed.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
