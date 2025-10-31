"""
main.py

This is the main controller script that runs the entire ML data pipeline.
It orchestrates all stages — from data fetching to preprocessing — in order.

Each step is defined in a separate module under 'src/', and each module
has a dedicated function with the same name as the file (for example,
`data_cleaning()` inside data_cleaning.py).

Pipeline Order:
1. Data Fetching
2. Data Cleaning
3. Feature Engineering
4. Preprocessing

"""

# ======================================================================
# Imports
# ======================================================================

import logging
import sys
import os

# Add src directory to the path so imports work when running main.py directly
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import pipeline modules
from src.data_fetch import data_fetch
from src.data_cleaning import data_cleaning
from src.feature_engineering import feature_engineering
from src.preprocessor import preprocessor
# ======================================================================
# Logging Configuration
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================================================================
# Main Orchestrator
# ======================================================================

def run_pipeline():
    """Run the entire ML data pipeline sequentially."""
    logging.info("=" * 70)
    logging.info("🚀 Starting End-to-End ML Pipeline Execution")
    logging.info("=" * 70)

    try:
        # 1️⃣ Fetch Data
        logging.info("\n--- [1/4] Fetching Raw Data ---")
        data_fetch()

        # 2️⃣ Clean Data
        logging.info("\n--- [2/4] Cleaning Data ---")
        data_cleaning()

        # 3️⃣ Feature Engineering
        logging.info("\n--- [3/4] Generating Features ---")
        feature_engineering()

        # 4️⃣ Preprocessing
        logging.info("\n--- [4/4] Running Preprocessor ---")
        preprocessor()

        logging.info("\n✅ Pipeline completed successfully.")
    except Exception as e:
        logging.exception("❌ Pipeline failed due to an unexpected error.")

# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    run_pipeline()
