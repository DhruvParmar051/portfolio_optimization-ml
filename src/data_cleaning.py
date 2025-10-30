"""
data_cleaning.py

This module performs cleaning and validation of stock market data.
It loads raw data (with sectors), analyzes missing values, trims
invalid or incomplete records, and saves the cleaned dataset.

Pipeline Steps:
1. Load raw dataset from parquet file
2. Identify missing data stretches per stock
3. Analyze and summarize missing patterns
4. Trim data to start from first valid 'Close' price for each stock
5. Save cleaned dataset to a parquet file

Author: Dhruv
Date: 2025-10-29
"""

import os
import pandas as pd
import numpy as np
import logging
import warnings
import fastparquet

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# File paths
INPUT_PATH = "data/raw/all_stocks_data_with_sector.parquet"
OUTPUT_PATH = "data/cleaned/cleaned_data.parquet"


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw parquet dataset.

    Parameters:
        path (str): Path to the input parquet file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Input file not found at {path}")
    
    df = pd.read_parquet(path, engine='fastparquet')
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def find_missing_stretches(group: pd.DataFrame) -> pd.DataFrame:
    """
    Identify consecutive missing stretches for a single stock.

    Parameters:
        group (pd.DataFrame): Data for one stock.

    Returns:
        pd.DataFrame: Summary of missing data stretches.
    """
    group = group.copy()
    # Create a unique ID whenever the missing flag changes
    group['gap_id'] = (group['is_missing'].ne(group['is_missing'].shift())).cumsum()
    
    # Aggregate missing periods
    missing_stretches = (
        group[group['is_missing'] == 1]
        .groupby('gap_id')
        .agg(
            Stock=('Stock', 'first'),
            start_date=('Date', 'min'),
            end_date=('Date', 'max'),
            missing_days=('Date', 'count')
        )
        .reset_index(drop=True)
    )
    return missing_stretches


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing data patterns across all stocks.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: DataFrame with missing value flags added.
    """
    # Flag missing 'Close' values
    df['is_missing'] = df['Close'].isna().astype(int)
    
    # Apply missing stretch detection per stock
    missing_summary = df.groupby('Stock', group_keys=False).apply(find_missing_stretches)
    missing_summary = missing_summary.sort_values('missing_days', ascending=False)
    
    # Log summary statistics
    logging.info("Top 10 longest missing data stretches:")
    logging.info(f"\n{missing_summary.head(10)}")
    
    long_gaps = missing_summary[missing_summary['missing_days'] > 10]
    tickers_with_long_gaps = long_gaps['Stock'].nunique()
    logging.info(f"Number of tickers with gaps > 10 days: {tickers_with_long_gaps}")
    
    return df


def trim_to_first_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim data to start from the first valid Close value per stock.

    Parameters:
        df (pd.DataFrame): Input dataset with potential leading NaNs.

    Returns:
        pd.DataFrame: Trimmed dataset with all leading missing data removed.
    """
    # Sort to ensure proper time order
    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
    
    # Define trimming logic per stock
    def _trim(group):
        first_valid = group['Close'].first_valid_index()
        return group.loc[first_valid:] if first_valid is not None else group
    
    # Apply trimming
    df_trimmed = df.groupby('Stock', group_keys=False).apply(_trim)
    
    # Drop helper columns if present
    df_trimmed = df_trimmed.drop(columns=['is_missing'], errors='ignore')
    
    logging.info(f"Trimmed data shape: {df_trimmed.shape}")
    return df_trimmed


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to a parquet file.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        output_path (str): Output file path.
    """
    # Ensure target folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save parquet file
    df.to_parquet(output_path, index=False)
    logging.info(f"Cleaned dataset saved to {output_path}")


def main():
    """
    Execute the data cleaning pipeline:
      1. Load data
      2. Analyze missing values
      3. Trim invalid portions
      4. Save cleaned data
    """
    try:
        # Step 1: Load raw data
        df = load_data(INPUT_PATH)
        
        # Step 2: Analyze and flag missing data
        df = analyze_missing_data(df)
        
        # Step 3: Trim incomplete leading data
        df_cleaned = trim_to_first_valid(df)
        
        # Step 4: Save final cleaned dataset
        save_data(df_cleaned, OUTPUT_PATH)
        
        logging.info("Data cleaning pipeline completed successfully.")
    except Exception as e:
        logging.exception("Pipeline failed due to an error.")


# Entry point for script execution
if __name__ == "__main__":
    main()
