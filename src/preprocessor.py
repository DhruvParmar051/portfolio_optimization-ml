"""
preprocessor.py

This module prepares the engineered dataset for modeling.
It handles scaling, encoding, and other transformations to make sure
the features are clean, consistent, and ready for the machine learning model.

Pipeline Steps:
1. Load engineered data
2. Handle missing values and outliers
3. Encode categorical variables
4. Scale numerical features
5. Save preprocessed data for model training

Author: Dhruv
Date: 2025-10-31
"""

# ======================================================================
# Imports
# ======================================================================

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ======================================================================
# Logging Configuration
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================================================================
# File Paths
# ======================================================================

INPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "engineered_data.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")

# ======================================================================
# Core Functions
# ======================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the engineered dataset from parquet file.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Input file not found at {path}")

    df = pd.read_parquet(path)
    logging.info(f"Loaded engineered dataset successfully. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values depending on feature type.
    """
    # Fill numeric NaNs with median (more robust than mean)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaNs with mode (most frequent value)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    logging.info("Handled missing values for numeric and categorical features.")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using LabelEncoder.
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 0:
        logging.info("No categorical columns found for encoding.")
        return df

    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        logging.info(f"Encoded column: {col}")

    return df


def scale_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scale numerical features using the specified method.
    Options: 'standard' (z-score) or 'minmax' (0-1 scaling).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info(f"Scaled numerical features using {method} scaler.")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the preprocessed dataset to parquet file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved preprocessed dataset at: {output_path}")


# ======================================================================
# Main Pipeline
# ======================================================================
 
def preprocessor():
    """
    Run the preprocessing pipeline:
      1. Load engineered data
      2. Handle missing values
      3. Encode categoricals
      4. Scale numericals
      5. Save processed output
    """
    try:
        df = load_data(INPUT_PATH)
        df = handle_missing_values(df)
        df = encode_categorical(df)
        df = scale_features(df, method="standard")
        save_data(df, OUTPUT_PATH)
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.exception("Preprocessing failed due to an unexpected error.")

