"""
preprocessor.py

Production-level preprocessing module for stock-level and sector-level features.

Responsibilities:
1. Handle missing values in numeric and categorical columns.
2. Encode selected categorical variables (Sector, Industry) using one-hot encoding.
3. Scale numeric features with StandardScaler.
4. Persist preprocessing artifacts (scaler and metadata) for reproducibility.
5. Save preprocessed dataset ready for model training.
"""

# ===========================================================
# Imports
# ===========================================================

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ===========================================================
# Configuration
# ===========================================================

INPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data.parquet")


ARTIFACT_DIR = os.path.join(os.getcwd(), "artifacts", "preprocessing")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ENCODER_METADATA_PATH = os.path.join(ARTIFACT_DIR, "encoder_columns.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===========================================================
# Helper Functions
# ===========================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for both numeric and categorical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill numeric columns with median (robust to outliers)
    for col in num_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    # Fill categoricals with mode (most frequent)
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode.iloc[0], inplace=True)
        else:
            df[col].fillna("Unknown", inplace=True)

    logging.info("Missing values handled successfully.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 'Sector' and 'Industry' using one-hot encoding.
    Persists encoder column list for consistency during inference.
    """
    encode_cols = [col for col in ["Sector", "Industry"] if col in df.columns]
    if not encode_cols:
        logging.warning("No categorical columns found for encoding.")
        return df

    df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True)
    encoded_columns = df_encoded.columns.tolist()

    # Persist column list for future inference alignment
    joblib.dump(encoded_columns, ENCODER_METADATA_PATH)
    logging.info(f"Categoricals encoded: {encode_cols}. Saved metadata → {ENCODER_METADATA_PATH}")
    return df_encoded


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features excluding identifiers and targets.
    Saves the fitted scaler for reuse during inference.
    """
    exclude = {"Date", "Stock", "Next_Return"}
    num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude]

    if not num_cols:
        logging.warning("No numeric features found to scale.")
        return df

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save fitted scaler
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Numeric features scaled and scaler saved → {SCALER_PATH}")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize numeric data types to reduce memory usage."""
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    logging.info("Memory optimization complete.")
    return df


# ===========================================================
# Main Preprocessing Pipeline
# ===========================================================

def preprocessor():
    """Run the complete preprocessing pipeline."""
    try:
        logging.info("Loading featured dataset...")
        df = pd.read_parquet(INPUT_PATH)
        logging.info(f"Initial dataset shape: {df.shape}")

        df = handle_missing_values(df)
        df = encode_categoricals(df)
        df = scale_numeric_features(df)
        df = reduce_memory(df)

        # Save processed dataset
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)

        logging.info(f"Preprocessing complete. Final shape: {df.shape}")
        logging.info(f"Saved preprocessed dataset → {OUTPUT_PATH}")

    except Exception as e:
        logging.exception(f"Preprocessing failed: {e}")
