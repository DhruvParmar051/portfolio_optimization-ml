"""
preprocessor.py

Cleans missing values, one-hot encodes categoricals, and scales numerics.
"""

import os, logging, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "preprocessed_data", "preprocessed_data.parquet")


def handle_missing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols: df[c] = df[c].fillna(df[c].mode()[0])
    return df


def encode_categoricals(df):
    cat_cols = ["Stock", "Sector", "Industry"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    logging.info("Categoricals encoded.")
    return df


def scale(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info("Numerical features scaled.")
    return df


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

