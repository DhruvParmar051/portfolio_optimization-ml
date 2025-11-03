"""
preprocessor.py

Preprocesses cleaned and feature-engineered stock market data.
Handles missing values, scaling, and encoding for both training and backtesting.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === File Paths ===
FEATURED_TRAIN_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")
FEATURED_BACKTEST_PATH = os.path.join(os.getcwd(), "data", "backtest", "featured_data_backtest.parquet")
PROCESSED_TRAIN_PATH = os.path.join(os.getcwd(), "data", "processed_data", "processed_data.parquet")
PROCESSED_BACKTEST_PATH = os.path.join(os.getcwd(), "data", "backtest", "processed_data", "processed_data.parquet")
SCALER_PATH = os.path.join(os.getcwd(), "models", "scaler.joblib")
ENCODER_PATH = os.path.join(os.getcwd(), "models", "encoder.joblib")

# === Utility Functions ===

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by forward/backward filling and dropping residual NaNs."""
    id_col = "Stock" if "Stock" in df.columns else "Symbol"
    df = df.sort_values(by=[id_col, "Date"])
    df = df.groupby(id_col).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    df = df.dropna()
    logger.info("Missing values handled successfully.")
    return df


def encode_categorical_features(df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
    """
    Encode categorical variables (like Sector, Stock).
    Saves encoder mappings during training; loads and applies during backtesting.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) == 0:
        return df

    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

    try:
        if training_mode:
            logger.info("Fitting label encoders on categorical columns...")
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            joblib.dump(encoders, ENCODER_PATH)
            logger.info(f"✅ Encoders saved successfully → {ENCODER_PATH}")
        else:
            logger.info("Loading pre-fitted label encoders for backtest...")
            saved_encoders = joblib.load(ENCODER_PATH)
            for col in cat_cols:
                if col in saved_encoders:
                    le = saved_encoders[col]
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
                else:
                    logger.warning(f"Column {col} not found in saved encoders; skipping.")
            logger.info("✅ Categorical encoding aligned with training metadata.")
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise

    return df


def scale_numeric_features(df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
    """
    Standardize numeric features using StandardScaler.
    During training, it fits and saves the scaler.
    During backtest/inference, it loads and applies the saved scaler.
    """
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for scaling.")
        return df

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    try:
        if training_mode:
            logger.info("Fitting StandardScaler on numeric features...")
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"✅ Scaler saved successfully → {SCALER_PATH}")
        else:
            logger.info("Loading pre-fitted StandardScaler for inference/backtest...")
            saved_obj = joblib.load(SCALER_PATH)
            scaler = saved_obj["scaler"] if isinstance(saved_obj, dict) and "scaler" in saved_obj else saved_obj
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            logger.info("✅ Numeric features scaled successfully using loaded scaler.")
    except Exception as e:
        logger.error(f"Error while scaling numeric features: {e}")
        raise

    return df


# === Main Function ===

def preprocessor(training_mode: bool = True, backtest: bool = False) -> pd.DataFrame:
    """
    Main preprocessing pipeline. Handles missing values, encoding, and scaling.

    Parameters
    ----------
    training_mode : bool
        If True, preprocess training data.
    backtest : bool
        If True, run in backtest mode.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    try:
        if backtest:
            input_path = FEATURED_BACKTEST_PATH
            output_path = PROCESSED_BACKTEST_PATH
            logger.info(f"Running in BACKTEST mode using {input_path}")
        else:
            input_path = FEATURED_TRAIN_PATH
            output_path = PROCESSED_TRAIN_PATH
            logger.info(f"Running in TRAINING mode using {input_path}")

        df = pd.read_parquet(input_path)
        logger.info(f"Initial dataset shape: {df.shape}")

        df = handle_missing_values(df)
        df = encode_categorical_features(df, training_mode=training_mode)
        df = scale_numeric_features(df, training_mode=training_mode)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Preprocessed data saved → {output_path}")

        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
