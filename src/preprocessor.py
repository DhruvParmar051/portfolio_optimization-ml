"""
preprocessor.py

Preprocesses cleaned and feature-engineered stock market data.
Handles missing values, encoding, and scaling for both training and backtesting.

Includes automatic column alignment between training and backtesting.
"""

# ===========================================================
# Imports
# ===========================================================
import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ===========================================================
# Logging Configuration
# ===========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================
# File Paths
# ===========================================================
FEATURED_TRAIN_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")
FEATURED_BACKTEST_PATH = os.path.join(os.getcwd(), "data", "backtest", "featured_data", "featured_data.parquet")

PROCESSED_TRAIN_PATH = os.path.join(os.getcwd(), "data", "processed_data", "processed_data.parquet")
PROCESSED_BACKTEST_PATH = os.path.join(os.getcwd(), "data", "backtest", "processed_data", "processed_data.parquet")

SCALER_PATH = os.path.join(os.getcwd(), "models", "scaler.joblib")
ENCODER_PATH = os.path.join(os.getcwd(), "models", "encoder.joblib")

# ===========================================================
# Utility Functions
# ===========================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by forward/backward filling and dropping residual NaNs."""
    id_col = "Stock" if "Stock" in df.columns else "Symbol"
    df = df.sort_values(by=[id_col, "Date"])
    df = df.groupby(id_col).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    df = df.dropna()
    logger.info("Missing values handled successfully.")
    return df


def fit_and_save_encoders(df: pd.DataFrame) -> dict:
    """Fit label encoders on training dataset and save them."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoders[col] = le
        logger.info(f"Encoder fitted for '{col}' with {len(le.classes_)} classes.")
    joblib.dump(encoders, ENCODER_PATH)
    logger.info(f"✅ Encoders saved successfully → {ENCODER_PATH}")
    return encoders


def encode_categorical_features(df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
    """
    Encode categorical variables (like Sector, Stock).
    Automatically rebuilds encoders if missing.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) == 0:
        logger.info("No categorical columns found for encoding.")
        return df

    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

    try:
        if training_mode:
            encoders = fit_and_save_encoders(df)
        else:
            logger.info("Loading pre-fitted label encoders for backtest...")
            if not os.path.exists(ENCODER_PATH):
                logger.warning("⚠️ Encoder file missing — retraining encoders using training dataset.")
                if not os.path.exists(FEATURED_TRAIN_PATH):
                    raise FileNotFoundError("Training data not found for retraining encoders.")
                train_df = pd.read_parquet(FEATURED_TRAIN_PATH)
                fit_and_save_encoders(train_df)

            saved_encoders = joblib.load(ENCODER_PATH)
            for col in cat_cols:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                if col in saved_encoders:
                    le = saved_encoders[col]
                    if len(le.classes_) == 0:
                        logger.warning(f"⚠️ Encoder for '{col}' is empty. Skipping encoding.")
                        continue
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
                    if "UNKNOWN" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "UNKNOWN")
                    df[col] = le.transform(df[col])
                else:
                    logger.warning(f"⚠️ Column '{col}' not found in saved encoders. Skipping.")
            logger.info("✅ Categorical encoding aligned with training metadata.")

    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise

    return df


def fit_and_save_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler and save it."""
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump({"scaler": scaler, "columns": numeric_cols}, SCALER_PATH)
    logger.info(f"✅ Scaler trained and saved → {SCALER_PATH}")
    return scaler


def scale_numeric_features(df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
    """
    Standardize numeric features using StandardScaler.
    Automatically rebuilds scaler if missing and aligns columns.
    """
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for scaling.")
        return df

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    try:
        if training_mode:
            fit_and_save_scaler(df)
        else:
            logger.info("Loading pre-fitted StandardScaler for inference/backtest...")
            if not os.path.exists(SCALER_PATH):
                logger.warning("⚠️ Scaler file missing — retraining scaler using training dataset.")
                if not os.path.exists(FEATURED_TRAIN_PATH):
                    raise FileNotFoundError("Training data not found for retraining scaler.")
                train_df = pd.read_parquet(FEATURED_TRAIN_PATH)
                fit_and_save_scaler(train_df)

            saved_obj = joblib.load(SCALER_PATH)
            scaler = saved_obj.get("scaler", saved_obj)
            train_cols = saved_obj.get("columns", numeric_cols)

            # === Align Columns ===
            for col in train_cols:
                if col not in df.columns:
                    df[col] = 0.0  # Add missing with neutral value
                    logger.warning(f"⚠️ Added missing feature '{col}' with zeros for alignment.")
            df = df[[c for c in train_cols if c in df.columns]]  # reorder and remove extras

            df[train_cols] = scaler.transform(df[train_cols])
            logger.info("✅ Numeric features scaled successfully and aligned with training metadata.")

    except Exception as e:
        logger.error(f"Error while scaling numeric features: {e}")
        raise

    return df


# ===========================================================
# Main Preprocessing Function
# ===========================================================
def preprocessor(training_mode: bool = True, backtest: bool = False) -> pd.DataFrame:
    """
    Main preprocessing pipeline. Handles missing values, encoding, and scaling.
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

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input data not found at {input_path}")

        df = pd.read_parquet(input_path)
        logger.info(f"Initial dataset shape: {df.shape}")

        df = handle_missing_values(df)
        df = encode_categorical_features(df, training_mode=training_mode)
        df = scale_numeric_features(df, training_mode=training_mode)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Preprocessed data saved → {output_path}")
        logger.info(f"Final dataset shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("=== Running Preprocessor Standalone ===")
    df = preprocessor(training_mode=True, backtest=False)
    logger.info(f"✅ Preprocessing complete. Final shape: {df.shape}")
