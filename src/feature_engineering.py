import os, pandas as pd, numpy as np, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = os.path.join(os.getcwd(), "data", "cleaned_data", "cleaned_data.parquet")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "featured_data", "featured_data.parquet")

def create_stock_features(df):
    df = df.sort_values(["Stock", "Date"])
    df["Daily_Return"] = df.groupby("Stock")["Close"].pct_change()
    df["Volatility_7d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(7, min_periods=1).std())
    df["Volatility_30d"] = df.groupby("Stock")["Daily_Return"].transform(lambda x: x.rolling(30, min_periods=1).std())
    df["MA_20"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["MA_50"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x / x.shift(10) - 1)
    df["Log_Volume"] = np.log1p(df["Volume"])
    logging.info("Stock-level features created successfully.")
    return df

def create_sector_features(df):
    sector = df.groupby(["Date", "Sector"]).agg({"Close": "mean", "Daily_Return": "mean"}).reset_index()
    sector["Sector_Return_7d"] = sector.groupby("Sector")["Daily_Return"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df = df.merge(sector[["Date", "Sector", "Sector_Return_7d"]], on=["Date", "Sector"], how="left")
    logging.info("Sector-level features created successfully.")
    return df

def create_targets(df):
    df["Next_Return"] = df.groupby("Stock")["Close"].pct_change().shift(-1)
    df["Next_Return"].fillna(0, inplace=True)
    logging.info("Target (Next_Return) created.")
    return df

def feature_engineering():
    df = pd.read_parquet(INPUT_PATH)
    df = create_stock_features(df)
    df = create_sector_features(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info(f"✅ Feature engineering complete. Shape: {df.shape}")
