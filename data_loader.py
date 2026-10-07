"""Data loading, feature definitions and train/test splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "credit_card_fraud_10k.csv"

TARGET = "is_fraud"
ID_COL = "transaction_id"
NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
BINARY_FEATURES = ["foreign_transaction", "location_mismatch"]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and drop the ID column."""
    df = pd.read_csv(path)
    return df.drop(columns=[ID_COL])


def split_data(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def dataset_summary(df: pd.DataFrame) -> dict:
    """Basic statistics used by the dashboard."""
    counts = df[TARGET].value_counts()
    return {
        "n_rows": len(df),
        "n_features": len(FEATURES),
        "n_legit": int(counts.get(0, 0)),
        "n_fraud": int(counts.get(1, 0)),
        "fraud_rate": float(df[TARGET].mean()),
        "imbalance_ratio": float(counts.get(0, 0) / max(counts.get(1, 0), 1)),
        "class_means": df.groupby(TARGET)[NUMERIC_FEATURES + BINARY_FEATURES].mean(),
    }
