"""Data loading, cleaning, and train/validation split utilities."""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_INPUT, DEFAULT_OUTPUT, PROCESSED_DIR, RAW_DIR, SEED, TEST_SIZE
from .utils import ensure_dir, setup_logging


EXPECTED_COLUMNS = [
    "timestamp",
    "appliance_id",
    "appliance_name",
    "power_w",
    "voltage",
    "current",
    "duration_s",
    "usage_kwh",
    "tariff_per_kwh",
    "day_of_week",
    "hour",
    "ambient_temp",
    "occupancy",
]


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.ffill().bfill()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derive time-based fields if missing or corrupted
    if "day_of_week" not in df or df["day_of_week"].isna().any():
        df["day_of_week"] = df["timestamp"].dt.weekday
    if "hour" not in df or df["hour"].isna().any():
        df["hour"] = df["timestamp"].dt.hour

    df = df.dropna(subset=["timestamp", "usage_kwh", "tariff_per_kwh"])
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["occupancy"] = df.get("occupancy", 0).fillna(0).astype(int)
    df["cost"] = df["usage_kwh"] * df["tariff_per_kwh"]
    df["duration_hours"] = df["duration_s"] / 3600.0
    return df


def split_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = df.drop(columns=[target])
    labels = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    return X_train, X_val, y_train, y_val


def save_processed(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw energy CSVs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Processed CSV path")
    args = parser.parse_args()

    setup_logging()
    df_raw = load_raw_csv(args.input)
    df_clean = preprocess(df_raw)
    save_processed(df_clean, args.output)


if __name__ == "__main__":
    main()
