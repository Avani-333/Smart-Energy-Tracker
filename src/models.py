"""Regression models for consumption and cost estimation."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_OUTPUT, SEED, TEST_SIZE
from .data_prep import preprocess
from .features import add_features
from .utils import ensure_dir, setup_logging


def build_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_elastic_pipeline(alpha: float = 0.1, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)),
        ]
    )


def evaluate(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, float]:
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    return mae, rmse


def select_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = {target, "timestamp", "appliance_name"}
    # drop appliance_id if not numeric
    if target != "appliance_id":
        drop_cols.add("appliance_id")
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    features = features.select_dtypes(include=["number", "bool"])
    if features.empty:
        raise ValueError("No numeric features available after preprocessing. Check input columns.")
    labels = df[target]
    return features, labels


def train_and_eval(
    df: pd.DataFrame,
    target: str,
    alpha: float,
    l1_ratio: float,
) -> Tuple[Dict[str, float], Pipeline]:
    features, labels = select_features(df, target)
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    lin = build_linear_pipeline()
    lin.fit(X_train, y_train)
    lin_mae, lin_rmse = evaluate(lin, X_val, y_val)

    elastic = build_elastic_pipeline(alpha=alpha, l1_ratio=l1_ratio)
    elastic.fit(X_train, y_train)
    el_mae, el_rmse = evaluate(elastic, X_val, y_val)

    metrics = {
        "linear_mae": lin_mae,
        "linear_rmse": lin_rmse,
        "elastic_mae": el_mae,
        "elastic_rmse": el_rmse,
    }
    best_model = elastic if el_mae <= lin_mae else lin
    return metrics, best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regression models")
    parser.add_argument("--train", type=Path, default=DEFAULT_OUTPUT, help="Processed CSV path")
    parser.add_argument("--target", type=str, default="usage_kwh", help="Target column")
    parser.add_argument("--alpha", type=float, default=0.1, help="ElasticNet alpha")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="ElasticNet l1_ratio")
    parser.add_argument("--save", type=Path, default=Path("reports/model.joblib"), help="Where to save model")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"), help="Where to save metrics")
    args = parser.parse_args()

    setup_logging()
    df = pd.read_csv(args.train)
    df = preprocess(df)
    df = add_features(df)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns: {df.columns.tolist()}")

    metrics, best_model = train_and_eval(df, args.target, args.alpha, args.l1_ratio)

    ensure_dir(args.save.parent)
    ensure_dir(args.metrics.parent)
    joblib.dump(best_model, args.save)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.save}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()
