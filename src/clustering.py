"""K-Means clustering for appliance usage profiles."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from .config import DEFAULT_OUTPUT
from .data_prep import preprocess
from .features import add_features
from .utils import setup_logging


def build_cluster_pipeline(n_clusters: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=0)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster appliance usage")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT, help="Processed CSV path")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    args = parser.parse_args()

    setup_logging()
    df = pd.read_csv(args.input)
    df = preprocess(df)
    df = add_features(df)
    feature_cols = [c for c in df.columns if c not in {"timestamp", "appliance_name", "appliance_id"}]
    X = df[feature_cols].select_dtypes(include=["number", "bool"])

    pipe = build_cluster_pipeline(args.clusters)
    labels = pipe.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"Silhouette score: {sil:.4f}")


if __name__ == "__main__":
    main()
