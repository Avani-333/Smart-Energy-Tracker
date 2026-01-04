"""Visualization helpers."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_consumption_over_time(df: pd.DataFrame, output_path: str = None) -> None:
    plt.figure(figsize=(10, 4))
    df_sorted = df.sort_values("timestamp")
    plt.plot(df_sorted["timestamp"], df_sorted["usage_kwh"], label="Usage (kWh)")
    plt.xlabel("Time")
    plt.ylabel("Usage kWh")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_cluster_profiles(df: pd.DataFrame, labels, output_path: str = None) -> None:
    temp = df.copy()
    temp["cluster"] = labels
    cluster_mean = temp.groupby("cluster").mean(numeric_only=True)
    cluster_mean.T.plot(kind="bar", figsize=(10, 5))
    plt.title("Cluster Profiles")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
