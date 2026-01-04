"""Feature engineering utilities."""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    angle = df["hour"] * 2 * np.pi / 24.0
    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_peak_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    peak_hours = set(range(17, 23))
    df["is_peak"] = df["hour"].isin(peak_hours).astype(int)
    df["is_offpeak"] = (~df["hour"].isin(peak_hours)).astype(int)
    return df


def add_lag_rolling(
    df: pd.DataFrame,
    group_col: str = "appliance_id",
    targets: tuple = ("usage_kwh", "power_w"),
    lags: tuple = (1, 2, 3),
    windows: tuple = (3, 6),
) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values([group_col, "timestamp"]) if group_col in df.columns else df.sort_values("timestamp")

    group_obj = df.groupby(group_col) if group_col in df.columns else [(None, df)]
    frames = []
    for _, g in group_obj:
        g_aug = g.copy()
        for col in targets:
            if col not in g_aug.columns:
                continue
            for lag in lags:
                g_aug[f"{col}_lag{lag}"] = g_aug[col].shift(lag)
            for w in windows:
                g_aug[f"{col}_rollmean{w}"] = g_aug[col].rolling(window=w, min_periods=1).mean()
        frames.append(g_aug)

    df_out = pd.concat(frames, axis=0).sort_index()
    df_out = df_out.fillna(0)
    return df_out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_peak_flags(df)
    df = add_lag_rolling(df)
    return df
