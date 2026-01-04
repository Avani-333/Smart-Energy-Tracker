import pandas as pd

from src.data_prep import preprocess
from src.features import add_features


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "appliance_id": [0, 0, 0, 1, 1, 1],
            "appliance_name": ["AC"] * 3 + ["Fridge"] * 3,
            "power_w": [100, 110, 120, 150, 160, 170],
            "voltage": [220] * 6,
            "current": [0.45] * 6,
            "duration_s": [900, 1200, 1500, 900, 1200, 1500],
            "usage_kwh": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "tariff_per_kwh": [0.15] * 6,
            "ambient_temp": [20, 21, 22, 19, 20, 21],
            "occupancy": [1, None, 1, 0, None, 1],
        }
    )


def test_preprocess_derives_time_and_cost():
    df = preprocess(sample_df())
    assert "day_of_week" in df.columns
    assert "hour" in df.columns
    assert df["day_of_week"].dtype.kind in {"i", "u"}
    assert df["hour"].dtype.kind in {"i", "u"}
    assert df["cost"].notna().all()
    assert df["duration_hours"].notna().all()
    assert df["occupancy"].isna().sum() == 0


def test_features_add_lags_and_flags():
    df = preprocess(sample_df())
    df_feat = add_features(df)
    for col in ["hour_sin", "hour_cos", "is_weekend", "is_peak", "is_offpeak"]:
        assert col in df_feat.columns
    for lag_col in ["usage_kwh_lag1", "power_w_lag1"]:
        assert lag_col in df_feat.columns
    for roll_col in ["usage_kwh_rollmean3", "power_w_rollmean3"]:
        assert roll_col in df_feat.columns
    # Lags introduce leading zeros after fillna
    assert (df_feat[["usage_kwh_lag1", "power_w_lag1"]].iloc[0] == 0).all()
