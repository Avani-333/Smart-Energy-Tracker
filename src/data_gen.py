"""Synthetic data generator for Smart Energy Tracker."""

from datetime import datetime, timedelta
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .config import RAW_DIR, SEED
from .utils import ensure_dir, setup_logging


APPLIANCES = [
    ("AC", 1200, 0.2),
    ("Fridge", 150, 0.9),
    ("Washer", 500, 0.1),
    ("Dryer", 1800, 0.05),
    ("Heater", 2000, 0.15),
    ("TV", 120, 0.4),
    ("Laptop", 65, 0.6),
    ("LED Bulbs", 15, 0.7),
    ("Ceiling Fan", 75, 0.5),
    ("Microwave", 1200, 0.1),
    ("Iron", 1000, 0.05),
    ("Water Heater", 2500, 0.2),
    ("Router/WiFi", 10, 0.95),
    ("Phone Charger", 5, 0.3),
    ("Desktop PC", 150, 0.4),
]


def generate(rows: int, start: datetime) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    for i in range(rows):
        ts = start + timedelta(minutes=30 * i)
        appliance_name, base_power, usage_prob = APPLIANCES[i % len(APPLIANCES)]
        active = rng.random() < usage_prob
        power_w = base_power * (0.8 + 0.4 * rng.random()) if active else base_power * 0.05
        voltage = 220 + rng.normal(0, 3)
        current = power_w / max(voltage, 1)
        duration_s = rng.integers(300, 3600)
        usage_kwh = (power_w * duration_s) / (1000 * 3600)
        tariff = 6.0 + 2.0 * rng.random()  # ₹6-8 per kWh (realistic Indian tariff)
        day_of_week = ts.weekday()
        hour = ts.hour
        ambient_temp = 20 + 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.5)
        occupancy = rng.integers(0, 2)
        records.append(
            {
                "timestamp": ts.isoformat(),
                "appliance_id": i % len(APPLIANCES),
                "appliance_name": appliance_name,
                "power_w": round(power_w, 2),
                "voltage": round(voltage, 2),
                "current": round(current, 3),
                "duration_s": int(duration_s),
                "usage_kwh": round(usage_kwh, 4),
                "tariff_per_kwh": round(tariff, 4),
                "day_of_week": int(day_of_week),
                "hour": int(hour),
                "ambient_temp": round(ambient_temp, 2),
                "occupancy": int(occupancy),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic energy dataset")
    parser.add_argument("--rows", type=int, default=240, help="Number of rows to generate")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "sample.csv", help="Output CSV path")
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.output.parent)
    df = generate(args.rows, start=datetime.now().replace(minute=0, second=0, microsecond=0))
    df.to_csv(args.output, index=False)
    print(f"Wrote synthetic data to {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
