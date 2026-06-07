"""
csv_backfill.py
---------------
Reads the AQICN historical CSV (Islamabad US Embassy PM2.5 data),
engineers all features, and pushes to Hopsworks — replacing the
flat 74.1 placeholder data with real varied readings.

Usage:
    py -3.11 csv_backfill.py

Place the CSV file at:
    data/islamabad-us_embassy__pakistan-air-quality.csv
"""

import logging
import os
import sys
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
CITY_NAME         = os.getenv("CITY_NAME", "islamabad")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Climate averages for Islamabad (for weather approximation) ────────────────
ISLAMABAD_CLIMATE = {
    1:  {"temp": 9.5,  "humidity": 68, "wind": 2.5, "pressure": 1018},
    2:  {"temp": 11.5, "humidity": 65, "wind": 2.8, "pressure": 1015},
    3:  {"temp": 17.0, "humidity": 60, "wind": 3.2, "pressure": 1010},
    4:  {"temp": 23.0, "humidity": 55, "wind": 3.5, "pressure": 1005},
    5:  {"temp": 29.0, "humidity": 40, "wind": 4.0, "pressure": 1000},
    6:  {"temp": 33.5, "humidity": 45, "wind": 4.5, "pressure":  996},
    7:  {"temp": 31.0, "humidity": 70, "wind": 3.8, "pressure":  994},
    8:  {"temp": 29.5, "humidity": 72, "wind": 3.2, "pressure":  995},
    9:  {"temp": 26.0, "humidity": 65, "wind": 2.8, "pressure": 1000},
    10: {"temp": 21.0, "humidity": 55, "wind": 2.2, "pressure": 1008},
    11: {"temp": 14.5, "humidity": 60, "wind": 2.0, "pressure": 1014},
    12: {"temp": 10.0, "humidity": 68, "wind": 2.2, "pressure": 1017},
}


def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration to AQI using US EPA breakpoints."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm25 = min(pm25, 500.4)
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0


def get_approximate_weather(dt: pd.Timestamp) -> dict:
    """Return approximate weather from Islamabad monthly climate averages."""
    base = ISLAMABAD_CLIMATE[dt.month]
    rng  = np.random.default_rng(seed=int(dt.timestamp()))
    return {
        "temperature":    round(base["temp"]    + rng.normal(0, 2.0), 2),
        "humidity":       int(np.clip(base["humidity"] + rng.integers(-8, 8), 10, 100)),
        "pressure":       int(np.clip(base["pressure"] + rng.integers(-5, 5), 980, 1030)),
        "wind_speed":     round(max(0, base["wind"]    + rng.normal(0, 0.8)), 2),
        "wind_direction": int(rng.integers(0, 360)),
        "visibility":     int(rng.choice([5000, 8000, 10000, 10000, 10000])),
    }


def engineer_time_features(dt: pd.Timestamp) -> dict:
    """Extract cyclical and categorical time features."""
    hour    = dt.hour
    month   = dt.month
    weekday = dt.weekday()
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    return {
        "hour":         hour,
        "day":          dt.day,
        "month":        month,
        "weekday":      weekday,
        "season":       season_map[month],
        "is_weekend":   int(weekday >= 5),
        "hour_sin":     round(np.sin(2 * np.pi * hour    / 24), 6),
        "hour_cos":     round(np.cos(2 * np.pi * hour    / 24), 6),
        "month_sin":    round(np.sin(2 * np.pi * month   / 12), 6),
        "month_cos":    round(np.cos(2 * np.pi * month   / 12), 6),
        "weekday_sin":  round(np.sin(2 * np.pi * weekday / 7),  6),
        "weekday_cos":  round(np.cos(2 * np.pi * weekday / 7),  6),
    }


def build_features_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV, compute AQI from PM2.5, add weather and time features.
    Returns a complete feature DataFrame ready to push to Hopsworks.
    """
    log.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = ['date', 'pm25']
    df['pm25']  = pd.to_numeric(df['pm25'], errors='coerce')
    df['date']  = pd.to_datetime(df['date'])
    df = df.dropna(subset=['pm25']).sort_values('date').reset_index(drop=True)
    log.info(f"Loaded {len(df)} rows: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    rows = []
    prev_aqi = np.nan

    for _, row in df.iterrows():
        dt   = row['date']
        pm25 = float(row['pm25'])
        aqi  = pm25_to_aqi(pm25)

        # AQI change rate
        change_rate = 0.0
        if not np.isnan(prev_aqi) and prev_aqi != 0:
            change_rate = round((aqi - prev_aqi) / prev_aqi * 100, 4)
        prev_aqi = aqi

        weather       = get_approximate_weather(dt)
        time_feats    = engineer_time_features(dt)

        # Weather interaction features
        temp     = weather["temperature"]
        humidity = weather["humidity"]
        wind     = weather["wind_speed"]
        pressure = weather["pressure"]

        feature_row = {
            "city":                  CITY_NAME,
            "timestamp":             dt.replace(hour=12).tz_localize("UTC"),
            "aqi":                   aqi,
            "pm25":                  pm25,
            "pm10":                  np.nan,
            "no2":                   np.nan,
            "o3":                    np.nan,
            "so2":                   np.nan,
            "co":                    np.nan,
            "temperature":           temp,
            "humidity":              humidity,
            "pressure":              pressure,
            "wind_speed":            wind,
            "wind_direction":        float(weather["wind_direction"]),
            "visibility":            float(weather["visibility"]),
            **time_feats,
            "temp_humidity_index":   round(temp * humidity / 100, 4),
            "wind_dispersion":       round(wind ** 1.5, 4),
            "atmospheric_stability": round(pressure / (wind + 0.1), 4),
            "aqi_change_rate":       change_rate,
        }
        rows.append(feature_row)

    result = pd.DataFrame(rows)

    # Fix dtypes to match Hopsworks feature group schema
    nullable_cols = ['pm10', 'no2', 'o3', 'so2', 'co']
    for col in nullable_cols:
        result[col] = result[col].astype('float64')

    log.info(f"Built {len(result)} feature rows with {len(result.columns)} columns")
    return result


def push_to_hopsworks(df: pd.DataFrame):
    """Push the feature DataFrame to Hopsworks in batches."""
    log.info("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Hourly AQI and weather features for Islamabad",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )

    # Push in batches of 100
    batch_size = 100
    total      = len(df)
    pushed     = 0

    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size]
        fg.insert(batch, write_options={"wait_for_job": False, "use_kafka": False})
        pushed += len(batch)
        log.info(f"Pushed {pushed}/{total} rows...")

    log.info(f"Done. All {total} rows pushed to feature group 'aqi_features'.")


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "islamabad-us_embassy__pakistan-air-quality.csv"

    if not csv_path.exists():
        log.error(f"CSV not found at {csv_path}")
        log.error("Please copy the CSV file to your data/ folder first.")
        sys.exit(1)

    df = build_features_from_csv(csv_path)

    # Preview
    print("\nSample rows:")
    print(df[['timestamp', 'aqi', 'pm25', 'temperature', 'month']].head(5).to_string())
    print(f"\nAQI stats:\n{df['aqi'].describe()}")

    push_to_hopsworks(df)