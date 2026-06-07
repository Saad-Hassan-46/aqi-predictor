"""
feature_pipeline.py
--------------------
Fetches raw weather + AQI data, engineers features, and upserts
them into the Hopsworks feature store.

Run manually:
    python pipelines/feature_pipeline.py

Run for a specific date (used by backfill.py):
    python pipelines/feature_pipeline.py --date 2025-03-15
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
# Fix Hopsworks /tmp path issue on Windows
import tempfile
tmpdir = os.getenv("HOPSWORKS_TMPDIR")
if tmpdir:
    tempfile.tempdir = tmpdir

AQICN_API_KEY       = os.getenv("56902632900ff5599183fbded99acd3dffd62219")
OPENWEATHER_API_KEY = os.getenv("37b7a3ab54ab4b514cf7d81efc0fd411")
HOPSWORKS_API_KEY   = os.getenv("PDNDYtjehc4whxLP.0HAPAs4ZShvlnuUeyJmSuw4xwPBN3pfwCdhqVYaKspUMdwHlUg0NopCBRNclB6au")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")
CITY_NAME           = os.getenv("CITY_NAME", "Rawalpindi")
CITY_LAT            = float(os.getenv("CITY_LAT", 33.5651))
CITY_LON            = float(os.getenv("CITY_LON", 73.0169))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_aqi_data(city: str = None) -> dict:
    key = os.getenv("AQICN_API_KEY")
    if city is None:
        city = os.getenv("CITY_NAME", "islamabad")
    url = f"https://api.waqi.info/feed/{city}/?token={key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN API error: {data.get('data', 'unknown error')}")

    iaqi = data["data"].get("iaqi", {})

    return {
        "aqi":  data["data"].get("aqi", np.nan),
        "pm25": iaqi.get("pm25", {}).get("v", np.nan),
        "pm10": iaqi.get("pm10", {}).get("v", np.nan),
        "no2":  iaqi.get("no2",  {}).get("v", np.nan),
        "o3":   iaqi.get("o3",   {}).get("v", np.nan),
        "so2":  iaqi.get("so2",  {}).get("v", np.nan),
        "co":   iaqi.get("co",   {}).get("v", np.nan),
    }


def fetch_weather_data(lat: float = None, lon: float = None) -> dict:
    key = os.getenv("OPENWEATHER_API_KEY")
    if lat is None: lat = float(os.getenv("CITY_LAT", 33.7235))
    if lon is None: lon = float(os.getenv("CITY_LON", 73.11822))
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={key}&units=metric"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "temperature":      data["main"]["temp"],
        "humidity":         data["main"]["humidity"],
        "pressure":         data["main"]["pressure"],
        "wind_speed":       data["wind"]["speed"],
        "wind_direction":   data["wind"].get("deg", np.nan),
        "visibility":       data.get("visibility", np.nan),
        "weather_desc":     data["weather"][0]["description"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_time_features(dt: datetime) -> dict:
    """
    Extract cyclical and categorical time features from a datetime.

    Why cyclical encoding?
    Hour 23 and hour 0 are only 1 hour apart, but numerically they are
    23 apart. Sine/cosine encoding wraps the scale so the model sees
    them as close together. Same principle applies to month and weekday.
    """
    hour    = dt.hour
    month   = dt.month
    weekday = dt.weekday()   # 0=Monday, 6=Sunday

    # Cyclical encoding using sine and cosine
    hour_sin    = np.sin(2 * np.pi * hour    / 24)
    hour_cos    = np.cos(2 * np.pi * hour    / 24)
    month_sin   = np.sin(2 * np.pi * month   / 12)
    month_cos   = np.cos(2 * np.pi * month   / 12)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

    return {
        "hour":         hour,
        "day":          dt.day,
        "month":        month,
        "weekday":      weekday,
        "season":       season_map[month],
        "is_weekend":   int(weekday >= 5),
        "hour_sin":     round(hour_sin,    6),
        "hour_cos":     round(hour_cos,    6),
        "month_sin":    round(month_sin,   6),
        "month_cos":    round(month_cos,   6),
        "weekday_sin":  round(weekday_sin, 6),
        "weekday_cos":  round(weekday_cos, 6),
    }


def engineer_weather_interaction_features(weather: dict) -> dict:
    """
    Create derived features that capture interactions between weather variables.

    Meteorological context:
    - Heat index: high temperature + high humidity = worse pollutant trapping
    - Wind chill effect: strong wind disperses pollutants faster
    - Atmospheric stability: high pressure + low wind = pollutants accumulate
    """
    temp     = weather["temperature"]
    humidity = weather["humidity"]
    wind     = weather["wind_speed"]
    pressure = weather["pressure"]

    return {
        # Temperature-humidity interaction (proxy for atmospheric stagnation)
        "temp_humidity_index":    round(temp * humidity / 100, 4),
        # Wind dispersion factor: higher = better pollutant dispersal
        "wind_dispersion":        round(wind ** 1.5, 4),
        # Atmospheric stability index: high pressure + calm wind = stable air
        "atmospheric_stability":  round(pressure / (wind + 0.1), 4),
    }


def compute_aqi_change_rate(current_aqi: float, previous_aqi: float) -> float:
    """
    AQI change rate = percentage change from previous reading to current.

    Returns 0.0 when previous_aqi is missing (first data point).
    A positive value means AQI is worsening; negative means improving.
    """
    if np.isnan(previous_aqi) or previous_aqi == 0:
        return 0.0
    return round((current_aqi - previous_aqi) / previous_aqi * 100, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HOPSWORKS FEATURE STORE
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_feature_group(fs):
    """
    Connect to (or create) the AQI feature group in Hopsworks.

    A Feature Group is like a managed table in the feature store.
    The primary key + event_time combination ensures that if you
    re-run the pipeline for the same timestamp, it updates rather
    than duplicates the row (upsert behaviour).
    """
    feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Hourly AQI and weather features for Islamabad",
    primary_key=["city", "timestamp"],
    event_time="timestamp",
    online_enabled=False,   # Changed from True
)


def fetch_previous_aqi_from_store(fs, city: str) -> float:
    """
    Retrieve the most recent AQI value from the feature store.

    Used to compute the AQI change rate feature. Returns np.nan
    if no previous records exist (first ever pipeline run).
    """
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        # Read the last 2 rows for this city, sorted by time
        df = fg.read()
        city_df = df[df["city"] == city].sort_values("timestamp", ascending=False)
        if len(city_df) > 0:
            return float(city_df.iloc[0]["aqi"])
    except Exception as e:
        log.warning(f"Could not fetch previous AQI from store: {e}")
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(target_date: datetime = None) -> pd.DataFrame:
    """
    Full pipeline: fetch → engineer → validate → push to feature store.

    Args:
        target_date: If None, uses current UTC time (live mode).
                     Pass a datetime for historical backfill mode.

    Returns:
        DataFrame of the row that was pushed to the feature store.
    """
    # ── 1. Determine timestamp ────────────────────────────────────────────────
    if target_date is None:
        now = datetime.now(timezone.utc)
        log.info(f"Running in LIVE mode for {now.isoformat()}")
    else:
        now = target_date
        log.info(f"Running in BACKFILL mode for {now.isoformat()}")

    # ── 2. Fetch raw data ─────────────────────────────────────────────────────
    log.info("Fetching AQI data from AQICN...")
    aqi_raw = fetch_aqi_data(CITY_NAME)
    log.info(f"  AQI={aqi_raw['aqi']}, PM2.5={aqi_raw['pm25']}, PM10={aqi_raw['pm10']}")

    log.info("Fetching weather data from OpenWeather...")
    weather_raw = fetch_weather_data(CITY_LAT, CITY_LON)
    log.info(f"  Temp={weather_raw['temperature']}°C, Humidity={weather_raw['humidity']}%, Wind={weather_raw['wind_speed']}m/s")

    # ── 3. Connect to Hopsworks ───────────────────────────────────────────────
    log.info("Connecting to Hopsworks feature store...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    # ── 4. Get previous AQI for change rate feature ───────────────────────────
    previous_aqi = fetch_previous_aqi_from_store(fs, CITY_NAME)

    # ── 5. Engineer all features ──────────────────────────────────────────────
    log.info("Engineering features...")
    time_features        = engineer_time_features(now)
    interaction_features = engineer_weather_interaction_features(weather_raw)
    aqi_change_rate      = compute_aqi_change_rate(aqi_raw["aqi"], previous_aqi)

    # ── 6. Assemble the feature row ───────────────────────────────────────────
    row = {
        # Identity
        "city":           CITY_NAME,
        "timestamp":      now.replace(minute=0, second=0, microsecond=0),  # floor to hour

        # Target variable (what we're predicting)
        "aqi":            float(aqi_raw["aqi"]),

        # Raw pollutants
        "pm25":           float(aqi_raw["pm25"])  if not np.isnan(aqi_raw["pm25"])  else None,
        "pm10":           float(aqi_raw["pm10"])  if not np.isnan(aqi_raw["pm10"])  else None,
        "no2":            float(aqi_raw["no2"])   if not np.isnan(aqi_raw["no2"])   else None,
        "o3":             float(aqi_raw["o3"])    if not np.isnan(aqi_raw["o3"])    else None,
        "so2":            float(aqi_raw["so2"])   if not np.isnan(aqi_raw["so2"])   else None,
        "co":             float(aqi_raw["co"])    if not np.isnan(aqi_raw["co"])    else None,

        # Raw weather
        "temperature":    weather_raw["temperature"],
        "humidity":       weather_raw["humidity"],
        "pressure":       weather_raw["pressure"],
        "wind_speed":     weather_raw["wind_speed"],
        "wind_direction": weather_raw["wind_direction"] if not np.isnan(weather_raw["wind_direction"]) else None,
        "visibility":     weather_raw["visibility"] if not np.isnan(weather_raw["visibility"]) else None,

        # Engineered: time
        **time_features,

        # Engineered: weather interactions
        **interaction_features,

        # Engineered: AQI change rate
        "aqi_change_rate": aqi_change_rate,
    }

    # NOTE: Lag features (lag_1h, lag_24h, rolling_avg_7d etc.) are computed
    # during the training pipeline, not here. This is intentional — computing
    # lags here would require fetching historical data on every run, which is
    # slow and error-prone. The feature store's point-in-time query handles
    # lag computation cleanly during training.

    df = pd.DataFrame([row])

    # Cast all nullable pollutant/weather columns to float64
    # Hopsworks does not accept pandas 'null' dtype (happens when all values are None)
    nullable_cols = ['pm10', 'no2', 'o3', 'so2', 'co', 'wind_direction', 'visibility']
    for col in nullable_cols:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    
    log.info(f"Feature row assembled: {len(df.columns)} features")

    # ── 7. Validate before pushing ────────────────────────────────────────────
    assert not df["aqi"].isna().any(),       "AQI value is missing — aborting."
    assert df["aqi"].iloc[0] >= 0,           "AQI cannot be negative."
    assert df["aqi"].iloc[0] <= 500,         "AQI above 500 is invalid."
    assert df["temperature"].iloc[0] > -90,  "Temperature below -90°C is invalid."
    log.info("Data validation passed.")

    # ── 8. Push to feature store ──────────────────────────────────────────────
    fg = get_or_create_feature_group(fs)
    fg.insert(df, write_options={"wait_for_job": False, "use_kafka": False})
    log.info(f"Successfully pushed 1 row to feature group 'aqi_features'.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Feature Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD or YYYY-MM-DDTHH:MM format. "
             "Defaults to current UTC time if not provided.",
    )
    args = parser.parse_args()

    if args.date:
        try:
            # Accept both date-only and datetime formats
            if "T" in args.date:
                target = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
            else:
                target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            log.error("Invalid date format. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM")
            sys.exit(1)
    else:
        target = None

    result = run_pipeline(target_date=target)
    print("\n── Feature row pushed ──")
    print(result.T.to_string())
