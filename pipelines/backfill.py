"""
backfill.py
-----------
Populates the Hopsworks feature store with historical AQI + weather data
for model training. Fetches from OpenAQ (free, no key required).

Usage:
    # Backfill last 90 days (recommended before training)
    py -3.11 backfill.py --days 90

    # Backfill a specific date range
    py -3.11 backfill.py --start 2025-12-01 --end 2026-03-01

    # Dry run (fetch and print without pushing to Hopsworks)
    py -3.11 backfill.py --days 7 --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from wsgiref import headers

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
CITY_NAME         = os.getenv("CITY_NAME", "islamabad")
CITY_LAT          = float(os.getenv("CITY_LAT", 33.7235))
CITY_LON          = float(os.getenv("CITY_LON", 73.11822))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Checkpoint file (saves progress so crashes don't lose work) ───────────────
CHECKPOINT_FILE = Path(__file__).resolve().parent.parent / "data" / "backfill_checkpoint.json"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OPENAQ HISTORICAL FETCHER
# OpenAQ is free, open, no API key needed.
# Islamabad US Embassy station ID on OpenAQ: we'll search by coordinates.
# ══════════════════════════════════════════════════════════════════════════════

OPENAQ_BASE = "https://api.openaq.org/v3"

def fetch_openaq_historical(location_id: int, date: datetime) -> dict:
    """
    Fetch daily PM2.5 measurements from OpenAQ sensor endpoint.
    Uses Islamabad AirNow sensor ID 1343270.
    """
    SENSOR_ID = 1343270
    date_from = date.replace(hour=0,  minute=0, second=0, microsecond=0)
    date_to   = date.replace(hour=23, minute=59, second=59, microsecond=0)

    url = f"{OPENAQ_BASE}/sensors/{SENSOR_ID}/measurements"
    params = {
        "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit":     100,
    }
    headers = {"X-API-Key": os.getenv("OPENAQ_API_KEY", "")}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            return {}

        pm25_values = [m["value"] for m in results
                       if m.get("value") is not None and m["value"] >= 0]
        if not pm25_values:
            return {}

        pm25 = round(np.mean(pm25_values), 2)
        aqi  = pm25_to_aqi(pm25)

        return {
            "aqi":  aqi,
            "pm25": pm25,
            "pm10": None,
            "no2":  None,
            "o3":   None,
            "so2":  None,
            "co":   None,
        }
    except Exception as e:
        log.warning(f"OpenAQ fetch failed for {date.date()}: {e}")
        return {}

def pm25_to_aqi(pm25: float) -> float:
    """
    Convert PM2.5 concentration (µg/m³) to AQI using US EPA linear interpolation.

    Breakpoints from: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    """
    # (C_low, C_high, AQI_low, AQI_high)
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
            aqi = (aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo
            return round(aqi, 1)
    return 500.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIMULATED WEATHER FOR HISTORICAL DATES
# OpenWeather historical API requires a paid plan.
# We use a seasonal approximation for Islamabad based on climatological averages.
# This is clearly documented as approximate — good enough for ML training features.
# ══════════════════════════════════════════════════════════════════════════════

# Monthly climate averages for Islamabad (temp °C, humidity %, wind m/s, pressure hPa)
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

def get_approximate_weather(dt: datetime) -> dict:
    """
    Return approximate weather for a historical date using Islamabad
    climatological monthly averages with small random noise for realism.

    This is clearly an approximation — used only when historical API
    data is unavailable. Documented in the project report.
    """
    base = ISLAMABAD_CLIMATE[dt.month]
    rng  = np.random.default_rng(seed=int(dt.timestamp()))  # deterministic per date

    return {
        "temperature":    round(base["temp"]     + rng.normal(0, 2.0), 2),
        "humidity":       int(np.clip(base["humidity"]  + rng.integers(-8, 8), 10, 100)),
        "pressure":       int(np.clip(base["pressure"]  + rng.integers(-5, 5), 980, 1030)),
        "wind_speed":     round(max(0, base["wind"]     + rng.normal(0, 0.8)), 2),
        "wind_direction": int(rng.integers(0, 360)),
        "visibility":     int(rng.choice([5000, 8000, 10000, 10000, 10000])),
        "weather_desc":   "historical_approximate",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING (reused from feature_pipeline.py)
# Duplicated here to keep backfill.py self-contained and runnable independently.
# ══════════════════════════════════════════════════════════════════════════════

def engineer_time_features(dt: datetime) -> dict:
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


def engineer_weather_interaction_features(weather: dict) -> dict:
    temp     = weather["temperature"]
    humidity = weather["humidity"]
    wind     = weather["wind_speed"]
    pressure = weather["pressure"]
    return {
        "temp_humidity_index":   round(temp * humidity / 100, 4),
        "wind_dispersion":       round(wind ** 1.5, 4),
        "atmospheric_stability": round(pressure / (wind + 0.1), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHECKPOINT LOGIC
# Saves processed dates so we can resume after crashes without re-processing.
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> set:
    """Load set of already-processed date strings (YYYY-MM-DD)."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        log.info(f"Checkpoint loaded: {len(data['processed'])} dates already done")
        return set(data["processed"])
    return set()


def save_checkpoint(processed: set) -> None:
    """Save the set of processed dates to disk."""
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": sorted(processed)}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN BACKFILL ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_row(dt: datetime, aqi_data: dict, weather: dict, prev_aqi: float) -> dict:
    """Assemble one complete feature row from raw data + engineering."""
    time_feats        = engineer_time_features(dt)
    interaction_feats = engineer_weather_interaction_features(weather)

    aqi = aqi_data.get("aqi")
    if aqi is None:
        return None   # skip rows with no AQI value

    prev = prev_aqi if prev_aqi and not np.isnan(prev_aqi) else aqi
    change_rate = round((aqi - prev) / prev * 100, 4) if prev != 0 else 0.0

    return {
        "city":                  CITY_NAME,
        "timestamp":             dt.replace(minute=0, second=0, microsecond=0),
        "aqi":                   float(aqi),
        "pm25":                  float(aqi_data["pm25"]) if aqi_data.get("pm25") is not None else np.nan,
        "pm10":                  float(aqi_data["pm10"]) if aqi_data.get("pm10") is not None else np.nan,
        "no2":                   float(aqi_data["no2"])  if aqi_data.get("no2")  is not None else np.nan,
        "o3":                    float(aqi_data["o3"])   if aqi_data.get("o3")   is not None else np.nan,
        "so2":                   float(aqi_data["so2"])  if aqi_data.get("so2")  is not None else np.nan,
        "co":                    float(aqi_data["co"])   if aqi_data.get("co")   is not None else np.nan,
        "temperature":           weather["temperature"],
        "humidity":              int(weather["humidity"]),
        "pressure":              int(weather["pressure"]),
        "wind_speed":            weather["wind_speed"],
        "wind_direction":        float(weather["wind_direction"]),
        "visibility":            float(weather["visibility"]),
        **time_feats,
        **interaction_feats,
        "aqi_change_rate":       change_rate,
    }


def run_backfill(start_date: datetime, end_date: datetime, dry_run: bool = False):
    """
    Main backfill loop.

    For each date in range:
    1. Check checkpoint — skip if already processed
    2. Fetch AQI from OpenAQ
    3. Get approximate weather from climate averages
    4. Engineer features
    5. Batch insert into Hopsworks every 10 days
    """
    log.info(f"Backfill range: {start_date.date()} → {end_date.date()}")
    log.info(f"Dry run: {dry_run}")

    # ── Connect to Hopsworks ──────────────────────────────────────────────────
    if not dry_run:
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
        online_enabled=False,   # Changed from True
        )

    # ── Find OpenAQ station ───────────────────────────────────────────────────
    location_id = 233470
    # ── Load checkpoint ───────────────────────────────────────────────────────
    processed = load_checkpoint()

    # ── Generate list of dates ────────────────────────────────────────────────
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)

    total       = len(all_dates)
    skipped     = 0
    failed      = 0
    rows_batch  = []
    prev_aqi    = np.nan

    log.info(f"Total dates to process: {total}")

    for i, date in enumerate(all_dates, 1):
        date_str = date.strftime("%Y-%m-%d")

        # Skip already processed dates
        if date_str in processed:
            skipped += 1
            continue

        log.info(f"[{i}/{total}] Processing {date_str}...")

        try:
            # Fetch AQI data
            if location_id:
                aqi_data = fetch_openaq_historical(location_id, date)
            else:
                aqi_data = {}

            # If OpenAQ has no data for this date, skip gracefully
            if not aqi_data or aqi_data.get("aqi") is None:
                log.warning(f"  No AQI data available for {date_str} — skipping")
                processed.add(date_str)   # mark as processed to avoid retrying
                failed += 1
                continue

            # Get weather (approximate for historical)
            weather = get_approximate_weather(date)

            # Build feature row
            row = build_feature_row(date, aqi_data, weather, prev_aqi)
            if row is None:
                log.warning(f"  Could not build feature row for {date_str}")
                failed += 1
                continue

            prev_aqi = row["aqi"]
            rows_batch.append(row)
            processed.add(date_str)

            log.info(f"  AQI={row['aqi']}, PM2.5={row['pm25']:.1f}, Temp={row['temperature']}°C")

            # ── Batch insert every 10 rows ────────────────────────────────────
            if len(rows_batch) >= 10 and not dry_run:
                df_batch = pd.DataFrame(rows_batch)
                nullable_cols = ['pm10', 'no2', 'o3', 'so2', 'co', 'wind_direction', 'visibility']
                for col in nullable_cols:
                    if col in df_batch.columns:
                        df_batch[col] = df_batch[col].astype('float64')

                fg.insert(df_batch, write_options={"wait_for_job": False, "use_kafka": False})
                log.info(f"  Batch of {len(rows_batch)} rows pushed to feature store")
                rows_batch = []
                save_checkpoint(processed)

            # Be respectful to OpenAQ API — small delay between requests
            time.sleep(0.5)

        except KeyboardInterrupt:
            log.info("Interrupted by user — saving checkpoint...")
            save_checkpoint(processed)
            sys.exit(0)

        except Exception as e:
            log.error(f"  Failed for {date_str}: {e}")
            failed += 1
            continue

    # ── Push any remaining rows ───────────────────────────────────────────────
    if rows_batch and not dry_run:
        df_batch = pd.DataFrame(rows_batch)
        nullable_cols = ['pm10', 'no2', 'o3', 'so2', 'co', 'wind_direction', 'visibility']
        for col in nullable_cols:
            if col in df_batch.columns:
                df_batch[col] = df_batch[col].astype('float64')

        fg.insert(df_batch, write_options={"wait_for_job": False})
        log.info(f"Final batch of {len(rows_batch)} rows pushed.")

    # ── Save final checkpoint ─────────────────────────────────────────────────
    save_checkpoint(processed)

    # ── Summary ───────────────────────────────────────────────────────────────
    successful = total - skipped - failed
    log.info("=" * 50)
    log.info(f"Backfill complete.")
    log.info(f"  Total dates:  {total}")
    log.info(f"  Pushed:       {successful}")
    log.info(f"  Skipped:      {skipped} (already in checkpoint)")
    log.info(f"  Failed:       {failed} (no data available)")
    log.info("=" * 50)

    if dry_run:
        log.info("DRY RUN — nothing was pushed to Hopsworks.")
        if rows_batch:
            print("\nSample rows that would be pushed:")
            print(pd.DataFrame(rows_batch).head(3).T.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Historical Backfill")
    parser.add_argument("--days",    type=int, default=90,
                        help="Number of past days to backfill (default: 90)")
    parser.add_argument("--start",  type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end",    type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and print data without pushing to Hopsworks")
    args = parser.parse_args()

    # Determine date range
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    end_date = end_date.replace(hour=12, minute=0, second=0, microsecond=0)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_date = end_date - timedelta(days=args.days)

    run_backfill(start_date, end_date, dry_run=args.dry_run)