from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "city",
    "timestamp",
    "aqi",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
}


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Feature frame is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    if out["timestamp"].isna().any():
        raise ValueError("One or more timestamps could not be parsed.")
    if out["aqi"].isna().any():
        raise ValueError("AQI value is missing.")
    if not out["aqi"].between(0, 500).all():
        raise ValueError("AQI must be between 0 and 500.")
    if not out["humidity"].between(0, 100).all():
        raise ValueError("Humidity must be between 0 and 100.")
    if not out["temperature"].between(-90, 70).all():
        raise ValueError("Temperature is outside a realistic range.")
    if (out["pressure"] <= 0).any():
        raise ValueError("Pressure must be positive.")
    if (out["wind_speed"] < 0).any():
        raise ValueError("Wind speed cannot be negative.")

    return out
