from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


SEASON_BY_MONTH = {
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
}


def engineer_time_features(dt: datetime) -> dict:
    hour = dt.hour
    month = dt.month
    weekday = dt.weekday()

    return {
        "hour": hour,
        "day": dt.day,
        "month": month,
        "weekday": weekday,
        "season": SEASON_BY_MONTH[month],
        "is_weekend": int(weekday >= 5),
        "hour_sin": round(float(np.sin(2 * np.pi * hour / 24)), 6),
        "hour_cos": round(float(np.cos(2 * np.pi * hour / 24)), 6),
        "month_sin": round(float(np.sin(2 * np.pi * month / 12)), 6),
        "month_cos": round(float(np.cos(2 * np.pi * month / 12)), 6),
        "weekday_sin": round(float(np.sin(2 * np.pi * weekday / 7)), 6),
        "weekday_cos": round(float(np.cos(2 * np.pi * weekday / 7)), 6),
    }


def engineer_weather_interaction_features(weather: dict) -> dict:
    temp = float(weather["temperature"])
    humidity = float(weather["humidity"])
    wind = float(weather["wind_speed"])
    pressure = float(weather["pressure"])

    return {
        "temp_humidity_index": round(temp * humidity / 100, 4),
        "wind_dispersion": round(wind ** 1.5, 4),
        "atmospheric_stability": round(pressure / (wind + 0.1), 4),
    }


def compute_aqi_change_rate(current_aqi: float, previous_aqi: float | None) -> float:
    if previous_aqi is None or pd.isna(previous_aqi) or previous_aqi == 0:
        return 0.0
    return round((float(current_aqi) - float(previous_aqi)) / float(previous_aqi) * 100, 4)


def add_daily_lag_features(df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    required = {"timestamp", "aqi", "temperature", "humidity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for lag engineering: {sorted(missing)}")

    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    for lag in [1, 2, 3, 7, 14]:
        if lag <= lookback_days:
            out[f"aqi_lag_{lag}d"] = out["aqi"].shift(lag)

    out["aqi_rolling_mean_3d"] = out["aqi"].shift(1).rolling(3, min_periods=1).mean()
    out["aqi_rolling_mean_7d"] = out["aqi"].shift(1).rolling(7, min_periods=1).mean()
    out["aqi_rolling_mean_14d"] = out["aqi"].shift(1).rolling(14, min_periods=1).mean()
    out["aqi_rolling_std_7d"] = out["aqi"].shift(1).rolling(7, min_periods=2).std().fillna(0)
    out["temp_rolling_mean_3d"] = out["temperature"].shift(1).rolling(3, min_periods=1).mean()
    out["humidity_rolling_mean_3d"] = out["humidity"].shift(1).rolling(3, min_periods=1).mean()
    out["target_aqi_1d"] = out["aqi"].shift(-1)
    out["target_aqi_2d"] = out["aqi"].shift(-2)
    out["target_aqi_3d"] = out["aqi"].shift(-3)

    critical = ["aqi_lag_1d", "aqi_rolling_mean_7d", "target_aqi_1d"]
    return out.dropna(subset=critical).reset_index(drop=True)
