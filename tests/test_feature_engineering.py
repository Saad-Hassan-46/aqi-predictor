from datetime import datetime, timezone

import pandas as pd

from pipelines.feature_engineering import (
    add_daily_lag_features,
    compute_aqi_change_rate,
    engineer_time_features,
    engineer_weather_interaction_features,
)


def test_engineer_time_features_uses_cyclical_columns():
    features = engineer_time_features(datetime(2026, 6, 8, 6, tzinfo=timezone.utc))

    assert features["hour"] == 6
    assert features["month"] == 6
    assert features["season"] == 2
    assert features["hour_sin"] == 1.0
    assert abs(features["hour_cos"]) < 0.000001


def test_engineer_weather_interactions_are_deterministic():
    features = engineer_weather_interaction_features(
        {"temperature": 30, "humidity": 60, "wind_speed": 4, "pressure": 1000}
    )

    assert features["temp_humidity_index"] == 18.0
    assert features["wind_dispersion"] == 8.0
    assert features["atmospheric_stability"] == round(1000 / 4.1, 4)


def test_compute_aqi_change_rate_handles_missing_previous_value():
    assert compute_aqi_change_rate(120, None) == 0.0
    assert compute_aqi_change_rate(120, 100) == 20.0


def test_add_daily_lag_features_creates_targets_without_leakage():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
            "aqi": [100, 110, 120, 130, 140, 150],
            "temperature": [20, 21, 22, 23, 24, 25],
            "humidity": [40, 41, 42, 43, 44, 45],
        }
    )

    out = add_daily_lag_features(df, lookback_days=3)

    assert "aqi_lag_1d" in out.columns
    assert "target_aqi_1d" in out.columns
    assert out.iloc[0]["aqi"] == 110
    assert out.iloc[0]["aqi_lag_1d"] == 100
    assert out.iloc[0]["target_aqi_1d"] == 120
