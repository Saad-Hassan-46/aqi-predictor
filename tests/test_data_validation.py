import pandas as pd
import pytest

from pipelines.data_validation import validate_feature_frame


def valid_frame():
    return pd.DataFrame(
        [
            {
                "city": "Rawalpindi",
                "timestamp": "2026-06-08T00:00:00Z",
                "aqi": 120,
                "temperature": 33,
                "humidity": 45,
                "pressure": 1004,
                "wind_speed": 2.5,
            }
        ]
    )


def test_validate_feature_frame_accepts_valid_data():
    out = validate_feature_frame(valid_frame())

    assert str(out.loc[0, "timestamp"].tz) == "UTC"


def test_validate_feature_frame_rejects_invalid_aqi():
    df = valid_frame()
    df.loc[0, "aqi"] = 900

    with pytest.raises(ValueError, match="AQI"):
        validate_feature_frame(df)


def test_validate_feature_frame_rejects_missing_required_column():
    df = valid_frame().drop(columns=["pressure"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_feature_frame(df)
