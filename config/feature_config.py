RAW_POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "o3", "so2", "co"]

RAW_WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "visibility",
]

TIME_FEATURE_COLUMNS = [
    "hour",
    "day",
    "month",
    "weekday",
    "season",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
]

INTERACTION_FEATURE_COLUMNS = [
    "temp_humidity_index",
    "wind_dispersion",
    "atmospheric_stability",
    "aqi_change_rate",
]

LAG_FEATURE_COLUMNS = [
    "aqi_lag_1d",
    "aqi_lag_2d",
    "aqi_lag_3d",
    "aqi_lag_7d",
    "aqi_rolling_mean_3d",
    "aqi_rolling_mean_7d",
    "aqi_rolling_mean_14d",
    "aqi_rolling_std_7d",
    "temp_rolling_mean_3d",
    "humidity_rolling_mean_3d",
]

TARGET_COLUMNS = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]

FEATURE_STORE_COLUMNS = (
    ["city", "timestamp", "aqi"]
    + RAW_POLLUTANT_COLUMNS
    + RAW_WEATHER_COLUMNS
    + TIME_FEATURE_COLUMNS
    + INTERACTION_FEATURE_COLUMNS
)
