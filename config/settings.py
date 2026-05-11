import os
from dotenv import load_dotenv

load_dotenv()

# API keys
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")

# Location
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")
CITY_LAT = float(os.getenv("CITY_LAT", 33.5651))
CITY_LON = float(os.getenv("CITY_LON", 73.0169))

# Feature store config
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_forecaster"

# Model training
FORECAST_HORIZON = 3        # days ahead to predict
LOOKBACK_HOURS = 72         # hours of history as input features
TEST_SIZE_DAYS = 14         # hold-out test set size
RANDOM_STATE = 42