from pathlib import Path
from dotenv import load_dotenv
import os
import hopsworks

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")

if not api_key:
    raise RuntimeError("HOPSWORKS_API_KEY is missing. Add it to .env or GitHub Actions secrets.")

project = hopsworks.login(
    api_key_value=api_key,
    project=project_name,
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    "aqi_features",
    version=1
)

df = fg.read()

print("Total rows:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

print(
    df[["timestamp", "aqi", "pm25", "temperature"]]
    .sort_values("timestamp")
    .head(10)
)
