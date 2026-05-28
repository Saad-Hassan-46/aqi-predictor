from pathlib import Path
from dotenv import load_dotenv
import os
import hopsworks

load_dotenv(dotenv_path=Path(r"D:\Internship Project\aqi-predictor") / ".env")

project = hopsworks.login(
    api_key_value=os.getenv("PDNDYtjehc4whxLP.0HAPAs4ZShvlnuUeyJmSuw4xwPBN3pfwCdhqVYaKspUMdwHlUg0NopCBRNclB6au"),
    project=os.getenv("aqi_predictor_model")
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