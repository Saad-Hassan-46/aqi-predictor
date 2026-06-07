from pathlib import Path
from dotenv import load_dotenv
import os
import hopsworks

<<<<<<< HEAD
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")

if not api_key:
    raise RuntimeError("HOPSWORKS_API_KEY is missing. Add it to .env or GitHub Actions secrets.")

project = hopsworks.login(
    api_key_value=api_key,
    project=project_name,
=======
load_dotenv(dotenv_path=Path(r"D:\Internship Project\aqi-predictor") / ".env")

project = hopsworks.login(
    api_key_value=os.getenv("PDNDYtjehc4whxLP.0HAPAs4ZShvlnuUeyJmSuw4xwPBN3pfwCdhqVYaKspUMdwHlUg0NopCBRNclB6au"),
    project=os.getenv("aqi_predictor_model")
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
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
<<<<<<< HEAD
)
=======
)
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
