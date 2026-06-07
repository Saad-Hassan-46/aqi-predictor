from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import hopsworks

# =========================
# Load Environment Variables
# =========================
load_dotenv(
    dotenv_path=Path(r"D:\Internship Project\aqi-predictor") / ".env"
)

# =========================
# Connect to Hopsworks
# =========================
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)

# =========================
# Access Feature Store
# =========================
fs = project.get_feature_store()

# =========================
# Read Feature Group
# =========================
fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

df = fg.read()

print("\n" + "=" * 60)
print("FULL DATASET ANALYSIS")
print("=" * 60)

print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nAQI Summary:")
print(f"Unique AQI Values: {df['aqi'].nunique()}")
print(f"Minimum AQI: {df['aqi'].min()}")
print(f"Maximum AQI: {df['aqi'].max()}")

# =========================
# Sort by Time
# =========================
df = df.sort_values("timestamp")

# =========================
# Create Test Set (Last 30 Days)
# =========================
cutoff = df["timestamp"].max() - pd.Timedelta(days=30)
test = df[df["timestamp"] > cutoff]

print("\n" + "=" * 60)
print("TEST SET ANALYSIS (LAST 30 DAYS)")
print("=" * 60)

print(f"Test Set Rows: {len(test)}")
print(f"Test Set AQI Unique Values: {test['aqi'].nunique()}")

# =========================
# AQI Statistics
# =========================
print("\nAQI Statistical Summary:")
print(test["aqi"].describe())

# =========================
# AQI Distribution
# =========================
print("\nAQI Value Counts:")
print(test["aqi"].value_counts().sort_index())

# =========================
# Most Common AQI Values
# =========================
print("\nTop 20 Most Frequent AQI Values:")
print(test["aqi"].value_counts().head(20))

# =========================
# Missing Values
# =========================
print("\nMissing Values:")
missing = test.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

# =========================
# Duplicate Rows
# =========================
print("\nDuplicate Rows:")
print(test.duplicated().sum())

# =========================
# Timestamp Range
# =========================
print("\nTimestamp Range:")
print(f"Start: {test['timestamp'].min()}")
print(f"End  : {test['timestamp'].max()}")

# =========================
# First and Last Few Records
# =========================
print("\nFirst 10 Test Records:")
print(test[["timestamp", "aqi"]].head(10).to_string(index=False))

print("\nLast 10 Test Records:")
print(test[["timestamp", "aqi"]].tail(10).to_string(index=False))

# =========================
# Full Timestamp + AQI Table
# =========================
print("\nFull Test Set (Timestamp + AQI):")
print(test[["timestamp", "aqi"]].to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)