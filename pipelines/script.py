import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(r"D:\Internship Project\aqi-predictor") / ".env")

key = os.getenv("OPENAQ_API_KEY")
print("Key:", repr(key))

r = requests.get(
    "https://api.openaq.org/v3/locations",
    params={
        "coordinates": "33.7235,73.11822",
        "radius": 25000,
        "limit": 10
    },
    headers={"X-API-Key": key}
)

print("Status:", r.status_code)
print(r.text)