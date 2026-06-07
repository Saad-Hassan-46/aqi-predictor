from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(r'D:\Internship Project\aqi-predictor') / '.env')

import hopsworks

project = hopsworks.login(
    api_key_value=os.getenv('HOPSWORKS_API_KEY'),
    project=os.getenv('HOPSWORKS_PROJECT_NAME')
)

fs = project.get_feature_store()
fg = fs.get_feature_group('aqi_features', version=1)

df = fg.read()
df = df.sort_values('timestamp')

print('AQI stats:')
print(df['aqi'].describe())

print()
print('Unique AQI values:', df['aqi'].nunique())

print()
print('AQI by year-month:')
df['ym'] = df['timestamp'].dt.to_period('M')
print(df.groupby('ym')['aqi'].agg(['mean', 'min', 'max', 'std']).to_string())