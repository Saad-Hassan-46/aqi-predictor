# AQI Predictor Architecture

## Goal

Build a 3-day AQI forecasting system for Islamabad/Rawalpindi using automated data collection, feature engineering, model training, model registry, scheduled CI/CD, and a Streamlit dashboard.

## System Flow

1. **Live feature pipeline**
   - Reads AQI from AQICN and weather from OpenWeather.
   - Engineers time, weather interaction, and AQI change-rate features.
   - Writes one hourly feature row to the Hopsworks feature store.

2. **Historical backfill**
   - Uses historical PM2.5 data from the included AQICN CSV and optional OpenAQ fetches.
   - Converts PM2.5 concentration into AQI.
   - Creates historical rows for training and validation.

3. **Training pipeline**
   - Reads feature rows from Hopsworks.
   - Adds lag features, rolling statistics, and one/two/three-day targets.
   - Compares Ridge Regression, Random Forest, Gradient Boosting, and XGBoost.
   - Saves the best local artifacts and attempts registration in Hopsworks Model Registry.

4. **Inference and dashboard**
   - Loads model artifacts from Hopsworks when available.
   - Falls back to local artifacts for demos/offline development.
   - Generates recursive 3-day AQI forecasts.
   - Shows AQI category bands, pollutant summaries, alerts, trend charts, and SHAP explanations.

5. **Automation**
   - GitHub Actions runs tests on pushes/PRs.
   - Feature pipeline is scheduled hourly.
   - Training pipeline is scheduled daily.
   - Streamlit deployment workflow smoke-tests the app entrypoint.

## Production Practices Covered

- Environment-based secrets instead of hardcoded keys.
- Time-based train/test splitting for forecasting.
- Feature store and model registry integration.
- Reproducible CI checks.
- Local fallback for demos and disaster recovery.
- Small, fast tests for feature engineering, validation, and inference formatting.
