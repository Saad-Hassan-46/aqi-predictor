# AQI Predictor

A 3-day Air Quality Index forecasting system using a 100% serverless ML stack.

## Live Demo
[Streamlit App](https://your-app-url.streamlit.app) <!-- update after deployment -->

## Architecture
Raw API data → Feature Pipeline → Hopsworks Feature Store → Training Pipeline → Model Registry → Streamlit Dashboard

## Stack
- **Data**: AQICN API / OpenWeather API
- **Feature store & model registry**: Hopsworks (free tier)
- **Models**: Ridge Regression, Random Forest, XGBoost, LSTM (TensorFlow)
- **Explainability**: SHAP
- **CI/CD**: GitHub Actions (hourly feature pipeline, daily training)
- **Dashboard**: Streamlit Community Cloud

## Setup
```bash
git clone https://github.com/your-username/aqi-predictor
cd aqi-predictor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

## Run pipelines locally
```bash
python pipelines/feature_pipeline.py     # fetch + engineer features
python pipelines/backfill.py             # populate historical data
python pipelines/training_pipeline.py    # train and register model
streamlit run app/app.py                 # launch dashboard
```

## Project structure
See `docs/architecture.md`