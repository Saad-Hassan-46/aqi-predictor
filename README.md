# AQI Predictor

A 3-day Air Quality Index forecasting system using a 100% serverless ML stack.

## Live Demo
[Streamlit App](https://your-app-url.streamlit.app) <!-- update after deployment -->

## Architecture
Raw API data -> Feature Pipeline -> Hopsworks Feature Store -> Training Pipeline -> Model Registry -> Streamlit Dashboard

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

On Windows, if `python` opens the Microsoft Store, install Python 3.11 from python.org or fix App Execution Aliases, then rerun the setup commands.

## Run pipelines locally
```bash
python pipelines/feature_pipeline.py     # fetch + engineer features
python pipelines/backfill.py             # populate historical data
python pipelines/training_pipeline.py    # train and register model
streamlit run app/app.py                 # launch dashboard
```

## Validate locally
```bash
python -m pytest -q
```

The dashboard can run from Hopsworks or fall back to the included local CSV/model artifacts for demos.

## Project structure
See `docs/architecture.md`

## Internship roadmap
See `docs/ROADMAP.md`

## Learning resources
See `docs/LEARNING_RESOURCES.md`
