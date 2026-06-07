# Internship Roadmap

Deadline context: the final deadline is **June 8, 2026**, so this project should now be treated as a finalization and polish sprint, not a long discovery project.

## Phase 1: Stabilize The Repository

**Objective:** make the project installable, testable, and safe to share.

**Learn:** virtual environments, dependency files, environment variables, GitHub secrets, pytest basics.

**Deliverables:**
- Working `.env.example`.
- Passing unit tests.
- No hardcoded secrets.
- README with exact run commands.
- CI workflow that runs tests.

**Avoid:** committing `.env`, relying on notebooks only, leaving empty placeholder files.

## Phase 2: Data Collection And Feature Store

**Objective:** automate live AQI/weather feature ingestion.

**Learn:** REST APIs, JSON parsing, AQI vs pollutant concentration, Hopsworks feature groups, primary keys, event time.

**Deliverables:**
- Hourly `feature_pipeline.py`.
- Historical `csv_backfill.py` or `backfill.py`.
- Validated feature schema.
- Hopsworks feature group named `aqi_features`.

**Avoid:** duplicate timestamps, unbounded API calls, mixing live and historical schemas.

## Phase 3: Feature Engineering

**Objective:** turn raw readings into forecast-ready features.

**Learn:** cyclical time encoding, lag features, rolling means/std, leakage prevention, weather interactions.

**Deliverables:**
- Shared feature engineering functions.
- Tests proving lags and targets are shifted correctly.
- Feature list stored with the trained model.

**Avoid:** using future AQI in input features, random train/test split, inconsistent features between training and inference.

## Phase 4: Model Training And Evaluation

**Objective:** compare baselines and choose a defensible model.

**Learn:** regression metrics, time-series validation, Random Forest, Gradient Boosting/XGBoost, Ridge baseline, optional LSTM.

**Deliverables:**
- Model comparison table with RMSE, MAE, and R2.
- Best model artifact.
- Imputer and feature column artifact.
- Model registry entry.

**Avoid:** reporting only accuracy, training on test data, ignoring simple baselines.

## Phase 5: Dashboard And Explainability

**Objective:** present predictions in a way a non-technical user can understand.

**Learn:** Streamlit, Plotly, AQI category bands, SHAP feature attribution, alert thresholds.

**Deliverables:**
- Current AQI card.
- 3-day forecast cards and chart.
- Pollutant breakdown.
- Hazard alert.
- SHAP/feature importance section.

**Avoid:** dashboard text that overexplains implementation, broken app when Hopsworks is unavailable, unlabeled units.

## Phase 6: Automation And Deployment

**Objective:** show end-to-end MLOps maturity.

**Learn:** GitHub Actions schedules, workflow secrets, Streamlit Community Cloud, monitoring failures.

**Deliverables:**
- Hourly feature workflow.
- Daily training workflow.
- CI test workflow.
- Deployed Streamlit URL in README.

**Avoid:** putting API keys in workflow YAML, no manual workflow trigger, no smoke test before deployment.

## Optional Resume Boosters

- Add model drift monitoring: compare recent prediction errors with training RMSE.
- Add FastAPI endpoint `/predict` for backend serving.
- Add a model card in `docs/model_card.md`.
- Add data quality report with missingness and outlier charts.
- Add multi-city support with a city selector.
- Add confidence intervals using quantile regression or bootstrapped tree predictions.
