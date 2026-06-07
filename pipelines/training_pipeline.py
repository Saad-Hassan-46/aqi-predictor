"""
training_pipeline.py
---------------------
Fetches features from Hopsworks, engineers lag features, trains and
compares multiple ML models, and registers the best one in the
Hopsworks Model Registry.

Run manually:
    py -3.11 training_pipeline.py

Run with specific lookback window:
    py -3.11 training_pipeline.py --lookback 14
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

<<<<<<< HEAD
try:
    from pipelines.feature_engineering import add_daily_lag_features
except ModuleNotFoundError:
    from feature_engineering import add_daily_lag_features

=======
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
warnings.filterwarnings("ignore")

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR    = Path(__file__).resolve().parent.parent / "models"
TEST_DAYS    = 365
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FETCH FEATURES FROM HOPSWORKS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_features(project) -> pd.DataFrame:
    """
    Read all rows from the aqi_features feature group.
    Returns a DataFrame sorted by timestamp ascending.
    """
    log.info("Fetching features from Hopsworks feature store...")
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Fetched {len(df)} rows spanning {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LAG FEATURE ENGINEERING
# Computed here (not in feature_pipeline.py) because they require the full
# dataset to look back across multiple rows.
# ══════════════════════════════════════════════════════════════════════════════

def engineer_lag_features(df: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """
    Add lag and rolling average features to the dataset.

    Why lag features matter for AQI forecasting:
    - Yesterday's AQI is the single strongest predictor of today's AQI
    - Rolling averages smooth out noise and capture trends
    - Change rates capture whether air quality is improving or worsening

    Args:
        df: Feature DataFrame sorted by timestamp
        lookback_days: Number of past days to create lag features for

    Returns:
        DataFrame with lag features added, NaN rows dropped
    """
    log.info(f"Engineering lag features (lookback={lookback_days} days)...")
<<<<<<< HEAD
    df = add_daily_lag_features(df, lookback_days=lookback_days)
=======
    df = df.copy()

    # Lag features: AQI values from past N days
    for lag in [1, 2, 3, 7, 14]:
        if lag <= lookback_days:
            df[f"aqi_lag_{lag}d"] = df["aqi"].shift(lag)

    # Rolling statistics
    df["aqi_rolling_mean_3d"]  = df["aqi"].shift(1).rolling(window=3,  min_periods=1).mean()
    df["aqi_rolling_mean_7d"]  = df["aqi"].shift(1).rolling(window=7,  min_periods=1).mean()
    df["aqi_rolling_mean_14d"] = df["aqi"].shift(1).rolling(window=14, min_periods=1).mean()
    df["aqi_rolling_std_7d"]   = df["aqi"].shift(1).rolling(window=7,  min_periods=2).std().fillna(0)

    # Rolling weather features
    df["temp_rolling_mean_3d"]     = df["temperature"].shift(1).rolling(window=3, min_periods=1).mean()
    df["humidity_rolling_mean_3d"] = df["humidity"].shift(1).rolling(window=3,    min_periods=1).mean()

    # Targets: AQI 1, 2, 3 days ahead
    # Train on 1-day-ahead; dashboard chains 3 predictions for 3-day forecast
    df["target_aqi_1d"] = df["aqi"].shift(-1)
    df["target_aqi_2d"] = df["aqi"].shift(-2)
    df["target_aqi_3d"] = df["aqi"].shift(-3)

    # Drop rows with NaN in critical columns
    required_cols = ["aqi_lag_1d", "aqi_rolling_mean_7d", "target_aqi_1d"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
    log.info(f"After lag engineering: {len(df)} rows, {len(df.columns)} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN / TEST SPLIT
# Never use random split for time series — always split by time.
# ══════════════════════════════════════════════════════════════════════════════

def time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """
    Split DataFrame into train and test sets by time.
    The last test_days days form the test set.
    """
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train  = df[df["timestamp"] <= cutoff].copy()
    test   = df[df["timestamp"] >  cutoff].copy()
    log.info(f"Train: {len(train)} rows ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
    log.info(f"Test:  {len(test)} rows  ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return input feature columns — exclude identity and target columns."""
    exclude = {
        "city", "timestamp", "aqi",
        "target_aqi_1d", "target_aqi_2d", "target_aqi_3d",
        "weather_desc",
    }
    return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Compute RMSE, MAE, and R2 for a fitted model."""
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    log.info(f"  {name:30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2, "preds": preds}


def train_all_models(X_train, y_train, X_test, y_test) -> list:
    """
    Train and evaluate all models.
    Returns list of result dicts sorted by RMSE ascending.
    """
    results = []

    # 1. Ridge Regression (linear baseline)
    log.info("Training Ridge Regression...")
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    ridge.fit(X_train, y_train)
    results.append(evaluate_model(ridge, X_test, y_test, "Ridge Regression"))

    # 2. Random Forest
    log.info("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    # 3. Gradient Boosting
    log.info("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    results.append(evaluate_model(gb, X_test, y_test, "Gradient Boosting"))

    # 4. XGBoost
    try:
        from xgboost import XGBRegressor
        log.info("Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model(xgb, X_test, y_test, "XGBoost"))
    except ImportError:
        log.warning("XGBoost not installed. Run: pip install xgboost")

    results.sort(key=lambda x: x["rmse"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def register_best_model(project, best: dict, feature_cols: list, metrics: dict):
    """
    Save the best model locally and register it in Hopsworks Model Registry.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    model_path    = MODEL_DIR / "best_model.pkl"
    features_path = MODEL_DIR / "feature_columns.pkl"

    joblib.dump(best["model"], model_path)
    joblib.dump(feature_cols, features_path)
    log.info(f"Model saved locally to {model_path}")

    mr = project.get_model_registry()

    try:
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        # With this:
        input_example_df  = pd.DataFrame([dict(zip(feature_cols, [0.0] * len(feature_cols)))])
        output_example_df = pd.DataFrame([{"aqi_forecast": 0.0}])
        input_schema  = Schema(input_example_df)
        output_schema = Schema(output_example_df)
        model_schema  = ModelSchema(input_schema=input_schema, output_schema=output_schema)

        hw_model = mr.sklearn.create_model(
            name="aqi_forecaster",
            metrics=metrics,
            model_schema=model_schema,
            description=f"Best model: {best['name']} | RMSE={metrics['rmse']:.2f}",
            input_example=[0.0] * len(feature_cols),
        )
        hw_model.save(str(MODEL_DIR))
        log.info("Model registered in Hopsworks Model Registry as 'aqi_forecaster'")
    except Exception as e:
        log.warning(f"Could not register in Hopsworks Model Registry: {e}")
        log.info("Model is still saved locally — inference pipeline will use local file.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pipeline(lookback_days: int = 7):
    """
    Full training pipeline:
    1. Fetch features from Hopsworks
    2. Engineer lag features
    3. Time-based train/test split
    4. Train and compare all models
    5. Register best model in Model Registry
    """

    # Connect to Hopsworks
    log.info("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    # Fetch and prepare data
    df = fetch_features(project)

    if len(df) < 50:
        log.error(f"Not enough data to train ({len(df)} rows). Run backfill first.")
        sys.exit(1)

    df = engineer_lag_features(df, lookback_days=lookback_days)

    # Split
    train_df, test_df = time_based_split(df)
    feature_cols = get_feature_columns(df)
    target_col   = "target_aqi_1d"

    # Impute NaN values with column median before training
    # NaNs come from missing pollutants (pm10, no2, o3, so2, co)
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(train_df[feature_cols])
    X_test  = imputer.transform(test_df[feature_cols])
    y_train = train_df[target_col].values
    y_test  = test_df[target_col].values

<<<<<<< HEAD
    MODEL_DIR.mkdir(exist_ok=True)
=======
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
    # Save imputer alongside model for use in inference
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")

    log.info(f"Features: {len(feature_cols)} columns")
    log.info(f"Training on {len(X_train)} rows, evaluating on {len(X_test)} rows")

    # Train all models
    log.info("\n── Model Comparison ─────────────────────────────────")
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Print final rankings
    log.info("\n── Final Rankings ───────────────────────────────────")
    log.info(f"{'Rank':<6}{'Model':<30}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
    log.info("-" * 62)
    for i, r in enumerate(results, 1):
        log.info(f"{i:<6}{r['name']:<30}{r['rmse']:>8.2f}{r['mae']:>8.2f}{r['r2']:>8.4f}")

    best = results[0]
    log.info(f"\nBest model: {best['name']} (RMSE={best['rmse']:.2f})")

    # Register best model
    metrics = {
        "rmse": round(best["rmse"], 4),
        "mae":  round(best["mae"],  4),
        "r2":   round(best["r2"],   4),
    }
    register_best_model(project, best, feature_cols, metrics)

    log.info("\nTraining pipeline complete.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Training Pipeline")
    parser.add_argument(
        "--lookback", type=int, default=7,
        help="Number of past days for lag features (default: 7)"
    )
    args = parser.parse_args()

    try:
        import xgboost
    except ImportError:
        log.info("Installing xgboost...")
        os.system(f"{sys.executable} -m pip install xgboost")

<<<<<<< HEAD
    run_training_pipeline(lookback_days=args.lookback)
=======
    run_training_pipeline(lookback_days=args.lookback)
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
