"""
training_pipeline.py
---------------------
Reads features from the Hopsworks feature store, engineers lag/rolling
features, trains four model families (Ridge, RandomForest, XGBoost, LSTM),
evaluates them, and registers the best model to the Hopsworks Model Registry.

Forecast task: predict AQI at +24 h, +48 h, and +72 h ahead
(multi-output regression, one row of outputs per input window).

Run manually:
    python pipelines/training_pipeline.py

Force a specific model to be registered regardless of metric:
    python pipelines/training_pipeline.py --force-model xgboost
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb


# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Fix Hopsworks /tmp path issue on Windows
tmpdir = os.getenv("HOPSWORKS_TMPDIR")
if tmpdir:
    tempfile.tempdir = tmpdir

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME            = "aqi_forecaster"
MODEL_VERSION         = 1

LOOKBACK_HOURS  = 72   # input window fed to LSTM; also controls lag depth
FORECAST_HORIZONS = [24, 48, 72]   # hours ahead to predict (3-day forecast)
TEST_SIZE_DAYS  = 14
RANDOM_STATE    = 42

# Raw + engineered feature columns that exist in the feature store
# (we add lag/rolling columns on top of these during training)
BASE_FEATURE_COLS = [
    "aqi", "pm25", "pm10", "no2", "o3", "so2", "co",
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_direction", "visibility",
    "hour", "day", "month", "weekday", "season", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos",
    "temp_humidity_index", "wind_dispersion", "atmospheric_stability",
    "aqi_change_rate",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_features_from_store() -> pd.DataFrame:
    """
    Pull all rows from the aqi_features feature group.

    Returns a DataFrame sorted chronologically with a proper DatetimeIndex.
    """
    log.info("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    log.info(f"Reading feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # Coerce timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING (LAG + ROLLING)
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-window features on top of what the feature pipeline
    already computed.  These are calculated here (not in the feature pipeline)
    because computing them there would require fetching history on every hourly
    run, which is slow and fragile.

    Lags capture short-term autocorrelation; rolling windows capture trend.
    All shifts are in hours (1 row = 1 hour).
    """
    log.info("Engineering lag and rolling features...")

    aqi = df["aqi"]

    # Point-in-time lags (how bad was the air 1h / 6h / 24h / 48h / 72h ago?)
    for lag_h in [1, 6, 24, 48, 72]:
        df[f"aqi_lag_{lag_h}h"] = aqi.shift(lag_h)

    # Rolling statistics (local trend and volatility)
    df["aqi_rolling_mean_6h"]  = aqi.shift(1).rolling(6).mean()
    df["aqi_rolling_mean_24h"] = aqi.shift(1).rolling(24).mean()
    df["aqi_rolling_std_24h"]  = aqi.shift(1).rolling(24).std()
    df["aqi_rolling_mean_7d"]  = aqi.shift(1).rolling(24 * 7).mean()

    # PM2.5 lags (highest-weight pollutant for AQI in South Asia)
    if "pm25" in df.columns:
        for lag_h in [1, 24]:
            df[f"pm25_lag_{lag_h}h"] = df["pm25"].shift(lag_h)

    # Wind speed lag (dispersion signal)
    if "wind_speed" in df.columns:
        df["wind_lag_1h"] = df["wind_speed"].shift(1)

    # Drop rows where lags are undefined (first LOOKBACK_HOURS rows)
    n_before = len(df)
    df = df.dropna(subset=[f"aqi_lag_{LOOKBACK_HOURS}h"])
    log.info(f"  Dropped {n_before - len(df)} rows with undefined lags → {len(df):,} rows remain.")

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift AQI forward to create multi-step prediction targets.

    target_24h = AQI 24 hours from now (shift by -24)
    target_48h = AQI 48 hours from now (shift by -48)
    target_72h = AQI 72 hours from now (shift by -72)

    Rows near the end of the dataset won't have complete targets → drop them.
    """
    log.info("Building prediction targets (future AQI at +24h, +48h, +72h)...")
    for h in FORECAST_HORIZONS:
        df[f"target_{h}h"] = df["aqi"].shift(-h)

    # Drop the last FORECAST_HORIZONS[-1] rows (no future data available)
    df = df.dropna(subset=[f"target_{h}h" for h in FORECAST_HORIZONS])
    log.info(f"  {len(df):,} rows after dropping incomplete targets.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_features_targets(df: pd.DataFrame):
    target_cols  = [f"target_{h}h" for h in FORECAST_HORIZONS]

    # Columns that cause leakage — they are derived directly from AQI
    # and are perfectly correlated with future AQI targets.
    # A model using these is not forecasting — it is copying.
    LEAKY_COLS = [
        "aqi",
        "aqi_lag_1h",
        "aqi_lag_6h",
        "aqi_lag_24h",
        "aqi_lag_48h",        # ← add this
        "aqi_lag_72h",        # ← add this
        "aqi_rolling_mean_6h",
        "aqi_rolling_mean_24h",
        "aqi_rolling_mean_7d",
        "pm25",
        "pm25_lag_1h",
        "pm25_lag_24h",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in target_cols
        and c != "city"
        and c not in LEAKY_COLS
    ]

    X = df[feature_cols].values
    y = df[target_cols].values

    # Hold out the last TEST_SIZE_DAYS as the test set
    split_idx = len(df) - TEST_SIZE_DAYS * 24
    if split_idx < 100:
        log.warning(
            f"Dataset is small ({len(df)} rows). Defaulting to 80/20 split."
        )
        split_idx = int(len(df) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info(
        f"  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows | "
        f"Features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test, feature_cols
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(X_train, y_train, X_test, y_test, scaler):
    """Ridge Regression — strong baseline, interpretable coefficients."""
    import pandas as pd
    import numpy as np

    # Step 1: Drop columns that are entirely NaN
    X_train_df = pd.DataFrame(X_train)
    X_test_df  = pd.DataFrame(X_test)

    all_nan_cols = X_train_df.columns[X_train_df.isnull().all()]
    if len(all_nan_cols) > 0:
        log.info(f"  Dropping {len(all_nan_cols)} fully-NaN columns: {list(all_nan_cols)}")
        X_train_df = X_train_df.drop(columns=all_nan_cols)
        X_test_df  = X_test_df.drop(columns=all_nan_cols)

    X_train = X_train_df.values
    X_test  = X_test_df.values

    # Step 2: Impute any remaining NaNs
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    # Step 3: Re-fit scaler on cleaned data (column count may have changed)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    # Step 4: Train model
    log.info("Training Ridge Regression...")
    model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=RANDOM_STATE))
    model.fit(Xs_train, y_train)
    preds = model.predict(Xs_test)
    metrics = evaluate(y_test, preds, "Ridge")
    return model, metrics, preds


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest — captures non-linearity, feature importance for free."""
    log.info("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    # MultiOutputRegressor wraps RF natively for multi-target output
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, "RandomForest")
    return model, metrics, preds


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost — typically best single-model performance on tabular data."""
    log.info("Training XGBoost...")

    # XGBoost doesn't natively support multi-output → wrap with sklearn
    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
    )
    model.fit(X_train, y_train, **{
        # Pass eval set per-estimator for early stopping awareness
    })
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, "XGBoost")
    return model, metrics, preds


def train_lstm(X_train, y_train, X_test, y_test, n_features: int):
    """
    LSTM — captures long-range temporal dependencies.

    Architecture rationale:
    - Input: (batch, LOOKBACK_HOURS=72, n_features) is ideal for LSTM, but
      the flat tabular format from sklearn means we reshape the lag features
      into a pseudo-sequence.  We use the last 72 rows of X directly as a
      sequence by grouping data into overlapping windows of size 72.
    - Two stacked LSTM layers with dropout prevent overfitting.
    - Output: 3 neurons (one per forecast horizon), linear activation.
    """
    try:
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam
    except ImportError:
        log.warning("TensorFlow not installed. Skipping LSTM.")
        return None, None, None

    log.info("Training LSTM...")

    # ── Reshape flat X into (samples, timesteps, features) ───────────────────
    # We use a sliding window of LOOKBACK_HOURS over the raw data.
    # X_train rows are already offset correctly (each row is one hourly step),
    # so we create windows directly from the sequence.
    SEQ_LEN = min(LOOKBACK_HOURS, X_train.shape[0] // 4)  # safety cap for small datasets

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    Xt_train, yt_train = make_sequences(X_train, y_train, SEQ_LEN)
    Xt_test,  yt_test  = make_sequences(X_test,  y_test,  SEQ_LEN)

    if len(Xt_train) < 10:
        log.warning("Not enough data for LSTM sequences after windowing. Skipping.")
        return None, None, None

    n_timesteps = Xt_train.shape[1]
    n_feat      = Xt_train.shape[2]
    n_outputs   = yt_train.shape[1]

    model = Sequential([
        LSTM(128, input_shape=(n_timesteps, n_feat), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(n_outputs, activation="linear"),   # linear for regression
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    model.fit(
        Xt_train, yt_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    preds = model.predict(Xt_test, verbose=0)
    metrics = evaluate(yt_test, preds, "LSTM")

    # Attach seq_len so the inference pipeline can reconstruct sequences
    model._seq_len = SEQ_LEN
    return model, metrics, preds


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute MAE and RMSE per forecast horizon, plus overall mean MAE.
    Lower is better. MAE is the primary ranking metric (more robust to outliers
    than RMSE, and directly interpretable as average AQI-point error).
    """
    horizon_metrics = {}
    maes = []
    for i, h in enumerate(FORECAST_HORIZONS):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        horizon_metrics[f"mae_{h}h"]  = round(mae, 3)
        horizon_metrics[f"rmse_{h}h"] = round(rmse, 3)
        maes.append(mae)

    mean_mae = round(float(np.mean(maes)), 3)
    horizon_metrics["mean_mae"] = mean_mae

    log.info(
        f"  [{name}] MAE → 24h: {horizon_metrics['mae_24h']:.2f}  "
        f"48h: {horizon_metrics['mae_48h']:.2f}  "
        f"72h: {horizon_metrics['mae_72h']:.2f}  "
        f"| avg: {mean_mae:.2f}"
    )
    return horizon_metrics


def pick_best_model(results: dict) -> str:
    """Return the model name with the lowest mean MAE across horizons."""
    valid = {k: v for k, v in results.items() if v["metrics"] is not None}
    best  = min(valid, key=lambda k: valid[k]["metrics"]["mean_mae"])
    log.info(
        f"\nBest model: {best} "
        f"(mean MAE = {valid[best]['metrics']['mean_mae']:.2f})"
    )
    return best


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def register_model(
    project,
    model_obj,
    model_name_tag: str,
    metrics: dict,
    feature_cols: list,
    scaler: StandardScaler,
    is_lstm: bool = False,
    lstm_seq_len: int = None,
):
    """
    Save model artifacts to a temp dir and push them to the Hopsworks
    Model Registry.

    Artifacts saved:
    - model.pkl          sklearn/xgboost model (joblib)    [or]
    - model.keras        Keras SavedModel format           [LSTM only]
    - scaler.pkl         StandardScaler fitted on X_train  [Ridge only]
    - feature_columns.json  ordered list of input features
    - metrics.json       evaluation results
    - model_info.json    metadata (type, horizons, seq_len)
    """
    import tempfile, shutil
    tmpdir_path = Path(tempfile.mkdtemp())
    log.info(f"Saving model artifacts to {tmpdir_path}...")

    try:
        # ── Save model ────────────────────────────────────────────────────────
        if is_lstm:
            model_path = tmpdir_path / "model.keras"
            model_obj.save(str(model_path))
        else:
            model_path = tmpdir_path / "model.pkl"
            joblib.dump(model_obj, model_path)

        # ── Save scaler ───────────────────────────────────────────────────────
        if scaler is not None:
            joblib.dump(scaler, tmpdir_path / "scaler.pkl")

        # ── Save feature schema ───────────────────────────────────────────────
        (tmpdir_path / "feature_columns.json").write_text(
            json.dumps(feature_cols, indent=2)
        )

        # ── Save metrics ──────────────────────────────────────────────────────
        (tmpdir_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )

        # ── Save model info ───────────────────────────────────────────────────
        model_info = {
            "model_type":       model_name_tag,
            "forecast_horizons": FORECAST_HORIZONS,
            "is_lstm":          is_lstm,
            "lstm_seq_len":     lstm_seq_len,
            "n_features":       len(feature_cols),
            "trained_at":       datetime.now(timezone.utc).isoformat(),
        }
        (tmpdir_path / "model_info.json").write_text(
            json.dumps(model_info, indent=2)
        )

        # ── Push to Hopsworks Model Registry ─────────────────────────────────
        log.info(f"Registering model '{MODEL_NAME}' to Hopsworks Model Registry...")
        mr = project.get_model_registry()

        # get_or_create so re-runs don't fail on duplicate names
        hw_model = mr.sklearn.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            description=(
                f"AQI 3-day forecaster ({model_name_tag}). "
                f"Predicts AQI at +24h, +48h, +72h for Islamabad/Rawalpindi. "
                f"Mean MAE: {metrics['mean_mae']:.2f} AQI points."
            ),
            input_example=np.zeros((1, len(feature_cols))).tolist(),
            model_schema=None,
        )
        hw_model.save(str(tmpdir_path))

        log.info(
            f"Model registered successfully. "
            f"Version: {hw_model.version}  |  Artifacts: {list(tmpdir_path.iterdir())}"
        )
        return hw_model

    finally:
        shutil.rmtree(tmpdir_path, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_training(force_model: str = None):
    """
    End-to-end training run.

    Steps:
    1.  Load features from Hopsworks feature store.
    2.  Add lag / rolling features.
    3.  Build multi-step forecast targets.
    4.  Train / test split (temporal).
    5.  Fit StandardScaler on training set.
    6.  Train all four models.
    7.  Pick the best by mean MAE (or honour --force-model).
    8.  Register winner to Hopsworks Model Registry.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_features_from_store()

    if len(df) < 200:
        log.error(
            f"Only {len(df)} rows in the feature store — need at least 200 to train. "
            "Run backfill.py first."
        )
        sys.exit(1)

    # ── 2. Lag features ───────────────────────────────────────────────────────
    df = add_lag_and_rolling_features(df)

    # ── 3. Targets ────────────────────────────────────────────────────────────
    df = build_targets(df)

    # ── 4. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_cols = split_features_targets(df)

    # ── 5. Scale (fit only on train to prevent leakage) ───────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── 5. Scale (fit only on train to prevent leakage) ───────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── 6. Train all models ───────────────────────────────────────────────────
    results = {}

    ridge_model, ridge_metrics, _ = train_ridge(X_train, y_train, X_test, y_test, scaler)
    results["ridge"] = {"model": ridge_model, "metrics": ridge_metrics, "scaler": scaler, "is_lstm": False}

    rf_model, rf_metrics, _ = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = {"model": rf_model, "metrics": rf_metrics, "scaler": None, "is_lstm": False}

    xgb_model, xgb_metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)
    results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics, "scaler": None, "is_lstm": False}

    lstm_model, lstm_metrics, _ = train_lstm(X_train, y_train, X_test, y_test, n_features=X_train.shape[1])
    if lstm_model is not None:
        results["lstm"] = {
            "model": lstm_model,
            "metrics": lstm_metrics,
            "scaler": None,
            "is_lstm": True,
            "seq_len": getattr(lstm_model, "_seq_len", LOOKBACK_HOURS),
        }

    # ── 7. Pick winner ────────────────────────────────────────────────────────
    if force_model:
        if force_model not in results:
            log.error(f"--force-model '{force_model}' is not in {list(results.keys())}")
            sys.exit(1)
        best_name = force_model
        log.info(f"Forcing model selection: {best_name}")
    else:
        best_name = pick_best_model(results)

    best = results[best_name]

    # Print a summary comparison table
    log.info("\n── Model comparison ──────────────────────────────────────────────")
    log.info(f"{'Model':<16} {'MAE 24h':>8} {'MAE 48h':>8} {'MAE 72h':>8} {'Avg MAE':>8}")
    log.info("─" * 55)
    for name, r in results.items():
        m = r["metrics"]
        if m:
            marker = "  ← best" if name == best_name else ""
            log.info(
                f"{name:<16} {m['mae_24h']:>8.2f} {m['mae_48h']:>8.2f} "
                f"{m['mae_72h']:>8.2f} {m['mean_mae']:>8.2f}{marker}"
            )
    log.info("─" * 55)

    # ── 8. Register winner ────────────────────────────────────────────────────
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    seq_len = best.get("seq_len") if best["is_lstm"] else None

    register_model(
        project=project,
        model_obj=best["model"],
        model_name_tag=best_name,
        metrics=best["metrics"],
        feature_cols=feature_cols,
        scaler=best.get("scaler"),
        is_lstm=best["is_lstm"],
        lstm_seq_len=seq_len,
    )

    log.info("\nPhase 3 complete. Model registered to Hopsworks Model Registry.")
    log.info("Next: build pipelines/inference_pipeline.py (Phase 4).")

    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Training Pipeline")
    parser.add_argument(
        "--force-model",
        type=str,
        choices=["ridge", "random_forest", "xgboost", "lstm"],
        default=None,
        help="Force a specific model to be registered, ignoring metric ranking.",
    )
    args = parser.parse_args()
    run_training(force_model=args.force_model)