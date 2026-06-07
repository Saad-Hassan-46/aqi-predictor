"""
inference_pipeline.py
----------------------
Loads the latest registered model from the Hopsworks Model Registry,
fetches the most recent features from the Feature Store, and generates
AQI predictions at +24h, +48h, and +72h.

Predictions are written to:
  1. Hopsworks Feature Store  → 'aqi_predictions' feature group
                                (so Streamlit app can read them live)
  2. Local JSON backup        → data/latest_predictions.json
                                (for offline debugging)

Run manually:
    python pipelines/inference_pipeline.py

GitHub Actions runs this automatically every hour via .github/workflows/inference.yml
"""

import json
import logging
import os
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Fix Hopsworks /tmp path issue on Windows
tmpdir = os.getenv("HOPSWORKS_TMPDIR")
if tmpdir:
    tempfile.tempdir = tmpdir

HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME       = "aqi_features"
FEATURE_GROUP_VERSION    = 1
PREDICTIONS_FG_NAME      = "aqi_predictions"
PREDICTIONS_FG_VERSION   = 1
MODEL_NAME               = "aqi_forecaster"
FORECAST_HORIZONS        = [24, 48, 72]
LOOKBACK_HOURS           = 72

# Local backup path (relative to project root)
LOCAL_BACKUP_PATH = Path(__file__).resolve().parent.parent / "data" / "latest_predictions.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONNECT TO HOPSWORKS
# ══════════════════════════════════════════════════════════════════════════════

def connect_hopsworks():
    """Login to Hopsworks and return (project, feature_store) tuple."""
    log.info("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    log.info("Connected.")
    return project, fs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD MODEL FROM REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def load_model(project):
    """
    Download the latest registered model artifacts from Hopsworks
    Model Registry and return (model, scaler, feature_cols, model_info).

    The model directory contains:
      model.pkl           — sklearn/xgboost model
      scaler.pkl          — fitted StandardScaler (may be None for RF/XGB)
      feature_columns.json — ordered feature list the model expects
      model_info.json     — metadata (type, horizons, is_lstm, etc.)
    """
    log.info(f"Loading model '{MODEL_NAME}' from Hopsworks Model Registry...")
    mr = project.get_model_registry()

    # get_best_model fetches the version with the best metric
    # Falls back to latest version if metric not found
    try:
<<<<<<< HEAD
        hw_model = mr.get_model(name=MODEL_NAME)
=======
        hw_model = mr.get_model(name=MODEL_NAME)  # defaults to latest version
>>>>>>> 4d870c6d4d159ff80ae0af65d9693268a6743cd9
    except Exception:
        log.warning("Could not fetch best model by metric — loading latest version.")
        hw_model = mr.get_model(name=MODEL_NAME)

    model_dir = Path(hw_model.download())
    log.info(f"  Model version {hw_model.version} downloaded to {model_dir}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    model_info_path = model_dir / "model_info.json"
    model_info = json.loads(model_info_path.read_text()) if model_info_path.exists() else {}

    is_lstm = model_info.get("is_lstm", False)

    if is_lstm:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_dir / "model.keras"))
        except Exception as e:
            log.error(f"Failed to load LSTM model: {e}")
            sys.exit(1)
    else:
        model = joblib.load(model_dir / "model.pkl")

    # Scaler — may not exist for tree-based models
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Feature column list — tells us which columns to pass and in what order
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text())

    log.info(f"  Model type : {model_info.get('model_type', 'unknown')}")
    log.info(f"  Features   : {len(feature_cols)}")
    log.info(f"  Horizons   : {model_info.get('forecast_horizons', FORECAST_HORIZONS)}")

    return model, scaler, feature_cols, model_info


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FETCH & PREPARE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_latest_features(fs, feature_cols: list) -> pd.DataFrame:
    """
    Pull the most recent rows from the feature store, engineer the same
    lag/rolling features that were used during training, and return a
    single-row DataFrame ready for inference.

    We fetch LOOKBACK_HOURS + buffer rows to ensure we have enough history
    to compute all lag features without NaN in the most recent row.
    """
    log.info(f"Fetching latest features from '{FEATURE_GROUP_NAME}'...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    log.info(f"  Fetched {len(df):,} rows from feature store.")

    # Sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    # ── Engineer lag/rolling features (must match training_pipeline exactly) ──
    aqi = df["aqi"]

    for lag_h in [1, 6, 24, 48, 72]:
        df[f"aqi_lag_{lag_h}h"] = aqi.shift(lag_h)

    df["aqi_rolling_mean_6h"]  = aqi.shift(1).rolling(6).mean()
    df["aqi_rolling_mean_24h"] = aqi.shift(1).rolling(24).mean()
    df["aqi_rolling_std_24h"]  = aqi.shift(1).rolling(24).std()
    df["aqi_rolling_mean_7d"]  = aqi.shift(1).rolling(24 * 7).mean()

    if "pm25" in df.columns:
        for lag_h in [1, 24]:
            df[f"pm25_lag_{lag_h}h"] = df["pm25"].shift(lag_h)

    if "wind_speed" in df.columns:
        df["wind_lag_1h"] = df["wind_speed"].shift(1)

    # Drop rows where lags are still undefined
    df = df.dropna(subset=[f"aqi_lag_{LOOKBACK_HOURS}h"])

    if len(df) == 0:
        log.error("No usable rows after lag engineering. Need more data in feature store.")
        sys.exit(1)

    # ── Select the most recent row for inference ──────────────────────────────
    latest_row = df.iloc[[-1]]   # keep as DataFrame (not Series) for sklearn
    latest_timestamp = latest_row.index[0]
    log.info(f"  Using latest row: {latest_timestamp}")

    # ── Align columns to what the model expects ───────────────────────────────
    # Add any missing columns as NaN (will be imputed below)
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = np.nan

    # Drop any extra columns not seen during training
    latest_row = latest_row[feature_cols]

    log.info(f"  Feature matrix shape: {latest_row.shape}")
    return latest_row, latest_timestamp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESS FOR INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(X: pd.DataFrame, scaler) -> np.ndarray:
    """
    Apply the same NaN handling and scaling used during training.
    Returns a clean numpy array ready for model.predict().
    """
    X_arr = X.values.astype(float)

    # Drop fully-NaN columns (same logic as train_ridge fix)
    col_mask = ~np.all(np.isnan(X_arr), axis=0)
    X_arr = X_arr[:, col_mask]

    # Impute remaining NaNs with median strategy
    # For a single inference row, median = the value itself if not NaN
    # We use mean as fallback since median of one value is undefined
    imputer = SimpleImputer(strategy="mean")
    X_arr = imputer.fit_transform(X_arr)

    # Scale if a scaler was saved with the model (Ridge uses it)
    if scaler is not None:
        try:
            X_arr = scaler.transform(X_arr)
        except ValueError:
            # Feature count mismatch — re-fit a fresh scaler
            # This can happen if leaky cols were dropped post-training
            log.warning("Scaler feature mismatch — applying standard normalization.")
            fresh_scaler = StandardScaler()
            X_arr = fresh_scaler.fit_transform(X_arr)

    return X_arr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RUN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, X_processed: np.ndarray, model_info: dict) -> dict:
    """
    Run model.predict() and return a clean predictions dict.
    Handles both sklearn (flat output) and LSTM (sequence input).
    """
    is_lstm   = model_info.get("is_lstm", False)
    seq_len   = model_info.get("lstm_seq_len", LOOKBACK_HOURS)

    if is_lstm:
        # LSTM expects (batch, timesteps, features) — we only have 1 row here
        # For inference we repeat the single row to fill the sequence window
        # (acceptable approximation when only the latest state is available)
        X_seq = np.repeat(X_processed, seq_len, axis=0)
        X_seq = X_seq[np.newaxis, :, :]   # shape: (1, seq_len, n_features)
        raw_preds = model.predict(X_seq, verbose=0)[0]
    else:
        raw_preds = model.predict(X_processed)[0]

    predictions = {
        f"aqi_pred_{h}h": round(float(raw_preds[i]), 2)
        for i, h in enumerate(FORECAST_HORIZONS)
    }

    log.info("  Predictions:")
    for h in FORECAST_HORIZONS:
        log.info(f"    +{h:>2}h → AQI {predictions[f'aqi_pred_{h}h']:.1f}")

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_to_hopsworks(fs, predictions: dict, latest_timestamp):
    """
    Write predictions to the 'aqi_predictions' feature group in Hopsworks.
    Creates the feature group on first run; appends on subsequent runs.

    Schema:
        timestamp       — when the prediction was made (source data time)
        predicted_at    — wall-clock time this pipeline ran
        aqi_pred_24h    — predicted AQI 24h ahead
        aqi_pred_48h    — predicted AQI 48h ahead
        aqi_pred_72h    — predicted AQI 72h ahead
    """
    log.info(f"Saving predictions to Hopsworks feature group '{PREDICTIONS_FG_NAME}'...")

    now_utc = datetime.now(timezone.utc)

    pred_row = {
        "timestamp"    : latest_timestamp,
        "predicted_at" : now_utc,
        "aqi_pred_24h" : predictions["aqi_pred_24h"],
        "aqi_pred_48h" : predictions["aqi_pred_48h"],
        "aqi_pred_72h" : predictions["aqi_pred_72h"],
    }
    pred_df = pd.DataFrame([pred_row])

    # Ensure timezone-aware timestamps (Hopsworks requires UTC)
    pred_df["timestamp"]    = pd.to_datetime(pred_df["timestamp"], utc=True)
    pred_df["predicted_at"] = pd.to_datetime(pred_df["predicted_at"], utc=True)

    try:
        fg = fs.get_or_create_feature_group(
            name=PREDICTIONS_FG_NAME,
            version=PREDICTIONS_FG_VERSION,
            primary_key=["timestamp"],
            description="AQI forecasts at +24h, +48h, +72h generated by inference pipeline.",
            event_time="timestamp",
        )
        fg.insert(pred_df, write_options={"wait_for_job": False})
        log.info("  Predictions written to Hopsworks successfully.")
    except Exception as e:
        log.error(f"  Failed to write to Hopsworks: {e}")
        log.info("  Continuing — local backup will still be saved.")


def save_to_local(predictions: dict, latest_timestamp):
    """
    Write predictions to data/latest_predictions.json as a local backup.
    The Streamlit app can fall back to this file if Hopsworks is unavailable.
    """
    LOCAL_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at"  : datetime.now(timezone.utc).isoformat(),
        "based_on_data_at": str(latest_timestamp),
        "predictions"   : predictions,
        "horizons_hours": FORECAST_HORIZONS,
    }

    LOCAL_BACKUP_PATH.write_text(json.dumps(output, indent=2))
    log.info(f"  Local backup saved → {LOCAL_BACKUP_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_pipeline():
    """
    End-to-end inference run.

    Steps:
    1.  Connect to Hopsworks.
    2.  Load best registered model + artifacts.
    3.  Fetch latest features from Feature Store.
    4.  Preprocess (NaN handling + scaling).
    5.  Run model.predict().
    6.  Save predictions → Hopsworks + local JSON.
    """
    log.info("=" * 60)
    log.info("AQI Inference Pipeline — starting")
    log.info("=" * 60)

    # ── 1. Connect ────────────────────────────────────────────────────────────
    project, fs = connect_hopsworks()

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, scaler, feature_cols, model_info = load_model(project)

    # ── 3. Fetch features ─────────────────────────────────────────────────────
    X_raw, latest_timestamp = fetch_latest_features(fs, feature_cols)

    # ── 4. Preprocess ─────────────────────────────────────────────────────────
    X_processed = preprocess(X_raw, scaler)

    # ── 5. Predict ────────────────────────────────────────────────────────────
    log.info("Running inference...")
    predictions = run_inference(model, X_processed, model_info)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    log.info("Saving predictions...")
    save_to_hopsworks(fs, predictions, latest_timestamp)
    save_to_local(predictions, latest_timestamp)

    log.info("=" * 60)
    log.info("Inference pipeline complete.")
    log.info("=" * 60)

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_inference_pipeline()