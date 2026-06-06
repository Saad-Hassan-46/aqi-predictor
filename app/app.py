"""
app.py
-------
Main Streamlit application for the AQI Forecaster.

Pages:
  1. Forecast        — AQI predictions at +24h, +48h, +72h
  2. Historical      — Historical AQI trend chart
  3. Explainability  — SHAP feature importance
  4. About           — Project info and architecture

Run locally:
    streamlit run app/app.py

Deployed on Streamlit Community Cloud.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.aqi_categories import AQI_CATEGORIES, get_aqi_category

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecaster — Rawalpindi",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #ccc;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 13px; color: #666; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 36px; font-weight: 700; color: #111; }
    .metric-category { font-size: 14px; font-weight: 600; margin-top: 4px; }
    .metric-advice { font-size: 12px; color: #555; margin-top: 6px; }
    .section-header { font-size: 20px; font-weight: 600; color: #111; margin: 24px 0 12px 0; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_predictions() -> dict | None:
    """
    Load latest predictions. Tries Hopsworks first, falls back to local JSON.
    Returns None if neither source is available.
    """
    # ── Try local backup first (fast, no network) ─────────────────────────────
    local_path = ROOT / "data" / "latest_predictions.json"
    if local_path.exists():
        try:
            data = json.loads(local_path.read_text())
            data["_source"] = "local"
            return data
        except Exception:
            pass

    # ── Try Hopsworks ─────────────────────────────────────────────────────────
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env")

        import hopsworks
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model"),
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_predictions", version=1)
        df = fg.read()
        if len(df) == 0:
            return None

        df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
        latest = df.sort_values("predicted_at").iloc[-1]

        return {
            "generated_at": str(latest["predicted_at"]),
            "based_on_data_at": str(latest["timestamp"]),
            "predictions": {
                "aqi_pred_24h": float(latest["aqi_pred_24h"]),
                "aqi_pred_48h": float(latest["aqi_pred_48h"]),
                "aqi_pred_72h": float(latest["aqi_pred_72h"]),
            },
            "horizons_hours": [24, 48, 72],
            "_source": "hopsworks",
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical() -> pd.DataFrame | None:
    """
    Load historical AQI data from Hopsworks feature store.
    Falls back to None if unavailable.
    """
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env")

        import hopsworks
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model"),
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=1)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌬️ AQI Forecaster")
    st.markdown("**Rawalpindi / Islamabad**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Forecast", "📈 Historical", "🔍 Explainability", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Data refreshes every hour via GitHub Actions.")
    st.caption("Model: Ridge Regression | Features: Weather + Pollutants")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def render_forecast():
    st.title("📊 AQI Forecast")
    st.markdown("Predicted Air Quality Index for Rawalpindi at +24h, +48h, and +72h.")

    with st.spinner("Loading latest predictions..."):
        data = load_predictions()

    if data is None:
        st.error("Predictions unavailable. The inference pipeline may not have run yet.")
        st.info("Run `python pipelines/inference_pipeline.py` to generate predictions.")
        return

    preds = data["predictions"]
    generated_at = data.get("generated_at", "Unknown")
    based_on = data.get("based_on_data_at", "Unknown")
    source = data.get("_source", "unknown")

    # ── Freshness banner ──────────────────────────────────────────────────────
    try:
        gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
        if age_hours < 2:
            st.success(f"✅ Predictions fresh — generated {age_hours:.1f}h ago")
        elif age_hours < 6:
            st.warning(f"⚠️ Predictions are {age_hours:.1f}h old — next update coming soon")
        else:
            st.error(f"🔴 Predictions are {age_hours:.1f}h old — pipeline may be down")
    except Exception:
        st.info(f"Generated at: {generated_at}")

    st.caption(f"Based on data at: {based_on} | Source: {source}")
    st.markdown("---")

    # ── Metric cards ──────────────────────────────────────────────────────────
    horizons = [
        ("24h", preds.get("aqi_pred_24h", 0)),
        ("48h", preds.get("aqi_pred_48h", 0)),
        ("72h", preds.get("aqi_pred_72h", 0)),
    ]

    cols = st.columns(3)
    for col, (label, aqi_val) in zip(cols, horizons):
        cat = get_aqi_category(aqi_val)
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {cat['color']};">
                <div class="metric-label">+{label} Forecast</div>
                <div class="metric-value">{aqi_val:.0f}</div>
                <div class="metric-category" style="color: {cat['color']};">
                    {cat['label']}
                </div>
                <div class="metric-advice">{cat['advice']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Forecast bar chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Forecast Trend</div>', unsafe_allow_html=True)

    aqi_vals = [v for _, v in horizons]
    colors   = [get_aqi_category(v)["color"] for v in aqi_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"+{h}" for h, _ in horizons],
        y=aqi_vals,
        marker_color=colors,
        text=[f"{v:.0f}" for v in aqi_vals],
        textposition="outside",
        width=0.4,
    ))

    # AQI threshold lines
    thresholds = [(50, "Good", "#00E400"), (100, "Moderate", "#FFFF00"),
                  (150, "Unhealthy SG", "#FF7E00"), (200, "Unhealthy", "#FF0000")]
    for threshold, label, color in thresholds:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color=color,
            annotation_text=label, annotation_position="right",
            line_width=1,
        )

    fig.update_layout(
        xaxis_title="Forecast Horizon",
        yaxis_title="AQI",
        yaxis=dict(range=[0, max(max(aqi_vals) * 1.3, 220)]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=40, l=40, r=80),
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")

    st.plotly_chart(fig, use_container_width=True)

    # ── AQI scale reference ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">AQI Reference Scale</div>', unsafe_allow_html=True)

    ref_cols = st.columns(len(AQI_CATEGORIES))
    for col, cat in zip(ref_cols, AQI_CATEGORIES):
        with col:
            st.markdown(f"""
            <div style="background:{cat['color']}22; border-radius:8px; padding:10px;
                        border-top: 4px solid {cat['color']}; text-align:center;">
                <div style="font-weight:600; font-size:13px;">{cat['label']}</div>
                <div style="font-size:12px; color:#555;">{cat['min']}–{cat['max']}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HISTORICAL
# ══════════════════════════════════════════════════════════════════════════════

def render_historical():
    st.title("📈 Historical AQI")
    st.markdown("Hourly AQI readings from the Hopsworks feature store.")

    with st.spinner("Loading historical data..."):
        df = load_historical()

    if df is None or len(df) == 0:
        st.error("Historical data unavailable. Check your Hopsworks connection.")
        return

    st.success(f"Loaded {len(df):,} hourly readings from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        days_back = st.slider("Show last N days", min_value=3, max_value=21, value=7, step=1)
    with col2:
        show_pollutants = st.checkbox("Overlay PM2.5", value=False)

    cutoff = df["timestamp"].max() - pd.Timedelta(days=days_back)
    df_filtered = df[df["timestamp"] >= cutoff].copy()

    # ── Main AQI line chart ───────────────────────────────────────────────────
    fig = go.Figure()

    # Colour the line by AQI category using segments
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp"],
        y=df_filtered["aqi"],
        mode="lines",
        name="AQI",
        line=dict(color="#2563EB", width=2),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.08)",
    ))

    if show_pollutants and "pm25" in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered["timestamp"],
            y=df_filtered["pm25"],
            mode="lines",
            name="PM2.5",
            line=dict(color="#DC2626", width=1.5, dash="dot"),
        ))

    # Threshold bands
    for threshold, label, color in [
        (50, "Good", "#00E400"), (100, "Moderate", "#FFFF00"),
        (150, "Unhealthy SG", "#FF7E00"), (200, "Unhealthy", "#FF0000"),
    ]:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color=color,
            line_width=1,
            annotation_text=label, annotation_position="right",
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=20, b=40, l=40, r=80),
        font=dict(family="Inter, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)

    s = df_filtered["aqi"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average AQI", f"{s.mean():.1f}")
    m2.metric("Max AQI", f"{s.max():.0f}")
    m3.metric("Min AQI", f"{s.min():.0f}")
    m4.metric("Std Dev", f"{s.std():.1f}")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("View raw data"):
        show_cols = ["timestamp", "aqi", "pm25", "pm10", "temperature", "humidity", "wind_speed"]
        show_cols = [c for c in show_cols if c in df_filtered.columns]
        st.dataframe(
            df_filtered[show_cols].sort_values("timestamp", ascending=False).head(200),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def render_explainability():
    st.title("🔍 Model Explainability")
    st.markdown("SHAP (SHapley Additive exPlanations) shows which features drive each prediction.")

    with st.spinner("Loading model and computing SHAP values..."):
        result = compute_shap()

    if result is None:
        st.warning("SHAP computation requires the model and feature data to be available.")
        st.info("Make sure you have run the training pipeline and inference pipeline first.")
        return

    shap_vals, feature_names, X_sample = result

    # ── Global feature importance ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Global Feature Importance (Mean |SHAP|)</div>',
                unsafe_allow_html=True)
    st.caption("Averaged across all test samples and forecast horizons. Higher = more influential.")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap,
    }).sort_values("Mean |SHAP|", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=importance_df["Mean |SHAP|"],
        y=importance_df["Feature"],
        orientation="h",
        marker_color="#2563EB",
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP| value",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=10, b=40, l=160, r=20),
        font=dict(family="Inter, sans-serif", size=13),
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-horizon breakdown ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Importance by Forecast Horizon</div>',
                unsafe_allow_html=True)

    horizon_labels = ["+24h", "+48h", "+72h"]
    tabs = st.tabs(horizon_labels)

    for tab, h_idx, h_label in zip(tabs, range(3), horizon_labels):
        with tab:
            h_shap = np.abs(shap_vals[:, :, h_idx]).mean(axis=0) if shap_vals.ndim == 3 else np.abs(shap_vals).mean(axis=0)
            h_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP|": h_shap,
            }).sort_values("Mean |SHAP|", ascending=True).tail(10)

            fig_h = go.Figure(go.Bar(
                x=h_df["Mean |SHAP|"],
                y=h_df["Feature"],
                orientation="h",
                marker_color="#2563EB",
            ))
            fig_h.update_layout(
                xaxis_title="Mean |SHAP|",
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=320,
                margin=dict(t=10, b=30, l=160, r=20),
                font=dict(family="Inter, sans-serif", size=12),
            )
            fig_h.update_xaxes(gridcolor="#f0f0f0")
            fig_h.update_yaxes(showgrid=False)
            st.plotly_chart(fig_h, use_container_width=True)

    # ── Explainer note ────────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "**How to read this:** SHAP values measure each feature's contribution to the "
        "prediction. A high SHAP value means the feature pushed the AQI prediction higher; "
        "a negative value means it pushed it lower. The bar chart shows the average magnitude "
        "across all predictions — larger bars = more important features overall."
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_shap():
    """Compute SHAP values for the current model. Cached for 1 hour."""
    try:
        import shap
        import os
        import joblib
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env")

        import hopsworks
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model"),
        )

        # Load model
        mr = project.get_model_registry()
        hw_model = mr.get_model(name="aqi_forecaster")
        model_dir = Path(hw_model.download())
        model = joblib.load(model_dir / "model.pkl")
        feature_cols = json.loads((model_dir / "feature_columns.json").read_text())

        # Load features for background data
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=1)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Engineer same lag features as training
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

        df = df.dropna(subset=["aqi_lag_72h"])

        # Align to model's feature set
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        X = df[feature_cols].fillna(df[feature_cols].median()).values

        # Use a small sample for speed (SHAP is slow on large datasets)
        n_sample = min(50, len(X))
        X_sample = X[:n_sample]

        # SHAP explainer — TreeExplainer for RF/XGB, LinearExplainer for Ridge
        model_type = json.loads((model_dir / "model_info.json").read_text()).get("model_type", "")
        if model_type in ("ridge",):
            explainer = shap.LinearExplainer(
                model.estimators_[0],
                X_sample,
                feature_names=feature_cols,
            )
            shap_vals_list = np.stack([
                explainer.shap_values(X_sample)
                for est in model.estimators_
            ], axis=-1)
        else:
            explainer = shap.TreeExplainer(model.estimators_[0])
            shap_vals_list = np.stack([
                explainer.shap_values(X_sample)
                for est in model.estimators_
            ], axis=-1)

        return shap_vals_list, feature_cols, X_sample

    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about():
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### AQI Forecaster — Rawalpindi / Islamabad

    This project is a **production-grade MLOps pipeline** that forecasts Air Quality Index (AQI)
    up to 72 hours ahead for Rawalpindi and Islamabad, Pakistan — one of the most
    air-pollution-affected regions in South Asia.
    """)

    st.markdown("---")

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown("### 🏗️ Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Data Pipeline (hourly, automated)**
        - Fetches AQI from AQICN API
        - Fetches weather from OpenWeatherMap API
        - Engineers features (lag, rolling, cyclical time encoding)
        - Writes to Hopsworks Feature Store

        **Training Pipeline (on-demand)**
        - Reads from Feature Store
        - Trains Ridge, Random Forest, XGBoost, LSTM
        - Evaluates with temporal train/test split
        - Registers best model to Hopsworks Model Registry
        """)

    with col2:
        st.markdown("""
        **Inference Pipeline (hourly, automated)**
        - Loads latest model from Model Registry
        - Fetches most recent features from Feature Store
        - Generates +24h, +48h, +72h predictions
        - Writes predictions to Hopsworks + local backup

        **This App**
        - Reads predictions from Hopsworks / local JSON
        - Displays forecast, historical trends, SHAP explainability
        - Deployed on Streamlit Community Cloud
        - Auto-refreshes as new predictions arrive
        """)

    st.markdown("---")

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown("### 🛠️ Tech Stack")

    tech = {
        "Feature Store": "Hopsworks (serverless, free tier)",
        "Model Registry": "Hopsworks Model Registry",
        "ML Models": "scikit-learn, XGBoost, TensorFlow/Keras",
        "Explainability": "SHAP (SHapley Additive exPlanations)",
        "Orchestration": "GitHub Actions (hourly cron jobs)",
        "Dashboard": "Streamlit + Plotly",
        "Data Sources": "AQICN API + OpenWeatherMap API",
        "Language": "Python 3.11",
    }

    for k, v in tech.items():
        st.markdown(f"- **{k}:** {v}")

    st.markdown("---")

    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Model Details")
    st.markdown("""
    | Property | Value |
    |---|---|
    | Task | Multi-output regression (3 horizons) |
    | Targets | AQI at +24h, +48h, +72h |
    | Primary metric | Mean Absolute Error (MAE) |
    | Train/test split | Temporal (no shuffling) |
    | Feature count | 29 engineered features |
    | Data frequency | 1 observation per hour |
    """)

    st.info(
        "**Note on metrics:** The model was trained on ~21 days of initial data. "
        "MAE will improve automatically as more hourly data accumulates via the "
        "automated GitHub Actions pipeline."
    )

    st.markdown("---")
    st.caption("Built as part of an MLOps internship project. Data is real and refreshed hourly.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Forecast":
    render_forecast()
elif page == "📈 Historical":
    render_historical()
elif page == "🔍 Explainability":
    render_explainability()
elif page == "ℹ️ About":
    render_about()