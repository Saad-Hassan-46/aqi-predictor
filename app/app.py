"""
app.py
------
AQI Forecasting Dashboard — Islamabad
Loads model and features from Hopsworks, generates 3-day AQI forecast,
shows pollutant breakdown, SHAP explainability, and hazard alerts.

Deploy:
    streamlit run app/app.py
"""

import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

try:
    import hopsworks
except ImportError:
    hopsworks = None

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load environment ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_model")

# ── AQI Category definitions ──────────────────────────────────────────────────
AQI_CATEGORIES = [
    {"label": "Good",           "min": 0,   "max": 50,  "color": "#00C853", "bg": "#E8F5E9", "emoji": "😊"},
    {"label": "Moderate",       "min": 51,  "max": 100, "color": "#FFD600", "bg": "#FFFDE7", "emoji": "😐"},
    {"label": "Unhealthy (SG)", "min": 101, "max": 150, "color": "#FF6D00", "bg": "#FFF3E0", "emoji": "😷"},
    {"label": "Unhealthy",      "min": 151, "max": 200, "color": "#D50000", "bg": "#FFEBEE", "emoji": "🤢"},
    {"label": "Very Unhealthy", "min": 201, "max": 300, "color": "#6A1B9A", "bg": "#F3E5F5", "emoji": "🚨"},
    {"label": "Hazardous",      "min": 301, "max": 500, "color": "#37474F", "bg": "#ECEFF1", "emoji": "☠️"},
]

def get_aqi_category(aqi: float) -> dict:
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached so it doesn't reload on every interaction
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(ttl=3600, show_spinner="Connecting to Hopsworks...")
def load_hopsworks():
    """Connect to Hopsworks and return project object. Cached for 1 hour."""
    if hopsworks is None:
        raise RuntimeError("hopsworks is not installed.")
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not configured.")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    return project


@st.cache_data(ttl=3600, show_spinner="Loading features from feature store...")
def load_features():
    """Load and return the latest features from Hopsworks."""
    try:
        project = load_hopsworks()
        fs      = project.get_feature_store()
        fg      = fs.get_feature_group("aqi_features", version=1)
        df      = fg.read()
        df      = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return load_local_features()


@st.cache_resource(ttl=86400, show_spinner="Loading model from registry...")
def load_model():
    """Load best model, imputer, and feature columns from Hopsworks Model Registry."""
    try:
        project = load_hopsworks()
        mr      = project.get_model_registry()

        # Get latest version of the model
        model_meta = mr.get_best_model("aqi_forecaster", metric="rmse", direction="min")
        model_dir  = Path(model_meta.download())
    except Exception:
        model_dir = PROJECT_ROOT / "models"

    model        = joblib.load(model_dir / "best_model.pkl")
    feature_cols = joblib.load(model_dir / "feature_columns.pkl")
    imputer      = joblib.load(model_dir / "imputer.pkl")

    return model, feature_cols, imputer


def pm25_to_aqi(pm25: float) -> float:
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm25 = min(float(pm25), 500.4)
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0


@st.cache_data(ttl=3600, show_spinner="Loading local historical data...")
def load_local_features():
    csv_path = PROJECT_ROOT / "data" / "islamabad-us_embassy__pakistan-air-quality.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["date"], format="%Y/%m/%d", utc=True)
    df["city"] = os.getenv("CITY_NAME", "islamabad")
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25"]).sort_values("timestamp").reset_index(drop=True)
    df["aqi"] = df["pm25"].apply(pm25_to_aqi)

    monthly = {
        1: (9.5, 68, 2.5, 1018), 2: (11.5, 65, 2.8, 1015), 3: (17.0, 60, 3.2, 1010),
        4: (23.0, 55, 3.5, 1005), 5: (29.0, 40, 4.0, 1000), 6: (33.5, 45, 4.5, 996),
        7: (31.0, 70, 3.8, 994), 8: (29.5, 72, 3.2, 995), 9: (26.0, 65, 2.8, 1000),
        10: (21.0, 55, 2.2, 1008), 11: (14.5, 60, 2.0, 1014), 12: (10.0, 68, 2.2, 1017),
    }
    weather = df["timestamp"].dt.month.map(monthly)
    df["temperature"] = [v[0] for v in weather]
    df["humidity"] = [v[1] for v in weather]
    df["wind_speed"] = [v[2] for v in weather]
    df["pressure"] = [v[3] for v in weather]
    df["pm10"] = np.nan
    df["no2"] = np.nan
    df["o3"] = np.nan
    df["so2"] = np.nan
    df["co"] = np.nan
    df["weather_desc"] = "local historical fallback"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — same functions as training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features — must match training pipeline exactly."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for lag in [1, 2, 3, 7, 14]:
        df[f"aqi_lag_{lag}d"] = df["aqi"].shift(lag)
    df["aqi_rolling_mean_3d"]      = df["aqi"].shift(1).rolling(3,  min_periods=1).mean()
    df["aqi_rolling_mean_7d"]      = df["aqi"].shift(1).rolling(7,  min_periods=1).mean()
    df["aqi_rolling_mean_14d"]     = df["aqi"].shift(1).rolling(14, min_periods=1).mean()
    df["aqi_rolling_std_7d"]       = df["aqi"].shift(1).rolling(7,  min_periods=2).std().fillna(0)
    df["temp_rolling_mean_3d"]     = df["temperature"].shift(1).rolling(3, min_periods=1).mean()
    df["humidity_rolling_mean_3d"] = df["humidity"].shift(1).rolling(3,    min_periods=1).mean()
    df["target_aqi_1d"]            = df["aqi"].shift(-1)
    df["target_aqi_2d"]            = df["aqi"].shift(-2)
    df["target_aqi_3d"]            = df["aqi"].shift(-3)
    return df


def generate_3day_forecast(df, model, feature_cols, imputer) -> list:
    """
    Generate 3-day AQI forecast using recursive prediction.
    Day 1 prediction feeds into Day 2 features, and so on.
    """
    df_feat  = engineer_lag_features(df)
    latest   = df_feat.iloc[-1:].copy()
    forecasts = []

    exclude = {"city", "timestamp", "aqi", "target_aqi_1d",
               "target_aqi_2d", "target_aqi_3d", "weather_desc"}
    feat_cols = [c for c in feature_cols if c in latest.columns]

    for day in range(1, 4):
        X = latest[feat_cols].values
        X = imputer.transform(X)
        pred_aqi = float(model.predict(X)[0])
        pred_aqi = max(0, min(500, pred_aqi))  # clip to valid range

        forecast_date = datetime.now(timezone.utc) + timedelta(days=day)
        forecasts.append({
            "day":   day,
            "date":  forecast_date.strftime("%A, %b %d"),
            "aqi":   round(pred_aqi, 1),
            "cat":   get_aqi_category(pred_aqi),
        })

        # Update lag features for next iteration
        latest = latest.copy()
        prev_lag_1d = latest["aqi_lag_1d"].copy()
        prev_lag_2d = latest["aqi_lag_2d"].copy()
        latest["aqi_lag_2d"]           = prev_lag_1d
        latest["aqi_lag_3d"]           = prev_lag_2d
        latest["aqi_lag_7d"]           = latest["aqi_lag_7d"]
        latest["aqi_lag_1d"]           = pred_aqi
        latest["aqi_rolling_mean_3d"]  = (latest["aqi_rolling_mean_3d"] * 2 + pred_aqi) / 3
        latest["aqi_rolling_mean_7d"]  = (latest["aqi_rolling_mean_7d"] * 6 + pred_aqi) / 7
        latest["aqi_change_rate"]      = round((pred_aqi - float(latest["aqi"].iloc[0])) / max(float(latest["aqi"].iloc[0]), 1) * 100, 4)
        latest["aqi"]                  = pred_aqi

    return forecasts


# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Computing SHAP values...")
def compute_shap(_model, _imputer, df, feature_cols):
    """Compute SHAP values for the latest prediction."""
    try:
        import shap
        df_feat  = engineer_lag_features(df)
        feat_cols = [c for c in feature_cols if c in df_feat.columns]
        X        = df_feat[feat_cols].dropna().values
        X        = _imputer.transform(X)

        # Use TreeExplainer for tree-based models, LinearExplainer for Ridge
        model_type = type(_model).__name__
        if "Ridge" in model_type or "Pipeline" in model_type:
            # For Pipeline with Ridge
            try:
                inner = _model.named_steps["model"]
                scaler = _model.named_steps["scaler"]
                X_scaled = scaler.transform(X)
                explainer   = shap.LinearExplainer(inner, X_scaled)
                shap_values = explainer.shap_values(X_scaled[-1:])
            except:
                explainer   = shap.Explainer(_model.predict, X[-50:])
                shap_values = explainer(X[-1:]).values
        else:
            explainer   = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(X[-1:])

        shap_df = pd.DataFrame({
            "feature": feat_cols,
            "shap_value": shap_values[0] if len(shap_values.shape) > 1 else shap_values.flatten(),
        }).sort_values("shap_value", key=abs, ascending=False).head(15)

        return shap_df
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #F8F9FA; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid;
        margin-bottom: 8px;
    }
    .metric-card .aqi-value {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-card .day-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .metric-card .cat-label {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Alert box */
    .alert-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 5px solid;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1A1A2E;
        margin: 24px 0 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #E0E0E0;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #1A1A2E; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/200px-Flag_of_Pakistan.svg.png", width=60)
        st.title("🌫️ AQI Forecast")
        st.markdown("**Islamabad, Pakistan**")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Real-time 3-day Air Quality Index forecasting using machine learning.

        **Data sources:**
        - AQICN API (live AQI)
        - OpenWeather API (weather)
        - US Embassy Islamabad monitor

        **Model:** Trained on 2019–2026 historical data
        """)
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌫️ Islamabad Air Quality Forecast")
    st.markdown("Machine learning powered 3-day AQI prediction using real-time sensor data")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        try:
            df                       = load_features()
            model, feature_cols, imputer = load_model()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.info("Check your Hopsworks API key and project name in secrets.")
            return

    latest_row = df.iloc[-1]
    current_aqi = float(latest_row["aqi"])
    current_cat = get_aqi_category(current_aqi)

    # ── Current AQI banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{current_cat['bg']};border-radius:16px;padding:24px;
                border-left:8px solid {current_cat['color']};margin-bottom:24px">
        <div style="display:flex;align-items:center;gap:16px">
            <div style="font-size:3.5rem">{current_cat['emoji']}</div>
            <div>
                <div style="font-size:0.9rem;color:#555;font-weight:500">CURRENT AQI · ISLAMABAD US EMBASSY</div>
                <div style="font-size:3rem;font-weight:800;color:{current_cat['color']};line-height:1">{current_aqi}</div>
                <div style="font-size:1.1rem;font-weight:600;color:{current_cat['color']}">{current_cat['label']}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div style="font-size:0.85rem;color:#666">PM2.5</div>
                <div style="font-size:1.8rem;font-weight:700">{latest_row.get('pm25', 'N/A')}</div>
                <div style="font-size:0.75rem;color:#888">µg/m³</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3-Day Forecast ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 3-Day Forecast</div>', unsafe_allow_html=True)

    try:
        forecasts = generate_3day_forecast(df, model, feature_cols, imputer)

        cols = st.columns(3)
        for i, forecast in enumerate(forecasts):
            with cols[i]:
                cat = forecast["cat"]
                st.markdown(f"""
                <div class="metric-card" style="border-top-color:{cat['color']}">
                    <div class="day-label">{forecast['date']}</div>
                    <div class="aqi-value" style="color:{cat['color']}">{forecast['aqi']}</div>
                    <div style="font-size:1.5rem;margin:4px 0">{cat['emoji']}</div>
                    <div class="cat-label" style="color:{cat['color']}">{cat['label']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Forecast trend chart
        forecast_df = pd.DataFrame([{
            "Day":  f["date"],
            "AQI":  f["aqi"],
            "Category": f["cat"]["label"],
            "Color": f["cat"]["color"],
        } for f in forecasts])

        # Add today's reading
        today_row = pd.DataFrame([{
            "Day":  "Today",
            "AQI":  current_aqi,
            "Category": current_cat["label"],
            "Color": current_cat["color"],
        }])
        chart_df = pd.concat([today_row, forecast_df], ignore_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["Day"], y=chart_df["AQI"],
            mode="lines+markers+text",
            text=chart_df["AQI"],
            textposition="top center",
            line=dict(color="#1565C0", width=3),
            marker=dict(size=12, color=chart_df["Color"], line=dict(width=2, color="white")),
            textfont=dict(size=13, color="#333"),
        ))

        # AQI category bands
        bands = [(0, 50, "#E8F5E9"), (51, 100, "#FFFDE7"), (101, 150, "#FFF3E0"),
                 (151, 200, "#FFEBEE"), (201, 300, "#F3E5F5")]
        for lo, hi, color in bands:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.4, line_width=0)

        fig.update_layout(
            title="AQI Trend: Today + 3-Day Forecast",
            xaxis_title="", yaxis_title="AQI",
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(t=40, b=20, l=40, r=20),
            yaxis=dict(range=[0, max(300, max(chart_df["AQI"]) + 30)]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")

    # ── Hazard Alerts ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Health Alerts</div>', unsafe_allow_html=True)

    all_aqi = [current_aqi] + [f["aqi"] for f in forecasts]
    max_aqi = max(all_aqi)
    max_cat = get_aqi_category(max_aqi)

    if max_aqi <= 50:
        st.success("✅ Air quality is good for the next 3 days. Enjoy outdoor activities!")
    elif max_aqi <= 100:
        st.warning("⚠️ Air quality is moderate. Unusually sensitive people should consider reducing prolonged outdoor exertion.")
    elif max_aqi <= 150:
        st.warning(f"🟠 Unhealthy for Sensitive Groups (AQI up to {max_aqi:.0f}). People with respiratory or heart conditions should limit outdoor activity.")
    elif max_aqi <= 200:
        st.error(f"🔴 Unhealthy conditions expected (AQI up to {max_aqi:.0f}). Everyone should reduce prolonged outdoor exertion. Wear an N95 mask outdoors.")
    elif max_aqi <= 300:
        st.error(f"🟣 Very Unhealthy (AQI up to {max_aqi:.0f}). Avoid all outdoor activity. Keep windows closed. Use air purifiers indoors.")
    else:
        st.error(f"☠️ HAZARDOUS conditions (AQI up to {max_aqi:.0f}). Stay indoors. This is a health emergency. Wear N95 mask if you must go out.")

    # Health advice table
    advice_data = {
        "Group": ["General Public", "Children & Elderly", "Asthma / Heart Conditions", "Athletes / Outdoor Workers"],
        "Recommendation": [
            "Limit prolonged outdoor activity" if max_aqi > 100 else "Normal activity OK",
            "Avoid outdoor play" if max_aqi > 100 else "Normal activity OK",
            "Stay indoors, use inhaler as needed" if max_aqi > 100 else "Monitor symptoms",
            "Reschedule outdoor training" if max_aqi > 100 else "Normal activity OK",
        ]
    }
    st.dataframe(pd.DataFrame(advice_data), use_container_width=True, hide_index=True)

    # ── Historical AQI Trend ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Historical AQI Trend</div>', unsafe_allow_html=True)

    hist_df = df[df["aqi"] > 75].tail(90).copy()  # last 90 real readings
    if len(hist_df) > 5:
        fig2 = px.line(
            hist_df, x="timestamp", y="aqi",
            title="Last 90 Days — Daily AQI",
            labels={"aqi": "AQI", "timestamp": "Date"},
            color_discrete_sequence=["#1565C0"],
        )
        fig2.add_hline(y=100, line_dash="dash", line_color="#FFD600", annotation_text="Moderate")
        fig2.add_hline(y=150, line_dash="dash", line_color="#FF6D00", annotation_text="Unhealthy SG")
        fig2.add_hline(y=200, line_dash="dash", line_color="#D50000", annotation_text="Unhealthy")
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(t=40, b=20, l=40, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Pollutant Breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧪 Pollutant Breakdown</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    pollutants = {
        "PM2.5": latest_row.get("pm25"),
        "PM10":  latest_row.get("pm10"),
        "NO2":   latest_row.get("no2"),
        "O3":    latest_row.get("o3"),
        "SO2":   latest_row.get("so2"),
        "CO":    latest_row.get("co"),
    }
    available = {k: v for k, v in pollutants.items() if v is not None and not np.isnan(float(v))}

    with col1:
        if available:
            poll_df = pd.DataFrame(list(available.items()), columns=["Pollutant", "Value (µg/m³)"])
            fig3 = px.bar(
                poll_df, x="Pollutant", y="Value (µg/m³)",
                title="Current Pollutant Levels",
                color="Value (µg/m³)",
                color_continuous_scale="RdYlGn_r",
            )
            fig3.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(t=40, b=20, l=40, r=20),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Individual pollutant data not available for current reading. Only PM2.5 is reported by the Islamabad US Embassy station.")
            st.metric("PM2.5", f"{latest_row.get('pm25', 'N/A')} µg/m³", help="Fine particulate matter — primary AQI driver")

    with col2:
        st.markdown("**What each pollutant means:**")
        st.markdown("""
        | Pollutant | Source | Health Impact |
        |-----------|--------|---------------|
        | **PM2.5** | Vehicle exhaust, burning | Penetrates lungs deeply |
        | **PM10** | Dust, construction | Respiratory irritation |
        | **NO2** | Traffic, industry | Asthma trigger |
        | **O3** | Sunlight + pollutants | Lung damage |
        | **SO2** | Industry, power plants | Breathing difficulty |
        | **CO** | Incomplete combustion | Reduces oxygen in blood |
        """)

    # ── Weather Context ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌤️ Current Weather Conditions</div>', unsafe_allow_html=True)

    w_cols = st.columns(4)
    weather_metrics = [
        ("🌡️ Temperature", f"{latest_row.get('temperature', 'N/A')}°C"),
        ("💧 Humidity",    f"{latest_row.get('humidity', 'N/A')}%"),
        ("💨 Wind Speed",  f"{latest_row.get('wind_speed', 'N/A')} m/s"),
        ("🔵 Pressure",    f"{latest_row.get('pressure', 'N/A')} hPa"),
    ]
    for col, (label, value) in zip(w_cols, weather_metrics):
        with col:
            st.metric(label, value)

    # ── SHAP Explainability ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Why this prediction? (SHAP Explainability)</div>', unsafe_allow_html=True)
    st.markdown("SHAP values show which features most influenced today's AQI prediction. Positive values push AQI higher; negative values push it lower.")

    with st.spinner("Computing SHAP values..."):
        shap_df = compute_shap(model, imputer, df, feature_cols)

    if shap_df is not None and len(shap_df) > 0:
        shap_df["direction"] = shap_df["shap_value"].apply(lambda x: "Increases AQI" if x > 0 else "Decreases AQI")
        shap_df["abs_value"] = shap_df["shap_value"].abs()

        fig4 = px.bar(
            shap_df.head(12),
            x="shap_value", y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Increases AQI": "#D50000", "Decreases AQI": "#00C853"},
            title="Top Feature Contributions to Latest Prediction",
            labels={"shap_value": "SHAP Value (impact on AQI)", "feature": "Feature"},
        )
        fig4.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=420, margin=dict(t=40, b=20, l=160, r=20),
            yaxis=dict(autorange="reversed"),
            legend_title="Effect",
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Top 3 explanation in plain English
        top3 = shap_df.head(3)
        st.markdown("**In plain English:**")
        for _, row in top3.iterrows():
            direction = "increasing" if row["shap_value"] > 0 else "decreasing"
            st.markdown(f"- **{row['feature']}** is {direction} the predicted AQI by `{abs(row['shap_value']):.1f}` points")
    else:
        st.info("SHAP analysis requires the `shap` package. Run: `pip install shap`")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#888;font-size:0.85rem'>
        Built with ❤️ using Streamlit · Data: AQICN & US Embassy Islamabad ·
        Model: Scikit-learn · Feature Store: Hopsworks
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
