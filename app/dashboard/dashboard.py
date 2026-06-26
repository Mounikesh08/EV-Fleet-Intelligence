import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

BATTERY_DATA_PATH = "data/battery_features.csv"
FSD_DATA_PATH = "data/fsd_features.csv"


st.set_page_config(
    page_title="EV Fleet Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0b1020 0%, #111827 45%, #020617 100%);
            color: #f8fafc;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        h1, h2, h3 {
            color: #f8fafc;
            font-weight: 800;
        }

        .hero-card {
            padding: 28px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(34,197,94,0.12));
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            margin-bottom: 22px;
        }

        .hero-title {
            font-size: 42px;
            font-weight: 900;
            letter-spacing: -1px;
            margin-bottom: 8px;
            color: #ffffff;
        }

        .hero-subtitle {
            font-size: 17px;
            color: #cbd5e1;
            line-height: 1.6;
        }

        .module-pill {
            display: inline-block;
            padding: 8px 14px;
            margin-right: 8px;
            margin-top: 12px;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            border: 1px solid rgba(96,165,250,0.35);
            color: #dbeafe;
            font-weight: 700;
            font-size: 13px;
        }

        .metric-card {
            padding: 18px;
            border-radius: 18px;
            background: rgba(15,23,42,0.92);
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 16px 40px rgba(0,0,0,0.30);
            min-height: 122px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .metric-value {
            color: #ffffff;
            font-size: 30px;
            font-weight: 900;
            margin-top: 8px;
        }

        .metric-note {
            color: #94a3b8;
            font-size: 12px;
            margin-top: 5px;
        }

        .section-card {
            padding: 22px;
            border-radius: 20px;
            background: rgba(15,23,42,0.86);
            border: 1px solid rgba(148,163,184,0.20);
            box-shadow: 0 14px 35px rgba(0,0,0,0.22);
            margin-top: 16px;
            margin-bottom: 16px;
        }

        .success-box {
            padding: 16px;
            border-radius: 16px;
            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.35);
            color: #dcfce7;
        }

        .warning-box {
            padding: 16px;
            border-radius: 16px;
            background: rgba(245,158,11,0.13);
            border: 1px solid rgba(245,158,11,0.35);
            color: #fef3c7;
        }

        .danger-box {
            padding: 16px;
            border-radius: 16px;
            background: rgba(239,68,68,0.14);
            border: 1px solid rgba(239,68,68,0.38);
            color: #fee2e2;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 13px;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            height: 48px;
            font-weight: 800;
            border: 1px solid rgba(96,165,250,0.45);
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: white;
        }

        .stButton > button:hover {
            border: 1px solid rgba(125,211,252,0.9);
            color: white;
        }

        div[data-testid="stMetric"] {
            background: rgba(15,23,42,0.82);
            padding: 18px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.20);
        }

        .footer {
            margin-top: 35px;
            color: #64748b;
            font-size: 13px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data Loaders
# -----------------------------
@st.cache_data
def load_battery_data() -> pd.DataFrame:
    return pd.read_csv(BATTERY_DATA_PATH)


@st.cache_data
def load_fsd_data() -> pd.DataFrame:
    return pd.read_csv(FSD_DATA_PATH)


def safe_load_battery_data():
    try:
        return load_battery_data()
    except FileNotFoundError:
        st.error("Battery feature data not found. Run: python app/features/battery_features.py")
        st.stop()


def safe_load_fsd_data():
    try:
        return load_fsd_data()
    except FileNotFoundError:
        st.error("FSD feature data not found. Run: python app/features/fsd_features.py")
        st.stop()


# -----------------------------
# Helper Components
# -----------------------------
def hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">EV Fleet Intelligence Platform</div>
            <div class="hero-subtitle">
                Production-style AI platform for electric vehicle fleet monitoring, battery intelligence,
                FSD scenario risk mining, predictive maintenance, and engineer-friendly risk explanation.
            </div>
            <span class="module-pill">Battery Intelligence</span>
            <span class="module-pill">FSD Scenario Intelligence</span>
            <span class="module-pill">FastAPI + MLflow + Streamlit</span>
            <span class="module-pill">Tesla-style Fleet Analytics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_box(level: str, title: str, text: str):
    css_class = "success-box"
    if level == "MEDIUM":
        css_class = "warning-box"
    elif level == "HIGH":
        css_class = "danger-box"

    st.markdown(
        f"""
        <div class="{css_class}">
            <b>{title}</b><br/>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def plot_dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=20, color="#ffffff"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", color="#cbd5e1")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", color="#cbd5e1")
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## ⚡ EV Fleet AI")
st.sidebar.markdown("Battery + FSD intelligence demo")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Battery Intelligence",
        "Vehicle Deep Dive",
        "FSD Scenario Intelligence",
        "Live API Demo",
        "Model Operations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### API Status")

health = check_api_health()

if health:
    st.sidebar.success("FastAPI online")
    st.sidebar.caption(f"Service: {health.get('service', 'EV Fleet Intelligence Platform')}")
else:
    st.sidebar.error("FastAPI offline")
    st.sidebar.caption("Run: uvicorn app.api.main:app --reload")

st.sidebar.markdown("---")
st.sidebar.caption(f"API URL: {API_URL}")


# -----------------------------
# Pages
# -----------------------------
hero()

battery_df = safe_load_battery_data()
fsd_df = safe_load_fsd_data()

battery_df["timestamp"] = pd.to_datetime(battery_df["timestamp"])
fsd_df["timestamp"] = pd.to_datetime(fsd_df["timestamp"])


if page == "Executive Overview":
    st.header("Executive Overview")

    total_battery_vehicles = battery_df["vehicle_id"].nunique()
    total_battery_rows = len(battery_df)
    battery_anomaly_rows = int(battery_df["is_anomaly"].sum())
    avg_soh = battery_df["soh"].mean()

    total_fsd_vehicles = fsd_df["vehicle_id"].nunique()
    total_fsd_events = len(fsd_df)
    risky_fsd_events = int(fsd_df["is_risky_event"].sum())
    interventions = int(fsd_df["driver_intervention"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Battery Vehicles", f"{total_battery_vehicles}", "Battery telemetry monitored")
    with c2:
        metric_card("Battery Rows", f"{total_battery_rows:,}", "Synthetic telemetry records")
    with c3:
        metric_card("FSD Events", f"{total_fsd_events:,}", "Driving scenario events")
    with c4:
        metric_card("Risky FSD Events", f"{risky_fsd_events:,}", "Scenario mining labels")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Platform Modules")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🔋 Battery Intelligence")
        st.write(
            "Detects abnormal battery behavior and forecasts State-of-Health using telemetry such as "
            "voltage, current, temperature, SOC, capacity, cycle count, impedance, and engineered health features."
        )
        st.metric("Battery Anomaly Rows", f"{battery_anomaly_rows:,}")
        st.metric("Average SOH", f"{avg_soh:.4f}")

    with col_b:
        st.markdown("### 🧠 FSD Scenario Intelligence")
        st.write(
            "Mines risky driving scenarios using FSD-style event signals such as speed, lead-vehicle distance, "
            "relative speed, lane offset, braking, pedestrian count, weather, road type, and driver intervention."
        )
        st.metric("Driver Interventions", f"{interventions:,}")
        st.metric("FSD Risk Rate", f"{(risky_fsd_events / total_fsd_events) * 100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Fleet Risk Snapshot")

    risk_summary = pd.DataFrame(
        {
            "Module": ["Battery Intelligence", "FSD Scenario Intelligence"],
            "Total Records": [total_battery_rows, total_fsd_events],
            "Risk Records": [battery_anomaly_rows, risky_fsd_events],
            "Risk Rate %": [
                round((battery_anomaly_rows / total_battery_rows) * 100, 2),
                round((risky_fsd_events / total_fsd_events) * 100, 2),
            ],
        }
    )

    fig = px.bar(
        risk_summary,
        x="Module",
        y="Risk Rate %",
        text="Risk Rate %",
        title="Risk Rate by Intelligence Module",
    )
    fig = plot_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(risk_summary, use_container_width=True)


elif page == "Battery Intelligence":
    st.header("Battery Intelligence")

    total_vehicles = battery_df["vehicle_id"].nunique()
    total_rows = len(battery_df)
    anomaly_rows = int(battery_df["is_anomaly"].sum())
    avg_soh = battery_df["soh"].mean()
    min_soh = battery_df["soh"].min()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Vehicles", f"{total_vehicles}", "Battery fleet size")
    with c2:
        metric_card("Telemetry Rows", f"{total_rows:,}", "Battery records")
    with c3:
        metric_card("Anomaly Rows", f"{anomaly_rows:,}", "Detected risk labels")
    with c4:
        metric_card("Minimum SOH", f"{min_soh:.4f}", "Lowest observed health")

    st.subheader("Battery Anomaly Distribution")

    anomaly_by_vehicle = (
        battery_df.groupby("vehicle_id")["is_anomaly"]
        .sum()
        .reset_index()
        .sort_values("is_anomaly", ascending=False)
    )

    fig = px.bar(
        anomaly_by_vehicle,
        x="vehicle_id",
        y="is_anomaly",
        title="Anomaly Count by Vehicle",
        labels={"vehicle_id": "Vehicle ID", "is_anomaly": "Anomaly Count"},
    )
    fig = plot_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average SOH by Vehicle")

    soh_by_vehicle = (
        battery_df.groupby("vehicle_id")["soh"]
        .mean()
        .reset_index()
        .sort_values("soh")
    )

    fig2 = px.bar(
        soh_by_vehicle,
        x="vehicle_id",
        y="soh",
        title="Average Battery State-of-Health by Vehicle",
        labels={"vehicle_id": "Vehicle ID", "soh": "Average SOH"},
    )
    fig2 = plot_dark_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Temperature vs Impedance Risk Map")

    sampled = battery_df.sample(min(4000, len(battery_df)), random_state=42)

    fig3 = px.scatter(
        sampled,
        x="temperature",
        y="impedance",
        color="is_anomaly",
        hover_data=["vehicle_id", "soh", "cycle_count"],
        title="Battery Thermal and Impedance Risk Space",
        labels={
            "temperature": "Temperature",
            "impedance": "Impedance",
            "is_anomaly": "Anomaly",
        },
    )
    fig3 = plot_dark_layout(fig3)
    st.plotly_chart(fig3, use_container_width=True)


elif page == "Vehicle Deep Dive":
    st.header("Vehicle Deep Dive")

    vehicle_ids = sorted(battery_df["vehicle_id"].unique())
    selected_vehicle = st.selectbox("Select Battery Vehicle", vehicle_ids)

    vehicle_df = battery_df[battery_df["vehicle_id"] == selected_vehicle].copy()
    latest = vehicle_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Latest Voltage", f"{latest['voltage']:.3f} V", selected_vehicle)
    with c2:
        metric_card("Temperature", f"{latest['temperature']:.2f} °C", "Latest reading")
    with c3:
        metric_card("Impedance", f"{latest['impedance']:.5f} Ω", "Internal resistance")
    with c4:
        metric_card("SOH", f"{latest['soh']:.4f}", "State-of-Health")

    st.subheader("Battery Telemetry Timeline")

    fig_temp = px.line(
        vehicle_df,
        x="timestamp",
        y="temperature",
        color="is_anomaly",
        title=f"Temperature Trend — {selected_vehicle}",
    )
    fig_temp = plot_dark_layout(fig_temp)
    st.plotly_chart(fig_temp, use_container_width=True)

    fig_soh = px.line(
        vehicle_df,
        x="timestamp",
        y="soh",
        color="is_anomaly",
        title=f"SOH Trend — {selected_vehicle}",
    )
    fig_soh = plot_dark_layout(fig_soh)
    st.plotly_chart(fig_soh, use_container_width=True)

    st.subheader("Run Battery ML + GenAI Risk Explanation")

    payload = {
        "vehicle_id": selected_vehicle,
        "telemetry": {
            "voltage": float(latest["voltage"]),
            "current": float(latest["current"]),
            "temperature": float(latest["temperature"]),
            "soc": float(latest["soc"]),
            "capacity": float(latest["capacity"]),
            "cycle_count": int(latest["cycle_count"]),
            "impedance": float(latest["impedance"]),
            "soh": float(latest["soh"]),
        },
    }

    if st.button("Run Battery Intelligence Check"):
        try:
            response = requests.post(f"{API_URL}/explain/risk", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            risk_box(result["risk_level"], f"Risk Level: {result['risk_level']}", result["summary"])

            st.markdown("### Retrieved Battery Context")
            for item in result["retrieved_context"]:
                st.write(f"- {item}")

            st.markdown("### Forecast")
            st.write(result["forecast_summary"])

            st.markdown("### Recommended Action")
            st.write(result["recommended_action"])

        except Exception as e:
            st.error(f"Battery API request failed: {e}")


elif page == "FSD Scenario Intelligence":
    st.header("FSD Scenario Intelligence")

    total_events = len(fsd_df)
    total_vehicles = fsd_df["vehicle_id"].nunique()
    risky_events = int(fsd_df["is_risky_event"].sum())
    interventions = int(fsd_df["driver_intervention"].sum())
    near_collisions = int(fsd_df["near_collision"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("FSD Events", f"{total_events:,}", "Driving event records")
    with c2:
        metric_card("Vehicles", f"{total_vehicles}", "Fleet size")
    with c3:
        metric_card("Risky Events", f"{risky_events:,}", "Scenario risk labels")
    with c4:
        metric_card("Interventions", f"{interventions:,}", "Driver takeovers")
    with c5:
        metric_card("Near Collisions", f"{near_collisions:,}", "Safety-critical events")

    st.subheader("FSD Scenario Type Distribution")

    scenario_counts = fsd_df["scenario_type"].value_counts().reset_index()
    scenario_counts.columns = ["scenario_type", "count"]

    fig_scenario = px.bar(
        scenario_counts,
        x="scenario_type",
        y="count",
        title="Scenario Mining Distribution",
        labels={"scenario_type": "Scenario Type", "count": "Event Count"},
    )
    fig_scenario = plot_dark_layout(fig_scenario)
    st.plotly_chart(fig_scenario, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk by Weather")
        risk_by_weather = (
            fsd_df.groupby("weather_condition")["is_risky_event"]
            .sum()
            .reset_index()
            .sort_values("is_risky_event", ascending=False)
        )

        fig_weather = px.bar(
            risk_by_weather,
            x="weather_condition",
            y="is_risky_event",
            title="Risky Events by Weather Condition",
            labels={"weather_condition": "Weather", "is_risky_event": "Risky Events"},
        )
        fig_weather = plot_dark_layout(fig_weather)
        st.plotly_chart(fig_weather, use_container_width=True)

    with col_right:
        st.subheader("Risk by Road Type")
        risk_by_road = (
            fsd_df.groupby("road_type")["is_risky_event"]
            .sum()
            .reset_index()
            .sort_values("is_risky_event", ascending=False)
        )

        fig_road = px.bar(
            risk_by_road,
            x="road_type",
            y="is_risky_event",
            title="Risky Events by Road Type",
            labels={"road_type": "Road Type", "is_risky_event": "Risky Events"},
        )
        fig_road = plot_dark_layout(fig_road)
        st.plotly_chart(fig_road, use_container_width=True)

    st.subheader("FSD Risk Space")

    sampled_fsd = fsd_df.sample(min(4000, len(fsd_df)), random_state=42)

    fig_space = px.scatter(
        sampled_fsd,
        x="lead_vehicle_distance",
        y="lane_offset",
        color="is_risky_event",
        size="brake_pressure",
        hover_data=["vehicle_id", "scenario_type", "relative_speed", "weather_condition"],
        title="Lead Distance vs Lane Offset Risk Space",
        labels={
            "lead_vehicle_distance": "Lead Vehicle Distance",
            "lane_offset": "Lane Offset",
            "is_risky_event": "Risky Event",
        },
    )
    fig_space = plot_dark_layout(fig_space)
    st.plotly_chart(fig_space, use_container_width=True)

    st.subheader("Run Custom FSD Scenario Risk Prediction")

    vehicle_id = st.text_input("FSD Vehicle ID", "EV_021")

    col_a, col_b = st.columns(2)

    with col_a:
        speed = st.number_input("Speed", value=52.0)
        lead_vehicle_distance = st.number_input("Lead Vehicle Distance", value=7.5)
        relative_speed = st.number_input("Relative Speed", value=-19.0)
        lane_offset = st.number_input("Lane Offset", value=0.88)
        steering_angle = st.number_input("Steering Angle", value=10.5)
        brake_pressure = st.number_input("Brake Pressure", value=0.76)

    with col_b:
        object_count = st.number_input("Object Count", value=14)
        pedestrian_count = st.number_input("Pedestrian Count", value=3)
        autopilot_engaged = st.selectbox("Autopilot Engaged", [1, 0])
        traffic_light_state = st.selectbox("Traffic Light State", ["green", "yellow", "red", "none"], index=2)
        weather_condition = st.selectbox("Weather Condition", ["clear", "rain", "fog", "snow"], index=1)
        road_type = st.selectbox("Road Type", ["highway", "urban", "suburban"], index=1)

    fsd_payload = {
        "vehicle_id": vehicle_id,
        "scenario": {
            "speed": float(speed),
            "lead_vehicle_distance": float(lead_vehicle_distance),
            "relative_speed": float(relative_speed),
            "lane_offset": float(lane_offset),
            "steering_angle": float(steering_angle),
            "brake_pressure": float(brake_pressure),
            "object_count": int(object_count),
            "pedestrian_count": int(pedestrian_count),
            "autopilot_engaged": int(autopilot_engaged),
            "traffic_light_state": traffic_light_state,
            "weather_condition": weather_condition,
            "road_type": road_type,
        },
    }

    if st.button("Run FSD Scenario Risk Check"):
        try:
            response = requests.post(f"{API_URL}/predict/fsd-risk", json=fsd_payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            risk_box(
                result["risk_level"],
                f"FSD Risk Level: {result['risk_level']}",
                result["scenario_summary"],
            )

            st.markdown("### Prediction Response")
            st.json(result)

            st.markdown("### Recommended Action")
            st.write(result["recommended_action"])

        except Exception as e:
            st.error(f"FSD API request failed: {e}")


elif page == "Live API Demo":
    st.header("Live API Demo")

    st.write("Use this page to test both Battery Intelligence and FSD Scenario Intelligence from the frontend.")

    tab1, tab2 = st.tabs(["Battery API Demo", "FSD API Demo"])

    with tab1:
        st.subheader("Battery Intelligence Request")

        vehicle_id = st.text_input("Battery Vehicle ID", "EV_DEMO")

        col1, col2 = st.columns(2)

        with col1:
            voltage = st.number_input("Voltage", value=3.92)
            current = st.number_input("Current", value=-3.4)
            temperature = st.number_input("Temperature", value=39.5)
            soc = st.number_input("SOC", value=0.58)

        with col2:
            capacity = st.number_input("Capacity", value=91.2)
            cycle_count = st.number_input("Cycle Count", value=420)
            impedance = st.number_input("Impedance", value=0.046, format="%.5f")
            soh = st.number_input("SOH", value=0.912, format="%.4f")

        custom_payload = {
            "vehicle_id": vehicle_id,
            "telemetry": {
                "voltage": float(voltage),
                "current": float(current),
                "temperature": float(temperature),
                "soc": float(soc),
                "capacity": float(capacity),
                "cycle_count": int(cycle_count),
                "impedance": float(impedance),
                "soh": float(soh),
            },
        }

        if st.button("Run Battery Full Check"):
            try:
                anomaly_response = requests.post(f"{API_URL}/predict/anomaly", json=custom_payload, timeout=10)
                forecast_response = requests.post(f"{API_URL}/predict/forecast", json=custom_payload, timeout=10)
                explain_response = requests.post(f"{API_URL}/explain/risk", json=custom_payload, timeout=10)

                anomaly_response.raise_for_status()
                forecast_response.raise_for_status()
                explain_response.raise_for_status()

                st.write("### Anomaly Prediction")
                st.json(anomaly_response.json())

                st.write("### SOH Forecast")
                st.json(forecast_response.json())

                st.write("### GenAI Risk Explanation")
                st.json(explain_response.json())

            except Exception as e:
                st.error(f"Battery API request failed: {e}")

    with tab2:
        st.subheader("FSD Scenario Request")

        demo_payload = {
            "vehicle_id": "EV_021",
            "scenario": {
                "speed": 52.0,
                "lead_vehicle_distance": 7.5,
                "relative_speed": -19.0,
                "lane_offset": 0.88,
                "steering_angle": 10.5,
                "brake_pressure": 0.76,
                "object_count": 14,
                "pedestrian_count": 3,
                "autopilot_engaged": 1,
                "traffic_light_state": "red",
                "weather_condition": "rain",
                "road_type": "urban",
            },
        }

        st.json(demo_payload)

        if st.button("Run FSD Demo Request"):
            try:
                response = requests.post(f"{API_URL}/predict/fsd-risk", json=demo_payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                risk_box(
                    result["risk_level"],
                    f"FSD Risk Level: {result['risk_level']}",
                    result["scenario_summary"],
                )

                st.json(result)

            except Exception as e:
                st.error(f"FSD API request failed: {e}")


elif page == "Model Operations":
    st.header("Model Operations")

    st.write("MLOps and model artifact overview for the EV Fleet Intelligence Platform.")

    model_files = {
        "Battery Anomaly Model": "models/anomaly_model.joblib",
        "Battery Anomaly Scaler": "models/anomaly_scaler.joblib",
        "SOH Forecast Model": "models/forecast_model.joblib",
        "FSD Scenario Risk Model": "models/fsd_risk_model.joblib",
        "Battery Anomaly Metrics": "models/anomaly_metrics.json",
        "SOH Forecast Metrics": "models/forecast_metrics.json",
        "FSD Risk Metrics": "models/fsd_risk_metrics.json",
        "MLflow SQLite DB": "mlflow.db",
    }

    rows = []
    for name, path in model_files.items():
        exists = os.path.exists(path)
        size_kb = round(os.path.getsize(path) / 1024, 2) if exists else 0
        rows.append(
            {
                "Artifact": name,
                "Path": path,
                "Exists": exists,
                "Size KB": size_kb,
            }
        )

    artifact_df = pd.DataFrame(rows)

    st.subheader("Model Artifacts")
    st.dataframe(artifact_df, use_container_width=True)

    st.subheader("API Health")

    if health:
        st.success("FastAPI is online")
        st.json(health)
    else:
        st.error("FastAPI is offline")

    st.subheader("MLOps Commands")

    st.code(
        """
# Generate battery data
python scripts/generate_battery_data.py

# Build battery features
python app/features/battery_features.py

# Train battery models
python app/ml/train_anomaly_model.py
python app/ml/train_forecast_model.py

# Generate FSD data
python scripts/generate_fsd_data.py

# Build FSD features
python app/features/fsd_features.py

# Train FSD model
python app/ml/train_fsd_risk_model.py

# Run API
uvicorn app.api.main:app --reload

# Run Dashboard
streamlit run app/dashboard/dashboard.py

# Open MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
        """,
        language="bash",
    )


st.markdown(
    """
    <div class="footer">
        EV Fleet Intelligence Platform · Battery Intelligence + FSD Scenario Intelligence · Production-style Applied AI Demo
    </div>
    """,
    unsafe_allow_html=True,
)