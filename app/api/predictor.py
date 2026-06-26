import os
from typing import Dict

import joblib
import pandas as pd


ANOMALY_MODEL_PATH = "models/anomaly_model.joblib"
ANOMALY_SCALER_PATH = "models/anomaly_scaler.joblib"
FORECAST_MODEL_PATH = "models/forecast_model.joblib"
FSD_RISK_MODEL_PATH = "models/fsd_risk_model.joblib"


FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "soc",
    "capacity",
    "cycle_count",
    "impedance",
    "soh",
    "delta_voltage",
    "rolling_mean_temp_10",
    "rolling_max_temp_10",
    "rolling_std_current_10",
    "soc_change",
    "capacity_fade_rate",
    "impedance_rise",
    "temperature_risk_score",
    "impedance_risk_score",
    "soh_drop_from_new",
]


FORECAST_FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "soc",
    "capacity",
    "cycle_count",
    "impedance",
    "delta_voltage",
    "rolling_mean_temp_10",
    "rolling_max_temp_10",
    "rolling_std_current_10",
    "soc_change",
    "capacity_fade_rate",
    "impedance_rise",
    "temperature_risk_score",
    "impedance_risk_score",
    "soh_drop_from_new",
]


FSD_FEATURE_COLUMNS = [
    "speed",
    "lead_vehicle_distance",
    "relative_speed",
    "lane_offset",
    "steering_angle",
    "brake_pressure",
    "object_count",
    "pedestrian_count",
    "autopilot_engaged",
    "time_gap",
    "closing_risk",
    "lane_departure_risk",
    "hard_brake_risk",
    "crowded_scene_risk",
    "low_visibility_flag",
    "red_light_flag",
    "urban_flag",
    "highway_flag",
]


def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


anomaly_model = load_model(ANOMALY_MODEL_PATH)
anomaly_scaler = load_model(ANOMALY_SCALER_PATH)
forecast_model = load_model(FORECAST_MODEL_PATH)
fsd_risk_model = load_model(FSD_RISK_MODEL_PATH)


def engineer_single_record_features(record: Dict) -> pd.DataFrame:
    voltage = record["voltage"]
    current = record["current"]
    temperature = record["temperature"]
    soc = record["soc"]
    capacity = record["capacity"]
    cycle_count = record["cycle_count"]
    impedance = record["impedance"]
    soh = record["soh"]

    features = {
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "soc": soc,
        "capacity": capacity,
        "cycle_count": cycle_count,
        "impedance": impedance,
        "soh": soh,
        "delta_voltage": 0.0,
        "rolling_mean_temp_10": temperature,
        "rolling_max_temp_10": temperature,
        "rolling_std_current_10": 0.0,
        "soc_change": 0.0,
        "capacity_fade_rate": 0.0,
        "impedance_rise": 0.0,
        "temperature_risk_score": max(0.0, temperature - 25.0),
        "impedance_risk_score": max(0.0, impedance - 0.03),
        "soh_drop_from_new": 1.0 - soh,
    }

    return pd.DataFrame([features])


def get_risk_level(anomaly_detected: bool, temperature: float, impedance: float, soh: float) -> str:
    if anomaly_detected and (temperature > 35 or impedance > 0.04 or soh < 0.90):
        return "HIGH"
    if anomaly_detected:
        return "MEDIUM"
    return "LOW"


def predict_anomaly(record: Dict) -> Dict:
    if anomaly_model is None or anomaly_scaler is None:
        raise RuntimeError("Anomaly model or scaler is not loaded. Train the model first.")

    features_df = engineer_single_record_features(record)
    X = features_df[FEATURE_COLUMNS]
    X_scaled = anomaly_scaler.transform(X)

    raw_prediction = anomaly_model.predict(X_scaled)[0]
    anomaly_detected = bool(raw_prediction == -1)

    anomaly_score = float(anomaly_model.decision_function(X_scaled)[0])

    risk_level = get_risk_level(
        anomaly_detected=anomaly_detected,
        temperature=record["temperature"],
        impedance=record["impedance"],
        soh=record["soh"],
    )

    if anomaly_detected:
        message = "Battery behavior is abnormal. Vehicle should be reviewed by fleet maintenance team."
    else:
        message = "Battery behavior is within expected operating range."

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_score": round(anomaly_score, 6),
        "risk_level": risk_level,
        "message": message,
    }


def predict_forecast(record: Dict) -> Dict[str, float]:
    if forecast_model is None:
        raise RuntimeError("Forecast model is not loaded. Train the forecast model first.")

    features_df = engineer_single_record_features(record)
    X = features_df[FORECAST_FEATURE_COLUMNS]

    prediction = forecast_model.predict(X)[0]

    return {
        "1_day": round(float(prediction[0]), 4),
        "3_days": round(float(prediction[1]), 4),
        "7_days": round(float(prediction[2]), 4),
    }


def engineer_fsd_features(record: Dict) -> pd.DataFrame:
    speed = record["speed"]
    lead_vehicle_distance = record["lead_vehicle_distance"]
    relative_speed = record["relative_speed"]
    lane_offset = record["lane_offset"]
    steering_angle = record["steering_angle"]
    brake_pressure = record["brake_pressure"]
    object_count = record["object_count"]
    pedestrian_count = record["pedestrian_count"]
    autopilot_engaged = record["autopilot_engaged"]
    traffic_light_state = record["traffic_light_state"]
    weather_condition = record["weather_condition"]
    road_type = record["road_type"]

    features = {
        "speed": speed,
        "lead_vehicle_distance": lead_vehicle_distance,
        "relative_speed": relative_speed,
        "lane_offset": lane_offset,
        "steering_angle": steering_angle,
        "brake_pressure": brake_pressure,
        "object_count": object_count,
        "pedestrian_count": pedestrian_count,
        "autopilot_engaged": autopilot_engaged,
        "time_gap": lead_vehicle_distance / max(speed, 1),
        "closing_risk": max(0.0, -relative_speed),
        "lane_departure_risk": abs(lane_offset),
        "hard_brake_risk": brake_pressure,
        "crowded_scene_risk": object_count + (pedestrian_count * 3),
        "low_visibility_flag": int(weather_condition in ["fog", "snow", "rain"]),
        "red_light_flag": int(traffic_light_state == "red"),
        "urban_flag": int(road_type == "urban"),
        "highway_flag": int(road_type == "highway"),
    }

    return pd.DataFrame([features])


def get_fsd_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def predict_fsd_risk(record: Dict) -> Dict:
    if fsd_risk_model is None:
        raise RuntimeError("FSD risk model is not loaded. Train the FSD risk model first.")

    features_df = engineer_fsd_features(record)
    X = features_df[FSD_FEATURE_COLUMNS]

    probability = float(fsd_risk_model.predict_proba(X)[0][1])
    detected = probability >= 0.50
    risk_level = get_fsd_risk_level(probability)

    reasons = []

    if record["lead_vehicle_distance"] < 12:
        reasons.append("short lead-vehicle distance")

    if record["relative_speed"] < -10:
        reasons.append("rapid closing speed")

    if abs(record["lane_offset"]) > 0.7:
        reasons.append("lane departure risk")

    if record["brake_pressure"] > 0.6:
        reasons.append("hard braking signal")

    if record["pedestrian_count"] >= 3:
        reasons.append("dense pedestrian scene")

    if record["traffic_light_state"] == "red" and record["speed"] > 25:
        reasons.append("red-light approach risk")

    if record["weather_condition"] in ["fog", "snow", "rain"]:
        reasons.append("low-visibility weather")

    if reasons:
        reason_text = ", ".join(reasons)
    else:
        reason_text = "no dominant high-risk signal"

    scenario_summary = (
        f"FSD scenario risk is {risk_level}. "
        f"Main contributing signals: {reason_text}."
    )

    if risk_level == "HIGH":
        recommended_action = (
            "Prioritize this scenario for fleet review, model evaluation, and training data mining."
        )
    elif risk_level == "MEDIUM":
        recommended_action = (
            "Store this event for offline analysis and compare against similar intervention scenarios."
        )
    else:
        recommended_action = (
            "No immediate action required. Continue normal fleet analytics monitoring."
        )

    return {
        "fsd_risk_detected": detected,
        "risk_probability": round(probability, 4),
        "risk_level": risk_level,
        "scenario_summary": scenario_summary,
        "recommended_action": recommended_action,
    }


def models_loaded_status() -> Dict[str, bool]:
    return {
        "anomaly_model_loaded": anomaly_model is not None and anomaly_scaler is not None,
        "forecast_model_loaded": forecast_model is not None,
        "fsd_risk_model_loaded": fsd_risk_model is not None,
    }