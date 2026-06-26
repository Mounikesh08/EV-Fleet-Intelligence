from typing import Dict, List


BATTERY_KNOWLEDGE_BASE = [
    {
        "condition": "high_temperature",
        "text": "High battery temperature can indicate thermal stress, cooling inefficiency, aggressive charging, or early cell-level instability.",
    },
    {
        "condition": "high_impedance",
        "text": "Rising internal impedance often indicates battery aging, increased resistance, and reduced efficiency under load.",
    },
    {
        "condition": "low_soh",
        "text": "Low State-of-Health means the battery has lost usable capacity compared to its original condition.",
    },
    {
        "condition": "low_voltage",
        "text": "Unexpected voltage drop may indicate cell imbalance, degradation, or abnormal discharge behavior.",
    },
    {
        "condition": "high_current_variation",
        "text": "Unstable current behavior can suggest irregular charging, sensor noise, or stress on the battery management system.",
    },
]


def retrieve_relevant_context(record: Dict) -> List[str]:
    context = []

    if record["temperature"] >= 35:
        context.append(BATTERY_KNOWLEDGE_BASE[0]["text"])

    if record["impedance"] >= 0.04:
        context.append(BATTERY_KNOWLEDGE_BASE[1]["text"])

    if record["soh"] <= 0.90:
        context.append(BATTERY_KNOWLEDGE_BASE[2]["text"])

    if record["voltage"] <= 3.95:
        context.append(BATTERY_KNOWLEDGE_BASE[3]["text"])

    if abs(record["current"]) >= 3.0:
        context.append(BATTERY_KNOWLEDGE_BASE[4]["text"])

    if not context:
        context.append(
            "Battery telemetry is within normal operating range based on available rule-based context."
        )

    return context


def generate_risk_explanation(vehicle_id: str, record: Dict, anomaly_result: Dict, forecast_result: Dict) -> Dict:
    retrieved_context = retrieve_relevant_context(record)

    risk_level = anomaly_result["risk_level"]
    anomaly_detected = anomaly_result["anomaly_detected"]

    reasons = []

    if record["temperature"] >= 35:
        reasons.append(f"temperature is elevated at {record['temperature']}°C")

    if record["impedance"] >= 0.04:
        reasons.append(f"internal impedance is high at {record['impedance']} ohms")

    if record["soh"] <= 0.90:
        reasons.append(f"State-of-Health is low at {record['soh']}")

    if record["voltage"] <= 3.95:
        reasons.append(f"voltage is lower than expected at {record['voltage']}V")

    if abs(record["current"]) >= 3.0:
        reasons.append(f"current draw/charge behavior is aggressive at {record['current']}A")

    if reasons:
        reason_text = ", ".join(reasons)
    else:
        reason_text = "battery telemetry does not show strong rule-based risk signals"

    if anomaly_detected:
        summary = (
            f"Vehicle {vehicle_id} was flagged as {risk_level} risk because {reason_text}. "
            f"The anomaly model detected behavior outside the learned normal operating pattern."
        )
    else:
        summary = (
            f"Vehicle {vehicle_id} is currently {risk_level} risk. "
            f"The anomaly model did not detect abnormal behavior, and {reason_text}."
        )

    forecast_message = (
        f"The SOH forecast is {forecast_result['1_day']} after 1 day, "
        f"{forecast_result['3_days']} after 3 days, and "
        f"{forecast_result['7_days']} after 7 days."
    )

    if risk_level == "HIGH":
        recommended_action = (
            "Schedule diagnostic inspection, review recent thermal and impedance trends, "
            "and prioritize this vehicle for fleet maintenance."
        )
    elif risk_level == "MEDIUM":
        recommended_action = (
            "Continue monitoring this vehicle and compare the next telemetry window "
            "against recent operating history."
        )
    else:
        recommended_action = (
            "No immediate maintenance action required. Continue normal fleet monitoring."
        )

    return {
        "vehicle_id": vehicle_id,
        "risk_level": risk_level,
        "summary": summary,
        "retrieved_context": retrieved_context,
        "forecast_summary": forecast_message,
        "recommended_action": recommended_action,
    }