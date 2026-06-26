import requests


API_URL = "http://127.0.0.1:8000"


battery_payload = {
    "vehicle_id": "EV_005",
    "telemetry": {
        "voltage": 3.92,
        "current": -3.4,
        "temperature": 39.5,
        "soc": 0.58,
        "capacity": 91.2,
        "cycle_count": 420,
        "impedance": 0.046,
        "soh": 0.912,
    },
}


fsd_payload = {
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


def call_api(method: str, endpoint: str, payload=None):
    url = f"{API_URL}{endpoint}"

    if method == "GET":
        response = requests.get(url, timeout=10)
    else:
        response = requests.post(url, json=payload, timeout=10)

    print(f"\n{method} {endpoint}")
    print(f"Status Code: {response.status_code}")

    response.raise_for_status()

    data = response.json()
    print(data)

    return data


def main():
    print("Running EV Fleet Intelligence Platform smoke tests...")

    health = call_api("GET", "/health")
    assert health["status"] == "ok"
    assert health["anomaly_model_loaded"] is True
    assert health["forecast_model_loaded"] is True
    assert health["fsd_risk_model_loaded"] is True

    anomaly = call_api("POST", "/predict/anomaly", battery_payload)
    assert "anomaly_detected" in anomaly
    assert "risk_level" in anomaly

    forecast = call_api("POST", "/predict/forecast", battery_payload)
    assert "soh_forecast" in forecast
    assert "1_day" in forecast["soh_forecast"]
    assert "3_days" in forecast["soh_forecast"]
    assert "7_days" in forecast["soh_forecast"]

    explanation = call_api("POST", "/explain/risk", battery_payload)
    assert "summary" in explanation
    assert "recommended_action" in explanation

    fsd = call_api("POST", "/predict/fsd-risk", fsd_payload)
    assert "fsd_risk_detected" in fsd
    assert "risk_probability" in fsd
    assert "scenario_summary" in fsd

    print("\nAll smoke tests passed successfully.")


if __name__ == "__main__":
    main()