import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


OUTPUT_PATH = "data/fsd_driving_events.csv"


def generate_event(vehicle_id: str, timestamp: datetime, risky: bool = False):
    speed = np.random.normal(55, 12)
    speed = max(0, min(speed, 95))

    lead_vehicle_distance = np.random.normal(45, 15)
    lead_vehicle_distance = max(2, lead_vehicle_distance)

    relative_speed = np.random.normal(0, 8)
    lane_offset = np.random.normal(0, 0.25)
    steering_angle = np.random.normal(0, 4)
    brake_pressure = max(0, np.random.normal(0.15, 0.08))

    object_count = np.random.poisson(5)
    pedestrian_count = np.random.poisson(1)

    traffic_light_state = np.random.choice(
        ["green", "yellow", "red", "none"],
        p=[0.45, 0.12, 0.18, 0.25],
    )

    weather_condition = np.random.choice(
        ["clear", "rain", "fog", "snow"],
        p=[0.70, 0.18, 0.08, 0.04],
    )

    road_type = np.random.choice(
        ["highway", "urban", "suburban"],
        p=[0.45, 0.35, 0.20],
    )

    autopilot_engaged = int(np.random.random() > 0.18)

    driver_intervention = 0
    near_collision = 0
    scenario_type = "normal_driving"

    if risky:
        risk_case = np.random.choice(
            [
                "cut_in",
                "hard_brake",
                "lane_departure",
                "pedestrian_crossing",
                "red_light_risk",
                "low_visibility",
            ]
        )

        scenario_type = risk_case

        if risk_case == "cut_in":
            lead_vehicle_distance = np.random.uniform(4, 12)
            relative_speed = np.random.uniform(-25, -8)
            brake_pressure = np.random.uniform(0.5, 0.95)

        elif risk_case == "hard_brake":
            lead_vehicle_distance = np.random.uniform(3, 10)
            relative_speed = np.random.uniform(-30, -12)
            brake_pressure = np.random.uniform(0.7, 1.0)

        elif risk_case == "lane_departure":
            lane_offset = np.random.uniform(0.75, 1.5)
            steering_angle = np.random.uniform(8, 18)

        elif risk_case == "pedestrian_crossing":
            pedestrian_count = np.random.randint(3, 8)
            speed = np.random.uniform(25, 45)
            brake_pressure = np.random.uniform(0.5, 1.0)

        elif risk_case == "red_light_risk":
            traffic_light_state = "red"
            speed = np.random.uniform(30, 60)
            brake_pressure = np.random.uniform(0.05, 0.25)

        elif risk_case == "low_visibility":
            weather_condition = np.random.choice(["fog", "snow", "rain"])
            object_count = np.random.randint(8, 18)
            lane_offset = np.random.uniform(0.4, 1.1)

        driver_intervention = int(np.random.random() > 0.35)
        near_collision = int(np.random.random() > 0.70)

    return {
        "vehicle_id": vehicle_id,
        "timestamp": timestamp,
        "speed": round(speed, 3),
        "lead_vehicle_distance": round(lead_vehicle_distance, 3),
        "relative_speed": round(relative_speed, 3),
        "lane_offset": round(lane_offset, 3),
        "steering_angle": round(steering_angle, 3),
        "brake_pressure": round(brake_pressure, 3),
        "object_count": int(object_count),
        "pedestrian_count": int(pedestrian_count),
        "traffic_light_state": traffic_light_state,
        "weather_condition": weather_condition,
        "road_type": road_type,
        "autopilot_engaged": autopilot_engaged,
        "driver_intervention": driver_intervention,
        "near_collision": near_collision,
        "scenario_type": scenario_type,
        "is_risky_event": int(risky),
    }


def main():
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    vehicle_ids = [f"EV_{i:03d}" for i in range(1, 51)]
    start_time = datetime(2026, 1, 1)

    rows = []
    events_per_vehicle = 500

    for vehicle_id in vehicle_ids:
        for i in range(events_per_vehicle):
            timestamp = start_time + timedelta(minutes=5 * i)
            risky = np.random.random() < 0.12
            rows.append(generate_event(vehicle_id, timestamp, risky=risky))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"FSD driving events saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Vehicles: {df['vehicle_id'].nunique()}")
    print(f"Risky events: {df['is_risky_event'].sum()}")
    print(df.head())


if __name__ == "__main__":
    main()