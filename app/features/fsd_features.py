import os

import pandas as pd


RAW_DATA_PATH = "data/fsd_driving_events.csv"
FEATURE_DATA_PATH = "data/fsd_features.csv"


def build_fsd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["vehicle_id", "timestamp"])

    df["time_gap"] = df["lead_vehicle_distance"] / df["speed"].replace(0, 1)
    df["closing_risk"] = (-df["relative_speed"]).clip(lower=0)
    df["lane_departure_risk"] = df["lane_offset"].abs()
    df["hard_brake_risk"] = df["brake_pressure"]
    df["crowded_scene_risk"] = df["object_count"] + (df["pedestrian_count"] * 3)

    df["low_visibility_flag"] = df["weather_condition"].isin(["fog", "snow", "rain"]).astype(int)
    df["red_light_flag"] = (df["traffic_light_state"] == "red").astype(int)
    df["urban_flag"] = (df["road_type"] == "urban").astype(int)
    df["highway_flag"] = (df["road_type"] == "highway").astype(int)

    df["intervention_or_collision"] = (
        (df["driver_intervention"] == 1) | (df["near_collision"] == 1)
    ).astype(int)

    feature_columns = [
        "vehicle_id",
        "timestamp",
        "speed",
        "lead_vehicle_distance",
        "relative_speed",
        "lane_offset",
        "steering_angle",
        "brake_pressure",
        "object_count",
        "pedestrian_count",
        "autopilot_engaged",
        "traffic_light_state",
        "weather_condition",
        "road_type",
        "scenario_type",
        "time_gap",
        "closing_risk",
        "lane_departure_risk",
        "hard_brake_risk",
        "crowded_scene_risk",
        "low_visibility_flag",
        "red_light_flag",
        "urban_flag",
        "highway_flag",
        "driver_intervention",
        "near_collision",
        "intervention_or_collision",
        "is_risky_event",
    ]

    return df[feature_columns]


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw FSD data not found at {RAW_DATA_PATH}. "
            "Run scripts/generate_fsd_data.py first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    features_df = build_fsd_features(df)

    features_df.to_csv(FEATURE_DATA_PATH, index=False)

    print(f"FSD feature dataset saved to: {FEATURE_DATA_PATH}")
    print(f"Rows: {len(features_df)}")
    print(f"Columns: {len(features_df.columns)}")
    print(features_df.head())


if __name__ == "__main__":
    main()