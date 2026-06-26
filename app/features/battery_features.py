import os

import pandas as pd


RAW_DATA_PATH = "data/battery_telemetry.csv"
FEATURE_DATA_PATH = "data/battery_features.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw EV battery telemetry into ML-ready battery health features.
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["vehicle_id", "timestamp"])

    # Voltage behavior
    df["delta_voltage"] = df.groupby("vehicle_id")["voltage"].diff().fillna(0)

    # Temperature behavior
    df["rolling_mean_temp_10"] = (
        df.groupby("vehicle_id")["temperature"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_max_temp_10"] = (
        df.groupby("vehicle_id")["temperature"]
        .rolling(window=10, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    # Current instability
    df["rolling_std_current_10"] = (
        df.groupby("vehicle_id")["current"]
        .rolling(window=10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # SOC movement
    df["soc_change"] = df.groupby("vehicle_id")["soc"].diff().fillna(0)

    # Capacity degradation behavior
    df["capacity_fade_rate"] = (
        df.groupby("vehicle_id")["capacity"]
        .diff()
        .fillna(0)
    )

    # Impedance growth
    df["impedance_rise"] = (
        df.groupby("vehicle_id")["impedance"]
        .diff()
        .fillna(0)
    )

    # Risk-style engineered signals
    df["temperature_risk_score"] = (df["temperature"] - 25).clip(lower=0)
    df["impedance_risk_score"] = (df["impedance"] - 0.03).clip(lower=0)
    df["soh_drop_from_new"] = 1.0 - df["soh"]

    feature_columns = [
        "vehicle_id",
        "timestamp",
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
        "is_anomaly",
    ]

    return df[feature_columns]


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run scripts/generate_battery_data.py first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    features_df = build_features(df)

    features_df.to_csv(FEATURE_DATA_PATH, index=False)

    print(f"Feature dataset saved to: {FEATURE_DATA_PATH}")
    print(f"Rows: {len(features_df)}")
    print(f"Columns: {len(features_df.columns)}")
    print(features_df.head())


if __name__ == "__main__":
    main()