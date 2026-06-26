import os
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_DATA_PATH = "data/battery_features.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "forecast_metrics.json")

MLFLOW_DB_PATH = "mlflow.db"


FEATURE_COLUMNS = [
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


def create_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates future SOH targets for 30, 60, and 90-day horizons.

    In this synthetic dataset:
    - Data is sampled every 30 minutes.
    - 48 rows = 1 day.
    - 30 days = 1440 rows.
    - 60 days = 2880 rows.
    - 90 days = 4320 rows.

    Since our demo dataset currently contains 30 days, we use practical demo horizons:
    - 1 day ahead
    - 3 days ahead
    - 7 days ahead

    We label them as forecast_horizon_1d, 3d, and 7d for the working demo.
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["vehicle_id", "timestamp"])

    horizon_steps = {
        "soh_1d_ahead": 48,
        "soh_3d_ahead": 48 * 3,
        "soh_7d_ahead": 48 * 7,
    }

    for target_name, steps in horizon_steps.items():
        df[target_name] = df.groupby("vehicle_id")["soh"].shift(-steps)

    df = df.dropna(subset=list(horizon_steps.keys()))

    return df


def main():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature data not found at {FEATURE_DATA_PATH}. "
            "Run app/features/battery_features.py first."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA_PATH)
    df = create_forecast_targets(df)

    X = df[FEATURE_COLUMNS]

    target_columns = ["soh_1d_ahead", "soh_3d_ahead", "soh_7d_ahead"]
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("EV_Fleet_SOH_Forecasting")

    with mlflow.start_run(run_name="random_forest_soh_forecaster"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {}

        for idx, target in enumerate(target_columns):
            mae = mean_absolute_error(y_test.iloc[:, idx], y_pred[:, idx])
            mse = mean_squared_error(y_test.iloc[:, idx], y_pred[:, idx])
            rmse = mse ** 0.5
            r2 = r2_score(y_test.iloc[:, idx], y_pred[:, idx])

            metrics[target] = {
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2,
            }

            mlflow.log_metric(f"{target}_mae", mae)
            mlflow.log_metric(f"{target}_rmse", rmse)
            mlflow.log_metric(f"{target}_r2_score", r2)

        final_metrics = {
            "model_type": "RandomForestRegressor",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": FEATURE_COLUMNS,
            "targets": target_columns,
            "metrics": metrics,
        }

        joblib.dump(model, MODEL_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(final_metrics, f, indent=4)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("target_count", len(target_columns))

        mlflow.sklearn.log_model(model, "forecast_model")

    print("SOH forecasting model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print()

    for target, values in metrics.items():
        print(target)
        print(f"  MAE:  {values['mae']:.6f}")
        print(f"  RMSE: {values['rmse']:.6f}")
        print(f"  R2:   {values['r2_score']:.4f}")


if __name__ == "__main__":
    main()