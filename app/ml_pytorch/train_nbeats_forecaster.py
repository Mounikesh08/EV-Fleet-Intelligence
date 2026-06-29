import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data/nasa_processed/nasa_battery_cycles.csv")
MODEL_DIR = Path("models")

FORECAST_OUTPUT_PATH = MODEL_DIR / "nbeats_soh_forecast_predictions.csv"
METRICS_PATH = MODEL_DIR / "nbeats_soh_metrics.json"
MODEL_SAVE_DIR = MODEL_DIR / "nbeats_soh_forecaster"

MLFLOW_DB_PATH = "mlflow.db"

HORIZON = 7
INPUT_SIZE = 30
MAX_STEPS = 300
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def load_cycle_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NASA cycle-level data not found at {DATA_PATH}. "
            "Run app/nasa/prepare_nasa_battery_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = ["battery_id", "cycle_index", "soh"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["battery_id", "cycle_index", "soh"])
    df = df.sort_values(["battery_id", "cycle_index"]).copy()

    return df


def prepare_neuralforecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    NeuralForecast expects:
    unique_id = time-series ID
    ds        = timestamp
    y         = target value

    NASA cycle_index is numeric, so we convert cycle order into pseudo-dates.
    This keeps each battery as one time series.
    """

    rows = []

    base_date = datetime(2026, 1, 1)

    for battery_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle_index").reset_index(drop=True)

        for i, row in group.iterrows():
            rows.append(
                {
                    "unique_id": battery_id,
                    "ds": base_date + timedelta(days=i),
                    "y": float(row["soh"]),
                    "cycle_index": int(row["cycle_index"]),
                }
            )

    nf_df = pd.DataFrame(rows)
    nf_df = nf_df.sort_values(["unique_id", "ds"])

    return nf_df


def train_test_split_by_battery(nf_df: pd.DataFrame):
    """
    Uses the last HORIZON cycles of each battery as test set.
    Trains on all earlier cycles.
    """

    train_parts = []
    test_parts = []

    for battery_id, group in nf_df.groupby("unique_id"):
        group = group.sort_values("ds")

        if len(group) <= HORIZON + INPUT_SIZE:
            raise ValueError(
                f"Battery {battery_id} has too few cycles for "
                f"INPUT_SIZE={INPUT_SIZE} and HORIZON={HORIZON}."
            )

        train_parts.append(group.iloc[:-HORIZON])
        test_parts.append(group.iloc[-HORIZON:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def calculate_metrics(merged: pd.DataFrame, prediction_col: str):
    y_true = merged["y"].values
    y_pred = merged[prediction_col].values

    overall_metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2_score": float(r2_score(y_true, y_pred)),
    }

    horizon_metrics = {}

    for horizon_step in [1, 3, 7]:
        step_df = merged[merged["horizon_step"] == horizon_step]

        if len(step_df) == 0:
            continue

        step_true = step_df["y"].values
        step_pred = step_df[prediction_col].values

        horizon_metrics[f"{horizon_step}_cycle_ahead"] = {
            "mae": float(mean_absolute_error(step_true, step_pred)),
            "rmse": float(mean_squared_error(step_true, step_pred) ** 0.5),
            "r2_score": float(r2_score(step_true, step_pred))
            if len(step_true) > 1
            else None,
        }

    return overall_metrics, horizon_metrics


def main():
    np.random.seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NASA cycle-level SOH data...")
    cycle_df = load_cycle_data()

    print(f"Cycle rows: {len(cycle_df):,}")
    print(f"Batteries: {cycle_df['battery_id'].nunique()}")

    nf_df = prepare_neuralforecast_data(cycle_df)

    print("Prepared NeuralForecast data:")
    print(nf_df.head())

    train_df, test_df = train_test_split_by_battery(nf_df)

    print()
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Forecast horizon: {HORIZON} cycles")

    model = NBEATS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED,
        scaler_type="standard",
    )

    nf = NeuralForecast(
        models=[model],
        freq="D",
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("NASA_NBEATS_SOH_Forecasting")

    with mlflow.start_run(run_name="nbeats_soh_forecaster_nasa"):
        mlflow.log_param("model_type", "N-BEATS")
        mlflow.log_param("dataset", "NASA Battery Dataset")
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("input_size", INPUT_SIZE)
        mlflow.log_param("max_steps", MAX_STEPS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("frequency", "D")
        mlflow.log_param("target", "SOH")

        print()
        print("Training N-BEATS SOH forecaster...")
        nf.fit(df=train_df[["unique_id", "ds", "y"]])

        print("Generating forecasts...")
        forecast_df = nf.predict()

        prediction_columns = [
            col for col in forecast_df.columns if col not in ["unique_id", "ds"]
        ]

        if not prediction_columns:
            raise ValueError("No prediction column found in forecast output.")

        prediction_col = prediction_columns[0]

        merged = test_df.merge(
            forecast_df,
            on=["unique_id", "ds"],
            how="inner",
        )

        if merged.empty:
            raise ValueError(
                "Forecast and test data did not align. "
                "Check ds/frequency handling."
            )

        merged = merged.sort_values(["unique_id", "ds"]).copy()
        merged["horizon_step"] = (
            merged.groupby("unique_id").cumcount() + 1
        )

        overall_metrics, horizon_metrics = calculate_metrics(
            merged=merged,
            prediction_col=prediction_col,
        )

        metrics = {
            "model_type": "N-BEATS",
            "dataset": "NASA Battery Dataset",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": "SOH",
            "horizon": HORIZON,
            "input_size": INPUT_SIZE,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "prediction_column": prediction_col,
            "overall_metrics": overall_metrics,
            "horizon_metrics": horizon_metrics,
        }

        merged.to_csv(FORECAST_OUTPUT_PATH, index=False)

        with open(METRICS_PATH, "w") as file:
            json.dump(metrics, file, indent=4)

        nf.save(
            path=str(MODEL_SAVE_DIR),
            overwrite=True,
            save_dataset=False,
        )

        mlflow.log_metric("overall_mae", overall_metrics["mae"])
        mlflow.log_metric("overall_rmse", overall_metrics["rmse"])
        mlflow.log_metric("overall_r2_score", overall_metrics["r2_score"])

        for horizon_name, values in horizon_metrics.items():
            mlflow.log_metric(f"{horizon_name}_mae", values["mae"])
            mlflow.log_metric(f"{horizon_name}_rmse", values["rmse"])

            if values["r2_score"] is not None:
                mlflow.log_metric(f"{horizon_name}_r2_score", values["r2_score"])

        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(FORECAST_OUTPUT_PATH))

    print()
    print("NASA N-BEATS SOH forecaster trained successfully.")
    print(f"Forecast predictions saved to: {FORECAST_OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Model directory saved to: {MODEL_SAVE_DIR}")

    print()
    print("Overall Metrics:")
    print(f"MAE:  {overall_metrics['mae']:.6f}")
    print(f"RMSE: {overall_metrics['rmse']:.6f}")
    print(f"R2:   {overall_metrics['r2_score']:.4f}")

    print()
    print("Horizon Metrics:")
    for horizon_name, values in horizon_metrics.items():
        print(horizon_name)
        print(f"  MAE:  {values['mae']:.6f}")
        print(f"  RMSE: {values['rmse']:.6f}")
        if values["r2_score"] is not None:
            print(f"  R2:   {values['r2_score']:.4f}")
        else:
            print("  R2:   N/A")


if __name__ == "__main__":
    main()