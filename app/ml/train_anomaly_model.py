import os
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_DATA_PATH = "data/battery_features.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "anomaly_scaler.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "anomaly_metrics.json")

MLFLOW_DB_PATH = "mlflow.db"


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


def main():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature data not found at {FEATURE_DATA_PATH}. "
            "Run app/features/battery_features.py first."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["is_anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.07,
        random_state=42,
        n_jobs=-1,
    )

    # MLflow SQLite backend.
    # This avoids Windows path issues and avoids MLflow's file-store maintenance-mode error.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("EV_Fleet_Anomaly_Detection")

    with mlflow.start_run(run_name="isolation_forest_anomaly_detector"):
        model.fit(X_train_scaled)

        raw_preds = model.predict(X_test_scaled)

        # IsolationForest output:
        #  1 = normal
        # -1 = anomaly
        y_pred = [1 if pred == -1 else 0 for pred in raw_preds]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "model_type": "IsolationForest",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": FEATURE_COLUMNS,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix,
            "classification_report": report,
        }

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("contamination", 0.07)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "anomaly_model")

    print("Anomaly detection model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"MLflow database saved to: {MLFLOW_DB_PATH}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()