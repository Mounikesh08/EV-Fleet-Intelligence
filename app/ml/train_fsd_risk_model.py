import os
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


FEATURE_DATA_PATH = "data/fsd_features.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "fsd_risk_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "fsd_risk_metrics.json")

MLFLOW_DB_PATH = "mlflow.db"


FEATURE_COLUMNS = [
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


def main():
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"FSD feature data not found at {FEATURE_DATA_PATH}. "
            "Run app/features/fsd_features.py first."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["is_risky_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("FSD_Scenario_Risk_Model")

    with mlflow.start_run(run_name="random_forest_fsd_scenario_risk"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred).tolist()

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        feature_importance = dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        metrics = {
            "model_type": "RandomForestClassifier",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": FEATURE_COLUMNS,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix,
            "classification_report": report,
            "feature_importance": feature_importance,
        }

        joblib.dump(model, MODEL_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 250)
        mlflow.log_param("max_depth", 14)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "fsd_risk_model")

    print("FSD scenario risk model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(matrix)
    print()
    print("Top Feature Importances:")
    for feature, value in list(feature_importance.items())[:10]:
        print(f"{feature}: {value:.4f}")


if __name__ == "__main__":
    main()