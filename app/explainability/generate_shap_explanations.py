import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


DATA_PATH = Path("data/fsd_features.csv")
MODEL_PATH = Path("models/fsd_risk_model.joblib")

OUTPUT_DIR = Path("reports/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_VALUES_CSV = OUTPUT_DIR / "fsd_shap_values_sample.csv"
TOP_FEATURES_JSON = OUTPUT_DIR / "fsd_shap_top_features.json"


TARGET_COLUMN = "is_risky_event"

NON_FEATURE_COLUMNS = [
    "vehicle_id",
    "timestamp",
    "scenario_type",
    "is_risky_event",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"FSD feature data not found at {DATA_PATH}. "
            "Run: python app/features/fsd_features.py"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FSD risk model not found at {MODEL_PATH}. "
            "Run: python app/ml/train_fsd_risk_model.py"
        )

    return joblib.load(MODEL_PATH)


def get_feature_columns(df: pd.DataFrame, model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    feature_columns = [
        col for col in numeric_columns
        if col not in NON_FEATURE_COLUMNS
    ]

    return feature_columns


def normalize_shap_values(shap_values):
    """
    Handles SHAP output shape differences across versions.

    For binary classification, SHAP may return:
    - list[class_0_values, class_1_values]
    - array with shape (samples, features, classes)
    - array with shape (samples, features)
    """

    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return shap_values[1]
        return shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def main():
    print("Loading FSD feature data...")
    df = load_data()

    print("Loading FSD risk model...")
    model = load_model()

    feature_columns = get_feature_columns(df, model)

    print(f"Feature columns used: {len(feature_columns)}")
    print(feature_columns)

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    sample_size = min(1000, len(X))
    X_sample = X.sample(sample_size, random_state=42)

    print(f"SHAP sample size: {sample_size}")

    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    shap_values_raw = explainer.shap_values(X_sample)
    shap_values = normalize_shap_values(shap_values_raw)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top_features = importance_df.head(15)

    top_features_payload = {
        "model": "FSD Scenario Risk Classifier",
        "method": "SHAP TreeExplainer",
        "sample_size": int(sample_size),
        "target": TARGET_COLUMN,
        "top_features": top_features.to_dict(orient="records"),
    }

    with open(TOP_FEATURES_JSON, "w") as file:
        json.dump(top_features_payload, file, indent=4)

    shap_sample_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{col}" for col in feature_columns],
    )

    shap_sample_df.to_csv(SHAP_VALUES_CSV, index=False)

    print()
    print("SHAP explainability completed successfully.")
    print(f"Top features saved to: {TOP_FEATURES_JSON}")
    print(f"SHAP values sample saved to: {SHAP_VALUES_CSV}")
    print()
    print("Top SHAP Features:")
    print(top_features)


if __name__ == "__main__":
    main()