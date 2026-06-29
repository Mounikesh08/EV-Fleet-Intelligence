from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


DATA_PATH = Path("data/fsd_features.csv")
REPORT_DIR = Path("reports/drift")
REPORT_HTML_PATH = REPORT_DIR / "fsd_data_drift_report.html"
REPORT_JSON_PATH = REPORT_DIR / "fsd_data_drift_report.json"


TARGET_COLUMN = "is_risky_event"

EXCLUDED_COLUMNS = [
    "vehicle_id",
    "timestamp",
    "scenario_type",
    TARGET_COLUMN,
]


def load_fsd_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"FSD feature data not found at {DATA_PATH}. "
            "Run: python scripts/generate_fsd_data.py and python app/features/fsd_features.py"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("FSD feature dataset is empty.")

    return df


def prepare_drift_data(df: pd.DataFrame):
    """
    Creates reference/current datasets for drift monitoring.

    In a real production system:
    - reference_data = training baseline
    - current_data = latest production batch

    For this demo:
    - reference_data = first 70% of FSD events by timestamp
    - current_data = last 30% of FSD events by timestamp
    """

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    feature_columns = [
        col for col in numeric_columns
        if col not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found for drift detection.")

    split_index = int(len(df) * 0.70)

    reference_data = df.iloc[:split_index][feature_columns].copy()
    current_data = df.iloc[split_index:][feature_columns].copy()

    if reference_data.empty or current_data.empty:
        raise ValueError("Reference or current drift dataset is empty.")

    return reference_data, current_data, feature_columns


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FSD feature data...")
    df = load_fsd_features()

    print(f"Rows: {len(df):,}")

    reference_data, current_data, feature_columns = prepare_drift_data(df)

    print(f"Reference rows: {len(reference_data):,}")
    print(f"Current rows:   {len(current_data):,}")
    print(f"Features used:  {len(feature_columns)}")
    print(feature_columns)

    print()
    print("Generating Evidently data drift report...")

    report = Report(
        [
            DataDriftPreset(method="psi")
        ],
        include_tests=True,
    )

    evaluation = report.run(reference_data, current_data)

    evaluation.save_html(str(REPORT_HTML_PATH))

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as file:
        file.write(evaluation.json())

    print()
    print("Evidently drift report generated successfully.")
    print(f"HTML report saved to: {REPORT_HTML_PATH}")
    print(f"JSON report saved to: {REPORT_JSON_PATH}")
    print()
    print("Open the HTML report locally in your browser to inspect feature drift.")


if __name__ == "__main__":
    main()