import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports/validation")
REPORT_PATH = REPORT_DIR / "data_quality_validation_report.json"

NASA_CYCLES_PATH = Path("data/nasa_processed/nasa_battery_cycles.csv")
NASA_TIMESERIES_PATH = Path("data/nasa_processed/nasa_battery_discharge_timeseries.csv")
FSD_FEATURES_PATH = Path("data/fsd_features.csv")


def check_required_columns(df: pd.DataFrame, required_columns, dataset_name: str, results: list):
    missing = [col for col in required_columns if col not in df.columns]

    passed = len(missing) == 0

    results.append(
        {
            "dataset": dataset_name,
            "check": "required_columns",
            "passed": passed,
            "details": {
                "missing_columns": missing,
                "required_columns": required_columns,
            },
        }
    )


def check_not_null(df: pd.DataFrame, columns, dataset_name: str, results: list):
    for column in columns:
        if column not in df.columns:
            continue

        null_count = int(df[column].isna().sum())
        passed = null_count == 0

        results.append(
            {
                "dataset": dataset_name,
                "check": f"{column}_not_null",
                "passed": passed,
                "details": {
                    "null_count": null_count,
                    "total_rows": int(len(df)),
                },
            }
        )


def check_range(df: pd.DataFrame, column: str, min_value, max_value, dataset_name: str, results: list):
    if column not in df.columns:
        return

    invalid_mask = (df[column] < min_value) | (df[column] > max_value)
    invalid_count = int(invalid_mask.sum())
    passed = invalid_count == 0

    results.append(
        {
            "dataset": dataset_name,
            "check": f"{column}_between_{min_value}_and_{max_value}",
            "passed": passed,
            "details": {
                "invalid_count": invalid_count,
                "min_observed": float(df[column].min()) if len(df) else None,
                "max_observed": float(df[column].max()) if len(df) else None,
            },
        }
    )


def check_allowed_values(df: pd.DataFrame, column: str, allowed_values, dataset_name: str, results: list):
    if column not in df.columns:
        return

    observed_values = set(df[column].dropna().unique().tolist())
    allowed_set = set(allowed_values)

    invalid_values = sorted(list(observed_values - allowed_set))
    passed = len(invalid_values) == 0

    results.append(
        {
            "dataset": dataset_name,
            "check": f"{column}_allowed_values",
            "passed": passed,
            "details": {
                "allowed_values": allowed_values,
                "invalid_values": invalid_values,
            },
        }
    )


def validate_nasa_cycles(results: list):
    dataset_name = "NASA Battery Cycles"

    if not NASA_CYCLES_PATH.exists():
        results.append(
            {
                "dataset": dataset_name,
                "check": "file_exists",
                "passed": False,
                "details": {"path": str(NASA_CYCLES_PATH)},
            }
        )
        return

    df = pd.read_csv(NASA_CYCLES_PATH)

    required_columns = [
        "battery_id",
        "cycle_index",
        "capacity",
        "ambient_temperature",
        "voltage_mean",
        "voltage_min",
        "voltage_max",
        "current_mean",
        "temperature_mean",
        "temperature_max",
        "discharge_duration",
        "sample_count",
        "initial_capacity",
        "soh",
        "capacity_fade",
        "voltage_drop",
        "temperature_rise",
        "is_low_soh",
    ]

    check_required_columns(df, required_columns, dataset_name, results)
    check_not_null(df, ["battery_id", "cycle_index", "capacity", "soh"], dataset_name, results)

    check_range(df, "soh", 0.0, 1.2, dataset_name, results)
    check_range(df, "capacity", 0.0, 3.0, dataset_name, results)
    check_range(df, "voltage_min", 0.0, 5.0, dataset_name, results)
    check_range(df, "voltage_max", 0.0, 5.0, dataset_name, results)
    check_range(df, "temperature_mean", 0.0, 80.0, dataset_name, results)
    check_range(df, "temperature_max", 0.0, 100.0, dataset_name, results)
    check_range(df, "sample_count", 1, 10000, dataset_name, results)
    check_allowed_values(df, "is_low_soh", [0, 1], dataset_name, results)


def validate_nasa_timeseries(results: list):
    dataset_name = "NASA Battery Discharge Time-Series"

    if not NASA_TIMESERIES_PATH.exists():
        results.append(
            {
                "dataset": dataset_name,
                "check": "file_exists",
                "passed": False,
                "details": {"path": str(NASA_TIMESERIES_PATH)},
            }
        )
        return

    df = pd.read_csv(NASA_TIMESERIES_PATH)

    required_columns = [
        "battery_id",
        "cycle_index",
        "sample_index",
        "cycle_type",
        "ambient_temperature",
        "time_seconds",
        "voltage_measured",
        "current_measured",
        "temperature_measured",
        "current_load",
        "voltage_load",
        "capacity",
    ]

    check_required_columns(df, required_columns, dataset_name, results)
    check_not_null(
        df,
        [
            "battery_id",
            "cycle_index",
            "sample_index",
            "time_seconds",
            "voltage_measured",
            "current_measured",
            "temperature_measured",
            "capacity",
        ],
        dataset_name,
        results,
    )

    check_range(df, "voltage_measured", 0.0, 5.0, dataset_name, results)
    check_range(df, "current_measured", -10.0, 10.0, dataset_name, results)
    check_range(df, "temperature_measured", 0.0, 100.0, dataset_name, results)
    check_range(df, "voltage_load", 0.0, 5.0, dataset_name, results)
    check_range(df, "capacity", 0.0, 3.0, dataset_name, results)
    check_range(df, "sample_index", 0, 10000, dataset_name, results)


def validate_fsd_features(results: list):
    dataset_name = "FSD Feature Dataset"

    if not FSD_FEATURES_PATH.exists():
        results.append(
            {
                "dataset": dataset_name,
                "check": "file_exists",
                "passed": False,
                "details": {"path": str(FSD_FEATURES_PATH)},
            }
        )
        return

    df = pd.read_csv(FSD_FEATURES_PATH)

    required_columns = [
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
        "driver_intervention",
        "near_collision",
        "scenario_type",
        "is_risky_event",
        "time_gap",
        "closing_risk",
        "lane_departure_risk",
        "hard_brake_risk",
    ]

    check_required_columns(df, required_columns, dataset_name, results)
    check_not_null(df, ["vehicle_id", "timestamp", "speed", "is_risky_event"], dataset_name, results)

    check_range(df, "speed", 0.0, 150.0, dataset_name, results)
    check_range(df, "lead_vehicle_distance", 0.0, 300.0, dataset_name, results)
    check_range(df, "lane_offset", 0.0, 5.0, dataset_name, results)
    check_range(df, "brake_pressure", 0.0, 1.0, dataset_name, results)
    check_range(df, "object_count", 0, 100, dataset_name, results)
    check_range(df, "pedestrian_count", 0, 50, dataset_name, results)

    check_allowed_values(df, "driver_intervention", [0, 1], dataset_name, results)
    check_allowed_values(df, "near_collision", [0, 1], dataset_name, results)
    check_allowed_values(df, "is_risky_event", [0, 1], dataset_name, results)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    validate_nasa_cycles(results)
    validate_nasa_timeseries(results)
    validate_fsd_features(results)

    total_checks = len(results)
    passed_checks = sum(1 for item in results if item["passed"])
    failed_checks = total_checks - passed_checks

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "Great Expectations-style data quality validation",
        "note": (
            "This script implements explicit expectation checks for the project datasets. "
            "It is designed to mirror production data-quality validation before model training."
        ),
        "summary": {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "pass_rate": round((passed_checks / total_checks) * 100, 2) if total_checks else 0,
        },
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Data quality validation completed.")
    print(f"Report saved to: {REPORT_PATH}")
    print()
    print("Summary:")
    print(f"Total checks:  {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Failed checks: {failed_checks}")
    print(f"Pass rate:     {report['summary']['pass_rate']}%")

    if failed_checks > 0:
        print()
        print("Failed checks:")
        for item in results:
            if not item["passed"]:
                print(f"- {item['dataset']} | {item['check']} | {item['details']}")


if __name__ == "__main__":
    main()