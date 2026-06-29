import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat


RAW_DIR = Path("data/nasa_raw")
PROCESSED_DIR = Path("data/nasa_processed")

CYCLE_OUTPUT_PATH = PROCESSED_DIR / "nasa_battery_cycles.csv"
TIMESERIES_OUTPUT_PATH = PROCESSED_DIR / "nasa_battery_discharge_timeseries.csv"


def safe_get(obj: Any, key: str, default=None):
    """
    Safely extracts a field from MATLAB-loaded structures.
    NASA .mat files often load as nested numpy structured arrays.
    """
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)

        if hasattr(obj, "dtype") and obj.dtype.names and key in obj.dtype.names:
            return obj[key]

        if hasattr(obj, key):
            return getattr(obj, key)

    except Exception:
        return default

    return default


def unwrap(value: Any):
    """
    Repeatedly unwraps nested numpy arrays until we get the useful object.
    """
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def to_float_array(value: Any) -> np.ndarray:
    """
    Converts MATLAB arrays into a clean 1D numpy float array.
    """
    value = unwrap(value)

    try:
        arr = np.array(value, dtype=float).flatten()
        return arr
    except Exception:
        return np.array([], dtype=float)


def to_scalar(value: Any) -> Optional[float]:
    """
    Converts MATLAB scalar-like objects into Python float.
    """
    try:
        value = unwrap(value)
        arr = np.array(value, dtype=float).flatten()
        if len(arr) == 0:
            return None
        return float(arr[0])
    except Exception:
        return None


def extract_battery_name(mat_data: Dict[str, Any]) -> Optional[str]:
    """
    NASA battery files usually contain one main key like B0005, B0006, B0007, B0018.
    """
    ignored_keys = {"__header__", "__version__", "__globals__"}

    for key in mat_data.keys():
        if key not in ignored_keys:
            return key

    return None


def extract_cycles_from_battery(battery_name: str, battery_obj: Any) -> List[Dict]:
    """
    Extracts discharge cycles from one NASA battery object.

    Expected NASA structure:
    battery_name
      -> cycle
        -> type: charge/discharge/impedance
        -> ambient_temperature
        -> time
        -> data
            -> Voltage_measured
            -> Current_measured
            -> Temperature_measured
            -> Current_load
            -> Voltage_load
            -> Time
            -> Capacity
    """

    battery_obj = unwrap(battery_obj)
    cycle_data = safe_get(battery_obj, "cycle")

    if cycle_data is None:
        print(f"No cycle data found for {battery_name}")
        return []

    cycle_data = unwrap(cycle_data)

    if not isinstance(cycle_data, np.ndarray):
        cycle_data = np.array([cycle_data])

    rows = []

    for cycle_index, cycle in enumerate(cycle_data.flatten(), start=1):
        cycle = unwrap(cycle)

        cycle_type = unwrap(safe_get(cycle, "type", "unknown"))
        if isinstance(cycle_type, bytes):
            cycle_type = cycle_type.decode("utf-8")

        if isinstance(cycle_type, np.ndarray):
            cycle_type = str(unwrap(cycle_type))

        cycle_type = str(cycle_type).lower()

        if cycle_type != "discharge":
            continue

        ambient_temperature = to_scalar(safe_get(cycle, "ambient_temperature"))
        cycle_time = safe_get(cycle, "time")
        data = unwrap(safe_get(cycle, "data"))

        if data is None:
            continue

        voltage_measured = to_float_array(safe_get(data, "Voltage_measured"))
        current_measured = to_float_array(safe_get(data, "Current_measured"))
        temperature_measured = to_float_array(safe_get(data, "Temperature_measured"))
        current_load = to_float_array(safe_get(data, "Current_load"))
        voltage_load = to_float_array(safe_get(data, "Voltage_load"))
        time_seconds = to_float_array(safe_get(data, "Time"))
        capacity = to_scalar(safe_get(data, "Capacity"))

        n = min(
            len(voltage_measured),
            len(current_measured),
            len(temperature_measured),
            len(time_seconds),
        )

        if n == 0:
            continue

        cycle_capacity = capacity if capacity is not None else np.nan

        for i in range(n):
            rows.append(
                {
                    "battery_id": battery_name,
                    "cycle_index": cycle_index,
                    "sample_index": i,
                    "cycle_type": cycle_type,
                    "ambient_temperature": ambient_temperature,
                    "time_seconds": float(time_seconds[i]),
                    "voltage_measured": float(voltage_measured[i]),
                    "current_measured": float(current_measured[i]),
                    "temperature_measured": float(temperature_measured[i]),
                    "current_load": float(current_load[i]) if i < len(current_load) else np.nan,
                    "voltage_load": float(voltage_load[i]) if i < len(voltage_load) else np.nan,
                    "capacity": cycle_capacity,
                }
            )

    return rows


def build_cycle_level_features(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts time-series discharge samples into one row per battery cycle.
    This gives us cycle-level SOH and health features.
    """

    grouped = timeseries_df.groupby(["battery_id", "cycle_index"], as_index=False)

    cycle_df = grouped.agg(
        capacity=("capacity", "mean"),
        ambient_temperature=("ambient_temperature", "mean"),
        voltage_mean=("voltage_measured", "mean"),
        voltage_min=("voltage_measured", "min"),
        voltage_max=("voltage_measured", "max"),
        current_mean=("current_measured", "mean"),
        current_min=("current_measured", "min"),
        current_max=("current_measured", "max"),
        temperature_mean=("temperature_measured", "mean"),
        temperature_max=("temperature_measured", "max"),
        discharge_duration=("time_seconds", "max"),
        sample_count=("sample_index", "count"),
    )

    cycle_df = cycle_df.sort_values(["battery_id", "cycle_index"])

    cycle_df["initial_capacity"] = cycle_df.groupby("battery_id")["capacity"].transform("first")
    cycle_df["soh"] = cycle_df["capacity"] / cycle_df["initial_capacity"]

    cycle_df["capacity_fade"] = cycle_df.groupby("battery_id")["capacity"].diff().fillna(0)
    cycle_df["voltage_drop"] = cycle_df["voltage_max"] - cycle_df["voltage_min"]
    cycle_df["temperature_rise"] = cycle_df["temperature_max"] - cycle_df["temperature_mean"]

    cycle_df["is_low_soh"] = (cycle_df["soh"] < 0.80).astype(int)

    return cycle_df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = list(RAW_DIR.glob("*.mat"))

    if not mat_files:
        print("No NASA .mat files found.")
        print(f"Place files like B0005.mat, B0006.mat, B0007.mat, B0018.mat inside: {RAW_DIR}")
        return

    all_rows = []

    for mat_file in mat_files:
        print(f"Reading: {mat_file}")

        try:
            mat_data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            print(f"Failed to read {mat_file}: {exc}")
            continue

        battery_name = extract_battery_name(mat_data)

        if battery_name is None:
            print(f"No battery key found inside {mat_file}")
            continue

        battery_obj = mat_data[battery_name]
        battery_rows = extract_cycles_from_battery(battery_name, battery_obj)

        print(f"Extracted rows from {battery_name}: {len(battery_rows)}")
        all_rows.extend(battery_rows)

    if not all_rows:
        print("No discharge rows extracted. Please confirm NASA .mat files are valid.")
        return

    timeseries_df = pd.DataFrame(all_rows)

    timeseries_df.to_csv(TIMESERIES_OUTPUT_PATH, index=False)

    cycle_df = build_cycle_level_features(timeseries_df)
    cycle_df.to_csv(CYCLE_OUTPUT_PATH, index=False)

    print()
    print("NASA preprocessing completed successfully.")
    print(f"Time-series output: {TIMESERIES_OUTPUT_PATH}")
    print(f"Cycle-level output: {CYCLE_OUTPUT_PATH}")
    print()
    print(f"Time-series rows: {len(timeseries_df):,}")
    print(f"Cycle-level rows: {len(cycle_df):,}")
    print(f"Batteries: {cycle_df['battery_id'].nunique()}")
    print()
    print("Cycle-level preview:")
    print(cycle_df.head())


if __name__ == "__main__":
    main()