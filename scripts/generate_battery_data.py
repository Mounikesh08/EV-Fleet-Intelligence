import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_vehicle_data(vehicle_id: str, start_time: datetime, periods: int, is_risky: bool = False):
    rows = []

    base_voltage = 4.15
    base_current = -2.0
    base_temp = 25.0
    base_soc = 1.0
    base_capacity = 100.0
    base_impedance = 0.030

    for i in range(periods):
        timestamp = start_time + timedelta(minutes=30 * i)

        cycle_count = i // 48
        soc = max(0.05, base_soc - (i % 48) * 0.018)

        voltage = base_voltage - (i % 48) * 0.005 + np.random.normal(0, 0.01)
        current = base_current + np.random.normal(0, 0.15)
        temperature = base_temp + np.random.normal(0, 1.0)
        capacity = base_capacity - cycle_count * 0.015 + np.random.normal(0, 0.05)
        impedance = base_impedance + cycle_count * 0.00002 + np.random.normal(0, 0.0005)

        is_anomaly = 0

        if is_risky and i > periods * 0.65:
            temperature += np.random.uniform(5, 12)
            current += np.random.normal(0, 0.8)
            impedance += np.random.uniform(0.005, 0.015)
            capacity -= np.random.uniform(1.0, 3.0)
            voltage -= np.random.uniform(0.05, 0.15)
            is_anomaly = 1

        soh = max(0.60, capacity / base_capacity)

        rows.append(
            {
                "vehicle_id": vehicle_id,
                "timestamp": timestamp,
                "voltage": round(voltage, 4),
                "current": round(current, 4),
                "temperature": round(temperature, 4),
                "soc": round(soc, 4),
                "capacity": round(capacity, 4),
                "cycle_count": cycle_count,
                "impedance": round(impedance, 5),
                "soh": round(soh, 4),
                "is_anomaly": is_anomaly,
            }
        )

    return rows


def main():
    np.random.seed(42)

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime(2026, 1, 1)
    periods = 30 * 48

    all_rows = []

    vehicle_ids = [f"EV_{i:03d}" for i in range(1, 21)]

    for idx, vehicle_id in enumerate(vehicle_ids):
        is_risky = idx in [4, 9, 14, 19]
        all_rows.extend(
            generate_vehicle_data(
                vehicle_id=vehicle_id,
                start_time=start_time,
                periods=periods,
                is_risky=is_risky,
            )
        )

    df = pd.DataFrame(all_rows)

    output_path = os.path.join(output_dir, "battery_telemetry.csv")
    df.to_csv(output_path, index=False)

    print(f"Generated battery telemetry data: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Vehicles: {df['vehicle_id'].nunique()}")
    print(f"Anomaly rows: {df['is_anomaly'].sum()}")
    print(df.head())


if __name__ == "__main__":
    main()