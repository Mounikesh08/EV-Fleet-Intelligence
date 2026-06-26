import json


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def main():
    anomaly = load_json("models/anomaly_metrics.json")
    forecast = load_json("models/forecast_metrics.json")
    fsd = load_json("models/fsd_risk_metrics.json")

    print("=" * 70)
    print("EV FLEET INTELLIGENCE PLATFORM - MODEL PERFORMANCE")
    print("=" * 70)

    print("\nBATTERY ANOMALY DETECTION")
    print(f"Precision: {anomaly['precision']:.4f}")
    print(f"Recall:    {anomaly['recall']:.4f}")
    print(f"F1 Score:  {anomaly['f1_score']:.4f}")

    print("\nBATTERY SOH FORECASTING")
    for horizon, metrics in forecast["metrics"].items():
        print(f"{horizon}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R2:   {metrics['r2_score']:.4f}")

    print("\nFSD SCENARIO RISK CLASSIFICATION")
    print(f"Precision: {fsd['precision']:.4f}")
    print(f"Recall:    {fsd['recall']:.4f}")
    print(f"F1 Score:  {fsd['f1_score']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()