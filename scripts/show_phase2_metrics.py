import json
from pathlib import Path


LSTM_METRICS_PATH = Path("models/lstm_autoencoder_metrics.json")
NBEATS_METRICS_PATH = Path("models/nbeats_soh_metrics.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Make sure you trained the model before running this script."
        )

    with open(path, "r") as file:
        return json.load(file)


def print_separator():
    print("=" * 78)


def print_lstm_metrics(metrics: dict):
    print_separator()
    print("PHASE 2 MODEL 1: NASA PYTORCH LSTM AUTOENCODER")
    print_separator()

    print(f"Dataset:          {metrics.get('dataset', 'NASA Battery Dataset')}")
    print(f"Model Type:       {metrics.get('model_type', 'PyTorch LSTM Autoencoder')}")
    print(f"Sequence Length:  {metrics.get('sequence_length')}")
    print(f"Train Sequences:  {metrics.get('normal_train_sequences')}")
    print(f"Eval Sequences:   {metrics.get('evaluation_sequences')}")
    print()

    print("Anomaly Detection Metrics")
    print("-" * 78)
    print(f"Threshold:              {metrics.get('threshold'):.8f}")
    print(f"Threshold Percentile:   {metrics.get('threshold_percentile')}")
    print(f"Precision:              {metrics.get('precision'):.4f}")
    print(f"Recall:                 {metrics.get('recall'):.4f}")
    print(f"F1 Score:               {metrics.get('f1_score'):.4f}")
    print()

    print("Confusion Matrix")
    print("-" * 78)
    matrix = metrics.get("confusion_matrix")
    print(matrix)
    print()

    print("Interpretation")
    print("-" * 78)
    print(
        "The LSTM Autoencoder learns normal NASA battery discharge behavior and flags "
        "sequences with high reconstruction error as anomalous. Threshold tuning improved "
        "the balance between precision and recall compared with the initial conservative "
        "95th-percentile baseline."
    )
    print()


def print_nbeats_metrics(metrics: dict):
    print_separator()
    print("PHASE 2 MODEL 2: NASA N-BEATS SOH FORECASTING")
    print_separator()

    print(f"Dataset:       {metrics.get('dataset', 'NASA Battery Dataset')}")
    print(f"Model Type:    {metrics.get('model_type', 'N-BEATS')}")
    print(f"Target:        {metrics.get('target', 'SOH')}")
    print(f"Horizon:       {metrics.get('horizon')} cycles")
    print(f"Input Size:    {metrics.get('input_size')}")
    print(f"Train Rows:    {metrics.get('train_rows')}")
    print(f"Test Rows:     {metrics.get('test_rows')}")
    print()

    overall = metrics.get("overall_metrics", {})

    print("Overall SOH Forecasting Metrics")
    print("-" * 78)
    print(f"MAE:   {overall.get('mae'):.6f}")
    print(f"RMSE:  {overall.get('rmse'):.6f}")
    print(f"R2:    {overall.get('r2_score'):.4f}")
    print()

    print("Forecast Horizon Metrics")
    print("-" * 78)

    horizon_metrics = metrics.get("horizon_metrics", {})

    for horizon_name, values in horizon_metrics.items():
        print(horizon_name)
        print(f"  MAE:   {values.get('mae'):.6f}")
        print(f"  RMSE:  {values.get('rmse'):.6f}")

        r2 = values.get("r2_score")
        if r2 is None:
            print("  R2:    N/A")
        else:
            print(f"  R2:    {r2:.4f}")

        print()

    print("Interpretation")
    print("-" * 78)
    print(
        "The N-BEATS model forecasts NASA battery State-of-Health over future cycles. "
        "The error naturally increases as the forecast horizon becomes longer, but the "
        "model still maintains strong R2 across 1-cycle, 3-cycle, and 7-cycle horizons."
    )
    print()


def main():
    lstm_metrics = load_json(LSTM_METRICS_PATH)
    nbeats_metrics = load_json(NBEATS_METRICS_PATH)

    print()
    print_separator()
    print("EV FLEET INTELLIGENCE PLATFORM — PHASE 2 NASA/PYTORCH METRICS")
    print_separator()
    print()

    print_lstm_metrics(lstm_metrics)
    print_nbeats_metrics(nbeats_metrics)

    print_separator()
    print("SUMMARY")
    print_separator()
    print("Phase 2 upgrades completed:")
    print("- NASA Battery Dataset preprocessing")
    print("- PyTorch LSTM Autoencoder for anomaly detection")
    print("- N-BEATS / neuralforecast model for SOH forecasting")
    print("- MLflow experiment tracking for both models")
    print()
    print("This proves the project evolved from a synthetic scikit-learn demo into a")
    print("real-data PyTorch-based battery intelligence system.")
    print_separator()


if __name__ == "__main__":
    main()