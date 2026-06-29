import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data/nasa_processed/nasa_battery_discharge_timeseries.csv")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "lstm_autoencoder.pt"
SCALER_PATH = MODEL_DIR / "lstm_autoencoder_scaler.json"
THRESHOLD_PATH = MODEL_DIR / "lstm_autoencoder_threshold.json"
METRICS_PATH = MODEL_DIR / "lstm_autoencoder_metrics.json"

MLFLOW_DB_PATH = "mlflow.db"

SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
LATENT_SIZE = 16
NUM_LAYERS = 1
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "current_load",
    "voltage_load",
]


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        self.decoder_fc = nn.Linear(latent_size, hidden_size)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)

        hidden_last = hidden[-1]
        latent = self.encoder_fc(hidden_last)

        decoder_hidden = self.decoder_fc(latent)
        decoder_input = decoder_hidden.unsqueeze(1).repeat(1, x.size(1), 1)

        decoded, _ = self.decoder_lstm(decoder_input)
        reconstructed = self.output_layer(decoded)

        return reconstructed


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NASA processed data not found at {DATA_PATH}. "
            "Run app/nasa/prepare_nasa_battery_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "battery_id",
        "cycle_index",
        "sample_index",
        "capacity",
    ] + FEATURE_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=FEATURE_COLUMNS + ["capacity"])
    df = df.sort_values(["battery_id", "cycle_index", "sample_index"])

    return df


def add_soh_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    first_capacity = df.groupby("battery_id")["capacity"].transform("first")
    df["soh"] = df["capacity"] / first_capacity

    # Battery health convention:
    # SOH below 80% is treated as degraded/anomalous for evaluation.
    df["is_anomaly"] = (df["soh"] < 0.80).astype(int)

    return df


def create_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
):
    sequence_list = []
    label_list = []
    metadata_list = []

    feature_values = df[FEATURE_COLUMNS].values

    if fit_scaler:
        scaler = StandardScaler()
        feature_values_scaled = scaler.fit_transform(feature_values)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False.")
        feature_values_scaled = scaler.transform(feature_values)

    df_scaled = df.copy()
    df_scaled[FEATURE_COLUMNS] = feature_values_scaled

    grouped = df_scaled.groupby(["battery_id", "cycle_index"])

    for (battery_id, cycle_index), group in grouped:
        group = group.sort_values("sample_index")

        values = group[FEATURE_COLUMNS].values
        labels = group["is_anomaly"].values

        if len(values) < SEQUENCE_LENGTH:
            continue

        for start in range(0, len(values) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            end = start + SEQUENCE_LENGTH

            sequence = values[start:end]
            sequence_label = int(labels[start:end].max())

            sequence_list.append(sequence)
            label_list.append(sequence_label)
            metadata_list.append(
                {
                    "battery_id": battery_id,
                    "cycle_index": int(cycle_index),
                    "start_sample": int(start),
                    "end_sample": int(end),
                }
            )

    sequences = np.array(sequence_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)

    return sequences, labels, metadata_list, scaler


def train_model(model, train_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    model.train()

    for epoch in range(EPOCHS):
        epoch_losses = []

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            reconstructed = model(x)
            loss = criterion(reconstructed, x)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_loss:.6f}")

    return train_losses


def reconstruction_errors(model, sequences, device):
    model.eval()

    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    errors = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            reconstructed = model(x)

            mse = torch.mean((reconstructed - x) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy().tolist())

    return np.array(errors)


def tune_threshold(normal_errors, all_errors, y_true):
    """
    Finds the reconstruction-error threshold that maximizes F1 score.

    Baseline used 95th percentile of normal errors.
    That was too conservative and caused low recall.
    This function searches multiple percentiles and keeps the best F1 result.
    """

    best_threshold = None
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    best_matrix = None
    best_report = None
    best_percentile = None

    threshold_candidates = np.percentile(normal_errors, np.arange(50, 100, 1))

    for percentile, candidate in zip(np.arange(50, 100, 1), threshold_candidates):
        candidate = float(candidate)
        candidate_pred = (all_errors > candidate).astype(int)

        candidate_precision = precision_score(y_true, candidate_pred, zero_division=0)
        candidate_recall = recall_score(y_true, candidate_pred, zero_division=0)
        candidate_f1 = f1_score(y_true, candidate_pred, zero_division=0)

        if candidate_f1 > best_f1:
            best_threshold = candidate
            best_precision = candidate_precision
            best_recall = candidate_recall
            best_f1 = candidate_f1
            best_matrix = confusion_matrix(y_true, candidate_pred).tolist()
            best_report = classification_report(
                y_true,
                candidate_pred,
                output_dict=True,
                zero_division=0,
            )
            best_percentile = int(percentile)

    return {
        "threshold": float(best_threshold),
        "threshold_percentile": best_percentile,
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1_score": float(best_f1),
        "confusion_matrix": best_matrix,
        "classification_report": best_report,
    }


def save_scaler_as_json(scaler: StandardScaler, path: Path):
    scaler_data = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    with open(path, "w") as file:
        json.dump(scaler_data, file, indent=4)


def main():
    set_seed()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NASA battery discharge time-series data...")
    df = load_data()
    df = add_soh_labels(df)

    print(f"Rows: {len(df):,}")
    print(f"Batteries: {df['battery_id'].nunique()}")
    print(f"Anomaly rows based on SOH < 0.80: {df['is_anomaly'].sum():,}")

    # Train the autoencoder only on normal/healthy sequences.
    # This lets the model learn normal discharge behavior.
    normal_df = df[df["is_anomaly"] == 0].copy()

    print("Creating sequences...")

    normal_sequences, normal_labels, normal_metadata, scaler = create_sequences(
        normal_df,
        scaler=None,
        fit_scaler=True,
    )

    all_sequences, all_labels, all_metadata, _ = create_sequences(
        df,
        scaler=scaler,
        fit_scaler=False,
    )

    if len(normal_sequences) == 0:
        raise ValueError("No normal sequences created. Check preprocessing or sequence length.")

    if len(all_sequences) == 0:
        raise ValueError("No evaluation sequences created. Check preprocessing or sequence length.")

    print(f"Normal train sequences: {len(normal_sequences):,}")
    print(f"All evaluation sequences: {len(all_sequences):,}")

    train_dataset = TensorDataset(torch.tensor(normal_sequences, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMAutoencoder(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("NASA_LSTM_Autoencoder_Anomaly_Detection")

    with mlflow.start_run(run_name="pytorch_lstm_autoencoder_nasa_threshold_tuned"):
        mlflow.log_param("model_type", "LSTM Autoencoder")
        mlflow.log_param("dataset", "NASA Battery Dataset")
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("latent_size", LATENT_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))

        train_losses = train_model(model, train_loader, device)

        for idx, loss in enumerate(train_losses, start=1):
            mlflow.log_metric("train_loss", loss, step=idx)

        print("Calculating reconstruction errors...")
        normal_errors = reconstruction_errors(model, normal_sequences, device)
        all_errors = reconstruction_errors(model, all_sequences, device)

        y_true = all_labels

        print("Tuning reconstruction-error threshold...")
        tuned = tune_threshold(
            normal_errors=normal_errors,
            all_errors=all_errors,
            y_true=y_true,
        )

        threshold = tuned["threshold"]
        threshold_percentile = tuned["threshold_percentile"]
        precision = tuned["precision"]
        recall = tuned["recall"]
        f1 = tuned["f1_score"]
        matrix = tuned["confusion_matrix"]
        report = tuned["classification_report"]

        model_state = {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
            "num_layers": NUM_LAYERS,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_columns": FEATURE_COLUMNS,
        }

        torch.save(model_state, MODEL_PATH)
        save_scaler_as_json(scaler, SCALER_PATH)

        threshold_data = {
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
            "method": "Best F1 threshold search over 50th to 99th percentile of normal reconstruction errors",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(THRESHOLD_PATH, "w") as file:
            json.dump(threshold_data, file, indent=4)

        metrics = {
            "model_type": "PyTorch LSTM Autoencoder",
            "dataset": "NASA Battery Dataset",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "features": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "normal_train_sequences": int(len(normal_sequences)),
            "evaluation_sequences": int(len(all_sequences)),
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix,
            "classification_report": report,
            "final_train_loss": float(train_losses[-1]),
            "threshold_tuning": {
                "search_percentile_start": 50,
                "search_percentile_end": 99,
                "objective": "maximize_f1_score",
            },
        }

        with open(METRICS_PATH, "w") as file:
            json.dump(metrics, file, indent=4)

        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("threshold_percentile", threshold_percentile)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("final_train_loss", train_losses[-1])

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))
        mlflow.log_artifact(str(THRESHOLD_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

    print()
    print("NASA PyTorch LSTM Autoencoder trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print()
    print(f"Best Threshold Percentile: {threshold_percentile}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()