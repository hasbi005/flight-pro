
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


# Project paths
BASE_DIR = Path(__file__).parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "daily_airport_delays.csv"
)

MODEL_FILE = BASE_DIR / "models" / "delay_lstm.pt"


# Same LSTM architecture used during training
class DelayLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x):
        result, _ = self.lstm(x)
        return self.output(result[:, -1, :])


def predict_airport_delay(airport_code=None):

    # Check required files
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained LSTM model not found: {MODEL_FILE}. "
            "Run python train_lstm.py first."
        )

    print("Loading trained LSTM model...")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = DelayLSTM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading historical airport data...")

    df = pd.read_csv(
        DATA_FILE,
        dtype={"ORIGIN_AIRPORT": str},
    )

    df["FLIGHT_DATE"] = pd.to_datetime(
        df["FLIGHT_DATE"]
    )

    df = df.sort_values(
        ["ORIGIN_AIRPORT", "FLIGHT_DATE"]
    )

    # Select an airport
    if airport_code is not None:
        airport_code = str(airport_code).strip()

        df = df[
            df["ORIGIN_AIRPORT"] == airport_code
        ]

        if df.empty:
            raise ValueError(
                f"No historical data found for airport "
                f"{airport_code}. Check its identifier."
            )

    else:
        # Use an airport with sufficient observations
        counts = df.groupby("ORIGIN_AIRPORT").size()

        eligible = counts[counts >= 7].index

        if len(eligible) == 0:
            raise ValueError(
                "No airport has enough historical observations."
            )

        airport_code = str(eligible[0])

        df = df[
            df["ORIGIN_AIRPORT"] == airport_code
        ]

    # Keep only rows with valid delay values
    df = df.dropna(
        subset=["AVG_ARRIVAL_DELAY", "FLIGHT_DATE"]
    )

    df = df.sort_values("FLIGHT_DATE")

    if len(df) < 7:
        raise ValueError(
            "At least seven valid delay observations are required."
        )

    # Get the last seven recorded observations
    last_rows = df.tail(7)

    values = last_rows[
        "AVG_ARRIVAL_DELAY"
    ].to_numpy(dtype=np.float32)

    # Apply the same scaling used in training
    x_mean = float(checkpoint["x_mean"])
    x_std = float(checkpoint["x_std"]) or 1.0

    y_mean = float(checkpoint["y_mean"])
    y_std = float(checkpoint["y_std"]) or 1.0

    scaled_values = (values - x_mean) / x_std

    input_tensor = torch.tensor(
        scaled_values,
        dtype=torch.float32,
    ).reshape(1, checkpoint["window"], 1)

    # Predict the next observation
    with torch.no_grad():
        predicted_scaled = model(input_tensor).item()

    predicted_delay = predicted_scaled * y_std + y_mean

    print("\n--- LSTM DELAY PREDICTION ---")
    print("Airport:", airport_code)
    print(
        "Latest historical date:",
        last_rows["FLIGHT_DATE"].iloc[-1].date(),
    )
    print("Last seven recorded delays:")

    for i, value in enumerate(values, start=1):
        print(f"  Observation {i}: {value:.2f} minutes")

    print(
        f"\nPredicted arrival delay: "
        f"{predicted_delay:.2f} minutes"
    )

    return float(predicted_delay)


if __name__ == "__main__":
    predict_airport_delay()
