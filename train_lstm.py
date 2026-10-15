
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "processed" / "daily_airport_delays.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

WINDOW = 7
EPOCHS = 10
BATCH_SIZE = 256

print("Loading daily airport data...")

df = pd.read_csv(DATA_FILE, dtype={"ORIGIN_AIRPORT": str})
df["FLIGHT_DATE"] = pd.to_datetime(df["FLIGHT_DATE"])
df = df.sort_values(["ORIGIN_AIRPORT", "FLIGHT_DATE"])

# Use airports with enough historical observations.
valid_airports = (
    df.groupby("ORIGIN_AIRPORT").size()
    .loc[lambda counts: counts >= 60]
    .index
)
df = df[df["ORIGIN_AIRPORT"].isin(valid_airports)].copy()

print("Airports used:", df["ORIGIN_AIRPORT"].nunique())
print("Daily records used:", len(df))

# Create 7-observation input windows separately for each airport.
sequences = []
targets = []
dates = []

for airport, group in df.groupby("ORIGIN_AIRPORT"):
    values = group["AVG_ARRIVAL_DELAY"].to_numpy(dtype=np.float32)
    airport_dates = group["FLIGHT_DATE"].to_numpy()

    for i in range(WINDOW, len(values)):
        sequences.append(values[i-WINDOW:i])
        targets.append(values[i])
        dates.append(airport_dates[i])

X = np.asarray(sequences, dtype=np.float32).reshape(-1, WINDOW, 1)
y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
dates = pd.to_datetime(dates)

# Chronological split: earlier 80% for training, later 20% for testing.
cutoff = dates.sort_values()[int(len(dates) * 0.8)]
train_mask = dates < cutoff
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough data for training/testing.")

# Normalize using training data only.
x_mean = float(X_train.mean())
x_std = float(X_train.std()) or 1.0
y_mean = float(y_train.mean())
y_std = float(y_train.std()) or 1.0

X_train = (X_train - x_mean) / x_std
X_test = (X_test - x_mean) / x_std
y_train_scaled = (y_train - y_mean) / y_std

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train_scaled)
)
loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class DelayLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        result, _ = self.lstm(x)
        return self.output(result[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DelayLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print("\nTraining LSTM on", device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_x)

    average_loss = total_loss / len(train_dataset)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Training loss: {average_loss:.4f}"
    )

# Evaluate on the later time period.
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test).to(device)
    predicted_scaled = model(test_tensor).cpu().numpy()

predicted = predicted_scaled * y_std + y_mean

mae = float(np.mean(np.abs(predicted - y_test)))
rmse = float(np.sqrt(np.mean((predicted - y_test) ** 2)))

model_path = MODEL_DIR / "delay_lstm.pt"
torch.save(
    {
        "model_state_dict": model.cpu().state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "window": WINDOW,
    },
    model_path
)

print("\n--- LSTM TRAINING COMPLETE ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print("Saved model:", model_path)
