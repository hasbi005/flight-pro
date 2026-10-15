
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "flights_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading cleaned dataset...")

# Load only the columns needed for daily delay forecasting.
df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "FLIGHT_DATE",
        "ORIGIN_AIRPORT",
        "ARRIVAL_DELAY",
    ],
    parse_dates=["FLIGHT_DATE"],
)

print("Loaded records:", f"{len(df):,}")

# Calculate daily average arrival delay for each origin airport.
daily = (
    df.groupby(
        ["FLIGHT_DATE", "ORIGIN_AIRPORT"],
        as_index=False
    )
    .agg(
        AVG_ARRIVAL_DELAY=("ARRIVAL_DELAY", "mean"),
        FLIGHT_COUNT=("ARRIVAL_DELAY", "size"),
        DELAYED_FLIGHTS=(
            "ARRIVAL_DELAY",
            lambda values: (values > 15).sum()
        ),
    )
)

daily["DELAYED_FLIGHT_RATE"] = (
    daily["DELAYED_FLIGHTS"] / daily["FLIGHT_COUNT"]
)

daily = daily.sort_values(
    ["ORIGIN_AIRPORT", "FLIGHT_DATE"]
).reset_index(drop=True)

output_file = OUTPUT_DIR / "daily_airport_delays.csv"
daily.to_csv(output_file, index=False)

print("\n--- DAILY DATA CREATED ---")
print("Daily airport records:", f"{len(daily):,}")
print("Number of airports:", daily["ORIGIN_AIRPORT"].nunique())
print("Date range:", daily["FLIGHT_DATE"].min(), "to",
      daily["FLIGHT_DATE"].max())
print("Saved to:", output_file)

print("\nSample records:")
print(daily.head(10).to_string(index=False))
