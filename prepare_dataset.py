
import pandas as pd
from pathlib import Path

DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = DATA_FOLDER / "processed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

FLIGHTS_FILE = DATA_FOLDER / "flights.csv"

# Select only the columns needed for this project.
USE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_DELAY",
    "SCHEDULED_TIME",
    "DISTANCE",
    "ARRIVAL_DELAY",
    "CANCELLED",
    "DIVERTED",
]

chunks = []
total_rows = 0

print("Reading flights.csv in chunks...")

for chunk in pd.read_csv(
    FLIGHTS_FILE,
    usecols=USE_COLUMNS,
    dtype={
        "ORIGIN_AIRPORT": "string",
        "DESTINATION_AIRPORT": "string",
        "TAIL_NUMBER": "string",
        "AIRLINE": "string",
    },
    chunksize=100_000,
    low_memory=True,
):
    total_rows += len(chunk)

    # Keep records useful for delay prediction.
    chunk = chunk[
        (chunk["CANCELLED"] == 0)
        & (chunk["DIVERTED"] == 0)
        & chunk["ARRIVAL_DELAY"].notna()
        & chunk["SCHEDULED_TIME"].notna()
        & chunk["DISTANCE"].notna()
        & chunk["TAIL_NUMBER"].notna()
    ].copy()

    chunks.append(chunk)

    if total_rows % 500_000 == 0:
        print(f"Read approximately {total_rows:,} rows")

print("\nCombining cleaned data...")
df = pd.concat(chunks, ignore_index=True)
del chunks

# Build a date column.
df["FLIGHT_DATE"] = pd.to_datetime(
    {
        "year": df["YEAR"],
        "month": df["MONTH"],
        "day": df["DAY"],
    },
    errors="coerce",
)

# Remove rows with missing or invalid essential values.
df = df.dropna(
    subset=[
        "FLIGHT_DATE",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ARRIVAL_DELAY",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
)

# Keep reasonable values for the first modelling version.
df = df[
    (df["SCHEDULED_TIME"] > 0)
    & (df["DISTANCE"] > 0)
].copy()

# Save only the cleaned records.
output_file = OUTPUT_FOLDER / "flights_clean.csv"
df.to_csv(output_file, index=False)

print("\n--- CLEANING COMPLETE ---")
print("Original rows read:", f"{total_rows:,}")
print("Cleaned rows:", f"{len(df):,}")
print("Saved to:", output_file)
print("\nSample:")
print(df.head(5).to_string(index=False))
print("\nMissing values in important columns:")
print(
    df[
        [
            "AIRLINE",
            "TAIL_NUMBER",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "ARRIVAL_DELAY",
        ]
    ].isna().sum()
)
