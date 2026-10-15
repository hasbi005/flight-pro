
import pandas as pd
from pathlib import Path

DATA_FOLDER = Path(__file__).parent / "data"

# Check small CSV files
for filename in ["airlines.csv", "airports.csv"]:
    file_path = DATA_FOLDER / filename
    df = pd.read_csv(file_path)

    print(f"\n--- {filename} ---")
    print("Rows and columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

# Read the large flights file in chunks
flights_path = DATA_FOLDER / "flights.csv"

total_rows = 0

print("\n--- flights.csv ---")

for chunk in pd.read_csv(flights_path, chunksize=100_000):
    total_rows += len(chunk)

    if total_rows == len(chunk):
        print("Columns:", chunk.columns.tolist())
        print("\nFirst 5 rows:")
        print(chunk.head().to_string(index=False))

print("\nTotal flight records:", total_rows)
