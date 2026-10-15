
import pandas as pd
from pathlib import Path

# Find the data folder
DATA_FOLDER = Path(__file__).parent / "data"

# Read flight details from CSV
flights = pd.read_csv(DATA_FOLDER / "flights.csv")

# Display the project title
print("===================================")
print(" AIRCRAFT REALLOCATION SYSTEM")
print("===================================")

# Display flight information
print("\nFlight Details:")
print(flights.to_string(index=False))

# Step 3: Find disrupted flights

disrupted_flights = flights[
    flights["status"] == "Disrupted"
]

print("\nDisrupted Flights:")
print(disrupted_flights.to_string(index=False))

# Show the affected flight ID
if not disrupted_flights.empty:
    affected_flight = disrupted_flights.iloc[0]

    print("\nAffected Flight ID:", affected_flight["flight_id"])
    print("Origin Airport:", affected_flight["origin"])
    print("Destination Airport:", affected_flight["destination"])
    print("Scheduled Departure:", affected_flight["scheduled_departure"])
aircraft = pd.read_csv(DATA_FOLDER / "aircraft.csv")

print("\nAircraft CSV Columns:")
print(aircraft.columns.tolist())

print("\nAircraft Data:")
print(aircraft.to_string(index=False))


# Step 4B: Find replacement aircraft

affected_flight = disrupted_flights.iloc[0]

departure_time = pd.to_datetime(
    affected_flight["scheduled_departure"]
)

aircraft["next_available_time"] = pd.to_datetime(
    aircraft["next_available_time"]
)

candidates = aircraft[
    (aircraft["airline_id"] == affected_flight["airline_id"]) &
    (aircraft["current_airport"] == affected_flight["origin"]) &
    (aircraft["available"] == "Yes") &
    (aircraft["technical_status"] == "Serviceable") &
    (aircraft["route_approved"] == "Yes") &
    (aircraft["next_available_time"] <= departure_time)
]

print("\n=== REPLACEMENT AIRCRAFT RESULT ===")

if not candidates.empty:
    print("Affected Flight:", affected_flight["flight_id"])
    print("Route:", affected_flight["origin"], "to",
          affected_flight["destination"])
    print("Departure:", departure_time)

    print("\nPossible replacement aircraft:")
    print(candidates.to_string(index=False))
else:
    print("No suitable replacement aircraft found.")


# Step 5: Check existing replacement assignments

assignments = pd.read_csv(DATA_FOLDER / "assignments.csv")

print("\n=== EXISTING ASSIGNMENTS ===")
print(assignments.to_string(index=False))

if not candidates.empty:
    candidate_aircraft_id = candidates.iloc[0]["aircraft_id"]

    # Check whether this aircraft is already selected
    # as a replacement for another flight
    existing_assignment = assignments[
        (assignments["replacement_aircraft_id"] == candidate_aircraft_id) &
        (assignments["flight_id"] != affected_flight["flight_id"])
    ]

    print("\n=== ASSIGNMENT CONFLICT CHECK ===")

    if existing_assignment.empty:
        print("No other replacement assignment found for",
              candidate_aircraft_id)
        print("Preliminary result: Candidate can proceed to further checks.")
    else:
        print("Potential assignment conflict for", candidate_aircraft_id)
        print(existing_assignment.to_string(index=False))
else:
    print("No replacement candidate available.")

# Step 6: Check whether the candidate has another scheduled flight

if not candidates.empty:
    candidate_aircraft_id = candidates.iloc[0]["aircraft_id"]

    # Find flights already using this aircraft
    candidate_flights = flights[
        (flights["aircraft_id"] == candidate_aircraft_id) &
        (flights["flight_id"] != affected_flight["flight_id"]) &
        (flights["status"].isin(["Scheduled", "Disrupted"]))
    ].copy()

    print("\n=== SCHEDULE CONFLICT CHECK ===")
    print("Candidate aircraft:", candidate_aircraft_id)

    if candidate_flights.empty:
        print("No other scheduled or disrupted flights found.")
        print("Proceed to further feasibility checks.")
    else:
        print("Potential schedule conflict found:")
        print(candidate_flights[
            ["flight_id", "origin", "destination",
             "scheduled_departure", "scheduled_arrival", "status"]
        ].to_string(index=False))
else:
    print("No candidate aircraft to check.")
