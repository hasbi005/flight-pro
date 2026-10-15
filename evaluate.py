
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from aircraft_env import AircraftReallocationEnv


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ppo_aircraft.zip"

env = AircraftReallocationEnv()
model = PPO.load(str(MODEL_PATH))

# Synthetic scenarios for testing the policy.
scenarios = [
    {"name": "A", "delay": 10, "disruption": 0, "spare": 1},
    {"name": "B", "delay": 60, "disruption": 1, "spare": 1},
    {"name": "C", "delay": 90, "disruption": 1, "spare": 0},
    {"name": "D", "delay": 120, "disruption": 1, "spare": 1},
]

results = []

for scenario in scenarios:
    delay = float(scenario["delay"])
    disruption = int(scenario["disruption"])
    spare = int(scenario["spare"])

    # Baseline: keep original aircraft.
    env.reset(seed=42)
    env.predicted_delay = delay
    env.disruption = disruption
    env.spare_aircraft = spare
    env.availability = 1 - disruption

    baseline_obs = env._get_observation()
    _, baseline_reward, _, _, baseline_info = env.step(0)

    # PPO: reproduce exactly the same scenario.
    env.reset(seed=42)
    env.predicted_delay = delay
    env.disruption = disruption
    env.spare_aircraft = spare
    env.availability = 1 - disruption

    observation = env._get_observation()
    action, _ = model.predict(observation, deterministic=True)

    _, ppo_reward, _, _, ppo_info = env.step(int(action))

    results.append({
        "Scenario": scenario["name"],
        "Predicted Delay": delay,
        "Disruption": disruption,
        "Spare Aircraft": spare,
        "Baseline Delay": baseline_info["final_delay_minutes"],
        "PPO Action": int(action),
        "PPO Delay": ppo_info["final_delay_minutes"],
        "Baseline Reward": round(baseline_reward, 3),
        "PPO Reward": round(ppo_reward, 3),
    })

df = pd.DataFrame(results)

print("\n--- SCENARIO EVALUATION ---")
print(df.to_string(index=False))

print("\n--- SUMMARY ---")
print(f"Scenarios tested: {len(results)}")
print(
    "Baseline average reward:",
    round(df["Baseline Reward"].mean(), 3)
)
print(
    "PPO average reward:",
    round(df["PPO Reward"].mean(), 3)
)

print(
    "Scenarios with improved reward:",
    int((df["PPO Reward"] > df["Baseline Reward"]).sum()),
    "out of",
    len(results),
)

env.close()
