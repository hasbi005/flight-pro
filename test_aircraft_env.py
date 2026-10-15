
from aircraft_env import AircraftReallocationEnv

env = AircraftReallocationEnv()

observation, info = env.reset(seed=42)

print("Environment created successfully!")
print("Initial observation:", observation)

action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

print("Selected action:", action)
print("Reward:", reward)
print("Scenario result:", info)

env.close()
